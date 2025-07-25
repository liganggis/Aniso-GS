#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

@torch.cuda.amp.autocast(enabled=False)
def contract(x):
    shape, C = x.shape[:-1], x.shape[-1]
    x = x.view(-1, C)
    mag, idx = x.abs().max(1, keepdim=True)
    scale = 1 / mag.repeat(1, C)
    scale.scatter_(1, idx, (2 - 1 / mag) / mag)
    z = torch.where(mag < 1, x, x * scale)
    return z.view(*shape, C)

@torch.cuda.amp.autocast(enabled=False)
def contract1(x):
    # x: [..., C]
    shape, C = x.shape[:-1], x.shape[-1]
    x = x.view(-1, C)
    mag, idx = torch.sum(x**2, axis=-1, keepdim=True).max(1, keepdim=True)
    z = torch.where(mag <= 1, x, ((2 * torch.sqrt(mag) - 1) / mag) * x)
    return z.view(*shape, C)

@torch.cuda.amp.autocast(enabled=False)
def uncontract(z):
    # z: [..., C]
    shape, C = z.shape[:-1], z.shape[-1]
    z = z.view(-1, C)
    mag, idx = z.abs().max(1, keepdim=True)
    scale = 1 / (2 - mag.repeat(1, C)).clamp(min=1e-8)
    scale.scatter_(1, idx, 1 / (2 * mag - mag * mag).clamp(min=1e-8))
    x = torch.where(mag < 1, z, z * scale)
    return x.view(*shape, C)

# 图像渲染主函数
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, offset = 0.3, isContracting=False, isTraning = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    conics2D = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        conics2D.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # tanfovx 是水平视场角一半的正切值。它定义了视锥（view frustum）在水平方向的范围，可以用来确定屏幕上每个像素在水平方向上的世界空间单位
    # tanfovy 是垂直视场角一半的正切值。它定义了视锥在垂直方向的范围，可以用来确定屏幕上每个像素在垂直方向上的世界空间单位。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        # sh_degree=pc.active_sh_degree,
        # campos=viewpoint_camera.camera_center,
        offset = offset,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # scales = None
    # rotations = None
    # cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    #         dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     else:
    #         shs = pc.get_features
    # else:
    #     colors_precomp = override_color
    pixes2D, conics, radii, tiles_touched = rasterizer.preprocess(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
    )
    mask = (radii > 0)
    if isContracting:
        temPos = contract(pc.get_xyz[mask].detach())
    else:
        temPos = pc.get_xyz[mask].detach()

    colors_precomps = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device='cuda')
    opacities = torch.zeros(pc.get_xyz.shape[0], 1, dtype=pc.get_xyz.dtype, device='cuda')
    
    dir_pp = (temPos - viewpoint_camera.camera_center)
    # dir_dist = dir_pp.norm(dim=1, keepdim=True)
    # dir_pp_normalized = dir_pp / dir_dist
    colors_precomp, opacity = pc._nlgs(temPos, dir_pp, pc.active_sh_degree + 1)
    
    colors_precomps[mask] = colors_precomp
    opacities[mask] = opacity
    if isTraning:
        pc.get_opacity[mask] = opacity.detach()

    rendered_image = rasterizer(
        means2D = means2D,
        conics2D = conics2D,
        opacities = opacities,
        colors_precomp = colors_precomps,
        radii=radii,
        tiles_touched=tiles_touched,
        pixes2D = pixes2D,
        conics = conics)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image = rasterizer(
    #     means2D = means2D,
    #     opacities = opacities,
    #     colors_precomp = colors_precomps,
    #     radii=radii,
    #     tiles_touched=tiles_touched,
    #     pixes2D = pixes2D,
    #     conics = conics)

    # print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2)} MB")
    # print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2)} MB")

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "conics2D": conics2D,
            "visibility_filter" : radii > 0,
            "radii": radii}
