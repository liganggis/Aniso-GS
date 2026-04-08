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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    # means3D,
    means2D,
    conics2D,
    # num_rendered,
    colors_precomp,
    opacities,
    tiles_touched,
    pixes2D,
    conics,
    radii,
    raster_settings,
    # geomBuffer,
):
    return _RasterizeGaussians.apply(
        # means3D,
        means2D,
        conics2D,
        # sh,
        # num_rendered,
        colors_precomp,
        opacities,
        tiles_touched,
        pixes2D,
        conics,
        radii,
        raster_settings,
        # geomBuffer,
    )

def preprocess_gaussians(
    means3D,
    # means2D,
    # sh,
    # colors_precomp,
    # opacities,
    scales,
    rotations,
    # cov3Ds_precomp,
    raster_settings,
    # geomBuffer,
):
    return _PreprocessGaussians.apply(
        means3D,
        # means2D,
        # sh,
        # colors_precomp,
        # opacities,
        scales,
        rotations,
        # cov3Ds_precomp,
        raster_settings,
        # geomBuffer,
    )

class _PreprocessGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        # means2D,
        # sh,
        # colors_precomp,
        # opacities,
        scales,
        rotations,
        # cov3Ds_precomp,
        raster_settings,
        # geomBuffer,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            # geomBuffer,
            means3D,                                # (P, 3)                            每个3D gaussian的XYZ均值
            # opacities,                              # (P, 1)  0.1                       不透明度
            scales,                                 # (P, 3)                            每个3D gaussian的XYZ尺度
            rotations,                              # (P, 4)  [1., 0., 0., 0.]          每个3D gaussian的旋转四元组
            raster_settings.scale_modifier,         # 1.0                               
            # cov3Ds_precomp,                         # []                                提前计算好的每个3D gaussian的协方差矩阵
            raster_settings.viewmatrix,             # (4, 4)                            相机外参矩阵 world to camera
            raster_settings.projmatrix,             # (4, 4)                            相机内参矩阵 camera to image
            raster_settings.tanfovx,                # 0.841174315841308                 水平视场角一半的正切值
            raster_settings.tanfovy,                # 0.4717713779864031                垂直视场角一半的正切值
            raster_settings.image_height,           # 546                               图像高度
            raster_settings.image_width,            # 979                               图像宽度
            # raster_settings.sh_degree,              # 0 -- > 1 -- > 2 -- > 3            球谐函数的次数, 最开始是0, 每隔1000次迭代, 将球谐函数的次数增加1
            # raster_settings.campos,                 # (3,) [-3.9848, -0.3486,  0.1481]  所有相机的中心点坐标
            raster_settings.offset,
            raster_settings.prefiltered,            # False
            raster_settings.debug                   # False
        )

        radii, means2D, conics, cov3Ds, tiles_touched = _C.preprocess(*args)
        
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        # ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, cov3Ds, radii)
        return means2D, conics, radii, tiles_touched

    @staticmethod
    def backward(ctx, grad_out_mean2D, grad_out_conic, grad_radii, grad_tiles_touched):

        # Restore necessary values from context
        # num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        means3D, scales, rotations, cov3Ds, radii = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
                means3D, 
                radii, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                raster_settings.image_height, 
                raster_settings.image_width, 
                raster_settings.offset,
                grad_out_mean2D,
                grad_out_conic,
                # raster_settings.sh_degree, 
                # raster_settings.campos,
                # geomBuffer,
                # num_rendered,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means3D, grad_scales, grad_rotations = _C.preprocess_backward(*args)

        grads = (
            grad_means3D,
            # grad_means2D,
            # grad_sh,
            # grad_colors_precomp,
            # grad_opacities,
            grad_scales,
            grad_rotations,
            # grad_cov3Ds,
            None,
            # None,
        )

        return grads

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        # means3D,
        means2D,
        conics2D,
        # sh,
        # num_rendered,
        colors_precomp,
        opacities,
        tiles_touched,
        pixes2D,
        conics,
        radii,
        # opacities,
        # scales,
        # rotations,
        # cov3Ds_precomp,
        raster_settings,
        # geomBuffer,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            # geomBuffer,
            raster_settings.bg,                     # (3,)    [0., 0., 0.]              背景颜色
            # means3D,                                # (P, 3)                            每个3D gaussian的XYZ均值
            colors_precomp,                         # []                                提前计算好的每个3D gaussian的颜色
            opacities,
            # opacities,                              # (P, 1)  0.1                       不透明度
            # scales,                                 # (P, 3)                            每个3D gaussian的XYZ尺度
            # rotations,                              # (P, 4)  [1., 0., 0., 0.]          每个3D gaussian的旋转四元组
            # raster_settings.scale_modifier,         # 1.0                               
            # cov3Ds_precomp,                         # []                                提前计算好的每个3D gaussian的协方差矩阵
            # raster_settings.viewmatrix,             # (4, 4)                            相机外参矩阵 world to camera
            # raster_settings.projmatrix,             # (4, 4)                            相机内参矩阵 camera to image
            # raster_settings.tanfovx,                # 0.841174315841308                 水平视场角一半的正切值
            # raster_settings.tanfovy,                # 0.4717713779864031                垂直视场角一半的正切值
            raster_settings.image_height,           # 546                               图像高度
            raster_settings.image_width,            # 979                               图像宽度
            # sh,                                     # (P, 16, 3)                        每个3D gaussian对应的球谐系数, R、G、B3个通道分别对应16个球谐系数
            radii,
            pixes2D,
            conics,
            tiles_touched,
            # raster_settings.sh_degree,              # 0 -- > 1 -- > 2 -- > 3            球谐函数的次数, 最开始是0, 每隔1000次迭代, 将球谐函数的次数增加1
            # raster_settings.campos,                 # (3,) [-3.9848, -0.3486,  0.1481]  所有相机的中心点坐标
            raster_settings.prefiltered,            # False
            # num_rendered,
            raster_settings.debug                   # False
        )
        # print('===================-----------------')
        # print(radii)
        # print(pixes2D)
        # print(conics)

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, pixes2D, conics, opacities, binningBuffer, imgBuffer)
        return color

    @staticmethod
    def backward(ctx, grad_out_color):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, pixes2D, conics, opacities, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                # means3D, 
                # radii, 
                colors_precomp, 
                # scales, 
                # rotations, 
                # raster_settings.scale_modifier, 
                # cov3Ds_precomp, 
                # raster_settings.viewmatrix, 
                # raster_settings.projmatrix, 
                # raster_settings.tanfovx, 
                # raster_settings.tanfovy, 
                pixes2D,
                conics,
                opacities,
                grad_out_color, 
                # sh, 
                # raster_settings.sh_degree, 
                # raster_settings.campos,
                # geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_conics, grad_opacities = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_conics, grad_opacities = _C.rasterize_gaussians_backward(*args)

        grads = (
            # None,
            grad_means2D,
            grad_conics,
            # grad_sh,
            # None,
            grad_colors_precomp,
            grad_opacities,
            None,
            grad_means2D,
            grad_conics,
            # grad_cov3Ds_precomp,
            None,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    # sh_degree : int
    # campos : torch.Tensor
    offset: float
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings
        # self.geomBuffer = torch.empty(0)

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible
    
    def preprocess(self, means3D, scales = None, rotations = None):
        raster_settings = self.raster_settings

        # if ((scales is None or rotations is None)) or ((scales is not None or rotations is not None)):
        #     raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        # if cov3D_precomp is None:
        #     cov3D_precomp = torch.Tensor([])

        # self.geomBuffer = self.geomBuffer.to(means3D.device)
        return preprocess_gaussians(
            means3D,
            # opacities,
            scales,
            rotations,
            # cov3D_precomp,
            raster_settings,
            # self.geomBuffer,
        )

    def forward(self, means2D, conics2D, opacities=None, colors_precomp = None, radii=None, tiles_touched = None, pixes2D = None, conics = None):
        
        raster_settings = self.raster_settings

        if opacities is None:
            opacities = torch.Tensor([])
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')


        if colors_precomp is None:
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if tiles_touched is None:
            tiles_touched = torch.Tensor([])
        if pixes2D is None:
            pixes2D = torch.Tensor([])
        if conics is None:
            conics = torch.Tensor([])

        # self.geomBuffer = self.geomBuffer.to(means3D.device)

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            # means3D,
            means2D,
            conics2D,
            # num_rendered,
            colors_precomp,
            opacities,
            tiles_touched,
            pixes2D, 
            conics,
            radii,
            raster_settings, 
            # self.geomBuffer,
        )

