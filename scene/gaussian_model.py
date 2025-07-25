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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from scene.nerf_model import NLGS
from utils.ema import EMA
from utils.ema2 import ExponentialMovingAverage

class GaussianModel:
    # 从尺度和旋转参数中去构建3Dgaussian的协方差矩阵
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp             # 将尺度限制为非负数
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid         # 将不透明度限制在0-1的范围内
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, bound=2.0, tensor_fusion=False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)                  # 中心点位置, 也即3Dgaussian的均值
        # self._features_dc = torch.empty(0)          # 第一个球谐系数, 球谐系数用来表示RGB颜色
        # self._features_rest = torch.empty(0)        # 其余球谐系数
        self._scaling = torch.empty(0)              # 尺度
        self._rotation = torch.empty(0)             # 旋转参数, 四元组
        self._opacity = torch.empty(0)              # 不透明度
        self.max_radii2D = torch.empty(0)           # 投影到2D时, 每个2D gaussian最大的半径
        self.xyz_gradient_accum = torch.empty(0)    # 3Dgaussian的均值的累积梯度
        self.opacity_accum = torch.empty(0)    # 3Dgaussian的均值的累积梯度
        self.denom = torch.empty(0)
        self.optimizer = None                       # 上述各参数的优化器
        self.percent_dense = 0  
        self.spatial_lr_scale = 0
        self.bound = bound                             # hashgrid的边框(bound)的大小
        # if not self.bound:
        self._nlgs = NLGS(self.bound, tensor_fusion).to('cuda')
        self.ema = None
        self.nlgs_optimizer = None
        self.densify = False
        self.densify_ema = False
        self.densify_decay = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            # self._features_dc,
            # self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self._nlgs.state_dict(),
            self.nlgs_optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        # self._features_dc, 
        # self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self._nlgs.load_state_dict(opt_dict)
        self.nlgs_optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    # @property
    # def get_features(self):
    #     features_dc = self._features_dc
    #     features_rest = self._features_rest
    #     return torch.cat((features_dc, features_rest), dim=1)
    
    # @property
    # def get_opacity(self):
    #     return self.opacity_activation(self._opacity)
    
    @property
    def get_opacity(self):
        return self._opacity
    
    @get_opacity.setter
    def get_opacity(self, opacity):
        self._opacity = opacity
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features[:, :3, 0 ] = fused_color
        # features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        self._xyz = nn.Parameter(fused_point_cloud)
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        # self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rots)
        # self._opacity = nn.Parameter(opacities)
        self._opacity = opacities
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense    # 0.01
        # 存储每个3D gaussian的均值xyz的梯度, 用于判断是否对该3D gaussian进行克隆或者
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # (P, 1)
        self.opacity_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # (P, 1)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")     # (P, 1)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self._nlgs.grid_params(), 'lr': 0.005, "name": "grid"},
            {'params': self._nlgs.view_params(), 'lr': 0.0001, "name": "view"},
            {'params': self._nlgs.sigma_params(), 'lr': 0.001, "name": "sigma"},
        ]
        # 创建optimizer
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # 创建对xyz参数进行学习率调整的scheduler
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.densify = training_args.densify
        self.densify_ema = training_args.densify_ema
        self.densify_decay = training_args.densify_decay
        self.ema = EMA(self._nlgs, 0.995)

    # 对xyz的学习率进行调整
    def update_learning_rate(self, iteration, total_iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "view":
                warm_up = total_iteration * 0.1
                if iteration <= warm_up:
                    lr = (0.01 + 0.99 * (iteration / warm_up))
                else:
                    lr = (0.01 ** ((iteration-warm_up) / (total_iteration - warm_up)))
                param_group['lr'] = 0.01 * lr
            if param_group["name"] == "sigma":
                warm_up = total_iteration * 0.2
                if iteration <= warm_up:
                    lr = (0.01 + 0.99 * (iteration / warm_up))
                else:
                    lr = (0.1 ** ((iteration-warm_up) / (total_iteration - warm_up)))
                param_group['lr'] = 0.01 * lr

    def construct_list_of_attributes(self):
        # l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        #     l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        # l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # normals = np.zeros_like(xyz)
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_mlp_checkpoints(self,path,ema_path):
        mkdir_p(os.path.dirname(path))
        torch.save({
            'nlgs_mlp': self._nlgs.state_dict(),
        }, path)
        if self.ema.isStart:
            self.ema.apply_shadow()
            torch.save({
                'nlgs_mlp': self._nlgs.state_dict(),
            }, ema_path)
            self.ema.restore()

    def load_mlp_checkpoints(self,path):
        checkpoint = torch.load(path)
        self._nlgs.load_state_dict(checkpoint['nlgs_mlp'])

    # 重置不透明度
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # features_dc = np.zeros((xyz.shape[0], 3, 1))
        # features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 删除不符合要求的3D gaussian在self.optimizer中对应的参数(均值、球谐系数、不透明度、尺度、旋转参数)
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'alpha' in group['name']:
                continue
            if 'grid' in group['name']:
                continue
            if 'view' in group['name']:
                continue
            if 'sigma' in group['name']:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._opacity = optimizable_tensors["opacity"]
        self._xyz = optimizable_tensors["xyz"]

        self._opacity = self._opacity[valid_points_mask]
        
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        # self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'alpha' in group['name']:
                continue
            if 'grid' in group['name']:
                continue
            if 'view' in group['name']:
                continue
            if 'sigma' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # 将挑选出来的3D gaussian的参数拼接到原有的参数之后
    def densification_postfix(self, new_xyz, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        # "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._opacity = optimizable_tensors["opacity"]
        self._opacity = torch.cat((self._opacity, new_opacities), dim=0)
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # 对于那些均值的梯度超过一定阈值且尺度大于一定阈值的3D gaussian进行分割操作
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")                   # (P, )
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # padded_grad2 = torch.zeros((n_init_points), device="cuda")                   # (P, )
        # padded_grad2[:grads2.shape[0]] = grads2.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False) # (P, )
        # selected_pts_mask1 = torch.where(padded_grad2 >= grad_threshold, True, False) # (P, )
        # postfix_pts_mask = torch.where(padded_grad >= postfix_threshold, True, False)
        # selected_pts_mask = torch.logical_and(selected_pts_mask0, selected_pts_mask1)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)                      # (2 * P, 3)
        means =torch.zeros((stds.size(0), 3),device="cuda")                         # (2 * P, 3)
        samples = torch.normal(mean=means, std=stds)                                # (2 * P, 3)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)      # (2 * P, 3, 3)

        # 在以原来3Dgaussian的均值xyz为中心, stds为形状, rots为方向的椭球内随机采样新的3Dgaussian
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)     # (2 * P, 3)

        # 由于原来的3D gaussian的尺度过大, 现在将3D gaussian的尺度缩小为原来的1/1.6
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))        # (2 * P, 3)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)                # (2 * P, 4)
        # new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)        # (2 * P, 1, 3)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)    # (2 * P, 15, 3)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)                  # (2 * P, 1)
        # new_alpha = self._alpha[selected_pts_mask].repeat(N,1)                  # (2 * P, 1)

        self.densification_postfix(new_xyz, new_opacity, new_scaling, new_rotation)

        # 将原来的那些均值的梯度超过一定阈值且尺度大于一定阈值的3D gaussian进行删除 (因为已经将它们分割成了两个新的3D gaussian，原先的不再需要了)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    # 对于那些均值的梯度超过一定阈值且尺度小于一定阈值的3D gaussian进行克隆操作
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.where(grads.squeeze() >= grad_threshold, True, False)
        # selected_pts_mask1 = torch.where(grads2.squeeze() >= grad_threshold, True, False)
        # postfix_pts_mask = torch.where(grads.squeeze() >= postfix_threshold, True, False)
        # selected_pts_mask = torch.logical_or(selected_pts_mask, postfix_pts_mask)
        # selected_pts_mask = torch.logical_and(selected_pts_mask0, selected_pts_mask1)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]                          # (P, 3)
        # new_features_dc = self._features_dc[selected_pts_mask]          # (P, 1)
        # new_features_rest = self._features_rest[selected_pts_mask]      # (P, 15)
        new_opacities = self._opacity[selected_pts_mask]                # (P, 1)
        new_scaling = self._scaling[selected_pts_mask]                  # (P, 1)
        new_rotation = self._rotation[selected_pts_mask]                # (P, 4)
        # new_alpha = self._alpha[selected_pts_mask]                  # (2 * P, 1)

        self.densification_postfix(new_xyz, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration):
        if self.densify_ema:
            # selected_pts_mask0 = torch.where(self.denom < 12.5, True, False)
            # selected_pts_mask1 = torch.where(self.denom > 0, True, False)
            # selected_pts_mask = torch.where(self._xyz.abs().max(-1, keepdim=True).values > extent*3, True, False)
            grads = self.xyz_gradient_accum                 # 3Dgaussian的均值的累积梯度
            # grads[selected_pts_mask] = self.xyz_gradient_accum2[selected_pts_mask]
            grads[grads.isnan()] = 0.0
            # grads2 = self.xyz_gradient_accum2                # 3Dgaussian的均值的累积梯度
            # grads2[grads2.isnan()] = 0.0
            # mask = torch.where(grads2.squeeze() >= max_grad, True, False)
            # Q = torch.quantile(grads.reshape(-1), 1 - (torch.sum(mask) / self.denom.shape[0]))
        else:
            grads = self.xyz_gradient_accum / self.denom                 # 3Dgaussian的均值的累积梯度
            grads[grads.isnan()] = 0.0
            # Q = max_grad

        # mask = torch.where(grads.squeeze() >= max_grad, True, False)
        # Q = torch.quantile(grads.reshape(-1), 1 - (torch.sum(mask) / self.denom.shape[0])*3)

        self.densify_and_clone(grads, max_grad, extent)             # 如果某些3Dgaussian的均值的梯度过大且尺度小于一定阈值，说明是欠重建，则对它们进行克隆
        self.densify_and_split(grads, max_grad, extent)             # 如果某些3Dgaussian的均值的梯度过大且尺度超过一定阈值，说明是过重建，则对它们进行切分

        # selected_pts_mask = torch.where(self._xyz.abs().max(-1, keepdim=True).values > extent, True, False)
        # temp = self.get_opacity
        # temp[selected_pts_mask] -= min_opacity
        prune_mask = (self.get_opacity < min_opacity).squeeze()     # 删除不透明度小于一定阈值的3Dgaussian
        
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size      # 删除2D半径超过2D尺寸阈值的高斯
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent   # 删除尺度超过一定阈值的高斯
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)   
            # prune_mask = torch.logical_or(big_points_vs, big_points_ws)   
        self.prune_points(prune_mask)                               # 对不符合要求的高斯进行删除
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.opacity_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, conics2D, update_filter):
        if self.densify:
            geo_grads = torch.cat((viewspace_point_tensor.grad[update_filter, :2], 
                conics2D.grad[update_filter]), dim=-1)
        else:
            geo_grads = viewspace_point_tensor.grad[update_filter, :2]
        if self.densify_ema:
            temp = torch.norm(geo_grads, p=2, dim=-1, keepdim=True)
            self.xyz_gradient_accum[update_filter] = self.xyz_gradient_accum[update_filter] * self.densify_decay + temp * (1.0 - self.densify_decay)
            # self.xyz_gradient_accum[update_filter] += (temp - self.xyz_gradient_accum[update_filter]) / (1.0+self.denom[update_filter])
            # decay = 0.5
            # self.opacity_accum[update_filter] = self.opacity_accum[update_filter] * decay + temp * (1.0 - decay)
            # self.opacity_accum[update_filter] += temp
            # self.denom[update_filter] += 1
        else:
            self.xyz_gradient_accum[update_filter] += torch.norm(geo_grads, p=2, dim=-1, keepdim=True)
            self.denom[update_filter] += 1
        # self.opacity_accum[update_filter] += self._opacity[update_filter]
        # self.denom[update_filter] += 1

        


