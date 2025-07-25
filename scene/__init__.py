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

import os
import shutil
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

# 数据加载 + 模型重建
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # 寻找是否有训练过的记录, 如果没有则为初次训练, 需要从COLMAP创建的点云中初始化每个点对应的3D gaussian
        # 以及将每张图片对应的相机参数dump到`cameras.json`文件中
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # 从COLMAP或Blender中读取每张图片, 以及每张图片对应的相机内外参
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found transforms.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # 将每张图片对应的相机参数dump到`cameras.json`文件中
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 随机打乱所有图片和对应相机的顺序
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # 所有相机的中心点位置到最远camera的距离
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 多分辨率图像输入，默认scale：1.0
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # 如果是初次训练, 则从COLMAP创建的点云中初始化每个点对应的3D gaussian, 否则直接从之前保存的模型文件中读取3D gaussian
        if self.loaded_iter:
            if self.loaded_iter == 40000:
                self.gaussians.load_ply(os.path.join(self.model_path,"point_cloud","best_result","point_cloud.ply"))
                if not args.isUsingEMA:
                    check_path = os.path.join(self.model_path,"point_cloud","best_result","checkpoint_ema.pth")
                else:
                    check_path = os.path.join(self.model_path,"point_cloud","best_result","checkpoint.pth")
                if not os.path.exists(check_path):
                    check_path = os.path.join(self.model_path,"point_cloud","best_result","checkpoint.pth")
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,"point_cloud","iteration_" + str(self.loaded_iter),"point_cloud.ply"))
                if not args.isUsingEMA:
                    check_path = os.path.join(self.model_path,"point_cloud","iteration_" + str(self.loaded_iter),"checkpoint_ema.pth")
                else:
                    check_path = os.path.join(self.model_path,"point_cloud","iteration_" + str(self.loaded_iter),"checkpoint.pth")
                if not os.path.exists(check_path):
                    check_path = os.path.join(self.model_path,"point_cloud","iteration_" + str(self.loaded_iter),"checkpoint.pth")
            print(check_path)
            self.gaussians.load_mlp_checkpoints(path=check_path)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, isBest=False):
        if not isBest:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            if os.path.exists(point_cloud_path) and os.path.isdir(point_cloud_path):
                shutil.rmtree(point_cloud_path)
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            self.gaussians.save_mlp_checkpoints(os.path.join(point_cloud_path, "checkpoint.pth"), os.path.join(point_cloud_path, "checkpoint_ema.pth"))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/best_result")
            if os.path.exists(point_cloud_path) and os.path.isdir(point_cloud_path):
                shutil.rmtree(point_cloud_path)
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            self.gaussians.save_mlp_checkpoints(os.path.join(point_cloud_path, "checkpoint.pth"), os.path.join(point_cloud_path, "checkpoint_ema.pth"))

    def shuffleCameras(self, scale=1.0):
        random.shuffle(self.train_cameras[scale])
        
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]