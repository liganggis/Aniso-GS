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
import torch
from random import randint
import random
import numpy as np
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from os import makedirs
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, load_iteration, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    if dataset.contract:
        bound = 2
    # 创建 `GaussianModel` 模型，给点云中的每个点去创建一个3D gaussian
    gaussians = GaussianModel(dataset.sh_degree, bound=dataset.bound, tensor_fusion=dataset.tensor_fusion)
    # 加载数据集和每张图片对应的camera的参数
    if load_iteration == -1:
        scene = Scene(dataset, gaussians)
    else:
        scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
        tb_writer = None
    # 为3D gaussian的各组参数创建optimizer以及lr_scheduler
    gaussians.training_setup(opt)

    # 选择背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 设置test和save的迭代次数
    train_img_nums = len(scene.getTrainCameras())
    opt.iterations = (np.ceil(opt.iterations / train_img_nums).astype(np.int32)) * train_img_nums
    nums = np.ceil(train_img_nums / 5).astype(np.int32)

    start_ema = opt.iterations - train_img_nums * 2

    testing_iterations = [opt.iterations - 5*i for i in range(nums)]
    saving_iterations = [opt.iterations]
    if load_iteration == -1:
        saving_iterations.append(start_ema)
    best_result_test = 0.0

    # torch.cuda.Event()精确记录GPU的运行时间
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):   
        iter_start.record()

        if iteration > start_ema:
            gaussians._xyz.requires_grad = False
            gaussians._scaling.requires_grad = False
            gaussians._rotation.requires_grad = False

        # 对xyz的学习率进行调整
        gaussians.update_learning_rate(iteration, opt.iterations)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 将球谐函数的次数增加1
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # 随机选择一个图片及其相应的相机视角(内外参)
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        if load_iteration != -1 and iteration <= load_iteration:
            continue

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 根据3D gaussian渲染该相机视角的图像
        if iteration < opt.densify_until_iter:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, offset=dataset.offset, isContracting=dataset.contract, isTraning=True)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, offset=dataset.offset, isContracting=dataset.contract, isTraning=False)
        image, viewspace_point_tensor, conics2D, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["conics2D"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            # ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_loss_for_log = loss.item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if iteration > start_ema and iteration <= (opt.iterations - train_img_nums):
                    if not gaussians.ema.isStart:
                        gaussians.ema.register()
                    if gaussians.ema.isStart:
                        gaussians.ema.update(0.995)
                if iteration > (opt.iterations - train_img_nums):
                    if gaussians.ema.isStart:
                        gaussians.ema.update(0.995)
                if iteration == (opt.iterations - train_img_nums):
                    gaussians.ema.store()

            # Log and save
            result_test = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians.ema, render, (pipe, background, dataset.offset, dataset.contract, False))
            if iteration >= 26000 and best_result_test < result_test:
                best_result_test = result_test
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, isBest=True)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 统计3D gaussian均值(xyz)的梯度, 用于对3D gaussians的克隆或者切分
                gaussians.add_densification_stats(viewspace_point_tensor, conics2D, visibility_filter)

                # 对3D gaussians进行克隆或者切分, 并将opacity小于一定阈值的3D gaussians进行删除
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, iteration)
                
                # 对3D gaussians的不透明度进行重置
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            # Optimizer step
            # if iteration < opt.iterations:
            #     gaussians.optimizer.step()
            #     gaussians.optimizer.zero_grad(set_to_none = True)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, ema, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    result_test = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
        #                       {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        validation_configs = [{'name': 'test', 'cameras' : scene.getTestCameras()}]
        ema.apply_shadow()
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # if tb_writer and (idx < 5):
                    #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    #     if iteration == testing_iterations[0]:
                    #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # if tb_writer:
                #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if config['name'] == 'test':
                    result_test = psnr_test
        print("\n----------------------[ITER {}]".format(iteration))
        ema.restore()        
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    return result_test

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_iteration", default=-1, type=int)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    makedirs(args.model_path, exist_ok=True)
    logger_path = os.path.join(args.model_path, 'logger.txt')
    sys.stdout = open(logger_path, 'w')
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.load_iteration, args.debug_from)

    # All done
    print("\nTraining complete.")
