/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

// 创建并返回一个 lambda 表达式，该表达式用于调整 torch::Tensor 对象的大小，并返回一个指向它数据的原始指针
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
PreprocessCUDA(
	// torch::Tensor& geomBuffer,
	const torch::Tensor& means3D,
    // const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	// const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const float offset,
	// const int degree,
	// const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

//   torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor means2D = torch::full({P, 3}, 0, means3D.options().dtype(torch::kFloat32));
  torch::Tensor conics = torch::full({P, 3}, 0, means3D.options().dtype(torch::kFloat32));
  torch::Tensor cov3Ds = torch::full({P, 6}, 0, means3D.options().dtype(torch::kFloat32));
  torch::Tensor tiles_touched = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
//   torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
//   torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
//   torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
//   std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
//   std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
//   std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  if(P != 0)
  {
	  CudaRasterizer::Rasterizer::prepForward(
	    // geomFunc,
	    P,									// 3D gaussian的个数, 球谐函数的次数, 球谐系数的个数 (球谐系数用于表示颜色)
		W, H,											// 图像的宽和高
		means3D.contiguous().data<float>(),				// 每个3D gaussian的XYZ均值
		// opacity.contiguous().data<float>(), 			// 每个3D gaussian的不透明度
		scales.contiguous().data_ptr<float>(),			// 每个3D gaussian的XYZ尺度
		scale_modifier,									// 尺度缩放系数, 1.0
		rotations.contiguous().data_ptr<float>(),		// 每个3D gaussian的旋转四元组
		// cov3D_precomp.contiguous().data<float>(), 		// 提前计算好的每个3D gaussian的协方差矩阵, []
		viewmatrix.contiguous().data<float>(), 			// 相机外参矩阵, world to camera
		projmatrix.contiguous().data<float>(),			// 投影矩阵, world to image
		// campos.contiguous().data<float>(),				// 所有相机的中心点XYZ坐标
		tan_fovx,										// 水平视场角一半的正切值
		tan_fovy,										// 垂直视场角一半的正切值
		offset,
		prefiltered,									// 是否预先过滤掉了中心点(均值XYZ)不在视锥（frustum）内的3D gaussian, False
		radii.contiguous().data<int>(),					// 存储每个2D gaussian在图像上的半径
		(float3*)means2D.contiguous().data<float>(),
		(float3*)conics.contiguous().data<float>(),
		cov3Ds.contiguous().data<float>(),
		(uint32_t*)tiles_touched.contiguous().data<int>(),
		debug);											// False
  }
  return std::make_tuple(radii, means2D, conics, cov3Ds, tiles_touched);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
 PreprocessBackwardCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3Ds,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const float offset,
	const torch::Tensor& dL_dout_means2D,
	const torch::Tensor& dL_dout_conic,
	// const int degree,
	// const torch::Tensor& campos,
	// const torch::Tensor& geomBuffer,
	// const int R,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
//   torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
//   torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
//   torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
//   torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
//   torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::prepBackward(P,
	  W, H, 
	  means3D.contiguous().data<float>(),
	//   sh.contiguous().data<float>(),
	//   colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3Ds.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	//   campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  offset,
	  radii.contiguous().data<int>(),
	//   reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  dL_dout_means2D.contiguous().data<float>(),
	  dL_dout_conic.contiguous().data<float>(),  
	//   dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	//   dL_dsh.contiguous().data<float>(),
	//   dL_dopacity.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans3D, dL_dscales, dL_drotations);
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	// torch::Tensor& geomBuffer,
	const torch::Tensor& background,
	// const torch::Tensor& means3D,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
    const int image_height,
    const int image_width,
	// const torch::Tensor& sh,
	const torch::Tensor& radii,
	const torch::Tensor& means2D,
	const torch::Tensor& conics,
	const torch::Tensor& tiles_touched,
	// const int degree,
	// const torch::Tensor& campos,
	const bool prefiltered,
	// const int num_rendered,
	const bool debug)
{
//   if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
//     AT_ERROR("means3D must have dimensions (num_points, 3)");
//   }
  
  const int P = opacities.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = opacities.options().dtype(torch::kInt32);
  auto float_opts = opacities.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
//   torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if(P != 0)
  {
	rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, 									// 3D gaussian的个数, 球谐函数的次数, 球谐系数的个数 (球谐系数用于表示颜色)
		background.contiguous().data<float>(),			// 背景颜色, [0, 0, 0]
		W, H,											// 图像的宽和高
		colors.contiguous().data<float>(), 				// 提前计算好的每个3D gaussian的颜色, []
		opacities.contiguous().data<float>(),
		prefiltered,									// 是否预先过滤掉了中心点(均值XYZ)不在视锥（frustum）内的3D gaussian, False
		out_color.contiguous().data<float>(),			// 在指定的视角下, 对所有3D gaussian进行投影和渲染得到的图像
		(float3*)means2D.contiguous().data<float>(),
		(float3*)conics.contiguous().data<float>(),
		radii.contiguous().data<int>(),					// 存储每个2D gaussian在图像上的半径
		(uint32_t*)tiles_touched.contiguous().data<int>(),
		debug);											// False
  }
  return std::make_tuple(rendered, out_color, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	// const torch::Tensor& means3D,
    const torch::Tensor& colors,
	const torch::Tensor& means2D,
	const torch::Tensor& conics,
	const torch::Tensor& opacities,
    const torch::Tensor& dL_dout_color,
	// const torch::Tensor& sh,
	// const int degree,
	// const torch::Tensor& campos,
	// const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
  const int P = opacities.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, opacities.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, opacities.options());
  torch::Tensor dL_dconic = torch::zeros({P, 3}, opacities.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, opacities.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  colors.contiguous().data<float>(),
	  (float3*)means2D.contiguous().data<float>(),
	  (float3*)conics.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),
	//   reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),    
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dconic, dL_dopacity);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}