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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <cuda.h>
#include "cuda_runtime.h"

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static void prepForward(
			// std::function<char* (size_t)> geometryBuffer,
			const int P,
			// const float* background,
			const int width, int height,
			const float* means3D,
			// const float* shs,
			// const float* colors_precomp,
			// const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			// const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			// const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const float offset,
			const bool prefiltered,
			int* radii,
			float3* means2D,
			float3* conics,
			float* cov3Ds,
			uint32_t* tiles_touched,
			bool debug = false);
			
		static void prepBackward(
			const int P,
			const int width, int height,
			const float* means3D,
			// const float* shs,
			// const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3Ds,
			const float* viewmatrix,
			const float* projmatrix,
			// const float* campos,
			const float tan_fovx, float tan_fovy,
			const float offset,
			const int* radii,
			// char* geom_buffer,
			const float* dL_dmean2D,
			const float* dL_dconic,
			// float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			// float* dL_dopacity,
			// float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool debug);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const float* background,
			const int width, int height,
			const float* colors_precomp,
			const float* opacities,
			const bool prefiltered,
			float* out_color,
			float3* means2D,
			float3* conics,
			int* radii,
			uint32_t* tiles_touched,
			bool debug = false);

		static void backward(
			const int P, int R,
			const float* background,
			const int width, int height,
			const float* colors_precomp,
			const float3* means2D,
			const float3* conics,
			const float* opacities,
			// char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			bool debug);
	};
};

#endif