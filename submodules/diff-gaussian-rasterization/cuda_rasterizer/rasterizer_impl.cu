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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

__global__ void shEncoder(int idx, const glm::vec3* means, glm::vec3 campos, float * outputs) {
	// 获取当前点的中心位置
	glm::vec3 pos = means[idx];
	// 计算从相机位置到当前点的位置向量
	glm::vec3 dir = pos - campos;
	// 将位置向量归一化
	dir = dir / glm::length(dir);

	outputs += 16 * idx;

	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	float xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z, xyz=xy*z;
	float x4=x2*x2, y4=y2*y2, z4=z2*z2;
	float x6=x4*x2, y6=y4*y2, z6=z4*z2;

	outputs[0] = 0.28209479177387814f ;                          // 1/(2*sqrt(pi))
	outputs[1] = -0.48860251190291987f*y ;                               // -sqrt(3)*y/(2*sqrt(pi))
	outputs[2] = 0.48860251190291987f*z ;                                // sqrt(3)*z/(2*sqrt(pi))
	outputs[3] = -0.48860251190291987f*x ;                               // -sqrt(3)*x/(2*sqrt(pi))
	outputs[4] = 1.0925484305920792f*xy ;                                // sqrt(15)*xy/(2*sqrt(pi))
	outputs[5] = -1.0925484305920792f*yz ;                               // -sqrt(15)*yz/(2*sqrt(pi))
	outputs[6] = 0.94617469575755997f*z2 - 0.31539156525251999f ;                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
	outputs[7] = -1.0925484305920792f*xz ;                               // -sqrt(15)*xz/(2*sqrt(pi))
	outputs[8] = 0.54627421529603959f*x2 - 0.54627421529603959f*y2 ;                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
	outputs[9] = 0.59004358992664352f*y*(-3.0f*x2 + y2) ;                         // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
	outputs[10] = 2.8906114426405538f*xy*z ;                             // sqrt(105)*xy*z/(2*sqrt(pi))
	outputs[11] = 0.45704579946446572f*y*(1.0f - 5.0f*z2) ;                                // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
	outputs[12] = 0.3731763325901154f*z*(5.0f*z2 - 3.0f) ;                         // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
	outputs[13] = 0.45704579946446572f*x*(1.0f - 5.0f*z2) ;                                // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
	outputs[14] = 1.4453057213202769f*z*(x2 - y2) ;                              // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
	outputs[15] = 0.59004358992664352f*x*(-x2 + 3.0f*y2) ;                                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
// 寻找给定无符号整数 n 的最高有效位（Most Significant Bit, MSB）的下一个最高位
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
// 计算2d高斯椭圆中心点points_xy在2d像素平面上占据的tile的tileID，并将tileID|depth组合成64位的key值，value值为高斯球的编号
__global__ void duplicateWithKeys(
	int P,
	const float3* points_xy,
	// const float* depths,
	const uint32_t* offsets,				// 累计的tiles数量的数组
	uint64_t* gaussian_keys_unsorted,		// 未排序的key（tileID|depth）
	uint32_t* gaussian_values_unsorted,		// 未排序的valu（depth）
	int* radii,								// 高斯球的半径
	dim3 grid)								// block编号的xy两个极大值
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		// 第idx个高斯球前面已经占据的tiles总数
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		const float3 point_xyz = points_xy[idx];
		// 计算像素点points_xy[idx]在半径为radii[idx]的圆所占据的网格编号的最小值和最大值
		getRect(point_xyz, radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x; 			// 计算当前block所在的tileID
				key <<= 32;								// 左移32位，把后面的32位空出来
				key |= *((uint32_t*)&point_xyz.z);		// 后32位存高斯的深度值
				gaussian_keys_unsorted[off] = key;		// 把上步得到的key，存到第off个（也即是当前高斯球前面的所有高斯所占据的tiles数量）gaussian_keys_unsorted的uint64_t数组中
				gaussian_values_unsorted[off] = idx;	// values设置为idx
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
// 识别每个瓦片（tile）在排序后的高斯ID列表中的范围
// 目的是确定哪些高斯ID属于哪个瓦片，并记录每个瓦片的开始和结束位置
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

// 在给定的内存块中初始化 GeometryState 结构
// chunk（一个指向内存块的指针引用），P（元素的数量）
// 使用 obtain 函数为 GeometryState 的不同成员分配空间，并返回一个初始化的 GeometryState 实例
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	// obtain(chunk, geom.depths, P, 128);
	// obtain(chunk, geom.clamped, P * 3, 128);
	// obtain(chunk, geom.internal_radii, P, 128);
	// obtain(chunk, geom.means2D, P, 128);
	// obtain(chunk, geom.cov3D, P * 6, 128);
	// obtain(chunk, geom.conic_opacity, P, 128);
	// obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

// 初始化 BinningState 实例，分配所需的内存，并执行排序操作
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	// 在 GPU 上进行基数排序, 将 point_list_keys_unsorted 作为键，point_list_unsorted 作为值进行排序，排序结果存储在 point_list_keys 和 point_list 中
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

void CudaRasterizer::Rasterizer::prepForward(
	// std::function<char* (size_t)> geometryBuffer,
	const int P,
	const int width, int height,
	const float* means3D,
	// const float* shs,
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
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);					// 垂直方向的焦距 focal_y		
	const float focal_x = width / (2.0f * tan_fovx);					// 水平方向的焦距 focal_x

	// size_t chunk_size = required<GeometryState>(P);						// 计算存储所有3D gaussian的各个参数所需要的空间大小
	// char* chunkptr = geometryBuffer(chunk_size);						// 给所有3D gaussian的各个参数分配存储空间, 并返回存储空间的指针
	// GeometryState geomState = GeometryState::fromChunk(chunkptr, P);	// 在给定的内存块中初始化 GeometryState 结构体, 为不同成员分配空间，并返回一个初始化的实例

	// if (radii == nullptr)
	// {
	// 	radii = geomState.internal_radii;		// 指向radii数据的指针
	// }

	// 定义了一个三维网格（dim3 是 CUDA 中定义三维网格维度的数据类型），确定了在水平和垂直方向上需要多少个块来覆盖整个渲染区域
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	// 确定了每个块在 X（水平）和 Y（垂直）方向上的线程数
	// dim3 block(BLOCK_X, BLOCK_Y, 1);

	// if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	// {
	// 	throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	// }

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P,							// 3D gaussian的个数, 球谐函数的次数, 球谐系数的个数 (球谐系数用于表示颜色)
		means3D,							// 每个3D gaussian的XYZ均值
		(glm::vec3*)scales,					// 每个3D gaussian的XYZ尺度
		scale_modifier,						// 尺度缩放系数, 1.0
		(glm::vec4*)rotations,				// 每个3D gaussian的旋转四元组
		// opacities,							// 每个3D gaussian的不透明度
		// geomState.clamped,					// 存储每个3D gaussian的R、G、B是否小于0
		// cov3D_precomp,						// 提前计算好的每个3D gaussian的协方差矩阵, []
		// colors_precomp,						// 提前计算好的每个3D gaussian的颜色, []
		viewmatrix, projmatrix,				// 投影矩阵, world to image
		// (glm::vec3*)cam_pos,				// 所有相机的中心点XYZ坐标
		width, height,						// 图像的宽和高
		focal_x, focal_y,					// 水平、垂直方向的焦距
		tan_fovx, tan_fovy,					// 水平、垂直视场角一半的正切值
		offset,
		radii,								// 存储每个2D gaussian在图像上的半径
		means2D,					// 存储每个2D gaussian的均值
		// depths,					// 存储每个2D gaussian的深度
		cov3Ds,					// 存储每个3D gaussian的协方差矩阵
		conics,			// 存储每个2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
		tile_grid,							// 在水平和垂直方向上需要多少个块来覆盖整个渲染区域
		tiles_touched,			// 存储每个2D gaussian覆盖了多少个tile
		prefiltered							// 是否预先过滤掉了中心点(均值XYZ)不在视锥（frustum）内的3D gaussian
	), debug)
}

void CudaRasterizer::Rasterizer::prepBackward(
	const int P,
	const int width, int height,
	const float* means3D,
	// const float* shs,
	// const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	// const float* cov3D_precomp,
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
	bool debug)
{
	// GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);

	// if (radii == nullptr)
	// {
	// 	radii = geomState.internal_radii;
	// }

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	// const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P,
		(float3*)means3D,
		radii,
		// shs,
		// geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3Ds,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		offset,
		// (glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		// dL_dcolor,
		dL_dcov3D,
		// dL_dopacity,
		// dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
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
	bool debug)
{
	// GeometryState geomState = GeometryState::fromChunk(geometryBuffer, P);
	size_t chunk_size = required<GeometryState>(P);						// 计算存储所有3D gaussian的各个参数所需要的空间大小
	char* chunkptr = geometryBuffer(chunk_size);						// 给所有3D gaussian的各个参数分配存储空间, 并返回存储空间的指针
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);	// 在给定的内存块中初始化 GeometryState 结构体, 为不同成员分配空间，并返回一个初始化的实例

	// geomState.tiles_touched = tiles_touched;
	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// 同步运行InclusiveSum，获取tiles_touched数组的前缀和，存到point_offsets中
	// CHECK_CUDA(cub::DeviceScan::InclusiveSum(nullptr, geomState.scan_size, tiles_touched, tiles_touched, P), debug);
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;	// 存储所有的2D gaussian总共覆盖了多少个tile，允许不同2D gaussian在相同的tileid
	// 将 geomState.point_offsets 数组中最后一个元素的值复制到主机内存中的变量 num_rendered
	// point_offsets + P - 1），也就是point_offsets数组的最后一个元素的值，赋给num_rendered，也就是总共覆盖的tiles数量
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// 定义了一个三维网格（dim3 是 CUDA 中定义三维网格维度的数据类型），确定了在水平和垂直方向上需要多少个块来覆盖整个渲染区域
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	// 确定了每个块在 X（水平）和 Y（垂直）方向上的线程数
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// 计算所需的BinningState的数量，即每个高斯球覆盖的tile都有对应的装箱状态BinningState数据
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	// 调整显存块大小，并返回首地址
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	// 用显存块首地址作为参数，调用fromChunk函数来申请显存
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// 将每个3D gaussian的对应的tile index和深度存到point_list_keys_unsorted中
    // 将每个3D gaussian的对应的index（第几个3D gaussian）存到point_list_unsorted中
	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		means2D,
		// geomState.depths,
		geomState.point_offsets,					// 这里用到上步InclusiveSum得到的累计高斯球touch的tiles数
		binningState.point_list_keys_unsorted,		// 存储key（tileID|depth）
		binningState.point_list_unsorted,			// 存储对应的高斯球idx
		radii,										// 像素平面上高斯圆的半径，最长轴的3倍
		tile_grid)									// 全图中tile的数量
	CHECK_CUDA(, debug)								// 同步，并检查错误

	// 查找tile_grid.x * tile_grid.y的最高位
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// 对一个键值对列表进行排序。这里的键值对由 binningState.point_list_keys_unsorted 和 binningState.point_list_unsorted 组成
    // 排序后的结果存储在 binningState.point_list_keys 和 binningState.point_list 中
    // binningState.list_sorting_space 和 binningState.sorting_size 指定了排序操作所需的临时存储空间和其大小
    // num_rendered 是要排序的元素总数。0, 32 + bit 指定了排序的最低位和最高位，这里用于确保排序考虑到了足够的位数，以便正确处理所有的键值对
	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,										// 辅助空间
		binningState.sorting_size,												// 辅助空间大小
		binningState.point_list_keys_unsorted, binningState.point_list_keys,	// d_keys_in,d_keys_out
		binningState.point_list_unsorted, binningState.point_list,				// d_values_in, d_values_out
		num_rendered, 0, 32 + bit), debug)										// 总共覆盖的tiles数量，开始bit位，结束bit位

	// 将 imgState.ranges 数组中的所有元素设置为0，长度为tile_grid.x * tile_grid.y * sizeof(uint2)
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// 识别每个瓦片（tile）在排序后的高斯ID列表中的范围
    // 目的是确定哪些高斯ID属于哪个瓦片，并记录每个瓦片的开始和结束位置
	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	// const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, 						// 在水平和垂直方向上需要多少个块来覆盖整个渲染区域
		block,							// 每个块在 X（水平）和 Y（垂直）方向上的线程数
		imgState.ranges,				// 每个瓦片（tile）在排序后的高斯ID列表中的范围
		binningState.point_list,		// 排序后的3D gaussian的id列表	
		width, height,					// 图像的宽和高
		means2D,				// 每个2D gaussian在图像上的中心点位置
		colors_precomp,					// 每个3D gaussian对应的RGB颜色
		opacities,
		conics,		// 每个2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
		imgState.accum_alpha,			// 渲染过程后每个像素的最终透明度或透射率值
		imgState.n_contrib,				// 每个pixel的最后一个贡献的2D gaussian是谁
		background,						// 背景颜色
		out_color), debug)				// 输出图像

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int R,
	const float* background,
	const int width, int height,
	const float* colors_precomp,
	const float3* means2D,
	const float3* conics,
	const float* opacities,
	// char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	bool debug)
{
	// GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	// const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		means2D,
		conics,
		opacities,
		colors_precomp,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float3*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

}