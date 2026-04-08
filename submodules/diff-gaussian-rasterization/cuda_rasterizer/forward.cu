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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
// 从每个3D gaussian对应的球谐系数中计算对应的颜色
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	// 获取当前点的中心位置
	glm::vec3 pos = means[idx];
	// 计算从相机位置到当前点的位置向量
	glm::vec3 dir = pos - campos;
	// 将位置向量归一化
	dir = dir / glm::length(dir);

	// 获取当前点的 SH 系数
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	// 计算 SH 零阶系数的颜色值
	glm::vec3 result = SH_C0 * sh[0];

	// 如果阶数大于 0，则计算一阶 SH 系数的颜色值
	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		// 如果阶数大于 1，则计算二阶 SH 系数的颜色值
		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			// 如果阶数大于 2，则计算三阶 SH 系数的颜色值
			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	// 为结果颜色值加上一个偏移量
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	// 将 RGB 颜色值限制在正值范围内。如果值被限制，则需要在反向传播过程中记录此信息。
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float offset, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// 该函数实现了 "EWA Splatting" (Zwicker et al., 2002) 中的公式29和31，
	// Additionally considers aspect / scaling of viewport.
	// 同时考虑了视口的纵横比/缩放
	// Transposes used to account for row-/column-major conventions.
	// 转置用于处理行/列优先顺序。

	// 将当前3D gaussian的中心点从世界坐标系投影到相机坐标系
	float3 t = transformPoint4x3(mean, viewmatrix);

	// 定义x和y方向的视锥限制， 长宽高为1.3f*2的最大立方体
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	// 限制x和y方向的值，使其不超过视锥限制
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// 透视变换是非线性的，因为一个点的屏幕空间坐标与其深度（Z值）成非线性关系。雅可比矩阵 J 提供了一个在特定点附近的线性近似，这使得计算变得简单且高效
	// 计算雅可比矩阵J，表示从3D到2D的投影变换
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	// 提取视图矩阵的前3x3部分，用于线性变换
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	// 从输入的3D协方差矩阵中构建Vrk矩阵
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// 计算最终的2D协方差矩阵
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	// 应用低通滤波器：每个高斯函数应至少在一个像素宽/高
    // 丢弃第3行和第3列
	cov[0][0] += offset;
	cov[1][1] += offset;
	// 返回2D协方差矩阵的三个元素
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
// 根据当前3D gaussian的尺度和旋转参数计算其对应的协方差矩阵
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	// 创建缩放矩阵
	glm::mat3 S = glm::mat3(0.0000001f);		// 初始化了一个3x3的单位阵
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	// 标准化四元数以获得有效的旋转
	glm::vec4 q = rot;// / glm::length(rot); // 假设已经是单位四元数，不再进行额外的标准化
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	// 根据四元数计算旋转矩阵
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// 计算M矩阵，即缩放后的旋转矩阵
	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	// 计算3D世界协方差矩阵Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	// 协方差矩阵是对称的，只存储上三角部分
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
// 在光栅化之前，对每个高斯进行初始预处理步骤。
template<int C>
__global__ void preprocessCUDA(int P,
	const float* orig_points,						// 原始点的数组，3D高斯中心点
	const glm::vec3* scales,						// 缩放因子数组
	const float scale_modifier,						// 缩放因子修正值
	const glm::vec4* rotations,						// 四元数旋转数组
	// const float* opacities,							// 透明度数组
	// bool* clamped,									// 是否被夹住的标志数组
	// const float* cov3D_precomp,						// 预计算的3D协方差矩阵
	// const float* colors_precomp,					// 预计算的颜色数组
	const float* viewmatrix,						// 视图矩阵
	const float* projmatrix,						// 投影矩阵
	// const glm::vec3* cam_pos,						// 相机位置
	const int W, int H,								// 图像宽度和高度
	const float tan_fovx, float tan_fovy,			// tan(fov_x)和tan(fov_y)
	const float focal_x, float focal_y,				// 焦距
	const float offset,
	int* radii,										// 半径数组
	float3* points_xy_image,						// 点在图像上的坐标数组
	// float* depths,									// 深度数组
	float* cov3Ds,									// 3D协方差矩阵数组
	float3* conics,							// 圆锥参数和透明度数组
	const dim3 grid,								// 网格大小
	uint32_t* tiles_touched,						// 被触及的图块数数组
	bool prefiltered)								// 是否进行预过滤的标志
{
	// 每个线程处理一个3D gaussian, index超过3D gaussian总数的线程直接返回, 防止数组越界访问
	// thread_rank() 对占用的[0, num_threads]线程进行排序
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	// 初始化半径和触及的图块数为0。如果这些不被改变，
	// 那么这个高斯将不会进一步处理。
	radii[idx] = 0;
	tiles_touched[idx] = 0; // 当前高斯与那些图像块相交

	// 判断当前处理的3D gaussian的中心点(均值XYZ)是否在视锥（frustum）内, 如果不在则直接返回
	// Perform near culling, quit if outside.
	// 执行近裁剪，如果超出视锥体外，则退出。
	float3 p_view;	// 用于存储将 p_orig 通过视图矩阵 viewmatrix 转换到视图空间后的点坐标
	// if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
	// 	return;

	// Transform point by projecting
	// 将点通过投影变换到视图空间
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	p_view = transformPoint4x3(p_orig, viewmatrix);
	if (p_view.z <= 0.2f)
		return;
	// 将当前3D gaussian的中心点从世界坐标系投影到裁剪坐标系
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);		// 避免除零错误
	// 将当前3D gaussian的中心点从裁剪坐标转变到归一化设备坐标（Normalized Device Coordinates, NDC）
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	//if (p_proj.x < -1.5 || p_proj.x > 1.5 || p_proj.y < -1.5 || p_proj.y > 1.5)
		//return;

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	// 如果预计算的3D协方差矩阵存在，则使用它；否则根据缩放和旋转参数计算。
	const float* cov3D; // 单独提取，是为了下面计算2D屏幕空间协方差矩阵
	// if (cov3D_precomp != nullptr)
	// {
	// 	cov3D = cov3D_precomp + idx * 6;
	// }
	// else
	// {
	// 根据当前3D gaussian的尺度和旋转参数计算其对应的协方差矩阵
	// scales和rotations每次都会更新，故也需要更新协方差矩阵数组cov3Ds
	computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
	cov3D = cov3Ds + idx * 6;
	// }

	// Compute 2D screen-space covariance matrix
	// 将当前的3D gaussian投影到2D图像，得到对应的2D gaussian的协方差矩阵cov
	// 计算2D屏幕空间协方差矩阵
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, offset, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	// 计算当前2D gaussian的协方差矩阵cov的逆矩阵（EWA算法）
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	// 计算2D gaussian的协方差矩阵cov的特征值lambda1, lambda2, 从而计算2D gaussian的最大半径
    // 对协方差矩阵进行特征值分解时，可以得到描述分布形状的主轴（特征向量）以及这些轴上分布的宽度（特征值）

	// 计算在屏幕空间中的扩展范围（通过计算2D协方差矩阵的特征值）。使用扩展范围计算与此高斯重叠的屏幕空间矩形的边界框。
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

	// 将归一化设备坐标（Normalized Device Coordinates, NDC）转换为像素坐标
	float3 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H), p_view.z };
	uint2 rect_min, rect_max;

	// 计算当前的2D gaussian落在哪几个tile上
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	// 如果没有命中任何一个title则直接返回。如果矩形覆盖0个图块，则退出。
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// 如果颜色已经预计算，则使用它们；否则将球谐系数转换为RGB颜色。

	// 到这里可以获取到哪些3dgs参与了当前视角的计算，可以在此将预处理过程分为两个部分
	// if (colors_precomp == nullptr)
	// {
	// 	// 从每个3D gaussian对应的球谐系数中计算对应的颜色
	// 	glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
	// 	rgb[idx * C + 0] = result.x;
	// 	rgb[idx * C + 1] = result.y;
	// 	rgb[idx * C + 2] = result.z;
	// }
	// shEncoder(idx, (glm::vec3*)orig_points, *cam_pos, )

	// Store some useful helper data for the next steps.
	// 存储一些有用的下一步的辅助数据。
	// depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	// 将逆2D协方差和透明度整齐地打包到一个float4中
	// conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	conics[idx] = { conic.x, conic.y, conic.z};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 主光栅化方法。每个块协作处理一个图块，每个线程处理一个像素。在数据获取和光栅化之间交替进行。
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,				// 点范围数组
	const uint32_t* __restrict__ point_list,		// 点列表
	int W, int H,									// 图像宽度和高度
	const float3* __restrict__ points_xy_image,		// 点在图像上的坐标数组
	const float3* __restrict__ features,				// 特征数组
	const float* __restrict__ opacities,			// 透明度数组
	const float3* __restrict__ conics,		// 圆锥参数和透明度数组
	float* __restrict__ final_T,					// 最终T数组
	uint32_t* __restrict__ n_contrib,				// 贡献数量数组
	const float* __restrict__ bg_color,				// 背景颜色数组
	float* __restrict__ out_color)					// 输出颜色数组
{
	// Identify current tile and associated min/max pixel range.
	// 识别当前图块及其关联的最小/最大像素范围。
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	// 当前处理的tile的左上角的像素坐标
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	// 当前处理的tile的右下角的像素坐标
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	// 当前处理的像素坐标
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	// 当前处理的像素id
	uint32_t pix_id = W * pix.y + pix.x;
	// 当前处理的像素坐标
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	// 检查此线程是否与有效像素关联或位于外部。
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	// 完成的线程可以帮助获取数据，但不进行光栅化。
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	// 当前处理的tile对应的3D gaussian的起始id和结束id
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// 还有多少3D gaussian需要处理
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	// 分配存储批处理收集数据的空间。
	// 高斯球编号,坐标,透明度
	// __shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_xyz[BLOCK_SIZE];
	__shared__ float3 collected_conics[BLOCK_SIZE];
	__shared__ float3 collected_colors[BLOCK_SIZE];
	__shared__ float collected_opacities[BLOCK_SIZE];

	// Initialize helper variables
	// 初始化辅助变量；穿透率,高斯贡献数量,最后高斯数量,颜色
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float3 C = { 0, 0, 0 };

	// Iterate over batches until all done or range is complete
	// 迭代处理批次，直到所有处理完成或范围完成。
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) // 循环与该tile相交的所有2D gaussian
	{
		// End if entire block votes that it is done rasterizing
		// 如果整个块都标记为完成光栅化，则结束。
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		// 从全局内存中共同获取每个高斯数据到共享内存中
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			// 当前处理的3D gaussian的id
			int coll_id = point_list[range.x + progress];
			// collected_id[block.thread_rank()] = coll_id;
			collected_xyz[block.thread_rank()] = points_xy_image[coll_id];
			collected_conics[block.thread_rank()] = conics[coll_id];
			collected_colors[block.thread_rank()] = features[coll_id];
			collected_opacities[block.thread_rank()] = opacities[coll_id];
		}
		block.sync();

		// Iterate over current batch
		// 迭代处理当前批次中的数据
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) // 循环所有2D gaussian中一个批次（KBLOC_SIZE个2D gaussian）
		{
			// Keep track of current position in range
			// 跟踪当前范围内的位置
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			// 使用圆锥矩阵进行重采样
			float3 xy = collected_xyz[j];					// 当前处理的2D gaussian在图像上的中心点坐标
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };	// 当前处理的2D gaussian的中心点到当前处理的pixel的offset
			float3 con_o = collected_conics[j];		// 当前处理的2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
			// 计算高斯分布的强度（或权重），用于确定像素在光栅化过程中的贡献程度，power是到圆心的平方距离
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y + 2 * con_o.y * d.x * d.y);
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// 从3dgs论文中的公式（2）中获得alpha。
			// 通过高斯透明度及其从均值指数衰减获得。
			// 避免数值不稳定性（参见论文附录）。由2D GS中心点向周围逐渐衰减，超出2D椭圆范围衰减到无限接近于0
			float alpha = min(0.99f, collected_opacities[j] * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// float3 dir = {d.x, d.y, -1};
			// dir = dir / glm::length(dir);
			// float3 color = transformPoint3x3(dir, collected_colors[j]);

			// Eq. (3) from 3D Gaussian splatting paper.
			// 计算颜色
			// for (int ch = 0; ch < CHANNELS; ch++)
			// 	C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			float3 color = collected_colors[j];
			// C = {color.x * alpha * T, color.y * alpha * T, color.z * alpha * T};
			C.x += color.x * alpha * T;
			C.y += color.y * alpha * T;
			C.z += color.z * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			// 跟踪最后一个范围条目，以更新此像素。
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	// 所有处理有效像素的线程将最终渲染数据写入帧和辅助缓冲区。
	if (inside)
	{
		final_T[pix_id] = T;						// 渲染过程后每个像素的最终透明度或透射率值
		n_contrib[pix_id] = last_contributor;		// 最后一个贡献的2D gaussian是谁
		// for (int ch = 0; ch < CHANNELS; ch++)
		// 	out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_color[0 * H * W + pix_id] = C.x + T * bg_color[0];
		out_color[1 * H * W + pix_id] = C.y + T * bg_color[1];
		out_color[2 * H * W + pix_id] = C.z + T * bg_color[2];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float3* means2D,
	const float* colors,
	const float* opacities,
	const float3* conics,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,				// 每个瓦片（tile）在排序后的高斯ID列表中的范围
		point_list,			// 排序后的3D gaussian的id列表
		W, H,				// 图像的宽和高
		means2D,			// 每个2D gaussian在图像上的中心点位置
		(float3*)colors,				// 每个3D gaussian对应的RGB颜色
		opacities,
		conics,		// 每个2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
		final_T,			// 渲染过程后每个像素的最终透明度或透射率值
		n_contrib,			// 每个pixel的最后一个贡献的2D gaussian是谁
		bg_color,			// 背景颜色
		out_color);			// 输出图像
}

void FORWARD::preprocess(int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	// const float* opacities,
	// const float* shs,
	// bool* clamped,
	// const float* cov3D_precomp,
	// const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	// const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float offset,
	int* radii,
	float3* means2D,
	// float* depths,
	float* cov3Ds,
	// float* rgb,
	float3* conics,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P,				// 3D gaussian的个数, 球谐函数的次数, 球谐系数的个数 (球谐系数用于表示颜色)
		means3D,				// 每个3D gaussian的XYZ均值
		scales,					// 每个3D gaussian的XYZ尺度
		scale_modifier,			// 尺度缩放系数, 1.0
		rotations,				// 每个3D gaussian的旋转四元组
		// opacities,				// 每个3D gaussian的不透明度
		// shs,					// 每个3D gaussian的球谐系数, 用于表示颜色
		// clamped,				// 存储每个3D gaussian的R、G、B是否小于0
		// cov3D_precomp,			// 提前计算好的每个3D gaussian的协方差矩阵, []
		// colors_precomp,			// 提前计算好的每个3D gaussian的颜色, []
		viewmatrix, 			// 相机外参矩阵, world to camera
		projmatrix,				// 投影矩阵, world to image
		// cam_pos,				// 所有相机的中心点XYZ坐标
		W, H,					// 图像的宽和高
		tan_fovx, tan_fovy,		// 水平、垂直视场角一半的正切值
		focal_x, focal_y,		// 水平、垂直方向的焦距
		offset,
		radii,					// 存储每个2D gaussian在图像上的半径
		means2D,				// 存储每个2D gaussian的均值
		// depths,					// 存储每个2D gaussian的深度
		cov3Ds,					// 存储每个3D gaussian的协方差矩阵
		// rgb,					// 存储每个2D pixel的颜色
		conics,			// 存储每个2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
		grid,					// 在水平和垂直方向上需要多少个tile来覆盖整个渲染区域
		tiles_touched,			// 存储每个2D gaussian覆盖了多少个tile
		prefiltered				// 是否预先过滤掉了中心点(均值XYZ)不在视锥（frustum）内的3D gaussian
		);
}