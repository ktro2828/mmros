// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMROS__TENSORRT__PLUGIN__MULTI_LEVEL_ROI_ALIGN__MULTI_LEVEL_ROI_ALIGN_KERNEL_HPP_
#define MMROS__TENSORRT__PLUGIN__MULTI_LEVEL_ROI_ALIGN__MULTI_LEVEL_ROI_ALIGN_KERNEL_HPP_
#include <cuda_runtime.h>

namespace mmros::plugin
{
/**
 * @brief Applies multi-level ROI (Region of Interest) alignment across multiple feature maps.
 *
 * @tparam T Data type of features and output.
 * @param[out] output Pointer to device memory where aligned ROIs will be written. The expected size
 * is @p num_rois * @p c * @p aligned_height * @p aligned_width.
 * @param[in] rois Pointer to device memory containing ROI coordinates.
 * @param[in] num_rois Total number of ROIs to process.
 * @param[in] feats Pointer to an array of pointers to the feature maps. Each feature map in
 * @p feats should be contiguous in memory.
 * @param[in] num_feats Number of feature maps in @p feats`.
 * @param[in] n Batch size.
 * @param[in] c Number of channels in each feature map.
 * @param[in] h Pointer to an array of length @p num_feats indicating the height for each feature
 * map level. For example `h[0]` is the height of `feats[0]`.
 * @param[in] w Pointer to an array of length @p num_feats indicating the width for each feature map
 * level. For example `w[0]` is the width of `feats[0]`.
 * @param[in] strides Pointer to an array of length `num_feats` specifying the stride between each
 * feature map level and the original image resolution.
 * @param[in] aligned_height The desired output height for each pooled ROI.
 * @param[in] aligned_width The desired output width for each pooled ROI.
 * @param[in] pool_mode Specifies the pooing mode.
 * @param[in] sample_num Number of samples used in each bin for interpolation.
 * @param[in] roi_scale_factor Additional scale factor for ROIs.
 * @param[in] finest_scale A threshold used to decide which level of feature map to use for a given
 * ROI.
 * @param[in] aligned If true, aligns the interpolation points to the pixel centers.
 * @param[in] stream CUDA stream on which the kernel is launched.
 */
template <typename T>
extern void multi_level_roi_align(
  T * output, const T * rois, int num_rois, const void * const * feats, int num_feats, int n, int c,
  int * h, int * w, float * strides, int aligned_height, int aligned_width, int pool_mode,
  int sample_num, float roi_scale_factor, int finest_scale, bool aligned, cudaStream_t stream);
}  // namespace mmros::plugin

#endif  // MMROS__TENSORRT__PLUGIN__MULTI_LEVEL_ROI_ALIGN__MULTI_LEVEL_ROI_ALIGN_KERNEL_HPP_
