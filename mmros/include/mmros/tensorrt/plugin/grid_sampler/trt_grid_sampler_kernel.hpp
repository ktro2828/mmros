// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMROS__TENSORRT__PLUGIN__GRID_SAMPLER__TRT_GRID_SAMPLER_KERNEL_HPP_
#define MMROS__TENSORRT__PLUGIN__GRID_SAMPLER__TRT_GRID_SAMPLER_KERNEL_HPP_

#include <cuda_runtime.h>

#include <cstdint>

namespace mmros::plugin
{
/**
 * @brief Specifies the interpolation method used for grid sampling.
 */
enum class GridSamplerInterpolation {
  Bilinear,  //!< Bilinear interpolation.
  Nearest    //!< Nearest-neighbor interpolation.
};

/**
 * @brief Specifies the padding method used when the sampling index is out of range.
 */
enum class GridSamplerPadding {
  Zeros,      //!< Pads out-of-range elements with zeros.
  Border,     //!< Uses the value at the border for out-of-range elements.
  Reflection  //!< Reflects the index back into the valid range.
};

/**
 * @brief Samples an input tensor according to the coordinates in a grid tensor.
 *
 * This function applies a grid-based sampling of the @p input tensor to produce an @p output
 * tensor. The sampling grid is specified by @p grid. The dimensional information of the output,
 * input, and grid tensors are provided via @p output_dims, @p input_dims, and @p grid_dims,
 * respectively.
 *
 * @tparam T The data type of the input, grid and output.
 * @param[out] output Pointer to the memory where the sampled output will be stored.
 * @param[in] input Pointer to the input tensor to be sampled.
 * @param[in] grid Pointer to the grid tensor containing the sampling coordinates.
 * @param[in] output_dims Pointer to an array describing the dimensions of the output tensor.
 * @param[in] input_dims Pointer to an array describing the dimensions of the input tensor.
 * @param[in] grid_dims Pointer to an array describing the dimensions of the grid tensor.
 * @param[in] nb_dims The number of dimensions to consider for both input and output tensors.
 * @param[in] interp The interpolation method.
 * @param[in] padding THe padding mode to use when sampling out of range.
 * @param[in] align_corners If @c true, the corner points of the grid are aligned with the boundary
 * of the input tensor.
 * @param[in] stream The CUDA stream on which to run this operation.
 */
template <typename T>
void grid_sample(
  T * output, const T * input, const T * grid, int64_t * output_dims, int64_t * input_dims,
  int64_t * grid_dims, int nb_dims, GridSamplerInterpolation interp, GridSamplerPadding padding,
  bool align_corners, cudaStream_t stream);
}  // namespace mmros::plugin
#endif  // MMROS__TENSORRT__PLUGIN__GRID_SAMPLER__TRT_GRID_SAMPLER_KERNEL_HPP_
