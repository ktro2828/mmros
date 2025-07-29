// Copyright (c) OpenMMLab. All rights reserved.
#ifndef MMROS__TENSORRT__PLUGIN__GATHER_TOPK__GATHER_TOPK_KERNEL_HPP_
#define MMROS__TENSORRT__PLUGIN__GATHER_TOPK__GATHER_TOPK_KERNEL_HPP_

#include <cuda_runtime.h>

#include <cstdint>

namespace mmros::plugin
{
/**
 * @brief Gathers elements from a multi-dimensional input tensor using specified index locations.
 *
 * @tparam scalar_t Data type of the input and output tensors.
 * @param[in] input Pointer to the input tensor data on device memory.
 * @param[in] indices Pointer to the index tensor data on device memory. Each entry specifies a
 * location in the input tensor from which to gather an element.
 * @param[in] dims Pointer to an array describing the shape of the input tensor.
 * @param[in] nbDims Number of dimensions in the input tensor.
 * @param[in] indices_dims Pointer to an array describing the shape of the indices tensor.
 * @param[in] indice_nbDims Number of dimensions in the indices tensor.
 * @param[out] output Pointer to the output buffer on device  memory where gathered elements are
 * stored.
 * @param[in] stream CUDA stream on which to schedule the gather kernel.
 */
template <typename scalar_t>
extern void gather_topk_impl(
  const scalar_t * input, const int * indices, const int64_t * dims, int nbDims,
  const int64_t * indices_dims, int indice_nbDims, scalar_t * output, cudaStream_t stream);
}  // namespace mmros::plugin
#endif  // MMROS__TENSORRT__PLUGIN__GATHER_TOPK__GATHER_TOPK_KERNEL_HPP_
