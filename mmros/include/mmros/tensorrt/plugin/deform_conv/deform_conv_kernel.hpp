// Copyright (c) OpenMMLab. All rights reserved
#ifndef MMROS__TENSORRT__PLUGIN__DEFORM_CONV__DEFORM_CONV_KERNEL_HPP_
#define MMROS__TENSORRT__PLUGIN__DEFORM_CONV__DEFORM_CONV_KERNEL_HPP_

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace mmros::plugin
{
template <typename scalar_t>
extern void deform_conv_im2col(
  const scalar_t * input, const scalar_t * offset, scalar_t * column, const int channels,
  const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h,
  const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
  const int dilation_w, const int parallel_imgs, const int deformable_group, cudaStream_t stream);

template <typename scalar_t>
extern void deform_conv(
  const scalar_t * input, const scalar_t * weight, const scalar_t * offset, scalar_t * output,
  void * workspace, int batchSize, int nInputPlane, int inputHeight, int inputWidth,
  int nOutputPlane, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW,
  int dilationH, int group, int deformable_group, int im2col_step, cublasHandle_t cublas_handle,
  cudaStream_t stream);
}  // namespace mmros::plugin
#endif  // MMROS__TENSORRT__PLUGIN__DEFORM_CONV__DEFORM_CONV_KERNEL_HPP_
