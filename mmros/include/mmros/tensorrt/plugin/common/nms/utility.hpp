// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#ifndef MMROS__TENSORRT__PLUGIN__COMMON__NMS__UTILITY_HPP_
#define MMROS__TENSORRT__PLUGIN__COMMON__NMS__UTILITY_HPP_

#include "mmros/tensorrt/plugin/common/trt_plugin_helper.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdio>

#define DEBUG_ENABLE 0

namespace mmros::plugin
{
template <typename T>
struct Bbox
{
  T xmin, ymin, xmax, ymax;
  Bbox(T xmin, T ymin, T xmax, T ymax) : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax) {}
  Bbox() = default;
};

/**
 * @brief Get CUDA architecture as `<MAJOR> * 100 + <MINOR> * 10`.
 *
 * @param[in] devID Device ID.
 * @return size_t CUDA architecture.
 */
size_t get_cuda_arch(int devID);

int8_t * alignPtr(int8_t * ptr, uintptr_t to);

int8_t * nextWorkspacePtr(int8_t * ptr, uintptr_t previousWorkspaceSize);

void setUniformOffsets(cudaStream_t stream, int num_segments, int offset, int * d_offsets);

size_t detectionForwardBBoxDataSize(int N, int C1, nvinfer1::DataType DT_BBOX);

size_t detectionForwardBBoxPermuteSize(
  bool shareLocation, int N, int C1, nvinfer1::DataType DT_BBOX);

size_t calculateTotalWorkspaceSize(size_t * workspaces, int count);

size_t detectionForwardPreNMSSize(int N, int C2);

size_t detectionForwardPostNMSSize(int N, int numClasses, int topK);

size_t detectionInferenceWorkspaceSize(
  bool shareLocation, int N, int C1, int C2, int numClasses, int numPredsPerClass, int topK,
  nvinfer1::DataType DT_BBOX, nvinfer1::DataType DT_SCORE);
}  // namespace mmros::plugin
#endif  // MMROS__TENSORRT__PLUGIN__COMMON__NMS__UTILITY_HPP_
