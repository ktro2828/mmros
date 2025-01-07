// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#ifndef MMROS__TENSORRT__PLUGIN__COMMON__NMS__PERMUTE_HPP_
#define MMROS__TENSORRT__PLUGIN__COMMON__NMS__PERMUTE_HPP_

#include "mmros/tensorrt/plugin/common/trt_plugin_helper.hpp"

#include <NvInferRuntimeBase.h>

namespace mmros::plugin
{
pluginStatus_t permuteData(
  cudaStream_t stream, int nthreads, int num_classes, int num_data, int num_dim,
  nvinfer1::DataType DT_DATA, bool confSigmoid, const void * data, void * new_data);
}
#endif  // MMROS__TENSORRT__PLUGIN__COMMON__NMS__PERMUTE_HPP_
