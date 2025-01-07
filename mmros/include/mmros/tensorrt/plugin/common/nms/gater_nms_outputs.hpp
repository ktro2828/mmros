// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#ifndef MMROS__TENSORRT__PLUGIN__COMMON__NMS__GATER_NMS_OUTPUTS_HPP_
#define MMROS__TENSORRT__PLUGIN__COMMON__NMS__GATER_NMS_OUTPUTS_HPP_

#include "mmros/tensorrt/plugin/common/trt_plugin_helper.hpp"

#include <NvInferRuntimeBase.h>

namespace mmros::plugin
{
pluginStatus_t gatherNMSOutputs(
  cudaStream_t stream, bool shareLocation, int numImages, int numPredsPerClass, int numClasses,
  int topK, int keepTopK, nvinfer1::DataType DT_BBOX, nvinfer1::DataType DT_SCORE,
  const void * indices, const void * scores, const void * bboxData, void * nmsedDets,
  void * nmsedLabels, void * nmsedIndex = nullptr, bool clipBoxes = true, bool rotated = false);
}
#endif  // MMROS__TENSORRT__PLUGIN__COMMON__NMS__GATER_NMS_OUTPUTS_HPP_
