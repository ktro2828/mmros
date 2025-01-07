// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#ifndef MMROS__TENSORRT__PLUGIN__COMMON__NMS__ALL_CLASS_NMS_HPP_
#define MMROS__TENSORRT__PLUGIN__COMMON__NMS__ALL_CLASS_NMS_HPP_

#include "mmros/tensorrt/plugin/common/trt_plugin_helper.hpp"

#include <NvInferRuntimeBase.h>

namespace mmros::plugin
{
pluginStatus_t allClassNMS(
  cudaStream_t stream, const int num, const int num_classes, const int num_preds_per_class,
  const int top_k, const float nms_threshold, const bool share_location, const bool isNormalized,
  const nvinfer1::DataType DT_SCORE, const nvinfer1::DataType DT_BBOX, void * bbox_data,
  void * beforeNMS_scores, void * beforeNMS_index_array, void * afterNMS_scores,
  void * afterNMS_index_array, bool flipXY);
}
#endif  // MMROS__TENSORRT__PLUGIN__COMMON__NMS__ALL_CLASS_NMS_HPP_
