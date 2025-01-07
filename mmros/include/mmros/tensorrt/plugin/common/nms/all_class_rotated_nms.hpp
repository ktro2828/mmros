// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#ifndef MMROS__TENSORRT__PLUGIN__COMMON__NMS__ALL_CLASS_ROTATED_NMS_HPP_
#define MMROS__TENSORRT__PLUGIN__COMMON__NMS__ALL_CLASS_ROTATED_NMS_HPP_

#include "mmros/tensorrt/plugin/common/trt_plugin_helper.hpp"

#include <NvInferRuntimeBase.h>

namespace mmros::plugin
{
pluginStatus_t allClassRotatedNMS(
  cudaStream_t stream, int num, int num_classes, int num_preds_per_class, int top_k,
  float nms_threshold, bool share_location, bool isNormalized, nvinfer1::DataType DT_SCORE,
  nvinfer1::DataType DT_BBOX, void * bbox_data, void * beforeNMS_scores,
  void * beforeNMS_index_array, void * afterNMS_scores, void * afterNMS_index_array,
  bool flipXY = false);
}
#endif  // MMROS__TENSORRT__PLUGIN__COMMON__NMS__ALL_CLASS_ROTATED_NMS_HPP_
