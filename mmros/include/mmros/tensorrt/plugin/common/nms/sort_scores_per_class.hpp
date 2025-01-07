// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#ifndef MMROS__TENSORRT__PLUGIN__COMMON__NMS__SORT_SCORES_PER_CLASS_HPP_
#define MMROS__TENSORRT__PLUGIN__COMMON__NMS__SORT_SCORES_PER_CLASS_HPP_

#include "mmros/tensorrt/plugin/common/trt_plugin_helper.hpp"

#include <NvInferRuntimeBase.h>

#include <cstddef>

namespace mmros::plugin
{
pluginStatus_t sortScoresPerClass(
  cudaStream_t stream, int num, int num_classes, int num_preds_per_class, int background_label_id,
  float confidence_threshold, nvinfer1::DataType DT_SCORE, void * conf_scores_gpu,
  void * index_array_gpu, void * workspace);

size_t sortScoresPerClassWorkspaceSize(
  int num, int num_classes, int num_preds_per_class, nvinfer1::DataType DT_CONF);
}  // namespace mmros::plugin
#endif  // MMROS__TENSORRT__PLUGIN__COMMON__NMS__SORT_SCORES_PER_CLASS_HPP_
