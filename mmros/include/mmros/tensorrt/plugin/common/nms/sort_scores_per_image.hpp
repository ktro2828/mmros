// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#ifndef MMROS__TENSORRT__PLUGIN__COMMON__NMS__SORT_SCORES_PER_IMAGE_HPP_
#define MMROS__TENSORRT__PLUGIN__COMMON__NMS__SORT_SCORES_PER_IMAGE_HPP_

#include "mmros/tensorrt/plugin/common/trt_plugin_helper.hpp"

#include <NvInferRuntimeBase.h>

#include <cstddef>

namespace mmros::plugin
{
pluginStatus_t sortScoresPerImage(
  cudaStream_t stream, int num_images, int num_items_per_image, nvinfer1::DataType DT_SCORE,
  void * unsorted_scores, void * unsorted_bbox_indices, void * sorted_scores,
  void * sorted_bbox_indices, void * workspace);

size_t sortScoresPerImageWorkspaceSize(
  int num_images, int num_items_per_image, nvinfer1::DataType DT_SCORE);
}  // namespace mmros::plugin
#endif  // MMROS__TENSORRT__PLUGIN__COMMON__NMS__SORT_SCORES_PER_IMAGE_HPP_
