// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#ifndef MMROS__TENSORRT__PLUGIN__BATCHED_NMS__BATCHED_NMS_KERNEL_HPP_
#define MMROS__TENSORRT__PLUGIN__BATCHED_NMS__BATCHED_NMS_KERNEL_HPP_

#include "mmros/tensorrt/plugin/common/trt_plugin_helper.hpp"

#include <NvInferRuntimeBase.h>
#include <cuda_runtime_api.h>

namespace mmros::plugin
{
/**
 * @brief Performs Non-Maximum SUppression (NMS) on a batch of bounding boxes.
 *
 * This function applies NMS on bounding boxes and confidence scores to filter out overlapping
 * low-confidence detections. The function writes the resulting bounding boxes, labels, and indices
 * of retained detections into the provided output buffers.
 *
 * @param[in] stream The CUDA stream on which this operation will be executed.
 * @param[in] N The batch size.
 * @param[in] perBatchBoxesSize The size (in number of elements) of the the boxes for each batch.
 * @param[in] perBatchScoresSize The size (in number of elements) of the scores for each batch.
 * @param[in] shareLocation Indicates whether boxes are shared across different classes.
 * @param[in] backgroundLabelId The label ID
 * @param[in] numPredsPerClass The number of predictions per class.
 * @param[in] numClasses The number classes.
 * @param[in] topK The maximum number of boxes to be considered before NMS (per batch and per
 * class).
 * @param[in] keepTopK The minimum number of detections to be kept after NMD (per batch).
 * @param[in] scoreThreshold The minimum confidence threshold for a box to be considered in NMS.
 * @param[in] iouThreshold The IoU (Intersection over Union) threshold used for suppressing
 * overlapping boxes.
 * @param[in] DT_BBOX The data type of the bounding box input (@p locData).
 * @param[in] locData Pointer to the memory containing bounding box data.
 * @param[in] DT_SCORE The data type of the confidence score (@p condData).
 * @param[in] confData Pointer to the memory containing confidence scores.
 * @param[out] nmsedDets Pointer to the output buffer for NMS-filtered bounding boxes.
 * @param[out] nmsedLabels Pointer to the output buffer for class labels of retained detections.
 * @param[out] nmsedIndex Pointer the output buffer for the original indices of retained detections.
 * @param[in,out] workspace Pointer to the workspace buffer required for temporary storage during
 * NMS.
 * @param[in] isNormalized Indicates whether the box coordinates are in a normalized format ([0,1]
 * range).
 * @param[in] confSigmoid Indicates whether to apply a sigmoid functions to the confidence scores
 * before NMS.
 * @param[in] clipBoxes Indicates whether to clip the boxes to the image boundary.
 * @param[in] rotated Indicates whether the boxes are rotated (default is @c false).
 * @return pluginStatus_t A status code of type \c pluginStatus_t indicating the operation succeeded
 * or not.
 */
extern pluginStatus_t nmsInference(
  cudaStream_t stream, const int N, const int perBatchBoxesSize, const int perBatchScoresSize,
  const bool shareLocation, const int backgroundLabelId, const int numPredsPerClass,
  const int numClasses, const int topK, const int keepTopK, const float scoreThreshold,
  const float iouThreshold, const nvinfer1::DataType DT_BBOX, const void * locData,
  const nvinfer1::DataType DT_SCORE, const void * confData, void * nmsedDets, void * nmsedLabels,
  void * nmsedIndex, void * workspace, bool isNormalized, bool confSigmoid, bool clipBoxes,
  bool rotated = false);
}  // namespace mmros::plugin
#endif  // MMROS__TENSORRT__PLUGIN__BATCHED_NMS__BATCHED_NMS_KERNEL_HPP_
