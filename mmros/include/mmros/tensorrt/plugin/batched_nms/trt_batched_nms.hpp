// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#ifndef MMROS__TENSORRT__PLUGIN__BATCHED_NMS__TRT_BATCHED_NMS_HPP_
#define MMROS__TENSORRT__PLUGIN__BATCHED_NMS__TRT_BATCHED_NMS_HPP_

#include "mmros/tensorrt/plugin/common/trt_plugin_base.hpp"

#include <NvInferPluginBase.h>
#include <NvInferPluginUtils.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeBase.h>
#include <NvInferRuntimePlugin.h>

#include <string>
#include <vector>

namespace mmros::plugin
{
enum NMSReturnType { RETURN_DETS = 1, RETURN_INDEX = 1 << 1 };

//!
//! \brief The NMSParameters are used by the BatchedNMSPlugin for performing
//! the non_max_suppression operation over boxes for object detection networks.
//!
//! \deprecated Deprecated in TensorRT 10.0. BatchedNMSPlugin plugin is deprecated.
//!
struct NMSParameters
{
  bool shareLocation;  //!< If set to true, the boxes inputs are shared across all classes.
                       //!< If set to false, the boxes input should account for per class box data.
  int32_t backgroundLabelId;  //!< Label ID for the background class.
                              //!< If there is no background class, set it as -1
  int32_t numClasses;         //!< Number of classes in the network.
  int32_t topK;               //!< Number of bounding boxes to be fed into the NMS step.
  int32_t keepTopK;      //!< Number of total bounding boxes to be kept per image after NMS step.
                         //!< Should be less than or equal to the topK value.
  float scoreThreshold;  //!< Scalar threshold for score (low scoring boxes are removed).
  float iouThreshold;    //!< A scalar threshold for IOU (new boxes that have high IOU overlap
                         //!< with previously selected boxes are removed).
  bool isNormalized;     //!< Set to false, if the box coordinates are not normalized,
                         //!< i.e. not in the range [0,1]. Defaults to false.
};

class TRTBatchedNMS : public nvinfer1::IPluginV3,
                      public nvinfer1::IPluginV3OneCore,
                      public nvinfer1::IPluginV3OneBuild,
                      public nvinfer1::IPluginV3OneRuntime
{
public:
  TRTBatchedNMS(const std::string & name, NMSParameters param, bool clipBoxes, bool returnIndex)
    TRT_NOEXCEPT;

  ~TRTBatchedNMS() TRT_NOEXCEPT override = default;

  // === IPluginV3 Methods ===
  nvinfer1::IPluginCapability * getCapabilityInterface(nvinfer1::PluginCapabilityType type)
    TRT_NOEXCEPT override;

  nvinfer1::IPluginV3 * clone() TRT_NOEXCEPT override;

  char const * getPluginName() const TRT_NOEXCEPT override;

  char const * getPluginVersion() const TRT_NOEXCEPT override;

  char const * getPluginNamespace() const TRT_NOEXCEPT override;

  // === IPluginV3OneBuild Methods ===
  int32_t getNbOutputs() const TRT_NOEXCEPT override;

  int32_t configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc * inputs, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * outputs, int32_t nbOutputs) TRT_NOEXCEPT override;

  bool supportsFormatCombination(
    int32_t pos, const nvinfer1::DynamicPluginTensorDesc * ioDesc, int32_t nbInputs,
    int32_t nbOutputs) TRT_NOEXCEPT override;

  int32_t getOutputDataTypes(
    nvinfer1::DataType * outputTypes, int32_t nbOutputs, nvinfer1::DataType const * inputTypes,
    int32_t nbInputs) const TRT_NOEXCEPT override;

  int32_t getOutputShapes(
    nvinfer1::DimsExprs const * inputs, int32_t nbInputs, nvinfer1::DimsExprs const * shapeInputs,
    int32_t nbShapeInputs, nvinfer1::DimsExprs * outputs, int32_t nbOutputs,
    nvinfer1::IExprBuilder & exprBuilder) TRT_NOEXCEPT override;

  // === IPluginV3OneRuntime Methods ===
  int32_t enqueue(
    const nvinfer1::PluginTensorDesc * inputDesc, const nvinfer1::PluginTensorDesc * outputDesc,
    const void * const * inputs, void * const * outputs, void * workSpace,
    cudaStream_t stream) TRT_NOEXCEPT override;

  int32_t onShapeChange(
    nvinfer1::PluginTensorDesc const * in, int32_t nbInputs, nvinfer1::PluginTensorDesc const * out,
    int32_t nbOutputs) TRT_NOEXCEPT override;

  nvinfer1::IPluginV3 * attachToContext(nvinfer1::IPluginResourceContext * context)
    TRT_NOEXCEPT override;

  nvinfer1::PluginFieldCollection const * getFieldsToSerialize() TRT_NOEXCEPT override;

  size_t getWorkspaceSize(
    const nvinfer1::DynamicPluginTensorDesc * inputs, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * outputs,
    int32_t nbOutputs) const TRT_NOEXCEPT override;

private:
  std::string mLayerName;
  NMSParameters param{};
  bool mClipBoxes{};
  bool mReturnIndex{};
  std::vector<nvinfer1::PluginField> mFields;
  nvinfer1::PluginFieldCollection mFC;
};

class TRTBatchedNMSCreator : public nvinfer1::IPluginCreatorV3One
{
public:
  TRTBatchedNMSCreator();

  ~TRTBatchedNMSCreator() TRT_NOEXCEPT override = default;

  const char * getPluginNamespace() const TRT_NOEXCEPT override;

  const char * getPluginName() const TRT_NOEXCEPT override;

  const char * getPluginVersion() const TRT_NOEXCEPT override;

  nvinfer1::PluginFieldCollection const * getFieldNames() TRT_NOEXCEPT override;

  nvinfer1::IPluginV3 * createPlugin(
    const char * name, const nvinfer1::PluginFieldCollection * fc,
    nvinfer1::TensorRTPhase phase) TRT_NOEXCEPT override;

private:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
};
}  // namespace mmros::plugin
#endif  // MMROS__TENSORRT__PLUGIN__BATCHED_NMS__TRT_BATCHED_NMS_HPP_
