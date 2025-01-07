// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#ifndef MMROS__TENSORRT__PLUGIN__BATCHED_NMS__TRT_BATCHED_NMS_HPP_
#define MMROS__TENSORRT__PLUGIN__BATCHED_NMS__TRT_BATCHED_NMS_HPP_

#include "mmros/tensorrt/plugin/common/trt_plugin_base.hpp"

#include <NvInferPluginUtils.h>
#include <NvInferRuntime.h>

#include <string>

namespace mmros::plugin
{
enum NMSReturnType { RETURN_DETS = 1, RETURN_INDEX = 1 << 1 };

class TRTBatchedNMS : public TRTPluginBase
{
public:
  TRTBatchedNMS(const std::string & name, nvinfer1::plugin::NMSParameters param, bool returnIndex);

  TRTBatchedNMS(const std::string & name, const void * data, size_t length);

  ~TRTBatchedNMS() TRT_NOEXCEPT override = default;

  int32_t getNbOutputs() const TRT_NOEXCEPT override;

  nvinfer1::DimsExprs getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs * inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder & exprBuilder) TRT_NOEXCEPT override;

  size_t getWorkspaceSize(
    const nvinfer1::PluginTensorDesc * inputs, int32_t nbInputs,
    const nvinfer1::PluginTensorDesc * outputs, int32_t nbOutputs) const TRT_NOEXCEPT override;

  int32_t enqueue(
    const nvinfer1::PluginTensorDesc * inputDesc, const nvinfer1::PluginTensorDesc * outputDesc,
    const void * const * inputs, void * const * outputs, void * workSpace,
    cudaStream_t stream) TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override;

  void serialize(void * buffer) const TRT_NOEXCEPT override;

  void configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc * inputs, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * outputs, int32_t nbOutputs) TRT_NOEXCEPT override;

  bool supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc * ioDesc, int32_t nbInputs,
    int32_t nbOutputs) TRT_NOEXCEPT override;

  const char * getPluginType() const TRT_NOEXCEPT override;

  const char * getPluginVersion() const TRT_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt * clone() const TRT_NOEXCEPT override;

  nvinfer1::DataType getOutputDataType(
    int32_t index, const nvinfer1::DataType * inputType,
    int32_t nbInputs) const TRT_NOEXCEPT override;

  void setClipParam(bool clip);

private:
  nvinfer1::plugin::NMSParameters param{};
  bool mClipBoxes{};
  bool mReturnIndex{};
};

class TRTBatchedNMSCreator : public TRTPluginCreatorBase
{
public:
  TRTBatchedNMSCreator();

  ~TRTBatchedNMSCreator() TRT_NOEXCEPT override = default;

  const char * getPluginName() const TRT_NOEXCEPT override;

  const char * getPluginVersion() const TRT_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt * createPlugin(
    const char * name, const nvinfer1::PluginFieldCollection * fc) TRT_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt * deserializePlugin(
    const char * name, const void * serialData, size_t serialLength) TRT_NOEXCEPT override;
};
}  // namespace mmros::plugin
#endif  // MMROS__TENSORRT__PLUGIN__BATCHED_NMS__TRT_BATCHED_NMS_HPP_
