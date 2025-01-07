// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// Modified from:
// https://github.com/NVIDIA/TensorRT/blob/master/plugin/instanceNormalizationPlugin/instanceNormalizationPlugin.h

#ifndef MMROS__TENSORRT__PLUGIN__INSTANCE_NORM__TRT_INSTANCE_NORM_HPP_
#define MMROS__TENSORRT__PLUGIN__INSTANCE_NORM__TRT_INSTANCE_NORM_HPP_

#include "mmros/tensorrt/plugin/common/trt_plugin_base.hpp"

#include <string>

namespace mmros::plugin
{
class TRTInstanceNormalization : public TRTPluginBase
{
public:
  TRTInstanceNormalization(const std::string & name, float epsilon);

  TRTInstanceNormalization(const std::string & name, void const * serialData, size_t seralLength);

  TRTInstanceNormalization() = delete;

  ~TRTInstanceNormalization() TRT_NOEXCEPT override = default;

  int32_t getNbOutputs() const TRT_NOEXCEPT override;

  // DynamicExt plugins returns DimsExprs class instead of Dims
  nvinfer1::DimsExprs getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs * inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder & exprBuilder) TRT_NOEXCEPT override;

  size_t getWorkspaceSize(
    const nvinfer1::PluginTensorDesc * inputs, int32_t nbInputs,
    const nvinfer1::PluginTensorDesc * outputs, int32_t nbOutputs) const TRT_NOEXCEPT override;

  int32_t enqueue(
    const nvinfer1::PluginTensorDesc * inputDesc, const nvinfer1::PluginTensorDesc * outputDesc,
    const void * const * inputs, void * const * outputs, void * workspace,
    cudaStream_t stream) TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override;

  void serialize(void * buffer) const TRT_NOEXCEPT override;

  // DynamicExt plugin supportsFormat update.
  bool supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc * ioDesc, int32_t nbInputs,
    int32_t nbOutputs) TRT_NOEXCEPT override;

  const char * getPluginType() const TRT_NOEXCEPT override;

  const char * getPluginVersion() const TRT_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt * clone() const TRT_NOEXCEPT override;

  nvinfer1::DataType getOutputDataType(
    int32_t index, const nvinfer1::DataType * inputTypes,
    int32_t nbInputs) const TRT_NOEXCEPT override;

  void attachToContext(
    cudnnContext * cudnn, cublasContext * cublas,
    nvinfer1::IGpuAllocator * allocator) TRT_NOEXCEPT override;

  void detachFromContext() TRT_NOEXCEPT override;

  void configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc * in, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * out, int32_t nbOutputs) TRT_NOEXCEPT override;

private:
  float mEpsilon{};
  cudnnHandle_t _cudnn_handle{};
  cudnnTensorDescriptor_t _x_desc{}, _y_desc{}, _b_desc{};
  std::string mPluginNamespace{};
};

class TRTInstanceNormalizationCreator : public TRTPluginCreatorBase
{
public:
  TRTInstanceNormalizationCreator();

  ~TRTInstanceNormalizationCreator() override = default;

  const char * getPluginName() const TRT_NOEXCEPT override;

  const char * getPluginVersion() const TRT_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt * createPlugin(
    const char * name, const nvinfer1::PluginFieldCollection * fc) TRT_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt * deserializePlugin(
    const char * name, const void * serialData, size_t serialLength) TRT_NOEXCEPT override;
};
}  // namespace mmros::plugin
#endif  // MMROS__TENSORRT__PLUGIN__INSTANCE_NORM__TRT_INSTANCE_NORM_HPP_
