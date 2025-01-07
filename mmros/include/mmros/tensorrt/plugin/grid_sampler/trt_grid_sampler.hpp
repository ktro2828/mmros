// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMROS__TENSORRT__PLUGIN__GRID_SAMPLER__TRT_GRID_SAMPLER_HPP_
#define MMROS__TENSORRT__PLUGIN__GRID_SAMPLER__TRT_GRID_SAMPLER_HPP_

#include "mmros/tensorrt/plugin/common/trt_plugin_base.hpp"

#include <NvInferRuntime.h>
#include <cublas_v2.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mmros::plugin
{
class TRTGridSampler : public TRTPluginBase
{
public:
  TRTGridSampler(const std::string & name, int mode, int paddingMode, bool alignCorners);

  TRTGridSampler(const std::string name, const void * data, size_t length);

  TRTGridSampler() = delete;

  ~TRTGridSampler() TRT_NOEXCEPT override = default;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt * clone() const TRT_NOEXCEPT override;

  nvinfer1::DimsExprs getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs * inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder & exprBuilder) TRT_NOEXCEPT override;

  bool supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc * ioDesc, int32_t nbInputs,
    int32_t nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc * in, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * out, int32_t nbOutputs) TRT_NOEXCEPT override;

  size_t getWorkspaceSize(
    const nvinfer1::PluginTensorDesc * inputs, int32_t nbInputs,
    const nvinfer1::PluginTensorDesc * outputs, int32_t nbOutputs) const TRT_NOEXCEPT override;

  int32_t enqueue(
    const nvinfer1::PluginTensorDesc * inputDesc, const nvinfer1::PluginTensorDesc * outputDesc,
    const void * const * inputs, void * const * outputs, void * workspace,
    cudaStream_t stream) TRT_NOEXCEPT override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(
    int32_t index, const nvinfer1::DataType * inputTypes,
    int32_t nbInputs) const TRT_NOEXCEPT override;

  // IPluginV2 Methods
  const char * getPluginType() const TRT_NOEXCEPT override;

  const char * getPluginVersion() const TRT_NOEXCEPT override;

  int32_t getNbOutputs() const TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override;

  void serialize(void * buffer) const TRT_NOEXCEPT override;

private:
  int mMode;
  int mPaddingMode;
  bool mAlignCorners;
};

class TRTGridSamplerCreator : public TRTPluginCreatorBase
{
public:
  TRTGridSamplerCreator();

  ~TRTGridSamplerCreator() TRT_NOEXCEPT override = default;

  const char * getPluginName() const TRT_NOEXCEPT override;

  const char * getPluginVersion() const TRT_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt * createPlugin(
    const char * name, const nvinfer1::PluginFieldCollection * fc) TRT_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt * deserializePlugin(
    const char * name, const void * serialData, size_t serialLength) TRT_NOEXCEPT override;
};
}  // namespace mmros::plugin
#endif  // MMROS__TENSORRT__PLUGIN__GRID_SAMPLER__TRT_GRID_SAMPLER_HPP_
