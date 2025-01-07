// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMROS__TENSORRT__PLUGIN__COMMON__TRT_PLUGIN_BASE_HPP_
#define MMROS__TENSORRT__PLUGIN__COMMON__TRT_PLUGIN_BASE_HPP_

#include "mmros/tensorrt/plugin/common/trt_plugin_helper.hpp"

#include <NvInferRuntime.h>
#include <NvInferVersion.h>

#include <cstdint>
#include <string>
#include <vector>

namespace mmros::plugin
{
#if NV_TENSORRT_MAJOR > 7
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif

class TRTPluginBase : public nvinfer1::IPluginV2DynamicExt
{
public:
  explicit TRTPluginBase(const std::string & name) : mLayerName(name) {}

  // IPluginV2 Methods
  const char * getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  int initialize() TRT_NOEXCEPT override { return STATUS_SUCCESS; }

  void terminate() TRT_NOEXCEPT override {}

  void destroy() TRT_NOEXCEPT override { delete this; }

  void setPluginNamespace(const char * pluginNamespace) TRT_NOEXCEPT override
  {
    mNamespace = pluginNamespace;
  }

  const char * getPluginNamespace() const TRT_NOEXCEPT override { return mNamespace.c_str(); }

  void configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc * in, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * out, int32_t nbOutputs) TRT_NOEXCEPT override
  {
  }

  size_t getWorkspaceSize(
    const nvinfer1::PluginTensorDesc * inputs, int32_t nbInputs,
    const nvinfer1::PluginTensorDesc * outputs, int32_t nbOutputs) const TRT_NOEXCEPT override
  {
    return 0;
  }

  void attachToContext(
    cudnnContext * cudnnContext, cublasContext * cublasContext,
    nvinfer1::IGpuAllocator * gpuAllocator) TRT_NOEXCEPT override
  {
  }

  void detachFromContext() TRT_NOEXCEPT override {}

protected:
  const std::string mLayerName;
  std::string mNamespace;
};

class TRTPluginCreatorBase : public nvinfer1::IPluginCreator
{
public:
  const char * getPluginVersion() const TRT_NOEXCEPT override { return "1"; };

  const nvinfer1::PluginFieldCollection * getFieldNames() TRT_NOEXCEPT override { return &mFC; }

  void setPluginNamespace(const char * pluginNamespace) TRT_NOEXCEPT override
  {
    mNamespace = pluginNamespace;
  }

  const char * getPluginNamespace() const TRT_NOEXCEPT override { return mNamespace.c_str(); }

protected:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};
}  // namespace mmros::plugin
#endif  // MMROS__TENSORRT__PLUGIN__COMMON__TRT_PLUGIN_BASE_HPP_
