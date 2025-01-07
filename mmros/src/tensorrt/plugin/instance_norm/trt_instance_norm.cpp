// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// Modified from:
// https://github.com/NVIDIA/TensorRT/blob/master/plugin/instanceNormalizationPlugin/instanceNormalizationPlugin.cpp

#include "mmros/tensorrt/plugin/instance_norm/trt_instance_norm.hpp"

#include "mmros/tensorrt/plugin/common/trt_plugin_helper.hpp"
#include "mmros/tensorrt/plugin/common/trt_serialize.hpp"

namespace mmros::plugin
{
namespace
{
constexpr const char * PLUGIN_VERSION{"1"};
constexpr const char * PLUGIN_NAME{"TRTInstanceNormalization"};
}  // namespace

TRTInstanceNormalization::TRTInstanceNormalization(const std::string & name, float epsilon)
: TRTPluginBase(name), mEpsilon(epsilon)
{
}

TRTInstanceNormalization::TRTInstanceNormalization(
  const std::string & name, void const * serialData, size_t serialLength)
: TRTPluginBase(name)
{
  deserialize_value(&serialData, &serialLength, &mEpsilon);
}

// TRTInstanceNormalization returns one output.
int32_t TRTInstanceNormalization::getNbOutputs() const TRT_NOEXCEPT
{
  return 1;
}

nvinfer1::DimsExprs TRTInstanceNormalization::getOutputDimensions(
  int32_t outputIndex, const nvinfer1::DimsExprs * inputs, int32_t nbInputs,
  nvinfer1::IExprBuilder & exprBuilder) TRT_NOEXCEPT
{
  nvinfer1::DimsExprs output(inputs[0]);
  return output;
}

size_t TRTInstanceNormalization::getWorkspaceSize(
  const nvinfer1::PluginTensorDesc * inputs, int32_t nbInputs,
  const nvinfer1::PluginTensorDesc * outputs, int32_t nbOutputs) const TRT_NOEXCEPT
{
  int n = inputs[0].dims.d[0];
  int c = inputs[0].dims.d[1];
  int elem_size = sizeof(float);
  return getAlignedSize(n * c * elem_size) * 2;
}

int32_t TRTInstanceNormalization::enqueue(
  const nvinfer1::PluginTensorDesc * inputDesc, const nvinfer1::PluginTensorDesc * outputDesc,
  const void * const * inputs, void * const * outputs, void * workspace,
  cudaStream_t stream) TRT_NOEXCEPT
{
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  int32_t n = input_dims.d[0];
  int32_t c = input_dims.d[1];
  int32_t h = input_dims.d[2];
  int32_t w = input_dims.nbDims > 3 ? input_dims.d[3] : 1;
  int32_t elem_size = sizeof(float);

  void * n_scales = (void *)workspace;
  void * n_bias = (void *)((char *)workspace + getAlignedSize(n * c * elem_size));

  const void * scales = (const void *)inputs[1];
  const void * bias = (const void *)inputs[2];

  for (int32_t i = 0; i < n; ++i) {
    cudaMemcpyAsync(
      (char *)n_scales + i * c * elem_size, scales, c * elem_size, cudaMemcpyDeviceToDevice,
      stream);
    cudaMemcpyAsync(
      (char *)n_bias + i * c * elem_size, bias, c * elem_size, cudaMemcpyDeviceToDevice, stream);
  }

  cudnnSetTensor4dDescriptor(_b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n * c, 1, 1);
  cudnnDataType_t cudnn_dtype{};
  convert_trt2cudnn_dtype(inputDesc[0].type, &cudnn_dtype);
  cudnnSetTensor4dDescriptor(_x_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w);
  cudnnSetTensor4dDescriptor(_y_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w);
  float alpha = 1;
  float beta = 0;
  void const * x_ptr = inputs[0];
  void * y_ptr = outputs[0];
  cudnnSetStream(_cudnn_handle, stream);
  // Note: Use of CUDNN_BATCHNORM_SPATIAL_PERSISTENT can cause numerical
  //       overflows (NaNs) for fp32 data in some circumstances. The lower-
  //       performance CUDNN_BATCHNORM_SPATIAL should be used if this is not
  //       acceptable.
  cudnnBatchNormalizationForwardTraining(
    _cudnn_handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha, &beta, _x_desc, x_ptr, _y_desc,
    y_ptr, _b_desc, n_scales, n_bias, 1., nullptr, nullptr, mEpsilon, nullptr, nullptr);
  return 0;
}

size_t TRTInstanceNormalization::getSerializationSize() const TRT_NOEXCEPT
{
  return serialized_size(mEpsilon);
}

void TRTInstanceNormalization::serialize(void * buffer) const TRT_NOEXCEPT
{
  serialize_value(&buffer, mEpsilon);
}

bool TRTInstanceNormalization::supportsFormatCombination(
  int32_t pos, const nvinfer1::PluginTensorDesc * ioDesc, int32_t nbInputs,
  int32_t nbOutputs) TRT_NOEXCEPT
{
  switch (pos) {
    case 0:
    case 3:
      return (
        (ioDesc[pos].type == nvinfer1::DataType::kFLOAT ||
         ioDesc[pos].type == nvinfer1::DataType::kHALF) &&
        ioDesc[pos].format == nvinfer1::PluginFormat::kLINEAR &&
        ioDesc[pos].type == ioDesc[0].type);
    case 1:
    case 2:
      return ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
             ioDesc[pos].format == nvinfer1::PluginFormat::kLINEAR;
    default:
      return false;
  }
  return false;
}

const char * TRTInstanceNormalization::getPluginType() const TRT_NOEXCEPT
{
  return PLUGIN_NAME;
}

const char * TRTInstanceNormalization::getPluginVersion() const TRT_NOEXCEPT
{
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2DynamicExt * TRTInstanceNormalization::clone() const TRT_NOEXCEPT
{
  auto * plugin = new TRTInstanceNormalization{mLayerName, mEpsilon};
  plugin->setPluginNamespace(mPluginNamespace.c_str());
  return plugin;
}

nvinfer1::DataType TRTInstanceNormalization::getOutputDataType(
  int32_t index, const nvinfer1::DataType * inputTypes, int32_t nbInputs) const TRT_NOEXCEPT
{
  return inputTypes[0];
}

// Attach the plugin object to an execution context and grant the plugin the
// access to some context resource.
void TRTInstanceNormalization::attachToContext(
  cudnnContext * cudnnContext, cublasContext * cublasContext,
  nvinfer1::IGpuAllocator * gpuAllocator) TRT_NOEXCEPT
{
  _cudnn_handle = cudnnContext;
  cudnnCreateTensorDescriptor(&_b_desc);
  cudnnCreateTensorDescriptor(&_x_desc);
  cudnnCreateTensorDescriptor(&_y_desc);
}

// Detach the plugin object from its execution context.
void TRTInstanceNormalization::detachFromContext() TRT_NOEXCEPT
{
  if (_y_desc) {
    cudnnDestroyTensorDescriptor(_y_desc);
    _y_desc = nullptr;
  }
  if (_x_desc) {
    cudnnDestroyTensorDescriptor(_x_desc);
    _x_desc = nullptr;
  }
  if (_b_desc) {
    cudnnDestroyTensorDescriptor(_b_desc);
    _b_desc = nullptr;
  }
}

void TRTInstanceNormalization::configurePlugin(
  const nvinfer1::DynamicPluginTensorDesc * in, int32_t nbInputs,
  const nvinfer1::DynamicPluginTensorDesc * out, int32_t nbOutputs) TRT_NOEXCEPT
{
}

// TRTInstanceNormalizationCreator methods
TRTInstanceNormalizationCreator::TRTInstanceNormalizationCreator()
{
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("epsilon", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char * TRTInstanceNormalizationCreator::getPluginName() const TRT_NOEXCEPT
{
  return PLUGIN_NAME;
}

const char * TRTInstanceNormalizationCreator::getPluginVersion() const TRT_NOEXCEPT
{
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2DynamicExt * TRTInstanceNormalizationCreator::createPlugin(
  const char * name, const nvinfer1::PluginFieldCollection * fc) TRT_NOEXCEPT
{
  float epsilon = 1e-5;
  const nvinfer1::PluginField * fields = fc->fields;
  for (int32_t i = 0; i < fc->nbFields; ++i) {
    const char * attrName = fields[i].name;
    if (!strcmp(attrName, "epsilon")) {
      epsilon = *(static_cast<const float *>(fields[i].data));
    }
  }

  TRTInstanceNormalization * obj = new TRTInstanceNormalization(name, epsilon);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

nvinfer1::IPluginV2DynamicExt * TRTInstanceNormalizationCreator::deserializePlugin(
  const char * name, const void * serialData, size_t serialLength) TRT_NOEXCEPT
{
  TRTInstanceNormalization * obj = new TRTInstanceNormalization{name, serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

REGISTER_TENSORRT_PLUGIN(TRTInstanceNormalizationCreator);
}  // namespace mmros::plugin
