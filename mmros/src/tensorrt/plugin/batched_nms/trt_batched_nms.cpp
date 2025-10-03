// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#include "mmros/tensorrt/plugin/batched_nms/trt_batched_nms.hpp"

#include "mmros/tensorrt/plugin/batched_nms/batched_nms_kernel.hpp"
#include "mmros/tensorrt/plugin/common/nms/utility.hpp"
#include "mmros/tensorrt/plugin/common/trt_plugin_helper.hpp"
#include "mmros/tensorrt/plugin/common/trt_serialize.hpp"

#include <NvInferPluginBase.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeBase.h>

#include <exception>
#include <string>

namespace mmros::plugin
{
namespace
{
static const char * NMS_PLUGIN_VERSION{"1"};           //!< Plugin version.
static const char * NMS_PLUGIN_NAME{"TRTBatchedNMS"};  //!< Plugin name.
static const char * NMS_PLUGIN_NAMESPACE{""};          //!< Plugin namespace.
}  // namespace

TRTBatchedNMS::TRTBatchedNMS(
  const std::string & name, NMSParameters params, bool clipBoxes, bool returnIndex) TRT_NOEXCEPT
: mLayerName(name),
  param(params),
  mClipBoxes(clipBoxes),
  mReturnIndex(returnIndex)
{
  mFields.clear();
  mFC.nbFields = mFields.size();
  mFC.fields = mFields.data();
}

nvinfer1::IPluginCapability * TRTBatchedNMS::getCapabilityInterface(
  nvinfer1::PluginCapabilityType type) TRT_NOEXCEPT
{
  try {
    if (type == nvinfer1::PluginCapabilityType::kBUILD) {
      return static_cast<nvinfer1::IPluginV3OneBuild *>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME) {
      return static_cast<nvinfer1::IPluginV3OneRuntime *>(this);
    }
    ASSERT(type == nvinfer1::PluginCapabilityType::kCORE);
    return static_cast<nvinfer1::IPluginV3OneCore *>(this);
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

nvinfer1::IPluginV3 * TRTBatchedNMS::clone() TRT_NOEXCEPT
{
  return new (std::nothrow) TRTBatchedNMS(mLayerName, param, mClipBoxes, mReturnIndex);
}

char const * TRTBatchedNMS::getPluginName() const TRT_NOEXCEPT
{
  return NMS_PLUGIN_NAME;
}

char const * TRTBatchedNMS::getPluginVersion() const TRT_NOEXCEPT
{
  return NMS_PLUGIN_VERSION;
}

char const * TRTBatchedNMS::getPluginNamespace() const TRT_NOEXCEPT
{
  return NMS_PLUGIN_NAMESPACE;
}

int32_t TRTBatchedNMS::getNbOutputs() const TRT_NOEXCEPT
{
  return mReturnIndex ? 3 : 2;
}

int32_t TRTBatchedNMS::configurePlugin(
  const nvinfer1::DynamicPluginTensorDesc * inputs, int32_t nbInputs,
  const nvinfer1::DynamicPluginTensorDesc * outputs, int32_t nbOutputs) TRT_NOEXCEPT
{
  // check the number of I/O
  ASSERT(nbInputs == 2);
  if (mReturnIndex) {
    ASSERT(nbOutputs == 3);
  } else {
    ASSERT(nbOutputs == 2);
  }

  // check the number of I/O dimensions
  ASSERT(inputs[0].desc.dims.nbDims == 4);
  ASSERT(inputs[1].desc.dims.nbDims == 3);
  ASSERT(outputs[0].desc.dims.nbDims == 3);
  ASSERT(outputs[1].desc.dims.nbDims == 2);
  if (mReturnIndex) {
    ASSERT(outputs[2].desc.dims.nbDims == 2);
  }

  for (auto i = 0; i < nbOutputs; ++i) {
    ASSERT(outputs[i].desc.dims.d[0] == inputs[0].desc.dims.d[0]);  // check batch size
    ASSERT(outputs[i].desc.dims.d[1] == param.keepTopK);            // check topK
  }

  // check the data type of I/O
  ASSERT(inputs[0].desc.type == inputs[1].desc.type);
  ASSERT(outputs[0].desc.type == inputs[0].desc.type);
  ASSERT(
    outputs[1].desc.type == nvinfer1::DataType::kINT32 ||
    outputs[1].desc.type == nvinfer1::DataType::kINT64);
  if (mReturnIndex) {
    ASSERT(
      outputs[2].desc.type == nvinfer1::DataType::kINT32 ||
      outputs[2].desc.type == nvinfer1::DataType::kINT64);
  }

  return 0;
}

bool TRTBatchedNMS::supportsFormatCombination(
  int32_t pos, const nvinfer1::DynamicPluginTensorDesc * ioDesc, int32_t nbInputs,
  int32_t nbOutputs) TRT_NOEXCEPT
{
  if (pos == 3 || pos == 4 || pos == 5) {
    return ioDesc[pos].desc.type == nvinfer1::DataType::kINT32 &&
           ioDesc[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
  }
  return ioDesc[pos].desc.type == nvinfer1::DataType::kFLOAT &&
         ioDesc[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
}

int32_t TRTBatchedNMS::getOutputDataTypes(
  nvinfer1::DataType * outputTypes, int32_t nbOutputs, nvinfer1::DataType const * inputTypes,
  int32_t nbInputs) const TRT_NOEXCEPT
{
  // check the number of I/O
  ASSERT(nbInputs == 2);
  if (mReturnIndex) {
    ASSERT(nbOutputs == 3);
  } else {
    ASSERT(nbOutputs == 2);
  }

  outputTypes[0] = inputTypes[0];
  outputTypes[1] = nvinfer1::DataType::kINT32;
  if (mReturnIndex) {
    outputTypes[2] = nvinfer1::DataType::kINT32;
  }
  return 0;
}

int32_t TRTBatchedNMS::getOutputShapes(
  nvinfer1::DimsExprs const * inputs, int32_t nbInputs, nvinfer1::DimsExprs const * shapeInputs,
  int32_t nbShapeInputs, nvinfer1::DimsExprs * outputs, int32_t nbOutputs,
  nvinfer1::IExprBuilder & exprBuilder) TRT_NOEXCEPT
{
  for (auto i = 0; i < nbOutputs; ++i) {
    outputs[i].d[0] = inputs[0].d[0];
    outputs[i].d[1] = exprBuilder.constant(param.keepTopK);
    switch (i) {
      case 0:
        outputs[i].nbDims = 3;
        outputs[i].d[2] = exprBuilder.constant(5);
        break;
      case 1:
        outputs[i].nbDims = 2;
        break;
      case 2:
        outputs[i].nbDims = 2;
        break;
      default:
        break;
    }
  }
  return 0;
}

int32_t TRTBatchedNMS::enqueue(
  const nvinfer1::PluginTensorDesc * inputDesc, const nvinfer1::PluginTensorDesc * outputDesc,
  const void * const * inputs, void * const * outputs, void * workSpace,
  cudaStream_t stream) TRT_NOEXCEPT
{
  const void * const locData = inputs[0];
  const void * const confData = inputs[1];

  void * nmsedDets = outputs[0];
  void * nmsedLabels = outputs[1];
  void * nmsedIndex = mReturnIndex ? outputs[2] : nullptr;

  size_t batch_size = inputDesc[0].dims.d[0];
  size_t boxes_size = inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
  size_t score_size = inputDesc[1].dims.d[1] * inputDesc[1].dims.d[2];
  size_t num_priors = inputDesc[0].dims.d[1];
  bool shareLocation = (inputDesc[0].dims.d[2] == 1);

  int32_t topk =
    param.topK > 0 && param.topK <= inputDesc[1].dims.d[1] ? param.topK : inputDesc[1].dims.d[1];
  bool rotated = false;
  pluginStatus_t status = nmsInference(
    stream, batch_size, boxes_size, score_size, shareLocation, param.backgroundLabelId, num_priors,
    param.numClasses, topk, param.keepTopK, param.scoreThreshold, param.iouThreshold,
    nvinfer1::DataType::kFLOAT, locData, nvinfer1::DataType::kFLOAT, confData, nmsedDets,
    nmsedLabels, nmsedIndex, workSpace, param.isNormalized, false, mClipBoxes, rotated);
  ASSERT(status == STATUS_SUCCESS);

  return 0;
}

int32_t TRTBatchedNMS::onShapeChange(
  nvinfer1::PluginTensorDesc const * in, int32_t nbInputs, nvinfer1::PluginTensorDesc const * out,
  int32_t nbOutputs) TRT_NOEXCEPT
{
  return 0;
}

nvinfer1::IPluginV3 * TRTBatchedNMS::attachToContext(nvinfer1::IPluginResourceContext *)
  TRT_NOEXCEPT
{
  return clone();
}

nvinfer1::PluginFieldCollection const * TRTBatchedNMS::getFieldsToSerialize() TRT_NOEXCEPT
{
  return &mFC;
}

size_t TRTBatchedNMS::getWorkspaceSize(
  const nvinfer1::DynamicPluginTensorDesc * inputs, int32_t nbInputs,
  const nvinfer1::DynamicPluginTensorDesc * outputs, int32_t nbOutputs) const TRT_NOEXCEPT
{
  size_t batch_size = inputs[0].desc.dims.d[0];
  size_t boxes_size =
    inputs[0].desc.dims.d[1] * inputs[0].desc.dims.d[2] * inputs[0].desc.dims.d[3];
  size_t score_size = inputs[1].desc.dims.d[1] * inputs[1].desc.dims.d[2];
  size_t num_priors = inputs[0].desc.dims.d[1];
  bool shareLocation = (inputs[0].desc.dims.d[2] == 1);
  int32_t topk = param.topK > 0 && param.topK <= inputs[1].desc.dims.d[1]
                   ? param.topK
                   : inputs[1].desc.dims.d[1];
  return detectionInferenceWorkspaceSize(
    shareLocation, batch_size, boxes_size, score_size, param.numClasses, num_priors, topk,
    nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT);
}

// === PluginCreator ===
TRTBatchedNMSCreator::TRTBatchedNMSCreator()
{
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("background_label_id", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("num_classes", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("topk", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("keep_topk", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("score_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("iou_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("is_normalized", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("clip_boxes", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("return_index", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char * TRTBatchedNMSCreator::getPluginNamespace() const TRT_NOEXCEPT
{
  return NMS_PLUGIN_NAMESPACE;
}

const char * TRTBatchedNMSCreator::getPluginName() const TRT_NOEXCEPT
{
  return NMS_PLUGIN_NAME;
}

const char * TRTBatchedNMSCreator::getPluginVersion() const TRT_NOEXCEPT
{
  return NMS_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const * TRTBatchedNMSCreator::getFieldNames() TRT_NOEXCEPT
{
  return &mFC;
}

nvinfer1::IPluginV3 * TRTBatchedNMSCreator::createPlugin(
  const char * name, const nvinfer1::PluginFieldCollection * fc,
  nvinfer1::TensorRTPhase) TRT_NOEXCEPT
{
  const nvinfer1::PluginField * fields = fc->fields;
  bool clipBoxes = true;
  bool returnIndex = false;
  NMSParameters params{};

  for (int i = 0; i < fc->nbFields; ++i) {
    const char * attrName = fields[i].name;
    if (!strcmp(attrName, "background_label_id")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      params.backgroundLabelId = *(static_cast<const int *>(fields[i].data));
    } else if (!strcmp(attrName, "num_classes")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      params.numClasses = *(static_cast<const int *>(fields[i].data));
    } else if (!strcmp(attrName, "topk")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      params.topK = *(static_cast<const int *>(fields[i].data));
    } else if (!strcmp(attrName, "keep_topk")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      params.keepTopK = *(static_cast<const int *>(fields[i].data));
    } else if (!strcmp(attrName, "score_threshold")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      params.scoreThreshold = *(static_cast<const float *>(fields[i].data));
    } else if (!strcmp(attrName, "iou_threshold")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      params.iouThreshold = *(static_cast<const float *>(fields[i].data));
    } else if (!strcmp(attrName, "is_normalized")) {
      params.isNormalized = *(static_cast<const bool *>(fields[i].data));
    } else if (!strcmp(attrName, "clip_boxes")) {
      clipBoxes = *(static_cast<const bool *>(fields[i].data));
    } else if (!strcmp(attrName, "return_index")) {
      returnIndex = *(static_cast<const bool *>(fields[i].data));
    }
  }
  return new (std::nothrow) TRTBatchedNMS(std::string(name), params, clipBoxes, returnIndex);
}

REGISTER_TENSORRT_PLUGIN(TRTBatchedNMSCreator);
}  // namespace mmros::plugin
