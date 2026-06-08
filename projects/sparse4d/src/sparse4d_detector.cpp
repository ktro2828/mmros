// Copyright 2025 Kotaro Uetake.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sparse4d/sparse4d_detector.hpp"

#include "mmros/archetype/exception.hpp"
#include "mmros/tensorrt/cuda_check_error.hpp"

#include <opencv2/imgproc.hpp>

#include <NvInferRuntimeBase.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace sparse4d
{
namespace
{
constexpr size_t kNumClasses = 10;
constexpr size_t kBoxDim = 11;
constexpr size_t kQualityDim = 2;

constexpr size_t kX = 0;
constexpr size_t kY = 1;
constexpr size_t kZ = 2;
constexpr size_t kW = 3;
constexpr size_t kL = 4;
constexpr size_t kH = 5;
constexpr size_t kSinYaw = 6;
constexpr size_t kCosYaw = 7;
constexpr size_t kVx = 8;

constexpr size_t kCenterness = 0;

float sigmoid(float x)
{
  return 1.0F / (1.0F + std::exp(-x));
}

std::string dims_to_string(const nvinfer1::Dims & dims)
{
  std::ostringstream os;
  os << '[';
  for (int index = 0; index < dims.nbDims; ++index) {
    if (index > 0) {
      os << ", ";
    }
    os << dims.d[index];
  }
  os << ']';
  return os.str();
}

std::string data_type_to_string(const nvinfer1::DataType data_type)
{
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      return "FLOAT";
    case nvinfer1::DataType::kHALF:
      return "HALF";
    case nvinfer1::DataType::kINT8:
      return "INT8";
    case nvinfer1::DataType::kINT32:
      return "INT32";
    case nvinfer1::DataType::kBOOL:
      return "BOOL";
    case nvinfer1::DataType::kUINT8:
      return "UINT8";
    case nvinfer1::DataType::kFP8:
      return "FP8";
    case nvinfer1::DataType::kBF16:
      return "BF16";
    case nvinfer1::DataType::kINT64:
      return "INT64";
    case nvinfer1::DataType::kINT4:
      return "INT4";
    case nvinfer1::DataType::kFP4:
      return "FP4";
  }
  return "UNKNOWN";
}

std::string io_mode_to_string(const nvinfer1::TensorIOMode io_mode)
{
  switch (io_mode) {
    case nvinfer1::TensorIOMode::kINPUT:
      return "INPUT";
    case nvinfer1::TensorIOMode::kOUTPUT:
      return "OUTPUT";
    case nvinfer1::TensorIOMode::kNONE:
      return "NONE";
  }
  return "UNKNOWN";
}

size_t count_finite(const std::vector<float> & values)
{
  return static_cast<size_t>(
    std::count_if(values.begin(), values.end(), [](float value) { return std::isfinite(value); }));
}
}  // namespace

Sparse4dDetector::Sparse4dDetector(
  std::string onnx_path, std::string precision, Sparse4dConfig && detector_config)
: onnx_path_(std::move(onnx_path)),
  precision_(std::move(precision)),
  has_logged_runtime_shapes_(false),
  has_logged_invalid_output_stats_(false)
{
  detector_config_ = std::make_unique<Sparse4dConfig>(std::move(detector_config));
  setup_trt();
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
}

mmros::archetype::Result<Sparse4dDetector::output_type> Sparse4dDetector::do_inference(
  const Sparse4dInferenceInput & input) noexcept
{
  if (input.images.size() != detector_config_->num_cameras) {
    std::ostringstream os;
    os << "Sparse4D expects " << detector_config_->num_cameras << " input images, got "
       << input.images.size();
    return mmros::archetype::make_err<output_type>(
      mmros::archetype::MmRosError_t::UNKNOWN, os.str());
  }

  if (input.camera_inputs.size() != detector_config_->num_cameras) {
    std::ostringstream os;
    os << "Sparse4D expects " << detector_config_->num_cameras << " camera inputs, got "
       << input.camera_inputs.size();
    return mmros::archetype::make_err<output_type>(
      mmros::archetype::MmRosError_t::UNKNOWN, os.str());
  }

  try {
    init_cuda_ptr(1);
    preprocess(input);
  } catch (const mmros::archetype::MmRosException & e) {
    return mmros::archetype::make_err<output_type>(mmros::archetype::MmRosError_t::CUDA, e.what());
  }

  std::vector<void *> buffers{image_d_.get(),          projection_mat_d_.get(), image_wh_d_.get(),
                              classification_d_.get(), prediction_d_.get(),     quality_d_.get(),
                              instance_id_d_.get()};
  if (!trt_common_->setTensorsAddresses(buffers)) {
    std::ostringstream os;
    os << "@" << __FILE__ << ", #F:" << __FUNCTION__ << ", #L:" << __LINE__;
    return mmros::archetype::make_err<output_type>(
      mmros::archetype::MmRosError_t::TENSORRT, os.str());
  }

  if (!trt_common_->enqueueV3(stream_)) {
    std::ostringstream os;
    os << "@" << __FILE__ << ", #F:" << __FUNCTION__ << ", #L:" << __LINE__;
    return mmros::archetype::make_err<output_type>(
      mmros::archetype::MmRosError_t::TENSORRT, os.str());
  }

  return postprocess(1);
}

void Sparse4dDetector::setup_trt()
{
  trt_common_ = std::make_unique<mmros::tensorrt::TrtCommon>(
    mmros::tensorrt::TrtCommonConfig(onnx_path_, precision_));

  const auto image_dims = trt_common_->getTensorShape(0);
  in_height_ = image_dims.d[3];
  in_width_ = image_dims.d[4];

  std::vector<mmros::tensorrt::ProfileDims> profile_dims{
    {0,
     {5, {1, 6, 3, in_height_, in_width_}},
     {5, {1, 6, 3, in_height_, in_width_}},
     {5, {1, 6, 3, in_height_, in_width_}}},
    {1, {4, {1, 6, 4, 4}}, {4, {1, 6, 4, 4}}, {4, {1, 6, 4, 4}}},
    {2, {3, {1, 6, 2}}, {3, {1, 6, 2}}, {3, {1, 6, 2}}},
  };

  std::vector<mmros::tensorrt::NetworkIO> network_io{
    {"img", {5, {1, 6, 3, in_height_, in_width_}}},
    {"projection_mat", {4, {1, 6, 4, 4}}},
    {"image_wh", {3, {1, 6, 2}}},
    {"classification", {3, {1, 900, 10}}},
    {"prediction", {3, {1, 900, 11}}},
    {"quality", {3, {1, 900, 2}}},
    {"instance_id", {2, {1, 900}}},
  };

  auto profile_dims_ptr = std::make_unique<std::vector<mmros::tensorrt::ProfileDims>>(profile_dims);
  auto network_io_ptr = std::make_unique<std::vector<mmros::tensorrt::NetworkIO>>(network_io);
  if (!trt_common_->setup(std::move(profile_dims_ptr), std::move(network_io_ptr))) {
    throw mmros::archetype::MmRosException(
      mmros::archetype::MmRosError_t::TENSORRT, "Failed to setup TensorRT engine.");
  }

  log_tensor_shapes();
}

void Sparse4dDetector::init_cuda_ptr(size_t batch_size)
{
  auto get_dim_size = [](const nvinfer1::Dims & dims) {
    return std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1, std::multiplies<int>());
  };

  const auto image_dims = trt_common_->getInputDims(0);
  const auto image_size = get_dim_size(image_dims);
  if (!image_d_) {
    image_d_ = mmros::cuda::make_unique<float[]>(image_size * batch_size);
  } else {
    mmros::cuda::clear_async(image_d_.get(), image_size * batch_size, stream_);
  }

  const auto projection_dims = trt_common_->getInputDims(1);
  const auto projection_size = get_dim_size(projection_dims);
  if (!projection_mat_d_) {
    projection_mat_d_ = mmros::cuda::make_unique<float[]>(projection_size * batch_size);
  } else {
    mmros::cuda::clear_async(projection_mat_d_.get(), projection_size * batch_size, stream_);
  }

  const auto image_wh_dims = trt_common_->getInputDims(2);
  const auto image_wh_size = get_dim_size(image_wh_dims);
  if (!image_wh_d_) {
    image_wh_d_ = mmros::cuda::make_unique<float[]>(image_wh_size * batch_size);
  } else {
    mmros::cuda::clear_async(image_wh_d_.get(), image_wh_size * batch_size, stream_);
  }

  const auto classification_dims = trt_common_->getOutputDims(0);
  const auto classification_size = get_dim_size(classification_dims);
  if (!classification_d_) {
    classification_d_ = mmros::cuda::make_unique<float[]>(classification_size * batch_size);
  } else {
    mmros::cuda::clear_async(classification_d_.get(), classification_size * batch_size, stream_);
  }

  const auto prediction_dims = trt_common_->getOutputDims(1);
  const auto prediction_size = get_dim_size(prediction_dims);
  if (!prediction_d_) {
    prediction_d_ = mmros::cuda::make_unique<float[]>(prediction_size * batch_size);
  } else {
    mmros::cuda::clear_async(prediction_d_.get(), prediction_size * batch_size, stream_);
  }

  const auto quality_dims = trt_common_->getOutputDims(2);
  const auto quality_size = get_dim_size(quality_dims);
  if (!quality_d_) {
    quality_d_ = mmros::cuda::make_unique<float[]>(quality_size * batch_size);
  } else {
    mmros::cuda::clear_async(quality_d_.get(), quality_size * batch_size, stream_);
  }

  const auto instance_id_dims = trt_common_->getOutputDims(3);
  const auto instance_id_size = get_dim_size(instance_id_dims);
  if (!instance_id_d_) {
    instance_id_d_ = mmros::cuda::make_unique<int64_t[]>(instance_id_size * batch_size);
  } else {
    mmros::cuda::clear_async(instance_id_d_.get(), instance_id_size * batch_size, stream_);
  }
}

void Sparse4dDetector::preprocess(const Sparse4dInferenceInput & input)
{
  std::vector<float> image_h(detector_config_->num_cameras * 3 * in_height_ * in_width_, 0.0F);
  std::vector<float> projection_h(detector_config_->num_cameras * 16, 0.0F);
  std::vector<float> image_wh_h(detector_config_->num_cameras * 2, 0.0F);

  const auto plane_size = static_cast<size_t>(in_height_ * in_width_);
  for (size_t camera_id = 0; camera_id < input.images.size(); ++camera_id) {
    const auto & image = input.images.at(camera_id);
    const auto resize = std::max(
      static_cast<float>(in_height_) / static_cast<float>(image.rows),
      static_cast<float>(in_width_) / static_cast<float>(image.cols));
    const int resized_width = static_cast<int>(image.cols * resize);
    const int resized_height = static_cast<int>(image.rows * resize);
    const int crop_top = std::max(0, resized_height - static_cast<int>(in_height_));
    const int crop_left = std::max(0, resized_width - static_cast<int>(in_width_)) / 2;

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(resized_width, resized_height));
    resized_image = resized_image(
      cv::Rect(crop_left, crop_top, static_cast<int>(in_width_), static_cast<int>(in_height_)));

    // Reference preprocessing normalizes in RGB order. The node already converts ROS images to
    // RGB8.
    for (int row = 0; row < resized_image.rows; ++row) {
      for (int column = 0; column < resized_image.cols; ++column) {
        const auto pixel = resized_image.at<cv::Vec3b>(row, column);
        for (size_t channel = 0; channel < 3; ++channel) {
          const auto index =
            camera_id * 3 * plane_size + channel * plane_size + row * resized_image.cols + column;
          image_h.at(index) =
            (static_cast<float>(pixel[channel]) - detector_config_->mean.at(channel)) /
            detector_config_->std.at(channel);
        }
      }
    }

    std::array<float, 16> image_aug_mat = {
      resize,
      0.0F,
      -static_cast<float>(crop_left),
      0.0F,
      0.0F,
      resize,
      -static_cast<float>(crop_top),
      0.0F,
      0.0F,
      0.0F,
      1.0F,
      0.0F,
      0.0F,
      0.0F,
      0.0F,
      1.0F,
    };
    std::array<float, 16> scaled_projection{};
    const auto & projection = input.camera_inputs.at(camera_id).projection_mat;
    for (size_t row = 0; row < 4; ++row) {
      for (size_t column = 0; column < 4; ++column) {
        float value = 0.0F;
        for (size_t inner = 0; inner < 4; ++inner) {
          value += image_aug_mat.at(row * 4 + inner) * projection.at(inner * 4 + column);
        }
        scaled_projection.at(row * 4 + column) = value;
      }
    }

    std::memcpy(projection_h.data() + camera_id * 16, scaled_projection.data(), sizeof(float) * 16);

    const std::array<float, 2> resized_image_wh = {
      static_cast<float>(in_width_),
      static_cast<float>(in_height_),
    };
    std::memcpy(image_wh_h.data() + camera_id * 2, resized_image_wh.data(), sizeof(float) * 2);
  }

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    image_d_.get(), image_h.data(), sizeof(float) * image_h.size(), cudaMemcpyHostToDevice,
    stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    projection_mat_d_.get(), projection_h.data(), sizeof(float) * projection_h.size(),
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    image_wh_d_.get(), image_wh_h.data(), sizeof(float) * image_wh_h.size(), cudaMemcpyHostToDevice,
    stream_));
}

mmros::archetype::Result<Sparse4dDetector::output_type> Sparse4dDetector::postprocess(
  size_t batch_size) noexcept
{
  const auto classification_dims = trt_common_->getOutputDims(0);
  const auto prediction_dims = trt_common_->getOutputDims(1);
  const auto quality_dims = trt_common_->getOutputDims(2);
  const auto instance_id_dims = trt_common_->getOutputDims(3);

  auto get_dim_size = [](const nvinfer1::Dims & dims) {
    return std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1, std::multiplies<int>());
  };

  std::vector<float> classification_h(batch_size * get_dim_size(classification_dims));
  std::vector<float> prediction_h(batch_size * get_dim_size(prediction_dims));
  std::vector<float> quality_h(batch_size * get_dim_size(quality_dims));
  std::vector<int64_t> instance_id_h(batch_size * get_dim_size(instance_id_dims));

  try {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      classification_h.data(), classification_d_.get(), sizeof(float) * classification_h.size(),
      cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      prediction_h.data(), prediction_d_.get(), sizeof(float) * prediction_h.size(),
      cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      quality_h.data(), quality_d_.get(), sizeof(float) * quality_h.size(), cudaMemcpyDeviceToHost,
      stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      instance_id_h.data(), instance_id_d_.get(), sizeof(int64_t) * instance_id_h.size(),
      cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  } catch (const mmros::archetype::MmRosException & e) {
    return mmros::archetype::make_err<output_type>(mmros::archetype::MmRosError_t::CUDA, e.what());
  }

  if (!has_logged_runtime_shapes_) {
    has_logged_runtime_shapes_ = true;
  }

  output_type boxes;
  const size_t num_pred = static_cast<size_t>(classification_dims.d[1]);
  std::vector<std::tuple<float, size_t, size_t, float>> scored;
  scored.reserve(num_pred * kNumClasses);

  for (size_t pred_index = 0; pred_index < num_pred; ++pred_index) {
    const auto centerness = sigmoid(quality_h.at(pred_index * kQualityDim + kCenterness));
    if (!std::isfinite(centerness)) {
      continue;
    }
    for (size_t class_index = 0; class_index < kNumClasses; ++class_index) {
      const auto cls_score = sigmoid(classification_h.at(pred_index * kNumClasses + class_index));
      const auto score = cls_score * centerness;
      if (!std::isfinite(cls_score) || !std::isfinite(score)) {
        continue;
      }
      if (score < detector_config_->score_threshold) {
        continue;
      }
      scored.emplace_back(score, pred_index, class_index, cls_score);
    }
  }

  const auto num_output = std::min(detector_config_->num_output, scored.size());
  std::partial_sort(
    scored.begin(), scored.begin() + num_output, scored.end(),
    [](const auto & lhs, const auto & rhs) { return std::get<0>(lhs) > std::get<0>(rhs); });

  boxes.reserve(num_output);
  for (size_t output_index = 0; output_index < num_output; ++output_index) {
    const auto [score, pred_index, class_index, cls_score] = scored.at(output_index);
    (void)cls_score;
    const float * box_ptr = prediction_h.data() + pred_index * kBoxDim;

    const auto yaw = std::atan2(box_ptr[kSinYaw], box_ptr[kCosYaw]);
    const auto length = std::exp(box_ptr[kW]);
    const auto width = std::exp(box_ptr[kL]);
    const auto height = std::exp(box_ptr[kH]);

    if (
      !std::isfinite(box_ptr[kX]) || !std::isfinite(box_ptr[kY]) || !std::isfinite(box_ptr[kZ]) ||
      !std::isfinite(length) || !std::isfinite(width) || !std::isfinite(height) ||
      !std::isfinite(yaw) || !std::isfinite(score)) {
      continue;
    }

    boxes.emplace_back(
      box_ptr[kX], box_ptr[kY], box_ptr[kZ], length, width, height, yaw, score,
      static_cast<int>(class_index));
  }

  if (boxes.empty() && !has_logged_invalid_output_stats_) {
    has_logged_invalid_output_stats_ = true;
    std::cerr << "Sparse4D finite stats: classification=" << count_finite(classification_h) << '/'
              << classification_h.size() << ", prediction=" << count_finite(prediction_h) << '/'
              << prediction_h.size() << ", quality=" << count_finite(quality_h) << '/'
              << quality_h.size() << std::endl;
  }

  return mmros::archetype::make_ok(boxes);
}

void Sparse4dDetector::log_tensor_shapes() const
{
  std::ostringstream os;
  os << "Sparse4D TensorRT IO metadata:";
  for (int32_t index = 0; index < trt_common_->getNbIOTensors(); ++index) {
    os << ' ' << trt_common_->getIOTensorName(index)
       << "{mode=" << io_mode_to_string(trt_common_->getTensorIOMode(index))
       << ", dtype=" << data_type_to_string(trt_common_->getTensorDataType(index))
       << ", shape=" << dims_to_string(trt_common_->getTensorShape(index)) << '}';
    if (index + 1 < trt_common_->getNbIOTensors()) {
      os << ',';
    }
  }
  std::cerr << os.str() << std::endl;
}
}  // namespace sparse4d
