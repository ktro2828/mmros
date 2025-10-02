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

#include "mmros/detector/semantic_segmenter2d.hpp"

#include "mmros/archetype/exception.hpp"
#include "mmros/archetype/result.hpp"
#include "mmros/process/image.hpp"
#include "mmros/tensorrt/cuda_check_error.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include <NvInferRuntime.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace mmros::detector
{
using outputs_type = SemanticSegmenter2D::outputs_type;

SemanticSegmenter2D::SemanticSegmenter2D(
  tensorrt::TrtCommonConfig && trt_config, SemanticSegmenter2dConfig && detector_config)
{
  trt_common_ = std::make_unique<tensorrt::TrtCommon>(std::move(trt_config));
  detector_config_ = std::make_unique<SemanticSegmenter2dConfig>(std::move(detector_config));

  const auto network_input_dims = trt_common_->getTensorShape(0);
  const auto batch_size = network_input_dims.d[0];
  const auto in_channel = network_input_dims.d[1];
  in_height_ = network_input_dims.d[2];
  in_width_ = network_input_dims.d[3];

  std::vector<tensorrt::ProfileDims> profile_dims;
  if (batch_size == -1) {
    // dynamic shape inference
    profile_dims = {
      {0,
       {4, 1, in_channel, in_height_, in_width_},
       {4, 5, in_channel, in_height_, in_width_},
       {4, 10, in_channel, in_height_, in_width_}}};
  } else {
    // static shape inference
    profile_dims = {
      {0,
       {4, batch_size, in_channel, in_height_, in_width_},
       {4, batch_size, in_channel, in_height_, in_width_},
       {4, batch_size, in_channel, in_height_, in_width_}}};
  }

  auto profile_dims_ptr = std::make_unique<std::vector<tensorrt::ProfileDims>>(profile_dims);

  if (!trt_common_->setup(std::move(profile_dims_ptr))) {
    throw archetype::MmRosException(
      archetype::MmRosError_t::TENSORRT, "Failed to setup TensorRT engine.");
  }

  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
}

archetype::Result<outputs_type> SemanticSegmenter2D::do_inference(
  const std::vector<cv::Mat> & images) noexcept
{
  if (images.empty()) {
    return archetype::make_err<outputs_type>(archetype::MmRosError_t::UNKNOWN, "No image.");
  }

  // 1. Init CUDA pointers
  try {
    init_cuda_ptr(images.size());
  } catch (const archetype::MmRosException & e) {
    return archetype::make_err<outputs_type>(archetype::MmRosError_t::CUDA, e.what());
  }

  // 2. Execute preprocess
  try {
    preprocess(images);
  } catch (const archetype::MmRosException & e) {
    return archetype::make_err<outputs_type>(archetype::MmRosError_t::CUDA, e.what());
  }

  // 3. Set tensors
  std::vector<void *> buffers{input_d_.get(), output_d_.get()};
  if (!trt_common_->setTensorsAddresses(buffers)) {
    std::ostringstream os;
    os << "@" << __FILE__ << ", #F:" << __FUNCTION__ << ", #L:" << __LINE__;
    return archetype::make_err<outputs_type>(archetype::MmRosError_t::TENSORRT, os.str());
  }

  // 4. Execute inference
  if (!trt_common_->enqueueV3(stream_)) {
    std::ostringstream os;
    os << "@" << __FILE__ << ", #F:" << __FUNCTION__ << ", #L:" << __LINE__;
    return archetype::make_err<outputs_type>(archetype::MmRosError_t::TENSORRT, os.str());
  }

  // 5. Execute postprocess
  return postprocess(images);
}

void SemanticSegmenter2D::init_cuda_ptr(size_t batch_size)
{
  auto get_dim_size = [&](const nvinfer1::Dims & dims) {
    return std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1, std::multiplies<int>());
  };

  auto in_dims = trt_common_->getInputDims(0);
  const auto in_size = get_dim_size(in_dims);
  if (!input_d_) {
    input_d_ = cuda::make_unique<float[]>(in_size * batch_size);
  } else {
    cuda::clear_async(input_d_.get(), in_size * batch_size, stream_);
  }

  auto out_dims = trt_common_->getOutputDims(0);
  const auto out_size = get_dim_size(out_dims);
  if (!output_d_) {
    output_d_ = cuda::make_unique<int64_t[]>(out_size * batch_size);
  } else {
    cuda::clear_async(output_d_.get(), out_size * batch_size, stream_);
  }
}

void SemanticSegmenter2D::preprocess(const std::vector<cv::Mat> & images)
{
  process::preprocess_image(
    input_d_.get(), scales_, images, in_width_, in_height_, detector_config_->mean.get(),
    detector_config_->std.get(), stream_);
}

archetype::Result<outputs_type> SemanticSegmenter2D::postprocess(
  const std::vector<cv::Mat> & images) noexcept
{
  const auto batch_size = images.size();

  const auto out_dims = trt_common_->getOutputDims(0);
  const auto output_height = static_cast<size_t>(out_dims.d[2]);
  const auto output_width = static_cast<size_t>(out_dims.d[3]);

  std::vector<int64_t> output_h(batch_size * 1 * output_width * output_height);
  try {
    CHECK_CUDA_ERROR(
      ::cudaMemcpy(
        output_h.data(), output_d_.get(),
        sizeof(int64_t) * batch_size * 1 * output_width * output_height, ::cudaMemcpyDeviceToHost));
  } catch (const archetype::MmRosException & e) {
    return archetype::make_err<outputs_type>(archetype::MmRosError_t::CUDA, e.what());
  }

  outputs_type output;
  output.reserve(batch_size);

  constexpr int64_t max_class_id = 255;
  // Convert int64_t class IDs to unsigned char and resize to original image size
  for (size_t i = 0; i < batch_size; ++i) {
    const auto & orig_image = images[i];
    const float scale = scales_[i];
    // Create model output mask
    cv::Mat model_mask(output_height, output_width, CV_8UC1, cv::Scalar(0));
    for (size_t j = 0; j < output_height * output_width; ++j) {
      int64_t class_id = output_h[i * output_height * output_width + j];
      model_mask.data[j] =
        static_cast<unsigned char>(std::min(std::max(class_id, 0L), max_class_id));
    }

    // Calculate the actual resized dimensions based on letterbox scale
    const int scaled_w = static_cast<int>(orig_image.cols * scale);
    const int scaled_h = static_cast<int>(orig_image.rows * scale);

    // Extract the valid region (letterbox uses top-left alignment)
    cv::Rect valid_region(
      0, 0, std::min(scaled_w, static_cast<int>(output_width)),
      std::min(scaled_h, static_cast<int>(output_height)));

    cv::Mat valid_mask = model_mask(valid_region);
    // Resize back to original image size using nearest neighbor to preserve class labels
    output_type final_mask;
    cv::resize(
      valid_mask, final_mask, cv::Size(orig_image.cols, orig_image.rows), 0, 0, cv::INTER_NEAREST);

    output.push_back(std::move(final_mask));
  }

  return archetype::make_ok(output);
}
}  // namespace mmros::detector
