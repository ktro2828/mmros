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

#include "mmros/archetype/result.hpp"
#include "mmros/preprocess/image.hpp"
#include "mmros/tensorrt/cuda_unique_ptr.hpp"

#include <opencv2/core/mat.hpp>

#include <NvInferRuntime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mmros
{
using outputs_type = SemanticSegmenter2D::outputs_type;

SemanticSegmenter2D::SemanticSegmenter2D(const TrtCommonConfig & config)
{
  trt_common_ = std::make_unique<TrtCommon>(config);

  const auto network_input_dims = trt_common_->getTensorShape(0);
  const auto batch_size = network_input_dims.d[0];
  const auto in_channel = network_input_dims.d[1];
  const auto in_height = network_input_dims.d[2];
  const auto in_width = network_input_dims.d[3];

  std::vector<ProfileDims> profile_dims;
  if (batch_size == -1) {
    // dynamic shape inference
    profile_dims = {
      {0,
       {4, 1, in_channel, in_height, in_width},
       {4, 5, in_channel, in_height, in_width},
       {4, 10, in_channel, in_height, in_width}}};
  } else {
    // static shape inference
    profile_dims = {
      {0,
       {4, batch_size, in_channel, in_height, in_width},
       {4, batch_size, in_channel, in_height, in_width},
       {4, batch_size, in_channel, in_height, in_width}}};
  }

  auto profile_dims_ptr = std::make_unique<std::vector<ProfileDims>>(profile_dims);

  if (!trt_common_->setup(std::move(profile_dims_ptr))) {
    throw std::runtime_error("Failed to setup TensorRT engine.");
  }

  cudaStreamCreate(&stream_);
}

Result<outputs_type> SemanticSegmenter2D::doInference(const std::vector<cv::Mat> & images) noexcept
{
  if (images.empty()) {
    return Err<outputs_type>(InferenceError_t::UNKNOWN, "No image.");
  }

  // 1. Init CUDA pointers
  initCudaPtr(images.size());

  // 2. Execute preprocess
  if (const auto err = preprocess(images); err != ::cudaSuccess) {
    std::ostringstream os;
    os << ::cudaGetErrorName(err) << " (" << err << ")@" << __FILE__ << "#L" << __LINE__ << ": "
       << ::cudaGetErrorString(err);
    return Err<outputs_type>(InferenceError_t::CUDA, os.str());
  }

  // 3. Set tensors
  std::vector<void *> buffers{input_d_.get(), output_d_.get()};
  if (!trt_common_->setTensorsAddresses(buffers)) {
    std::ostringstream os;
    os << "@" << __FILE__ << ", #F:" << __FUNCTION__ << ", #L:" << __LINE__;
    return Err<outputs_type>(InferenceError_t::TENSORRT, os.str());
  }

  // 4. Execute inference
  if (!trt_common_->enqueueV3(stream_)) {
    std::ostringstream os;
    os << "@" << __FILE__ << ", #F:" << __FUNCTION__ << ", #L:" << __LINE__;
    return Err<outputs_type>(InferenceError_t::TENSORRT, os.str());
  }

  // 5. Execute postprocess
  return postprocess(images);
}

void SemanticSegmenter2D::initCudaPtr(size_t batch_size) noexcept
{
  auto get_dim_size = [&](const nvinfer1::Dims & dims) {
    return std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1, std::multiplies<int>());
  };

  auto in_dims = trt_common_->getInputDims(0);
  const auto in_size = get_dim_size(in_dims);
  input_d_ = cuda::make_unique<float[]>(in_size * batch_size);

  auto out_dims = trt_common_->getOutputDims(0);
  const auto out_size = get_dim_size(out_dims);
  output_d_ = cuda::make_unique<int[]>(out_size * batch_size);
}

cudaError_t SemanticSegmenter2D::preprocess(const std::vector<cv::Mat> & images) noexcept
{
  // (B, C, H, W)
  const auto batch_size = images.size();
  auto in_dims = trt_common_->getTensorShape(0);

  cuda::CudaUniquePtrHost<unsigned char[]> img_buf_h;
  cuda::CudaUniquePtr<unsigned char[]> img_buf_d;

  const float input_height = static_cast<float>(in_dims.d[2]);
  const float input_width = static_cast<float>(in_dims.d[3]);
  scales_.clear();
  for (auto b = 0; b < images.size(); ++b) {
    const auto & img = images.at(b);
    if (!img_buf_h) {
      img_buf_h = cuda::make_unique_host<unsigned char[]>(
        img.cols * img.rows * 3 * batch_size, cudaHostAllocWriteCombined);
      img_buf_d = cuda::make_unique<unsigned char[]>(img.cols * img.rows * 3 * batch_size);
    }
    const float scale = std::min(input_width / img.cols, input_height / img.rows);
    scales_.emplace_back(scale);

    int index = b * img.cols * img.rows * 3;
    // Copy into pinned memory
    memcpy(img_buf_h.get() + index, &img.data[0], img.cols * img.rows * 3 * sizeof(unsigned char));
  }

  if (const auto err = ::cudaMemcpyAsync(
        img_buf_d.get(), img_buf_h.get(),
        images[0].cols * images[0].rows * 3 * batch_size * sizeof(unsigned char),
        ::cudaMemcpyHostToDevice, stream_);
      err != ::cudaSuccess) {
    return err;
  }

  preprocess::resize_bilinear_letterbox_nhwc_to_nchw32_batch_gpu(
    input_d_.get(), img_buf_d.get(), input_width, input_height, 3, images[0].cols, images[0].rows,
    3, batch_size, 1.0, stream_);

  return cudaGetLastError();
}

Result<outputs_type> SemanticSegmenter2D::postprocess(const std::vector<cv::Mat> & images) noexcept
{
  const auto batch_size = images.size();

  const auto out_dims = trt_common_->getOutputDims(0);
  const auto num_class = static_cast<size_t>(out_dims.d[1]);
  const auto output_height = static_cast<size_t>(out_dims.d[2]);
  const auto output_width = static_cast<size_t>(out_dims.d[3]);

  auto out_mask = std::make_unique<int[]>(batch_size * num_class * output_width * output_height);

  if (const auto err = ::cudaMemcpyAsync(
        out_mask.get(), output_d_.get(),
        sizeof(int) * batch_size * num_class * output_width * output_height,
        ::cudaMemcpyDeviceToHost, stream_);
      err != ::cudaSuccess) {
    std::ostringstream os;
    os << ::cudaGetErrorName(err) << " (" << err << ")@" << __FILE__ << "#L" << __LINE__ << ": "
       << ::cudaGetErrorString(err);
    return Err<outputs_type>(InferenceError_t::CUDA, os.str());
  }

  outputs_type output;
  output.reserve(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    output_type mask = cv::Mat::zeros(output_height, output_width, CV_8UC1);
    std::memcpy(
      mask.data, out_mask.get() + i * output_height * output_width,
      sizeof(unsigned char) * 1 * output_height * output_width);
    output.emplace_back(mask);
  }

  return Ok(output);
}
}  // namespace mmros
