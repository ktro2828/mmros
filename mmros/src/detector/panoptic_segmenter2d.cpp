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

#include "mmros/detector/panoptic_segmenter2d.hpp"

#include "mmros/archetype/box.hpp"
#include "mmros/archetype/result.hpp"
#include "mmros/preprocess/image.hpp"
#include "mmros/tensorrt/cuda_unique_ptr.hpp"

#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

namespace mmros
{
using outputs_type = PanopticSegmenter2D::outputs_type;

PanopticSegmenter2D::PanopticSegmenter2D(
  const TrtCommonConfig & trt_config, const PanopticSegmenter2dConfig & detector_config)
{
  trt_common_ = std::make_unique<TrtCommon>(trt_config);
  detector_config_ = std::make_unique<PanopticSegmenter2dConfig>(detector_config);

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
       {4, 1, in_channel, 352, 512},
       {4, 5, in_channel, 800, 1344},
       {4, 10, in_channel, 1344, 1344}}};
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

Result<outputs_type> PanopticSegmenter2D::doInference(const std::vector<cv::Mat> & images) noexcept
{
  // TODO(ktro2828): Implementation
  return Err<outputs_type>(InferenceError_t::UNKNOWN);
}

void PanopticSegmenter2D::initCudaPtr(size_t batch_size) noexcept
{
  auto get_dim_size = [&](const nvinfer1::Dims & dims) {
    return std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1, std::multiplies<int>());
  };

  auto in_dims = trt_common_->getInputDims(0);
  const auto in_size = get_dim_size(in_dims);
  input_d_ = cuda::make_unique<float[]>(in_size * batch_size);

  auto out_dims0 = trt_common_->getOutputDims(0);
  const auto out_size0 = get_dim_size(out_dims0);
  out_boxes_d_ = cuda::make_unique<float[]>(out_size0 * batch_size);

  auto out_dims1 = trt_common_->getOutputDims(1);
  const auto out_size1 = get_dim_size(out_dims1);
  out_labels_d_ = cuda::make_unique<int[]>(out_size1 * batch_size);

  auto out_dims2 = trt_common_->getOutputDims(2);
  const auto out_size2 = get_dim_size(out_dims2);
  out_masks_d_ = cuda::make_unique<float[]>(out_size2 * batch_size);

  auto out_dims3 = trt_common_->getOutputDims(3);
  const auto out_size3 = get_dim_size(out_dims3);
  out_segments_d_ = cuda::make_unique<float[]>(out_size3 * batch_size);
}

cudaError_t PanopticSegmenter2D::preprocess(const std::vector<cv::Mat> & images) noexcept
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

Result<outputs_type> PanopticSegmenter2D::postprocess(const std::vector<cv::Mat> & images) noexcept
{
  const auto batch_size = images.size();

  const auto out_boxes_dims = trt_common_->getOutputDims(0);
  const auto num_detection = static_cast<size_t>(out_boxes_dims.d[1]);

  auto out_boxes = std::make_unique<float[]>(batch_size * 5 * num_detection);
  auto out_labels = std::make_unique<int[]>(batch_size * num_detection);

  // copy boxes
  if (const auto err = ::cudaMemcpyAsync(
        out_boxes.get(), out_boxes_d_.get(), sizeof(float) * batch_size * 5 * num_detection,
        ::cudaMemcpyDeviceToHost, stream_);
      err != ::cudaSuccess) {
    std::ostringstream os;
    os << ::cudaGetErrorName(err) << " (" << err << ")@" << __FILE__ << "#L" << __LINE__ << ": "
       << ::cudaGetErrorString(err);
    return Err<outputs_type>(InferenceError_t::CUDA, os.str());
  }

  // copy labels
  if (const auto err = ::cudaMemcpyAsync(
        out_labels.get(), out_labels_d_.get(), sizeof(int) * batch_size * num_detection,
        ::cudaMemcpyDeviceToHost, stream_);
      err != ::cudaSuccess) {
    std::ostringstream os;
    os << ::cudaGetErrorName(err) << " (" << err << ")@" << __FILE__ << "#L" << __LINE__ << ": "
       << ::cudaGetErrorString(err);
    return Err<outputs_type>(InferenceError_t::CUDA, os.str());
  }

  const auto out_segments_dims = trt_common_->getOutputDims(3);
  const auto num_class = static_cast<size_t>(out_segments_dims.d[1]);
  const auto output_height = static_cast<size_t>(out_segments_dims.d[2]);
  const auto output_width = static_cast<size_t>(out_segments_dims.d[3]);

  auto out_segments =
    std::make_unique<float[]>(batch_size * num_class * output_height * output_width);

  // copy segments
  if (const auto err = ::cudaMemcpyAsync(
        out_segments.get(), out_segments_d_.get(),
        sizeof(float) * batch_size * num_class * output_height * output_width,
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
    Boxes2D boxes;
    boxes.reserve(num_detection);
    const auto & scale = scales_.at(i);
    for (size_t j = 0; j < num_detection; ++j) {
      const auto score = out_boxes[i * num_detection * 5 + j * 5 + 4];
      if (score < detector_config_->score_threshold) {
        continue;
      }
      const auto xmin = out_boxes[i * num_detection * 5 + j * 5] / scale;
      const auto ymin = out_boxes[i * num_detection * 5 + j * 5 + 1] / scale;
      const auto xmax = out_boxes[i * num_detection * 5 + j * 5 + 2] / scale;
      const auto ymax = out_boxes[i * num_detection * 5 + j * 5 + 3] / scale;
      const auto label = out_labels[i * num_detection + j];
      boxes.emplace_back(xmin, ymin, xmax, ymax, score, label);
    }
    cv::Mat mask = cv::Mat::zeros(output_height, output_width, CV_8UC1);
    // TODO(ktro2828): postprocess of output segments mask (B, N, H, W)
    output.emplace_back(boxes, mask);
  }

  return Ok(output);
}
}  // namespace mmros
