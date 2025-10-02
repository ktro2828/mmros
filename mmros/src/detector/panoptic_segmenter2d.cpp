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
#include "mmros/archetype/exception.hpp"
#include "mmros/archetype/result.hpp"
#include "mmros/detector/utility.hpp"
#include "mmros/process/image.hpp"
#include "mmros/tensorrt/cuda_check_error.hpp"
#include "mmros/tensorrt/cuda_unique_ptr.hpp"

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace mmros::detector
{
using outputs_type = PanopticSegmenter2D::outputs_type;

PanopticSegmenter2D::PanopticSegmenter2D(
  tensorrt::TrtCommonConfig && trt_config, PanopticSegmenter2dConfig && detector_config)
{
  trt_common_ = std::make_unique<tensorrt::TrtCommon>(std::move(trt_config));
  detector_config_ = std::make_unique<PanopticSegmenter2dConfig>(std::move(detector_config));

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

archetype::Result<outputs_type> PanopticSegmenter2D::doInference(
  const std::vector<cv::Mat> & images) noexcept
{
  if (images.empty()) {
    return archetype::Err<outputs_type>(archetype::MmRosError_t::UNKNOWN, "No image.");
  }

  // 1. Init CUDA pointers
  try {
    initCudaPtr(images.size());
  } catch (const archetype::MmRosException & e) {
    return archetype::Err<outputs_type>(archetype::MmRosError_t::CUDA, e.what());
  }

  // 2. Execute preprocess
  try {
    preprocess(images);
  } catch (const archetype::MmRosException & e) {
    return archetype::Err<outputs_type>(archetype::MmRosError_t::CUDA, e.what());
  }

  // 3. Set tensors
  std::vector<void *> buffers{
    input_d_.get(), out_boxes_d_.get(), out_labels_d_.get(), out_masks_d_.get(),
    out_segments_d_.get()};
  if (!trt_common_->setTensorsAddresses(buffers)) {
    std::ostringstream os;
    os << "@" << __FILE__ << ", #F:" << __FUNCTION__ << ", #L:" << __LINE__;
    return archetype::Err<outputs_type>(archetype::MmRosError_t::TENSORRT, os.str());
  }

  // 4. Execute inference
  if (!trt_common_->enqueueV3(stream_)) {
    std::ostringstream os;
    os << "@" << __FILE__ << ", #F:" << __FUNCTION__ << ", #L:" << __LINE__;
    return archetype::Err<outputs_type>(archetype::MmRosError_t::TENSORRT, os.str());
  }

  // 5. Execute postprocess
  return postprocess(images);
}

void PanopticSegmenter2D::initCudaPtr(size_t batch_size)
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

  auto out_dims0 = trt_common_->getOutputDims(0);
  const auto out_size0 = get_dim_size(out_dims0);
  if (!out_boxes_d_) {
    out_boxes_d_ = cuda::make_unique<float[]>(out_size0 * batch_size);
  } else {
    cuda::clear_async(out_boxes_d_.get(), out_size0 * batch_size, stream_);
  }

  auto out_dims1 = trt_common_->getOutputDims(1);
  const auto out_size1 = get_dim_size(out_dims1);
  if (!out_labels_d_) {
    out_labels_d_ = cuda::make_unique<int[]>(out_size1 * batch_size);
  } else {
    cuda::clear_async(out_labels_d_.get(), out_size1 * batch_size, stream_);
  }

  auto out_dims2 = trt_common_->getOutputDims(2);
  const auto out_size2 = get_dim_size(out_dims2);
  if (!out_masks_d_) {
    out_masks_d_ = cuda::make_unique<float[]>(out_size2 * batch_size);
  } else {
    cuda::clear_async(out_masks_d_.get(), out_size2 * batch_size, stream_);
  }

  auto out_dims3 = trt_common_->getOutputDims(3);
  const auto out_size3 = get_dim_size(out_dims3);
  if (!out_segments_d_) {
    out_segments_d_ = cuda::make_unique<float[]>(out_size3 * batch_size);
  } else {
    cuda::clear_async(out_segments_d_.get(), out_size3 * batch_size, stream_);
  }
}

void PanopticSegmenter2D::preprocess(const std::vector<cv::Mat> & images)
{
  // (B, C, H, W)
  const auto batch_size = images.size();
  auto in_dims = trt_common_->getTensorShape(0);

  cuda::CudaUniquePtrHost<unsigned char[]> img_buf_h;
  cuda::CudaUniquePtr<unsigned char[]> img_buf_d;

  scales_.clear();
  for (auto b = 0; b < images.size(); ++b) {
    const auto & img = images.at(b);
    if (!img_buf_h) {
      img_buf_h = cuda::make_unique_host<unsigned char[]>(
        img.cols * img.rows * 3 * batch_size, cudaHostAllocWriteCombined);
      img_buf_d = cuda::make_unique<unsigned char[]>(img.cols * img.rows * 3 * batch_size);
    }
    const float scale =
      std::min(static_cast<float>(in_width_) / img.cols, static_cast<float>(in_height_) / img.rows);
    scales_.emplace_back(scale);

    int index = b * img.cols * img.rows * 3;
    // Copy into pinned memory
    memcpy(img_buf_h.get() + index, &img.data[0], img.cols * img.rows * 3 * sizeof(unsigned char));
  }

  CHECK_CUDA_ERROR(
    ::cudaMemcpyAsync(
      img_buf_d.get(), img_buf_h.get(),
      images[0].cols * images[0].rows * 3 * batch_size * sizeof(unsigned char),
      ::cudaMemcpyHostToDevice, stream_));

  process::resize_bilinear_letterbox_nhwc_to_nchw32_batch_gpu(
    input_d_.get(), img_buf_d.get(), in_width_, in_height_, 3, images[0].cols, images[0].rows, 3,
    batch_size, detector_config_->mean.get(), detector_config_->std.get(), stream_);

  CHECK_CUDA_ERROR(cudaGetLastError());
}

archetype::Result<outputs_type> PanopticSegmenter2D::postprocess(
  const std::vector<cv::Mat> & images) noexcept
{
  const auto batch_size = images.size();

  const auto out_boxes_dims = trt_common_->getOutputDims(0);
  const auto num_detection = static_cast<size_t>(out_boxes_dims.d[1]);

  const auto out_segments_dims = trt_common_->getOutputDims(3);
  const auto num_segment = static_cast<size_t>(out_segments_dims.d[1]);
  const auto output_height = static_cast<size_t>(out_segments_dims.d[2]);
  const auto output_width = static_cast<size_t>(out_segments_dims.d[3]);

  auto argmax_d =
    cuda::make_unique<unsigned char[]>(batch_size * num_segment * output_height * output_width);

  process::argmax_gpu(
    argmax_d.get(), out_segments_d_.get(), output_width, output_height, output_width, output_height,
    num_segment, batch_size, stream_);

  size_t box_dim, class_dim;
  if (archetype::include_score_in_box(detector_config_->box_format)) {
    box_dim = 5;
    class_dim = 1;
  } else {
    box_dim = 4;
    class_dim = static_cast<size_t>(trt_common_->getOutputDims(1).d[2]);
  }

  std::vector<float> out_boxes(batch_size * num_detection * box_dim);
  std::vector<int> out_labels(batch_size * num_detection * class_dim);
  std::vector<unsigned char> out_segments(batch_size * num_segment * output_height * output_width);
  try {
    CHECK_CUDA_ERROR(
      ::cudaMemcpyAsync(
        out_boxes.data(), out_boxes_d_.get(), sizeof(float) * batch_size * num_detection * box_dim,
        ::cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA_ERROR(
      ::cudaMemcpyAsync(
        out_labels.data(), out_labels_d_.get(),
        sizeof(int) * batch_size * num_detection * class_dim, ::cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA_ERROR(
      ::cudaMemcpyAsync(
        out_segments.data(), argmax_d.get(),
        sizeof(unsigned char) * batch_size * 1 * output_height * output_width,
        ::cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  } catch (const archetype::MmRosException & e) {
    return archetype::Err<outputs_type>(archetype::MmRosError_t::CUDA, e.what());
  }

  outputs_type output;
  output.reserve(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    // Output boxes
    archetype::Boxes2D boxes;
    boxes.reserve(num_detection);
    const auto & scale = scales_.at(i);
    for (size_t j = 0; j < num_detection; ++j) {
      const auto box_ptr = out_boxes.data() + (i * num_detection + j) * box_dim;
      const auto label_ptr = out_labels.data() + (i * num_detection + j) * class_dim;

      const auto box = to_box2d(
        box_ptr, label_ptr, box_dim, class_dim, in_height_, in_width_, scale,
        detector_config_->box_format);

      if (box.score < detector_config_->score_threshold) {
        continue;
      }

      boxes.push_back(std::move(box));
    }
    // Output mask
    cv::Mat mask = cv::Mat::zeros(output_height, output_width, CV_8UC1);
    std::memcpy(
      mask.data, out_segments.data() + i * output_height * output_width,
      sizeof(unsigned char) * 1 * output_height * output_width);
    output.emplace_back(std::move(boxes), std::move(mask));
  }

  return archetype::Ok(output);
}
}  // namespace mmros::detector
