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
#include "mmros/preprocess/image.hpp"
#include "mmros/tensorrt/cuda_check_error.hpp"
#include "mmros/tensorrt/cuda_unique_ptr.hpp"

#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

namespace mmros::detector
{
using outputs_type = PanopticSegmenter2D::outputs_type;

PanopticSegmenter2D::PanopticSegmenter2D(
  const tensorrt::TrtCommonConfig & trt_config, const PanopticSegmenter2dConfig & detector_config)
{
  trt_common_ = std::make_unique<tensorrt::TrtCommon>(trt_config);
  detector_config_ = std::make_unique<PanopticSegmenter2dConfig>(detector_config);

  const auto network_input_dims = trt_common_->getTensorShape(0);
  const auto batch_size = network_input_dims.d[0];
  const auto in_channel = network_input_dims.d[1];
  const auto in_height = network_input_dims.d[2];
  const auto in_width = network_input_dims.d[3];

  std::vector<tensorrt::ProfileDims> profile_dims;
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

void PanopticSegmenter2D::preprocess(const std::vector<cv::Mat> & images)
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

  CHECK_CUDA_ERROR(::cudaMemcpyAsync(
    img_buf_d.get(), img_buf_h.get(),
    images[0].cols * images[0].rows * 3 * batch_size * sizeof(unsigned char),
    ::cudaMemcpyHostToDevice, stream_));

  // TODO(ktro2828): Refactoring not to load mean/std array every loop
  std::vector<float> mean_h(detector_config_->mean.begin(), detector_config_->mean.end());
  std::vector<float> std_h(detector_config_->std.begin(), detector_config_->std.end());
  auto mean_d = cuda::make_unique<float[]>(mean_h.size());
  auto std_d = cuda::make_unique<float[]>(std_h.size());

  CHECK_CUDA_ERROR(::cudaMemcpyAsync(
    mean_d.get(), mean_h.data(), mean_h.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(::cudaMemcpyAsync(
    std_d.get(), std_h.data(), std_h.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));

  preprocess::resize_bilinear_letterbox_nhwc_to_nchw32_batch_gpu(
    input_d_.get(), img_buf_d.get(), input_width, input_height, 3, images[0].cols, images[0].rows,
    3, batch_size, mean_d.get(), std_d.get(), stream_);

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

  preprocess::argmax_gpu(
    argmax_d.get(), out_segments_d_.get(), output_width, output_height, output_width, output_height,
    num_segment, batch_size, stream_);

  std::vector<float> out_boxes(batch_size * 5 * num_detection);
  std::vector<int> out_labels(batch_size * num_detection);
  std::vector<unsigned char> out_segments(batch_size * num_segment * output_height * output_width);
  try {
    CHECK_CUDA_ERROR(::cudaMemcpyAsync(
      out_boxes.data(), out_boxes_d_.get(), sizeof(float) * batch_size * 5 * num_detection,
      ::cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA_ERROR(::cudaMemcpyAsync(
      out_labels.data(), out_labels_d_.get(), sizeof(int) * batch_size * num_detection,
      ::cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA_ERROR(::cudaMemcpyAsync(
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
      const auto score = out_boxes[i * num_detection * 5 + j * 5 + 4];
      if (score < detector_config_->score_threshold) {
        continue;
      }

      float xmin, ymin, xmax, ymax;
      if (detector_config_->box_format == archetype::BoxFormat2D::XYXY) {
        xmin = out_boxes[i * num_detection * 5 + j * 5] / scale;
        ymin = out_boxes[i * num_detection * 5 + j * 5 + 1] / scale;
        xmax = out_boxes[i * num_detection * 5 + j * 5 + 2] / scale;
        ymax = out_boxes[i * num_detection * 5 + j * 5 + 3] / scale;
      } else {
        float cx = out_boxes[i * num_detection * 5 + j * 5] / scale;
        float cy = out_boxes[i * num_detection * 5 + j * 5 + 1] / scale;
        float width = out_boxes[i * num_detection * 5 + j * 5 + 2] / scale;
        float height = out_boxes[i * num_detection * 5 + j * 5 + 3] / scale;
        xmin = cx - 0.5 * width;
        ymin = cy - 0.5 * height;
        xmax = cx + 0.5 * width;
        ymax = cy + 0.5 * height;
      }

      const auto label = out_labels[i * num_detection + j];
      boxes.emplace_back(xmin, ymin, xmax, ymax, score, label);
    }
    // Output mask
    cv::Mat mask = cv::Mat::zeros(output_height, output_width, CV_8UC1);
    std::memcpy(
      mask.data, out_segments.data() + i * output_height * output_width,
      sizeof(unsigned char) * 1 * output_height * output_width);
    output.emplace_back(boxes, mask);
  }

  return archetype::Ok(output);
}
}  // namespace mmros::detector
