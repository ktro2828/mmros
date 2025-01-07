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

#include "mmros/detector/detector2d.hpp"

#include "mmros/archetype/result.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <vector>

namespace mmros
{
Detector2D::Detector2D(const TrtCommonConfig & config)
{
  trt_common_ = std::make_unique<TrtCommon>(config);

  const auto network_input_dims = trt_common_->getTensorShape(0);
  const auto batch_size = network_input_dims.d[0];
  const auto in_channel = network_input_dims.d[1];
  const auto in_height = network_input_dims.d[2];
  const auto in_width = network_input_dims.d[3];

  // TODO(ktro2828): Check batch size for dynamic shape inference
  auto profile_dims = std::vector<ProfileDims>(
    {{0,
      {4, batch_size, in_channel, in_height, in_width},
      {4, batch_size, in_channel, in_height, in_width},
      {4, batch_size, in_channel, in_height, in_width}}});

  auto profile_dims_ptr = std::make_unique<std::vector<ProfileDims>>(profile_dims);

  if (!trt_common_->setup(std::move(profile_dims_ptr))) {
    throw std::runtime_error("Failed to setup TensorRT engine.");
  }

  cudaStreamCreate(&stream_);
}

Result<Detector2D::outputs_type> Detector2D::doInference(
  const std::vector<cv::Mat> & images) noexcept
{
  std::vector<void *> buffers{};
  if (!trt_common_->setTensorsAddresses(buffers)) {
    std::ostringstream os;
    os << "@" << __FILE__ << ", #F:" << __FUNCTION__ << ", #L:" << __LINE__;
    return Err<outputs_type>(InferenceError_t::TENSORRT, os.str());
  }

  trt_common_->enqueueV3(stream_);

  const auto batch_size = images.size();
  const auto out_dims = trt_common_->getOutputDims(0);
  const auto num_detection = static_cast<size_t>(out_dims.d[1]);

  auto out_boxes = std::make_unique<float[]>(batch_size * 5 * num_detection);
  auto out_labels = std::make_unique<int32_t[]>(batch_size * num_detection);

  outputs_type output;
  output.reserve(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    output_type boxes;
    boxes.reserve(num_detection);
    for (size_t j = 0; j < num_detection; ++j) {
      const auto xmin = out_boxes[i * num_detection * 5 + j * 5];
      const auto ymin = out_boxes[i * num_detection * 5 + j * 5 + 1];
      const auto xmax = out_boxes[i * num_detection * 5 + j * 5 + 2];
      const auto ymax = out_boxes[i * num_detection * 5 + j * 5 + 3];
      const auto score = out_boxes[i * num_detection * 5 + j * 5 + 4];
      const auto label = out_labels[i * num_detection + j];
      boxes.emplace_back(xmin, ymin, xmax, ymax, score, label);
    }
    output.emplace_back(boxes);
  }

  return Ok(output);
}
}  // namespace mmros
