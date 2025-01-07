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

#include <memory>
#include <optional>

namespace mmros
{
PanopticSegmenter2D::PanopticSegmenter2D(const TrtCommonConfig & config)
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
}

Result<PanopticSegmenter2D::outputs_type> PanopticSegmenter2D::doInference(
  const std::vector<cv::Mat> & images) noexcept
{
  // TODO(ktro2828): Implementation
  return {};
}
}  // namespace mmros
