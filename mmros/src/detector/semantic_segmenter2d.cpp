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

#include <algorithm>
#include <memory>
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
}

Result<outputs_type> SemanticSegmenter2D::doInference(const std::vector<cv::Mat> & images) noexcept
{
  // TODO(ktro2828): Implementation
  return Err<outputs_type>(InferenceError_t::UNKNOWN);
}
}  // namespace mmros
