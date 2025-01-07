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

#ifndef MMROS__DETECTOR__PANOPTIC_SEGMENTER2D_HPP_
#define MMROS__DETECTOR__PANOPTIC_SEGMENTER2D_HPP_

#include "mmros/archetype/box.hpp"
#include "mmros/archetype/result.hpp"
#include "mmros/tensorrt/tensorrt_common.hpp"
#include "mmros/tensorrt/utility.hpp"

#include <opencv2/core/mat.hpp>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace mmros
{
/**
 * @brief A class represents 2D panoptic segmenter using TensorRT.
 */
class PanopticSegmenter2D
{
public:
  using output_type = std::pair<Boxes2D, cv::Mat>;  //!< Output type of inference results.
  using outputs_type = std::vector<output_type>;    //!< Outputs type of batch results.

  /**
   * @brief
   *
   * @param config TensorRT common config.
   */
  explicit PanopticSegmenter2D(const TrtCommonConfig & config);

  /**
   * @brief Execute inference using input images. Returns `std::nullopt` if the inference fails.
   *
   * @param images Vector of multiple batch images.
   * @return Result<outputs_type>
   */
  Result<outputs_type> doInference(const std::vector<cv::Mat> & images) noexcept;

private:
  std::unique_ptr<TrtCommon> trt_common_;  //!< TrtCommon pointer.
};
}  // namespace mmros
#endif  // MMROS__DETECTOR__PANOPTIC_SEGMENTER2D_HPP_
