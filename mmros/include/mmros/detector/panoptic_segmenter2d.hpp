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
#include "mmros/tensorrt/cuda_unique_ptr.hpp"
#include "mmros/tensorrt/tensorrt_common.hpp"
#include "mmros/tensorrt/utility.hpp"

#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace mmros
{
struct PanopticSegmenter2dConfig
{
  std::vector<double> mean;
  std::vector<double> std;
  double score_threshold;
};

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
   * @param trt_config TensorRT common config.
   * @param detector_config Detector config.
   */
  explicit PanopticSegmenter2D(
    const TrtCommonConfig & trt_config, const PanopticSegmenter2dConfig & detector_config);

  /**
   * @brief Execute inference using input images. Returns `std::nullopt` if the inference fails.
   *
   * @param images Vector of multiple batch images.
   * @return Result<outputs_type>
   */
  Result<outputs_type> doInference(const std::vector<cv::Mat> & images) noexcept;

private:
  void initCudaPtr(size_t batch_size) noexcept;

  cudaError_t preprocess(const std::vector<cv::Mat> & images) noexcept;

  Result<outputs_type> postprocess(const std::vector<cv::Mat> & images) noexcept;

  std::unique_ptr<TrtCommon> trt_common_;                       //!< TrtCommon pointer.
  std::unique_ptr<PanopticSegmenter2dConfig> detector_config_;  //!< Detector config.
  cudaStream_t stream_;                                         //!< CUDA stream.

  std::vector<float> scales_;             //!< Image scales for each batch.
  cuda::CudaUniquePtr<float[]> input_d_;  //!< Input image pointer on the device. [B, 3, H, W].

  cuda::CudaUniquePtr<float[]> out_boxes_d_;  //!< Output boxes pointer on the device [B, N, 5].
  cuda::CudaUniquePtr<int[]> out_labels_d_;   //!< Output labels pointer on the device. [B, N].
  cuda::CudaUniquePtr<float[]> out_masks_d_;  //!< Output masks pointer on the device. [B, N, M, M].
  cuda::CudaUniquePtr<float[]>
    out_segments_d_;  //!< Output segments pointer on the device. [B, C, H, W].
};
}  // namespace mmros
#endif  // MMROS__DETECTOR__PANOPTIC_SEGMENTER2D_HPP_
