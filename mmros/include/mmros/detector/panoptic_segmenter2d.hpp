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

namespace mmros::detector
{
/**
 * @brief Configuration for 2D panoptic segmenter.
 */
struct PanopticSegmenter2dConfig
{
  PanopticSegmenter2dConfig(
    const std::vector<double> & _mean, const std::vector<double> & _std,
    archetype::BoxFormat2D _box_format, double _score_threshold)
  : box_format(_box_format), score_threshold(_score_threshold)
  {
    std::vector<float> mean_h(_mean.begin(), _mean.end());
    std::vector<float> std_h(_std.begin(), _std.end());
    mean = cuda::make_unique<float[]>(mean_h.size());
    std = cuda::make_unique<float[]>(std_h.size());

    CHECK_CUDA_ERROR(
      ::cudaMemcpy(
        mean.get(), mean_h.data(), mean_h.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
      ::cudaMemcpy(std.get(), std_h.data(), std_h.size() * sizeof(float), cudaMemcpyHostToDevice));
  }

  cuda::CudaUniquePtr<float[]> mean;  //!< Image mean on device.
  cuda::CudaUniquePtr<float[]> std;   //!< Image std on device.
  archetype::BoxFormat2D box_format;  //!< Box format.
  double score_threshold;             //!< Score threshold.
};

/**
 * @brief A class represents 2D panoptic segmenter using TensorRT.
 */
class PanopticSegmenter2D
{
public:
  using output_type =
    std::pair<archetype::Boxes2D, cv::Mat>;       //!< Output type of inference results.
  using outputs_type = std::vector<output_type>;  //!< Outputs type of batch results.

  /**
   * @brief
   *
   * @param trt_config TensorRT common config.
   * @param detector_config Detector config.
   */
  explicit PanopticSegmenter2D(
    tensorrt::TrtCommonConfig && trt_config, PanopticSegmenter2dConfig && detector_config);

  /**
   * @brief Execute inference using input images. Returns `std::nullopt` if the inference fails.
   *
   * @param images Vector of multiple batch images.
   * @return archetype::Result<outputs_type>
   */
  archetype::Result<outputs_type> do_inference(const std::vector<cv::Mat> & images) noexcept;

private:
  /**
   * @brief Initialize CUDA pointers.
   *
   * @param batch_size Batch size.
   *
   * @throw Throw `MmRosException` if any CUDA processing failed.
   */
  void init_cuda_ptr(size_t batch_size);

  /**
   * @brief Execute preprocessing.
   *
   * @param images Vector of images.
   *
   * @throw Throw `MmRosException` if any CUDA processing failed.
   */
  void preprocess(const std::vector<cv::Mat> & images);

  /**
   * @brief Execute postprocessing.
   *
   * @param images Vector of images.
   */
  archetype::Result<outputs_type> postprocess(const std::vector<cv::Mat> & images) noexcept;

  std::unique_ptr<tensorrt::TrtCommon> trt_common_;             //!< TrtCommon pointer.
  std::unique_ptr<PanopticSegmenter2dConfig> detector_config_;  //!< Detector config.
  cudaStream_t stream_;                                         //!< CUDA stream.

  std::vector<float> scales_;             //!< Image scales for each batch.
  cuda::CudaUniquePtr<float[]> input_d_;  //!< Input image pointer on the device. [B, 3, H, W].
  int64_t in_height_;                     //!< Model input height.
  int64_t in_width_;                      //!< Model input width.

  cuda::CudaUniquePtr<float[]> out_boxes_d_;  //!< Output boxes pointer on the device [B, N, D].
  cuda::CudaUniquePtr<int[]>
    out_labels_d_;  //!< Output labels pointer on the device. [B, N] or [B, N, C].
  cuda::CudaUniquePtr<float[]> out_masks_d_;  //!< Output masks pointer on the device. [B, N, M, M].
  cuda::CudaUniquePtr<float[]>
    out_segments_d_;  //!< Output segments pointer on the device. [B, C, H, W].
};
}  // namespace mmros::detector
#endif  // MMROS__DETECTOR__PANOPTIC_SEGMENTER2D_HPP_
