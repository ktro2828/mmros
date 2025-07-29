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

#ifndef MMROS__DETECTOR__INSTANCE_SEGMENTER2D_HPP_
#define MMROS__DETECTOR__INSTANCE_SEGMENTER2D_HPP_

#include "mmros/archetype/box.hpp"
#include "mmros/archetype/result.hpp"
#include "mmros/tensorrt/cuda_unique_ptr.hpp"
#include "mmros/tensorrt/tensorrt_common.hpp"
#include "mmros/tensorrt/utility.hpp"

#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
namespace mmros::detector
{
/**
 * @brief Configuration for 2D instance segmenter.
 */
struct InstanceSegmenter2dConfig
{
  std::vector<double> mean;           //!< Image mean.
  std::vector<double> std;            //!< Image std.
  archetype::BoxFormat2D box_format;  //!< Box format.
  double score_threshold;             //!< Score threshold.
};

/**
 * @brief A class represents 2D instance segmenter using TensorRT.
 */
class InstanceSegmenter2D
{
public:
  using output_type =
    std::pair<archetype::Boxes2D, std::vector<cv::Mat>>;  //!< Output type of single batch.
  using outputs_type = std::vector<output_type>;          //!< Output type of multiple batches.

  /**
   * @brief Construct a new InstanceSegmenter2D object.
   *
   * @param trt_config
   * @param detector_config
   */
  InstanceSegmenter2D(
    const tensorrt::TrtCommonConfig & trt_config,
    const InstanceSegmenter2dConfig & detector_config);

  /**
   * @brief Execute inference using input images.
   *
   * @param images Vector of multiple batch images.
   * @return archetype::Result<outputs_type>
   */
  archetype::Result<outputs_type> doInference(const std::vector<cv::Mat> & images) noexcept;

private:
  /**
   * @brief Initialize CUDA pointers.
   *
   * @param batch_size Batch size.
   *
   * @throw Throw `MmRosException` if any CUDA processing failed.
   */
  void initCudaPtr(size_t batch_size);

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
  std::unique_ptr<InstanceSegmenter2dConfig> detector_config_;  //!< Detector config.
  cudaStream_t stream_;                                         //!< CUDA stream.

  std::vector<float> scales_;             //!< Image scales for each batch.
  cuda::CudaUniquePtr<float[]> input_d_;  //!< Input image pointer on the device. [B, 3, H, W].
  int64_t in_height_;                     //!< Model input height.
  int64_t in_width_;                      //!< Model input width.

  cuda::CudaUniquePtr<float[]> out_boxes_d_;  //!< Output boxes pointer on the device. [B, N, 5].
  cuda::CudaUniquePtr<int[]> out_labels_d_;   //!< Output labels pointer on the device. [B, N].
  cuda::CudaUniquePtr<float[]>
    out_segments_d_;  //!< Output masks pointer on the device. [B, N, H, W].
};
}  // namespace mmros::detector
#endif  // MMROS__DETECTOR__INSTANCE_SEGMENTER2D_HPP_
