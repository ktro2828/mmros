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

#ifndef SPARSE4D__SPARSE4D_DETECTOR_HPP_
#define SPARSE4D__SPARSE4D_DETECTOR_HPP_

#include "mmros/archetype/box.hpp"
#include "mmros/archetype/result.hpp"
#include "mmros/tensorrt/cuda_unique_ptr.hpp"
#include "mmros/tensorrt/tensorrt_common.hpp"
#include "mmros/tensorrt/utility.hpp"

#include <opencv2/core/mat.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sparse4d
{
struct Sparse4dConfig
{
  std::vector<float> mean;
  std::vector<float> std;
  size_t num_cameras;
  size_t num_output;
  float score_threshold;
};

struct Sparse4dCameraInput
{
  std::array<float, 16> projection_mat;
  std::array<float, 2> image_wh;
};

struct Sparse4dInferenceInput
{
  std::vector<cv::Mat> images;
  std::vector<Sparse4dCameraInput> camera_inputs;
  int64_t timestamp_ns;
};

class Sparse4dDetector
{
public:
  using output_type = mmros::archetype::Boxes3D;

  explicit Sparse4dDetector(
    std::string onnx_path, std::string precision, Sparse4dConfig && detector_config);

  mmros::archetype::Result<output_type> do_inference(const Sparse4dInferenceInput & input) noexcept;

private:
  void setup_trt();
  void init_cuda_ptr(size_t batch_size);
  void preprocess(const Sparse4dInferenceInput & input);
  mmros::archetype::Result<output_type> postprocess(size_t batch_size) noexcept;
  void log_tensor_shapes() const;

  std::string onnx_path_;
  std::string precision_;
  std::unique_ptr<Sparse4dConfig> detector_config_;
  std::unique_ptr<mmros::tensorrt::TrtCommon> trt_common_;
  cudaStream_t stream_;

  int64_t in_height_;
  int64_t in_width_;

  mmros::cuda::CudaUniquePtr<float[]> image_d_;
  mmros::cuda::CudaUniquePtr<float[]> projection_mat_d_;
  mmros::cuda::CudaUniquePtr<float[]> image_wh_d_;

  mmros::cuda::CudaUniquePtr<float[]> classification_d_;
  mmros::cuda::CudaUniquePtr<float[]> prediction_d_;
  mmros::cuda::CudaUniquePtr<float[]> quality_d_;
  mmros::cuda::CudaUniquePtr<int64_t[]> instance_id_d_;

  bool has_logged_runtime_shapes_;
  bool has_logged_invalid_output_stats_;
};
}  // namespace sparse4d

#endif  // SPARSE4D__SPARSE4D_DETECTOR_HPP_
