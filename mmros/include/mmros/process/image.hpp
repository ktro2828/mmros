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

#ifndef MMROS__PROCESS__IMAGE_HPP_
#define MMROS__PROCESS__IMAGE_HPP_

#include "mmros/process/image_kernel.hpp"

#include <opencv2/core/mat.hpp>

#include <vector>

namespace mmros::process
{
/**
 * Run preprocessing for image.
 *
 * @param input_d Pointer to input image on device.
 * @param scales Mutable reference to the vector of scale factors.
 * @param images Read-only reference to the vector of source images.
 * @param in_width Model input width.
 * @param in_height Model input height.
 * @param mean Pointer to the image mean values.
 * @param std Pointer to the image std values.
 * @param stream CUDA stream.
 */
void preprocess_image(
  float * input_d, std::vector<float> & scales, const std::vector<cv::Mat> & images,
  int64_t in_width, int64_t in_height, float * mean, float * std, cudaStream_t stream);
}  // namespace mmros::process
#endif  // MMROS__PROCESS__IMAGE_HPP_
