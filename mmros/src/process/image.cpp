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

#include "mmros/process/image.hpp"

#include "mmros/tensorrt/cuda_unique_ptr.hpp"

#include <algorithm>
#include <vector>

namespace mmros::process
{
void preprocess_image(
  float * input_d, std::vector<float> & scales, const std::vector<cv::Mat> & images,
  int64_t in_width, int64_t in_height, float * mean, float * std, cudaStream_t stream)
{
  // (B, C, H, W)
  const auto batch_size = images.size();

  cuda::CudaUniquePtrHost<unsigned char[]> img_buf_h;
  cuda::CudaUniquePtr<unsigned char[]> img_buf_d;

  scales.clear();
  for (auto b = 0; b < images.size(); ++b) {
    const auto & img = images.at(b);
    if (!img_buf_h) {
      img_buf_h = cuda::make_unique_host<unsigned char[]>(
        img.cols * img.rows * 3 * batch_size, cudaHostAllocWriteCombined);
      img_buf_d = cuda::make_unique<unsigned char[]>(img.cols * img.rows * 3 * batch_size);
    }
    const float scale =
      std::min(static_cast<float>(in_width) / img.cols, static_cast<float>(in_height) / img.rows);
    scales.emplace_back(scale);

    int index = b * img.cols * img.rows * 3;
    // Copy into pinned memory
    memcpy(img_buf_h.get() + index, &img.data[0], img.cols * img.rows * 3 * sizeof(unsigned char));
  }

  CHECK_CUDA_ERROR(
    ::cudaMemcpyAsync(
      img_buf_d.get(), img_buf_h.get(),
      images[0].cols * images[0].rows * 3 * batch_size * sizeof(unsigned char),
      ::cudaMemcpyHostToDevice, stream));

  process::resize_bilinear_letterbox_nhwc_to_nchw32_batch_gpu(
    input_d, img_buf_d.get(), in_width, in_height, 3, images[0].cols, images[0].rows, 3, batch_size,
    mean, std, stream);

  CHECK_CUDA_ERROR(cudaGetLastError());
}
}  // namespace mmros::process
