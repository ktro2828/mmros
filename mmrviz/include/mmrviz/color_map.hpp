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

#ifndef MMRVIZ__COLOR_MAP_HPP_
#define MMRVIZ__COLOR_MAP_HPP_

#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/hal/interface.h>

#include <cstddef>
namespace mmrviz
{
class ColorMap
{
public:
  static const size_t num_color = 512;

  ColorMap()
  {
    cv::Mat src = cv::Mat::zeros(cv::Size(num_color, 1), CV_8UC1);
    for (size_t i = 0; i < num_color; ++i) {
      src.at<unsigned char>(0, i) = i;
    }
    cv::applyColorMap(src, table_, cv::COLORMAP_HSV);
  }

  cv::Scalar operator()(size_t index)
  {
    size_t idx = num_color % index;
    return table_.at<cv::Vec3b>(0, idx);
  }

private:
  cv::Mat table_;
};
}  // namespace mmrviz
#endif  // MMRVIZ__COLOR_MAP_HPP_
