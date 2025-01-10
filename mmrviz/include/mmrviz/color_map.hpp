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
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstddef>

namespace mmrviz
{
class ColorMap
{
public:
  static const size_t num_color = 80;

  ColorMap()
  {
    table_ = {{{0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},
               {77, 190, 238},  {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},
               {255, 128, 0},   {191, 191, 0},   {0, 255, 0},     {0, 0, 255},     {170, 0, 255},
               {85, 85, 0},     {85, 170, 0},    {85, 255, 0},    {170, 85, 0},    {170, 170, 0},
               {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},   {0, 85, 128},
               {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
               {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128},
               {255, 0, 128},   {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},
               {0, 170, 255},   {0, 255, 255},   {85, 0, 255},    {85, 85, 255},   {85, 170, 255},
               {85, 255, 255},  {170, 0, 255},   {170, 85, 255},  {170, 170, 255}, {170, 255, 255},
               {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},      {128, 0, 0},
               {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
               {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},
               {0, 0, 85},      {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},
               {0, 0, 0},       {36, 36, 36},    {73, 73, 73},    {109, 109, 109}, {145, 145, 145},
               {182, 182, 182}, {219, 219, 219}, {0, 100, 189},   {80, 183, 189},  {128, 128, 0}}};

    lut_ = cv::Mat(256, 1, CV_8UC3);
    for (int i = 0; i < 256; ++i) {
      auto color = table_[i % num_color];
      lut_.at<cv::Vec3b>(i, 0) = cv::Vec3b(
        static_cast<uchar>(color[0]), static_cast<uchar>(color[1]), static_cast<uchar>(color[2]));
    }
  }

  cv::Scalar operator()(size_t index)
  {
    size_t idx = index % num_color;
    return table_.at(idx);
  }

  const cv::Mat & getLookUpTable() const noexcept { return lut_; }

private:
  std::array<cv::Scalar, num_color> table_;
  cv::Mat lut_;
};
}  // namespace mmrviz
#endif  // MMRVIZ__COLOR_MAP_HPP_
