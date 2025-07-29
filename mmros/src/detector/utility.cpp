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

#include "mmros/detector/utility.hpp"

#include "mmros/archetype/box.hpp"

#include <limits>
#include <utility>

namespace mmros::detector
{
std::pair<float, int> compute_max_score(
  const float * box_ptr, const int * label_ptr, size_t box_dim, size_t class_dim,
  archetype::BoxFormat2D box_format)
{
  if (archetype::include_score_in_box(box_format)) {
    const float score = box_ptr[4];
    const int label = label_ptr[0];
    return {score, label};
  } else {
    float max_score = -std::numeric_limits<float>::max();
    int label = -1;
    for (size_t k = 0; k < class_dim; ++k) {
      float score = label_ptr[k];
      if (score > max_score) {
        max_score = score;
        label = k;
      }
    }
    return {max_score, label};
  }
}

archetype::Box2D to_box2d(
  const float * box_ptr, const int * label_ptr, size_t box_dim, size_t class_dim, int64_t in_height,
  int64_t in_width, float scale, archetype::BoxFormat2D box_format)
{
  const auto [score, label] = compute_max_score(box_ptr, label_ptr, box_dim, class_dim, box_format);

  float xmin, ymin, xmax, ymax;
  if (archetype::is_box_xyxy(box_format)) {
    xmin = box_ptr[0] / scale;
    ymin = box_ptr[1] / scale;
    xmax = box_ptr[2] / scale;
    ymax = box_ptr[3] / scale;
  } else {
    float cx = box_ptr[0];
    float cy = box_ptr[1];
    float width = box_ptr[2];
    float height = box_ptr[3];
    xmin = (cx - 0.5 * width) / scale;
    ymin = (cy - 0.5 * height) / scale;
    xmax = (cx + 0.5 * width) / scale;
    ymax = (cy + 0.5 * height) / scale;
  }

  if (archetype::is_box_normalized(box_format)) {
    xmin *= in_width;
    ymin *= in_height;
    xmax *= in_width;
    ymax *= in_height;
  }

  return {xmin, ymin, xmax, ymax, score, label};
}
}  // namespace mmros::detector
