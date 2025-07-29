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

#ifndef MMROS__DETECTOR__UTILITY_HPP_
#define MMROS__DETECTOR__UTILITY_HPP_

#include "mmros/archetype/box.hpp"

#include <utility>

namespace mmros::detector
{
/**
 * @brief Returns the maximum score and its corresponding label from the box and label pointers.
 *
 * @param box_ptr Pointer to the box data, which may include scores.
 * @param label_ptr Pointer to the label data.
 * @param box_dim Dimension of the box data.
 * @param class_dim Dimension of the class data.
 * @param box_format Format of the box data.
 * @return std::pair<float, int> A pair containing the maximum score and its corresponding label.
 */
std::pair<float, int> compute_max_score(
  const float * box_ptr, const int * label_ptr, size_t box_dim, size_t class_dim,
  archetype::BoxFormat2D box_format);

/**
 * @brief Converts the box and label pointers to a `Box2D` object.
 *
 * @param box_ptr Pointer to the box data.
 * @param label_ptr Pointer to the label data.
 * @param box_dim Dimension of the box data.
 * @param class_dim Dimension of the class data.
 * @param in_height Model input height.
 * @param in_width Model input width.
 * @param scale Scale factor for the box coordinates.
 * @param box_format Format of the box data.
 * @return archetype::Box2D
 */
archetype::Box2D to_box2d(
  const float * box_ptr, const int * label_ptr, size_t box_dim, size_t class_dim, int64_t in_height,
  int64_t in_width, float scale, archetype::BoxFormat2D box_format);
}  // namespace mmros::detector

#endif  // MMROS__DETECTOR__UTILITY_HPP_
