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

#ifndef MMROS__ARCHETYPE__BOX_HPP_
#define MMROS__ARCHETYPE__BOX_HPP_

#include <vector>

namespace mmros::archetype
{
/**
 * @brief An enum representing 2D box format.
 */
enum class BoxFormat2D {
  XYXY = 0,  //!< (xmin, ymin, xmax, ymax) order.
  XYWH = 1   //!< (cx, cy, width, height) order.
};

/**
 * @brief A class for 2D bounding box.
 */
struct Box2D
{
  float xmin;   //!< Top left x.
  float ymin;   //!< Top left y.
  float xmax;   //!< Bottom right x.
  float ymax;   //!< Bottom right y.
  float score;  //!< Score in [0.0, 1.0].
  int label;    //!< Label ID.

  /**
   * @brief Construct a new Box2D object.
   *
   * @param xmin Top left x.
   * @param ymin Top left y.
   * @param xmax Bottom right x.
   * @param ymax Bottom right y.
   * @param score Score in [0.0, 1.0].
   * @param label Label ID.
   */
  Box2D(float xmin, float ymin, float xmax, float ymax, float score, int label)
  : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax), score(score), label(label)
  {
  }
};

/**
 * @brief A class for 3D bounding box.
 */
struct Box3D
{
  float x;       //!< Center x.
  float y;       //!< Center y.
  float z;       //!< Center z.
  float length;  //!< X-direction length.
  float width;   //!< Y-direction width.
  float height;  //!< Z-direction height.
  float yaw;     //!< Box heading in [rad].
  float score;   //!< Score in [0.0, 1.0].
  int label;     //!< Label ID.

  /**
   * @brief Construct a new Box3D object.
   *
   * @param x Center x.
   * @param y Center y.
   * @param z Center z.
   * @param length X-direction length.
   * @param width Y-direction width.
   * @param height Z-direction height.
   * @param yaw Box heading in [rad].
   * @param score Score in [0.0, 1.0].
   * @param label Label ID.
   */
  Box3D(
    float x, float y, float z, float length, float width, float height, float yaw, float score,
    int label)
  : x(x),
    y(y),
    z(z),
    length(length),
    width(width),
    height(height),
    yaw(yaw),
    score(score),
    label(label)
  {
  }
};

using Boxes2D = std::vector<Box2D>;
using Boxes3D = std::vector<Box3D>;
}  // namespace mmros::archetype
#endif  // MMROS__ARCHETYPE__BOX_HPP_
