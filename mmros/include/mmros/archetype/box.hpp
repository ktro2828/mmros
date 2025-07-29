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

#include "mmros/archetype/exception.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace mmros::archetype
{
/**
 * @brief An enum representing 2D box format.
 */
enum class BoxFormat2D {
  XYXY = 0,        //!< (xmin, ymin, xmax, ymax) order.
  XYWH = 1,        //!< (cx, cy, width, height) order.
  XYXYS = 2,       //!< (xmin, ymin, xmax, ymax, score) order.
  XYWHS = 3,       //!< (cx, cy, width, height, score) order.
  XYXY_NORM = 4,   //!< (xmin, ymin, xmax, ymax) order normalized.
  XYWH_NORM = 5,   //!< (cx, cy, width, height) order normalized.
  XYXYS_NORM = 6,  //!< (xmin, ymin, xmax, ymax, score) order normalized.
  XYWHS_NORM = 7,  //!< (cx, cy, width, height, score) order normalized.
};

/**
 * @brief Check whether the output box format is xyxy order.
 *
 * @param format Box format.
 * @return bool Returns `true` if the `format` is the one of (`XYXY`, `XYXYS`, `XYXY_NORM`,
 * `XYXYS_NORM`).
 */
inline bool is_box_xyxy(BoxFormat2D format)
{
  return format == BoxFormat2D::XYXY || format == BoxFormat2D::XYXYS ||
         format == BoxFormat2D::XYXY_NORM || format == BoxFormat2D::XYXYS_NORM;
}

/**
 * @brief Check whether the box format indicates the score feature is included in the output box.
 *
 * @param format Box format.
 * @return bool Returns `true` if the `format` is the one of (`XYXYS`, `XYWHS`, `XYXYS_NORM`,
 * `XYWHS_NORM`).
 */
inline bool include_score_in_box(BoxFormat2D format)
{
  return format == BoxFormat2D::XYXYS || format == BoxFormat2D::XYWHS ||
         format == BoxFormat2D::XYXYS_NORM || format == BoxFormat2D::XYWHS_NORM;
}

/**
 * @brief Check whether the output box is normalized.
 *
 * @param format Box format.
 * @return bool Returns `true` if the `format` is the one of (`XYXY_NORM`, `XYWH_NORM`,
 * `XYXYS_NORM`, `XYWHS_NORM`).
 */
inline bool is_box_normalized(BoxFormat2D format)
{
  return format == BoxFormat2D::XYXY_NORM || format == BoxFormat2D::XYWH_NORM ||
         format == BoxFormat2D::XYXYS_NORM || format == BoxFormat2D::XYWHS_NORM;
}

/**
 * @brief Convert `std::string` to `BoxFormat2D`.
 *
 * @param format_str Format in string.
 * @return BoxFormat2D Returns the correponding box format.
 */
inline BoxFormat2D to_box_format2d(const std::string & format_str)
{
  if (format_str == "XYXY") return BoxFormat2D::XYXY;
  if (format_str == "XYXYS") return BoxFormat2D::XYXYS;
  if (format_str == "XYWH") return BoxFormat2D::XYWH;
  if (format_str == "XYWHS") return BoxFormat2D::XYWHS;
  if (format_str == "XYXY_NORM") return BoxFormat2D::XYXY_NORM;
  if (format_str == "XYWH_NORM") return BoxFormat2D::XYWH_NORM;
  if (format_str == "XYXYS_NORM") return BoxFormat2D::XYXYS_NORM;
  if (format_str == "XYWHS_NORM") return BoxFormat2D::XYWHS_NORM;
  throw MmRosException(MmRosError_t::INVALID_VALUE, "Unexpected box format: " + format_str);
}

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
