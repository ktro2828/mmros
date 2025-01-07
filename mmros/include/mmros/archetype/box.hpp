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

namespace mmros
{
struct Box2D
{
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  float score;
  int label;

  Box2D(float xmin, float ymin, float xmax, float ymax, float score, int label)
  : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax), score(score), label(label)
  {
  }
};

struct Box3D
{
  float x;
  float y;
  float z;
  float length;
  float width;
  float height;
  float yaw;
  float score;
  int label;

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
}  // namespace mmros
#endif  // MMROS__ARCHETYPE__BOX_HPP_
