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

#ifndef MASK2FORMER_NODE_HPP_
#define MASK2FORMER_NODE_HPP_

#include <mmros/node/semantic_segmentation2d_node.hpp>

namespace mmros::mask2former
{
class Mask2FormerNode : public SemanticSegmentation2dNode
{
public:
  explicit Mask2FormerNode(const rclcpp::NodeOptions & options);
};
}  // namespace mmros::mask2former
#endif  // MASK2FORMER_NODE_HPP_
