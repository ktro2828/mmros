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

#ifndef PANOPTIC_FPN_NODE_HPP_
#define PANOPTIC_FPN_NODE_HPP_

#include <mmros/node/panoptic_segmentation2d_node.hpp>
#include <rclcpp/rclcpp.hpp>

namespace mmros::panoptic_fpn
{
class PanopticFpnNode : public PanopticSegmentation2dNode
{
public:
  explicit PanopticFpnNode(const rclcpp::NodeOptions & options);
};
}  // namespace mmros::panoptic_fpn
#endif  // PANOPTIC_FPN_NODE_HPP_
