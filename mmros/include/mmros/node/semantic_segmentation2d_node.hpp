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

#ifndef MMROS__NODE__SEMANTIC_SEGMENTATION2D_NODE_HPP_
#define MMROS__NODE__SEMANTIC_SEGMENTATION2D_NODE_HPP_

#include "mmros/detector/semantic_segmenter2d.hpp"
#include "mmros/node/single_camera_node.hpp"

#include <sensor_msgs/msg/image.hpp>

#include <memory>

namespace mmros::node
{
/**
 * @brief A ROS 2 node class for 2D semantic segmenter.
 */
class SemanticSegmentation2dNode : public SingleCameraNode
{
public:
  /**
   * @brief Construct a new SemanticSegmentation2dNode object.
   *
   * @param options Node options.
   */
  explicit SemanticSegmentation2dNode(const rclcpp::NodeOptions & options);

private:
  /**
   * @brief Main callback for input image.
   *
   * @param msg Input image message.
   */
  void callback(sensor_msgs::msg::Image::ConstSharedPtr msg);

  std::unique_ptr<detector::SemanticSegmenter2D> detector_;    //!< TensorRT detector.
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;  //!< Output publisher.
};
}  // namespace mmros::node
#endif  // MMROS__NODE__SEMANTIC_SEGMENTATION2D_NODE_HPP_
