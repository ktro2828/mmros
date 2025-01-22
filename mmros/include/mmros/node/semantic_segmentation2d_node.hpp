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

#include <image_transport/subscriber.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <memory>
#include <optional>
#include <string>

namespace mmros::node
{
/**
 * @brief A ROS 2 node class for 2D semantic segmenter.
 */
class SemanticSegmentation2dNode : public rclcpp::Node
{
public:
  /**
   * @brief Construct a new SemanticSegmentation2dNode object.
   *
   * @param options Node options.
   */
  explicit SemanticSegmentation2dNode(const rclcpp::NodeOptions & options);

  /**
   * @brief Main callback for input image.
   *
   * @param msg Input image message.
   */
  virtual void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);

private:
  /**
   * @brief Check node connection and start subscribing.
   *
   * @param use_raw Indicates whether to use raw image.
   */
  void onConnect(bool use_raw);

  /**
   * @brief Return QoS of the specified topic.
   *
   * If it fails to load the specified QoS, returns `std::nullopt`.
   *
   * @param query_topic Topic name.
   */
  std::optional<rclcpp::QoS> getTopicQos(const std::string & query_topic);

  rclcpp::TimerBase::SharedPtr timer_;                         //!< Timer.
  std::unique_ptr<detector::SemanticSegmenter2D> detector_;    //!< TensorRT detector.
  image_transport::Subscriber sub_;                            //!< Input image subscription.
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;  //!< Output publisher.
};
}  // namespace mmros::node
#endif  // MMROS__NODE__SEMANTIC_SEGMENTATION2D_NODE_HPP_
