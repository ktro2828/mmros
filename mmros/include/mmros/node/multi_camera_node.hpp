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

#ifndef MMROS__NODE__MULTI_CAMERA_NODE_HPP_
#define MMROS__NODE__MULTI_CAMERA_NODE_HPP_

#include <image_transport/subscriber.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <functional>
#include <string>
#include <vector>

namespace mmros::node
{
/**
 * @brief MultiCameraNode class for subscribing to multiple image topics.
 */
class MultiCameraNode : public rclcpp::Node
{
public:
  using Callback = std::function<void(sensor_msgs::msg::Image::ConstSharedPtr, size_t)>;

  /**
   * @brief Constructor for MultiCameraNode.
   *
   * @param name Name of the node.
   * @param options Node options.
   */
  MultiCameraNode(const std::string & name, const rclcpp::NodeOptions & options);

  /**
   * @brief Connect to multiple image topics.
   *
   * @param image_topics Vector of image topic names.
   * @param callback Callback function to be called when a new image is received.
   * @param use_raw Whether to use raw images or not.
   */
  void onConnect(
    const std::vector<std::string> & image_topics, const Callback & callback, bool use_raw);

protected:
  rclcpp::TimerBase::SharedPtr connection_timer_;  //!< Topic connection timer.

private:
  /**
   * @brief Connect to a single image topic.
   *
   * @param camera_id Camera ID.
   * @param image_topic Image topic name.
   * @param callback Callback function to be called when a new image is received.
   * @param use_raw Whether to use raw images or not.
   * @return True if the connection was successful, false otherwise.
   */
  bool onConnectForSingleCamera(
    size_t camera_id, const std::string & image_topic, const Callback & callback, bool use_raw);

  std::vector<image_transport::Subscriber> subscriptions_;  //!< Subscribers for each camera topic.
};
}  // namespace mmros::node
#endif  // MMROS__NODE__MULTI_CAMERA_NODE_HPP_
