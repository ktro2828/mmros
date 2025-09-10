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

#ifndef MMROS__NODE__SINGLE_CAMERA_NODE_HPP_
#define MMROS__NODE__SINGLE_CAMERA_NODE_HPP_

#include <image_transport/subscriber.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <string>

namespace mmros::node
{
/**
 * @brief SingleCameraNode class for subscribing to a single image topic.
 */
class SingleCameraNode : public rclcpp::Node
{
public:
  /**
   * @brief Construct a new SingleCameraNode object.
   *
   * @param name Node name.
   * @param options Node options.
   */
  SingleCameraNode(const std::string & name, const rclcpp::NodeOptions & options);

  /**
   * @brief Check node connection and start subscribing.
   *
   * @param callback Callback function.
   * @param use_raw Indicates whether to use raw image.
   */
  void onConnect(
    const std::function<void(sensor_msgs::msg::Image::ConstSharedPtr)> & callback, bool use_raw);

protected:
  rclcpp::TimerBase::SharedPtr connection_timer_;  //!< Topic connection timer.

private:
  image_transport::Subscriber subscription_;  //!< Image subscription.
};
}  // namespace mmros::node
#endif  // MMROS__NODE__SINGLE_CAMERA_NODE_HPP_
