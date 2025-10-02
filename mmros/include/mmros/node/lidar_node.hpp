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

#ifndef MMROS__NODE__LIDAR_NODE_HPP_
#define MMROS__NODE__LIDAR_NODE_HPP_

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <functional>
#include <string>

namespace mmros::node
{
/**
 * @brief LidarNode class for subscribing to point cloud topic.
 */
class LidarNode : public rclcpp::Node
{
public:
  using Callback = std::function<void(sensor_msgs::msg::PointCloud2::ConstSharedPtr)>;

  /**
   * @brief Constructor for LidarNode.
   *
   * @param name The name of the node.
   * @param options The options for the node.
   */
  LidarNode(const std::string & name, const rclcpp::NodeOptions & options);

protected:
  /**
   * @brief Check node connection and start subscribing.
   *
   * @param callback Callback function.
   */
  void on_connect(const Callback & callback);

  rclcpp::TimerBase::SharedPtr connection_timer_;  //!< Topic connection timer.

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
    subscription_;  //!< PointCloud subscription.
};
}  // namespace mmros::node
#endif  // MMROS__NODE__LIDAR_NODE_HPP_
