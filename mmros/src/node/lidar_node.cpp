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

#include "mmros/node/lidar_node.hpp"

#include "mmros/node/utility.hpp"

#include <glog/logging.h>

#include <string>

namespace mmros::node
{
LidarNode::LidarNode(const std::string & name, const rclcpp::NodeOptions & options)
: rclcpp::Node(name, options)
{
  google::InitGoogleLogging(name.c_str());
  google::InstallFailureSignalHandler();
}

void LidarNode::onConnect(const Callback & callback)
{
  const auto pointcloud_topic = resolveTopicName(this, "~/input/pointcloud");

  const auto pointcloud_qos = getTopicQos(this, pointcloud_topic);
  if (pointcloud_qos) {
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      pointcloud_topic, *pointcloud_qos, callback);

    if (connection_timer_) {
      connection_timer_->cancel();
      RCLCPP_INFO(
        get_logger(), "Successfully subscribed to %s, connection timer canceled",
        pointcloud_topic.c_str());
    }
  } else {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000, "Failed to subscribe to %s", pointcloud_topic.c_str());
  }
}
}  // namespace mmros::node
