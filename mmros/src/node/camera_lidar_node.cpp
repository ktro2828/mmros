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

#include "mmros/node/camera_lidar_node.hpp"

#include "mmros/node/utility.hpp"

#include <image_transport/image_transport.hpp>

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace mmros::node
{
CameraLidarNode::CameraLidarNode(const std::string & name, const rclcpp::NodeOptions & options)
: rclcpp::Node(name, options)
{
  google::InitGoogleLogging(name.c_str());
  google::InstallFailureSignalHandler();
}

void CameraLidarNode::onConnect(
  const PointCloudCallback & pointcloud_callback, const std::vector<std::string> & image_topics,
  const ImageCallback & image_callback, bool use_raw)
{
  image_subscriptions_.clear();

  bool success =
    onConnectLidar(pointcloud_callback) &&
    std::all_of(image_topics.begin(), image_topics.end(), [&](const auto & image_topic) {
      return onConnectForSingleCamera(image_topic, image_callback, use_raw);
    });

  if (success && connection_timer_) {
    connection_timer_->cancel();
    RCLCPP_INFO(
      get_logger(), "Successfully connected to lidar and all cameras, connection timer canceled");
  }
}

bool CameraLidarNode::onConnectLidar(const PointCloudCallback & pointcloud_callback)
{
  const auto pointcloud_topic = resolveTopicName(this, "~/input/pointcloud");

  const auto pointcloud_qos = getTopicQos(this, pointcloud_topic);
  if (pointcloud_qos) {
    pointcloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      pointcloud_topic, *pointcloud_qos, pointcloud_callback);
    return true;
  } else {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000, "Failed to subscribe to %s", pointcloud_topic.c_str());
    return false;
  }
}

bool CameraLidarNode::onConnectForSingleCamera(
  const std::string & image_topic, const ImageCallback & image_callback, bool use_raw)
{
  const auto image_qos = getTopicQos(this, use_raw ? image_topic : image_topic + "/compressed");
  if (image_qos) {
    rclcpp::SubscriptionOptions options;
    options.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    const auto subscription = image_transport::create_subscription(
      this, image_topic, image_callback, use_raw ? "raw" : "compressed",
      image_qos->get_rmw_qos_profile(), options);

    image_subscriptions_.emplace_back(std::move(subscription));
    return true;
  } else {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000, "Failed to create subscription for topic '%s'",
      image_topic.c_str());
    return false;
  }
}
}  // namespace mmros::node
