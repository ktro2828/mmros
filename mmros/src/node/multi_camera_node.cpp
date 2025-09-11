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

#include "mmros/node/multi_camera_node.hpp"

#include "mmros/node/utility.hpp"

#include <image_transport/image_transport.hpp>

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace mmros::node
{
MultiCameraNode::MultiCameraNode(const std::string & name, const rclcpp::NodeOptions & options)
: rclcpp::Node(name, options)
{
  google::InitGoogleLogging(name.c_str());
  google::InstallFailureSignalHandler();
}

void MultiCameraNode::onConnect(
  const std::vector<std::string> & image_topics,
  const std::function<void(sensor_msgs::msg::Image::ConstSharedPtr)> & callback, bool use_raw)
{
  subscriptions_.clear();

  bool success =
    std::all_of(image_topics.begin(), image_topics.end(), [&](const auto & image_topic) {
      return onConnectForSingleCamera(image_topic, callback, use_raw);
    });

  if (success && connection_timer_) {
    connection_timer_->cancel();
    RCLCPP_INFO(get_logger(), "Successfully connected to all cameras, connection timer canceled");
  }
}

bool MultiCameraNode::onConnectForSingleCamera(
  const std::string & image_topic,
  const std::function<void(sensor_msgs::msg::Image::ConstSharedPtr)> & callback, bool use_raw)
{
  const auto image_qos = getTopicQos(this, use_raw ? image_topic : image_topic + "/compressed");
  if (image_qos) {
    rclcpp::SubscriptionOptions options;
    options.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    const auto subscription = image_transport::create_subscription(
      this, image_topic, callback, use_raw ? "raw" : "compressed", image_qos->get_rmw_qos_profile(),
      options);

    subscriptions_.emplace_back(std::move(subscription));
    return true;
  } else {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000, "Failed to create subscription for topic '%s'",
      image_topic.c_str());
    return false;
  }
}
}  // namespace mmros::node
