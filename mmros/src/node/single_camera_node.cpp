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

#include "mmros/node/single_camera_node.hpp"

#include <image_transport/image_transport.hpp>

#include <glog/logging.h>

#include <string>

namespace mmros::node
{
SingleCameraNode::SingleCameraNode(const std::string & name, const rclcpp::NodeOptions & options)
: rclcpp::Node(name)
{
  google::InitGoogleLogging(name.c_str());
  google::InstallFailureSignalHandler();
}

void SingleCameraNode::onConnect(
  const std::function<void(sensor_msgs::msg::Image::ConstSharedPtr)> & callback, bool use_raw)
{
  auto resolve_topic_name = [this](const std::string & query) {
    return this->get_node_topics_interface()->resolve_topic_name(query);
  };

  const auto image_topic = resolve_topic_name("~/input/image");
  const auto image_topic_for_qos_query = use_raw ? image_topic : image_topic + "/compressed";

  const auto image_qos = getTopicQos(image_topic_for_qos_query);
  if (image_qos) {
    const auto transport = use_raw ? "raw" : "compressed";
    subscription_ = image_transport::create_subscription(
      this, image_topic, callback, transport, image_qos->get_rmw_qos_profile());

    if (connection_timer_) {
      connection_timer_->cancel();
      RCLCPP_INFO(
        get_logger(), "Successfully subscribed to %s, connection timer canceled",
        image_topic.c_str());
    }
  }
}

std::optional<rclcpp::QoS> SingleCameraNode::getTopicQos(const std::string & query_topic)
{
  const auto publisher_info = get_publishers_info_by_topic(query_topic);
  if (publisher_info.size() != 1) {
    return std::nullopt;
  } else {
    return publisher_info[0].qos_profile();
  }
}
}  // namespace mmros::node
