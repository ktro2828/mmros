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

#ifndef MMRVIZ__VISUALIZER__SEGMENTATION2D_VISUALIZER_HPP_
#define MMRVIZ__VISUALIZER__SEGMENTATION2D_VISUALIZER_HPP_

#include <image_transport/publisher.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/timer.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

#include <memory>
#include <optional>
#include <string>

namespace mmrviz::visualizer
{
class Segmentation2dVisualizer : public rclcpp::Node
{
public:
  explicit Segmentation2dVisualizer(const rclcpp::NodeOptions & options);

  void callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & mask_msg);

private:
  void onConnect(bool use_raw);

  std::optional<rclcpp::QoS> getTopicQos(const std::string & query_topic);

  rclcpp::TimerBase::SharedPtr timer_;
  image_transport::SubscriberFilter image_sub_;
  image_transport::SubscriberFilter mask_sub_;

  using ExactTimeSyncPolicy =
    message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  using ExactTimeSync = message_filters::Synchronizer<ExactTimeSyncPolicy>;
  std::shared_ptr<ExactTimeSync> sync_ptr_;

  image_transport::Publisher pub_;
};
}  // namespace mmrviz::visualizer
#endif  // MMRVIZ__VISUALIZER__SEGMENTATION2D_VISUALIZER_HPP_
