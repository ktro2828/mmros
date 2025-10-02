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

#ifndef MMRVIZ__VISUALIZER__BOX_ARRAY2D_VISUALIZER_HPP_
#define MMRVIZ__VISUALIZER__BOX_ARRAY2D_VISUALIZER_HPP_

#include "mmrviz/color_map.hpp"

#include <image_transport/publisher.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <rclcpp/rclcpp.hpp>

#include <mmros_msgs/msg/box_array2d.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mmrviz::visualizer
{
class BoxArray2dVisualizer : public rclcpp::Node
{
public:
  explicit BoxArray2dVisualizer(const rclcpp::NodeOptions & options);

private:
  void on_connect(bool use_raw);

  void callback(
    const sensor_msgs::msg::Image::ConstSharedPtr image_msg,
    const mmros_msgs::msg::BoxArray2d::ConstSharedPtr boxes_msg);

  rclcpp::TimerBase::SharedPtr timer_;
  image_transport::SubscriberFilter image_sub_;
  message_filters::Subscriber<mmros_msgs::msg::BoxArray2d> boxes_sub_;

  using ExactTimeSyncPolicy =
    message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image, mmros_msgs::msg::BoxArray2d>;
  using ExactTimeSync = message_filters::Synchronizer<ExactTimeSyncPolicy>;
  std::shared_ptr<ExactTimeSync> sync_ptr_;

  image_transport::Publisher pub_;

  ColorMap color_map_;
};
}  // namespace mmrviz::visualizer
#endif  // MMRVIZ__VISUALIZER__BOX_ARRAY2D_VISUALIZER_HPP_
