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

#ifndef SPARSE4D__SPARSE4D_NODE_HPP_
#define SPARSE4D__SPARSE4D_NODE_HPP_

#include "sparse4d/sparse4d_detector.hpp"

#include <image_transport/subscriber.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

#include <mmros_msgs/msg/box_array3d.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sparse4d
{
class Sparse4dNode : public rclcpp::Node
{
public:
  using CameraInfoSubscription = rclcpp::Subscription<sensor_msgs::msg::CameraInfo>;

  explicit Sparse4dNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  [[nodiscard]] bool is_build_only() const noexcept;

private:
  void connect_topics(bool use_raw);
  void camera_info_callback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg, size_t camera_id);
  void callback(sensor_msgs::msg::Image::ConstSharedPtr msg, size_t camera_id);
  void maybe_run_inference();
  bool inputs_are_synchronized(const rclcpp::Time & reference_stamp) const;
  std::optional<Sparse4dCameraInput> build_camera_input(
    size_t camera_id, const rclcpp::Time & stamp);

  std::unique_ptr<Sparse4dDetector> detector_;
  rclcpp::Publisher<mmros_msgs::msg::BoxArray3d>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr connection_timer_;
  rclcpp::TimerBase::SharedPtr build_only_exit_timer_;
  std::vector<image_transport::Subscriber> subscriptions_;
  std::vector<CameraInfoSubscription::SharedPtr> camera_info_subscriptions_;
  std::vector<std::string> image_topics_;
  std::vector<std::string> camera_info_topics_;
  std::vector<std::optional<sensor_msgs::msg::Image::ConstSharedPtr>> latest_images_;
  std::vector<std::optional<sensor_msgs::msg::CameraInfo::ConstSharedPtr>> latest_camera_infos_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string lidar_frame_;
  double sync_tolerance_sec_;
  int64_t last_processed_stamp_ns_;
  bool build_only_;
};
}  // namespace sparse4d

#endif  // SPARSE4D__SPARSE4D_NODE_HPP_
