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

#include "sparse4d/sparse4d_node.hpp"

#include "mmros/archetype/exception.hpp"

#include <image_transport/image_transport.hpp>
#include <rclcpp/create_timer.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <cv_bridge/cv_bridge.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/exceptions.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sparse4d
{
namespace
{
std::vector<std::string> default_image_topics()
{
  return {
    "/nuscenes/CAM_FRONT/image",       "/nuscenes/CAM_FRONT_LEFT/image",
    "/nuscenes/CAM_FRONT_RIGHT/image", "/nuscenes/CAM_BACK/image",
    "/nuscenes/CAM_BACK_LEFT/image",   "/nuscenes/CAM_BACK_RIGHT/image",
  };
}

std::vector<std::string> default_camera_info_topics()
{
  return {
    "/nuscenes/CAM_FRONT/camera_info",       "/nuscenes/CAM_FRONT_LEFT/camera_info",
    "/nuscenes/CAM_FRONT_RIGHT/camera_info", "/nuscenes/CAM_BACK/camera_info",
    "/nuscenes/CAM_BACK_LEFT/camera_info",   "/nuscenes/CAM_BACK_RIGHT/camera_info",
  };
}

double stamp_diff_sec(const rclcpp::Time & lhs, const rclcpp::Time & rhs)
{
  return std::abs((lhs - rhs).seconds());
}
}  // namespace

Sparse4dNode::Sparse4dNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("sparse4d", options),
  tf_buffer_(get_clock()),
  tf_listener_(tf_buffer_),
  last_processed_stamp_ns_(-1),
  build_only_(false)
{
  auto onnx_path = declare_parameter<std::string>("tensorrt.onnx_path");
  auto precision = declare_parameter<std::string>("tensorrt.precision", "fp16");

  auto mean_param =
    declare_parameter<std::vector<double>>("detector.mean", {123.675, 116.28, 103.53});
  auto std_param = declare_parameter<std::vector<double>>("detector.std", {58.395, 57.12, 57.375});
  auto score_threshold = declare_parameter<double>("detector.score_threshold", 0.3);

  Sparse4dConfig detector_config;
  detector_config.mean.assign(mean_param.begin(), mean_param.end());
  detector_config.std.assign(std_param.begin(), std_param.end());
  detector_config.num_cameras = 6;
  detector_config.num_output = 300;
  detector_config.score_threshold = static_cast<float>(score_threshold);
  detector_ = std::make_unique<Sparse4dDetector>(
    std::move(onnx_path), std::move(precision), std::move(detector_config));

  image_topics_ =
    declare_parameter<std::vector<std::string>>("input.image_topics", default_image_topics());
  camera_info_topics_ = declare_parameter<std::vector<std::string>>(
    "input.camera_info_topics", default_camera_info_topics());
  lidar_frame_ = declare_parameter<std::string>("input.lidar_frame", "LIDAR_TOP");
  sync_tolerance_sec_ = declare_parameter<double>("input.sync_tolerance_sec", 0.05);

  if (image_topics_.size() != camera_info_topics_.size()) {
    throw mmros::archetype::MmRosException(
      mmros::archetype::MmRosError_t::INVALID_VALUE,
      "input.image_topics and input.camera_info_topics must have the same size");
  }

  latest_images_.resize(image_topics_.size());
  latest_camera_infos_.resize(image_topics_.size());

  using std::chrono_literals::operator""ms;
  bool use_raw = declare_parameter<bool>("use_raw", false);
  pub_ = create_publisher<mmros_msgs::msg::BoxArray3d>("~/output/boxes", 1);

  build_only_ = declare_parameter<bool>("build_only", false);
  if (build_only_) {
    RCLCPP_INFO(get_logger(), "Sparse4D package skeleton initialized and exit.");
    build_only_exit_timer_ = rclcpp::create_timer(this, get_clock(), 1ms, [this]() {
      if (build_only_exit_timer_) {
        build_only_exit_timer_->cancel();
      }
      rclcpp::shutdown();
    });
    return;
  }

  connection_timer_ = rclcpp::create_timer(
    this, get_clock(), 100ms, [this, use_raw]() { this->connect_topics(use_raw); });
}

bool Sparse4dNode::is_build_only() const noexcept
{
  return build_only_;
}

void Sparse4dNode::connect_topics(bool use_raw)
{
  if (!subscriptions_.empty()) {
    if (connection_timer_) {
      connection_timer_->cancel();
    }
    return;
  }

  const auto qos = rclcpp::SensorDataQoS();
  subscriptions_.reserve(image_topics_.size());
  camera_info_subscriptions_.reserve(camera_info_topics_.size());
  for (size_t camera_id = 0; camera_id < image_topics_.size(); ++camera_id) {
    const auto & image_topic = image_topics_.at(camera_id);
    rclcpp::SubscriptionOptions options;
    options.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    auto subscription = image_transport::create_subscription(
      this, image_topic,
      [this, camera_id](sensor_msgs::msg::Image::ConstSharedPtr msg) { callback(msg, camera_id); },
      use_raw ? "raw" : "compressed", qos.get_rmw_qos_profile(), options);
    subscriptions_.emplace_back(std::move(subscription));

    const auto & camera_info_topic = camera_info_topics_.at(camera_id);
    auto camera_info_subscription = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic, qos, [this, camera_id](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
        camera_info_callback(msg, camera_id);
      });
    camera_info_subscriptions_.emplace_back(std::move(camera_info_subscription));
  }

  if (connection_timer_) {
    connection_timer_->cancel();
  }
}

void Sparse4dNode::camera_info_callback(
  sensor_msgs::msg::CameraInfo::ConstSharedPtr msg, size_t camera_id)
{
  if (camera_id >= latest_camera_infos_.size()) {
    return;
  }

  latest_camera_infos_.at(camera_id) = msg;
  maybe_run_inference();
}

void Sparse4dNode::callback(sensor_msgs::msg::Image::ConstSharedPtr msg, size_t camera_id)
{
  if (camera_id >= latest_images_.size()) {
    return;
  }

  latest_images_.at(camera_id) = msg;
  maybe_run_inference();
}

void Sparse4dNode::maybe_run_inference()
{
  auto missing_image = std::find_if(
    latest_images_.cbegin(), latest_images_.cend(),
    [](const auto & image) { return !image.has_value(); });
  if (missing_image != latest_images_.cend()) {
    return;
  }

  auto missing_camera_info = std::find_if(
    latest_camera_infos_.cbegin(), latest_camera_infos_.cend(),
    [](const auto & camera_info) { return !camera_info.has_value(); });
  if (missing_camera_info != latest_camera_infos_.cend()) {
    return;
  }

  const rclcpp::Time reference_stamp(latest_images_.front().value()->header.stamp);
  if (reference_stamp.nanoseconds() == last_processed_stamp_ns_) {
    return;
  }

  if (!inputs_are_synchronized(reference_stamp)) {
    return;
  }

  Sparse4dInferenceInput input;
  input.images.reserve(latest_images_.size());
  input.camera_inputs.reserve(latest_images_.size());
  input.timestamp_ns = reference_stamp.nanoseconds();

  for (size_t camera_id = 0; camera_id < latest_images_.size(); ++camera_id) {
    const auto & image_msg = latest_images_.at(camera_id);
    cv_bridge::CvImagePtr in_image_ptr;
    try {
      in_image_ptr = cv_bridge::toCvCopy(*image_msg, sensor_msgs::image_encodings::RGB8);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_STREAM(get_logger(), "cv_bridge exception: " << e.what());
      return;
    }

    auto camera_input = build_camera_input(camera_id, reference_stamp);
    if (!camera_input.has_value()) {
      return;
    }

    input.images.push_back(in_image_ptr->image);
    input.camera_inputs.push_back(*camera_input);
  }

  try {
    const auto boxes = detector_->do_inference(input).unwrap();

    mmros_msgs::msg::BoxArray3d output_msg;
    output_msg.header = latest_images_.front().value()->header;
    output_msg.header.frame_id = lidar_frame_;
    output_msg.boxes.reserve(boxes.size());

    for (const auto & box : boxes) {
      mmros_msgs::msg::Box3d box_msg;
      box_msg.pose.position.x = box.x;
      box_msg.pose.position.y = box.y;
      box_msg.pose.position.z = box.z;

      tf2::Quaternion quaternion;
      quaternion.setRPY(0.0, 0.0, static_cast<double>(box.yaw));
      box_msg.pose.orientation.x = quaternion.x();
      box_msg.pose.orientation.y = quaternion.y();
      box_msg.pose.orientation.z = quaternion.z();
      box_msg.pose.orientation.w = quaternion.w();

      box_msg.dimensions.x = box.length;
      box_msg.dimensions.y = box.width;
      box_msg.dimensions.z = box.height;
      box_msg.score = box.score;
      box_msg.label = static_cast<uint32_t>(box.label);
      output_msg.boxes.push_back(std::move(box_msg));
    }

    pub_->publish(output_msg);
  } catch (const mmros::archetype::MmRosException & e) {
    RCLCPP_ERROR_STREAM(get_logger(), e.what());
    return;
  }
  last_processed_stamp_ns_ = reference_stamp.nanoseconds();
}

bool Sparse4dNode::inputs_are_synchronized(const rclcpp::Time & reference_stamp) const
{
  for (size_t camera_id = 0; camera_id < latest_images_.size(); ++camera_id) {
    const rclcpp::Time image_stamp(latest_images_.at(camera_id).value()->header.stamp);
    const rclcpp::Time camera_info_stamp(latest_camera_infos_.at(camera_id).value()->header.stamp);

    if (stamp_diff_sec(reference_stamp, image_stamp) > sync_tolerance_sec_) {
      return false;
    }

    if (stamp_diff_sec(reference_stamp, camera_info_stamp) > sync_tolerance_sec_) {
      return false;
    }
  }

  return true;
}

std::optional<Sparse4dCameraInput> Sparse4dNode::build_camera_input(
  size_t camera_id, const rclcpp::Time & stamp)
{
  const auto & camera_info_msg = latest_camera_infos_.at(camera_id).value();

  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_.lookupTransform(
      camera_info_msg->header.frame_id, lidar_frame_, stamp, rclcpp::Duration::from_seconds(0.01));
  } catch (const tf2::TransformException & e) {
    RCLCPP_WARN_STREAM_THROTTLE(get_logger(), *get_clock(), 5000, e.what());
    return std::nullopt;
  }

  tf2::Quaternion quaternion(
    transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z,
    transform.transform.rotation.w);
  tf2::Matrix3x3 rotation(quaternion);

  std::array<float, 9> intrinsic{};
  for (size_t index = 0; index < intrinsic.size(); ++index) {
    intrinsic.at(index) = static_cast<float>(camera_info_msg->k.at(index));
  }

  std::array<float, 12> extrinsic{};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      extrinsic.at(row * 4 + column) = static_cast<float>(rotation[row][column]);
    }
  }
  extrinsic.at(3) = static_cast<float>(transform.transform.translation.x);
  extrinsic.at(7) = static_cast<float>(transform.transform.translation.y);
  extrinsic.at(11) = static_cast<float>(transform.transform.translation.z);

  Sparse4dCameraInput camera_input{};
  camera_input.projection_mat.fill(0.0F);
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 4; ++column) {
      float value = 0.0F;
      for (int inner = 0; inner < 3; ++inner) {
        value += intrinsic.at(row * 3 + inner) * extrinsic.at(inner * 4 + column);
      }
      camera_input.projection_mat.at(row * 4 + column) = value;
    }
  }
  camera_input.projection_mat.at(15) = 1.0F;
  camera_input.image_wh = {
    static_cast<float>(camera_info_msg->width),
    static_cast<float>(camera_info_msg->height),
  };
  return camera_input;
}
}  // namespace sparse4d

RCLCPP_COMPONENTS_REGISTER_NODE(sparse4d::Sparse4dNode)
