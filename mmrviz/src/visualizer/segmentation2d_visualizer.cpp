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

#include "mmrviz/visualizer/segmentation2d_visualizer.hpp"

#include <image_transport/image_transport.hpp>
#include <mmros/node/utility.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/create_timer.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/qos.hpp>

#include <sensor_msgs/image_encodings.hpp>

#include <cv_bridge/cv_bridge.h>

#include <memory>
#include <optional>
#include <string>

namespace mmrviz::visualizer
{
Segmentation2dVisualizer::Segmentation2dVisualizer(const rclcpp::NodeOptions & options)
: rclcpp::Node("segmentation2d_visualizer", options)
{
  using std::chrono_literals::operator""ms;

  bool use_raw = declare_parameter<bool>("use_raw");
  timer_ = rclcpp::create_timer(
    this, get_clock(), 100ms, [this, use_raw]() { this->on_connect(use_raw); });

  pub_ = image_transport::create_publisher(this, "~/output/image");
}

void Segmentation2dVisualizer::on_connect(bool use_raw)
{
  const auto image_topic = mmros::node::resolve_topic_name(this, "~/input/image");
  const auto image_qos = use_raw ? mmros::node::to_topic_qos(this, image_topic)
                                 : mmros::node::to_topic_qos(this, image_topic + "/compressed");

  const auto mask_topic = mmros::node::resolve_topic_name(this, "~/input/mask");
  const auto mask_qos = mmros::node::to_topic_qos(this, mask_topic);

  if (image_qos && mask_qos) {
    const auto transport = use_raw ? "raw" : "compressed";
    image_sub_.subscribe(this, image_topic, transport, image_qos->get_rmw_qos_profile());
    mask_sub_.subscribe(this, mask_topic, "raw", mask_qos->get_rmw_qos_profile());
    sync_ptr_ = std::make_shared<ExactTimeSync>(ExactTimeSyncPolicy(10), image_sub_, mask_sub_);
    sync_ptr_->registerCallback(&Segmentation2dVisualizer::callback, this);

    timer_->cancel();
  }
}

void Segmentation2dVisualizer::callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & mask_msg)
{
  cv_bridge::CvImagePtr in_image_ptr, in_mask_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::RGB8);
    in_mask_ptr = cv_bridge::toCvCopy(mask_msg, sensor_msgs::image_encodings::MONO8);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR_STREAM(get_logger(), "cv_bridge exception: " << e.what());
    return;
  }

  // Check if image and mask sizes match (they should after letterbox correction)
  if (in_image_ptr->image.size() != in_mask_ptr->image.size()) {
    cv::Mat resized_mask;
    cv::resize(
      in_mask_ptr->image, resized_mask, in_image_ptr->image.size(), 0, 0, cv::INTER_NEAREST);
    in_mask_ptr->image = resized_mask;
  }

  // Apply color mapping
  const auto & lut = color_map_.lookup_table();
  cv::Mat color_mask;
  cv::applyColorMap(in_mask_ptr->image, color_mask, lut);

  // Convert color_mask from BGR to RGB for consistency with input image
  cv::cvtColor(color_mask, color_mask, cv::COLOR_BGR2RGB);

  // Create overlay - images should now be the same size
  cv::Mat overlay;
  cv::addWeighted(in_image_ptr->image, 0.6, color_mask, 0.4, 0.0, overlay);

  cv_bridge::CvImage out_image_msg;
  out_image_msg.header = image_msg->header;
  out_image_msg.image = overlay;
  out_image_msg.encoding = sensor_msgs::image_encodings::RGB8;
  pub_.publish(out_image_msg.toImageMsg());
}
}  // namespace mmrviz::visualizer

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mmrviz::visualizer::Segmentation2dVisualizer)
