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

#include "mmrviz/visualizer/instance_segmentation2d_visualizer.hpp"

#include "mmrviz/color_map.hpp"

#include <image_transport/image_transport.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/create_timer.hpp>

#include <sensor_msgs/image_encodings.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/hal/interface.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>

namespace mmrviz::visualizer
{
InstanceSegmentation2dVisualizer::InstanceSegmentation2dVisualizer(
  const rclcpp::NodeOptions & options)
: rclcpp::Node("instance_segmentation2d_visualizer", options)
{
  using std::chrono_literals::operator""ms;

  mask_threshold_ = declare_parameter<double>("mask_threshold", 0.8);

  bool use_raw = declare_parameter<bool>("use_raw");
  timer_ =
    rclcpp::create_timer(this, get_clock(), 100ms, [this, use_raw]() { this->onConnect(use_raw); });

  pub_ = image_transport::create_publisher(this, "~/output/image");
}

void InstanceSegmentation2dVisualizer::onConnect(bool use_raw)
{
  auto resolve_topic_name = [this](const std::string & query) {
    return this->get_node_topics_interface()->resolve_topic_name(query);
  };

  const auto image_topic = resolve_topic_name("~/input/image");
  const auto image_qos =
    use_raw ? getTopicQos(image_topic) : getTopicQos(image_topic + "/compressed");

  const auto segments_topic = resolve_topic_name("~/input/segments");
  const auto segments_qos = getTopicQos(segments_topic);

  if (image_qos && segments_qos) {
    const auto transport = use_raw ? "raw" : "compressed";
    image_sub_.subscribe(this, image_topic, transport, image_qos->get_rmw_qos_profile());
    segments_sub_.subscribe(this, segments_topic, segments_qos->get_rmw_qos_profile());

    sync_ptr_ = std::make_shared<ExactTimeSync>(ExactTimeSyncPolicy(10), image_sub_, segments_sub_);
    sync_ptr_->registerCallback(&InstanceSegmentation2dVisualizer::callback, this);

    timer_->cancel();
  }
}

std::optional<rclcpp::QoS> InstanceSegmentation2dVisualizer::getTopicQos(
  const std::string & query_topic)
{
  const auto publishers_info = get_publishers_info_by_topic(query_topic);
  if (publishers_info.size() != 1) {
    return std::nullopt;
  } else {
    return publishers_info[0].qos_profile();
  }
}

void InstanceSegmentation2dVisualizer::callback(
  const image_type::ConstSharedPtr & image_msg, const segment_type::ConstSharedPtr & segments_msg)
{
  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::RGB8);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR_STREAM(get_logger(), "cv_bridge exception: " << e.what());
    return;
  }

  const auto & lut = color_map_.getLookUpTable();
  auto & image = in_image_ptr->image;
  for (size_t i = 0; i < segments_msg->segments.size(); ++i) {
    const auto & box = segments_msg->segments[i].box;
    const auto xmin = std::max(0, static_cast<int>(box.x_offset));
    const auto ymin = std::max(0, static_cast<int>(box.y_offset));
    const auto xmax = std::min(static_cast<int>(box.x_offset + box.width), image.cols);
    const auto ymax = std::min(static_cast<int>(box.y_offset + box.height), image.rows);

    if (xmin >= xmax || ymin >= ymax) {
      continue;
    }

    const auto color = color_map_(box.label);

    // Draw box
    cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), color, 3);
    cv::putText(
      image, cv::format("%.1f%%", box.score * 100.0), cv::Point(xmin, ymin - 5),
      cv::FONT_HERSHEY_COMPLEX_SMALL,
      1,  // font scale
      color,
      1,  // thickness
      cv::LINE_AA);

    // Draw mask
    cv_bridge::CvImagePtr in_mask_ptr;
    try {
      in_mask_ptr =
        cv_bridge::toCvCopy(segments_msg->segments[i].mask, sensor_msgs::image_encodings::MONO8);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_STREAM(get_logger(), "cv_bridge exception: " << e.what());
      return;
    }

    // TODO(ktro2828): Improve performance
    auto process_pixel = [this, &color, &in_mask_ptr](
                           cv::Vec3b & pixel, const int * position) -> void {
      int i = position[0];
      int j = position[1];
      if (in_mask_ptr->image.at<uchar>(i, j) > static_cast<int>(this->mask_threshold_ * 255)) {
        cv::Vec3b color_weighted(
          color[0] * 0.5 + pixel[0] * 0.5, color[1] * 0.5 + pixel[2] * 0.5,
          color[2] * 0.5 + pixel[2] * 0.5);
        pixel = color_weighted;
      }
    };

    image.forEach<cv::Vec3b>(process_pixel);
  }

  cv_bridge::CvImage out_image_msg;
  out_image_msg.header = image_msg->header;
  out_image_msg.image = image;
  out_image_msg.encoding = sensor_msgs::image_encodings::RGB8;
  pub_.publish(out_image_msg.toImageMsg());
}
}  // namespace mmrviz::visualizer

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mmrviz::visualizer::InstanceSegmentation2dVisualizer)
