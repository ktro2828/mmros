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

#include "mmrviz/visualizer/box_array2d_visualizer.hpp"

#include <image_transport/image_transport.hpp>
#include <mmros/node/utility.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/qos.hpp>

#include <sensor_msgs/image_encodings.hpp>

#include <cv_bridge/cv_bridge.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mmrviz::visualizer
{
BoxArray2dVisualizer::BoxArray2dVisualizer(const rclcpp::NodeOptions & options)
: rclcpp::Node("box_array2d_visualizer", options)
{
  using std::chrono_literals::operator""ms;

  bool use_raw = declare_parameter<bool>("use_raw");
  timer_ = rclcpp::create_timer(
    this, get_clock(), 100ms, [this, use_raw]() { this->on_connect(use_raw); });

  pub_ = image_transport::create_publisher(this, "~/output/image");
}

void BoxArray2dVisualizer::on_connect(bool use_raw)
{
  const auto image_topic = mmros::node::resolve_topic_name(this, "~/input/image");
  const auto image_qos = use_raw ? mmros::node::to_topic_qos(this, image_topic)
                                 : mmros::node::to_topic_qos(this, image_topic + "/compressed");

  const auto boxes_topic = mmros::node::resolve_topic_name(this, "~/input/boxes");
  const auto boxes_qos = mmros::node::to_topic_qos(this, boxes_topic);

  if (image_qos && boxes_qos) {
    const auto transport = use_raw ? "raw" : "compressed";
    image_sub_.subscribe(this, image_topic, transport, image_qos->get_rmw_qos_profile());
    boxes_sub_.subscribe(this, boxes_topic, boxes_qos->get_rmw_qos_profile());
    sync_ptr_ = std::make_shared<ExactTimeSync>(ExactTimeSyncPolicy(10), image_sub_, boxes_sub_);
    sync_ptr_->registerCallback(&BoxArray2dVisualizer::callback, this);

    timer_->cancel();
  }
}

void BoxArray2dVisualizer::callback(
  const sensor_msgs::msg::Image::ConstSharedPtr image_msg,
  const mmros_msgs::msg::BoxArray2d::ConstSharedPtr boxes_msg)
{
  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::RGB8);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR_STREAM(get_logger(), "cv_bridge exception: " << e.what());
    return;
  }

  auto & image = in_image_ptr->image;
  for (const auto & box : boxes_msg->boxes) {
    const auto xmin = std::max(0, static_cast<int>(box.x_offset));
    const auto ymin = std::max(0, static_cast<int>(box.y_offset));
    const auto xmax = std::min(static_cast<int>(box.x_offset + box.width), image.cols);
    const auto ymax = std::min(static_cast<int>(box.y_offset + box.height), image.rows);

    if (xmin >= xmax || ymin >= ymax) {
      continue;
    }

    const auto color = color_map_(box.label);

    cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), color, 3);
    cv::putText(
      image, cv::format("%.1f%%", box.score * 100.0), cv::Point(xmin, ymin - 5),
      cv::FONT_HERSHEY_COMPLEX_SMALL,
      1,  // font scale
      color,
      1,  // thickness
      cv::LINE_AA);
  }

  cv_bridge::CvImage out_image_msg;
  out_image_msg.header = image_msg->header;
  out_image_msg.image = image;
  out_image_msg.encoding = sensor_msgs::image_encodings::RGB8;
  pub_.publish(out_image_msg.toImageMsg());
}
}  // namespace mmrviz::visualizer

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mmrviz::visualizer::BoxArray2dVisualizer)
