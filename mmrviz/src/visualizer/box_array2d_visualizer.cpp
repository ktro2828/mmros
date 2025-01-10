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
  timer_ =
    rclcpp::create_timer(this, get_clock(), 100ms, [this, use_raw]() { this->onConnect(use_raw); });

  pub_ = image_transport::create_publisher(this, "~/output/image");
}

void BoxArray2dVisualizer::onConnect(bool use_raw)
{
  auto resolve_topic_name = [this](const std::string & query) {
    return this->get_node_topics_interface()->resolve_topic_name(query);
  };

  const auto image_topic = resolve_topic_name("~/input/image");
  auto image_topic_for_qos_query = image_topic;
  if (!use_raw) {
    image_topic_for_qos_query += "/compressed";
  }
  const auto image_qos = getTopicQos(image_topic_for_qos_query);

  const auto boxes_topic = resolve_topic_name("~/input/boxes");
  const auto boxes_qos = getTopicQos(boxes_topic);

  if (image_qos && boxes_qos) {
    const auto transport = use_raw ? "raw" : "compressed";
    image_sub_.subscribe(this, image_topic, transport, image_qos->get_rmw_qos_profile());
    boxes_sub_.subscribe(this, boxes_topic, boxes_qos->get_rmw_qos_profile());
    sync_ptr_ = std::make_shared<ExactTimeSync>(ExactTimeSyncPolicy(10), image_sub_, boxes_sub_);
    sync_ptr_->registerCallback(&BoxArray2dVisualizer::callback, this);

    timer_->cancel();
  }
}

std::optional<rclcpp::QoS> BoxArray2dVisualizer::getTopicQos(const std::string & query_topic)
{
  const auto publisher_info = get_publishers_info_by_topic(query_topic);
  if (publisher_info.size() != 1) {
    return {};
  } else {
    return publisher_info[0].qos_profile();
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
    const auto ymax = std::max(static_cast<int>(box.y_offset + box.height), image.rows);

    const auto color = color_map_(box.label);

    cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), color);
    cv::putText(
      image, cv::format("%s", std::to_string(box.score).c_str()), cv::Point(xmin, ymin - 5),
      cv::FONT_HERSHEY_SIMPLEX, 1,  // font scale
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
