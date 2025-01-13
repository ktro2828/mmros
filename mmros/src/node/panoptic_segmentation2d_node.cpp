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

#include "mmros/node/panoptic_segmentation2d_node.hpp"

#include "mmros/detector/panoptic_segmenter2d.hpp"
#include "mmros/node/semantic_segmentation2d_node.hpp"

#include <image_transport/image_transport.hpp>
#include <opencv2/core/mat.hpp>
#include <rclcpp/qos.hpp>

#include <sensor_msgs/image_encodings.hpp>

#include <cv_bridge/cv_bridge.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mmros
{
PanopticSegmentation2dNode::PanopticSegmentation2dNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("panoptic_segmentation2d", options)
{
  {
    auto onnx_path = declare_parameter<std::string>("onnx_path");
    TrtCommonConfig trt_config(onnx_path);

    auto score_threshold = declare_parameter<double>("detector_config.score_threshold");
    PanopticSegmenter2dConfig detector_config{score_threshold};
    detector_ = std::make_unique<PanopticSegmenter2D>(trt_config, detector_config);
  }

  {
    // TODO(ktro2828): Subscribe and publish for multiple images.
    using std::chrono_literals::operator""ms;

    bool use_raw = declare_parameter<bool>("use_raw");
    timer_ = rclcpp::create_timer(
      this, get_clock(), 100ms, [this, use_raw]() { this->onConnect(use_raw); });

    pub_box_ = create_publisher<mmros_msgs::msg::BoxArray2d>("~/output/boxes", 1);
    pub_mask_ = create_publisher<sensor_msgs::msg::Image>("~/output/mask", 1);
  }

  if (declare_parameter<bool>("build_only")) {
    RCLCPP_INFO(get_logger(), "TensorRT engine file is built and exit.");
    rclcpp::shutdown();
  }
}

void PanopticSegmentation2dNode::onConnect(bool use_raw)
{
  using std::placeholders::_1;

  auto resolve_topic_name = [this](const std::string & query) {
    return this->get_node_topics_interface()->resolve_topic_name(query);
  };

  const auto image_topic = resolve_topic_name("~/input/image");
  auto image_topic_for_qos_query = image_topic;
  if (!use_raw) {
    image_topic_for_qos_query += "/compressed";
  }
  const auto image_qos = getTopicQos(image_topic_for_qos_query);
  if (image_qos) {
    const auto transport = use_raw ? "raw" : "compressed";
    sub_ = image_transport::create_subscription(
      this, image_topic, std::bind(&PanopticSegmentation2dNode::onImage, this, _1), transport,
      image_qos->get_rmw_qos_profile());

    timer_->cancel();
  }
}

std::optional<rclcpp::QoS> PanopticSegmentation2dNode::getTopicQos(const std::string & query_topic)
{
  const auto publisher_info = get_publishers_info_by_topic(query_topic);
  if (publisher_info.size() != 1) {
    return {};
  } else {
    return publisher_info[0].qos_profile();
  }
}

void PanopticSegmentation2dNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  // TODO(ktro2828): Implementation
  RCLCPP_INFO(get_logger(), "Subscribe input!!");

  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR_STREAM(get_logger(), "cv_bridge exception: " << e.what());
    return;
  }

  std::vector<std::pair<Boxes2D, cv::Mat>> batch_outputs;
  try {
    std::vector<cv::Mat> images{in_image_ptr->image};
    batch_outputs = detector_->doInference(images).unwrap();
  } catch (const InferenceException & e) {
    RCLCPP_ERROR_STREAM(get_logger(), e.what());
    return;
  }

  // TODO(ktro2828): Publish outputs for each image.
  for (const auto & [boxes, mask] : batch_outputs) {
    mmros_msgs::msg::BoxArray2d output_box_msg;
    output_box_msg.header = msg->header;
    for (const auto & box : boxes) {
      mmros_msgs::msg::Box2d box_msg;

      box_msg.x_offset = box.xmin;
      box_msg.y_offset = box.ymin;
      box_msg.width = box.xmax - box.xmin;
      box_msg.height = box.ymax - box.ymin;
      box_msg.score = box.score;
      box_msg.label = box.label;

      output_box_msg.boxes.emplace_back(box_msg);
    }
    pub_box_->publish(output_box_msg);

    sensor_msgs::msg::Image::ConstSharedPtr output_mask_msg =
      cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::MONO8, mask).toImageMsg();
    pub_mask_->publish(*output_mask_msg);
  }
}
}  // namespace mmros

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mmros::PanopticSegmentation2dNode)
