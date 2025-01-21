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

#include "mmros/node/semantic_segmentation2d_node.hpp"

#include "mmros/detector/semantic_segmenter2d.hpp"

#include <image_transport/image_transport.hpp>
#include <opencv2/core/mat.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/image_encodings.hpp>

#include <cv_bridge/cv_bridge.h>

#include <optional>
#include <string>
#include <vector>

namespace mmros
{
SemanticSegmentation2dNode::SemanticSegmentation2dNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("semantic_segmentation2d", options)
{
  {
    auto onnx_path = declare_parameter<std::string>("tensorrt.onnx_path");
    auto precision = declare_parameter<std::string>("tensorrt.precision");
    TrtCommonConfig trt_config(onnx_path, precision);

    auto mean = declare_parameter<std::vector<double>>("detector.mean");
    auto std = declare_parameter<std::vector<double>>("detector.std");
    SemanticSegmenter2dConfig detector_config{mean, std};
    detector_ = std::make_unique<SemanticSegmenter2D>(trt_config, detector_config);
  }

  {
    // TODO(ktro2828): Subscribe and publish for multiple images.
    using std::chrono_literals::operator""ms;

    bool use_raw = declare_parameter<bool>("use_raw");
    timer_ = rclcpp::create_timer(
      this, get_clock(), 100ms, [this, use_raw]() { this->onConnect(use_raw); });

    pub_ = create_publisher<sensor_msgs::msg::Image>("~/output/mask", 1);
  }

  if (declare_parameter<bool>("build_only")) {
    RCLCPP_INFO(get_logger(), "TensorRT engine file is built and exit.");
    rclcpp::shutdown();
  }
}

void SemanticSegmentation2dNode::onConnect(bool use_raw)
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
      this, image_topic, std::bind(&SemanticSegmentation2dNode::onImage, this, _1), transport,
      image_qos->get_rmw_qos_profile());

    timer_->cancel();
  }
}

std::optional<rclcpp::QoS> SemanticSegmentation2dNode::getTopicQos(const std::string & query_topic)
{
  const auto publisher_info = get_publishers_info_by_topic(query_topic);
  if (publisher_info.size() != 1) {
    return {};
  } else {
    return publisher_info[0].qos_profile();
  }
}

void SemanticSegmentation2dNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR_STREAM(get_logger(), "cv_bridge exception: " << e.what());
    return;
  }

  std::vector<cv::Mat> batch_masks;
  try {
    std::vector<cv::Mat> images{in_image_ptr->image};
    batch_masks = detector_->doInference(images).unwrap();
  } catch (const InferenceException & e) {
    RCLCPP_ERROR_STREAM(get_logger(), e.what());
    return;
  }

  // TODO(ktro2828): Publish outputs for each image.
  for (const auto & mask : batch_masks) {
    sensor_msgs::msg::Image::ConstSharedPtr output_msg =
      cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::MONO8, mask).toImageMsg();
    pub_->publish(*output_msg);
  }
}
}  // namespace mmros

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mmros::SemanticSegmentation2dNode)
