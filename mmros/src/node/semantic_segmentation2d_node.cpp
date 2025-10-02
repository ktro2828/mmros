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

#include "mmros/archetype/exception.hpp"
#include "mmros/detector/semantic_segmenter2d.hpp"

#include <image_transport/image_transport.hpp>
#include <opencv2/core/mat.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/image_encodings.hpp>

#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mmros::node
{
SemanticSegmentation2dNode::SemanticSegmentation2dNode(const rclcpp::NodeOptions & options)
: SingleCameraNode("semantic_segmentation2d", options)
{
  {
    auto onnx_path = declare_parameter<std::string>("tensorrt.onnx_path");
    auto precision = declare_parameter<std::string>("tensorrt.precision");
    tensorrt::TrtCommonConfig trt_config(std::move(onnx_path), std::move(precision));

    auto mean = declare_parameter<std::vector<double>>("detector.mean");
    auto std = declare_parameter<std::vector<double>>("detector.std");
    detector::SemanticSegmenter2dConfig detector_config{std::move(mean), std::move(std)};
    detector_ = std::make_unique<detector::SemanticSegmenter2D>(
      std::move(trt_config), std::move(detector_config));
  }

  {
    using std::chrono_literals::operator""ms;
    using std::placeholders::_1;

    bool use_raw = declare_parameter<bool>("use_raw");
    connection_timer_ = rclcpp::create_timer(this, get_clock(), 100ms, [this, use_raw]() {
      this->onConnect(std::bind(&SemanticSegmentation2dNode::onImage, this, _1), use_raw);
    });

    pub_ = create_publisher<sensor_msgs::msg::Image>("~/output/mask", 1);
  }

  if (declare_parameter<bool>("build_only")) {
    RCLCPP_INFO(get_logger(), "TensorRT engine file is built and exit.");
    rclcpp::shutdown();
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
    batch_masks = detector_->do_inference(images).unwrap();
  } catch (const archetype::MmRosException & e) {
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
}  // namespace mmros::node

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mmros::node::SemanticSegmentation2dNode)
