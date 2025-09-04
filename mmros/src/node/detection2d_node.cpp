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

#include "mmros/node/detection2d_node.hpp"

#include "mmros/archetype/box.hpp"
#include "mmros/archetype/exception.hpp"
#include "mmros/archetype/result.hpp"
#include "mmros/detector/detector2d.hpp"

#include <image_transport/image_transport.hpp>
#include <rclcpp/create_timer.hpp>
#include <rclcpp/qos.hpp>

#include <mmros_msgs/msg/detail/box_array2d__builder.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mmros::node
{
Detection2dNode::Detection2dNode(const rclcpp::NodeOptions & options)
: SingleCameraNode("detector2d", options)
{
  {
    auto onnx_path = declare_parameter<std::string>("tensorrt.onnx_path");
    auto precision = declare_parameter<std::string>("tensorrt.precision");
    tensorrt::TrtCommonConfig trt_config(onnx_path, precision);

    auto mean = declare_parameter<std::vector<double>>("detector.mean");
    auto std = declare_parameter<std::vector<double>>("detector.std");
    auto score_threshold = declare_parameter<float>("detector.score_threshold");
    auto box_format_str = declare_parameter<std::string>("detector.box_format");
    archetype::BoxFormat2D box_format = archetype::to_box_format2d(box_format_str);
    detector::Detector2dConfig detector_config{mean, std, box_format, score_threshold};
    detector_ = std::make_unique<detector::Detector2D>(trt_config, detector_config);
  }

  {
    using std::chrono_literals::operator""ms;
    using std::placeholders::_1;

    bool use_raw = declare_parameter<bool>("use_raw");
    connection_timer_ = rclcpp::create_timer(this, get_clock(), 100ms, [this, use_raw]() {
      this->onConnect(std::bind(&Detection2dNode::onImage, this, _1), use_raw);
    });

    pub_ = create_publisher<mmros_msgs::msg::BoxArray2d>("~/output/boxes", 1);
  }

  if (declare_parameter<bool>("build_only")) {
    RCLCPP_INFO(get_logger(), "TensorRT engine file is built and exit.");
    rclcpp::shutdown();
  }
}

void Detection2dNode::onImage(sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR_STREAM(get_logger(), "cv_bridge exception: " << e.what());
    return;
  }

  std::vector<archetype::Boxes2D> batch_boxes;
  try {
    std::vector<cv::Mat> images{in_image_ptr->image};
    batch_boxes = detector_->doInference(images).unwrap();
  } catch (const archetype::MmRosException & e) {
    RCLCPP_ERROR_STREAM(get_logger(), e.what());
    return;
  }

  // TODO(ktro2828): Publish outputs for each image.
  for (const auto & boxes : batch_boxes) {
    mmros_msgs::msg::BoxArray2d output_msg;
    output_msg.header = msg->header;
    for (const auto & box : boxes) {
      mmros_msgs::msg::Box2d box_msg;

      box_msg.x_offset = box.xmin;
      box_msg.y_offset = box.ymin;
      box_msg.width = box.xmax - box.xmin;
      box_msg.height = box.ymax - box.ymin;
      box_msg.score = box.score;
      box_msg.label = box.label;

      output_msg.boxes.emplace_back(box_msg);
    }
    pub_->publish(output_msg);
  }
}
}  // namespace mmros::node

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mmros::node::Detection2dNode)
