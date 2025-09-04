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

#include "mmros/node/instance_segmentation2d_node.hpp"

#include "mmros/archetype/box.hpp"
#include "mmros/archetype/exception.hpp"
#include "mmros/detector/instance_segmenter2d.hpp"
#include "mmros/node/single_camera_node.hpp"
#include "mmros/tensorrt/utility.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

#include <mmros_msgs/msg/box2d.hpp>
#include <mmros_msgs/msg/detail/instance_segment2d__struct.hpp>
#include <mmros_msgs/msg/instance_segment_array2d.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <cv_bridge/cv_bridge.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mmros::node
{
InstanceSegmentation2dNode::InstanceSegmentation2dNode(const rclcpp::NodeOptions & options)
: SingleCameraNode("instance_segmentation2d", options)
{
  {
    auto onnx_path = declare_parameter<std::string>("tensorrt.onnx_path");
    auto precision = declare_parameter<std::string>("tensorrt.precision");
    tensorrt::TrtCommonConfig trt_config(onnx_path, precision);

    auto mean = declare_parameter<std::vector<double>>("detector.mean");
    auto std = declare_parameter<std::vector<double>>("detector.std");
    auto score_threshold = declare_parameter<double>("detector.score_threshold");
    auto box_format_str = declare_parameter<std::string>("detector.box_format");
    archetype::BoxFormat2D box_format = archetype::to_box_format2d(box_format_str);
    detector::InstanceSegmenter2dConfig detector_config{mean, std, box_format, score_threshold};
    detector_ = std::make_unique<detector::InstanceSegmenter2D>(trt_config, detector_config);
  }

  {
    using std::chrono_literals::operator""ms;
    using std::placeholders::_1;

    bool use_raw = declare_parameter<bool>("use_raw");
    connection_timer_ = create_timer(this, get_clock(), 100ms, [this, use_raw]() {
      this->onConnect(std::bind(&InstanceSegmentation2dNode::onImage, this, _1), use_raw);
    });

    pub_segment_ =
      create_publisher<mmros_msgs::msg::InstanceSegmentArray2d>("~/output/segments", 1);
  }

  if (declare_parameter<bool>("build_only")) {
    RCLCPP_INFO(get_logger(), "TensorRT engine file is built and exit.");
    rclcpp::shutdown();
  }
}

void InstanceSegmentation2dNode::onImage(sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR_STREAM(get_logger(), "cv_bridge exception: " << e.what());
    return;
  }

  std::vector<std::pair<archetype::Boxes2D, std::vector<cv::Mat>>> batch_outputs;
  try {
    std::vector<cv::Mat> images{in_image_ptr->image};
    batch_outputs = detector_->doInference(images).unwrap();
  } catch (const archetype::MmRosException & e) {
    RCLCPP_ERROR_STREAM(get_logger(), e.what());
    return;
  }

  // TODO(ktro2828): Publish outputs for each image.
  for (const auto & [boxes, masks] : batch_outputs) {
    mmros_msgs::msg::InstanceSegmentArray2d segments_msg;
    segments_msg.header = msg->header;
    for (size_t i = 0; i < boxes.size(); ++i) {
      mmros_msgs::msg::InstanceSegment2d segment_msg;

      const auto & box = boxes.at(i);

      segment_msg.box.x_offset = box.xmin;
      segment_msg.box.y_offset = box.ymin;
      segment_msg.box.width = box.xmax - box.xmin;
      segment_msg.box.height = box.ymax - box.ymin;
      segment_msg.box.score = box.score;
      segment_msg.box.label = box.label;

      const auto & mask = masks.at(i);
      segment_msg.mask =
        *cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::MONO8, mask).toImageMsg();

      segments_msg.segments.emplace_back(segment_msg);
    }
    pub_segment_->publish(segments_msg);
  }
}
}  // namespace mmros::node

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mmros::node::InstanceSegmentation2dNode)
