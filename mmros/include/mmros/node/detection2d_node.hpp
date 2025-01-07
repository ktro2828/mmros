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

#ifndef MMROS__NODE__DETECTION2D_NODE_HPP_
#define MMROS__NODE__DETECTION2D_NODE_HPP_

#include "mmros/detector/detector2d.hpp"

#include <rclcpp/rclcpp.hpp>

#include <mmros_msgs/msg/box_array2d.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <memory>

namespace mmros
{
class Detection2dNode : public rclcpp::Node
{
public:
  explicit Detection2dNode(const rclcpp::NodeOptions & options);

  virtual void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);

private:
  std::unique_ptr<Detector2D> detector_;                           //!< TensorRT detector.
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;   //!< Input subscription.
  rclcpp::Publisher<mmros_msgs::msg::BoxArray2d>::SharedPtr pub_;  //!< Output publisher.
};
}  // namespace mmros
#endif  // MMROS__NODE__DETECTION2D_NODE_HPP_
