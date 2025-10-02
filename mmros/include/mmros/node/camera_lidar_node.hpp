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

#ifndef MMROS__NODE__CAMERA_LIDAR_NODE_HPP_
#define MMROS__NODE__CAMERA_LIDAR_NODE_HPP_

#include <image_transport/subscriber.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <functional>
#include <string>
#include <vector>

namespace mmros::node
{
/**
 * @brief CameraLidarNode class for subscribing to multiple image topics and pointcloud topic.
 */
class CameraLidarNode : public rclcpp::Node
{
public:
  using PointCloudCallback = std::function<void(sensor_msgs::msg::PointCloud2::ConstSharedPtr)>;
  using ImageCallback = std::function<void(sensor_msgs::msg::Image::ConstSharedPtr, size_t)>;

  /**
   * @brief Constructor for CameraLidarNode.
   *
   * @param name Name of the node.
   * @param options Options for the node.
   */
  CameraLidarNode(const std::string & name, const rclcpp::NodeOptions & options);

protected:
  /**
   * @brief Connects the node to the pointcloud and image topics.
   *
   * @param pointcloud_callback Callback function for pointcloud messages.
   * @param image_topics Vector of image topic names.
   * @param image_callback Callback function for image messages.
   * @param use_raw Whether to use raw image data.
   */
  void on_connect(
    const PointCloudCallback & pointcloud_callback, const std::vector<std::string> & image_topics,
    const ImageCallback & image_callback, bool use_raw);

  rclcpp::TimerBase::SharedPtr connection_timer_;  //!< Topic connection timer.

private:
  /**
   * @brief Connect to a single pointcloud topic.
   *
   * @param callback Callback function to be called when a new pointcloud is received.
   * @return True if the connection was successful, false otherwise.
   */
  bool on_connect_lidar(const PointCloudCallback & pointcloud_callback);

  /**
   * @brief Connect to a single image topic.
   *
   * @param camera_id Camera ID.
   * @param image_topic Image topic name.
   * @param image_callback Callback function to be called when a new image is received.
   * @param use_raw Whether to use raw images or not.
   * @return True if the connection was successful, false otherwise.
   */
  bool on_connect_for_single_camera(
    size_t camera_id, const std::string & image_topic, const ImageCallback & image_callback,
    bool use_raw);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
    pointcloud_subscription_;  //!< Subscriber for pointcloud topic
  std::vector<image_transport::Subscriber> image_subscriptions_;  //!< Subscribers for image topics
};
}  // namespace mmros::node
#endif  // MMROS__NODE__CAMERA_LIDAR_NODE_HPP_
