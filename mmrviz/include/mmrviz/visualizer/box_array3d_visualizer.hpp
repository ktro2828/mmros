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

#ifndef MMRVIZ__VISUALIZER__BOX_ARRAY3D_VISUALIZER_HPP_
#define MMRVIZ__VISUALIZER__BOX_ARRAY3D_VISUALIZER_HPP_

#include "mmrviz/color_map.hpp"

#include <image_transport/publisher.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/timer.hpp>

#include <mmros_msgs/msg/box_array3d.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <image_geometry/pinhole_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <memory>
#include <optional>
#include <string>

namespace mmrviz::visualizer
{
/**
 * @brief Visualizer class to render 3D boxes on image.
 */
class BoxArray3dVisualizer : public rclcpp::Node
{
public:
  /**
   * @brief Construct a new BoxArray3dVisualizer object.
   *
   * @param options Node options.
   */
  explicit BoxArray3dVisualizer(const rclcpp::NodeOptions & options);

  /**
   * @brief Render 3D boxes on the subscribed image.
   *
   * @param image_msg Image message.
   * @param camera_info_msg Camera info message.
   * @param boxes_msg 3D boxes message.
   */
  void callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg,
    const mmros_msgs::msg::BoxArray3d::ConstSharedPtr & boxes_msg);

private:
  /**
   * @brief Check connection and start subscribing topics.
   *
   * @param use_raw Indicates whether to use raw image.
   */
  void onConnect(bool use_raw);

  /**
   * @brief Return the QoS of the specified topic.
   *
   * @param query_topic Topic name.
   */
  std::optional<rclcpp::QoS> getTopicQos(const std::string & query_topic);

  rclcpp::TimerBase::SharedPtr timer_;           //!< Callback timer.
  image_transport::SubscriberFilter image_sub_;  //!< Image subscription.
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo>
    camera_info_sub_;  //!< Camera info subscription.
  message_filters::Subscriber<mmros_msgs::msg::BoxArray3d> boxes_sub_;  //!< 3D boxes subscription.

  using ApproximateTimeSyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo, mmros_msgs::msg::BoxArray3d>;
  using ApproximateTimeSync = message_filters::Synchronizer<ApproximateTimeSyncPolicy>;
  std::shared_ptr<ApproximateTimeSync> sync_ptr_;  //!< Pointer to messages synchronizer.

  tf2_ros::Buffer tf_buffer_;                        //!< Transform buffer.
  tf2_ros::TransformListener tf_listener_;           //!< Transform listener.
  image_geometry::PinholeCameraModel camera_model_;  //!< Camera model.

  image_transport::Publisher pub_;  //!< Image publisher.

  ColorMap color_map_;  //!< Color map.
};
}  // namespace mmrviz::visualizer
#endif  // MMRVIZ__VISUALIZER__BOX_ARRAY3D_VISUALIZER_HPP_
