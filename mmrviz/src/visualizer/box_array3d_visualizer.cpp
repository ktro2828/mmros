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

#include "mmrviz/visualizer/box_array3d_visualizer.hpp"

#include <image_transport/image_transport.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/create_timer.hpp>
#include <rclcpp/qos.hpp>

#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <tf2/convert.h>
#include <tf2/exceptions.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mmrviz::visualizer
{
namespace
{
/**
 * @brief Convert quaternion to 3x3 rotation matrix.
 *
 * @param q Quaternion.
 */
cv::Mat quaternionToRotationMatrix(const geometry_msgs::msg::Quaternion & q)
{
  // qx, qy, qz, qw
  double x = q.x;
  double y = q.y;
  double z = q.z;
  double w = q.w;

  double xx = x * x;
  double yy = y * y;
  double zz = z * z;
  double xy = x * y;
  double xz = x * z;
  double yz = y * z;
  double wx = w * x;
  double wy = w * y;
  double wz = w * z;

  cv::Mat R =
    (cv::Mat_<double>(3, 3) << 1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
     2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 2.0 * (xz - wy), 2.0 * (yz + wx),
     1.0 - 2.0 * (xx + yy));
  return R;
}
/**
 * @brief Compute 3D box corners.
 *
 * @param pose 3D pose of box.
 * @param dimensions 3D dimensions of box.
 */
std::vector<cv::Point3d> computeBoxCorners(
  const geometry_msgs::msg::Pose & pose, const geometry_msgs::msg::Vector3 & dimensions)
{
  const auto & dx = dimensions.x;
  const auto & dy = dimensions.y;
  const auto & dz = dimensions.z;
  std::vector<cv::Point3d> corners{
    cv::Point3d(0.5 * dx, 0.5 * dy, 0.5 * dz),   cv::Point3d(0.5 * dx, 0.5 * dy, -0.5 * dz),
    cv::Point3d(0.5 * dx, -0.5 * dy, 0.5 * dz),  cv::Point3d(0.5 * dx, -0.5 * dy, -0.5 * dz),
    cv::Point3d(-0.5 * dx, 0.5 * dy, 0.5 * dz),  cv::Point3d(-0.5 * dx, 0.5 * dy, -0.5 * dz),
    cv::Point3d(-0.5 * dx, -0.5 * dy, 0.5 * dz), cv::Point3d(-0.5 * dx, -0.5 * dy, -0.5 * dz)};

  const auto rotation = quaternionToRotationMatrix(pose.orientation);
  const cv::Mat translation =
    (cv::Mat_<double>(3, 1) << pose.position.x, pose.position.y, pose.position.z);

  for (auto & xyz : corners) {
    cv::Mat p = (cv::Mat_<double>(3, 1) << xyz.x, xyz.y, xyz.z);
    cv::Mat rp = rotation * p + translation;
    xyz.x = rp.at<double>(0, 0);
    xyz.y = rp.at<double>(1, 0);
    xyz.z = rp.at<double>(2, 0);
  }

  return corners;
}

/**
 * @brief Project box corners on image.
 *
 * @param pose 3D pose of box.
 * @param dimensions 3D dimensions of box.
 * @param camera_model Camera model.
 */
std::vector<cv::Point2d> projectBoxCorners(
  const geometry_msgs::msg::Pose & pose, const geometry_msgs::msg::Vector3 & dimensions,
  const image_geometry::PinholeCameraModel & camera_model)
{
  const auto corners3d = computeBoxCorners(pose, dimensions);

  std::vector<cv::Point2d> corners2d;
  for (const auto & xyz : corners3d) {
    if (xyz.z <= 0) {
      continue;
    }
    auto xy = camera_model.project3dToPixel(xyz);
    corners2d.emplace_back(xy);
  }
  return corners2d;
}

/**
 * @brief Render box corners and edges on image.
 *
 * @param image Source image.
 * @param corners Project 2D box corners.
 * @param score Box score [0.0, 1.0].
 * @param color Edge color.
 * @param thickness Edge thickness.
 */
void renderBoxEdges(
  cv::Mat & image, const std::vector<cv::Point2d> & corners, double score, const cv::Scalar & color,
  int thickness = 3)
{
  if (corners.size() == 0) {
    return;
  }

  static std::array<std::pair<size_t, size_t>, 12> edges{
    std::make_pair(0, 1), std::make_pair(0, 2), std::make_pair(0, 4), std::make_pair(1, 3),
    std::make_pair(1, 5), std::make_pair(2, 3), std::make_pair(2, 6), std::make_pair(3, 7),
    std::make_pair(4, 5), std::make_pair(4, 6), std::make_pair(5, 7), std::make_pair(6, 7)};

  for (const auto & e : edges) {
    const auto & i0 = e.first;
    const auto & i1 = e.second;
    if (i0 < corners.size() && i1 < corners.size()) {
      cv::line(image, corners[i0], corners[i1], color, thickness);
    }
  }

  const auto min_ptr = std::min_element(
    corners.cbegin(), corners.cend(),
    [&](const auto & xy1, const auto & xy2) { return xy1.y < xy2.y; });

  cv::putText(
    image, cv::format("%.1f%%", score * 100.0), cv::Point(min_ptr->x, min_ptr->y - 5),
    cv::FONT_HERSHEY_COMPLEX_SMALL,
    1,  // font scale
    color,
    1,  // thickness
    cv::LINE_AA);
}
}  // namespace

BoxArray3dVisualizer::BoxArray3dVisualizer(const rclcpp::NodeOptions & options)
: rclcpp::Node("box_array3d_visualizer", options),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_)
{
  using std::chrono_literals::operator""ms;

  bool use_raw = declare_parameter<bool>("use_raw");
  timer_ =
    rclcpp::create_timer(this, get_clock(), 100ms, [this, use_raw]() { this->onConnect(use_raw); });

  pub_ = image_transport::create_publisher(this, "~/output/image");
}

void BoxArray3dVisualizer::onConnect(bool use_raw)
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

  const auto camera_info_topic = resolve_topic_name("~/input/camera_info");
  const auto camera_info_qos = getTopicQos(camera_info_topic);

  const auto boxes_topic = resolve_topic_name("~/input/boxes");
  const auto boxes_qos = getTopicQos(boxes_topic);

  bool is_image_ok = image_qos.has_value();
  bool is_camera_info_ok = camera_info_qos.has_value();
  bool is_boxes_ok = boxes_qos.has_value();

  if (image_qos && camera_info_qos && boxes_qos) {
    const auto transport = use_raw ? "raw" : "compressed";
    image_sub_.subscribe(this, image_topic, transport, image_qos->get_rmw_qos_profile());
    camera_info_sub_.subscribe(this, camera_info_topic, camera_info_qos->get_rmw_qos_profile());
    boxes_sub_.subscribe(this, boxes_topic, boxes_qos->get_rmw_qos_profile());

    sync_ptr_ = std::make_shared<ApproximateTimeSync>(
      ApproximateTimeSyncPolicy(10), image_sub_, camera_info_sub_, boxes_sub_);
    sync_ptr_->registerCallback(&BoxArray3dVisualizer::callback, this);

    timer_->cancel();
  }
}

std::optional<rclcpp::QoS> BoxArray3dVisualizer::getTopicQos(const std::string & query_topic)
{
  const auto publishers_info = get_publishers_info_by_topic(query_topic);
  if (publishers_info.size() != 1) {
    return std::nullopt;
  } else {
    return publishers_info[0].qos_profile();
  }
}

void BoxArray3dVisualizer::callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg,
  const mmros_msgs::msg::BoxArray3d::ConstSharedPtr & boxes_msg)
{
  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::RGB8);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR_STREAM(get_logger(), "cv_bridge exception: " << e.what());
    return;
  }

  camera_model_.fromCameraInfo(camera_info_msg);

  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_.lookupTransform(
      camera_info_msg->header.frame_id, boxes_msg->header.frame_id, camera_info_msg->header.stamp);
  } catch (const tf2::TransformException & e) {
    RCLCPP_WARN_STREAM(get_logger(), e.what());
    return;
  }

  auto & image = in_image_ptr->image;
  for (const auto & box : boxes_msg->boxes) {
    // Project box corners on image
    geometry_msgs::msg::Pose pose_in_camera;
    tf2::doTransform(box.pose, pose_in_camera, transform);
    const auto corners = projectBoxCorners(pose_in_camera, box.dimensions, camera_model_);

    // Render corners on image.
    const auto color = color_map_(box.label);
    renderBoxEdges(image, corners, box.score, color);
  }

  cv_bridge::CvImage out_image_msg;
  out_image_msg.header = image_msg->header;
  out_image_msg.image = image;
  out_image_msg.encoding = sensor_msgs::image_encodings::RGB8;
  pub_.publish(out_image_msg.toImageMsg());
}
}  // namespace mmrviz::visualizer

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mmrviz::visualizer::BoxArray3dVisualizer)
