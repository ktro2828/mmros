# Copyright 2025 Kotaro Uetake.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os.path as osp
from typing import Any, Sequence

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from mmros_msgs.msg import Box3d, BoxArray3d


class NuScenesPublisher(Node):
    """Publisher of NuScenes."""

    TOPIC_NAMESPACE = "/nuscenes"

    EGO_FRAME_ID = "base_link"
    WORLD_FRAME_ID = "map"

    CATEGORY_MAPPING = {
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.wheelchair": "pedestrian",
        "human.pedestrian.stroller": "pedestrian",
        "human.pedestrian.personal_mobility": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "animal": "animal",
        "vehicle.car": "car",
        "vehicle.motorcycle": "motorcycle",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.truck": "truck",
        "vehicle.construction": "construction_vehicle",
        "vehicle.emergency.ambulance": "car",
        "vehicle.emergency.police": "car",
        "vehicle.trailer": "trailer",
        "movable_object.barrier": "barrier",
        "movable_object.trafficcone": "traffic_cone",
        "movable_object.pushable_pullable": "unknown",
        "movable_object.debris": "unknown",
        "static_object.bicycle_rack": "unknown",
    }

    LABEL_NAMES = (
        "car",
        "motorcycle",
        "bicycle",
        "bus",
        "truck",
        "construction_vehicle",
        "trailer",
        "barrier",
        "traffic_cone",
        "pedestrian",
        "unknown",
    )

    def __init__(self) -> None:
        super().__init__("nuscenes")

        # cv bridge
        self._cv_bridge = CvBridge()

        # parameters
        descriptor = ParameterDescriptor(dynamic_typing=True)

        self._version: str = (
            self.declare_parameter("version", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )
        self._data_root: str = (
            self.declare_parameter("data_root", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )

        self._timer_period_sec: float = (
            self.declare_parameter("timer_period_sec", descriptor=descriptor)
            .get_parameter_value()
            .double_value
        )

        self._publish_all_scenes: bool = (
            self.declare_parameter("publish_all_scenes", descriptor=descriptor)
            .get_parameter_value()
            .bool_value
        )

        # nuscenes
        self._nusc = NuScenes(self._version, self._data_root, verbose=False)

        if self._publish_all_scenes:
            self._current_scene_idx = 0
            self._current_scene_token = self._nusc.scene[self._current_scene_idx]["token"]
        else:
            self._current_scene_token = (
                self.declare_parameter("scene_token", descriptor=descriptor)
                .get_parameter_value()
                .string_value
            )
            self._current_scene_idx = [s["token"] for s in self._nusc.scene].index(
                self._current_scene_token
            )
        self._current_sample_token = self._nusc.scene[self._current_scene_idx]["first_sample_token"]

        self._publish_annotation: bool = (
            self.declare_parameter("publish_annotation", descriptor=descriptor)
            .get_parameter_value()
            .bool_value
        )

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # annotation publisher
        if self._publish_annotation:
            ann_topic = osp.join(self.TOPIC_NAMESPACE, "annotation", "boxes")
            self._boxes_pub: Publisher = self.create_publisher(
                BoxArray3d,
                ann_topic,
                qos_profile=qos_profile,
            )

        # sensor data publishers: {channel: publisher}
        self._image_pubs: dict[str, Publisher] = {}
        self._cam_info_pubs: dict[str, Publisher] = {}
        self._pointcloud_pubs: dict[str, Publisher] = {}
        for sensor_record in self._nusc.sensor:
            channel: str = sensor_record["channel"]
            if "camera" == sensor_record["modality"]:
                # image
                image_topic = osp.join(self.TOPIC_NAMESPACE, channel, "image/compressed")
                self._image_pubs[channel] = self.create_publisher(
                    CompressedImage,
                    image_topic,
                    qos_profile=qos_profile,
                )
                # camera info
                cam_info_topic = osp.join(self.TOPIC_NAMESPACE, channel, "camera_info")
                self._cam_info_pubs[channel] = self.create_publisher(
                    CameraInfo,
                    cam_info_topic,
                    qos_profile=qos_profile,
                )
            elif "lidar" == sensor_record["modality"]:
                lidar_topic = osp.join(self.TOPIC_NAMESPACE, channel, "pointcloud")
                self._pointcloud_pubs[channel] = self.create_publisher(
                    PointCloud2,
                    lidar_topic,
                    qos_profile=qos_profile,
                )

        # tf
        self._tf_broadcaster = TransformBroadcaster(self)
        self._tf_static_broadcaster = StaticTransformBroadcaster(self)
        self._broadcast_tf_static()

        # callback
        self._timer = self.create_timer(
            timer_period_sec=self._timer_period_sec,
            callback=self.callback,
        )

    def _broadcast_tf_static(self) -> None:
        """Broadcast `/tf_static` for each sensor."""
        for cs_record in self._nusc.calibrated_sensor:
            sensor_record = self._nusc.get("sensor", cs_record["sensor_token"])

            self._broadcast_tf(
                frame_id=self.EGO_FRAME_ID,
                child_frame_id=sensor_record["channel"],
                translation=cs_record["translation"],
                rotation=cs_record["rotation"],
                is_static=True,
            )

    def _broadcast_tf(
        self,
        frame_id: str,
        child_frame_id: str,
        translation: Sequence[float],
        rotation: Sequence[float],
        stamp: Time | None = None,
        *,
        is_static: bool = False,
    ) -> None:
        """Broadcast transform.

        Args:
            frame_id (str): Frame ID.
            child_frame_id (str): Chile frame ID.
            translation (Sequence[float]): 3D translation (x, y, z).
            rotation (Sequence[float]): Quaternion (w, x, y, z).
            stamp (Time | None, optional): Timestamp to publish.
                If None, current time will be used. Defaults to None.
            is_static (bool, optional): Indicates whether this transform is static or not.
                Defaults to False.
        """
        t = TransformStamped()

        # header
        t.header.stamp = self.get_clock().now().to_msg() if stamp is None else stamp
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id

        # translation
        tx, ty, tz = translation
        t.transform.translation.x = float(tx)
        t.transform.translation.y = float(ty)
        t.transform.translation.z = float(tz)

        # rotation
        qw, qx, qy, qz = rotation
        t.transform.rotation.w = float(qw)
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)

        if is_static:
            self._tf_static_broadcaster.sendTransform(t)
        else:
            self._tf_broadcaster.sendTransform(t)

    def callback(self) -> None:
        """Timer callback."""
        sample = self._nusc.get("sample", self._current_sample_token)
        for _, sd_token in sample["data"].items():
            stamp = self.get_clock().now().to_msg()
            sample_data = self._nusc.get("sample_data", sd_token)
            self._broadcast_ego_pose(sample_data["ego_pose_token"], stamp=stamp)
            if "camera" == sample_data["sensor_modality"]:
                self._publish_camera(sample_data, stamp=stamp)
            elif "lidar" == sample_data["sensor_modality"]:
                self._publish_lidar(sample_data, stamp=stamp)

        if self._publish_annotation:
            stamp = self.get_clock().now().to_msg()
            self._publish_boxes(annotations=sample["anns"], stamp=stamp)

        if sample["next"] != "":
            self._current_sample_token = sample["next"]
        else:
            if self._publish_all_scenes and self._current_scene_idx < len(self._nusc.scene) - 1:
                self._current_scene_idx += 1
                self._current_scene_token = self._nusc.scene[self._current_scene_idx]["token"]
                self._current_sample_token = self._nusc.scene[self._current_scene_idx][
                    "first_sample_token"
                ]
            else:
                self._timer.cancel()
                self.get_logger().info("Timer callback has been canceled.")

                self.destroy_node()
                rclpy.shutdown()

    def _broadcast_ego_pose(self, ego_pose_token: str, stamp: Time | None = None) -> None:
        """Broadcast transform of the corresponding ego pose.

        Args:
            ego_pose_token (str): Token of ego pose.
            stamp (Time | None, optional): Timestamp to publish.
                If None, current time will be used. Defaults to None.
        """
        ego_record = self._nusc.get("ego_pose", ego_pose_token)
        self._broadcast_tf(
            frame_id=self.WORLD_FRAME_ID,
            child_frame_id=self.EGO_FRAME_ID,
            translation=ego_record["translation"],
            rotation=ego_record["rotation"],
            stamp=stamp,
            is_static=False,
        )

    def _publish_camera(self, sample_data: dict[str, Any], stamp: Time | None = None) -> None:
        """Publish camera record.

        Args:
            sample_data (dict[str, Any]): Sample data record of camera.
            stamp (Time | None, optional): Timestamp to publish.
                If None, current time will be used. Defaults to None.
        """
        channel: str = sample_data["channel"]
        header = Header()
        header.frame_id = channel
        header.stamp = self.get_clock().now().to_msg() if stamp is None else stamp

        # === image ===
        image_path = osp.join(self._data_root, sample_data["filename"])
        image = cv2.imread(image_path)
        image_msg: CompressedImage = self._cv_bridge.cv2_to_compressed_imgmsg(image)
        image_msg.header = header
        self._image_pubs[channel].publish(image_msg)

        # === camera info ===
        cs_record = self._nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])  # [3, 3]

        camera_info_msg = CameraInfo()
        camera_info_msg.header = header

        # image size
        camera_info_msg.width = sample_data["width"]
        camera_info_msg.height = sample_data["height"]

        # intrinsic camera matrix K: (3x3)
        # cam_intrinsic = [[fx,  0, cx],
        #                  [ 0, fy, cy],
        #                  [ 0,  0,  1]]
        camera_info_msg.k = [
            cam_intrinsic[0][0],
            cam_intrinsic[0][1],
            cam_intrinsic[0][2],
            cam_intrinsic[1][0],
            cam_intrinsic[1][1],
            cam_intrinsic[1][2],
            cam_intrinsic[2][0],
            cam_intrinsic[2][1],
            cam_intrinsic[2][2],
        ]
        # projection matrix K: (3x4)
        # [fx, 0, cx, 0; 0, fy, cy; 0, 0, 1, 0]
        camera_info_msg.p = [
            cam_intrinsic[0][0],
            cam_intrinsic[0][1],
            cam_intrinsic[0][2],
            0.0,
            cam_intrinsic[1][0],
            cam_intrinsic[1][1],
            cam_intrinsic[1][2],
            0.0,
            cam_intrinsic[2][0],
            cam_intrinsic[2][1],
            cam_intrinsic[2][2],
            0.0,
        ]

        # row-major matrix R: (3x3)
        camera_info_msg.r[0] = 1.0
        camera_info_msg.r[3] = 1.0
        camera_info_msg.r[6] = 1.0

        # distortion parameter (assumed 0)
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        self._cam_info_pubs[channel].publish(camera_info_msg)

    def _publish_lidar(self, sample_data: dict[str, Any], stamp: Time | None = None) -> None:
        """Publish lidar record.

        Args:
            sample_data (dict[str, Any]): Sample data record of lidar.
            stamp (Time | None, optional): Timestamp to publish.
                If None, current time will be used. Defaults to None.
        """
        channel: str = sample_data["channel"]

        header = Header()
        header.frame_id = channel
        header.stamp = self.get_clock().now().to_msg() if stamp is None else stamp

        pointcloud_path = osp.join(self._data_root, sample_data["filename"])

        pointcloud = LidarPointCloud.from_file(pointcloud_path)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        pc2_msg = point_cloud2.create_cloud(
            header=header, fields=fields, points=pointcloud.points.T
        )

        self._pointcloud_pubs[channel].publish(pc2_msg)

    def _publish_boxes(self, annotations: Sequence[str], stamp: Time | None = None) -> None:
        """Publish annotation boxes.

        Args:
            annotations (Sequence[str]): Sequence of annotation tokens.
            stamp (Time | None, optional): Timestamp to publish.
                If None, current time will be used. Defaults to None.
        """
        boxes_msg = BoxArray3d()
        boxes_msg.header.frame_id = self.WORLD_FRAME_ID
        boxes_msg.header.stamp = self.get_clock().now().to_msg() if stamp is None else stamp
        for token in annotations:
            ann = self._nusc.get("sample_annotation", token)

            box_msg = Box3d()
            # translation
            tx, ty, tz = ann["translation"]
            box_msg.pose.position.x = float(tx)
            box_msg.pose.position.y = float(ty)
            box_msg.pose.position.z = float(tz)
            # rotation
            qw, qx, qy, qz = ann["rotation"]
            box_msg.pose.orientation.w = float(qw)
            box_msg.pose.orientation.x = float(qx)
            box_msg.pose.orientation.y = float(qy)
            box_msg.pose.orientation.z = float(qz)
            # size
            width, length, height = ann["size"]
            box_msg.dimensions.x = float(length)
            box_msg.dimensions.y = float(width)
            box_msg.dimensions.z = float(height)
            # score
            box_msg.score = 1.0
            # label
            box_msg.label = int(self._to_label_id(ann["category_name"]))

            boxes_msg.boxes.append(box_msg)

        self._boxes_pub.publish(boxes_msg)

    def _to_label_id(self, category: str) -> int:
        """Convert category name to integer ID.

        Args:
            category (str): Name of category.

        Returns:
            int: Corresponding ID.
        """
        if category in self.CATEGORY_MAPPING:
            label = self.CATEGORY_MAPPING[category]
        else:
            self.get_logger().warning(f"{category} is unexpected, use unknown")
            label = "unknown"
        return self.LABEL_NAMES.index(label)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = NuScenesPublisher()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
