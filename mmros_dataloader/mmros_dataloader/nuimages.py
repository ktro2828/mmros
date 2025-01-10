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
from nuimages import NuImages
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from mmros_dataloader.color import ColorMap
from mmros_msgs.msg import Box2d, BoxArray2d


class NuImagesPublisher(Node):
    """Publisher of NuImages."""

    TOPIC_NAMESPACE = "/nuimages"

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
        super().__init__("nuimages")

        # cv bridge
        self._cv_bridge = CvBridge()

        # color map
        self._color_map = ColorMap()

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

        self._publish_annotation: bool = (
            self.declare_parameter("publish_annotation", descriptor=descriptor)
            .get_parameter_value()
            .bool_value
        )

        # nuimages
        self._nuim = NuImages(self._version, self._data_root, verbose=False)
        self._current_sample_idx = 0
        self._current_sample_token = self._nuim.sample[self._current_sample_idx]["token"]

        # publishers: {channel: publisher}
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._image_pubs: dict[str, Publisher] = {}
        self._cam_info_pubs: dict[str, Publisher] = {}
        if self._publish_annotation:
            self._boxes_pubs: dict[str, Publisher] = {}
            self._semantic_mask_pubs: dict[str, Publisher] = {}
            self._instance_mask_pubs: dict[str, Publisher] = {}
        for sensor_record in self._nuim.sensor:
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
                # annotation
                if self._publish_annotation:
                    # boxes
                    boxes_topic = osp.join(self.TOPIC_NAMESPACE, channel, "annotation", "boxes")
                    self._boxes_pubs[channel] = self.create_publisher(
                        BoxArray2d,
                        boxes_topic,
                        qos_profile=qos_profile,
                    )
                    # semseg mask
                    semantic_mask_topic = osp.join(
                        self.TOPIC_NAMESPACE, channel, "annotation", "semantic_mask"
                    )
                    self._semantic_mask_pubs[channel] = self.create_publisher(
                        Image,
                        semantic_mask_topic,
                        qos_profile=qos_profile,
                    )
                    # instanceseg mask
                    instance_mask_topic = osp.join(
                        self.TOPIC_NAMESPACE, channel, "annotation", "instance_mask"
                    )
                    self._instance_mask_pubs[channel] = self.create_publisher(
                        Image,
                        instance_mask_topic,
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
        for cs_record in self._nuim.calibrated_sensor:
            sensor_record = self._nuim.get("sensor", cs_record["sensor_token"])

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
            is_static (bool, optional): Indicates whether this transform is static or not. Defaults to False.
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
        sd_tokens = self._nuim.get_sample_content(self._current_sample_token)
        is_all_sample_data_end = False
        for sd_token in sd_tokens:
            stamp = self.get_clock().now().to_msg()
            sample_data = self._nuim.get("sample_data", sd_token)
            self._broadcast_ego_pose(sample_data["ego_pose_token"], stamp=stamp)
            sensor_record = self._nuim.shortcut("sample_data", "sensor", sd_token)
            channel: str = sensor_record["channel"]
            if "camera" == sensor_record["modality"]:
                self._publish_camera(sample_data, channel, stamp=stamp)

            if self._publish_annotation:
                object_anns = [
                    ann for ann in self._nuim.object_ann if ann["sample_data_token"] == sd_token
                ]
                self._publish_boxes(object_anns, channel, stamp=stamp)

                self._publish_mask(sample_data, channel, stamp=stamp)

            is_all_sample_data_end &= sample_data["next"] == ""

        if self._current_sample_idx < len(self._nuim.sample) - 1:
            self._current_sample_idx += 1
            self._current_sample_token = self._nuim.sample[self._current_sample_idx]["token"]
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
        ego_record = self._nuim.get("ego_pose", ego_pose_token)
        self._broadcast_tf(
            frame_id=self.WORLD_FRAME_ID,
            child_frame_id=self.EGO_FRAME_ID,
            translation=ego_record["translation"],
            rotation=ego_record["rotation"],
            stamp=stamp,
            is_static=False,
        )

    def _publish_camera(
        self, sample_data: dict[str, Any], channel: str, stamp: Time | None = None
    ) -> None:
        """Publish camera record.

        Args:
            sample_data (dict[str, Any]): Sample data record of camera.
            channel (str): Channel name.
            stamp (Time | None, optional): Timestamp to publish.
                If None, current time will be used. Defaults to None.
        """
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
        cs_record = self._nuim.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
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

        # distortion parameter (assumed 0)
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        self._cam_info_pubs[channel].publish(camera_info_msg)

    def _publish_boxes(
        self, annotations: Sequence[dict[str, Any]], channel: str, stamp: Time | None = None
    ) -> None:
        """Publish annotation boxes.

        Args:
            annotations (Sequence[str]): Sequence of object_ann records.
        """
        boxes_msg = BoxArray2d()
        boxes_msg.header.frame_id = channel
        boxes_msg.header.stamp = self.get_clock().now().to_msg() if stamp is None else stamp
        for ann in annotations:
            box_msg = Box2d()
            # roi
            xmin, ymin, xmax, ymax = ann["bbox"]
            box_msg.x_offset = int(xmin)
            box_msg.y_offset = int(ymin)
            box_msg.width = int(xmax - xmin)
            box_msg.height = int(ymax - ymin)
            # score
            box_msg.score = 1.0
            # label
            category_record = self._nuim.get("category", ann["category_token"])
            box_msg.label = int(self._to_label_id(category_record["name"]))

            boxes_msg.boxes.append(box_msg)

        self._boxes_pubs[channel].publish(boxes_msg)

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

    def _publish_mask(
        self, sample_data: dict[str, Any], channel: str, stamp: Time | None = None
    ) -> None:
        """Publish annotation mask.

        Args:
            sample_data (dict[str, Any]): Sample data record.
            channel (str): Channel name.
        """
        if not sample_data["is_key_frame"]:
            return

        semantic_mask, instance_mask = self._nuim.get_segmentation(sd_token=sample_data["token"])

        header = Header()
        header.frame_id = channel
        header.stamp = self.get_clock().now().to_msg() if stamp is None else stamp

        # semantic segmentation
        semantic_mask_msg = self._cv_bridge.cv2_to_imgmsg(
            semantic_mask.astype(np.uint8),
            encoding="mono8",
            header=header,
        )
        self._semantic_mask_pubs[channel].publish(semantic_mask_msg)

        # instance segmentation
        instance_mask_msg = self._cv_bridge.cv2_to_imgmsg(
            instance_mask.astype(np.uint8),
            encoding="mono8",
            header=header,
        )
        self._instance_mask_pubs[channel].publish(instance_mask_msg)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = NuImagesPublisher()
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
