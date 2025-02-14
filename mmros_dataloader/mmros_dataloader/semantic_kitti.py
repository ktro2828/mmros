# cspell: ignore velodyne

from __future__ import annotations

import os.path as osp
from glob import glob

import numpy as np
import rclpy
import tf_transformations as tf
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from numpy.typing import NDArray
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster


class SemanticKittiPublisher(Node):
    TOPIC_NAMESPACE = "/semantic_kitti"

    EGO_FRAME_ID = "base_link"
    WORLD_FRAME_ID = "map"

    def __init__(self) -> None:
        super().__init__("semantic_kitti")

        # parameters
        descriptor = ParameterDescriptor(dynamic_typing=True)

        sequence: str = str(
            self.declare_parameter("sequence", descriptor=descriptor)
            .get_parameter_value()
            .integer_value
        ).zfill(2)

        if sequence not in [str(i).zfill(2) for i in range(22)]:
            raise ValueError(f"sequence must be in the range of [00, 21], but got {sequence}")

        data_root: str = (
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

        self._timestamps = self._load_timestamps(data_root=data_root, sequence=sequence)
        self._poses = self._load_poses(data_root=data_root, sequence=sequence)
        self._velodyne_paths = self._load_velodyne_paths(data_root=data_root, sequence=sequence)

        if len(self._timestamps) != len(self._poses) or len(self._timestamps) != len(
            self._velodyne_paths
        ):
            raise ValueError("The number of items is must be same.")

        if self._publish_annotation:
            self._label_paths = self._load_label_paths(data_root=data_root, sequence=sequence)
            if len(self._timestamps) != len(self._label_paths):
                raise ValueError("The number of items is must be same.")
        else:
            self._label_paths = None

        self._current_frame_index = 0
        self._num_sequence = len(self._timestamps)

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        if self._publish_annotation:
            self._point_xyzil_pub = self.create_publisher(
                PointCloud2,
                osp.join(self.TOPIC_NAMESPACE, "pointcloud/xyzil"),
                qos_profile=qos_profile,
            )

        self._point_xyzi_pub = self.create_publisher(
            PointCloud2,
            osp.join(self.TOPIC_NAMESPACE, "pointcloud/xyzi"),
            qos_profile=qos_profile,
        )

        self._odometry_pub = self.create_publisher(
            Odometry,
            osp.join(self.TOPIC_NAMESPACE, "localization/odometry"),
            qos_profile=qos_profile,
        )

        # tf
        self._tf_broadcaster = TransformBroadcaster(self)

        self._timer = self.create_timer(
            timer_period_sec=self._timer_period_sec,
            callback=self.callback,
        )

    @staticmethod
    def _load_timestamps(data_root: str, sequence: str) -> list[float]:
        timestamps_path = osp.join(data_root, "sequences", sequence, "times.txt")

        timestamps_list = []
        with open(timestamps_path, encoding="utf-8") as timestamps_file:
            for line in timestamps_file:
                number = float(line)
                timestamps_list.append(number)
        return timestamps_list

    @staticmethod
    def _load_poses(data_root: str, sequence: str) -> list[NDArray]:
        poses_path = osp.join(data_root, "sequences", sequence, "poses.txt")

        calibration = np.array(
            [
                [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02],
                [-7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02],
                [9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01],
                [0, 0, 0, 1],
            ]
        )
        calibration_inv = np.linalg.inv(calibration)

        poses_list = []
        with open(poses_path, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                pose = np.fromstring(line, dtype=np.float64, sep=" ")
                pose = pose.reshape(3, 4)
                pose = np.vstack((pose, [0, 0, 0, 1]))
                poses_list.append(np.matmul(calibration_inv, np.matmul(pose, calibration)))

        return poses_list

    @staticmethod
    def _load_velodyne_paths(data_root: str, sequence: str) -> list[str]:
        return sorted(glob(osp.join(data_root, "sequences", sequence, "velodyne/*.bin")))

    @staticmethod
    def _load_label_paths(data_root: str, sequence: str) -> list[str]:
        return sorted(glob(osp.join(data_root, "sequences", sequence, "labels/*.label")))

    def callback(self) -> None:
        timestamp = self._timestamps[self._current_frame_index]

        pose = self._poses[self._current_frame_index]
        self._publish_poses(pose, timestamp)

        velodyne_path = self._velodyne_paths[self._current_frame_index]
        self._publish_point_xyzi(velodyne_path, timestamp)

        if self._publish_annotation:
            label_path = self._label_paths[self._current_frame_index]
            self._publish_point_xyzil(velodyne_path, label_path, timestamp)

        self._current_frame_index += 1

        if self._current_frame_index >= self._num_sequence:
            self._timer.cancel()
            self.get_logger().info("Timer callback has been canceled.")

            self.destroy_node()
            rclpy.shutdown()

    def _publish_poses(self, pose: NDArray, timestamp: float) -> None:
        # odometry
        odometry_msg = Odometry()
        odometry_msg.header.frame_id = self.WORLD_FRAME_ID
        odometry_msg.header.stamp.sec = int(timestamp)
        odometry_msg.header.stamp.nanosec = int((timestamp - int(timestamp)) * 1e9)

        odometry_msg.pose.pose.position.x = float(pose[0][3])
        odometry_msg.pose.pose.position.y = float(pose[1][3])
        odometry_msg.pose.pose.position.z = float(pose[2][3])

        q = tf.quaternion_from_matrix(pose)
        odometry_msg.pose.pose.orientation.x = float(q[0])
        odometry_msg.pose.pose.orientation.y = float(q[1])
        odometry_msg.pose.pose.orientation.z = float(q[2])
        odometry_msg.pose.pose.orientation.w = float(q[3])

        self._odometry_pub.publish(odometry_msg)

        # tf
        tf_msg = TransformStamped()
        tf_msg.header.frame_id = self.WORLD_FRAME_ID
        tf_msg.header.stamp.sec = int(timestamp)
        tf_msg.header.stamp.nanosec = int((timestamp - int(timestamp)) * 1e9)
        tf_msg.child_frame_id = self.EGO_FRAME_ID

        tf_msg.transform.translation.x = float(pose[0][3])
        tf_msg.transform.translation.y = float(pose[1][3])
        tf_msg.transform.translation.z = float(pose[2][3])

        tf_msg.transform.rotation.x = float(q[0])
        tf_msg.transform.rotation.y = float(q[1])
        tf_msg.transform.rotation.z = float(q[2])
        tf_msg.transform.rotation.w = float(q[3])

        self._tf_broadcaster.sendTransform(tf_msg)

    def _publish_point_xyzi(self, velodyne_path: str, timestamp: float) -> None:
        velodyne_data = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)

        header = Header()
        header.frame_id = self.EGO_FRAME_ID
        header.stamp.sec = int(timestamp)
        header.stamp.nanosec = int((timestamp - int(timestamp)) * 1e9)

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        point_cloud2_msg = point_cloud2.create_cloud(header, fields, velodyne_data)

        self._point_xyzi_pub.publish(point_cloud2_msg)

    def _publish_point_xyzil(self, velodyne_path: str, label_path: str, timestamp: float) -> None:
        velodyne_data = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
        label_data = np.fromfile(label_path, dtype=np.uint32).reshape(-1, 1)
        label_data = (label_data & 0xFFFF).astype(np.uint16)
        points = np.empty(
            len(velodyne_data),
            dtype=np.dtype(
                [
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("intensity", np.float32),
                    ("label", np.uint16),
                ]
            ),
        )
        points["x"] = velodyne_data[:, 0]
        points["y"] = velodyne_data[:, 1]
        points["z"] = velodyne_data[:, 2]
        points["intensity"] = velodyne_data[:, 3]
        points["label"] = label_data[:, 0]

        header = Header()
        header.frame_id = self.EGO_FRAME_ID
        header.stamp.sec = int(timestamp)
        header.stamp.nanosec = int((timestamp - int(timestamp)) * 1e9)

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="label", offset=16, datatype=PointField.UINT16, count=1),
        ]
        point_cloud2_msg = point_cloud2.create_cloud(header, fields, points)

        self._point_xyzil_pub.publish(point_cloud2_msg)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = SemanticKittiPublisher()
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
