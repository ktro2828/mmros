# Semantic KITTI

Before running please download dataset from [Semantic KITTI Dataset](https://www.semantic-kitti.org/dataset.html#download).

Unpack the compressed file(s) into `mmros_dataloader/data/semantic_kitti` or other places and your folder structure should end up looking like this:

```shell
DATA_ROOT/
├── kitti
│   ├── dataset
│   |   └── sequences
|   │       ├── 00
│   │       │   ├── calib.txt
│   │       │   ├── labels
│   │       │   ├── poses.txt
│   │       │   ├── times.txt
│   │       │   └── velodyne
│   │       ├── 01
...
```

## Published Topics

Following topics are published with the timestamp at the corresponding timestamp preserved in `times.txt`.

| Topic                                   |             Type              | Description |
| :-------------------------------------- | :---------------------------: | :---------- |
| `/semantic_kitti/pointcloud/xyzi`       | `sensor_msgs/msg/PointCloud2` |             |
| `/semantic_kitti/localization/odometry` |    `nav_msgs/msg/Odometry`    |             |
| `/tf`                                   |   `tf2_msgs/msg/TFMessage`    |             |
| `/tf_static`                            |   `tf2_msgs/msg/TFMessage`    |             |
| `/semantic_kitti/pointcloud/xyzil`      | `sensor_msgs/msg/PointCloud2` |             |

## Parameters

| Name                 |  Type   | Description                                                  |
| :------------------- | :-----: | :----------------------------------------------------------- |
| `sequence`           |  `str`  | The number of sequence in the range of `[00, 21]`.           |
| `data_root`          |  `str`  | Root directory path of the Semantic KITTI dataset.           |
| `timer_period_sec`   | `float` | The period of the timer in [s]. (Default: `0.1`)             |
| `publish_annotation` | `bool`  | Indicates whether to publish annotations. (Default: `false`) |
