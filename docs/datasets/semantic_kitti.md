# Semantic KITTI

Before running please download dataset from [Semantic KITTI Dataset](https://www.semantic-kitti.org/dataset.html#download).

Unpack the compressed file(s) into `mmros_dataloader/data/semantic_kitti` or other places and your folder structure should end up looking like this:

```shell
DATA_ROOT/
└── sequences
    ├── 00
    │   ├── calib.txt
    │   ├── labels
    │   ├── poses.txt
    │   ├── times.txt
    │   └── velodyne
    ├── 01
...
```

## Published Topics

Following topics are published with the timestamp at the corresponding timestamp preserved in `times.txt`.

| Topic                                   |             Type              | Description                                                               |
| :-------------------------------------- | :---------------------------: | :------------------------------------------------------------------------ |
| `/semantic_kitti/pointcloud/xyzi`       | `sensor_msgs/msg/PointCloud2` | Pointcloud which includes x, y, z and intensity.                          |
| `/semantic_kitti/localization/odometry` |    `nav_msgs/msg/Odometry`    | Ego vehicle odometry.                                                     |
| `/tf`                                   |   `tf2_msgs/msg/TFMessage`    | Transform of the ego vehicle.                                             |
| `/tf_static`                            |   `tf2_msgs/msg/TFMessage`    | Static transform of each sensor.                                          |
| `/semantic_kitti/pointcloud/xyzil`      | `sensor_msgs/msg/PointCloud2` | Pointcloud with labels, which is published if `publish_annotation:=true`. |

## Parameters

| Name                 |  Type   | Description                                                  |
| :------------------- | :-----: | :----------------------------------------------------------- |
| `sequence`           |  `str`  | The number of sequence in the range of `[00, 21]`.           |
| `data_root`          |  `str`  | Root directory path of the Semantic KITTI dataset.           |
| `timer_period_sec`   | `float` | The period of the timer in [s]. (Default: `0.1`)             |
| `publish_annotation` | `bool`  | Indicates whether to publish annotations. (Default: `false`) |

## How to Run

First of all, please download Semantic KITTI dataset from [OFFICIAL WEBSITE](https://www.semantic-kitti.org/index.html).

Note that we assume that downloaded dataset is placed under `mmros_dataloader/data` by default.
If you want to specify an another directory, use `data_root:=<DATASET_ROOT>` option.

Start running the Semantic KITTI publisher as follows:

```shell
ros2 launch mmros_dataloader semantic_kitti.launch.xml data_root:=<SEMANTIC_KITTI_ROOT>
```

### Publish Annotation PointCloud

If you want to publish annotation pointcloud, please specify `publish_annotation:=true`:

```shell
ros2 launch mmros_dataloader semantic_kitti.launch.xml publish_annotation:=true
```
