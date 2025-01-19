# NuScenes

To install the nuScenes-lidarseg and/or Panoptic nuScenes expansion, download the dataset from <https://www.nuscenes.org/download>.

Unpack the compressed file(s) into `mmros_dataloader/data/nuscenes` or other places and your folder structure should end up looking like this:

```shell
DATA_ROOT/
├── Usual nuscenes folders (i.e. samples, sweep)
│
├── lidarseg
│   └── v1.0-{mini, test, trainval} <- Contains the .bin files; a .bin file
│                                      contains the labels of the points in a
│                                      point cloud (note that v1.0-test does not
│                                      have any .bin files associated with it)
│
├── panoptic
│   └── v1.0-{mini, test, trainval} <- Contains the *_panoptic.npz files; a .npz file
│                                      contains the panoptic labels of the points in a
│                                      point cloud (note that v1.0-test does not
│                                      have any .npz files associated with it)
└── v1.0-{mini, test, trainval}
    ├── Usual files (e.g. attribute.json, calibrated_sensor.json etc.)
    ├── lidarseg.json  <- contains the mapping of each .bin file to the token
    ├── panoptic.json  <- contains the mapping of each .npz file to the token
    └── category.json  <- contains the categories of the labels (note that the
                            category.json from nuScenes v1.0 is overwritten)
```

## Published Messages

Following topics are published with the timestamp at the current ROS 2 simulation time:

| Topic                                 |               Type                | Description                                                                                                       |
| :------------------------------------ | :-------------------------------: | :---------------------------------------------------------------------------------------------------------------- |
| `/nuscenes/<CAMERA>/image`            | `sensor_msgs/msg/CompressedImage` | Compressed image data of each camera.                                                                             |
| `/nuscenes/<CAMERA>/camera_info`      |   `sensor_msgs/msg/CameraInfo`    | Camera Info data of each camera.                                                                                  |
| `/nuscenes/<LIDAR>/pointcloud`        |   `sensor_msgs/msg/PointCloud2`   | PointCloud data of each lidar.                                                                                    |
| `/tf`                                 |     `tf2_msgs/msg/TFMessage`      | Transform of the ego vehicle.                                                                                     |
| `/tf_static`                          |     `tf2_msgs/msg/TFMessage`      | Static transform of each sensor.                                                                                  |
| `/nuscenes/annotation/boxes`          |    `mmros_msgs/msg/BoxArray3d`    | 3D annotation boxes, which is published if `publish_annotation:=true`.                                            |
| [TBD] `/nuscenes/annotation/lidarseg` |   `sensor_msgs/msg/PointCloud2`   | 3D annotation lidar segmentation pointcloud, which is published if `publish_annotation:=true` and files exist.    |
| [TBD] `/nuscenes/annotation/panoptic` |   `sensor_msgs/msg/PointCloud2`   | 3D annotation panoptic segmentation pointcloud, which is published if `publish_annotation:=true` and files exist. |

## Parameters

| Name                 |  Type   | Description                                                                                                               |
| :------------------- | :-----: | :------------------------------------------------------------------------------------------------------------------------ |
| `version`            |  `str`  | NuScenes version. (Default: `v1.0-mini`)                                                                                  |
| `data_root`          |  `str`  | Directory path to NuScenes. (Default: `mmros_dataloader/data/nuscenes`)                                                   |
| `timer_period_sec`   | `float` | The period of the timer in [s]. (Default: 0.1)                                                                            |
| `publish_all_scenes` | `bool`  | Indicates whether to publish all scenes. (Default: `true`)                                                                |
| `scene_token`        |  `str`  | If `publish_all_scenes:=false`, only the specified scene will be published. (Default: `bebf5f5b2a674631ab5c88fd1aa9e87a`) |
| `publish_annotation` | `bool`  | Indicates whether to publish annotation boxes. (Defaults: `false`)                                                        |

## How to Run

First of all, please download NuScenes dataset from [OFFICIAL WEBSITE](https://www.nuscenes.org/).

Note that we assume that downloaded dataset is placed under `mmros_dataloader/data/` by default.
If you want to specify an another directory, use `data_root:=<DATASET_ROOT>` option.

Start running the NuScenes publisher as follows:

```shell
ros2 launch mmros_dataloader nuscenes.launch.xml data_root:=<NUSCENES_ROOT>
```

### Publish Annotation Boxes

If you want to publish annotation boxes, please specify `publish_annotation:=true`:

```shell
ros2 launch mmros_dataloader nuscenes.launch.xml publish_annotation:=true
```
