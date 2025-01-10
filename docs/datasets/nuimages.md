# NuImages

## Published Messages

Following topics are published with the timestamp at the current ROS 2 simulation time:

| Topic                                         |             Type             | Description                                                                   |
| :-------------------------------------------- | :--------------------------: | :---------------------------------------------------------------------------- |
| `/nuimages/<CAMERA>/image`                    |   `sensor_msgs/msg/Image`    | Image data of each camera.                                                    |
| `/nuimages/<CAMERA>/camera_info`              | `sensor_msgs/msg/CameraInfo` | Camera Info data of each camera.                                              |
| `/tf`                                         |   `tf2_msgs/msg/TFMessage`   | Transform of the ego vehicle.                                                 |
| `/tf_static`                                  |   `tf2_msgs/msg/TFMessage`   | Static transform of each sensor.                                              |
| `/nuimages/<CAMERA>/annotation/boxes`         | `mmros_msgs/msg/BoxArray2d`  | 2D annotated boxes, which is published if `publish_annotation:=true`.         |
| `/nuimages/<CAMERA>/annotation/semantic_mask` |   `sensor_msgs/msg/Image`    | 2D annotated semantic mask, which is published if `publish_annotation:=true`. |
| `/nuimages/<CAMERA>/annotation/instance_mask` |   `sensor_msgs/msg/Image`    | 2D annotated instance mask, which is published if `publish_annotation:=true`. |

## Parameters

| Name                 |  Type   | Description                                                             |
| :------------------- | :-----: | :---------------------------------------------------------------------- |
| `version`            |  `str`  | NuImages version. (Default: `v1.0-mini`)                                |
| `data_root`          |  `str`  | Directory path to NuImages. (Default: `mmros_dataloader/data/nuimages`) |
| `timer_period_sec`   | `float` | The period of the timer in [s]. (Default: 0.1)                          |
| `publish_annotation` | `bool`  | Indicates whether to publish annotation boxes. (Defaults: `false`)      |

## How to Run

First of all, please download NuImages dataset from [OFFICIAL WEBSITE](https://www.nuscenes.org/).

Note that we assume that downloaded dataset is placed under `mmros_dataloader/data/`.
If your dataset is placed in another directory, use `data_root:=<DATASET_ROOT>` option.

Start running the NuImages publisher as follows:

```shell
ros2 launch mmros_dataloader nuimages.launch.xml data_root:=<NUIMAGES_ROOT>
```

### Publish Annotation Boxes

If you want to publish annotation boxes and masks, please specify `publish_annotation:=true`:

```shell
ros2 launch mmros_dataloader nuimages.launch.xml publish_annotation:=true
```
