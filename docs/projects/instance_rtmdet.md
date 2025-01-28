# New Project

For the details of the model configuration and results, please refer to [here]<LINK-TO-OpenMMLab-PROJECT>.

## ONNX Models

|    Model     | Input Shape |                                                    Config                                                    |                                                                         Checkpoint                                                                          |                                              ONNX                                               |
| :----------: | :---------: | :----------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------: |
| RTMDet-Ins-x |  3x640x640  | [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet-ins_x_8xb16-300e_coco.py) | [checkpoint](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_x_8xb16-300e_coco/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth) | [1x3x640x640](https://drive.google.com/uc?export=download&id=1LquFPJj-GpntYDdcuvSOfQbEhrzt5S1S) |

## Inference Times

<!-- Please describe inference time of the model using trtexec. -->

|    Model     | Input Shape | Precision | Device  | Median Enqueue Time (ms) |
| :----------: | :---------: | :-------: | :-----: | :----------------------: |
| RTMDet-Ins-x | 1x3x640x640 |   FP32    | RTX3060 |           3.04           |

## Custom TensorRT Plugins

<!-- Please describe custom TensorRT plugins. -->

|      Name       | Version |
| :-------------: | :-----: |
| `TRTBatchedNMS` |    1    |

## Inputs & Outputs in ROS 2

### Inputs

<!-- Input topics, types and descriptions -->

|      Topic      |          Type           | Description  |
| :-------------: | :---------------------: | :----------: |
| `~/input/image` | `sensor_msgs/msg/Image` | Input image. |

### Outputs

<!-- Output topics, types and descriptions. -->

|        Topic        |                     Type                     |        Description        |
| :-----------------: | :------------------------------------------: | :-----------------------: |
| `~/output/segments` | `/mmros_msgs/msg/InstanceSegmentArray2d.msg` | Output instance segments. |

## How to Run

<!-- Please describe launch command. -->

```shell
ros2 launch instance_rtmdet instance_rtmdet.launch.xml
```
