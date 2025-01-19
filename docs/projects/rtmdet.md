# RTMDet

For the details of the model configuration and results, please refer to [here](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet).

## ONNX Models

| Backbone | Input Shape |                                                  Config                                                  |                                                                     Checkpoint                                                                      |                                                                                                ONNX                                                                                                |
| :------: | :---------: | :------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RTMDet-x |  3x640x640  | [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet_x_8xb32-300e_coco.py) | [checkpoint](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth) | [1x3x640x640](https://drive.google.com/uc?export=download&id=1Dh8ABtyixDHVqOcPRdB701I9ifY2Rcpd) \| [Bx3x640x640](https://drive.google.com/uc?export=download&id=1PSfmeO2Z7lIboR7vhAiYxcWWBY5RpUVN) |

## Inference Times

| Backbone |                Input Shape                 | Precision | Device  | Median Enqueue Time (ms) |
| :------: | :----------------------------------------: | :-------: | :-----: | :----------------------: |
| RTMDet-x | 1x3x640x640 \| 5x3x640x640 \| 10x3x640x640 |   FP32    | RTX3060 | 2.91 \| 2.85 \| 3.13 \|  |

## Custom TensorRT Plugins

|      Name       | Version |
| :-------------: | :-----: |
| `TRTBatchedNMS` |    1    |

## Inputs & Outputs in ROS 2

### Inputs

|     Topic      |           Type           | Description  |
| :------------: | :----------------------: | :----------: |
| `/input/image` | `/sensor_msgs/msg/Image` | Input image. |

### Outputs

|      Topic      |             Type             |  Description  |
| :-------------: | :--------------------------: | :-----------: |
| `/output/boxes` | `/mmros_msgs/msg/BoxArray2d` | Output boxes. |

## How to Run

```shell
ros2 launch rtmdet rtmdet.launch.xml
```
