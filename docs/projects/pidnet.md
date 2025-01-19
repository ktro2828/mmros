# PIDNet

For the details of the model configuration and results, please refer to [here](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/pidnet).

## ONNX Models

| Backbone | Input Shape |                                                           Config                                                           |                                                                                      Checkpoint                                                                                      |                                                                                                  ONNX                                                                                                  |
| :------: | :---------: | :------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| PIDNet-L | 3x1024x1024 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py) | [checkpoint](https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes/pidnet-l_2xb6-120k_1024x1024-cityscapes_20230303_114514-0783ca6b.pth) | [1x3x1024x1024](https://drive.google.com/uc?export=download&id=1A5ykQop8_hQQqriu5R41-uakTuJi9DEb) \| [Bx3x1024x1024](https://drive.google.com/uc?export=download&id=1WCOizNcvReey-I3jISs6_WR_ZNpI_3ZX) |

## Inference Times

| Backbone |                   Input Shape                    | Precision | Device  | Median Enqueue Time (ms) |
| :------: | :----------------------------------------------: | :-------: | :-----: | :----------------------: |
| PIDNet-L | 1x3x1024x1024 \| 5x3x1024x1024 \| 10x3x1024x1024 |   FP32    | RTX3060 |   1.27 \| 1.39 \| 1.44   |

## Custom TensorRT Plugins

N/A

## Inputs & Outputs in ROS 2

### Inputs

|     Topic      |           Type           | Description  |
| :------------: | :----------------------: | :----------: |
| `/input/image` | `/sensor_msgs/msg/Image` | Input image. |

### Outputs

|     Topic      |           Type           |        Description        |
| :------------: | :----------------------: | :-----------------------: |
| `/output/mask` | `/sensor_msgs/msg/Image` | Output segmentation mask. |

## How to Run

```shell
ros2 launch pidnet pidnet.launch.xml
```
