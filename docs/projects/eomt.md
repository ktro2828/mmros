# EoMT

For the details of the model configuration and results, please refer to [here](https://github.com/ktro2828/eomt).

## ONNX Models

| Backbone | Input Shape |                                                        Config                                                        | Checkpoint |                                               ONNX                                                |
| :------: | :---------: | :------------------------------------------------------------------------------------------------------------------: | :--------: | :-----------------------------------------------------------------------------------------------: |
|  DinoV2  | 3x1024x1024 | [config](https://github.com/ktro2828/eomt/blob/develop/onnx/configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml) |    N/A     | [1x3x1024x1024](https://drive.google.com/uc?export=download&id=1cHOfOysS0grYzazHQwD_GogF1bWpJMOq) |

## Inference Times

<!-- Please describe inference time of the model using trtexec. -->

| Backbone |  Input Shape  | Precision |  Device  | Median Enqueue Time (ms) |
| :------: | :-----------: | :-------: | :------: | :----------------------: |
|  DinoV2  | 1x3x1024x1024 |   FP32    | RTX 3060 |          2.177           |

## Custom TensorRT Plugins

<!-- Please describe custom TensorRT plugins. -->

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
ros2 launch eomt eomt.launch.xml
```
