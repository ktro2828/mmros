# New Project

For the details of the model configuration and results, please refer to [here](https://github.com/ktro2828/DEIMv2).

## ONNX Models

| Backbone | Input Shape |                                                 Config                                                 |                                           Checkpoint                                           |                                              ONNX                                               |
| :------: | :---------: | :----------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------: |
|  DinoV3  |  3x640x640  | [config](https://github.com/ktro2828/DEIMv2/blob/develop/onnx/configs/deimv2/deimv2_dinov3_l_coco.yml) | [checkpoint](https://drive.google.com/uc?export=download&id=1dRJfVHr9HtpdvaHlnQP460yPVHynMray) | [1x3x640x640](https://drive.google.com/uc?export=download&id=1v6lCBrlsMYKIWUFDI1sGzx08wtSNIH1H) |

## Inference Times

| Backbone | Input Shape | Precision |  Device  | Median Enqueue Time (ms) |
| :------: | :---------: | :-------: | :------: | :----------------------: |
|  DinoV3  | 1x3x640x640 |   FP32    | RTX 3060 |          3.098           |

## Custom TensorRT Plugins

<!-- Please describe custom TensorRT plugins. -->

N/A

## Inputs & Outputs in ROS 2

### Inputs

<!-- Input topics, types and descriptions -->

|      Topic      |           Type           | Description  |
| :-------------: | :----------------------: | :----------: |
| `~/input/image` | `/sensor_msgs/msg/Image` | Input image. |

### Outputs

<!-- Output topics, types and descriptions. -->

|      Topic       |             Type             |  Description  |
| :--------------: | :--------------------------: | :-----------: |
| `~/output/boxes` | `/mmros_msgs/msg/BoxArray2d` | Output boxes. |

## How to Run

```shell
ros2 launch deimv2 deimv2.launch.xml
```
