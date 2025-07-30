# YOLOX

For the details of the model configuration and results, please refer to [here](https://github.com/open-mmlab/mmdetection/tree/main/configs/yolox).

## ONNX Models

| Backbone | Input Shape |                                                Config                                                 |                                                                  Checkpoint                                                                  |                                                                                                ONNX                                                                                                |
| :------: | :---------: | :---------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOX-l  |  3x640x640  | [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/yolox/yolox_l_8xb8-300e_coco.py) | [checkpoint](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth) | [1x3x640x640](https://drive.google.com/uc?export=download&id=1DWf4bAD4g6k8SAFEr-gOcaa1NEbXqscA) \| [Bx3x640x640](https://drive.google.com/uc?export=download&id=1YTnYTRYe3xf1XdnREyp_HM6FePzvFaqq) |

## Inference Times

| Backbone |                Input Shape                 | Precision |  Device  | Median Enqueue Time (ms) |
| :------: | :----------------------------------------: | :-------: | :------: | :----------------------: |
| YOLOX-l  | 1x3x640x640 \| 5x3x640x640 \| 10x3x640x640 |   FP32    | RTX 3060 |   2.01 \| 2.18 \| 2.28   |

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
ros2 launch yolox yolox.launch.xml
```
