# DETR

For the details of the model configuration and results, please refer to [here](https://github.com/open-mmlab/mmdetection/tree/main/configs/detr).

## ONNX Models

| Backbone | Input Shape |                                                Config                                                 |                                                                   Checkpoint                                                                    |                                                                                                 ONNX                                                                                                 |
| :------: | :---------: | :---------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R50    | 3x800x1344  | [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/detr/detr_r50_8xb2-150e_coco.py) | [checkpoint](https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth) | [1x3x800x1344](https://drive.google.com/uc?export=download&id=15Taa9qRKdVFfcHTw6bZeGwEMR9GaRZyO) \| [Bx3x800x1344](https://drive.google.com/uc?export=download&id=1ZFTLDwamiVTY0iXHeBX7snmqDAMDWdmI) |

## Inference Times

| Backbone |                  Input Shape                  | Precision |  Device  | Median Enqueue Time (ms) |
| :------: | :-------------------------------------------: | :-------: | :------: | :----------------------: |
|   R-50   | 1x3x800x1344 \| 5x3x800x1344 \| 10x3x800x1344 |   FP32    | RTX 3060 |   1.74 \| 1.78 \| 1.93   |

## Custom TensorRT Plugins

N/A

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
ros2 launch detr detr.launch.xml
```
