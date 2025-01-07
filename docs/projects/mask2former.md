# Mask2Former

For the details of the model configuration and results, please refer to [here](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/mask2former).

## ONNX Models

| Backbone | Input Shape |                                                                Config                                                                |                                                                                             Checkpoint                                                                                              |                                                                                                 ONNX                                                                                                 | Precision |
| :------: | :---------: | :----------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------: |
| R-50-D32 | 3x512x1024  | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py) | [checkpoint](https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024/mask2former_r50_8xb2-90k_cityscapes-512x1024_20221202_140802-ffd9d750.pth) | [1x3x512x1024](https://drive.google.com/uc?export=download&id=1eR-VWLxYBgwYoMvhpDfXeMmW_8e2dD-n) \| [Bx3x512x1024](https://drive.google.com/uc?export=download&id=1tjMjyD8VRG2bokpmPgD4PTAdoPhDxYIw) |   FP32    |

## Inference Times

| Backbone |                  Input Shape                  | Precision |  Device  | Median Enqueue Time (ms)  |
| :------: | :-------------------------------------------: | :-------: | :------: | :-----------------------: |
| R-50-D32 | 1x3x512x1024 \| 5x3x512x1024 \| 10x3x512x1024 |   FP32    | RTX 3060 | 53.67 \| 257.70 \| 499.44 |

## Custom TensorRT Plugins

|            Name            | Version |
| :------------------------: | :-----: |
| `TRTInstanceNormalization` |    1    |
|       `grid_sampler`       |    1    |

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
ros2 launch mask2former mask2former.launch.xml
```
