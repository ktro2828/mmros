# Panoptic FPN

For the details of the model configuration and results, please refer to [here](https://github.com/open-mmlab/mmdetection/tree/main/configs/panoptic_fpn).

## ONNX Models

| Backbone |                                                       Config                                                       |                                                                            Checkpoint                                                                             |                                                                                                 ONNX                                                                                                 |
| :------: | :----------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| R-50-FPN | [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/panoptic_fpn/panoptic-fpn_r50_fpn_1x_coco.py) | [checkpoint](https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco/panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth) | [1x3x800x1344](https://drive.google.com/uc?export=download&id=1FaTVAwdtKKm7i-9wSunZfHykL_gwmZDw) \| [Bx3x800x1344](https://drive.google.com/uc?export=download&id=1brJnPer7YGMQRGd7HGVH6N2lFRfaJ-8t) |

## Inference Times

| Backbone |                  Input Shape                  | Precision |  Device  | Median Enqueue Time (ms) |
| :------: | :-------------------------------------------: | :-------: | :------: | :----------------------: |
| R-50-FPN | 1x3x800x1344 \| 5x3x800x1344 \| 10x3x800x1344 |   FP32    | RTX 3060 |   2.29 \| 2.77 \| 3.34   |

## Custom TensorRT Plugins

|           Name           | Version |
| :----------------------: | :-----: |
|       `GatherTopk`       |    1    |
|     `TRTBatchedNMS`      |    1    |
| `MMCVMultiLevelRoiAlign` |    1    |

## Inputs & Outputs in ROS 2

### Inputs

|     Topic      |           Type           | Description  |
| :------------: | :----------------------: | :----------: |
| `/input/image` | `/sensor_msgs/msg/Image` | Input image. |

### Outputs

|          Topic          |             Type             |            Description             |
| :---------------------: | :--------------------------: | :--------------------------------: |
|     `/output/boxes`     | `/mmros_msgs/msg/BoxArray2d` |           Output boxes.            |
| `/output/semantic_mask` |   `/sensor_msgs/msg/Image`   | Output semantic segmentation mask. |

## How to Run

```shell
ros2 launch panoptic_fpn panoptic_fpn.launch.xml
```
