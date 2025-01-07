# Panoptic FPN

For the details of the model configuration and results, please refer to [here](https://github.com/open-mmlab/mmdetection/tree/main/configs/panoptic_fpn).

## ONNX Models

| Backbone |                                                       Config                                                       |                                                                            Checkpoint                                                                             |                                            ONNX                                             | Precision |
| :------: | :----------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: | :-------: |
| R-50-FPN | [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/panoptic_fpn/panoptic-fpn_r50_fpn_1x_coco.py) | [checkpoint](https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco/panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth) | [Dynamic](https://drive.google.com/uc?export=download&id=13wLV0AWPCNdE0unm1l-NDFHwkhCfWOCV) |   FP32    |

## Inference Times

N/A

## Custom TensorRT Plugins

N/A

## Inputs & Outputs in ROS 2

### Inputs

|     Topic      |           Type           | Description  |
| :------------: | :----------------------: | :----------: |
| `/input/image` | `/sensor_msgs/msg/Image` | Input image. |

### Outputs

TBD

## How to Run

```shell
ros2 launch panoptic_fpn panoptic_fpn.launch.xml
```
