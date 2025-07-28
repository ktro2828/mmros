# RF-DETR

For the details of the model configuration and results, please refer to [here](https://github.com/roboflow/rf-detr).

## ONNX Models

| Size   | Input Shape | ONNX                                                                                            |
| ------ | ----------- | ----------------------------------------------------------------------------------------------- |
| Base   | 3x560x560   | [1x3x560x560](https://drive.google.com/uc?export=download&id=1R14PgL1YxaZrwgz_iHgFApc7MAzzkgoh) |
| Nano   | 3x384x384   | [1x3x384x384](https://drive.google.com/uc?export=download&id=1HlGUp4E0smp70nqMa1rMuRVY2c7fwBEb) |
| Small  | 3x512x512   | [1x3x512x512](https://drive.google.com/uc?export=download&id=1iuF9gjcPqx5QQXSTcKvG6Mk5hsx8gljj) |
| Medium | 3x576x576   | [1x3x576x576](https://drive.google.com/uc?export=download&id=1dlKcmEeO1qloFaEZUnusGW_uNhABe7Zp) |
| Large  | 3x560x560   | [1x3x560x560](https://drive.google.com/uc?export=download&id=1vGeE1uIynRQzA_5b7j-txyTRo9M6dQA)  |

### How to Export ONNX

To install RF-DETR, please refer to [OFFICIAL DOCUMENTATION](https://rfdetr.roboflow.com/#install).

Here is the sample script to export the ONNX model:

```python
from __future__ import annotations

import argparse
from enum import Enum
from typing import TYPE_CHECKING

from rfdetr.detr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

if TYPE_CHECKING:
    from rfdetr.detr import RFDETR


class ModelSize(str, Enum):
    BASE = "base"
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    def build_model(self) -> RFDETR:
        if self == ModelSize.BASE:
            return RFDETRBase()
        elif self == ModelSize.NANO:
            return RFDETRNano()
        elif self == ModelSize.SMALL:
            return RFDETRSmall()
        elif self == ModelSize.MEDIUM:
            return RFDETRMedium()
        elif self == ModelSize.LARGE:
            return RFDETRLarge()
        else:
            raise ValueError(f"Unknown model size: {self}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_size",
        type=ModelSize,
        choices=["base", "nano", "small", "medium", "large"],
    )

    args = parser.parse_args()

    model_size: ModelSize = args.model_size
    model = model_size.build_model()

    model.export()


if __name__ == "__main__":
    main()
```

## Inference Times

| Size   | Input Shape | Precision | Device  | Median Enqueue Time (ms) |
| ------ | ----------- | --------- | ------- | ------------------------ |
| Base   | 1x3x560x560 | FP32      | RTX3060 | 1.400                    |
| Nano   | 1x3x384x384 | FP32      | RTX3060 | 1.441                    |
| Small  | 1x3x512x512 | FP32      | RTX3060 | 1.462                    |
| Medium | 1x3x576x576 | FP32      | RTX3060 | 1.656                    |
| Large  | 1x3x560x560 | FP32      | RTX3060 | 1.874                    |

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
ros2 launch rfdetr rfdetr.launch.xml
```
