# Panoptic Segmentation

Panoptic segmentation task performs to estimate objects' 2D/3D bounding boxes and segments for each object and panoptic scene segment.

## ONNX I/O Format

Expected I/O data and their types are shown as below table.

Note that I/O data are ordered from the top of the item in table.

### 2D Instance Segmentation

|  I/O   |   Shape   |  Type   | Description                                                                                        |
| :----: | :-------: | :-----: | -------------------------------------------------------------------------------------------------- |
| Input  | `Bx3xHxW` | `float` | Input camera image.                                                                                |
| Output |  `BxNx5`  | `float` | Estimated bounding boxes in `(xmin, ymin, xmax, ymax, score)` or `(cx, cy, width, height, score)`. |
|        |   `BxN`   |  `int`  | Estimated labels.                                                                                  |
|        | `BxNxMxM` | `float` | Estimated instance segments for each object.                                                       |
|        | `Bx1xMxM` | `float` | Estimated scene segments.                                                                          |

### 3D Instance Segmentation

TBD
