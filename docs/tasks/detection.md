# Detection

Detection task performs to estimate objects' 2D/3D bounding boxes.

## ONNX I/O Format

Expected I/O data and their types are shown as below table.

Note that I/O data are ordered from the top of the item in table.

### 2D Detection

|  I/O   |   Shape   |   Type    | Description               |
| :----: | :-------: | :-------: | ------------------------- |
| Input  | `Bx3xHxW` | `float[]` | Input camera image.       |
| Output |  `BxNxD`  | `float[]` | Estimated bounding boxes. |
|        |  `BxNxC`  |  `int[]`  | Estimated labels.         |

### 3D Detection

TBD
