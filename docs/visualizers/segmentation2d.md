# `Segmentation2dVisualizer`

This node performs to visualize 2D segmentation masks on the image.

## Inputs & Outputs

| Name             |          Topic          | Description                              |
| :--------------- | :---------------------: | :--------------------------------------- |
| `~/input/image`  | `sensor_msgs/msg/Image` | Source image.                            |
| `~/input/mask`   | `sensor_msgs/msg/Image` | 2D segmentation mask encoded by `MONO8`. |
| `~/output/image` | `sensor_msgs/msg/Image` | Rendering result.                        |

## Parameters

| Name      |  Type  | Description                                      |
| :-------- | :----: | :----------------------------------------------- |
| `use_raw` | `bool` | Indicates whether input image is not compressed. |
