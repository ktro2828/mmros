# MMRViz {#mainpage}

`mmrviz` features RViz plugins and visualizers for each topic used in MMROS.

## Supported RViz Plugins

TBD

## Supported Visualizers

Following visualizer classes are supported:

| Name                                                               | Description                                                               |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| [`BoxArray2dVisualizer`](./box_array2d.md)                         | Visualize `mmros_msgs/msg/BoxArray2d` messages.                           |
| [`Segmentation2dVisualizer`](./segmentation2d.md)                  | Visualize `sensor_msgs/msg/Image` messages, which are encoded by `MONO8`. |
| [`InstanceSegmentation2dVisualizer`](./instance_segmentation2d.md) | Visualize `mmros_msgs/msg/InstanceSegmentArray2d` messages.               |
| [`BoxArray3dVisualizer`](./box_array3d.md)                         | Visualize `mmros_msgs/msg/BoxArray3d` messages on image.                  |
