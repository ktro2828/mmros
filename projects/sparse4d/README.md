# Sparse4D

Initial independent ROS 2 package for Sparse4D integration.

## Status

This package currently provides the package-local node, launch file, and parameter surface.
The detector path is scaffolded and returns empty `BoxArray3d` outputs until Sparse4D TensorRT
preprocess and decode are implemented.

## Model

Default ONNX path:

`/home/ktro2828/myWorkspace/Sparse4D/artifacts/sparse4dv3.onnx`

## Run

```sh
ros2 launch sparse4d sparse4d.launch.xml
```
