# Define Project Specific Node

As an example, let's create a new project package called `my_project` which deals with 2D detection task.

For the other tasks, please refer to sample projects corresponding to the task.

```shell
cd /PATH/TO/MMROS/projects
ros2 pkg create my_project
```

## Using Default Node

```shell
projects/my_project/
├── CMakeLists.txt
├── config
│   └── my_project.param.yaml
├── data
├── launch
│   └── my_project.onnx
├── launch
│   └── my_project.launch.xml
└── package.xml
```

### Write the Launcher & Parameter Config

Edit your launch file, which is `projects/my_project/launch/my_project.launch.xml`, as follows:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<launch>
  <!-- Model parameters and onnx file path -->
  <arg name="param_path" default="$(find-pkg-share my_project)/config/my_project.param.yaml"/>
  <arg name="data_path" default="$(find-pkg-share my_project)/data"/>

  <!-- I/O topic names -->
  <arg name="input/image" default="input/image"/>
  <arg name="output/boxes" default="output/boxes"/>

  <!-- Whether to subscribe raw image (or compressed image) as an input -->
  <arg name="use_raw" default="false"/>

  <!-- Flag if only building TensorRT engine -->
  <arg name="build_only" default="false"/>

  <!-- Whether to launch visualizer node -->
  <arg name="visualize" default="true"/>

  <group>
    <push-ros-namespace namespace="my_project"/>
    <!-- Detector node -->
    <node pkg="mmros" exec="mmros_detection2d_exe" name="detector" output="screen">
      <param from="$(var param_path)" allow_substs="true"/>
      <remap from="~/input/image" to="$(var input/image)"/>
      <remap from="~/output/boxes" to="$(var output/boxes)"/>
      <param name="use_raw" value="$(var use_raw)"/>
      <param name="build_only" value="$(var build_only)"/>
    </node>

    <!-- Visualizer node -->
    <group if="$(var visualize)">
      <node pkg="mmrviz" exec="box_array2d_visualizer_exe" output="screen">
        <param name="use_raw" value="$(var use_raw)"/>
        <remap from="~/input/image" to="$(var input/image)"/>
        <remap from="~/input/boxes" to="$(var output/boxes)"/>
      </node>
    </group>
  <group>
</launch>
```

Edit your parameter file, which is `projects/my_project/config/my_config.param.yaml` as follows:

```yaml
/**:
  ros__parameters:
    tensorrt:
      onnx_path: $(var data_path)/my_project.onnx
      precision: fp32
    detector:
      mean: [0.0, 0.0, 0.0]
      std: [1.0, 1.0, 1.0]
      box_format: XYXYS # [xmin, ymin, xmax, ymax, score]
      score_threshold: 0.2
```
