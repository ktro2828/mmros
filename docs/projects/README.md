# Define Project Specific Node

As an example, let's create a new project package called `my_project` which deals with 2D detection task.

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
│   └── my_project.launch.xml
└── package.xml
```

### Write the Launcher

Edit your launch file, which is `projects/my_project/launch/my_project.launch.xml`, as follows:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<launch>
  <!-- Model parameters and onnx file path -->
  <arg name="param_path" default="$(find-pkg-share my_project)/config/my_project.param.yaml"/>
  <arg name="onnx_path" default="$(find-pkg-share my_project)/data/my_project.onnx"/>

  <!-- I/O topic names -->
  <arg name="input/image" default="input/image"/>
  <arg name="output/boxes" default="output/boxes"/>

  <!-- Flag if only building TensorRT engine -->
  <arg name="build_only" default="false"/>

  <node pkg="mmros" exec="mmros_detection2d_exe" name="my_project" output="screen">
    <param from="$(var param_path)" allow_substs="true"/>
    <param name="onnx_path" value="$(var onnx_path)"/>
    <remap from="~/input/image" to="$(var input/image)"/>
    <remap from="~/output/boxes" to="$(var output/boxes)"/>
    <param name="build_only" value="$(var build_only)"/>
  </node>
</launch>
```

## (OPTIONAL) Defining Custom Node

```shell
projects/my_project/
├── CMakeLists.txt
├── config
│   └── my_project.param.yaml
├── data
├── launch
│   └── my_project.launch.xml
├── package.xml
└── src
    ├── my_project_node.cpp
    └── my_project_node.hpp
```

### Write the Publisher Node

Edit your header file, which is `projects/my_project/src/my_project_node.hpp`, as follows:

```c++
#include <mmros/node/detection2d_node.hpp>

namespace mmros::my_project
{
class MyProject : public Detection2dNode
{
public:
    explicit MyProject(const rclcpp::NodeOptions & options);
};
}
```

Edit your source file, which is `projects/my_project/src/my_project_node.cpp`, as follows:

```c++
namespace mmros::my_project
{
MyProject::MyProject(const rclcpp::NodeOptions & options) : Detection2dNode("my_project", options)
{
}
}  // namespace mmros::my_project

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mmros::my_project::MyProjectNode)
```

### Write the Launcher

Edit your launch file, which is `projects/my_project/launch/my_project.launch.xml`, as follows:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<launch>
  <!-- Model parameters and onnx file path -->
  <arg name="param_path" default="$(find-pkg-share my_project)/config/my_project.param.yaml"/>
  <arg name="onnx_path" default="$(find-pkg-share my_project)/data/my_project.onnx"/>

  <!-- I/O topic names -->
  <arg name="input/image" default="input/image"/>
  <arg name="output/boxes" default="output/boxes"/>

  <!-- Flag if only building TensorRT engine -->
  <arg name="build_only" default="false"/>

  <node pkg="my_project" exec="my_project_exe" output="screen">
    <param from="$(var param_path)" allow_substs="true"/>
    <param name="onnx_path" value="$(var onnx_path)"/>
    <remap from="~/input/image" to="$(var input/image)"/>
    <remap from="~/output/boxes" to="$(var output/boxes)"/>
    <param name="build_only" value="$(var build_only)"/>
  </node>
</launch>
```

### Add Dependencies

Edit your project's `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_project</name>
  <version>0.0.0</version>
  <description>The ROS 2 package for my project.</description>
  <maintainer email="you@email.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake_auto</buildtool_depend>

  <depend>mmros</depend>
  <depend>rclcpp</depend>
  <depend>rclcpp_components</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### CMakeLists.txt

Edit your project's `CMakeLits.txt`:

```txt
cmake_minimum_required(VERSION 3.14)
project(my_project)

# -------- default to C++17 --------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# -------- create compile_commands.json --------
if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# -------- find dependencies --------
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# -------- link targets --------
file(GLOB_RECURSE MY_PROJECT_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
ament_auto_add_library(${PROJECT_NAME} SHARED ${MY_PROJECT_SOURCES})
ament_target_dependencies(${PROJECT_NAME} mmros mmros_msgs)

# -------- export component --------
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "mmros::yolox::YoloxNode"
  EXECUTABLE ${PROJECT_NAME}_exe
)

# -------- testing --------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# -------- install --------
ament_auto_package(INSTALL_TO_SHARE
  launch
  config
  data
)
```

## Build and Run

Before building your project, please run `rosdep` to install ROS 2 dependencies via `apt`:

```shell
cd /PATH/TO/MMROS
rosdep update
rosdep install -y --from-paths . --ignore-src --rosdistro $ROS_DISTRO
```

Build your new project, and source the setup files:

```shell
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to my_project
source install/setup.bash
```

Now launch your node:

```shell
ros2 launch my_project my_project.launch.xml
```

If you want to build only TensorRT engine, specify the `build_only:=true` option:

```shell
ros2 launch my_project my_project.launch.xml build_only:=true
```
