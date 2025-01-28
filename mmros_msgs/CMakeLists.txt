cmake_minimum_required(VERSION 3.14)
project(mmros_msgs)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  "msg/Box2d.msg"
  "msg/Box3d.msg"
  "msg/BoxArray2d.msg"
  "msg/BoxArray3d.msg"
  "msg/InstanceSegment2d.msg"
  "msg/InstanceSegmentArray2d.msg"
  "msg/Edge.msg"
  "msg/Skeleton.msg"
  "msg/SkeletonArray.msg"
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs)

ament_export_dependencies(rosidl_default_runtime)

ament_auto_package()
