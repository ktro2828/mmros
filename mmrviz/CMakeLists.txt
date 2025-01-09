cmake_minimum_required(VERSION 3.14)
project(mmrviz)

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

# -------- link visualizers --------
file(GLOB_RECURSE MMRVIZ_VISUALIZER_SOURCES
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
ament_auto_add_library(
  ${PROJECT_NAME}_visualizer SHARED
  ${MMRVIZ_VISUALIZER_SOURCES})

rclcpp_components_register_node(
  ${PROJECT_NAME}_visualizer PLUGIN
  "mmrviz::visualizer::BoxArray2dVisualizer"
  EXECUTABLE "box_array2d_visualizer_exe")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
