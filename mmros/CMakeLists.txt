cmake_minimum_required(VERSION 3.14)
project(mmros)

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

# -------- find CUDA/CUDNN/TensorRT dependencies --------
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CUDNN REQUIRED)
find_package(TENSORRT REQUIRED)

if(NOT
   (CUDAToolkit_FOUND
    AND CUDNN_FOUND
    AND TENSORRT_FOUND))
  message(
    FATAL_ERROR
      "cuda, cudnn, tensorrt libraries are not found"
  )
endif()

if(TENSORRT_VERSION VERSION_LESS 8.5)
  message(
    FATAL_ERROR
      "Unsupported version TensorRT ${TENSORRT_VERSION} detected. This package requires TensorRT 8.5 or later."
  )
endif()

# -------- link targets --------
file(GLOB_RECURSE MMROS_SOURCES
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)

cuda_add_library(${PROJECT_NAME} SHARED
                 ${MMROS_SOURCES})

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
    ${TENSORRT_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(
  ${PROJECT_NAME} stdc++fs CUDA::cudart
  ${OpenCV_LIBS} ${TENSORRT_LIBRARIES})

ament_target_dependencies(
  ${PROJECT_NAME}
  mmros_msgs
  cv_bridge
  rclcpp
  rclcpp_components
  sensor_msgs
  image_transport)

target_compile_definitions(
  ${PROJECT_NAME}
  PRIVATE
    TENSORRT_VERSION_MAJOR=${TENSORRT_VERSION_MAJOR}
)

# -------- register composable nodes --------
# detection2d
rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN
  "mmros::node::Detection2dNode" EXECUTABLE
  ${PROJECT_NAME}_detection2d_exe)

# instance segmentation2d
rclcpp_components_register_node(
  ${PROJECT_NAME}
  PLUGIN
  "mmros::node::InstanceSegmentation2dNode"
  EXECUTABLE
  ${PROJECT_NAME}_instance_segmentation2d_exe)

# semantic segmentation2d
rclcpp_components_register_node(
  ${PROJECT_NAME}
  PLUGIN
  "mmros::node::SemanticSegmentation2dNode"
  EXECUTABLE
  ${PROJECT_NAME}_semantic_segmentation2d_exe)

# panoptic segmentation2d
rclcpp_components_register_node(
  ${PROJECT_NAME}
  PLUGIN
  "mmros::node::PanopticSegmentation2dNode"
  EXECUTABLE
  ${PROJECT_NAME}_panoptic_segmentation2d_exe)

# -------- testing targets --------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# -------- install targets --------
install(TARGETS ${PROJECT_NAME}
        EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/
        DESTINATION include/${PROJECT_NAME})

ament_export_include_directories(
  "include/${PROJECT_NAME}")
ament_export_targets(export_${PROJECT_NAME})

ament_export_dependencies(
  "CUDA"
  "CUDAToolkit"
  "cudnn_cmake_module"
  "CUDNN"
  "rclcpp"
  "tensorrt_cmake_module"
  "TENSORRT")

ament_auto_package()
