cmake_minimum_required(VERSION 3.14)
project(sparse4d)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(cudnn_cmake_module REQUIRED)
find_package(tensorrt_cmake_module REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CUDNN REQUIRED)
find_package(TENSORRT REQUIRED)
find_library(
  CUBLAS_LIB cublas
  HINTS /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(
  ${PROJECT_NAME} SHARED
  src/sparse4d_detector.cpp src/sparse4d_node.cpp)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${TENSORRT_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(
  ${PROJECT_NAME} mmros::mmros CUDA::cudart
  ${TENSORRT_LIBRARIES} ${CUDNN_LIBRARIES}
  ${CUBLAS_LIB})

ament_target_dependencies(
  ${PROJECT_NAME}
  cv_bridge
  geometry_msgs
  image_transport
  mmros
  mmros_msgs
  rclcpp
  rclcpp_components
  sensor_msgs
  tf2
  tf2_ros)

rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN "sparse4d::Sparse4dNode"
  EXECUTABLE ${PROJECT_NAME}_component_exe)
target_link_options(
  ${PROJECT_NAME}_component_exe PRIVATE
  -Wl,--allow-shlib-undefined)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE launch config
                   USE_SCOPED_HEADER_INSTALL_DIR)
