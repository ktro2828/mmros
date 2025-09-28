# MMROS

ROS 2 support of [OpenMMLab](https://openmmlab.com/) projects using TensorRT.

![DEMO](./docs/assets/demo.gif)

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- TensorRT>=8.6

## Install

1. Install ROS 2. Please refer to [OFFICIAL DOCUMENT of ROS 2 HUMBLE](https://docs.ros.org/en/humble/Installation.html).
2. Install CUDA/CUDNN/TensorRT. Please refer to [OFFICIAL DOCUMENT of NVIDIA](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).
3. Clone repository and install dependencies:

   ```shell
   git clone git@github.com:ktro2828/mmros && cd mmros
   rosdep update
   rosdep install -y --from-paths . --ignore-src --rosdistro $ROS_DISTRO
   ```

4. Build all packages

   ```shell
   colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
   ```

5. (OPTIONAL) Build Only Specific Project

   Note that `<PROJECT_NAME>` represents the name of the particular package under the `projects` directory.

   ```shell
   colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to <PROJECT_NAME>
   ```

## Sample Projects

| Task                  | Modality | Model                                                              |
| --------------------- | -------- | ------------------------------------------------------------------ |
| Detection             | Camera   | [DETR (ECCV'2020)](./docs/projects/detr.md)                        |
|                       | Camera   | [YOLOX (CVPR'2021)](./docs/projects/yolox.md)                      |
|                       | Camera   | [RTMDet (ArXiv'2022)](./docs/projects/rtmdet.md)                   |
|                       | Camera   | [RF-DETR (ArXiv'2025)](./docs/projects/rfdetr.md)                  |
|                       | Camera   | [DEIMv2 (ArXiv'2025)](./docs/projects/deimv2.md)                   |
| Semantic Segmentation | Camera   | [Mask2Former (CVPR'2022)](./docs/projects/mask2former.md)          |
|                       | Camera   | [PIDNet (ArXiv'2022)](./docs/projects/pidnet.md)                   |
|                       | Camera   | [EoMT (CVPR'2025)](./docs/projects/eomt.md)                        |
| Instance Segmentation | Camera   | [Instance RTMDet (ArXiv'2022)](./docs/projects/instance_rtmdet.md) |
| Panoptic Segmentation | Camera   | [Panoptic FPN (CVPR'2019)](./docs/projects/panoptic_fpn.md)        |

## User Guides

- [😎 Supported Tasks](./docs/tasks/README.md)
- [💻 How to Deploy ONNX from MMLab Projects](./docs/deploys/README.md)
- [⚙️ Create Custom Projects](./docs/projects/README.md)
- [📚 MMROS DataLoader](./docs/datasets/README.md)
- [🖼️ MMRviz](./docs/visualizers/README.md)

## Contribution

We are welcome your contribution!!

Before starting your work, please follow the [Contributing Guidelines](./docs/CONTRIBUTING.md).
