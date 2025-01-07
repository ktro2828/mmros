# Deploy Config for MMDetection

> [!NOTE]
> Please edit input shapes suitable to your model configuration.

## Detection

In order to export onnx model with dynamic shapes, we edited deploy configs as follows:

```python
# mmdeploy/configs/_base_/onnx_config.py
onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=17,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['output'],
    input_shape=None,
    optimize=True)
```

```python
# mmdeploy/configs/mmdet/_base_/base_dynamic.py
_base_ = ['./base_static.py']
onnx_config = dict(
    dynamic_axes={
        'input': {
            0: 'batch',
        },
        'dets': {
            0: 'batch',
            1: 'num_dets',
        },
        'labels': {
            0: 'batch',
            1: 'num_dets',
        },
    }, )
```

```python
# mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-64x64-608x608.py
_base_ = ['../_base_/base_dynamic.py', '../../_base_/backends/tensorrt.py']

onnx_config = dict(input_shape=(640, 640))
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 640, 640],
                    opt_shape=[5, 3, 640, 640],
                    max_shape=[10, 3, 640, 640])))
    ])
```

## Panoptic Segmentation

TBD
