# Deploy Config for MMSegmentation

> [!NOTE]
> Please edit input shapes suitable to your model configuration.

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
# mmdeploy/configs/mmseg/segmentation_dynamic.py
_base_ = ['./segmentation_static.py']
onnx_config = dict(
    dynamic_axes={
        'input': {
            0: 'batch',
        },
        'output': {
            0: 'batch',
        },
    },
)
```

```python
# mmdeploy/configs/mmseg/segmentation_dynamic-512x1024-2048x2048.py
_base_ = ['./segmentation_dynamic.py', '../_base_/backends/tensorrt.py']

onnx_config = dict(input_shape=[1024, 512])
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 512, 1024],
                    opt_shape=[5, 3, 512, 1024],
                    max_shape=[10, 3, 512, 1024])))
])
```
