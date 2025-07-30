# FAQ

## Trouble Shooting

### Build Error While Compiling `TRTBatchedNMS`

While building a detector that includes `TRTBatchedNMS`, the following error may occur:

```shell
[mmros_detection2d_exe-1] [I] [TRT] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 35, GPU 921 (MiB)
[mmros_detection2d_exe-1] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +2645, GPU +412, now: CPU 2882, GPU 1333 (MiB)
[mmros_detection2d_exe-1] [I] [TRT] ----------------------------------------------------------------
[mmros_detection2d_exe-1] [I] [TRT] Input filename:   /home/ktro2828/myWorkspace/mmros/install/yolox/share/yolox/data/yolox_l_8x8_300e_coco-static-640x640.onnx
[mmros_detection2d_exe-1] [I] [TRT] ONNX IR version:  0.0.8
[mmros_detection2d_exe-1] [I] [TRT] Opset version:    17
[mmros_detection2d_exe-1] [I] [TRT] Producer name:    pytorch
[mmros_detection2d_exe-1] [I] [TRT] Producer version: 2.0.0
[mmros_detection2d_exe-1] [I] [TRT] Domain:
[mmros_detection2d_exe-1] [I] [TRT] Model version:    0
[mmros_detection2d_exe-1] [I] [TRT] Doc string:
[mmros_detection2d_exe-1] [I] [TRT] ----------------------------------------------------------------
[mmros_detection2d_exe-1] [I] [TRT] No checker registered for op: TRTBatchedNMS. Attempting to check as plugin.
[mmros_detection2d_exe-1] [I] [TRT] No importer registered for op: TRTBatchedNMS. Attempting to import as plugin.
[mmros_detection2d_exe-1] [I] [TRT] Searching for plugin: TRTBatchedNMS, plugin_version: 1, plugin_namespace:
[mmros_detection2d_exe-1] [I] [TRT] Successfully created plugin: TRTBatchedNMS
[mmros_detection2d_exe-1] [E] [TRT] ModelImporter.cpp:967: ERROR: ModelImporter.cpp:796 In function importModel:
[mmros_detection2d_exe-1] [8] Assertion failed: (output_tensor_ptr->getType() != nvinfer1::DataType::kINT32 || output_trt_dtype == nvinfer1::DataType::kINT32) && "For INT32 tensors, the output type must also be INT32."
[mmros_detection2d_exe-1] terminate called after throwing an instance of 'std::runtime_error'
[mmros_detection2d_exe-1]   what():  Failed to initialize TensorRT
```

This error is caused by a mismatch in data type (`dtype`) between ONNX and `TRTBatchedNMS` plugin.

As of TensorRT>=10.0, the framework performs stricter type checking. If the ONNX model specifies output tensor as `INT64`,
but the plugin expects `INT32`, TensorRT will throw a validation error during engine building.

Note that even if you modify expected data types from `INT32` to `INT64` by editing `TRTBatchedNMS`,
another error below will occur because TensorRT doesn't support `INT64` officially:

```shell
[mmros_detection2d_exe-1] [I] [TRT] Applying optimizations and building TensorRT CUDA engine. Please wait for a few minutes...
[mmros_detection2d_exe-1] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
[mmros_detection2d_exe-1] [E] [TRT] IBuilder::buildSerializedNetwork: Error Code 9: Internal Error (/TRTBatchedNMS: could not find any supported formats consistent with input/output data types)
[mmros_detection2d_exe-1] [E] [TRT] Fail to create host memory
```

To fix this issue, run the following script to convert the output data types from `INT64` to `INT32`:

```python
import argparse
import os.path as osp

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model data types")
    parser.add_argument("onnx", type=str, help="Path to the ONNX model file")
    args = parser.parse_args()

    onnx_path: str = args.onnx
    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)

    for output in graph.outputs:
        if output.dtype == np.int64:
            print(f"Converting output {output.name} from int64 to int32")
            output.dtype = np.int32

    model_name = osp.basename(onnx_path).removesuffix(".onnx")
    onnx.save(gs.export_onnx(graph), onnx_path.replace(model_name, f"{model_name}_converted_dtype"))


if __name__ == "__main__":
    main()
```

This script modifies the ONNX model to ensure that all output tensors use `INT32`, which is compatible with TensorRT.

If the above script didn't work, please try [aadhithya/onnx-typecast](https://github.com/aadhithya/onnx-typecast).

After converting the model and launching the node with the modified ONNX file, the error no longer occurs:

```shell
[INFO] [launch]: All log files can be found below /home/ktro2828/.ros/log/2025-07-31-00-36-42-070552-ktro2828-desktop-2577523
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [mmros_detection2d_exe-1]: process started with pid [2577524]
[mmros_detection2d_exe-1] [I] [TRT] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 35, GPU 952 (MiB)
[mmros_detection2d_exe-1] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +2645, GPU +411, now: CPU 2882, GPU 1363 (MiB)
[mmros_detection2d_exe-1] [I] [TRT] ----------------------------------------------------------------
[mmros_detection2d_exe-1] [I] [TRT] Input filename:   /home/ktro2828/myWorkspace/mmros/install/yolox/share/yolox/data/yolox_l_8x8_300e_coco-static-640x640_converted_dtype.onnx
[mmros_detection2d_exe-1] [I] [TRT] ONNX IR version:  0.0.10
[mmros_detection2d_exe-1] [I] [TRT] Opset version:    17
[mmros_detection2d_exe-1] [I] [TRT] Producer name:    pytorch
[mmros_detection2d_exe-1] [I] [TRT] Producer version: 2.0.0
[mmros_detection2d_exe-1] [I] [TRT] Domain:
[mmros_detection2d_exe-1] [I] [TRT] Model version:    0
[mmros_detection2d_exe-1] [I] [TRT] Doc string:
[mmros_detection2d_exe-1] [I] [TRT] ----------------------------------------------------------------
[mmros_detection2d_exe-1] [I] [TRT] No checker registered for op: TRTBatchedNMS. Attempting to check as plugin.
[mmros_detection2d_exe-1] [I] [TRT] No importer registered for op: TRTBatchedNMS. Attempting to import as plugin.
[mmros_detection2d_exe-1] [I] [TRT] Searching for plugin: TRTBatchedNMS, plugin_version: 1, plugin_namespace:
[mmros_detection2d_exe-1] [I] [TRT] Successfully created plugin: TRTBatchedNMS
[mmros_detection2d_exe-1] [W] [TRT] Engine is not initialized. Retrieving data from network
[mmros_detection2d_exe-1] [W] [TRT] Engine is not initialized. Retrieving data from network
[mmros_detection2d_exe-1] [I] [TRT] Setting optimization profile for tensor: input {min [1, 3, 640, 640], opt [1, 3, 640, 640], max [1, 3, 640, 640]}
[mmros_detection2d_exe-1] [I] [TRT] Starting to build engine
[mmros_detection2d_exe-1] [I] [TRT] Applying optimizations and building TensorRT CUDA engine. Please wait for a few minutes...
[mmros_detection2d_exe-1] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
[mmros_detection2d_exe-1] [I] [TRT] Applying optimizations and building TensorRT CUDA engine. Please wait for a few minutes...
[mmros_detection2d_exe-1] [I] [TRT] Applying optimizations and building TensorRT CUDA engine. Please wait for a few minutes...
[mmros_detection2d_exe-1] [I] [TRT] Applying optimizations and building TensorRT CUDA engine. Please wait for a few minutes...
[mmros_detection2d_exe-1] [I] [TRT] Applying optimizations and building TensorRT CUDA engine. Please wait for a few minutes...
[mmros_detection2d_exe-1] [I] [TRT] Applying optimizations and building TensorRT CUDA engine. Please wait for a few minutes...
[mmros_detection2d_exe-1] [I] [TRT] Compiler backend is used during engine build.
[mmros_detection2d_exe-1] [I] [TRT] Applying optimizations and building TensorRT CUDA engine. Please wait for a few minutes...
[mmros_detection2d_exe-1] [I] [TRT] Applying optimizations and building TensorRT CUDA engine. Please wait for a few minutes...
[mmros_detection2d_exe-1] [I] [TRT] Applying optimizations and building TensorRT CUDA engine. Please wait for a few minutes...
[mmros_detection2d_exe-1] [I] [TRT] Detected 1 inputs and 2 output network tensors.
[mmros_detection2d_exe-1] [I] [TRT] Total Host Persistent Memory: 809504 bytes
[mmros_detection2d_exe-1] [I] [TRT] Total Device Persistent Memory: 0 bytes
[mmros_detection2d_exe-1] [I] [TRT] Max Scratch Memory: 19463168 bytes
[mmros_detection2d_exe-1] [I] [TRT] [BlockAssignment] Started assigning block shifts. This will take 366 steps to complete.
[mmros_detection2d_exe-1] [I] [TRT] [BlockAssignment] Algorithm ShiftNTopDown took 14.3722ms to assign 13 blocks to 366 nodes requiring 76321280 bytes.
[mmros_detection2d_exe-1] [I] [TRT] Total Activation Memory: 76320256 bytes
[mmros_detection2d_exe-1] [I] [TRT] Total Weights Memory: 233261568 bytes
[mmros_detection2d_exe-1] [I] [TRT] Compiler backend is used during engine execution.
[mmros_detection2d_exe-1] [I] [TRT] Engine generation completed in 42.7299 seconds.
[mmros_detection2d_exe-1] [I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 18 MiB, GPU 227 MiB
[mmros_detection2d_exe-1] [I] [TRT] Loaded engine size: 228 MiB
[mmros_detection2d_exe-1] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +1, GPU +73, now: CPU 1, GPU 295 (MiB)
[mmros_detection2d_exe-1] [I] [TRT] The profiling verbosity was set to ProfilingVerbosity::kLAYER_NAMES_ONLY when the engine was built, so only the layer names will be returned. Rebuild the engine with ProfilingVerbosity::kDETAILED to get more verbose layer information.
[mmros_detection2d_exe-1] [I] [TRT] Engine build completed
[mmros_detection2d_exe-1] [I] [TRT] Engine setup completed
[mmros_detection2d_exe-1] [W] [TRT] Network IO is empty, skipping validation. It might lead to undefined behavior
[mmros_detection2d_exe-1] [INFO 1753889849.167994867] [yolox.detector]: TensorRT engine file is built and exit.
```
