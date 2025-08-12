# 3D Sparse Convolution Network
An opensource sparse inference engine for [3d sparse convolutional networks](https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/backbones/scn.py) based on [traveller59/spconv](https://github.com/traveller59/spconv) using int8/fp16.

An opensource library [libscn.so](#compile-and-run) is created which can be used as drop in replacement to libspconv.so

## Model && Data
Use the same model and dataset as mentioned in original [README](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/libraries/3DSparseConvolution/README.md)

## Accuracy on nuScenes Validation
TBD on new library

## Memory Usage
TBD on new library

## Install Pre-Requisities
1. libprotobuf_dev=3.7.0
2. Build traveller59/spconv libspconv.so

    2.1 Modify the [build_deps](./build_deps.sh) for your env
    ```
    export CUMM_CUDA_VERSION=12.0 # cuda version, required but only used for flag selection when build libspconv.
    export CUMM_CUDA_ARCH_LIST="7.5;8.6"
    ```
    2.2 Running build_deps.sh will install and build the necessary dependencies
    ```
    ./build_deps.sh
    ```

## Compile and Run
- To build the library libscn.so
```
$ cd path/to/New3DSparseConvolution
make all
```
- Build and run test
```
$ cd path/to/New3DSparseConvolution
$ make fp16 -j
🙌 Output.shape: 1 x 256 x 180 x 180
[PASSED 🤗], libspconv version is 1.0.0
To verify the results, you can execute the following command.
Verify Result:
  python tool/compare.py workspace/bevfusion/infer.xyz.dense workspace/bevfusion/output.xyz.dense --detail
[PASSED].
```

- Verify output
```
$ python tool/compare.py workspace/bevfusion/infer.xyz.dense workspace/bevfusion/output.xyz.dense --detail
================ Compare Information =================
 CPP     Tensor: 1 x 256 x 180 x 180, float16 : workspace/bevfusion/infer.xyz.dense
 PyTorch Tensor: 1 x 256 x 180 x 180, float16 : workspace/bevfusion/output.xyz.dense
[absdiff]: max:0.02734375, sum:527.194580, std:0.000441, mean:0.000064
CPP:   absmax:11.164062, min:0.000000, std:0.117200, mean:0.015906
Torch: absmax:11.148438, min:0.000000, std:0.117174, mean:0.015901
[absdiff > m75% --- 0.020508]: 0.000 %, 16
[absdiff > m50% --- 0.013672]: 0.002 %, 161
[absdiff > m25% --- 0.006836]: 0.046 %, 3823
[absdiff > 0]: 3.816 %, 316479
[absdiff = 0]: 96.184 %, 7977921
[cosine]: 99.999 %
======================================================
```

## For Python
```
$ make pyscn -j
Use Python Include: /usr/include/python3.8
Use Python SO Name: python3.8
Use Python Library: /usr/lib
Compile CXX src/pyscn.cpp
Link tool/pyscn.so
You can run "python tool/pytest.py" to test

$ python tool/pytest.py
[PASSED 🤗].
To verify result:
  python tool/compare.py workspace/centerpoint/out_dense.py.fp16.tensor workspace/centerpoint/out_dense.torch.fp16.tensor --detail
```

## Performance on ORIN
TBD on new library

## Note
- Supported operators:
  - SparseConvolution, Add, Relu, Add&Relu and ScatterDense&Reshape&Transpose.
- Supported SparseConvolution:
  - SpatiallySparseConvolution and SubmanifoldSparseConvolution.
- Supported properties of SparseConvolution:
  - activation, kernel_size, dilation, stride, padding, rulebook, subm, output_bound, precision and output_precision.