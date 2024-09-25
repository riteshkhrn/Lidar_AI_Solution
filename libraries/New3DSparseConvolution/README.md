# 3D Sparse Convolution Network
An opensource sparse inference engine for [3d sparse convolutional networks](https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/backbones/scn.py) based on [libspconv](https://github.com/traveller59/spconv) using int8/fp16.

## Model && Data
Use the same model and dataset as mentioned in original [README](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/libraries/3DSparseConvolution/README.md)

## Accuracy on nuScenes Validation
TBD on new library

## Memory Usage
TBD on new library

## Install Pre-Requisities
1. Install cumm==0.4.11 from source
2. Install spconv==2.3.6 from source(This directory would be **PATH_TO_INSTALLED_SPCONV**)
3. Build [libspconv.so](https://github.com/traveller59/spconv/blob/v2.3.6/example/libspconv/run_build.sh)
4. libprotobuf_dev==3.6.1

## Export ONNX
1. Download and configure the CenterPoint environment from https://github.com/tianweiy/CenterPoint
2. Export SCN ONNX
```
$ cp -r tool/centerpoint-export path/to/CenterPoint
$ cd path/to/CenterPoint
$ python centerpoint-export/export-scn.py --ckpt=epoch_20.pth --save-onnx=scn.nuscenes.onnx
$ cp scn.nuscenes.onnx path/to/3DSparseConvolution/workspace/
```

3. ## Compile && Run
- Change the path to libspconv.so in Makefile
```
include_paths := -Isrc -Ilibspconv/include -Ilibspconv/src -I$(CUDA_HOME)/include -I<PATH_TO_INSTALLED_SPCONV>/example/libspconv/spconv/include
link_flags    := -Llibspconv/lib/$(arch) -lscn -L$(CUDA_HOME)/lib64 -lcudart -lstdc++ -ldl -pthread -lprotobuf \
                -L<PATH_TO_INSTALLED_SPCONV>/example/libspconv/build/spconv/src -lspconv \
                -fopenmp -Wl,-rpath='$$ORIGIN' -Wl,-rpath=$(pwd)/libspconv/lib/$(arch) -Wl,-rpath=<PATH_TO_INSTALLED_SPCONV>/example/libspconv/build/spconv/src

```
- Build and run test
```
$ sudo apt-get install libprotobuf-dev
$ cd path/to/3DSparseConvolution
->>>>>> modify main.cpp:80 to scn.nuscenes.onnx
$ make fp16 -j
🙌 Output.shape: 1 x 256 x 180 x 180
[PASSED 🤗], libspconv version is 1.0.0
To verify the results, you can execute the following command.
Verify Result:
  python tool/compare.py workspace/centerpoint/out_dense.torch.fp16.tensor workspace/centerpoint/output.zyx.dense --detail
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