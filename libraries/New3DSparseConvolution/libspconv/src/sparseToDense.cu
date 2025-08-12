#include <cuda_fp16.h>

#include <iostream>
#include <numeric>
#include <sstream>

#include "sparseToDense.h"
#include "spconv/check.hpp"
#include "spconv/launch.cuh"

template <typename T>
__global__ void sparseToDense_kernel(size_t num_indices, T* features_data,
                                     int* indices_data, T* feature_map,
                                     bool isXYZ, int strideb, int stridec,
                                     int stridex, int stridey, int stridez,
                                     int out_channels) {
  int idx = cuda_linear_index;
  if (idx >= num_indices) return;
  int b, x, y, z;
  if (isXYZ) {
    b = indices_data[idx * 4 + 0];
    x = indices_data[idx * 4 + 1];
    y = indices_data[idx * 4 + 2];
    z = indices_data[idx * 4 + 3];
  } else {
    b = indices_data[idx * 4 + 0];
    z = indices_data[idx * 4 + 1];
    y = indices_data[idx * 4 + 2];
    x = indices_data[idx * 4 + 3];
  }
  for (int i = 0; i < out_channels; i++) {
    feature_map[b * strideb + i * stridec + z * stridez + x * stridex +
                y * stridey] = features_data[idx * out_channels + i];
  }
}

SparseToDense::SparseToDense(std::string input_name, std::string output_name,
                             std::string format, std::vector<int> spatial_shape,
                             std::vector<int> output_spatial_shape,
                             spconv::TensorLayout output_layout)
    : input_name_(input_name),
      output_name_(output_name),
      format_(format),
      spatial_shape_(spatial_shape),
      output_spatial_shape_(output_spatial_shape),
      output_layout_(output_layout) {
  // TODO: do some checks
  int64_t total_volume = std::accumulate(output_spatial_shape_.begin(),
                                        output_spatial_shape_.end(), int64_t(1),
                                        std::multiplies<int64_t>());
  out_channels_ = output_spatial_shape[1];
  output_size_ = total_volume * sizeof(half);
  checkRuntime(cudaMalloc(&output_, output_size_));
  stride_.resize(output_spatial_shape_.size());
  for (int i = 0; i < output_spatial_shape_.size(); i++) { 
    auto len = output_spatial_shape_[i];
    total_volume /= len;
    stride_[i] = total_volume;
  }
}

SparseToDense::~SparseToDense() {
  if (output_) checkRuntime(cudaFree(output_));
}

void SparseToDense::configure(
    spconv::Precision precision,
    std::unordered_map<std::string, float>& tensor_name_to_scale,
    std::unordered_map<std::string, std::shared_ptr<void>> parameters) {
  std::shared_ptr<std::vector<int64_t>> permute;
  for (auto& [key, value] : parameters) {
    if (key == "dims") {
      permute = std::static_pointer_cast<std::vector<int64_t>>(value);
      /*
      the output layout is NCHW output tensor [vb*b+vc*c+vx*x+vy*y+vz*z] then
        if format is XYZ
        (0, 1, 2, 3, 4)->(b, C, X, Y, Z)->(vb, vc, vx, vy, vz)
        vb = stridec*stridex*stridey*stridez
        vc = stridex*stridey*stridez
        vx = stridey*stridez
        vy = stridez
        vz=1
        else format is zyx
        (0, 1, 2, 3, 4)->(b, C, Z, Y, X)->(vb, vc, vz, vy, vx)
        vb = stridec*stridex*stridey*stridez
        vc = stridex*stridey*stridez
        vz = stridex*stridey
        vy = stridex
        vx=1
        transpose (0, 1, 4, 2, 3)
        vx = stridey, vy=1, vz = stridex*stridey
      */
      // tv::ssprint("permute", *permute);
      std::vector<int> _output_spatial_shape;
      stride_.resize(permute->size());
      int64_t total_volume =
          std::accumulate(output_spatial_shape_.begin(),
                          output_spatial_shape_.end(), int64_t(1),
                          std::multiplies<int64_t>());
      for (int i = 0; i < permute->size(); i++) { 
        auto len = output_spatial_shape_[permute->at(i)];
        total_volume /= len;
        stride_[permute->at(i)] = total_volume;
        _output_spatial_shape.push_back(len);
      }
      // tv::ssprint("strides", stride_);
      output_spatial_shape_ = _output_spatial_shape;
    }
    else if (key == "reshape") {
    }
  }
}

void SparseToDense::forward(
    std::unordered_map<std::string, std::shared_ptr<spconv::SparseDTensor>> &io_dict,
    void* stream) {
  // find the inputs
  auto input_iterator = io_dict.find(input_name_);
  THROW_COND_EXCEPTION((input_iterator != io_dict.end()), std::runtime_error, 
                       "Cannot find input", input_name_);

  auto input = input_iterator->second;

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  checkRuntime(cudaMemsetAsync(output_, 0, output_size_, _stream));
  auto input_shape = input->features().shape;
  int64_t num_indices = input_shape[0];
  // check coordinate order
  bool isXYZ = format_ == "xyz";
  int strideb, stridec, stridex, stridey, stridez;
  std::vector<int64_t> feature_map_shape;
  if (isXYZ) {
    strideb = stride_[0];
    stridec = stride_[1];
    stridex = stride_[2];
    stridey = stride_[3];
    stridez = stride_[4];
    feature_map_shape.insert(feature_map_shape.begin(),
                             {1, out_channels_ * spatial_shape_[2],
                              spatial_shape_[0], spatial_shape_[1]});
  } else {
    strideb = stride_[0];
    stridec = stride_[1];
    stridez = stride_[2];
    stridey = stride_[3];
    stridex = stride_[4];
    feature_map_shape.insert(feature_map_shape.begin(),
                             {1, out_channels_ * spatial_shape_[0],
                              spatial_shape_[1], spatial_shape_[2]});
  }
  // launch kernel
  cuda_linear_launch(sparseToDense_kernel<half>, _stream, num_indices,
                     reinterpret_cast<half*>(input->features().ptr()),
                     reinterpret_cast<int*>(input->indices().ptr()),
                     reinterpret_cast<half*>(output_), isXYZ, strideb, stridec,
                     stridex, stridey, stridez, out_channels_);
//  auto out = std::make_shared<spconv::SparseDTensorImplement>(output_name_);
  auto output_iterator = io_dict.find(output_name_);
  THROW_COND_EXCEPTION((output_iterator != io_dict.end()), std::out_of_range, 
                           "cannot find output in io_dict", output_name_);
  auto out = output_iterator->second;
  out->features().reference((void*)output_, feature_map_shape, input->features().dtype());
  out->indices().reference(input->indices().ptr(), input->indices().shape, input->indices().dtype());
  out->set_grid_size(spatial_shape_);
  // out->set_device(input->device());
  // io_dict[output_name_] = out ;
}
