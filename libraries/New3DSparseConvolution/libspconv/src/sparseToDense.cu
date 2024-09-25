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
                                     bool isXYZ, int sizex, int sizey,
                                     int sizez, int feature_map_channel) {
  int idx = cuda_linear_index;
  if (idx >= num_indices) return;
  int b, x, y, z;
  if (isXYZ) {
    b = indices_data[idx * 4 + 0];
    x = indices_data[idx * 4 + 1];
    y = indices_data[idx * 4 + 2];
    z = indices_data[idx * 4 + 3];
    for (int i = 0; i < feature_map_channel; i++) {
      feature_map[b * feature_map_channel * sizex * sizey * sizez +
                  i * sizez * sizex * sizey + z * sizex * sizey + x * sizey +
                  y] = features_data[idx * feature_map_channel + i];
    }
  } else {
    b = indices_data[idx * 4 + 0];
    z = indices_data[idx * 4 + 1];
    y = indices_data[idx * 4 + 2];
    x = indices_data[idx * 4 + 3];
    for (int i = 0; i < feature_map_channel; i++) {
      feature_map[b * feature_map_channel * sizex * sizey * sizez +
                  i * sizez * sizex * sizey + z * sizex * sizey + y * sizex +
                  x] = features_data[idx * feature_map_channel + i];
    }
  }
}

SparseToDense::SparseToDense(std::string input_name, std::string output_name,
                             std::string format, std::vector<int> spatial_shape,
                             int out_channels)
    : input_name_(input_name),
      output_name_(output_name),
      format_(format),
      spatial_shape_(spatial_shape),
      out_channels_(out_channels) {
  // TODO: do some checks
  int64_t spatial_volume =
      std::accumulate(spatial_shape_.begin(), spatial_shape_.end(), int64_t(1),
                      std::multiplies<int64_t>());
  output_size_ = spatial_volume * out_channels_ * sizeof(half);
  checkRuntime(cudaMalloc(&output_, output_size_));
}

SparseToDense::~SparseToDense() {
  if (output_) checkRuntime(cudaFree(output_));
}

void SparseToDense::forward(
    std::unordered_map<std::string, std::shared_ptr<spconv::DTensor>>& io_dict,
    void* stream) {
  // find the inputs
  auto input_iterator = io_dict.find(input_name_);
  THROW_COND_EXCEPTION(std::runtime_error, (input_iterator != io_dict.end()),
                       "Cannot find input", input_name_);

  auto input = input_iterator->second;

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  checkRuntime(cudaMemsetAsync(output_, 0, output_size_, _stream));
  auto input_shape = input->features_shape();
  int64_t num_indices = input_shape[0];
  // check coordinate order
  bool isXYZ = format_ == "xyz";
  int sizex, sizey, sizez;
  std::vector<int64_t> feature_map_shape;
  if (isXYZ) {
    sizex = spatial_shape_[0];
    sizey = spatial_shape_[1];
    sizez = spatial_shape_[2];
    feature_map_shape.insert(feature_map_shape.begin(),
                             {1, out_channels_ * spatial_shape_[2],
                              spatial_shape_[0], spatial_shape_[1]});
  } else {
    sizez = spatial_shape_[0];
    sizey = spatial_shape_[1];
    sizex = spatial_shape_[2];
    feature_map_shape.insert(feature_map_shape.begin(),
                             {1, out_channels_ * spatial_shape_[0],
                              spatial_shape_[1], spatial_shape_[2]});
  }

  cuda_linear_launch(sparseToDense_kernel<half>, _stream, num_indices,
                     reinterpret_cast<half*>(input->features_data()),
                     reinterpret_cast<int*>(input->indices_data()),
                     reinterpret_cast<half*>(output_), isXYZ, sizex, sizey,
                     sizez, out_channels_);
  io_dict[output_name_] = std::make_shared<spconv::DTensorImplement>(
      feature_map_shape, input->features_dtype(), (void*)output_,
      input->indices_shape(), input->indices_dtype(), input->indices_data(),
      input->grid_size(), input->device());
}
