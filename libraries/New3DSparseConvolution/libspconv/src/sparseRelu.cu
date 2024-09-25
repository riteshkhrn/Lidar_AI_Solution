#include <cuda_fp16.h>

#include <iostream>
#include <numeric>
#include <sstream>

#include "sparseRelu.h"
#include "spconv/check.hpp"
#include "spconv/launch.cuh"

template <typename T>
static __global__ void relu_kernel(size_t num_indices, T* input, T* output) {
  int idx = cuda_linear_index;
  if (idx >= num_indices) return;
  output[idx] = input[idx] > (T)0. ? input[idx] : (T)0.;
}

SparseRelu::SparseRelu(std::string input_name, std::string output_name,
                       int output_bound, int out_channels)
    : input_name_(input_name),
      output_name_(output_name),
      output_bound_(output_bound),
      out_channels_(out_channels) {
  // TODO: do some checks
  auto output_size = output_bound_ * out_channels_ * sizeof(half);
  checkRuntime(cudaMalloc(&output_, output_size));
}

SparseRelu::~SparseRelu() {
  if (output_) checkRuntime(cudaFree(output_));
}

void SparseRelu::set_precision(
    spconv::Precision precision,
    std::unordered_map<std::string, float>& tensor_name_to_scale) {
  // should do int8 inference
  int8_inference_ = precision == spconv::Precision::Int8;
  LOG("Doing int8 inference", int8_inference_);
}

void SparseRelu::forward(
    std::unordered_map<std::string, std::shared_ptr<spconv::DTensor>>& io_dict,
    void* stream) {
  // find the inputs
  auto input_iterator = io_dict.find(input_name_);
  THROW_COND_EXCEPTION(std::out_of_range, (input_iterator != io_dict.end()),
                       input_name_, "input not found");

  auto input = input_iterator->second;

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  auto input_shape = input->features_shape();
  int64_t num_indices = std::accumulate(input_shape.begin(), input_shape.end(),
                                        int64_t(1), std::multiplies<int64_t>());
  if (int8_inference_) {
    cuda_linear_launch(relu_kernel<int8_t>, _stream, num_indices,
                       reinterpret_cast<int8_t*>(input->features_data()),
                       reinterpret_cast<int8_t*>(output_));
  } else {
    cuda_linear_launch(relu_kernel<half>, _stream, num_indices,
                       reinterpret_cast<half*>(input->features_data()),
                       reinterpret_cast<half*>(output_));
  }

  io_dict[output_name_] = std::make_shared<spconv::DTensorImplement>(
      input->features_shape(), input->features_dtype(), (void*)output_,
      input->indices_shape(), input->indices_dtype(), input->indices_data(),
      input->grid_size(), input->device());
}
