#include <cuda_fp16.h>

#include <iostream>
#include <numeric>
#include <sstream>

#include "sparseFusedAddRelu.h"
#include "spconv/check.hpp"
#include "spconv/launch.cuh"

static __device__ int clamp(int x, int a, int b) { return max(a, min(b, x)); }

static __device__ int quantize(float inp, float scale) {
  return clamp(__float2int_rn(inp / scale), -128, 127);
}

template <typename Tin, typename Tout>
static __global__ void add_relu_kernel(size_t num_indices, Tin* input_1,
                                       float input_scale_1, Tin* input_2,
                                       float input_scale_2, Tout* output,
                                       float output_scale) {
  int idx = cuda_linear_index;
  if (idx >= num_indices) return;
  Tout out = input_1[idx] + input_2[idx];
  output[idx] = out > (Tout)0. ? out : (Tout)0.;
}

template <>
__global__ void add_relu_kernel<int8_t, int8_t>(
    size_t num_indices, int8_t* input_1, float input_scale_1, int8_t* input_2,
    float input_scale_2, int8_t* output, float output_scale) {
  int idx = cuda_linear_index;
  if (idx >= num_indices) return;
  // add
  float out = input_1[idx] * input_scale_1 + input_2[idx] * input_scale_2;
  // relu
  out = out > 0. ? out : 0.;
  output[idx] = quantize(out, output_scale);
}

template <>
__global__ void add_relu_kernel<half, int8_t>(
    size_t num_indices, half* input_1, float input_scale_1, half* input_2,
    float input_scale_2, int8_t* output, float output_scale) {
  int idx = cuda_linear_index;
  if (idx >= num_indices) return;
  // add
  float out = input_1[idx] + input_2[idx];
  // relu
  out = out > 0. ? out : 0.;
  output[idx] = quantize(out, output_scale);
}

SparseFusedAddRelu::SparseFusedAddRelu(std::string input_name_0,
                                       std::string input_name_1,
                                       std::string input_precision,
                                       std::string output_name,
                                       std::string output_precision,
                                       int output_bound, int out_channels)
    : input_name_1_(input_name_0),
      input_name_2_(input_name_1),
      input_precision_(input_precision),
      output_name_(output_name),
      output_precision_(output_precision),
      output_bound_(output_bound),
      out_channels_(out_channels) {
  // TODO: do some checks
  THROW_COND_EXCEPTION(
      std::runtime_error,
      (input_precision_ == "int8" || input_precision_ == "fp16"),
      "The input precision", input_precision_,
      "is not valid, the supported choices are int8 or fp16");
  THROW_COND_EXCEPTION(
      std::runtime_error,
      (output_precision_ == "int8" || output_precision_ == "fp16"),
      "The output precision", output_precision_,
      "is not valid, the supported choices are int8 or fp16");
  // THROW_COND_EXCEPTION(std::runtime_error,
  //                      (input_precision_ == output_precision_),
  //                      "The I/O precision is different",
  //                      input_precision_, output_precision_);
  // allocate the maximum size
  auto output_size = output_bound_ * out_channels_ * sizeof(half);
  checkRuntime(cudaMalloc(&output_, output_size));
  input_scale_1_ = 1.0;
  input_scale_2_ = 1.0;
  output_scale_ = 1.0;
}

SparseFusedAddRelu::~SparseFusedAddRelu() {
  if (output_) checkRuntime(cudaFree(output_));
}

void SparseFusedAddRelu::set_precision(
    spconv::Precision precision,
    std::unordered_map<std::string, float>& tensor_name_to_scale) {
  // should do int8 inference
  int8_inference_ = precision == spconv::Precision::Int8;
  int8_inference_ = int8_inference_ && !(input_precision_ == "fp16" &&
                                         output_precision_ == "fp16");
  LOG("Doing int8 inference", int8_inference_);
  if (!int8_inference_ && output_precision_ == "int8") {
    LOG("Ignoring node output precisions ", output_precision_,
        "cause model precision is not int8 reset it to fp16");
  }
  if (!int8_inference_ && input_precision_ == "int8") {
    LOG("Ignoring node input precisions", input_precision_,
        "cause model precision is not int8 reset it to fp16");
  }
  // if not int8 nothing to do more
  if (!int8_inference_) return;
  auto input_1_scale_iterator = tensor_name_to_scale.find(input_name_1_);
  THROW_COND_EXCEPTION(std::out_of_range,
                       (input_1_scale_iterator != tensor_name_to_scale.end()),
                       input_name_1_, "scale not found");
  input_scale_1_ = input_1_scale_iterator->second;
  auto input_2_scale_iterator = tensor_name_to_scale.find(input_name_2_);
  THROW_COND_EXCEPTION(std::out_of_range,
                       (input_2_scale_iterator != tensor_name_to_scale.end()),
                       input_name_2_, "scale not found");
  input_scale_2_ = input_2_scale_iterator->second;
  auto output_scale_iterator = tensor_name_to_scale.find(output_name_);
  THROW_COND_EXCEPTION(std::out_of_range,
                       (output_scale_iterator != tensor_name_to_scale.end()),
                       output_name_, "scale not found");
  output_scale_ = output_scale_iterator->second;
  // input_scale_2_ = input_scale_1_;
  // LOG("input_scale_1_", input_scale_1_, "input_scale_2_",
  // input_scale_2_, "output_scale_", output_scale_);
}

void SparseFusedAddRelu::forward(
    std::unordered_map<std::string, std::shared_ptr<spconv::DTensor>>& io_dict,
    void* stream) {
  // find the inputs
  auto input_1_iterator = io_dict.find(input_name_1_);
  THROW_COND_EXCEPTION(std::out_of_range, (input_1_iterator != io_dict.end()),
                       input_name_1_, "input not found");
  auto input1 = input_1_iterator->second;
  auto input_2_iterator = io_dict.find(input_name_2_);
  THROW_COND_EXCEPTION(std::out_of_range, (input_2_iterator != io_dict.end()),
                       input_name_2_, "input not found");
  auto input2 = input_2_iterator->second;
  assert(input1->features_shape() == input2->features_shape());

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  auto output_dtype = input1->features_dtype();
  auto input_shape = input1->features_shape();
  int64_t num_indices = std::accumulate(input_shape.begin(), input_shape.end(),
                                        int64_t(1), std::multiplies<int64_t>());
  if (int8_inference_) {
    if (input_precision_ == "int8" && output_precision_ == "int8") {
      cuda_linear_launch(
          add_relu_kernel, _stream, num_indices,
          reinterpret_cast<int8_t*>(input1->features_data()), input_scale_1_,
          reinterpret_cast<int8_t*>(input2->features_data()), input_scale_2_,
          reinterpret_cast<int8_t*>(output_), output_scale_);
    } else if (input_precision_ == "fp16" && output_precision_ == "int8") {
      cuda_linear_launch(
          add_relu_kernel, _stream, num_indices,
          reinterpret_cast<half*>(input1->features_data()), input_scale_1_,
          reinterpret_cast<half*>(input2->features_data()), input_scale_2_,
          reinterpret_cast<int8_t*>(output_), output_scale_);
    } else {
      // throw error
    }
    output_dtype = spconv::DType::Int8;
  } else {
    cuda_linear_launch(
        add_relu_kernel, _stream, num_indices,
        reinterpret_cast<half*>(input1->features_data()), input_scale_1_,
        reinterpret_cast<half*>(input2->features_data()), input_scale_2_,
        reinterpret_cast<half*>(output_), output_scale_);
  }

  io_dict[output_name_] = std::make_shared<spconv::DTensorImplement>(
      input1->features_shape(), output_dtype, (void*)output_,
      input1->indices_shape(), input1->indices_dtype(), input1->indices_data(),
      input1->grid_size(), input1->device());
}
