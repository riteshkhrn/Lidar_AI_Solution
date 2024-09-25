#ifndef _SPARSE_CONV_IMPLICIT_H_
#define _SPARSE_CONV_IMPLICIT_H_

#include <cuda_runtime_api.h>
#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/all/ops3d/Point2Voxel.h>
#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>
#include <spconvlib/spconv/csrc/sparse/convops/gemmops/GemmTunerSimple.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>
#include <spconvlib/spconv/csrc/sparse/inference/InferenceOps.h>
#include <tensorview/io/jsonarray.h>
#include <tensorview/parallel/map.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "spconv/common.hpp"  // for the DTensorImplementation definition
#include "spconv/engine.hpp"  // for the DTensor definition

using StaticAllocator = spconvlib::spconv::csrc::sparse::alloc::StaticAllocator;
using SpconvOps = spconvlib::spconv::csrc::sparse::all::SpconvOps;
using ConvMain = spconvlib::cumm::conv::main::ConvMainUnitTest;
using ConvTunerSimple =
    spconvlib::spconv::csrc::sparse::convops::spops::ConvTuner;
using ConvGemmOps =
    spconvlib::spconv::csrc::sparse::convops::spops::ConvGemmOps;
using SimpleExternalSpconvMatmul =
    spconvlib::spconv::csrc::sparse::convops::SimpleExternalSpconvMatmul;
using InferenceOps = spconvlib::spconv::csrc::sparse::inference::InferenceOps;
using Point2VoxelGPU3D =
    spconvlib::spconv::csrc::sparse::all::ops3d::Point2Voxel;
using GemmMain = spconvlib::cumm::gemm::main::GemmMainUnitTest;
using GemmTunerSimple =
    spconvlib::spconv::csrc::sparse::convops::spops::GemmTuner;

class SparseConvolution : public spconv::Operation {
 public:
  SparseConvolution(
      std::string input_name, std::string input_precision,
      std::string output_name, std::string output_precision, void* weights,
      std::vector<int> weight_shape, void* bias, std::vector<int> bias_shape,
      int ndim, std::vector<int> input_spatial_shape,
      std::vector<int> output_spatial_shape, int in_channels, int out_channels,
      std::vector<int> kernel_size, int output_bound, std::vector<int> stride,
      std::vector<int> dilation, std::vector<int> padding, int transposed,
      int inverse, std::vector<int> output_padding, int groups, int subm,
      std::string rulebook, std::string activation,
      std::vector<int> input_shape, std::vector<int> output_shape,
      float input_dynamic_range, std::vector<float> weight_dynamic_range);
  ~SparseConvolution();
  void set_precision(
      spconv::Precision precision,
      std::unordered_map<std::string, float>& tensor_name_to_scale);
  void forward(std::unordered_map<std::string,
                                  std::shared_ptr<spconv::DTensor>>& io_dict,
               void* stream);

 private:
  std::string input_name_;
  std::string output_name_;
  std::string input_precision_;
  std::string output_precision_;
  int ndim_;
  std::vector<int> input_spatial_shape_;
  std::vector<int> output_spatial_shape_;
  int in_channels_;
  int out_channels_;
  std::vector<int> kernel_size_;
  int output_bound_;
  std::vector<int> stride_;
  std::vector<int> dilation_;
  std::vector<int> padding_;
  int transposed_;
  int inverse_;
  std::vector<int> output_padding_;
  int groups_;
  int subm_;
  std::string rulebook_;
  std::string activation_;
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
  float input_dynamic_range_;
  std::vector<float> weight_dynamic_range_;
  float input_scale_;
  float output_scale_;
  bool int8_inference_;
  // spconv2
  tv::gemm::SparseConvAlgo conv_algo_;
  int out_inds_num_limit_;
  int static_num_act_in_;
  int max_act_out_theory_;
  bool use_direct_table_;
  // tensor view
  tv::Tensor weights_;
  tv::Tensor bias_;
  tv::Tensor workspace_;
  tv::Tensor pair_fwd_padded_;
  tv::Tensor pair_mask_fwd_padded_;
  tv::Tensor mask_argsort_fwd_padded_;
  tv::Tensor pair_bwd_padded_;
  tv::Tensor pair_mask_bwd_padded_;
  tv::Tensor mask_argsort_bwd_padded_;
  tv::Tensor indices_kernel_num_;
  tv::Tensor mask_tensor_;
  tv::Tensor out_inds_;
  tv::Tensor out_features_;
  tv::Tensor scale_;
  tv::DType output_dtype_;

  std::tuple<int, int> arch_;
  StaticAllocator* alloc_;
  ConvTunerSimple* conv_tuner_;
  tv::Context ctx_;
};

#endif  //_SPARSE_CONV_IMPLICIT_H_