#include "sparseConvImplicit.h"

tv::DType convert_to_tv_datatype(spconv::DType spconvType) {
  switch (spconvType) {
    case spconv::DType::Int32:
      return tv::int32;
    case spconv::DType::Float16:
      return tv::float16;
    case spconv::DType::Int8:
      return tv::int8;
    default:
      THROW_EXCEPTION(
          std::runtime_error, "The case of tv Data type",
          static_cast<std::underlying_type_t<spconv::DType>>(spconvType),
          "is not handled");
  }
}

spconv::DType convert_to_spconv_datatype(tv::DType tvType) {
  switch (tvType) {
    case tv::int32:
      return spconv::DType::Int32;
    case tv::float16:
      return spconv::DType::Float16;
    case tv::int8:
      return spconv::DType::Int8;
    default:
      THROW_EXCEPTION(std::runtime_error, "The case of tv Data type",
                      static_cast<std::underlying_type_t<tv::DType>>(tvType),
                      "is not handled");
  }
}

static __device__ int clamp(int x, int a, int b) { return max(a, min(b, x)); }

static __device__ int quantize(float inp, float scale) {
  return clamp(__float2int_rn(inp / scale), -128, 127);
}

SparseConvolution::SparseConvolution(
    std::string input_name, std::string input_precision,
    std::string output_name, std::string output_precision, void* weights,
    std::vector<int> weight_shape, void* bias, std::vector<int> bias_shape,
    int ndim, std::vector<int> input_spatial_shape,
    std::vector<int> output_spatial_shape, int in_channels, int out_channels,
    std::vector<int> kernel_size, int output_bound, std::vector<int> stride,
    std::vector<int> dilation, std::vector<int> padding, int transposed,
    int inverse_, std::vector<int> output_padding, int groups, int subm,
    std::string rulebook, std::string activation, std::vector<int> input_shape,
    std::vector<int> output_shape, float input_dynamic_range,
    std::vector<float> weight_dynamic_range)
    : input_name_(input_name),
      input_precision_(input_precision),
      output_name_(output_name),
      output_precision_(output_precision),
      ndim_(ndim),
      input_spatial_shape_(input_spatial_shape),
      output_spatial_shape_(output_spatial_shape),
      in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      output_bound_(output_bound),
      stride_(stride),
      dilation_(dilation),
      padding_(padding),
      transposed_(transposed),
      inverse_(inverse_),
      output_padding_(output_padding),
      groups_(groups),
      subm_(subm),
      rulebook_(rulebook),
      activation_(activation),
      input_shape_(input_shape),
      output_shape_(output_shape),
      input_dynamic_range_(input_dynamic_range),
      weight_dynamic_range_(weight_dynamic_range) {
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
  // output_bound_ = 100000;
  // weight layout is KRSC, [out channels, *kernel_size, in channels]
  weights_ = tv::from_blob(weights, weight_shape, tv::float16, -1).to(0);
  bias_ = tv::from_blob(bias, bias_shape, tv::float16, -1).to(0);
  // get_compute_capability is very slow, don't forget to cache arch result.
  arch_ = ConvGemmOps::get_compute_capability();
  conv_algo_ = tv::gemm::SparseConvAlgo::kMaskImplicitGemm;
  // conv_algo_ = tv::gemm::SparseConvAlgo::kMaskSplitImplicitGemm;
  // our algorithm support upper bound number of output indices.
  // in inference engine, the number of output points must have a upper bound.
  out_inds_num_limit_ = output_bound_;
  // static_num_act_in_ is just out_inds_num_limit_ of previous conv layer.
  static_num_act_in_ = output_bound_;
  int KV =
      kernel_size_[0] * kernel_size_[1] * kernel_size_[2];  // kernel volume
  // if shape is too large, we will use slower int64->int32 hash table instead
  // of int32->int32 table.
  int64_t out_spatial_volume = std::accumulate(
      output_spatial_shape_.begin(), output_spatial_shape_.end(), int64_t(1),
      std::multiplies<int64_t>());
  bool use_int64_hash_k =
      out_spatial_volume >= int64_t(std::numeric_limits<int>::max());
  // implicit gemm code example
  // direct table: a hash based algorithm that don't need unique. enabled
  // by default.
  bool direct_table = true;
  // only regular conv need direct table.
  use_direct_table_ = direct_table && !subm_;
  auto max_act_out_theory = SpconvOps::get_handcrafted_max_act_out(
      static_num_act_in_, kernel_size_, stride_, padding_, dilation_);
  // query workspace size.
  int workspace_size = SpconvOps::get_indice_gen_workspace_size(
      KV, static_num_act_in_, out_inds_num_limit_, max_act_out_theory, subm_,
      use_int64_hash_k, use_direct_table_);
  // you should return workspace size in tensorrt plugin method.
  workspace_ = tv::empty({workspace_size}, tv::uint8, 0);
  // pair can also have a upper bound.
  // !!!!!IMPORTANT!!!!!!! if you provide a static (padded) pair_fwd and
  // other indice data, the output layout is tight pair_fwd_correct =
  // pair_fwd_padded.view(-1)[:KV * real_pair_size].view(KV,
  // real_pair_size) this valid for pair_fwd, pair_bwd, pair_mask_fwd,
  // pair_mask_bwd, mask_argsort_fwd, mask_argsort_bwd.
  int pair_fwd_size_padded = subm_ ? static_num_act_in_ : out_inds_num_limit_;
  pair_fwd_padded_ = tv::empty({KV, pair_fwd_size_padded}, tv::int32, 0);
  // you can find equivalent python code of following code in python package
  bool is_split_mask =
      conv_algo_ == tv::gemm::SparseConvAlgo::kMaskSplitImplicitGemm;
  int mask_count = is_split_mask ? 2 : 1;
  pair_mask_fwd_padded_ =
      tv::empty({mask_count, pair_fwd_size_padded}, tv::int32, 0);
  mask_argsort_fwd_padded_ =
      tv::empty({mask_count, pair_fwd_size_padded}, tv::int32, 0);
  indices_kernel_num_ = tv::zeros({KV}, tv::int32, 0);
  out_inds_ =
      tv::empty({subm_ ? static_num_act_in_ : out_inds_num_limit_, ndim + 1},
                tv::int32, 0);
  mask_tensor_ = tv::zeros({pair_mask_fwd_padded_.dim(0)}, tv::uint32, -1);
  // get tensor map required by pair gen from workspace
  // keep in mind that our indice gen function use a "allocator" to alloc
  // temp/out tensors, in python we use TorchAllocator which is a simple
  // dynamic allocator, in c++ (inference engine) we need to use
  // fixed-size workspace and create a static allocator.
  auto ws_tensors = SpconvOps::get_indice_gen_tensors_from_workspace(
      workspace_.raw_data(), KV, static_num_act_in_,
      subm_ ? static_num_act_in_ : out_inds_num_limit_, max_act_out_theory,
      subm_, use_int64_hash_k, use_direct_table_);
  // create output tensors and insert them to static allocator
  // output tensors needed in subm get_indice_pairs_implicit_gemm,
  // saved to static allocator.
  ws_tensors.insert({SPCONV_ALLOC_PAIR_FWD, pair_fwd_padded_});
  ws_tensors.insert({SPCONV_ALLOC_PAIR_MASK, pair_mask_fwd_padded_});
  ws_tensors.insert({SPCONV_ALLOC_MASK_ARG_SORT, mask_argsort_fwd_padded_});
  ws_tensors.insert({SPCONV_ALLOC_OUT_INDICES, out_inds_});
  ws_tensors.insert({SPCONV_ALLOC_INDICE_NUM_PER_LOC, indices_kernel_num_});
  if (!subm_) {
    // WARNING be careful with inverse conv, understand python
    // code first. no inverse example here.
    // regular conv need more outputs, used for inversed conv.
    // bwd shape is [KV, static num_act_in (previous num_act_out_bound)]
    pair_bwd_padded_ = tv::empty({KV, static_num_act_in_}, tv::int32, 0);
    pair_mask_bwd_padded_ =
        tv::empty({mask_count, static_num_act_in_}, tv::int32, 0);
    mask_argsort_bwd_padded_ =
        tv::empty({mask_count, static_num_act_in_}, tv::int32, 0);
    ws_tensors.insert({SPCONV_ALLOC_PAIR_BWD, pair_bwd_padded_});
    ws_tensors.insert({SPCONV_ALLOC_PAIR_MASK_BWD, pair_mask_bwd_padded_});
    ws_tensors.insert(
        {SPCONV_ALLOC_MASK_ARG_SORT_BWD, mask_argsort_bwd_padded_});
  }
  alloc_ = new StaticAllocator(ws_tensors);
  conv_tuner_ = new ConvTunerSimple(ConvMain::get_all_conv_algo_desp());
}

SparseConvolution::~SparseConvolution() {
  if (alloc_) delete alloc_;
  if (conv_tuner_) delete conv_tuner_;
}

void SparseConvolution::set_precision(
    spconv::Precision precision,
    std::unordered_map<std::string, float>& tensor_name_to_scale) {
  output_scale_ = 1.0;
  input_scale_ = 1.0;
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
  output_dtype_ = int8_inference_
                      ? output_precision_ == "int8" ? tv::int8 : tv::float16
                      : tv::float16;
  out_features_ = tv::empty(
      {subm_ ? static_num_act_in_ : out_inds_num_limit_, out_channels_},
      output_dtype_, 0);
  // if not int8 nothing to do more
  if (!int8_inference_) return;
  // if int8 need to initialize weights, bias, scale
  // input_dynamic_range and weight_dynamic_range are the QAMAX
  // refer
  // https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/qat/lean/exptool.py#L179
  // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#intro-quantization
  if (output_precision_ == "int8") {
    auto output_scale_iterator = tensor_name_to_scale.find(output_name_);
    THROW_COND_EXCEPTION(std::out_of_range,
                         (output_scale_iterator != tensor_name_to_scale.end()),
                         output_name_, "scale not found");
    output_scale_ = output_scale_iterator->second;
  }
  auto input_scale_iterator = tensor_name_to_scale.find(input_name_);
  THROW_COND_EXCEPTION(std::out_of_range,
                       (input_scale_iterator != tensor_name_to_scale.end()),
                       input_name_, "scale not found");
  input_scale_ = input_scale_iterator->second;
  // LOG("input scale", input_scale_, "output scale", output_scale_);
  std::vector<int> weights_scale_shape{(int)weight_dynamic_range_.size()};
  // weights_scale_shape.push_back(weight_dynamic_range_.size());
  auto weights_scale = tv::from_blob(weight_dynamic_range_.data(),
                                     weights_scale_shape, tv::float32, -1)
                           .to(0);
  auto bias_fp32 = tv::empty(weights_scale_shape, tv::float32, 0);
  scale_ = tv::empty(weights_scale_shape, tv::float32, 0);
  // tv::ssprint("Weights scale ", weights_scale.shape(), " scale ",
  // scale_.shape());
  auto weights_scale_ptr = weights_scale.data_ptr<float>();
  auto scale_ptr = scale_.data_ptr<float>();
  auto bias_fp32_ptr = bias_fp32.data_ptr<float>();
  auto bias_ptr = bias_.data_ptr<__half>();
  float output_scale = output_scale_;
  float input_scale = input_scale_;
  tv::kernel_1d_map_cuda(
      0, weight_dynamic_range_.size(),
      [=] TV_GPU_LAMBDA_DEVICE(size_t i) mutable {
        weights_scale_ptr[i] = weights_scale_ptr[i] / 127;
        // https://github.com/traveller59/spconv/blob/master/docs/TENSORRT_INT8_GUIDE.md
        scale_ptr[i] = (weights_scale_ptr[i] * input_scale) / output_scale;
        bias_fp32_ptr[i] = __half2float(bias_ptr[i]) / output_scale;
      });
  bias_ = bias_fp32;

  auto weight_shape_iter = weights_.shape().begin();
  int weight_shape_0 = *weight_shape_iter;
  std::advance(weight_shape_iter, 1);
  int64_t weight_volume =
      std::accumulate(weight_shape_iter, weights_.shape().end(), int64_t(1),
                      std::multiplies<int64_t>());
  auto weights_int8 = tv::empty(weights_.shape(), tv::int8, 0);
  auto weights_int8_ptr = weights_int8.data_ptr<int8_t>();
  auto weights_ptr = weights_.data_ptr<__half>();
  tv::kernel_1d_map_cuda(
      0, weight_shape_0, [=] TV_GPU_LAMBDA_DEVICE(size_t i) mutable {
        for (int j = 0; j < weight_volume; j++) {
          int index = j + weight_volume * (i);
          weights_int8_ptr[index] =
              quantize(__half2float(weights_ptr[index]), weights_scale_ptr[i]);
        }
      });
  weights_ = weights_int8;
  // auto indices_cpu = weights_.cpu();
  // auto indices_cpu_data_ptr = indices_cpu.data_ptr<int8_t>();
  // for (int i = 0; i < 20; ++i) {
  //   auto cur_indices_cpu_data_ptr = indices_cpu_data_ptr + i * 4;
  //   tv::ssprint((int)cur_indices_cpu_data_ptr[0],
  //   (int)cur_indices_cpu_data_ptr[1],
  //               (int)cur_indices_cpu_data_ptr[2],
  //               (int)cur_indices_cpu_data_ptr[3]);
  // }
}

void SparseConvolution::forward(
    std::unordered_map<std::string, std::shared_ptr<spconv::DTensor>>& io_dict,
    void* stream) {
  float output_scale = output_scale_;
  float input_scale = input_scale_;
  // set the activation type
  tv::gemm::Activation act_type = tv::gemm::Activation::kNone;
  if (activation_ == "ReLU") act_type = tv::gemm::Activation::kReLU;
  // get feature tensor
  auto input_iterator = io_dict.find(input_name_);
  THROW_COND_EXCEPTION(std::out_of_range, (input_iterator != io_dict.end()),
                       input_name_, "input not found");
  auto input = input_iterator->second;
  auto input_dtype = int8_inference_
                         ? input_precision_ == "int8" ? tv::int8 : tv::float16
                         : tv::float16;
  auto tv_dtype = convert_to_tv_datatype(input->features_dtype());
  // tv::ssprint(int8_inference_, input_precision_, input_dtype, tv_dtype);
  THROW_COND_EXCEPTION(std::runtime_error, (input_dtype == tv_dtype),
                       "Invalid input type");
  tv::Tensor input_features = tv::Tensor(
      input->features_data(), input->features_shape(), input_dtype, 0);
  if (int8_inference_ && input_precision_ == "fp16") {
    LOG("Converting input from fp16 to int8");
    tv::Tensor input_features_int8 =
        tv::empty(input->features_shape(), tv::int8, 0);
    auto input_features_ptr = input_features.data_ptr<__half>();
    auto input_features_int8_ptr = input_features_int8.data_ptr<int8_t>();
    tv::kernel_1d_map_cuda(
        0, input_features.size(), [=] TV_GPU_LAMBDA_DEVICE(size_t i) mutable {
          input_features_int8_ptr[i] =
              quantize(__half2float(input_features_ptr[i]), input_scale);
        });
    input_features = input_features_int8;
  }
  // for regular conv, the input tensor has static shape, we should save a CPU
  // variable of real num_act_out. here we just use num_act_in.
  int real_num_features = input_features.dim(0);
  // get input indices
  tv::Tensor input_indices =
      tv::Tensor(input->indices_data(), input->indices_shape(), tv::int32, 0);
  // you need to slice all static inputs with real_num_act_in in
  // static inference engine, e.g. tensorrt. here we don't
  // need to do that.
  int batch_size = 1;
  // we shouldn't use standard transpose in any code.
  // int transpose = false;
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  if (!ctx_.has_cuda_stream()) {
    ctx_.set_cuda_stream_int(reinterpret_cast<std::uintptr_t>(_stream));
  }
  int KV =
      kernel_size_[0] * kernel_size_[1] * kernel_size_[2];  // kernel volume
  // https://github.com/traveller59/spconv/blob/master/docs/INT8_GUIDE.md#performance-guide
  bool do_sort = !int8_inference_;
  std::tuple<tv::Tensor, int> pair_res;
  pair_res = SpconvOps::get_indice_pairs_implicit_gemm(
      *alloc_, input_indices, batch_size, input_spatial_shape_,
      static_cast<int>(conv_algo_), kernel_size_, stride_, padding_, dilation_,
      {0, 0, 0}, subm_, transposed_, false /*is_train*/,
      reinterpret_cast<std::uintptr_t>(stream), out_inds_num_limit_,
      tv::CUDAKernelTimer(false), use_direct_table_, do_sort);

  // after get pair datas, we can start to do real convolution!
  // in static inference engine, you need to split pair-gen and conv to
  // different layers to reuse pair data
  // here we just use previous result.
  int num_act_out_real = std::get<1>(pair_res);

  auto out_features_real = out_features_.slice_first_axis(0, num_act_out_real);
  auto out_indices_real = out_inds_.slice_first_axis(0, num_act_out_real);

  bool is_mask_split = pair_mask_fwd_padded_.dim(0) > 1;
  int mask_split_cnt = pair_mask_fwd_padded_.dim(0);
  // tv::Tensor mask_tensor =
  //     tv::zeros({pair_mask_fwd_padded.dim(0)}, tv::uint32, -1);
  // create split mask
  // currently it's a constant for each algo.
  auto mask_tensor_ptr = mask_tensor_.data_ptr<uint32_t>();
  if (is_mask_split) {
    auto kv_div_2 = KV / 2;
    auto remain = KV - kv_div_2;
    uint64_t mask_np_1 = 1;
    uint64_t first = ((mask_np_1 << remain) - 1);
    uint64_t second = ((mask_np_1 << kv_div_2) - 1) << remain;
    mask_tensor_ptr[0] = uint32_t(first);
    mask_tensor_ptr[1] = uint32_t(second);
  } else {
    mask_tensor_ptr[0] = 0xffffffff;
  }
  std::vector<tv::Tensor> pair_mask_splits;
  std::vector<tv::Tensor> mask_argsort_splits;
  // size_t real_pair_size = KV * num_act_out_real;
  // keep in mind that pair_mask_fwd_padded is tight padded tensor, so we
  // must get real tensor before use them if inversed conv, use xxxx_bwd
  // here instead of fwd.
  auto pair_fwd_real = pair_fwd_padded_.view(-1)
                           .slice_first_axis(0, KV * num_act_out_real)
                           .view(KV, num_act_out_real);
  auto pair_mask_fwd_real =
      pair_mask_fwd_padded_.view(-1)
          .slice_first_axis(0, mask_split_cnt * num_act_out_real)
          .view(mask_split_cnt, num_act_out_real);
  auto mask_argsort_fwd_real =
      mask_argsort_fwd_padded_.view(-1)
          .slice_first_axis(0, mask_split_cnt * num_act_out_real)
          .view(mask_split_cnt, num_act_out_real);

  for (int i = 0; i < mask_split_cnt; ++i) {
    pair_mask_splits.push_back(pair_mask_fwd_real[i]);
    mask_argsort_splits.push_back(mask_argsort_fwd_real[i]);
  }
  // create output tensor allocator
  std::unordered_map<std::string, tv::Tensor> tensor_dict{
      {SPCONV_ALLOC_FEATURES, input_features},
      {SPCONV_ALLOC_FILTERS, weights_},
      {SPCONV_ALLOC_OUT_FEATURES, out_features_}};
  StaticAllocator alloc2(tensor_dict);
  // ConvTunerSimple tuner(ConvMain::get_all_conv_algo_desp());
  // auto output_add = tv::Tensor();
  auto conv_run_status = ConvGemmOps::implicit_gemm(
      alloc2, *conv_tuner_, input_features, weights_, pair_fwd_real,
      pair_mask_splits, mask_argsort_splits, num_act_out_real, mask_tensor_,
      arch_, false, subm_, reinterpret_cast<std::uintptr_t>(stream),
      tv::CUDAKernelTimer(false), false, false, bias_,
      1.0 /*bias alpha, only used for leaky relu*/, 0.0 /*unused for now*/,
      act_type, false, output_scale, scale_, tv::Tensor(), 1.0, output_dtype_);

  // convert the tv::Tensor -> Dtensor
  std::vector<int64_t> output_shape{out_features_real.shape().begin(),
                                    out_features_real.shape().end()};
  std::vector<int64_t> output_indices_shape{out_indices_real.shape().begin(),
                                            out_indices_real.shape().end()};
  io_dict[output_name_] = std::make_shared<spconv::DTensorImplement>(
      output_shape, convert_to_spconv_datatype(out_features_real.dtype()),
      (void*)out_features_real.raw_data(), output_indices_shape,
      convert_to_spconv_datatype(out_indices_real.dtype()),
      subm_ ? (void*)input_indices.raw_data()
            : (void*)out_indices_real.raw_data(),
      output_spatial_shape_, out_features_real.device());

  // checkCudaErrors(cudaStreamSynchronize(_stream));
}
