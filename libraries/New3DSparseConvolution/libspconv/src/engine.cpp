
#include <cuda_runtime.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <spconv/engine.hpp>
#include <unordered_map>
#include <vector>

#include "onnx.proto3.pb.h"
#include "sparseAdd.h"
#include "sparseConvImplicit.h"
#include "sparseFusedAddRelu.h"
#include "sparseRelu.h"
#include "sparseToDense.h"
#include "spconv/common.hpp"
#include "spconv/tensor.hpp"

namespace spconv {

class EngineImplement : public Engine {
 public:
  std::vector<std::shared_ptr<Operation>> operations_;
  std::unordered_map<std::string, float> tensor_name_to_scale_;
  std::string graph_input_name_;
  std::string graph_output_name_;
  std::shared_ptr<DTensor> result_;

  virtual ~EngineImplement() = default;

  void print_dim(const ::onnx::TensorShapeProto_Dimension& dim) {
    switch (dim.value_case()) {
      case onnx::TensorShapeProto_Dimension::ValueCase::kDimParam:
        std::cout << dim.dim_param();
        break;
      case onnx::TensorShapeProto_Dimension::ValueCase::kDimValue:
        std::cout << dim.dim_value();
        break;
      default:
        assert(false && "should never happen");
    }
  }

  void print_io_info(
      const ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto>&
          info) {
    for (auto input_data : info) {
      auto shape = input_data.type().tensor_type().shape();
      std::cout << "  " << input_data.name() << ":";
      std::cout << "[";
      if (shape.dim_size() != 0) {
        int size = shape.dim_size();
        for (int i = 0; i < size - 1; ++i) {
          print_dim(shape.dim(i));
          std::cout << ", ";
        }
        print_dim(shape.dim(size - 1));
      }
      std::cout << "]\n";
    }
  }

  std::shared_ptr<Operation> addSparseToDenseNode(
      onnx::NodeProto node, onnx::NodeProto prev_spconv_node,
      std::unordered_map<std::string, onnx::TensorProto> initializer) {
    // get the input names
    std::string input_name = node.input()[0];
    // get the ouput names
    std::string outputName = node.output()[0];
    outputName = graph_output_name_;
    // get last node outbound and out channels
    int out_channels;
    std::string format;
    std::vector<int> input_spatial_shape;
    for (auto at : prev_spconv_node.attribute()) {
      if (at.name() == "out_channels") {
        out_channels = at.i();
      }
    }
    for (auto at : node.attribute()) {
      if (at.name() == "input_spatial_shape") {
        input_spatial_shape.insert(input_spatial_shape.begin(),
                                   at.ints().begin(), at.ints().end());
      } else if (at.name() == "format") {
        format = at.s();
      }
    }

    return std::make_shared<SparseToDense>(input_name, outputName, format,
                                           input_spatial_shape, out_channels);
  }

  std::shared_ptr<Operation> addReluNode(
      onnx::NodeProto node, onnx::NodeProto prev_spconv_node,
      std::unordered_map<std::string, onnx::TensorProto> initializer) {
    // get the input names
    std::string input_name = node.input()[0];
    // get the ouput names
    std::string outputName = node.output()[0];
    // get last node outbound and out channels
    int out_channels;
    int output_bound;
    for (auto at : prev_spconv_node.attribute()) {
      if (at.name() == "out_channels") {
        out_channels = at.i();
      } else if (at.name() == "output_bound") {
        output_bound = at.i();
      }
    }

    return std::make_shared<SparseRelu>(input_name, outputName, output_bound,
                                        out_channels);
  }

  std::shared_ptr<Operation> addAddNode(
      onnx::NodeProto node, onnx::NodeProto prev_spconv_node,
      std::unordered_map<std::string, onnx::TensorProto> initializer) {
    // get the input names
    std::string input_name_0 = node.input()[0];
    std::string input_name_1 = node.input()[1];
    // get the ouput names
    std::string outputName = node.output()[0];
    // get the precision and input range
    std::string input_precision = "fp16";
    std::string output_precision = "fp16";
    float input0_dynamic_range;
    float input1_dynamic_range;
    for (auto at : node.attribute()) {
      if (at.name() == "precision") {
        input_precision = at.s();
      } else if (at.name() == "output_precision") {
        output_precision = at.s();
      } else if (at.name() == "input0_dynamic_range") {
        input0_dynamic_range = at.f();
      } else if (at.name() == "input1_dynamic_range") {
        input1_dynamic_range = at.f();
      }
    }
    tensor_name_to_scale_[input_name_0] = input0_dynamic_range / 127;
    tensor_name_to_scale_[input_name_1] = input1_dynamic_range / 127;
    // get last node outbound and out channels
    int out_channels;
    int output_bound;
    for (auto at : prev_spconv_node.attribute()) {
      if (at.name() == "out_channels") {
        out_channels = at.i();
      } else if (at.name() == "output_bound") {
        output_bound = at.i();
      }
    }

    LOG("Adding node", node.name(), "of type", node.op_type(),
        "with I/O precision", input_precision, "/", output_precision);
    return std::make_shared<SparseAdd>(
        input_name_0, input_name_1, input_precision, outputName,
        output_precision, output_bound, out_channels);
  }

  std::shared_ptr<Operation> addFusedAddReluNode(
      onnx::NodeProto add_node, onnx::NodeProto relu_node,
      onnx::NodeProto prev_spconv_node,
      std::unordered_map<std::string, onnx::TensorProto> initializer) {
    // get the input names
    std::string input_name_0 = add_node.input()[0];
    std::string input_name_1 = add_node.input()[1];
    // get the ouput names
    std::string outputName = relu_node.output()[0];
    // get the precision and input range
    std::string input_precision = "fp16";
    std::string output_precision = "fp16";
    float input0_dynamic_range;
    float input1_dynamic_range;
    for (auto at : add_node.attribute()) {
      if (at.name() == "precision") {
        input_precision = at.s();
      } else if (at.name() == "output_precision") {
        output_precision = at.s();
      } else if (at.name() == "input0_dynamic_range") {
        input0_dynamic_range = at.f();
      } else if (at.name() == "input1_dynamic_range") {
        input1_dynamic_range = at.f();
      }
    }
    tensor_name_to_scale_[input_name_0] = input0_dynamic_range / 127;
    tensor_name_to_scale_[input_name_1] = input1_dynamic_range / 127;
    // get last node outbound and out channels
    int out_channels;
    int output_bound;
    for (auto at : prev_spconv_node.attribute()) {
      if (at.name() == "out_channels") {
        out_channels = at.i();
      } else if (at.name() == "output_bound") {
        output_bound = at.i();
      }
    }

    LOG("Adding node", add_node.name(), "of type", add_node.op_type(),
        "with I/O precision", input_precision, "/", output_precision);
    return std::make_shared<SparseFusedAddRelu>(
        input_name_0, input_name_1, input_precision, outputName,
        output_precision, output_bound, out_channels);
  }

  std::shared_ptr<Operation> addSpconvNode(
      onnx::NodeProto node,
      std::unordered_map<std::string, onnx::TensorProto> initializer) {
    // get the input names
    std::string input_weight_name;
    std::string input_bias_name;
    std::string input_name;
    for (auto in : node.input()) {
      if (in.find("weight") != std::string::npos) {
        input_weight_name = in;
      } else if (in.find("bias") != std::string::npos) {
        input_bias_name = in;
      } else {
        input_name = in;
      }
    }
    // check for weights and bias
    THROW_COND_EXCEPTION(
        std::runtime_error,
        (initializer.find(input_weight_name) != initializer.end()),
        "Cannot find weight", input_weight_name);
    THROW_COND_EXCEPTION(
        std::runtime_error,
        (initializer.find(input_bias_name) != initializer.end()),
        "Cannot find bias", input_bias_name);
    // get weights and bias
    // std::cout << input_name << " " << input_weight_name << " " <<
    // input_bias_name << std::endl;
    auto weightTensor = initializer[input_weight_name];
    std::vector<int> weight_shape(weightTensor.dims().begin(),
                                  weightTensor.dims().end());
    int weight_vol = std::accumulate(weight_shape.begin(), weight_shape.end(),
                                     1, std::multiplies<int>());
    auto weightDataType = weightTensor.data_type();
    std::string weights = weightTensor.raw_data();
    ASSERT(weights.size() == weight_vol * 2, "Weights neeed to be in FP16");

    auto biasTensor = initializer[input_bias_name];
    std::vector<int> bias_shape(biasTensor.dims().begin(),
                                biasTensor.dims().end());
    auto biasDataType = biasTensor.data_type();
    std::string bias = biasTensor.raw_data();
    int bias_vol = std::accumulate(bias_shape.begin(), bias_shape.end(), 1,
                                   std::multiplies<int>());
    ASSERT(bias.size() == bias_vol * 2, "Bias neeed to be in FP16");

    // get the ouput names
    std::string outputName = node.output()[0];

    // get the attributes
    int ndim;
    std::vector<int> input_spatial_shape;
    std::vector<int> output_spatial_shape;
    int in_channels;
    int out_channels;
    std::vector<int> kernel_size;
    int output_bound;
    std::vector<int> stride;
    std::vector<int> dilation;
    std::vector<int> padding;
    int transposed;
    int inverse;
    std::vector<int> output_padding;
    int groups;
    int subm;
    std::string rulebook;
    std::string activation;
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    std::string input_precision = "fp16";
    std::string output_precision = "fp16";
    float input_dynamic_range;
    std::vector<float> weight_dynamic_ranges;
    for (auto at : node.attribute()) {
      if (at.name() == "ndim") {
        ndim = at.i();
      } else if (at.name() == "input_spatial_shape") {
        input_spatial_shape.insert(input_spatial_shape.begin(),
                                   at.ints().begin(), at.ints().end());
      } else if (at.name() == "output_spatial_shape") {
        output_spatial_shape.insert(output_spatial_shape.begin(),
                                    at.ints().begin(), at.ints().end());
      } else if (at.name() == "in_channels") {
        in_channels = at.i();
      } else if (at.name() == "out_channels") {
        out_channels = at.i();
      } else if (at.name() == "kernel_size") {
        kernel_size.insert(kernel_size.begin(), at.ints().begin(),
                           at.ints().end());
      } else if (at.name() == "output_bound") {
        output_bound = at.i();
      } else if (at.name() == "stride") {
        stride.insert(stride.begin(), at.ints().begin(), at.ints().end());
      } else if (at.name() == "dilation") {
        dilation.insert(dilation.begin(), at.ints().begin(), at.ints().end());
      } else if (at.name() == "padding") {
        padding.insert(padding.begin(), at.ints().begin(), at.ints().end());
      } else if (at.name() == "transposed") {
        transposed = at.i();
      } else if (at.name() == "inverse") {
        inverse = at.i();
      } else if (at.name() == "output_padding") {
        output_padding.insert(output_padding.begin(), at.ints().begin(),
                              at.ints().end());
      } else if (at.name() == "groups") {
        groups = at.i();
      } else if (at.name() == "subm") {
        subm = at.i();
      } else if (at.name() == "rulebook") {
        rulebook = at.s();
      } else if (at.name() == "activation") {
        activation = at.s();
      } else if (at.name() == "input_shape") {
        input_shape.insert(input_shape.begin(), at.ints().begin(),
                           at.ints().end());
      } else if (at.name() == "output_shape") {
        output_shape.insert(output_shape.begin(), at.ints().begin(),
                            at.ints().end());
      } else if (at.name() == "precision") {
        input_precision = at.s();
      } else if (at.name() == "output_precision") {
        output_precision = at.s();
      } else if (at.name() == "input_dynamic_range") {
        input_dynamic_range = at.f();
      } else if (at.name() == "weight_dynamic_ranges") {
        weight_dynamic_ranges.insert(weight_dynamic_ranges.begin(),
                                     at.floats().begin(), at.floats().end());
      } else {
        // throw std::error
      }
    }
    tensor_name_to_scale_[input_name] = input_dynamic_range / 127;

    LOG("Adding node", node.name(), "of type", node.op_type(),
        "with I/O precision", input_precision, "/", output_precision);
    return std::make_shared<SparseConvolution>(
        input_name, input_precision, outputName, output_precision,
        weights.data(), weight_shape, bias.data(), bias_shape, ndim,
        input_spatial_shape, output_spatial_shape, in_channels, out_channels,
        kernel_size, output_bound, stride, dilation, padding, transposed,
        inverse, output_padding, groups, subm, rulebook, activation,
        input_shape, output_shape, input_dynamic_range, weight_dynamic_ranges);
  }

  bool createEngine(onnx::ModelProto model, Precision precision) {
    // get the graph from the model
    auto graph = model.graph();
    // TODO some checks on the graph
    // create a map of initializer for easy parsing
    std::unordered_map<std::string, onnx::TensorProto> initializer;
    // Debug
    // std::cout << "graph inputs:\n";
    // print_io_info(graph.input());
    // std::cout << "graph outputs:\n";
    // print_io_info(graph.output());
    // get the model input and output name
    graph_input_name_ = graph.input()[0].name();
    graph_output_name_ = graph.output()[0].name();

    // std::cout << "Initializer_size " << graph.initializer_size() <<
    // std::endl;
    for (auto init : graph.initializer()) {
      // DEBUG
      // std::cout << init.name() << "[";
      // for (auto dim : init.dims()) {
      //   std::cout << dim << "x";
      // }
      // std::cout << "]\n";
      initializer[init.name()] = init;
    }
    // for (auto node : graph.node()) {
    //   std::cout << node.op_type() << " inputs " << node.input_size() <<
    //       " outputs " << node.output_size() << std::endl;
    //   for (auto in : node.input()) {
    //       std::cout << "[";
    //       std::cout << in;
    //       std::cout << "]\n";
    //     }
    //   std::cout << "attributes " << node.attribute_size() <<std::endl;
    //   for (auto at: node.attribute()) {
    //     std:: cout << at.name() << ", of type " << static_cast<int>
    //     (at.type())<<std::endl;
    //   }
    // }
    onnx::NodeProto prev_spconv_node = graph.node()[0];
    onnx::NodeProto prev_node = graph.node()[0];
    for (auto node : graph.node()) {
      if (node.op_type() == "SparseConvolution") {
        operations_.push_back(addSpconvNode(node, initializer));
        prev_spconv_node = node;
        prev_node = node;
      } else if (node.op_type() == "Add") {
        operations_.push_back(addAddNode(node, prev_spconv_node, initializer));
        prev_node = node;
      } else if (node.op_type() == "Relu") {
        if (prev_node.op_type() == "Add") {
          operations_.pop_back();
          operations_.push_back(addFusedAddReluNode(
              prev_node, node, prev_spconv_node, initializer));
        } else {
          operations_.push_back(
              addReluNode(node, prev_spconv_node, initializer));
        }
        prev_node = node;
      } else if (node.op_type() == "ScatterDense") {
        operations_.push_back(
            addSparseToDenseNode(node, prev_spconv_node, initializer));
        prev_node = node;
      } else if (node.op_type() == "Transpose" || node.op_type() == "Reshape") {
        if (prev_node.op_type() == "ScatterDense") {
          LOG("Merging", node.op_type(), "to ScatterDense.");
        }
      } else {
        THROW_EXCEPTION(std::runtime_error, "Operation not supported",
                        node.op_type());
      }
    }
    // set the precision
    for (auto operation : operations_) {
      operation->set_precision(precision, tensor_name_to_scale_);
    }
    return true;
  }

  bool load(const std::string& file, Precision precision) {
    // auto data = load_file(file);
    if (file.empty()) {
      printf(
          "An empty file has been loaded. Please confirm your file path: %s\n",
          file.c_str());
      return false;
    }
    // open file and move current position in file to the end
    std::ifstream input(file, std::ios::ate | std::ios::binary);
    std::streamsize size = input.tellg();  // get current position in file
    input.seekg(0, std::ios::beg);         // move to start of file
    std::vector<char> buffer(size);
    input.read(buffer.data(), size);  // read raw data

    onnx::ModelProto model;
    model.ParseFromArray(buffer.data(), size);  // parse protobuf

    // ONNX_NAMESPACE::shape_inference::InferShapes(model);
    return createEngine(model, precision);
  }

  virtual DTensor* forward(const std::vector<int64_t>& features_shape,
                           DType features_dtype, void* features_data,
                           const std::vector<int64_t>& indices_shape,
                           DType indices_dtype, void* indices_data, int batch,
                           std::vector<int> grid_size, void* stream = nullptr) {
    std::unordered_map<std::string, std::shared_ptr<DTensor>> io_dict;
    io_dict[graph_input_name_] = std::make_shared<DTensorImplement>(
        features_shape, features_dtype, features_data, indices_shape,
        indices_dtype, indices_data, grid_size);
    for (auto operation : operations_) {
      operation->forward(io_dict, stream);
    }
    auto io_dict_iterator = io_dict.find(graph_output_name_);
    THROW_COND_EXCEPTION(std::out_of_range, (io_dict_iterator != io_dict.end()),
                         "Cannot find graph output ", graph_output_name_);
    result_ = io_dict_iterator->second;
    return result_.get();
  }

  virtual void reconfigure(){};

  // If you want to execute an implicit PTQ calibration, you can enable
  // int8calibration by marking it and collecting the maximum value of the
  // tensor in the next forward.
  virtual void set_int8_calibration(bool enable){};

  // You can modify the precision of a node with this function, but don't forget
  // to call reconfigure
  virtual void set_node_precision_byname(const char* name,
                                         Precision compute_precision,
                                         Precision output_precision){};
  virtual void set_node_precision_byoptype(const char* optype,
                                           Precision compute_precision,
                                           Precision output_precision){};
};

std::shared_ptr<Engine> load_engine_from_onnx(const std::string& onnx_file,
                                              Precision precision) {
  std::shared_ptr<EngineImplement> impl(new EngineImplement());
  if (!impl->load(onnx_file, precision)) impl.reset();
  return impl;
}

bool verbose_ = false;
void set_verbose(bool enable) { verbose_ = enable; }
};  // namespace spconv