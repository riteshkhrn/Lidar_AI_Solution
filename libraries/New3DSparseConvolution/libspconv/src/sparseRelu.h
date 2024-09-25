#ifndef _SPARSE_RELU_H_
#define _SPARSE_RELU_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "spconv/common.hpp"  // for the DTensorImplementation definition
#include "spconv/engine.hpp"  // for the DTensor definition

class SparseRelu : public spconv::Operation {
 public:
  SparseRelu(std::string input_name, std::string output_name, int output_bound,
             int out_channels);
  ~SparseRelu();
  void set_precision(
      spconv::Precision precision,
      std::unordered_map<std::string, float>& tensor_name_to_scale);
  void forward(std::unordered_map<std::string,
                                  std::shared_ptr<spconv::DTensor>>& io_dict,
               void* stream);

 private:
  std::string input_name_;
  std::string output_name_;
  int output_bound_;
  int out_channels_;
  void* output_;
  bool int8_inference_;
};

#endif  //_SPARSE_RELU_H_