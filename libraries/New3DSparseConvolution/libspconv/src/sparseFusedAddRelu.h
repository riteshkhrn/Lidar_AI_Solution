#ifndef _SPARSE_FUSED_ADD_RELU_H_
#define _SPARSE_FUSED_ADD_RELU_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "spconv/common.hpp"  // for the DTensorImplementation definition
#include "spconv/engine.hpp"  // for the DTensor definition

class SparseFusedAddRelu : public spconv::Operation {
 public:
  SparseFusedAddRelu(std::string input_name_0, std::string input_name_1,
                     std::string input_precision, std::string output_name,
                     std::string output_precision, int output_bound,
                     int out_channels);
  ~SparseFusedAddRelu();
  void set_precision(
      spconv::Precision precision,
      std::unordered_map<std::string, float>& tensor_name_to_scale);
  void forward(std::unordered_map<std::string,
                                  std::shared_ptr<spconv::DTensor>>& io_dict,
               void* stream);

 private:
  bool int8_inference_;
  std::string input_name_1_;
  std::string input_name_2_;
  std::string input_precision_;
  std::string output_name_;
  std::string output_precision_;
  float input_scale_1_;
  float input_scale_2_;
  float output_scale_;
  int output_bound_;
  int out_channels_;
  void* output_;
};

#endif  //_SPARSE_ADD_H_