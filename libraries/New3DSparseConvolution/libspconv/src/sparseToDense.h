#ifndef _SPARSE_TO_DENSE_H_
#define _SPARSE_TO_DENSE_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "spconv/common.hpp"  // for the DTensorImplementation definition
#include "spconv/engine.hpp"  // for the DTensor definition

class SparseToDense : public spconv::Operation {
 public:
  SparseToDense(std::string input_name, std::string output_name,
                std::string format, std::vector<int> spatial_shape,
                std::vector<int> output_spatial_shape,
                spconv::TensorLayout output_layout);
  ~SparseToDense();
  virtual void configure(
      spconv::Precision precision,
      std::unordered_map<std::string, float>& tensor_name_to_scale,
      std::unordered_map<std::string, std::shared_ptr<void>> parameters);
  void forward(
    std::unordered_map<std::string, std::shared_ptr<spconv::SparseDTensor>> &io_dict,
    void* stream);

 private:
  std::string input_name_;
  std::string output_name_;
  std::string format_;
  std::vector<int> spatial_shape_;
  std::vector<int> output_spatial_shape_;
  std::vector<int> stride_;
  spconv::TensorLayout output_layout_;
  int out_channels_;
  int64_t output_size_;
  void* output_;
};

#endif  //_SPARSE_TO_DENSE_H_