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
                int out_channels);
  ~SparseToDense();
  void forward(std::unordered_map<std::string,
                                  std::shared_ptr<spconv::DTensor>>& io_dict,
               void* stream);

 private:
  std::string input_name_;
  std::string output_name_;
  std::string format_;
  std::vector<int> spatial_shape_;
  int64_t output_size_;
  int out_channels_;
  void* output_;
};

#endif  //_SPARSE_TO_DENSE_H_