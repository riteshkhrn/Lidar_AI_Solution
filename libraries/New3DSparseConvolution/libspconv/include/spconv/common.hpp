#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "spconv/engine.hpp"

#define LOG(...)                                                         \
  do {                                                                     \
    if(spconv::verbose_) spconv::ssprint(__FILE__, __LINE__, __VA_ARGS__); \
  } while (false)

#define THROW_EXCEPTION(e, ...)                                  \
  do {                                                           \
    std::stringstream msg;                                       \
    spconv::sstream_print(msg, __FILE__, __LINE__, __VA_ARGS__); \
    throw e(msg.str());                                          \
  } while (false)

#define THROW_COND_EXCEPTION(e, cond, ...)                         \
  do {                                                             \
    if (!cond) {                                                   \
      std::stringstream msg;                                       \
      spconv::sstream_print(msg, __FILE__, __LINE__, __VA_ARGS__); \
      throw e(msg.str());                                          \
    }                                                              \
  } while (false)

#define ASSERT(cond, ...)                                        \
  do {                                                           \
    if (!cond) spconv::ssprint(__FILE__, __LINE__, __VA_ARGS__); \
    assert(cond);                                                \
  } while (false)

namespace spconv {

extern bool verbose_;

template <char Sep = ' ', class SStream, class T>
void sstream_print(SStream& ss, T val) {
  ss << val;
}

template <char Sep = ' ', class SStream, class T, class... TArgs>
void sstream_print(SStream& ss, T val, TArgs... args) {
  ss << val << Sep;
  sstream_print<Sep>(ss, args...);
}

template <char Sep = ' ', class... TArgs>
std::string ssprint(TArgs... args) {
  std::stringstream ss;
  sstream_print<Sep>(ss, args...);
  std::cout << ss.str() << std::endl;
  return ss.str();
}

class DTensorImplement : public DTensor {
 public:
  DTensorImplement(std::vector<int64_t> features_shape, DType features_dtype,
                   void* features_data,
                   std::vector<int64_t> indices_shape = std::vector<int64_t>(),
                   DType indices_dtype = DType::None,
                   void* indices_data = nullptr,
                   std::vector<int> grid_size = std::vector<int>(),
                   int device = -1)
      : features_shape_(features_shape),
        features_dtype_(features_dtype),
        features_data_(features_data),
        indices_shape_(indices_shape),
        indices_dtype_(indices_dtype),
        indices_data_(indices_data),
        grid_size_(grid_size),
        device_(device) {}
  virtual std::vector<int64_t> features_shape() const {
    return features_shape_;
  }
  virtual DType features_dtype() const { return features_dtype_; }
  virtual void* features_data() { return features_data_; }

  virtual std::vector<int64_t> indices_shape() const { return indices_shape_; }
  virtual DType indices_dtype() const { return indices_dtype_; }
  virtual void* indices_data() { return indices_data_; }

  virtual std::vector<int> grid_size() const { return grid_size_; }
  virtual int device() const { return device_; }

 private:
  std::vector<int64_t> features_shape_;
  DType features_dtype_;
  void* features_data_;
  std::vector<int64_t> indices_shape_;
  DType indices_dtype_;
  void* indices_data_;
  std::vector<int> grid_size_;
  int device_;
};

class Operation {
 public:
  virtual void set_precision(
      spconv::Precision precision,
      std::unordered_map<std::string, float>& tensor_name_to_scale) {};
  virtual void forward(
      std::unordered_map<std::string, std::shared_ptr<spconv::DTensor>>&
          io_dict,
      void* stream) = 0;
};

// shared device code

// extern __device__ int clamp(int x, int a, int b);

// extern __device__ int quantize(float inp, float scale);

};  // namespace spconv

#endif  // __COMMON_HPP__