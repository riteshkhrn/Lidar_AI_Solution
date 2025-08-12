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

#define THROW_COND_EXCEPTION(cond, e,  ...)                         \
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

// overload the ostream << opertor for precision
std::ostream& operator<<(std::ostream& out, const spconv::Precision precision);

class Operation {
 public:
  virtual void configure(
      spconv::Precision precision,
      std::unordered_map<std::string, float>& tensor_name_to_scale,
      std::unordered_map<std::string, std::shared_ptr<void>> parameters) = 0;
  virtual void forward(
      std::unordered_map<std::string, std::shared_ptr<SparseDTensor>>&  io_dict,
      void* stream) = 0;
};

class UnspportedOperation : public Operation {
 public:
  virtual void configure(
      spconv::Precision precision,
      std::unordered_map<std::string, float>& tensor_name_to_scale,
      std::unordered_map<std::string, std::shared_ptr<void>> parameters) {
    THROW_EXCEPTION(std::runtime_error, "Operation not implemented yet");
  };
  virtual void forward(
      std::unordered_map<std::string, std::shared_ptr<SparseDTensor>>&  io_dict,
      void* stream) {
    THROW_EXCEPTION(std::runtime_error, "Operation not implemented yet");
  }
};

class IdenityOperation : public Operation {
  public:
   IdenityOperation(std::string input_name, std::string output_name) 
    :input_name_(input_name),
     output_name_(output_name) {}
     virtual void configure(
        spconv::Precision precision,
        std::unordered_map<std::string, float>& tensor_name_to_scale,
        std::unordered_map<std::string, std::shared_ptr<void>> parameters) {};
   virtual void forward(
       std::unordered_map<std::string, std::shared_ptr<SparseDTensor>>&  io_dict,
       void* stream) {
        auto input_iterator = io_dict.find(input_name_);
        THROW_COND_EXCEPTION((input_iterator != io_dict.end()), std::out_of_range, 
                            input_name_, "input not found");
        auto input = input_iterator->second;
        auto output_iterator = io_dict.find(output_name_);
        THROW_COND_EXCEPTION((output_iterator != io_dict.end()), std::out_of_range, 
                                 "cannot find output in io_dict", output_name_);
        auto out = output_iterator->second;
        out->features().reference((void*)input->features().ptr(), 
        input->features().shape, input->features().dtype());
        out->indices().reference((void*)input->indices().ptr(), 
        input->indices().shape, input->indices().dtype());
        out->set_grid_size(input->grid_size());
   }
  private:
  std::string input_name_;
  std::string output_name_;
};
};  // namespace spconv

#endif  // __COMMON_HPP__