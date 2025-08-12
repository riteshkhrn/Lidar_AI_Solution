
#include "spconv/common.hpp"

namespace spconv {

std::ostream& operator<<(std::ostream& out, const spconv::Precision precision) {
  switch(precision) {
    case spconv::Precision::Int8:
      return out << "int8";
    case spconv::Precision::Float16:
      return out << "fp16";
    case spconv::Precision::None:
      return out << "None";
    default:
    return out << "None";
  }
}
};  // namespace spconv