#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace mfq {class ModelSource;struct TqLayout;}
namespace mfq::tq_source {
std::vector<std::int64_t> logical_shape(const ModelSource&,const std::string&);
std::vector<std::string> row_parts(const ModelSource&,const std::string&);
std::vector<std::uint8_t> join_rows(const std::vector<std::vector<std::uint8_t>>&);
std::vector<std::uint8_t> tq_image(const std::vector<std::uint8_t>&,const TqLayout&);
bool is_row_table(const ModelSource&,const std::string&);
}
