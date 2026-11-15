#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class KlMmqMode;

namespace tyloi::cuda::internal {
std::vector<int64_t> parse_ids(const std::string& value);
KlMmqMode parse_kl_mmq_mode(const std::string& value);
std::vector<KlMmqMode> parse_kl_mmq_sequence(const std::string& value);
std::vector<int64_t> load_ids_file(const std::string& path);
} // namespace tyloi::cuda::internal
