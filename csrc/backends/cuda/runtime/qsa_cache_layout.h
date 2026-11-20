#pragma once
#include <string>

namespace mfq::cuda {
// Inspect cache ABI costs without constructing a GPU cache.
std::string qsa_cache_layout_json();
}
