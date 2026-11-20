#pragma once
#include <cstddef>

namespace mfq::cuda {
// Return only idle CPU pages; active tensor views retain their storage.
std::size_t reclaim_tensor_host_storage_cache() noexcept;
}
