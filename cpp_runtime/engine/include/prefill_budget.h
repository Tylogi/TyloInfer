#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mfq::engine {

inline constexpr int64_t max_prefill_compute_chunk = 8192;
// The native INT8 GEMM requests at most 32 MiB of library scratch. QSA's
// short-context staging and online KV encoder each need another 32 MiB.
// Weight tiles adapt to remaining memory independently of this planner.
inline constexpr std::size_t dense_prefill_fixed_bytes = 3*(std::size_t(32)<<20);

inline std::size_t prefill_budget_add(std::size_t a, std::size_t b) {
    return a > std::numeric_limits<std::size_t>::max() - b
        ? std::numeric_limits<std::size_t>::max() : a + b;
}

inline std::size_t prefill_activation_row_bytes(int64_t hidden,int64_t streams,int64_t dense_width=0) {
    if(hidden<=0 || streams<=0 || dense_width<0)
        throw std::invalid_argument("invalid prefill activation geometry");
    const auto maximum=std::numeric_limits<std::size_t>::max();
    if(std::uint64_t(hidden)>maximum/sizeof(float)/8/std::uint64_t(streams) ||
            std::uint64_t(dense_width)>maximum/sizeof(float)/3)return maximum;
    // Dense Gate, Up and their gated product coexist with the residual stream.
    return prefill_budget_add(std::size_t(hidden)*streams*sizeof(float)*8,
        std::size_t(dense_width)*sizeof(float)*3);
}

inline std::size_t prefill_sequence_row_bytes(int64_t hidden,int64_t streams) {
    if(hidden<=0 || streams<=0)throw std::invalid_argument("invalid prefill residual geometry");
    const auto maximum=std::numeric_limits<std::size_t>::max();
    if(std::uint64_t(hidden)>maximum/sizeof(float)/3/std::uint64_t(streams))return maximum;
    return std::size_t(hidden)*streams*sizeof(float)*3;
}

inline bool prefill_staging_fits(std::size_t available,std::size_t payload,
        std::size_t metadata,std::size_t protected_bytes) {
    if(protected_bytes>available || metadata>available-protected_bytes)return false;
    return payload<=available-protected_bytes-metadata;
}

// Select before allocating or borrowing. Transfer rings and persistent state
// have first claim on memory; only the remaining budget can grow the tile.
// A zero result means even the minimum tile cannot fit, never an OOM retry.
inline int64_t select_prefill_chunk(int64_t tokens, int64_t limit,
        std::size_t available, std::size_t fixed_bytes, std::size_t row_bytes,
        int64_t quantum=256) {
    if(tokens <= 0 || limit <= 0 || row_bytes == 0 || quantum<=0)
        throw std::invalid_argument("invalid prefill workspace geometry");
    const auto wanted = std::min({tokens, limit, max_prefill_compute_chunk});
    const auto rows = available > fixed_bytes ? (available-fixed_bytes)/row_bytes : 0;
    auto selected = static_cast<int64_t>(std::min<std::size_t>(wanted, rows));
    if(selected == wanted) return selected;
    const auto minimum = std::min<int64_t>(wanted, quantum);
    if(selected < minimum) return 0;
    return selected/quantum*quantum;
}

} // namespace mfq::engine
