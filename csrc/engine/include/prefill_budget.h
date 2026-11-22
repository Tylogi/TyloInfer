#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mfq::engine {

inline constexpr int64_t max_prefill_compute_chunk = 8192;

inline std::size_t prefill_budget_add(std::size_t a, std::size_t b) {
    return a > std::numeric_limits<std::size_t>::max() - b
        ? std::numeric_limits<std::size_t>::max() : a + b;
}

inline std::size_t prefill_budget_multiply(std::size_t a, std::size_t b) {
    return b && a > std::numeric_limits<std::size_t>::max() / b
        ? std::numeric_limits<std::size_t>::max() : a * b;
}

struct PrefillWorkspace {
    std::size_t fixed_bytes;
    std::size_t row_bytes;

    std::size_t bytes(int64_t rows, std::size_t attention_bytes = 0) const {
        return prefill_budget_add(prefill_budget_add(fixed_bytes, attention_bytes),
            prefill_budget_multiply(std::size_t(rows), row_bytes));
    }
};

inline bool prefill_layer_major_saves_transfers(std::size_t tokens, std::size_t prefix,
        std::size_t tile, std::size_t history_bytes_per_token, std::size_t activation_bytes_per_token) {
    if(!tile || tokens<=tile || !history_bytes_per_token)return false;
    const auto subsequent_tiles=(tokens-1)/tile;
    // Each subsequent token tile would otherwise stage the live history
    // again. Layer-major retains it, but exchanges hidden tiles at every layer.
    const auto triangular=prefill_budget_multiply(subsequent_tiles,subsequent_tiles-1)/2;
    const auto replayed_tokens=prefill_budget_add(prefill_budget_multiply(subsequent_tiles,prefix),
        prefill_budget_multiply(triangular,tile));
    return prefill_budget_multiply(replayed_tokens,history_bytes_per_token)>
        prefill_budget_multiply(tokens,activation_bytes_per_token);
}

// Use the same byte estimate for admission and execution. An attention tile
// can shrink independently of the projection alignment when it must fit.
template<class WorkspaceBytes>
int64_t select_prefill_chunk(int64_t tokens, int64_t limit, std::size_t available,
        WorkspaceBytes bytes, bool allow_small_tile) {
    if(tokens <= 0 || limit <= 0)
        throw std::invalid_argument("invalid prefill workspace geometry");
    const auto wanted = std::min({tokens, limit, max_prefill_compute_chunk});
    const auto fits=[&](int64_t rows) {
        const auto required=bytes(rows);
        return required!=std::numeric_limits<std::size_t>::max() && required<=available;
    };
    if(fits(wanted)) return wanted;
    int64_t low=0, high=wanted/256;
    while(low<high) {
        const auto middle=low+(high-low+1)/2;
        if(fits(256*middle)) low=middle;
        else high=middle-1;
    }
    if(low || !allow_small_tile) return low*256;
    auto rows=std::min<int64_t>(wanted,256);
    while(rows && !fits(rows)) rows/=2;
    return rows;
}

// Select before allocating or borrowing. Transfer rings and persistent state
// have first claim on memory; only the remaining budget can grow the tile.
// A zero result means even the minimum tile cannot fit, never an OOM retry.
inline int64_t select_prefill_chunk(int64_t tokens, int64_t limit,
        std::size_t available, std::size_t fixed_bytes, std::size_t row_bytes) {
    if(!row_bytes)
        throw std::invalid_argument("invalid prefill workspace geometry");
    const PrefillWorkspace workspace{fixed_bytes,row_bytes};
    return select_prefill_chunk(tokens,limit,available,[&](int64_t rows) {
        return workspace.bytes(rows);
    },false);
}

} // namespace mfq::engine
