#pragma once

#include "engine/tensor/tensor.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi::models::deepseek_v41 {

struct EngramHashSnapshot;

struct BlockSnapshot {
    ::tyloi::Tensor local_kv, compressed_kv, index_k, partial_kv, partial_score;
    int64_t compressed_length = 0, partial_length = 0;
    // Nonnegative for CPU tensors decoded from a paged CUDA block.
    int original_device = -1;

    template <class Visit>
    void tensors(const Visit& visit) const {
        if (local_kv) visit(local_kv);
        if (compressed_kv) visit(compressed_kv);
        if (index_k) visit(index_k);
        if (partial_kv) visit(partial_kv);
        if (partial_score) visit(partial_score);
    }
};

struct Snapshot {
    std::vector<BlockSnapshot> blocks;
    std::shared_ptr<const EngramHashSnapshot> engram;

    template <class Visit>
    void tensors(const Visit& visit) const {
        for (const auto& block : blocks) block.tensors(visit);
    }
};

struct MtpSnapshot {
    std::vector<::tyloi::Tensor> rings;
    int hidden_device = -1;
    std::vector<int> ring_devices;

    template <class Visit>
    void tensors(const Visit& visit) const {
        for (const auto& ring : rings) if (ring) visit(ring);
    }
};

} // namespace tyloi::models::deepseek_v41
