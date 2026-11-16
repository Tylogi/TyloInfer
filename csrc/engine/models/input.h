#pragma once

#include "engine/tensor/tensor.h"
#include "engine/ops/ops.h"

#include <cstdint>
#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::models {

struct Input {
    using Trace = std::vector<Tensor>;

    Tensor ids, embeddings, positions, lengths, cache_positions;
    Trace* trace = nullptr;
    Tensor* raw_hidden = nullptr;
    bool advance = false;
    std::int64_t confirmed = 0;
    std::int64_t kv_length = 0;
    std::int64_t attention_parts = 0;
    // Draft rows for an already-started speculative suffix transaction.
    bool suffix = false;
    Tensor host_ids, image_mask;

    std::pair<std::int64_t, std::int64_t> shape() const {
        std::int64_t batch = 0, tokens = 0;
        if (ids && ids.defined()) {
            if (ids.dim() == 1) { batch = 1; tokens = ids.size(0); }
            else if (ids.dim() == 2) { batch = ids.size(0); tokens = ids.size(1); }
            else throw std::invalid_argument("token IDs require [tokens] or [batch,tokens]");
        } else if (embeddings && embeddings.defined() && embeddings.dim() == 3) {
            batch = embeddings.size(0); tokens = embeddings.size(1);
        } else throw std::invalid_argument("model input requires token IDs or embeddings");
        if (batch <= 0 || tokens <= 0 || (embeddings && embeddings.defined() &&
            (embeddings.dim() != 3 || embeddings.size(0) != batch || embeddings.size(1) != tokens)))
            throw std::invalid_argument("model input batch and token dimensions must agree and be positive");
        return {batch, tokens};
    }
};

struct PrefillOptions {
    std::int64_t tokens = 0, compute = 0, first = 0, last = 0;
    bool transformed = false, retained = false, routed = false, coordinated = false, split = false;
    bool raw_host = false;
};

struct PrefillPlan {
    bool layered = false;
    // Bound the logical advance, independently of each compute chunk.
    std::int64_t max_span = 0;
};

struct Prefill {
    enum class Phase { progress, computed, encode, decode, complete };
    enum class Mode { ordinary, layered, windows };
    struct Progress {
        Device device;
        std::int64_t tokens = 0, chunk = 0, completed = 0, total = 0;
        std::chrono::steady_clock::time_point started;
    };
    Phase phase = Phase::complete;
    Mode mode = Mode::ordinary;
    Tensor hidden, raw;
    double elapsed_ms = 0;
    int first = 0, last = 0;
    std::optional<Progress> progress;
};

struct DecodeOptions {
    enum class Projection { hidden_only, hidden, logits, greedy };
    Projection projection = Projection::hidden_only;
    bool routed = false, compact_sampling = false, enabled = true;
    std::int64_t reserve_until = 0;
};

struct Decoded {
    Tensor hidden, raw, logits, compact, ids;
    std::vector<std::int32_t> tokens;
    bool initialized = false;
};

struct ForwardOptions {
    std::int64_t kv_capacity = 0;
    ops::Trace* trace = nullptr;
    int trace_layer = -1;
    int primary = 0;
    bool serial_experts = false;
};

struct Prepared {
    std::vector<std::int64_t> token_ids;
    Tensor embeddings;
    Tensor positions;
    int decode_position_delta = 0;
    std::string cache_key;
    Tensor image_mask;

    bool transformed() const noexcept {
        return (embeddings && embeddings.defined()) ||
            (positions && positions.defined()) ||
            decode_position_delta != 0;
    }
};

} // namespace tyloi::models
