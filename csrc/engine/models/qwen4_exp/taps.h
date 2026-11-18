#pragma once

#include "engine/models/common/check.h"
#include "engine/ops/ops.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace tyloi::models::qwen4_exp {

class Taps {
public:
    Taps(std::span<const std::int64_t> layers, std::int64_t streams, Device device)
        : layers_(layers), streams_(streams), device_(device), values_(layers.size()) {}

    bool contains(std::int64_t layer) const {
        return std::binary_search(layers_.begin(), layers_.end(), layer);
    }

    void capture(std::int64_t layer, const Tensor& hidden) {
        if (!contains(layer)) return;
        const auto slot = std::lower_bound(layers_.begin(), layers_.end(), layer) - layers_.begin();
        ops::DeviceScope scope(device_);
        auto grouped = ops::reshape(ops::copy(hidden, device_),
            {hidden.size(0), hidden.size(1), streams_, -1});
        values_[slot] = ops::contiguous(ops::mean(ops::cast(grouped, kFloat32), 2));
    }

    Tensor finish(Tensor fallback) {
        if (layers_.empty()) return fallback;
        require_model(std::all_of(values_.begin(), values_.end(),
            [](const Tensor& value) { return value.defined(); }),
            "DFlash2 target hidden states were not fully captured");
        ops::DeviceScope scope(device_);
        return ops::contiguous(ops::cat(values_, -1));
    }

private:
    std::span<const std::int64_t> layers_;
    std::int64_t streams_;
    Device device_;
    std::vector<Tensor> values_;
};

} // namespace tyloi::models::qwen4_exp
