#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <array>
#include <memory>
#include <vector>

namespace tyloi::models::qwen4_exp {

struct LayerSnapshot {
    ::tyloi::Tensor keys, values, index;
    ::tyloi::Tensor convolution, recurrent, ple_convolution;
    // Nonnegative indices identify host tensors decoded from paged CUDA data.
    std::array<int, 6> devices{-1, -1, -1, -1, -1, -1};
    int64_t ple_batch = 0;
    std::vector<int64_t> ple_tokens;

    template <class Visit>
    void tensors(const Visit& visit) const {
        if (keys) visit(keys);
        if (values) visit(values);
        if (index) visit(index);
        if (convolution) visit(convolution);
        if (recurrent) visit(recurrent);
        if (ple_convolution) visit(ple_convolution);
    }
};

struct Snapshot {
    std::vector<LayerSnapshot> layers;
    ::tyloi::Tensor positions;
    int positions_device = -1;

    template <class Visit>
    void tensors(const Visit& visit) const {
        if (positions) visit(positions);
        for (const auto& layer : layers) layer.tensors(visit);
    }
};

struct MtpSnapshot {
    std::vector<LayerSnapshot> layers;
    std::vector<::tyloi::Tensor> positions;
    std::vector<int64_t> lengths;
    std::vector<int> positions_devices;
    int hidden_device = -1;

    template <class Visit>
    void tensors(const Visit& visit) const {
        for (const auto& layer : layers) layer.tensors(visit);
        for (const auto& position : positions) if (position) visit(position);
    }
};

} // namespace tyloi::models::qwen4_exp
