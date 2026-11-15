#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <vector>

namespace tyloi::models::qwen4_exp {

struct LayerSnapshot {
    ::tyloi::Tensor keys, values, index;
    ::tyloi::Tensor convolution, recurrent, ple_convolution;
    // Paged host tensors retain their original accelerator device.
    std::array<std::optional<::tyloi::Device>, 6> devices{};
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
    std::optional<::tyloi::Device> positions_device;

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
    std::vector<std::optional<::tyloi::Device>> positions_devices;
    std::optional<::tyloi::Device> hidden_device;

    template <class Visit>
    void tensors(const Visit& visit) const {
        for (const auto& layer : layers) layer.tensors(visit);
        for (const auto& position : positions) if (position) visit(position);
    }
};

} // namespace tyloi::models::qwen4_exp
