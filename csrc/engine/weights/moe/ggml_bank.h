#pragma once

#include "engine/tensor/encoding/ggml.h"

namespace tyloi::weights { class GgmlExpertCache; }

namespace tyloi {

// Rank-three routed weights publish one quantized matrix per selected expert.
class GgmlExpertBank final : public TensorImpl {
public:
    GgmlQuantizedTensor image;
    DenseTensor resident_addresses;
    std::shared_ptr<weights::GgmlExpertCache> cache;
    Device target;
    std::uint32_t source_id = 0;
    std::int64_t experts = 0;
    std::int64_t expert_bytes = 0;

    bool defined() const noexcept override { return source_id && experts > 0; }
    std::int64_t dim() const noexcept override { return 3; }
    std::int64_t size(std::int64_t axis) const override {
        if (axis < 0) axis += 3;
        switch (axis) {
            case 0: return experts;
            case 1: return image.out;
            case 2: return image.neuron_len;
            default: throw std::out_of_range("GGUF expert bank dimension is out of range");
        }
    }
    std::int64_t numel() const override {
        TensorView view;
        view.rank = 3;
        view.sizes[0] = experts;
        view.sizes[1] = image.out;
        view.sizes[2] = image.neuron_len;
        return view.numel();
    }
    std::size_t nbytes() const override { return static_cast<std::size_t>(numel()) * sizeof(std::uint16_t); }
    std::size_t backing_nbytes() const override { return backing_bytes(buffers()); }
    ScalarType scalar_type() const noexcept override { return kFloat16; }
    const Device& device() const noexcept override { return target; }
    bool is_contiguous() const noexcept override { return false; }
    bool is_encoded() const noexcept override { return true; }
    std::vector<const DenseTensor*> buffers() const override {
        auto result = image.buffers();
        if (resident_addresses.defined()) result.push_back(&resident_addresses);
        return result;
    }
    void record_stream(std::uintptr_t stream, std::shared_ptr<void> lifetime = {}) const override {
        image.record_stream(stream, lifetime);
        if (resident_addresses.defined()) resident_addresses.record_stream(stream, std::move(lifetime));
    }
    std::shared_ptr<TensorImpl> relocated(Device) const override {
        throw std::invalid_argument("GGUF expert bank relocation is unsupported");
    }
};

} // namespace tyloi
