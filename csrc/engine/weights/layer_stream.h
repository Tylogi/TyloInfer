#pragma once

#include "engine/tensor/tensor.h"
#include "engine/weights/fields.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace tyloi::cuda { class Executor; }

namespace tyloi::weights {

// Prepare immutable device views while keeping their payloads in host memory.
class LayerStream {
public:
    template<class Layers>
    static std::shared_ptr<LayerStream> prepare(Device device, bool prefetch, Layers& layers) {
        auto stream = std::shared_ptr<LayerStream>(new LayerStream(device, prefetch));
        std::vector<std::size_t> sizes;
        sizes.reserve(layers.size());
        for (auto& layer : layers) {
            std::size_t bytes = 0;
            fields::tensors(layer, [&](const Tensor& weight) { measure(weight, bytes); });
            sizes.push_back(bytes);
        }
        stream->allocate(sizes);
        for (std::size_t index = 0; index < layers.size(); ++index) {
            stream->allocate_host(sizes[index]);
            std::size_t offset = 0;
            fields::tensors(layers[index], [&](Tensor& weight) { weight = stream->pack(index, weight, offset); });
        }
        bind(stream);
        return stream;
    }

    ~LayerStream();
    std::size_t host_bytes() const noexcept;
    std::size_t device_bytes() const noexcept;

private:
    friend class cuda::Executor;
    LayerStream(Device device, bool prefetch);
    static void measure(const Tensor&, std::size_t& bytes);
    void allocate(std::span<const std::size_t> sizes);
    void allocate_host(std::size_t bytes);
    Tensor pack(std::size_t layer, const Tensor&, std::size_t& offset);
    static void bind(const std::shared_ptr<LayerStream>& stream);
    void stage(std::size_t layer);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tyloi::weights
