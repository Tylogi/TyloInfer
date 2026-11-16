#include "engine/tensor/encoding/metal_experts.h"

#include "engine/backends/metal/ops/mfq/mlx_moe.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::encoding {

MetalExperts::MetalExperts(metal::MlxMfeWeight weight)
    : weight_(std::make_shared<const metal::MlxMfeWeight>(std::move(weight))) {
    if (weight_->projections() <= 0 || weight_->experts() <= 0 ||
        weight_->out_per_expert() <= 0 || weight_->neuron_len() <= 0)
        throw std::invalid_argument("invalid Metal expert bank geometry");
    if (static_cast<std::int64_t>(weight_->out_per_expert()) >
        std::numeric_limits<std::int64_t>::max() / weight_->projections())
        throw std::overflow_error("Metal expert bank output width overflow");
    shape_ = {weight_->experts(),
        static_cast<std::int64_t>(weight_->out_per_expert()) * weight_->projections(),
        weight_->neuron_len()};
    std::int64_t elements = 1;
    for (const auto extent : shape_) {
        if (elements > std::numeric_limits<std::int64_t>::max() / extent)
            throw std::overflow_error("Metal expert bank element count overflow");
        elements *= extent;
    }
    if (static_cast<std::uint64_t>(elements) >
        std::numeric_limits<std::size_t>::max() / scalar_size(kFloat16))
        throw std::overflow_error("Metal expert bank byte count overflow");
}

std::int64_t MetalExperts::numel() const {
    return shape_[0] * shape_[1] * shape_[2];
}

std::size_t MetalExperts::nbytes() const {
    return static_cast<std::size_t>(numel()) * scalar_size(kFloat16);
}

std::size_t MetalExperts::backing_nbytes() const {
    return backing_bytes(backings());
}

std::int64_t MetalExperts::size(std::int64_t dimension) const {
    if (dimension < 0) dimension += dim();
    return shape_.at(static_cast<std::size_t>(dimension));
}

std::vector<std::shared_ptr<TensorStorage>> MetalExperts::backings() const {
    return weight_->backings();
}

void MetalExperts::record_stream(std::uintptr_t, std::shared_ptr<void>) const {
    throw std::runtime_error("Metal expert bank stream recording is not supported");
}

} // namespace tyloi::encoding
