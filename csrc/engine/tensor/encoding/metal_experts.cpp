#include "engine/tensor/encoding/metal_experts.h"

#include "engine/backends/metal/ops/mfq/mlx_moe.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::encoding {

namespace {

void validate_shape(const std::array<std::int64_t, 3>& shape) {
    std::int64_t elements = 1;
    for (const auto extent : shape) {
        if (extent <= 0)
            throw std::invalid_argument("invalid Metal expert bank geometry");
        if (elements > std::numeric_limits<std::int64_t>::max() / extent)
            throw std::overflow_error("Metal expert bank element count overflow");
        elements *= extent;
    }
    if (static_cast<std::uint64_t>(elements) >
        std::numeric_limits<std::size_t>::max() / scalar_size(kFloat16))
        throw std::overflow_error("Metal expert bank byte count overflow");
}

std::array<std::int64_t, 3> resident_shape(const metal::MlxMfeWeight& weight) {
    if (weight.projections() <= 0 || weight.experts() <= 0 ||
        weight.out_per_expert() <= 0 || weight.neuron_len() <= 0)
        throw std::invalid_argument("invalid Metal expert bank geometry");
    if (static_cast<std::int64_t>(weight.out_per_expert()) >
        std::numeric_limits<std::int64_t>::max() / weight.projections())
        throw std::overflow_error("Metal expert bank output width overflow");
    const std::array<std::int64_t, 3> shape{
        weight.experts(),
        static_cast<std::int64_t>(weight.out_per_expert()) * weight.projections(),
        weight.neuron_len()};
    validate_shape(shape);
    return shape;
}

} // namespace

MetalExperts::MetalExperts(metal::MlxMfeWeight weight)
    : weight_(std::make_shared<const metal::MlxMfeWeight>(std::move(weight))) {
    shape_ = resident_shape(*weight_);
}

MetalExperts::MetalExperts(std::shared_ptr<metal::MlxMfeOffloadCache> cache,
        std::string record_name, std::array<std::int64_t, 3> shape)
    : cache_(std::move(cache)), record_name_(std::move(record_name)), shape_(shape) {
    if (!cache_)
        throw std::invalid_argument("Metal expert bank cache is null");
    if (record_name_.empty())
        throw std::invalid_argument("Metal expert bank record name is empty");
    validate_shape(shape_);
}

const metal::MlxMfeWeight& MetalExperts::weight() const {
    if (!weight_)
        throw std::logic_error("paged Metal expert bank has no resident weight");
    return *weight_;
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
    return weight_ ? weight_->backings() : std::vector<std::shared_ptr<TensorStorage>>{};
}

void MetalExperts::record_stream(std::uintptr_t, std::shared_ptr<void>) const {
    throw std::runtime_error("Metal expert bank stream recording is not supported");
}

} // namespace tyloi::encoding
