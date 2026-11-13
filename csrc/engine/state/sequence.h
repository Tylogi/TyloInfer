#pragma once

#include "engine/ops.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace tyloi::state {

class Sequence {
public:
    Sequence(std::int64_t maximum, std::int64_t width, ScalarType dtype = kFloat16)
        : maximum_(maximum), width_(width), dtype_(dtype) {
        if (maximum <= 0 || width <= 0) throw std::runtime_error("invalid sequence cache dimensions");
        if (!scalar_size(dtype)) throw std::invalid_argument("invalid sequence cache dtype");
    }
    void reset() { values_ = {}; position_ = 0; }
    void reset(std::int64_t batch, std::int64_t initial_capacity, Device device) {
        (void)reset_bytes(batch, initial_capacity);
        const auto capacity = std::min(maximum_, initial_capacity);
        auto next = ops::zeros({batch, capacity, width_}, TensorOptions().device(device).dtype(dtype_));
        values_ = std::move(next);
        position_ = 0;
    }
    std::size_t reset_bytes(std::int64_t batch, std::int64_t initial_capacity = 16) const {
        if (batch <= 0 || initial_capacity <= 0) throw std::invalid_argument("invalid sequence cache allocation");
        const std::array<std::int64_t, 3> shape{batch, std::min(maximum_, initial_capacity), width_};
        return make_contiguous_view(nullptr, shape, dtype_, {}).nbytes();
    }
    std::int64_t position() const { return position_; }
    const Tensor& storage() const { return values_; }
    Tensor view() const {
        if (!values_.defined()) throw std::runtime_error("sequence cache is not initialized");
        return ops::slice(values_, 1, 0, position_);
    }
    void release_storage() { values_ = {}; }
    void replace_storage(Tensor values) {
        if (!(values.dim() == 3 && values.size(1) >= position_ && values.size(1) <= maximum_ && values.size(2) == width_))
            throw std::runtime_error("replacement sequence cache geometry mismatch");
        values_ = std::move(values);
    }
    void prepare_fixed(std::int64_t capacity) {
        if (!values_.defined()) throw std::runtime_error("fixed sequence cache requires a prefilled state");
        if (!(capacity >= position_ && capacity <= maximum_)) throw std::runtime_error("invalid fixed sequence cache capacity");
        if (values_.size(1) == capacity) return;
        auto expanded = ops::zeros({values_.size(0), capacity, width_}, values_.options());
        if (position_) expanded = ops::update_slice(expanded, ops::slice(values_, 1, 0, position_), 1, 0);
        values_ = std::move(expanded);
    }
    Tensor append_fixed(const Tensor& value, const Tensor& positions) {
        if (!(values_.defined() && values_.size(1) <= maximum_ && value.dim() == 3 &&
            value.size(1) <= values_.size(1) - position_ && value.size(0) == values_.size(0) &&
            value.size(2) == width_ && positions.dim() == 1 && positions.size(0) == value.size(1)))
            throw std::runtime_error("fixed sequence cache append geometry mismatch");
        ops::index_copy(values_, 1, ops::cast(positions, kInt64), ops::cast(value, dtype_));
        return values_;
    }
    void advance_fixed(std::int64_t tokens) {
        if (!(tokens > 0 && tokens <= maximum_ - position_)) throw std::runtime_error("fixed sequence cache exceeds context");
        position_ += tokens;
    }
    void truncate(std::int64_t keep) {
        if (!(keep >= 0 && keep <= position_)) throw std::runtime_error("invalid sequence cache truncation");
        position_ = keep;
    }
    std::size_t growth_bytes(std::int64_t batch, std::int64_t tokens) const {
        if (!(batch > 0 && tokens > 0 && position_ >= 0 && position_ <= maximum_ && tokens <= maximum_ - position_))
            throw std::runtime_error("invalid sequence cache growth");
        if (values_.defined() && values_.size(0) != batch) throw std::runtime_error("reset sequence cache before changing batch");
        const auto required = position_ + tokens;
        if (values_.defined() && required <= values_.size(1)) return 0;
        const std::array<std::int64_t, 3> shape{batch, capacity_for(required), width_};
        return make_contiguous_view(nullptr, shape, dtype_, {}).nbytes();
    }
    Tensor append(const Tensor& value) {
        if (!(value.defined() && !value.is_encoded() && (value.is_cuda() || value.is_metal()) && value.dim() == 3 &&
            value.size(0) > 0 && value.size(1) > 0 && value.size(2) == width_))
            throw std::runtime_error("invalid sequence cache append");
        if (values_.defined() && (value.size(0) != values_.size(0) || value.device() != values_.device()))
            throw std::runtime_error("reset sequence cache before changing batch/device");
        if (value.size(1) > maximum_ - position_) throw std::runtime_error("sequence cache exceeds max_context");
        const auto required = position_ + value.size(1);
        if (!values_.defined() || required > values_.size(1)) {
            auto expanded = ops::zeros({value.size(0), capacity_for(required), width_}, value.options().dtype(dtype_));
            if (position_) expanded = ops::update_slice(expanded, ops::slice(values_, 1, 0, position_), 1, 0);
            values_ = std::move(expanded);
        }
        values_ = ops::update_slice(values_, ops::cast(value, dtype_), 1, position_);
        position_ = required;
        return ops::slice(values_, 1, 0, position_);
    }
private:
    std::int64_t capacity_for(std::int64_t required) const {
        auto capacity = values_.defined() ? values_.size(1) : std::min<std::int64_t>(16, maximum_);
        if (capacity == 0) capacity = std::min<std::int64_t>(16, maximum_);
        while (capacity < required) capacity = std::min(maximum_, capacity > maximum_ / 2 ? maximum_ : capacity * 2);
        return capacity;
    }
    std::int64_t maximum_, width_, position_ = 0;
    ScalarType dtype_;
    Tensor values_;
};

} // namespace tyloi::state
