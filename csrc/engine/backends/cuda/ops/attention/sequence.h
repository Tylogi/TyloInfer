#pragma once

#include "backend.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace tyloi::cuda {
class SequenceCache {
public:
    SequenceCache(int64_t maximum, int64_t width, ::tyloi::ScalarType dtype = ::tyloi::kFloat16)
        : maximum_(maximum), width_(width), dtype_(dtype) {
        TYLOINFER_RUNTIME_CHECK(maximum > 0 && width > 0, "invalid sequence cache dimensions");
    }
    /** Release the backing and reset the committed sequence position. */
    void reset() { values_ = {}; position_ = 0; }
    int64_t position() const { return position_; }
    /** Return the owning handle for the sequence backing. */
    const ::tyloi::Tensor& storage() const { return values_; }
    /** Release backing held by Graph KV while preserving the committed position. */
    void release_storage() { values_={}; }
    /** Replace the backing while preserving the committed position. */
    void replace_storage(::tyloi::Tensor values) {
        TYLOINFER_RUNTIME_CHECK(values.representation<DenseTensor>(), "replacement sequence cache must be dense");
        TYLOINFER_RUNTIME_CHECK(values.dim()==3 && values.size(1)>=position_ && values.size(1)<=maximum_ &&
            values.size(2)==width_,"replacement sequence cache geometry mismatch");
        values_=std::move(values);
    }
    /** Resize backing for static writes while preserving the committed prefix. */
    void prepare_fixed(int64_t capacity) {
        TYLOINFER_RUNTIME_CHECK(values_.defined(), "fixed sequence cache requires a prefilled state");
        TYLOINFER_RUNTIME_CHECK(capacity>=position_ && capacity<=maximum_, "invalid fixed sequence cache capacity");
        if(values_.size(1)==capacity)return;
        auto expanded=::tyloi::cuda::zeros({values_.size(0),capacity,width_},values_.dense().options());
        if(position_)expanded.narrow(1,0,position_).copy_(values_.dense().narrow(1,0,position_));
        values_=std::move(expanded);
    }
    /** Write rows into prepared backing without advancing the committed position. */
    DenseTensor append_fixed(const DenseTensor& value,const DenseTensor& positions) {
        TYLOINFER_RUNTIME_CHECK(values_.defined() && values_.size(1)<=maximum_ && value.dim()==3 &&
            value.size(1)<=values_.size(1)-position_ &&
            value.size(0)==values_.size(0) && value.size(2)==width_ && positions.dim()==1 &&
            positions.size(0)==value.size(1), "fixed sequence cache append geometry mismatch");
        values_.dense().index_copy_(1,positions.to(::tyloi::kInt64),value.to(dtype_));
        return values_.dense();
    }
    void advance_fixed(int64_t tokens) {
        TYLOINFER_RUNTIME_CHECK(tokens>0 && tokens<=maximum_-position_,"fixed sequence cache exceeds context");
        position_+=tokens;
    }
    void truncate(int64_t keep) {
        TYLOINFER_RUNTIME_CHECK(keep >= 0 && keep <= position_, "invalid sequence cache truncation");
        position_ = keep;
    }
    /** Return new backing bytes needed for an append, without allocating. */
    std::size_t growth(std::int64_t batch, std::int64_t tokens) const {
        TYLOINFER_RUNTIME_CHECK(batch > 0 && tokens > 0, "invalid sequence cache append");
        TYLOINFER_RUNTIME_CHECK(!values_.defined() || batch == values_.size(0),
            "reset sequence cache before changing batch/device");
        TYLOINFER_RUNTIME_CHECK(tokens <= maximum_ - position_, "sequence cache exceeds max_context");
        const auto required = position_ + tokens;
        if (values_.defined() && required <= values_.size(1)) return 0;
        const auto capacity = capacity_for(required);
        auto bytes = ::tyloi::scalar_size(dtype_);
        for (const auto dimension : {batch, capacity, width_}) {
            if (bytes > std::numeric_limits<std::size_t>::max() / std::size_t(dimension))
                throw std::overflow_error("sequence cache allocation size overflow");
            bytes *= std::size_t(dimension);
        }
        return bytes;
    }
    /** Append rows, growing backing if needed, and advance the committed position. */
    DenseTensor append(const DenseTensor& value) {
        TYLOINFER_RUNTIME_CHECK(value.is_cuda() && value.dim() == 3 && value.size(0) > 0 &&
            value.size(1) > 0 && value.size(2) == width_, "invalid sequence cache append");
        TYLOINFER_RUNTIME_CHECK(!values_.defined() || (value.size(0) == values_.size(0) &&
            value.device() == values_.device()), "reset sequence cache before changing batch/device");
        TYLOINFER_RUNTIME_CHECK(value.size(1) <= maximum_ - position_, "sequence cache exceeds max_context");
        const auto required = position_ + value.size(1);
        if (!values_.defined() || required > values_.size(1)) {
            const auto capacity = capacity_for(required);
            auto expanded = ::tyloi::cuda::zeros({value.size(0), capacity, width_}, value.options().dtype(dtype_));
            if (position_) expanded.narrow(1, 0, position_).copy_(values_.dense().narrow(1, 0, position_));
            values_ = std::move(expanded);
        }
        values_.dense().narrow(1, position_, value.size(1)).copy_(value.to(dtype_));
        position_ = required;
        return values_.dense().narrow(1, 0, position_);
    }
private:
    int64_t capacity_for(int64_t required) const {
        int64_t capacity = values_.defined() ? values_.size(1) : std::min<int64_t>(16, maximum_);
        while (capacity < required) capacity = std::min(maximum_, capacity > maximum_ / 2 ? maximum_ : capacity * 2);
        return capacity;
    }
    int64_t maximum_, width_, position_ = 0;
    ::tyloi::ScalarType dtype_;
    ::tyloi::Tensor values_;
};

} // namespace tyloi::cuda
