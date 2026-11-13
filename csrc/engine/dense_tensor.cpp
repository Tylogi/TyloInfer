// DenseTensor numeric convenience members forward to public ops.
#include "engine/tensor.h"
#include "engine/ops.h"

#include <initializer_list>
#include <span>
#include <stdexcept>

namespace tyloi {
namespace {

DenseTensor dense_result(const Tensor& result) {
    if (!result.defined()) throw std::invalid_argument("operation returned an undefined tensor");
    return result.dense();
}

}  // namespace

DenseTensor DenseTensor::index_select(std::int64_t dimension, const DenseTensor& indices) const {
    return dense_result(ops::index_select(*this, dimension, indices));
}

DenseTensor DenseTensor::gather(std::int64_t dimension, const DenseTensor& indices) const {
    return dense_result(ops::gather(*this, dimension, indices));
}

DenseTensor DenseTensor::repeat(std::span<const std::int64_t> repeats) const {
    return dense_result(ops::repeat(*this, repeats));
}

DenseTensor DenseTensor::repeat_interleave(std::int64_t repeats, std::int64_t dimension) const {
    return dense_result(ops::repeat_interleave(*this, repeats, dimension));
}

DenseTensor DenseTensor::mean(std::int64_t dimension, bool keep_dimension) const {
    return dense_result(ops::mean(*this, dimension, keep_dimension));
}

DenseTensor DenseTensor::mean() const { return reshape({-1}).mean(0, false); }
DenseTensor DenseTensor::sum(std::int64_t dimension, bool keep_dimension) const {
    return dense_result(ops::sum(*this, dimension, keep_dimension));
}
DenseTensor DenseTensor::sum() const { return reshape({-1}).sum(0, false); }
DenseTensor DenseTensor::max() const { return reshape({-1}).amax(0, false); }
DenseTensor DenseTensor::amax(std::int64_t dimension, bool keep_dimension) const {
    return dense_result(ops::amax(*this, dimension, keep_dimension));
}
DenseTensor DenseTensor::norm() const { return square().sum().sqrt(); }
DenseTensor DenseTensor::abs() const { return dense_result(ops::abs(*this)); }
DenseTensor DenseTensor::square() const { return dense_result(ops::square(*this)); }
DenseTensor DenseTensor::exp() const { return dense_result(ops::exp(*this)); }
DenseTensor DenseTensor::sqrt() const { return dense_result(ops::sqrt(*this)); }
DenseTensor DenseTensor::clamp(double minimum, double maximum) const {
    return dense_result(ops::clamp(*this, minimum, maximum));
}
DenseTensor DenseTensor::clamp_min(double minimum) const {
    return dense_result(ops::clamp_min(*this, minimum));
}
DenseTensor DenseTensor::clamp_max(double maximum) const {
    return dense_result(ops::clamp_max(*this, maximum));
}
DenseTensor DenseTensor::remainder(double divisor) const {
    return dense_result(ops::remainder(*this, divisor));
}
DenseTensor DenseTensor::argmax(std::int64_t dimension, bool keep_dimension) const {
    return dense_result(ops::argmax(*this, dimension, keep_dimension));
}
DenseTensor DenseTensor::eq(const DenseTensor& other) const { return dense_result(ops::eq(*this, other)); }
DenseTensor DenseTensor::eq(double other) const { return dense_result(ops::eq(*this, other)); }
DenseTensor DenseTensor::ne(const DenseTensor& other) const { return dense_result(ops::ne(*this, other)); }
DenseTensor DenseTensor::ne(double other) const { return dense_result(ops::ne(*this, other)); }
DenseTensor DenseTensor::all() const { return reshape({-1}).all(0, false); }
DenseTensor DenseTensor::all(std::int64_t dimension, bool keep_dimension) const {
    return dense_result(ops::all(*this, dimension, keep_dimension));
}
DenseTensor DenseTensor::any(std::int64_t dimension, bool keep_dimension) const {
    return dense_result(ops::any(*this, dimension, keep_dimension));
}
DenseTensor DenseTensor::logical_not() const { return dense_result(ops::logical_not(*this)); }
bool DenseTensor::equal(const DenseTensor& other) const { return ops::equal(*this, other); }
DenseTensor DenseTensor::masked_select(const DenseTensor& mask) const { return dense_result(ops::masked_select(*this, mask)); }
DenseTensor DenseTensor::tril(std::int64_t diagonal) const { return dense_result(ops::tril(*this, diagonal)); }
DenseTensor DenseTensor::masked_fill(const DenseTensor& mask, double value) const {
    return dense_result(ops::masked_fill(*this, mask, value));
}
DenseTensor& DenseTensor::masked_fill_(const DenseTensor& mask, double value) {
    Tensor self{*this};
    ops::masked_fill_inplace(self, mask, value);
    return *this;
}

DenseTensor DenseTensor::contiguous() const {
    return dense_result(ops::contiguous(*this));
}

DenseTensor DenseTensor::clone() const {
    return dense_result(ops::clone(*this));
}

DenseTensor DenseTensor::to(ScalarType type) const {
    return dense_result(ops::cast(*this, type));
}

DenseTensor DenseTensor::to(Device target_device) const {
    return dense_result(ops::copy(*this, target_device));
}

DenseTensor& DenseTensor::copy_(const DenseTensor& source, bool) {
    Tensor self{*this};
    ops::copy_into(self, source);
    return *this;
}

DenseTensor& DenseTensor::zero_() {
    return fill_(0.0);
}

DenseTensor& DenseTensor::fill_(double value) {
    Tensor self{*this};
    ops::fill(self, value);
    return *this;
}

DenseTensor& DenseTensor::add_(const DenseTensor& value) {
    auto result = ops::add(*this, value);
    Tensor self{*this};
    ops::copy_into(self, result);
    return *this;
}

DenseTensor& DenseTensor::scatter_add_(
    std::int64_t dimension, const DenseTensor& index, const DenseTensor& source) {
    Tensor self{*this};
    ops::scatter_add(self, dimension, index, source);
    return *this;
}

DenseTensor& DenseTensor::scatter_(
    std::int64_t dimension, const DenseTensor& index, const DenseTensor& source) {
    Tensor self{*this};
    ops::scatter(self, dimension, index, source);
    return *this;
}

DenseTensor& DenseTensor::index_copy_(
    std::int64_t dimension, const DenseTensor& index, const DenseTensor& source) {
    Tensor self{*this};
    ops::index_copy(self, dimension, index, source);
    return *this;
}

DenseTensor& DenseTensor::index_fill_(
    std::int64_t dimension, const DenseTensor& index, double value) {
    Tensor self{*this};
    ops::index_fill(self, dimension, index, value);
    return *this;
}

DenseTensor& DenseTensor::index_put_(
    std::initializer_list<TensorIndex> indices, const DenseTensor& value) {
    value.require_dense();
    Tensor view{index(indices)};
    ops::copy_into(view, value);
    return *this;
}

DenseTensor& DenseTensor::index_put_(
    std::initializer_list<TensorIndex> indices, double value) {
    index(indices).fill_(value);
    return *this;
}

}  // namespace tyloi
