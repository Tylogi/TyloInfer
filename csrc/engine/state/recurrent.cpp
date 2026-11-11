#include "engine/state/recurrent.h"

#include "engine/ops.h"

#include <stdexcept>
#include <utility>

namespace tyloi::state {

namespace {
::tyloi::Tensor storage(const ::tyloi::Tensor& target,
                        const ::tyloi::Tensor& source) {
    const auto& source_dense = source.dense();
    if (target.defined() && target.dense().sizes() == source_dense.sizes() &&
        target.scalar_type() == source.scalar_type() && target.device() == source.device())
        return target;
    return ::tyloi::ops::empty(source_dense.sizes(), source.options());
}
} // namespace

void Recurrent::reset(int64_t batch) {
    commit();
    if (defined() && state_.convolution.size(0) == batch && state_.recurrent.size(0) == batch) {
        ::tyloi::ops::fill(state_.convolution, 0);
        ::tyloi::ops::fill(state_.recurrent, 0);
    } else {
        state_ = {};
    }
}

Recurrent::State Recurrent::snapshot() const {
    if (!(defined() && !pending()))
        throw std::runtime_error("recurrent snapshot requires confirmed state");
    return {::tyloi::ops::clone(state_.convolution), ::tyloi::ops::clone(state_.recurrent), state_.transposed};
}

void Recurrent::copy(const State& source) {
    if (!source.defined()) throw std::runtime_error("recurrent snapshot is unavailable");
    // Prepare both allocations before changing either live tensor.
    auto convolution = storage(state_.convolution, source.convolution);
    auto recurrent = storage(state_.recurrent, source.recurrent);
    ::tyloi::ops::copy_into(convolution, source.convolution);
    ::tyloi::ops::copy_into(recurrent, source.recurrent);
    state_ = {std::move(convolution), std::move(recurrent), source.transposed};
}

void Recurrent::restore(const State& snapshot) {
    copy(snapshot);
    commit();
}

void Recurrent::begin(int64_t start, int64_t confirmed, int64_t tokens) {
    if (!(start >= 0 && confirmed > 0 && confirmed < tokens))
        throw std::runtime_error("invalid recurrent verification window");
    auto saved = snapshot();
    saved_ = std::move(saved);
    start_ = start;
    confirmed_ = confirmed;
    tokens_ = tokens;
}

int64_t Recurrent::rewind(int64_t position) {
    if (!(pending() && position >= start_ &&
        position - start_ >= confirmed_ && position - start_ < tokens_))
        throw std::runtime_error("invalid recurrent rollback position");
    copy(saved_);
    return position - start_;
}

void Recurrent::abort() {
    if (!pending()) return;
    copy(saved_);
    commit();
}

void Recurrent::commit() noexcept {
    saved_ = {};
    start_ = -1;
    confirmed_ = 0;
    tokens_ = 0;
}

} // namespace tyloi::state
