#include "engine/state/recurrent.h"

#include "engine/ops.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::state {

namespace {
::tyloi::Tensor storage(const ::tyloi::Tensor& target,
                        const ::tyloi::Tensor& source) {
    if (target.defined() && target.dim() == source.dim() &&
        target.scalar_type() == source.scalar_type() && target.device() == source.device()) {
        bool reuse = true;
        for (std::int64_t axis = 0; axis < source.dim(); ++axis)
            if (target.size(axis) != source.size(axis)) { reuse = false; break; }
        if (reuse) return target;
    }
    std::vector<std::int64_t> shape;
    shape.reserve(source.dim());
    for (std::int64_t axis = 0; axis < source.dim(); ++axis)
        shape.push_back(source.size(axis));
    return ::tyloi::ops::empty(shape, source.options());
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

Recurrent::State Recurrent::clone(const State& source) {
    if (!source.defined()) throw std::runtime_error("recurrent snapshot is unavailable");
    return {::tyloi::ops::clone(source.convolution), ::tyloi::ops::clone(source.recurrent), source.transposed};
}

Recurrent::State Recurrent::snapshot() const {
    if (!(defined() && !pending()))
        throw std::runtime_error("recurrent snapshot requires confirmed state");
    return clone(state_);
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

void Recurrent::begin(int64_t start, int64_t confirmed, int64_t tokens, State checkpoint) {
    if (!(start >= 0 && confirmed > 0 && confirmed < tokens))
        throw std::runtime_error("invalid recurrent verification window");
    if (!(defined() && !pending()))
        throw std::runtime_error("recurrent snapshot requires confirmed state");
    const auto matches = [](const Tensor& live, const Tensor& saved) {
        if (!live.defined() || !saved.defined() || live.is_encoded() || saved.is_encoded() ||
            live.dim() != saved.dim() || live.scalar_type() != saved.scalar_type() || live.device() != saved.device())
            return false;
        for (int64_t axis = 0; axis < live.dim(); ++axis)
            if (live.size(axis) != saved.size(axis)) return false;
        return true;
    };
    if (state_.transposed != checkpoint.transposed ||
        !matches(state_.convolution, checkpoint.convolution) ||
        !matches(state_.recurrent, checkpoint.recurrent))
        throw std::invalid_argument("recurrent checkpoint topology mismatch");
    saved_ = std::move(checkpoint);
    start_ = start;
    confirmed_ = confirmed;
    tokens_ = tokens;
}

Recurrent::Checkpoint Recurrent::checkpoint() const {
    if (!pending()) throw std::runtime_error("recurrent checkpoint is unavailable");
    return {saved_, start_, confirmed_, tokens_};
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
