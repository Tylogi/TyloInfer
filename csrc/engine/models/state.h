#pragma once

#include "engine/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::models {

struct ForwardOptions;
struct Input;
struct Mtp;

class MtpState {
public:
    explicit MtpState(const Mtp& model) : model_(&model) {}
    virtual ~MtpState() = default;
    MtpState(const MtpState&) = delete;
    MtpState& operator=(const MtpState&) = delete;
    bool belongs_to(const Mtp& model) const noexcept { return model_ == &model; }
    virtual std::int64_t position() const noexcept = 0;
private:
    const Mtp* model_;
};

class State {
public:
    State() = default;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    virtual ~State() = default;
    std::unique_ptr<MtpState> predictor;
    virtual void preallocate() = 0;
    virtual std::vector<std::pair<tyloi::Device, std::size_t>> preallocate_growth() const {
        throw std::logic_error("state preallocation planning is not implemented");
    }
    virtual void reset(std::int64_t batch) = 0;
    virtual std::int64_t position() const noexcept = 0;
    virtual void advance(std::int64_t tokens) noexcept = 0;
    virtual void set_position_delta(std::int64_t delta) noexcept = 0;
    // Additional backing for this input after reset(batch), including replacements and verification checkpoints.
    // Suffix setup, Graph preparation and session copies plan at their own allocation boundaries.
    virtual std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        const Input&, const ForwardOptions&) const {
        throw std::logic_error("state allocation planning is not implemented");
    }
    virtual void begin_speculative_suffix(std::int64_t) {
        throw std::logic_error("speculative suffix is not supported");
    }
    virtual void commit_speculative() {
        throw std::logic_error("speculative transactions are not supported");
    }
    virtual void rollback_speculative(std::int64_t) {
        throw std::logic_error("speculative transactions are not supported");
    }
    virtual bool retains_partial_prefix(std::int64_t) const noexcept { return false; }
    // Owners keep storage alive; deduplicate using backing identity, not object identity.
    virtual std::vector<::tyloi::Tensor> storage() const = 0;
    // Borrow writable objects from this state; their lifetime remains tied to it.
    virtual std::vector<tyloi::Tensor*> graph_state() = 0;
    virtual bool supports_text_session_state() const = 0;
    virtual bool supports_paged_text_session_state() const = 0;
};

} // namespace tyloi::models
