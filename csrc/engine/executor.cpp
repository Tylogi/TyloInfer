#include "engine/executor.h"

namespace tyloi {
void Executor::submit(std::span<const Tensor> outputs) {
    for (const auto& output : outputs)
        if (output.defined() && output.device().is_metal())
            throw std::invalid_argument("executor cannot submit lazy Metal tensors");
}
} // namespace tyloi

namespace tyloi::profiling {

thread_local Components* ProfileScope::active_ = nullptr;
thread_local double* EvaluationScope::active_ = nullptr;

void Components::record(std::string_view component, double elapsed_ms) {
    auto& timing = timings_[std::string(component)];
    timing.elapsed_ms += elapsed_ms;
    ++timing.evaluations;
}

const std::map<std::string, Timing>& Components::timings() const noexcept {
    return timings_;
}

double Components::evaluated_ms() const noexcept {
    double total = 0;
    for (const auto& [_, timing] : timings_) {
        total += timing.elapsed_ms;
    }
    return total;
}

ProfileScope::ProfileScope(Components* profile) noexcept : previous_(active_) {
    active_ = profile;
}

ProfileScope::~ProfileScope() {
    active_ = previous_;
}

EvaluationScope::EvaluationScope(double* elapsed_ms) noexcept : previous_(active_) {
    active_ = elapsed_ms;
}

EvaluationScope::~EvaluationScope() {
    active_ = previous_;
}

bool active() noexcept {
    return ProfileScope::active_ != nullptr;
}

void mark(std::string_view component) {
    if (ProfileScope::active_ != nullptr) {
        ProfileScope::active_->record(component, 0);
    }
}

} // namespace tyloi::profiling
