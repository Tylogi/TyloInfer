#include "profile.h"

#include "tensor/tensor.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

namespace tyloi::metal {
namespace {

using Clock = std::chrono::steady_clock;

void materialize(Stream& stream, std::initializer_list<Tensor> roots) {
    std::vector<mlx::core::array> arrays;
    arrays.reserve(roots.size());
    for (const auto& root : roots) {
        if (!root.defined()) continue;
        if (root.device() == Device{DeviceType::cpu, 0} && !root.is_encoded() &&
            root.representation<DenseTensor>()) continue;
        if (root.device() != Device{DeviceType::metal, 0} || root.is_encoded())
            throw std::invalid_argument("profiling root must be a dense Metal 0 array");
        arrays.push_back(metal::array(root));
    }
    if (!arrays.empty()) {
        StreamGuard guard(stream);
        tyloi::profiling::measure([&] { metal::eval(std::move(arrays)); });
    }
    drain(stream);
}

class Timer final : public tyloi::Executor::Timer {
public:
    explicit Timer(std::shared_ptr<Stream> stream) : stream_(std::move(stream)) {
        if (!stream_) throw std::invalid_argument("Metal timer requires a stream");
        drain(*stream_);
        started_ = Clock::now();
    }

    void stop() override { stop({}); }
    void stop(std::initializer_list<Tensor> roots) override {
        if (stopped_) return;
        materialize(*stream_, roots);
        elapsed_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - started_).count();
        stopped_ = true;
    }
    double elapsed_ms() const override {
        if (!stopped_) throw std::logic_error("Metal timer has not been stopped");
        return elapsed_ms_;
    }

private:
    std::shared_ptr<Stream> stream_;
    Clock::time_point started_;
    double elapsed_ms_ = 0;
    bool stopped_ = false;
};

class Profile final : public tyloi::Executor::Profile {
public:
    Profile(std::shared_ptr<Stream> stream, const char* name)
        : stream_(std::move(stream)) {
        if (!stream_) throw std::invalid_argument("Metal profile requires a stream");
        if (!name || !*name) throw std::invalid_argument("Metal profile requires a name");
        name_ = name;
        drain(*stream_);
        started_ = Clock::now();
        profile_scope_ = std::make_unique<profiling::ProfileScope>(&components_);
        evaluation_scope_ = std::make_unique<profiling::EvaluationScope>(&evaluation_ms_);
    }

    void finish() override { finish({}); }
    void finish(std::initializer_list<Tensor> roots) override {
        if (finished_) return;
        try {
            materialize(*stream_, roots);
        } catch (...) {
            restore();
            finished_ = true;
            throw;
        }
        const auto wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - started_).count();
        restore();
        finished_ = true;
        std::cerr << "[metal profile] " << name_ << " wall_ms=" << wall_ms
                  << " explicit_eval_ms=" << evaluation_ms_ << '\n';
        for (const auto& [name, timing] : components_.timings())
            std::cerr << "[metal profile] " << name << " calls=" << timing.evaluations
                      << " elapsed_ms=" << timing.elapsed_ms << '\n';
    }

private:
    void restore() noexcept {
        evaluation_scope_.reset();
        profile_scope_.reset();
    }

    std::shared_ptr<Stream> stream_;
    std::string name_;
    profiling::Components components_;
    double evaluation_ms_ = 0;
    Clock::time_point started_;
    std::unique_ptr<profiling::ProfileScope> profile_scope_;
    std::unique_ptr<profiling::EvaluationScope> evaluation_scope_;
    bool finished_ = false;
};

} // namespace

std::unique_ptr<tyloi::Executor::Timer> timer(std::shared_ptr<Stream> stream) {
    return std::make_unique<Timer>(std::move(stream));
}

std::unique_ptr<tyloi::Executor::Profile> profile(std::shared_ptr<Stream> stream, const char* name) {
    return std::make_unique<Profile>(std::move(stream), name);
}

} // namespace tyloi::metal
