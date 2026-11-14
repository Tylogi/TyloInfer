#include "engine/ops.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/decode_window.h"
#include "engine/backends/cuda/ops/profile.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/profile.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <optional>
#include <utility>

namespace tyloi::ops {
struct ProfileRange::Impl {
    std::optional<cuda::DeviceGuard> guard;
    std::optional<CudaProfiler::Range> range;
    Impl(CudaProfiler& profiler, const char* name, std::optional<Device> device) {
        if (device && device->is_cuda()) guard.emplace(device->index);
        range.emplace(profiler, name);
    }
};
ProfileRange::ProfileRange(const char* name, std::optional<Device> device) {
    auto& profiler = cuda::Executor::current().profiler();
    if (profiler.enabled && profiler.selected(name))
        impl_ = std::make_unique<Impl>(profiler, name, device);
}
ProfileRange::~ProfileRange() = default;
void ProfileRange::finish() { if (impl_) impl_->range->finish(); }
struct ProfileCapture::Impl {
    CudaProfiler& profiler;
    bool previous, enabled, selected;
    std::string report_name;
    Impl(CudaProfiler& profiler, bool selected, std::string report_name)
        : profiler(profiler), previous(profiler.enabled), enabled(selected || profiler.enabled),
          selected(selected), report_name(std::move(report_name)) {
        if (selected) {
            profiler.reset();
            profiler.enabled = true;
        }
    }
    ~Impl() {
        if (selected && !report_name.empty() && std::uncaught_exceptions() == 0) {
            try {
                profiler.report(report_name);
                profiler.reset();
            } catch (const std::exception& error) {
                std::cerr << "prefill_profile_error " << error.what() << '\n';
            }
        }
        profiler.enabled = previous;
    }
};
ProfileCapture::ProfileCapture(bool selected)
    : impl_(selected ? std::make_unique<Impl>(cuda::Executor::current().profiler(), true, std::string{}) : nullptr) {}
ProfileCapture::ProfileCapture(bool selected, const std::string& report_name)
    : impl_(std::make_unique<Impl>(cuda::Executor::current().profiler(), selected, report_name)) {}
ProfileCapture::~ProfileCapture() = default;
void ProfileCapture::report(const std::string& name) {
    if (impl_ && impl_->selected) impl_->profiler.report(name);
}
void ProfileCapture::resume() {
    if (impl_) impl_->profiler.enabled = impl_->enabled;
}
void ProfileCapture::suspend() {
    if (impl_) impl_->profiler.enabled = impl_->previous;
}
ProfileCapture ProfileCapture::prefill_once(std::int64_t tokens) {
    const auto* option = cuda::env_lookup("TYLOINFER_PROFILE_PREFILL_ONCE");
    const bool enabled = option && option[0] == '1';
    const auto* skip = cuda::env_lookup("TYLOINFER_PROFILE_PREFILL_SKIP");
    const auto skipped = skip ? std::max<long long>(0, std::atoll(skip)) : 0;
    static std::atomic<std::uint64_t> requests{0};
    const auto index = enabled ? requests.fetch_add(1) : 0;
    const bool selected = enabled && index == std::uint64_t(skipped);
    if (enabled) std::cerr << "prefill_profile_request index=" << index
        << " tokens=" << tokens << " collect=" << int(selected) << '\n';
    return ProfileCapture(selected, "prefill_regression_once");
}
}

void tyloi::cuda::append_graph_profile_metrics(const DecodeWindow& window, const std::string& prefix,
        std::vector<std::pair<std::string, double>>& result) {
    struct Sample { std::int64_t total_ns = 0, max_ns = 0; std::uint64_t samples = 0; };
    std::map<std::string, Sample> phases;
    std::uint64_t windows = 0;
    for (const auto& phase : window.gpu_phases()) if (phase.samples) {
        auto& sample = phases[phase.from + "_to_" + phase.to];
        sample.total_ns += phase.total_ns;
        sample.samples += phase.samples;
        sample.max_ns = std::max(sample.max_ns, phase.max_ns);
        windows = std::max(windows, phase.samples);
    }
    if (!windows) return;
    result.emplace_back(prefix + "windows", double(windows));
    for (const auto& [name, sample] : phases) {
        result.emplace_back(prefix + name + "_ms", sample.total_ns / 1e6);
        result.emplace_back(prefix + name + "_samples", double(sample.samples));
        result.emplace_back(prefix + name + "_max_ms", sample.max_ns / 1e6);
    }
}
