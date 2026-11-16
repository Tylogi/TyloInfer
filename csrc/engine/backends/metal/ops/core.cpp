#include "engine/ops/ops.h"
#include "engine/backends/metal/executor/executor.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <utility>

namespace tyloi::ops {
namespace {
const Device cpu{DeviceType::cpu, 0}, gpu{DeviceType::metal, 0};
struct Capture { bool selected=false, active=false; std::map<std::string, std::pair<std::size_t,double>> timings; };
thread_local Capture* capture = nullptr;
void report_capture(const Capture& value, const std::string& name) {
    for (const auto& [label, data] : value.timings)
        std::cerr << "profile_item report=" << name << " name=" << label << " calls=" << data.first << " elapsed_ms=" << data.second << '\n';
}
} // namespace

std::unique_ptr<::tyloi::Executor::Scope> dense_scope(::tyloi::Executor* executor,Device target) {
    if(!executor) return metal::Executor::current().enter();
    if(target!=gpu)throw std::invalid_argument("dense scope requires Metal 0");
    auto* backend=dynamic_cast<metal::Executor*>(executor);
    if(!backend)throw std::invalid_argument("dense scope requires a Metal executor");
    return backend->enter();
}
struct ProfileRange::Impl {
    Capture* owner;
    std::string name;
    std::unique_ptr<::tyloi::Executor::Timer> timer;
    bool finished=false;
    Impl(Capture* capture,const char* label) : owner(capture), name(label), timer(metal::Executor::current().timer()) {}
    void finish() {
        if(finished)return;
        timer->stop();auto& entry=owner->timings[name];++entry.first;entry.second+=timer->elapsed_ms();finished=true;
    }
};
ProfileRange::ProfileRange(const char* name,std::optional<Device> device) {
    if(device && *device!=gpu && *device!=cpu)throw std::invalid_argument("unsupported profile device");
    if(capture && capture->active)impl_=std::make_unique<Impl>(capture,name);
}
ProfileRange::~ProfileRange() { if(impl_ && !impl_->finished)try { impl_->finish(); } catch(...) {} }
void ProfileRange::finish() { if(impl_)impl_->finish(); }
struct ProfileCapture::Impl {
    Capture state;
    Capture* previous;
    std::string report_name;
    Impl(bool selected,std::string name) : previous(capture),report_name(std::move(name)) {
        state.selected=selected;state.active=selected;
        if(selected)capture=&state;
    }
    ~Impl() {
        if(state.selected && !report_name.empty() && std::uncaught_exceptions()==0)report_capture(state,report_name);
        capture=previous;
    }
};
ProfileCapture::ProfileCapture(bool selected) : impl_(std::make_unique<Impl>(selected,std::string{})) {}
ProfileCapture::ProfileCapture(bool selected,const std::string& name) : impl_(std::make_unique<Impl>(selected,name)) {}
ProfileCapture::~ProfileCapture() = default;
void ProfileCapture::report(const std::string& name) { if(impl_ && impl_->state.selected)report_capture(impl_->state,name); }
void ProfileCapture::resume() { if(impl_)impl_->state.active=impl_->state.selected; }
void ProfileCapture::suspend() { if(impl_)impl_->state.active=false; }
ProfileCapture ProfileCapture::prefill_once(std::int64_t tokens) {
    const auto* option=std::getenv("TYLOINFER_PROFILE_PREFILL_ONCE");
    const bool enabled=option && option[0]=='1';
    const auto* skip=std::getenv("TYLOINFER_PROFILE_PREFILL_SKIP");
    const auto skipped=skip ? std::max<long long>(0,std::atoll(skip)) : 0;
    static std::atomic<std::uint64_t> requests{0};
    const auto index=enabled ? requests.fetch_add(1) : 0;
    const bool selected=enabled && index==std::uint64_t(skipped);
    if(enabled)std::cerr << "prefill_profile_request index=" << index << " tokens=" << tokens << " collect=" << int(selected) << '\n';
    return ProfileCapture(selected,"prefill_regression_once");
}
} // namespace tyloi::ops
