#include "executor.h"

#include "engine/backends/metal/native/memory.h"
#include "engine/backends/metal/native/profile.h"
#include "engine/backends/metal/native/stream.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::metal {
namespace {

thread_local Executor* active_executor = nullptr;

void require_device(Device device) {
    if (device != Device{DeviceType::metal, 0})
        throw std::invalid_argument("Metal executor requires Metal device 0");
}

class ExecutorScope final : public tyloi::Executor::Scope {
public:
    explicit ExecutorScope(Executor& executor)
        : binding_(executor.stream()), previous_(active_executor) {
        active_executor = &executor;
    }
    ~ExecutorScope() override { active_executor = previous_; }

private:
    StreamGuard binding_;
    Executor* previous_;
};

} // namespace

Executor::Executor(std::shared_ptr<Stream> stream) : stream_(std::move(stream)) {
    if (!stream_) throw std::invalid_argument("Metal executor requires a stream");
}

Executor& Executor::current() {
    if (!active_executor) throw std::logic_error("no active Metal executor scope");
    return *active_executor;
}

void Executor::require_current() const {
    if (active_executor != this) throw std::logic_error("Metal executor is not active");
}

std::unique_ptr<tyloi::Executor::Scope> Executor::enter() {
    return std::make_unique<ExecutorScope>(*this);
}

std::unique_ptr<tyloi::Executor::Timer> Executor::timer() {
    require_current();
    return metal::timer(stream_);
}

std::unique_ptr<tyloi::Executor::Profile> Executor::profile(const char* name) {
    require_current();
    return metal::profile(stream_, name);
}

std::unique_ptr<tyloi::Executor::Graph> Executor::graph() {
    return {};
}

std::vector<tyloi::Executor::Memory> Executor::memory(std::span<const Device> devices) const {
    require_current();
    for (const auto device : devices)
        if (device != Device{DeviceType::cpu, 0}) require_device(device);
    if (devices.empty()) return {};
    const auto snapshot = metal::memory();
    const auto unlimited = std::numeric_limits<std::size_t>::max();
    if (snapshot.active > unlimited - snapshot.cache)
        throw std::overflow_error("Metal allocator accounting overflow");
    const auto reserved = snapshot.active + snapshot.cache;
    const auto resident = std::max(reserved, snapshot.footprint);
    const auto headroom = snapshot.limit > resident ? snapshot.limit - resident : 0;
    std::vector<tyloi::Executor::Memory> result;
    result.reserve(devices.size());
    for (const auto device : devices)
        result.push_back({device, snapshot.active, reserved, snapshot.available, snapshot.total,
            snapshot.limit == unlimited ? 0 : snapshot.limit, resident,
            std::min(snapshot.available, headroom), 0, Device{DeviceType::metal, 0}, snapshot.wired});
    return result;
}

void Executor::reclaim(Device device) {
    require_current();
    require_device(device);
    drain();
    metal::clear_cache();
}

void Executor::set_memory_limit(Device device, std::size_t bytes) {
    require_current();
    require_device(device);
    drain();
    if (bytes && metal::active_memory() > bytes)
        throw std::invalid_argument("Metal memory limit is smaller than live allocations");
    metal::clear_cache();
    if (bytes && metal::memory().footprint > bytes)
        throw std::invalid_argument("Metal memory limit is smaller than the process footprint");
    metal::set_memory_limit(bytes ? bytes : std::numeric_limits<std::size_t>::max());
}

void Executor::drain() {
    require_current();
    metal::drain(*stream_);
}

void Executor::clear() {
    require_current();
    drain();
    metal::clear_cache();
}

} // namespace tyloi::metal
