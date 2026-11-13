#include "allocation.h"
#include "resident_allocations.h"

namespace tyloi::metal {

void AllocationScope::Log::clear() {
    std::lock_guard lock(mutex);
    allocations.clear();
    pending_bytes = 0;
    recording = true;
}

AllocationScope::Log*& AllocationScope::current() noexcept {
    static thread_local Log* value = nullptr;
    return value;
}

AllocationScope::AllocationScope(Log* log) noexcept : previous_(current()) {
    current() = log;
}

AllocationScope::~AllocationScope() { current() = previous_; }

AllocationScope::Log* AllocationScope::log() noexcept { return current(); }

void AllocationScope::record(const mlx::core::array& value) {
    auto* log = current();
    if (!log || !value.data_shared_ptr()) return;
    std::unique_lock lock(log->mutex);
    if (!log->recording) return;
    log->allocations.push_back(value.data_shared_ptr());
    log->pending_bytes += value.buffer_size();
    if (log->pending_bytes < (std::size_t{4} << 30)) return;
    log->pending_bytes = 0;
    lock.unlock();
    ResidentAllocations::activate();
}

} // namespace tyloi::metal
