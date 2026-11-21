#pragma once
#include <cstddef>
#include <optional>

namespace mfq::cuda {
// Remaining WDDM shared-memory budget; absent on platforms without WDDM.
std::optional<std::size_t> shared_memory_available(int device);
// Keep dense prefill scratch independently reclaimable at the decode boundary.
class PrefillAllocationScope {
public:
    explicit PrefillAllocationScope(bool enabled,bool reuse_ordinary_pool=false) noexcept;
    ~PrefillAllocationScope() noexcept;
    PrefillAllocationScope(const PrefillAllocationScope&)=delete;
    PrefillAllocationScope& operator=(const PrefillAllocationScope&)=delete;
    static bool active() noexcept;
private:
    bool previous_,previous_isolation_;
};
// Keep long-lived request state out of the temporary async allocator pool.
class ResidentAllocationScope {
public:
    explicit ResidentAllocationScope(bool enabled=true) noexcept;
    ~ResidentAllocationScope() noexcept { finish(); }
    ResidentAllocationScope(const ResidentAllocationScope&)=delete;
    ResidentAllocationScope& operator=(const ResidentAllocationScope&)=delete;
    void finish() noexcept;
private:
    bool previous_,active_=true;
};
// Evictable routed images must not share allocator pages with model weights.
class RoutedWeightAllocationScope {
public:
    explicit RoutedWeightAllocationScope(bool enabled=true) noexcept;
    ~RoutedWeightAllocationScope() noexcept;
    RoutedWeightAllocationScope(const RoutedWeightAllocationScope&)=delete;
    RoutedWeightAllocationScope& operator=(const RoutedWeightAllocationScope&)=delete;
private:
    ResidentAllocationScope resident_;
    bool previous_;
};
// Keep temporarily movable dense matrices on their own allocator pages.
class DenseWeightAllocationScope {
public:
    DenseWeightAllocationScope() noexcept;
    ~DenseWeightAllocationScope() noexcept;
    DenseWeightAllocationScope(const DenseWeightAllocationScope&)=delete;
    DenseWeightAllocationScope& operator=(const DenseWeightAllocationScope&)=delete;
private:
    ResidentAllocationScope resident_;
    bool previous_;
};
// Migratable request state must be able to release pages independently of
// model weights and the persistent attention tables.
class RequestStateAllocationScope {
public:
    enum class Kind { Recurrent, Indexer };
    explicit RequestStateAllocationScope(bool enabled=true,Kind kind=Kind::Recurrent) noexcept;
    ~RequestStateAllocationScope() noexcept;
    RequestStateAllocationScope(const RequestStateAllocationScope&)=delete;
    RequestStateAllocationScope& operator=(const RequestStateAllocationScope&)=delete;
private:
    ResidentAllocationScope resident_;
    std::optional<Kind> previous_;
};
}
