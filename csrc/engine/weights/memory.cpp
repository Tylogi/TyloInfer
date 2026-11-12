#include "engine/weights/memory.h"

#include "engine/backends/cuda/native/allocation.h"
#include "engine/tensor.h"

#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace tyloi::weights {
namespace {
thread_local Memory* loading_weight_memory = nullptr;
}

void Memory::observe(const Tensor& tensor) {
    for (const auto& storage : tensor.backings()) {
        if (!storage) continue;
        if (storage->owner)
            backings_.push_back({storage->owner, storage->owner.get(), storage->device, storage->bytes});
        else
            weights_.push_back(storage);
    }
}

void Memory::observe_ple(const std::shared_ptr<void>& owner, std::size_t bytes) {
    ple_.push_back({owner, bytes});
}

std::size_t Memory::resident_bytes(int device) const {
    return resident_bytes(device, nullptr);
}

std::size_t Memory::resident_bytes(Device device) const {
    return resident_bytes(-1, &device);
}

std::size_t Memory::resident_bytes(int index, const Device* device) const {
    std::size_t bytes = 0;
    std::unordered_set<const void*> seen;
    const auto count = [&](const void* identity, Device actual, std::size_t size) {
        if (identity && (index < 0 || actual.index == index) && (!device || actual == *device) &&
            seen.insert(identity).second) {
            if (size > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::overflow_error("resident weight byte count overflow");
            bytes += size;
        }
    };
    for (const auto& weak : weights_)
        if (auto storage = weak.lock())
            count(storage->owner ? storage->owner.get() : storage->base, storage->device, storage->bytes);
    for (const auto& backing : backings_)
        if (auto owner = backing.owner.lock())
            count(backing.identity, backing.device, backing.bytes);
    return bytes;
}

std::size_t Memory::ple_bytes() const {
    std::size_t bytes = 0;
    for (const auto& payload : ple_)
        if (!payload.owner.expired())
            bytes += payload.bytes;
    return bytes;
}

LoadScope::LoadScope(Memory& ledger)
    : previous_(loading_weight_memory), allocation_(std::make_shared<tyloi::cuda::AllocationScope>(ledger.weights_)) {
    loading_weight_memory = &ledger;
}

LoadScope::~LoadScope() {
    loading_weight_memory = previous_;
}

void observe_streamed_ple(const std::shared_ptr<void>& owner, std::size_t bytes) {
    if (loading_weight_memory)
        loading_weight_memory->observe_ple(owner, bytes);
}

void observe_weight(const Tensor& tensor) {
    if (loading_weight_memory)
        loading_weight_memory->observe(tensor);
}
}
