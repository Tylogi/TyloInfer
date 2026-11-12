#include "engine/tensor.h"

#include <unordered_set>

namespace tyloi {
namespace {

void count_storage(const std::shared_ptr<TensorStorage>& storage,
    std::unordered_set<const void*>& seen, std::size_t& total) {
    if (!storage) return;
    const void* identity = storage->owner ? storage->owner.get() : storage->base;
    if (!identity || !seen.insert(identity).second) return;
    total += storage->bytes;
}

}  // namespace

std::size_t backing_bytes(std::span<const DenseTensor* const> buffers) {
    std::size_t total = 0;
    std::unordered_set<const void*> seen;
    for (const auto* buffer : buffers) {
        if (!buffer || !buffer->defined()) continue;
        count_storage(buffer->storage(), seen, total);
    }
    return total;
}

std::size_t backing_bytes(std::span<const std::shared_ptr<TensorStorage>> storages) {
    std::size_t total = 0;
    std::unordered_set<const void*> seen;
    for (const auto& storage : storages) {
        count_storage(storage, seen, total);
    }
    return total;
}

}  // namespace tyloi
