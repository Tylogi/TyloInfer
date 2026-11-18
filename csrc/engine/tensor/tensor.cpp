#include "engine/tensor/tensor.h"

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

bool Tensor::same_view(const Tensor& other) const {
    if (impl_ == other.impl_) return true;
    const auto* first = representation<DenseTensor>();
    const auto* second = other.representation<DenseTensor>();
    if (!first || !second || !first->defined() || !second->defined() ||
        first->device() != second->device() || first->scalar_type() != second->scalar_type() ||
        first->data_ptr() != second->data_ptr() || first->dim() != second->dim()) return false;
    for (std::int64_t axis = 0; axis < first->dim(); ++axis)
        if (first->size(axis) != second->size(axis) || first->stride(axis) != second->stride(axis)) return false;
    return true;
}

const DenseTensor& Tensor::dense() const {
    const auto* value = dynamic_cast<const DenseTensor*>(impl_.get());
    if (!value) throw std::invalid_argument("operation requires a dense tensor");
    return *value;
}

DenseTensor& Tensor::dense() {
    auto* value = dynamic_cast<DenseTensor*>(impl_.get());
    if (!value) throw std::invalid_argument("operation requires a dense tensor");
    return *value;
}

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
