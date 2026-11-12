#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <sys/mman.h>

namespace tyloi::metal {

// MLX stores every individual shape axis as int32 even though array::size()
// and the underlying allocation are size_t. Packed runtime streams are flat
// byte arenas, so their logical rank is irrelevant to every consumer. Split a
// large arena over two exact factors instead of rejecting it merely because a
// single axis would exceed INT32_MAX. No padding is permitted: descriptor
// offsets and slot-count validation both use the exact flat element count.
struct PackedLayout {
    int rows = 1;
    int columns = 1;

    constexpr bool is_matrix() const noexcept {
        return rows != 1;
    }
};

PackedLayout packed_layout(std::size_t elements);

// MLX's process-wide limit is a guideline, not a reservation of physical RAM.
struct Memory {
    std::size_t active, cache, wired, limit, available, total;
};

Memory memory();
std::size_t memory_limit();
void set_memory_limit(std::size_t bytes);
bool residency_available();
std::size_t active_memory();
std::size_t cache_memory();
std::size_t peak_memory();
std::size_t wired_memory();
void set_cache_limit(std::size_t limit);
void set_wired_limit(std::size_t limit);
void clear_cache();

// Large model-load scratch buffers must bypass Darwin malloc's large-object
// depot: freed multi-GiB vector capacities can otherwise remain resident and
// count against the model footprint. Anonymous mappings are returned to the
// kernel immediately when a staging vector dies.
template <typename T>
class StagingAllocator {
public:
    using value_type = T;
    using is_always_equal = std::true_type;

    StagingAllocator() noexcept = default;

    template <typename U>
    StagingAllocator(const StagingAllocator<U>&) noexcept {}

    T* allocate(std::size_t count) {
        if (count == 0) {
            return nullptr;
        }
        if (count > max_size()) {
            throw std::bad_array_new_length();
        }
        const auto bytes = count * sizeof(T);
        if (bytes < mapping_threshold) {
            if constexpr (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
                return static_cast<T*>(::operator new(bytes, std::align_val_t{alignof(T)}));
            } else {
                return static_cast<T*>(::operator new(bytes));
            }
        }
        void* mapping = ::mmap(
            nullptr,
            bytes,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0);
        if (mapping == MAP_FAILED) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(mapping);
    }

    void deallocate(T* pointer, std::size_t count) noexcept {
        if (pointer == nullptr || count == 0) {
            return;
        }
        const auto bytes = count * sizeof(T);
        if (bytes < mapping_threshold) {
            if constexpr (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
                ::operator delete(pointer, std::align_val_t{alignof(T)});
            } else {
                ::operator delete(pointer);
            }
        } else {
            ::munmap(pointer, bytes);
        }
    }

    constexpr std::size_t max_size() const noexcept {
        return std::numeric_limits<std::size_t>::max() / sizeof(T);
    }

private:
    static constexpr std::size_t mapping_threshold = 64 * 1024;
};

template <typename T, typename U>
bool operator==(
    const StagingAllocator<T>&,
    const StagingAllocator<U>&) noexcept {
    return true;
}

template <typename T>
using StagingVector = std::vector<T, StagingAllocator<T>>;

}  // namespace tyloi::metal
