#pragma once

#include "engine/backends/metal/native/pinned.h"
#include "engine/backends/metal/native/resident_allocations.h"
#include <mlx/array.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace tyloi::metal {

std::size_t pool_page();
struct TqRange { std::size_t offset = 0, bytes = 0; };
struct TqPoolLifetime {
    std::vector<mlx::core::array> chunks;
    mlx::core::array handle = mlx::core::array(0, mlx::core::uint64);
    std::shared_ptr<ResidentAllocations> residency;
};

struct TqPool {
    using Sizes = std::multimap<std::size_t, std::size_t>;
    struct FreeRange { std::size_t bytes; Sizes::iterator size_entry; };
    mlx::core::array words = mlx::core::array(0, mlx::core::uint32);
    std::byte* data = nullptr;
    std::size_t capacity = 0;
    std::shared_ptr<mfq::metal::detail::MlxPinnedPoolStorage> storage;
    std::size_t chunk_bytes = pool_page();
    std::vector<mlx::core::array> chunks;
    std::shared_ptr<TqPoolLifetime> lifetime;
    std::map<std::size_t, FreeRange> free;
    Sizes sizes;

    void insert_free(std::size_t offset, std::size_t bytes);
    std::uint64_t address(std::size_t offset) const;
    void erase_free(std::map<std::size_t, FreeRange>::iterator range);
    void reset_free(std::size_t offset);
    void allocate_storage(std::size_t bytes);
    void bind_storage(std::size_t bytes);
    // Call after resize commits, so failed rebinding can restore the old pages.
    void discard_tail(std::size_t previous_capacity);
    bool fits(std::size_t bytes, std::size_t limit = SIZE_MAX) const;
    TqRange take(std::size_t bytes, std::size_t limit = SIZE_MAX);
    void release(TqRange range);
    std::optional<TqRange> relocate(TqRange range, std::size_t limit);
};

} // namespace tyloi::metal
