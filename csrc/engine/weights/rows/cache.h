#pragma once

#include "engine/memory/host.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tyloi::weights {

class RowCache final : public HostCache {
public:
    ~RowCache();
    std::size_t copy(std::uint64_t key, std::byte* out, std::size_t capacity);
    void insert(std::uint64_t key, const std::byte* source, std::size_t bytes);

private:
    RowCache();
    std::size_t reclaim(std::size_t bytes) override;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend std::shared_ptr<RowCache> make_row_cache();
};

std::shared_ptr<RowCache> make_row_cache();

} // namespace tyloi::weights
