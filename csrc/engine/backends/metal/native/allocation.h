#pragma once

#include <mlx/array.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace tyloi::metal {

class AllocationScope {
public:
    struct Log {
        std::mutex mutex;
        std::vector<std::weak_ptr<mlx::core::array::Data>> allocations;
        std::size_t pending_bytes = 0;
        bool recording = true;
        void clear();
    };

    explicit AllocationScope(Log* log) noexcept;
    ~AllocationScope();
    AllocationScope(const AllocationScope&) = delete;
    AllocationScope& operator=(const AllocationScope&) = delete;

    static Log* log() noexcept;
    static void record(const mlx::core::array& value);

private:
    static Log*& current() noexcept;
    Log* previous_;
};

} // namespace tyloi::metal
