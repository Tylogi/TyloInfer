#pragma once

#include <mlx/array.h>
#include <mlx/backend/metal/device.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace tyloi::metal {

class ResidentAllocations {
    struct Capture {
        std::mutex mutex;
        std::vector<std::weak_ptr<mlx::core::array::Data>> allocations;
        std::size_t pending_bytes = 0;
    };
    struct Allocation {
        std::weak_ptr<mlx::core::array::Data> owner;
        NS::SharedPtr<MTL::Buffer> buffer;
        std::size_t set = 0;
        std::size_t bytes = 0;
    };

public:
    static void begin_load();

    static void track(const mlx::core::array& value);

    static std::unique_ptr<ResidentAllocations> finish_load(std::size_t limit);

    static std::unique_ptr<ResidentAllocations> retain_arrays(
        const std::vector<mlx::core::array>& values, std::size_t limit, bool indirect = false);

    void update_arrays(const std::vector<mlx::core::array>& values,std::size_t limit);

    void use_command_buffer(MTL::CommandBuffer* buffer);
    ~ResidentAllocations();

    ResidentAllocations(const ResidentAllocations&) = delete;
    ResidentAllocations& operator=(const ResidentAllocations&) = delete;

    std::size_t bytes() const noexcept;

private:
    static void activate();

    explicit ResidentAllocations(
        const std::vector<std::shared_ptr<mlx::core::array::Data>>& live,
        std::size_t limit, bool indirect = false);

    static bool supported();

    static std::shared_ptr<Capture>& capture();

    std::vector<NS::SharedPtr<MTL::ResidencySet>> sets_;
    std::vector<std::shared_ptr<mlx::core::array::Data>> indirect_owners_;
    std::vector<Allocation> allocations_;
    bool indirect_ = false;
    std::thread heartbeat_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopped_ = false;
    std::atomic<std::size_t> bytes_{0};
};

}
