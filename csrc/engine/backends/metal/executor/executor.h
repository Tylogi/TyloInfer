#pragma once

#include "engine/executor.h"

namespace tyloi::metal {

class Stream;

class Executor final : public tyloi::Executor {
public:
    explicit Executor(std::shared_ptr<Stream> stream);
    ~Executor() override = default;
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    std::unique_ptr<tyloi::Executor::Scope> enter() override;
    std::unique_ptr<tyloi::Executor::Timer> timer() override;
    std::unique_ptr<tyloi::Executor::Profile> profile(const char* name) override;
    std::unique_ptr<tyloi::Executor::Graph> graph() override;
    void submit(std::span<const Tensor> outputs) override;
    std::vector<tyloi::Executor::Memory> memory(std::span<const Device> devices) const override;
    void reclaim(Device device) override;
    void set_memory_limit(Device device, std::size_t bytes) override;
    void drain() override;
    void clear() override;

    Stream& stream() const noexcept { return *stream_; }
    static Executor& current();

private:
    void require_current() const;
    std::shared_ptr<Stream> stream_;
};

} // namespace tyloi::metal
