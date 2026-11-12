#pragma once

#include "engine/tensor.h"

#include <mlx/mlx.h>
#include <mutex>
#include <optional>
#include <vector>

namespace tyloi::metal {

class Stream final {
public:
    explicit Stream(mlx::core::Stream stream);
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    Stream(Stream&&) = delete;
    Stream& operator=(Stream&&) = delete;

    const mlx::core::Stream& get() const noexcept { return stream_; }

private:
    struct Submission {
        std::vector<mlx::core::array> outputs;
        std::vector<std::shared_ptr<TensorStorage>> backings;
        std::vector<mlx::core::Stream> streams;
    };
    void record(const std::vector<mlx::core::array>& values);
    void retire_ready();

    mlx::core::Stream stream_;
    std::mutex mutex_;
    std::vector<Submission> pending_;

    friend void eval(mlx::core::array&);
    friend void eval(std::vector<mlx::core::array>);
    friend void submit(std::vector<mlx::core::array>);
    friend void wait(Stream&);
    friend void drain(Stream&);
};

class StreamGuard final {
public:
    explicit StreamGuard(Stream& stream);
    ~StreamGuard();
    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
    StreamGuard(StreamGuard&&) = delete;
    StreamGuard& operator=(StreamGuard&&) = delete;

private:
    Stream* previous_;
    std::optional<mlx::core::Stream> previous_default_;
};

void eval(mlx::core::array& value);
void eval(std::vector<mlx::core::array> values);
void submit(std::vector<mlx::core::array> values);
void wait(Stream& stream);
void drain(Stream& stream);

} // namespace tyloi::metal
