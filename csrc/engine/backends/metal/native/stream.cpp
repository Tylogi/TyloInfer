#include "stream.h"

#include "tensor/tensor.h"

#include <mlx/primitives.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace tyloi::metal {
namespace {
thread_local Stream* active_stream = nullptr;
}

Stream::Stream(mlx::core::Stream stream) : stream_(stream) {
    if (stream.device != mlx::core::Device(mlx::core::Device::gpu, 0))
        throw std::invalid_argument("Metal stream requires GPU device 0");
}

Stream::~Stream() {
    try {
        drain(*this);
    } catch (...) {
        // Releasing owners without proving completion could free live device memory.
        std::terminate();
    }
}

StreamGuard::StreamGuard(Stream& stream)
    : previous_(active_stream),
      previous_default_(mlx::core::peek_default_stream(stream.get().device)) {
    mlx::core::set_default_stream(stream.get());
    active_stream = &stream;
}

StreamGuard::~StreamGuard() {
    active_stream = previous_;
    if (previous_default_) mlx::core::set_default_stream(*previous_default_);
}

void Stream::record(const std::vector<mlx::core::array>& values) {
    Submission submission;
    submission.outputs = values;
    std::vector<mlx::core::array> worklist = values;
    std::unordered_set<std::uintptr_t> visited;
    std::set<mlx::core::Stream> streams;
    for (const auto& value : values) {
        auto backings = metal::backings(value);
        submission.backings.insert(submission.backings.end(), backings.begin(), backings.end());
    }
    while (!worklist.empty()) {
        auto node = std::move(worklist.back());
        worklist.pop_back();
        if (!visited.insert(node.id()).second) continue;
        if (node.has_primitive()) streams.insert(node.primitive().stream());
        const auto& inputs = node.inputs();
        worklist.insert(worklist.end(), inputs.begin(), inputs.end());
    }
    submission.streams.assign(streams.begin(), streams.end());
    pending_.push_back(std::move(submission));
}

void Stream::retire_ready() {
    std::erase_if(pending_, [](const Submission& submission) {
        return std::all_of(
            submission.outputs.begin(), submission.outputs.end(),
            [](const auto& node) { return node.is_available(); });
    });
}

void eval(mlx::core::array& value) {
    if (!active_stream) {
        value.eval();
        return;
    }
    std::lock_guard lock(active_stream->mutex_);
    active_stream->retire_ready();
    active_stream->record({value});
    value.eval();
    active_stream->retire_ready();
}

void eval(std::vector<mlx::core::array> values) {
    if (!active_stream) {
        mlx::core::eval(std::move(values));
        return;
    }
    std::lock_guard lock(active_stream->mutex_);
    active_stream->retire_ready();
    active_stream->record(values);
    mlx::core::eval(std::move(values));
    active_stream->retire_ready();
}

void submit(std::vector<mlx::core::array> values) {
    if (!active_stream) throw std::logic_error("Metal submission requires an active stream");
    std::lock_guard lock(active_stream->mutex_);
    active_stream->retire_ready();
    active_stream->record(values);
    mlx::core::async_eval(std::move(values));
}

void wait(Stream& stream) {
    std::lock_guard lock(stream.mutex_);
    mlx::core::synchronize(stream.get());
    stream.retire_ready();
}

void drain(Stream& stream) {
    std::lock_guard lock(stream.mutex_);
    std::set<mlx::core::Stream> streams;
    for (const auto& submission : stream.pending_)
        streams.insert(submission.streams.begin(), submission.streams.end());
    streams.erase(stream.get());
    if (auto current = mlx::core::peek_default_stream(mlx::core::default_device()))
        streams.erase(*current);
    mlx::core::synchronize(stream.get());
    mlx::core::synchronize();
    for (const auto& recorded : streams) mlx::core::synchronize(recorded);
    stream.pending_.clear();
}

} // namespace tyloi::metal
