#pragma once

#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/tensor/tensor.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

namespace tyloi::cuda {

bool can_prefill_f16_residual(std::int64_t tokens, std::int64_t rows,
    std::int64_t columns, Device device, const DenseTensor& residual);
void prefill_f16_residual_tile(const DenseTensor& input, const DenseTensor& weights,
    const DenseTensor& residual, const DenseTensor& output, std::int64_t first,
    std::int64_t count, void* blas, DenseTensor& scratch);

// Select a row tile once during warmup. Capture reuses that geometry, while
// warmup admission checks the actual fragmented workspace holes.
inline std::int64_t prefill_weight_rows(std::int64_t rows, std::uint64_t bytes_per_row,
        std::int64_t alignment, int device, bool double_buffer = false) {
    const auto context = default_context(device);
    using Key = std::tuple<int, std::int64_t, std::uint64_t, std::int64_t, bool>;
    static thread_local std::map<Key, std::int64_t> prepared;
    const Key key{device, rows, bytes_per_row, alignment, double_buffer};
    const auto stream = current_stream(device).stream();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        if (const auto found = prepared.find(key); found != prepared.end()) return found->second;
        return std::min(rows, alignment);
    }
    const auto physical = context->available_memory();
    const auto reusable = context->reusable_workspace_bytes();
    const auto add = [](std::size_t left, std::size_t right) {
        return left > std::numeric_limits<std::size_t>::max() - right
            ? std::numeric_limits<std::size_t>::max() : left + right;
    };
    const auto ordinary = add(physical, reusable);
    const auto budget = add(ordinary, context->workspace_available());
    auto count = std::min<std::uint64_t>(rows, (budget / 2) / bytes_per_row);
    if (count < std::uint64_t(rows) && double_buffer) count /= 2;

    const auto fits = [&](std::int64_t candidate) {
        if (candidate <= 0 || std::uint64_t(candidate) >
                std::numeric_limits<std::size_t>::max() / bytes_per_row) return false;
        const auto bytes = std::size_t(candidate) * std::size_t(bytes_per_row);
        const std::vector<std::size_t> allocations = double_buffer
            ? std::vector<std::size_t>{bytes, bytes} : std::vector<std::size_t>{bytes};
        return context->workspace_allocations_fit(allocations, ordinary, stream);
    };
    std::uint64_t selected_count = count >= std::uint64_t(rows) ? count :
        count >= std::uint64_t(alignment)
            ? count / std::uint64_t(alignment) * std::uint64_t(alignment)
            : std::uint64_t(std::max<std::int64_t>(1, std::int64_t(count)));
    if (!fits(std::int64_t(selected_count))) {
        if (selected_count < std::uint64_t(alignment)) selected_count = 1;
        else {
            while (selected_count > std::uint64_t(alignment) &&
                   !fits(std::int64_t(selected_count)))
                selected_count -= std::uint64_t(alignment);
            if (!fits(std::int64_t(selected_count))) selected_count = 1;
        }
    }
    const auto selected = count >= std::uint64_t(rows) && selected_count == count ? rows :
        selected_count >= std::uint64_t(alignment)
            ? std::int64_t(selected_count) : std::max<std::int64_t>(1, std::int64_t(selected_count));
    prepared[key] = selected;
    return selected;
}

} // namespace tyloi::cuda
