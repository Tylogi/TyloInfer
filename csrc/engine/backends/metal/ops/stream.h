#pragma once

#include <mlx/mlx.h>
#include <vector>

namespace tyloi::metal::ops {

void eval(mlx::core::array& value);
void eval(std::vector<mlx::core::array> values);
// Submission does not wait; request owners retire at their existing completion boundary.
void submit(std::vector<mlx::core::array> values);
// MTP and sampling can use both the shared runtime stream and the calling thread's default.
// Drain both before releasing arrays or reclaiming allocator pages.
void drain(const mlx::core::Stream& runtime_stream);

} // namespace tyloi::metal::ops
