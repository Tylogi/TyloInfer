#include "execution.h"

#include <utility>

namespace tyloi::metal::ops {

void eval(mlx::core::array& value) {
    value.eval();
}

void eval(std::vector<mlx::core::array> values) {
    mlx::core::eval(std::move(values));
}

void submit(std::vector<mlx::core::array> values) {
    mlx::core::async_eval(std::move(values));
}

void drain(const mlx::core::Stream& runtime_stream) {
    mlx::core::synchronize(runtime_stream);
    mlx::core::synchronize();
}

} // namespace tyloi::metal::ops
