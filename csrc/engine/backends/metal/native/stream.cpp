#include "stream.h"

#include <utility>

namespace tyloi::metal {

void bind(const mlx::core::Stream& runtime_stream) {
    mlx::core::set_default_device(mlx::core::Device::gpu);
    mlx::core::set_default_stream(runtime_stream);
}

void eval(mlx::core::array& value) {
    value.eval();
}

void eval(std::vector<mlx::core::array> values) {
    mlx::core::eval(std::move(values));
}

void submit(std::vector<mlx::core::array> values) {
    mlx::core::async_eval(std::move(values));
}

void wait(const mlx::core::Stream& runtime_stream) {
    mlx::core::synchronize(runtime_stream);
}

void drain(const mlx::core::Stream& runtime_stream) {
    wait(runtime_stream);
    mlx::core::synchronize();
}

} // namespace tyloi::metal
