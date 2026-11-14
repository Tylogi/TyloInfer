#include "engine/load.h"
#include "engine/engine.h"
#include <stdexcept>
#include <utility>

#if TYLOINFER_HAS_CUDA
#include "engine/backends/cuda/load.h"
#endif
#if TYLOINFER_HAS_METAL
namespace tyloi::metal {
std::unique_ptr<engine::Engine> load(engine::EngineOptions);
void reload(engine::Engine&, engine::EngineOptions);
}
#endif

namespace tyloi::engine {
std::unique_ptr<Engine> load(EngineOptions options) {
#if TYLOINFER_HAS_CUDA
    if (options.device.is_cuda()) return cuda::load(std::move(options));
#endif
#if TYLOINFER_HAS_METAL
    if (options.device.is_metal()) return metal::load(std::move(options));
#endif
    throw std::invalid_argument("requested execution device is unavailable in this build");
}
void reload(Engine& engine, EngineOptions options) {
#if TYLOINFER_HAS_CUDA
    if (options.device.is_cuda()) return cuda::reload(engine, std::move(options));
#endif
#if TYLOINFER_HAS_METAL
    if (options.device.is_metal()) return metal::reload(engine, std::move(options));
#endif
    throw std::invalid_argument("requested execution device is unavailable in this build");
}
}
