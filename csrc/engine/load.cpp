#include "engine/load.h"
#include "engine/engine.h"
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>
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
    auto reload_options = options;
    std::unique_ptr<Engine> result;
#if TYLOINFER_HAS_CUDA
    if (options.device.is_cuda()) result = cuda::load(std::move(options));
#endif
#if TYLOINFER_HAS_METAL
    if (options.device.is_metal()) result = metal::load(std::move(options));
#endif
    if (!result) throw std::invalid_argument("requested execution device is unavailable in this build");
    result->start_reload(std::move(reload_options));
    return result;
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

void Engine::start_reload(EngineOptions options) {
    reload_worker_ = std::jthread([this, options = std::move(options)] {
        while (auto job = next_reload()) {
            try {
                {
                    auto reload_options = options;
                    reload_options.context_size = job->context;
                    engine::reload(*this, std::move(reload_options));
                }
                job->reply.set_value();
            } catch (...) {
                job->reply.set_exception(std::current_exception());
            }
        }
    });
    try { enable_reload(); }
    catch (...) {
        const auto failure = std::current_exception();
        try { shutdown(); }
        catch (const std::exception& error) {
            std::cerr << "runtime_unload_cleanup error=" << error.what() << '\n';
        }
        catch (...) { std::cerr << "runtime_unload_cleanup error=unknown\n"; }
        std::rethrow_exception(failure);
    }
}
}
