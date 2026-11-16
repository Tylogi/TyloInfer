#include "server.h"
#include "instance.h"
#include "engine/engine.h"
#include "engine/models/load.h"
#include "engine/models/source/model.h"

#include "engine/load.h"

#include <algorithm>
#include <charconv>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::atomic<bool> stop_requested{false};
static_assert(std::atomic<bool>::is_always_lock_free);

void stop(int) { stop_requested.store(true, std::memory_order_relaxed); }

tyloi::server::EngineFactory native_engine_factory() {
    tyloi::server::EngineFactory factory;
#if defined(TYLOINFER_SERVER_CUDA) || defined(TYLOINFER_SERVER_METAL)
#ifdef TYLOINFER_SERVER_CUDA
    factory.backend = "cuda";
#else
    factory.backend = "metal";
#endif
    factory.unavailable_reason = [](const tyloi::ModelSource& source) -> std::string {
#ifdef TYLOINFER_SERVER_CUDA
        return tyloi::models::unavailable_reason(
            source, tyloi::Device(tyloi::DeviceType::cuda, 0));
#else
        return tyloi::models::unavailable_reason(
            source, tyloi::Device(tyloi::DeviceType::metal, 0));
#endif
    };
    factory.load = [](const std::filesystem::path& path, const tyloi::server::LoadOptions& request) {
        tyloi::engine::EngineOptions options;
#ifdef TYLOINFER_SERVER_CUDA
        options.device = tyloi::Device(tyloi::DeviceType::cuda, 0);
#else
        options.device = tyloi::Device(tyloi::DeviceType::metal, 0);
#endif
        options.model_path = tyloi::server::path_utf8(path);
        options.context_size = request.context_size;
        return tyloi::engine::load(std::move(options));
    };
#endif
    return factory;
}

int main_impl(const std::vector<std::string>& arguments) {
    try {
        tyloi::server::Config config;
        for (std::size_t i = 1; i < arguments.size(); ++i) {
            const auto& option = arguments[i];
            if (option == "--help" || option == "-h") {
                std::cout << "TyloInfer native server " << TYLOINFER_SERVER_VERSION << "\n"
                    "Usage: tyloinfer-server [--host IP] [--port PORT]\n"
                    "                  [--data-dir PATH] [--db PATH] [--web-root PATH]\n"
                    "                  [--api-key-env NAME] [--model-dir PATH ...]\n"
                    "                  [--exit-on-stdin-close] [--studio-instance-id UUID]\n"
                    "Default port: 8090; port 0 selects an available port.\n"
                    "Non-loopback listeners require enabled authentication.\n"
                    "Root API key environment variable: TYLOINFER_SERVER_API_KEY.\n";
                return 0;
            }
            if (option == "--version") {
                std::cout << TYLOINFER_SERVER_VERSION << '\n';
                return 0;
            }
            if (option == "--exit-on-stdin-close") {
                config.exit_on_stdin_close = true;
                continue;
            }
            if (option != "--host" && option != "--port" && option != "--data-dir" &&
                option != "--web-root" && option != "--studio-instance-id" &&
                option != "--api-key-env" && option != "--db" && option != "--model-dir")
                throw std::invalid_argument("unknown option: " + option);
            if (++i == arguments.size() || arguments[i].empty())
                throw std::invalid_argument("missing value for " + option);
            const auto& value = arguments[i];
            if (option == "--host") config.host = value;
            else if (option == "--data-dir") config.data_dir = std::filesystem::u8path(value);
            else if (option == "--web-root") config.web_root = std::filesystem::u8path(value);
            else if (option == "--db") config.database = std::filesystem::u8path(value);
            else if (option == "--model-dir") config.model_dirs.push_back(std::filesystem::u8path(value));
            else if (option == "--api-key-env") config.api_key_env = value;
            else if (option == "--studio-instance-id") config.instance_id = value;
            else {
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), config.port);
                if (!std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }) ||
                    parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
                    config.port > 65535)
                    throw std::invalid_argument("--port must be an integer in [0, 65535]");
            }
        }
        if (!config.instance_id.empty() && !config.exit_on_stdin_close)
            throw std::invalid_argument("--studio-instance-id requires --exit-on-stdin-close");
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        return tyloi::server::run(config, stop_requested, native_engine_factory());
    } catch (const std::exception& error) {
        std::cerr << "TyloInfer server not started: " << error.what() << '\n';
        return 1;
    }
}

} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> arguments;
    for (int i = 0; i < argc; ++i) arguments.push_back(tyloi::server::path_utf8(std::filesystem::path(argv[i])));
    return main_impl(arguments);
}
#else
int main(int argc, char** argv) { return main_impl({argv, argv + argc}); }
#endif
