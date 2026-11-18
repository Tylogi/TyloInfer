#include "cli.h"
#include "run.h"
#include "engine/config.h"
#include "engine/engine.h"
#include "engine/generation/mtp/limits.h"
#include "engine/load.h"
#include "engine/models/source/model.h"
#include "engine/models/source/model_config.h"
#include "transport/include/transport.h"
#include "nlohmann/json.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tyloi::commands {
namespace {

void print_help() {
    std::cout
        << "TyloInfer runtime\n\n"
        << "Usage:\n"
        << "  tyloi-engine --model MODEL --transport stdio|http [OPTIONS]\n"
        << '\n'
        << "Runtime options:\n"
        << "  --model PATH                    model or split-model shard\n"
        << "  --inspect-model-json            inspect metadata without loading GPU weights\n"
        << "  --config PATH                   external model config\n"
        << "  --transport TYPE                stdio or http\n"
        << "  --host HOST                     HTTP bind host (default 127.0.0.1)\n"
        << "  --port N                        HTTP port (default 8080)\n"
        << "  --model-name NAME               API model name\n"
        << "  --api-key KEY                   API key\n"
        << "  --sampling-profile PATH         sampling profile override\n"
        << "  --continuous-batching N         maximum concurrent sequences\n"
        << "  --prefill-chunk-size auto|N     compute chunk limit (default auto, max 8192)\n"
        << "  --tokenizer PATH                external tokenizer GGUF\n"
        << "  --ctx-size N                    context size; 0 selects a default\n"
        << "  --moe-gpu-cache-gb N            bounded MoE GPU cache\n"
#if TYLOINFER_CLI_CUDA
        << "  -t, --threads N                 positive CPU thread count\n"
        << "  -ngl, --n-gpu-layers N          non-negative GPU layer count\n"
        << "  --cpu-offload-layers RANGES     explicit CPU-offloaded layers\n"
        << "  --moe-cache-profile PATH        MoE cache profile\n"
        << "  --dflash2-draft PATH             CUDA DFlash2 draft checkpoint for Qwen3.5 or Qwen4\n"
        << "  --tensor-parallel DEVICES       device count or comma-separated IDs\n"
        << "  --tensor-split WEIGHTS          tensor-parallel weights\n"
        << "  --expert-parallel DEVICES       expert-parallel devices\n"
        << "  --expert-split WEIGHTS          expert-parallel weights\n"
        << "  --layer-parallel DEVICES        layer-placement devices\n"
        << "  --layer-split WEIGHTS           layer-placement weights\n"
#endif
        << "  -h, --help                      show this help\n";
}

struct RuntimeOptions : tyloi::engine::EngineOptions {
    std::string transport_host = "127.0.0.1";
    std::string runtime_model_name = "mfq-model", transport_api_key;
    std::string runtime_sampling_profile;
    int transport_port = 8080;
    bool transport_mode = false;
    bool stdio_mode = false;
    bool inspect_model_json = false;
};

RuntimeOptions parse_runtime(ArgCursor& args) {
    RuntimeOptions result;
#if TYLOINFER_CLI_CUDA
    result.device = tyloi::Device(tyloi::DeviceType::cuda);
#elif TYLOINFER_CLI_METAL
    result.device = tyloi::Device(tyloi::DeviceType::metal);
#else
#error "tyloi-engine requires a selected runtime backend"
#endif
    bool transport_option = false;
    while (!args.empty()) {
        const std::string_view option = args.next();
        if (option == "--help" || option == "-h") {
            print_help();
            throw HelpRequested{};
        }
        if (parse_load_option(option, args, result)) continue;
        if (option == "--inspect-model-json") {
            result.inspect_model_json = true;
        }
        else if (option == "--transport") {
            if (result.transport_mode) {
                usage_error("runtime transport was specified more than once");
            }
            const std::string transport = args.value(option);
            if (transport != "stdio" && transport != "http") {
                usage_error("--transport must be stdio or http");
            }
            result.transport_mode = true;
            result.stdio_mode = transport == "stdio";
        }
        else if (option == "--host") {
            result.transport_host = args.value(option);
            transport_option = true;
        }
        else if (option == "--port") {
            result.transport_port = integer<int>(args.value(option), option);
            if (result.transport_port < 1 || result.transport_port > 65535) {
                usage_error("--port must be in [1, 65535]");
            }
            transport_option = true;
        }
        else if (option == "--continuous-batching") {
            result.continuous_batching = integer<int>(args.value(option), option);
            if (result.continuous_batching < 0) {
                usage_error("--continuous-batching must be non-negative");
            }
            transport_option = true;
        }
        else if (option == "--prefill-chunk-size") {
            const auto value = args.value(option);
            result.prefill_chunk_size = value == "auto" ? 0 : integer<int64_t>(value, option);
            if (value != "auto" && (result.prefill_chunk_size <= 0 || result.prefill_chunk_size > 8192)) {
                usage_error("--prefill-chunk-size must be auto or an integer in [1, 8192]");
            }
            transport_option = true;
        }
        else if (option == "--model-name") {
            result.runtime_model_name = args.value(option);
            transport_option = true;
        }
        else if (option == "--api-key") {
            result.transport_api_key = args.value(option);
            transport_option = true;
        }
        else if (option == "--sampling-profile") {
            result.runtime_sampling_profile = args.value(option);
            transport_option = true;
        }
        else usage_error("unknown option: " + std::string(option));
    }

    if (result.model_path.empty()) usage_error("--model is required");
    const int modes = static_cast<int>(result.transport_mode) +
        static_cast<int>(result.inspect_model_json);
    if (modes != 1) {
        usage_error("select exactly one execution mode: --inspect-model-json or --transport");
    }
    if (transport_option && !result.transport_mode) {
        usage_error("transport options require --transport");
    }
    if (result.transport_mode && !result.config_path.empty()) {
        usage_error("--config cannot be used with --transport");
    }
    return result;
}

std::string read_runtime_asset_text(
        const tyloi::ModelSource& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    if (bytes.empty()) return {};
    return std::string(
        reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

int run_transport_runtime(RuntimeOptions& options) {
    auto source = tyloi::open_model_source(options.model_path);
    if (!options.config_path.empty()) {
        throw std::runtime_error(
            "model runtime does not accept an external model config");
    }
    if (!source->has_asset(tyloi::kModelConfigAsset) ||
            (!source->has_asset(tyloi::kTokenizerGgufAsset) &&
             options.tokenizer_model.empty())) {
        throw std::runtime_error(
            "model runtime requires model config and tokenizer GGUF");
    }

    auto engine = tyloi::engine::load(options);
    const auto info = engine->info();
    if (options.transport_api_key.empty()) {
        const char* env_key = tyloi::engine::environment_value("TYLOINFER_API_KEY");
        if (env_key != nullptr) options.transport_api_key = env_key;
    }

    MfqHttpRuntimeTransportConfig transport_config;
    transport_config.host = options.transport_host;
    transport_config.port = options.transport_port;
    transport_config.model_name = options.runtime_model_name;
    transport_config.model_type = info.model_type;
    transport_config.api_key = options.transport_api_key;
    transport_config.max_context = info.max_context;
    transport_config.vocab_size = info.vocab_size;
    transport_config.model_capabilities = info.capabilities;
    const auto& runtime_assets = *source;
    const auto embedded_profile = runtime_assets.metadata().find(
        "runtime.sampling.v1");
    transport_config.runtime_profile = resolve_mfq_runtime_profile(
        options.model_path,
        info.capabilities.family,
        transport_config.model_type,
        transport_config.model_name,
        embedded_profile == runtime_assets.metadata().end()
            ? std::string()
            : embedded_profile->second,
        read_runtime_asset_text(runtime_assets, tyloi::kModelConfigAsset),
        options.runtime_sampling_profile);
    const auto* draft = tyloi::engine::environment_value("TYLOINFER_DFLASH2_DRAFT");
    if (options.device.is_cuda() && (!options.dflash2_draft_path.empty() || (draft && *draft)))
        transport_config.runtime_profile.chat.mtp_max_draft_tokens =
            tyloi::engine::mtp::kMaximumBlockwiseDraftDepth;

    auto transport = options.stdio_mode
        ? make_mfq_stdio_transport(transport_config)
        : make_mfq_http_transport(transport_config);
    return transport->run(*engine);
}

int execute_runtime(RuntimeOptions options) {
    return with_command_errors([&]() -> int {
        if (options.inspect_model_json) {
            const auto source = tyloi::open_model_source(options.model_path);
            nlohmann::json result{{"architecture", source->architecture()},
                {"config", nlohmann::json::parse(tyloi::models::config_json(*source, options.config_path))},
                {"paths", nlohmann::json::array()}, {"tensors", nlohmann::json::object()}};
            for (const auto& path : source->source_paths()) {
                const auto value = path.generic_u8string();
                result["paths"].push_back(std::string(value.begin(), value.end()));
            }
            for (const auto& tensor : source->tensors()) {
                if (tensor.name.starts_with("__mfq_asset__/")) continue;
                auto& record = result["tensors"][tensor.name];
                record = {{"dtype", tensor.dtype}, {"nbytes", tensor.nbytes}};
                if (tensor.shape) record["shape"] = *tensor.shape;
            }
            std::cout << result.dump() << '\n';
            return 0;
        }
        if (options.stdio_mode) prepare_mfq_stdio_transport();
        return run_transport_runtime(options);
    });
}

} // namespace

int run_runtime(int argc, char** argv) {
    try {
        ArgCursor args(argc, argv);
        auto options = parse_runtime(args);
        return execute_runtime(std::move(options));
    } catch (const HelpRequested&) {
        return 0;
    } catch (const UsageError& error) {
        return print_usage_error(error);
    }
}

} // namespace tyloi::commands
