#include "config.h"

#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/ops/attention/kv/quantization.h"
#include "engine/load.h"
#include "engine/models/load.h"
#include "engine/models/metadata.h"
#include "engine/weights/cache/cache.h"
#include "engine/weights/moe/profile.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

void validate_parallel_config(
        const ParallelConfig& config,
        const char* devices_option,
        const char* split_option,
        int available_devices,
        bool allow_duplicate_devices) {
    if (config.devices.empty()) {
        if (!config.split.empty()) {
            throw std::runtime_error(
                std::string(split_option) + " requires " + devices_option);
        }
        return;
    }
    if (config.devices.size() < 2) {
        throw std::runtime_error(
            std::string(devices_option) +
            " requires at least two devices");
    }

    std::unordered_set<int> unique_devices;
    for (const int device : config.devices) {
        if (device < 0 || device >= available_devices) {
            throw std::runtime_error(
                std::string(devices_option) +
                " CUDA device is unavailable: " +
                std::to_string(device));
        }
        const bool inserted = unique_devices.insert(device).second;
        if (!allow_duplicate_devices && !inserted) {
            throw std::runtime_error(
                std::string(devices_option) +
                " CUDA devices must be unique");
        }
    }

    if (!config.split.empty()) {
        if (config.split.size() != config.devices.size()) {
            throw std::runtime_error(
                std::string(split_option) + " count must match " +
                devices_option + " devices");
        }
        for (const double weight : config.split) {
            if (!std::isfinite(weight) || weight <= 0.0) {
                throw std::runtime_error(
                    std::string(split_option) +
                    " values must be finite and positive");
            }
        }
    }
}

void print_parallel_config(
        const char* label,
        const ParallelConfig& config) {
    if (!config.enabled()) return;
    std::cerr << label << " devices=";
    for (std::size_t index = 0; index < config.devices.size(); ++index) {
        if (index) std::cerr << ',';
        std::cerr << config.devices[index];
    }
    if (!config.split.empty()) {
        std::cerr << " split=";
        for (std::size_t index = 0; index < config.split.size(); ++index) {
            if (index) std::cerr << ',';
            std::cerr << config.split[index];
        }
    }
    std::cerr << '\n';
}

} // namespace

namespace tyloi::cuda {

void configure_load(CudaLoadConfig& config, const tyloi::engine::LoadOptions& options) {
    config.tensor_parallel = options.tensor_parallel;
    config.expert_parallel = options.expert_parallel;
    config.layer_placement = options.layer_placement;
    config.dsv4_cpu_offload_layers = options.cpu_offload_layers;
    config.dsv4_cpu_offload_host_bytes = 0;
    config.n_gpu_layers = options.n_gpu_layers_set ? options.n_gpu_layers : -1;
    config.dense_cpu_layer_count = 0;
    config.drop_file_cache = !options.cpu_offload_layers.empty();
    config.weights.min_slots = 8;
    config.cache_bytes = 0;
    config.cache_profile.clear();
    config.yarn_context_extension = options.yarn_context_extension;
    if (options.cpu_threads_set && options.cpu_threads <= 0)
        throw std::runtime_error("--threads must be positive");
    if (options.device.index < 0) throw std::invalid_argument("CUDA device index must be non-negative");
    int available = 0;
    TYLOINFER_CUDA_CHECK(cudaGetDeviceCount(&available));
    if (options.device.index >= available)
        throw std::invalid_argument("CUDA primary device is unavailable");
    if (!config.dsv4_cpu_offload_layers.empty()) {
        std::vector<int> ordered(config.dsv4_cpu_offload_layers.begin(), config.dsv4_cpu_offload_layers.end());
        std::sort(ordered.begin(), ordered.end());
        if (ordered.front() < 0)
            throw std::runtime_error("--cpu-offload-layers must be non-negative");
        std::cerr << "cpu_offload_layers=";
        for (std::size_t index = 0; index < ordered.size(); ++index) {
            if (index) std::cerr << ',';
            std::cerr << ordered[index];
        }
        std::cerr << std::endl;
    }
    if (options.n_gpu_layers_set && config.n_gpu_layers < 0)
        throw std::runtime_error("--n-gpu-layers must be non-negative");
    if (options.n_gpu_layers_set && !options.cpu_offload_layers.empty())
        throw std::runtime_error("--n-gpu-layers cannot be combined with --cpu-offload-layers");
    if (options.moe_gpu_cache_gb < 0.0 || !std::isfinite(options.moe_gpu_cache_gb))
        throw std::runtime_error("--moe-gpu-cache-gb must be finite and non-negative");
    if (options.moe_gpu_cache_gb > 0.0 && !options.cpu_offload_layers.empty())
        throw std::runtime_error("--moe-gpu-cache-gb cannot be combined with --cpu-offload-layers");
    if (options.moe_gpu_cache_gb > 0.0 &&
        (config.tensor_parallel.enabled() || config.expert_parallel.enabled()))
        throw std::runtime_error("--moe-gpu-cache-gb cannot be combined with tensor/expert parallelism");
    const auto* automatic = env_lookup("TYLOINFER_MOE_AUTO_CACHE_PROFILE");
    config.cache_profile = tyloi::resolve_moe_cache_profile_path(options.moe_cache_profile_path,
        options.moe_gpu_cache_gb > 0.0 ? options.model_path : std::string{}, !automatic || automatic[0] != '0');
    if (!config.cache_profile.empty() && options.moe_gpu_cache_gb <= 0.0)
        throw std::runtime_error("--moe-cache-profile requires --moe-gpu-cache-gb");
    if (options.moe_gpu_cache_gb > 0.0) {
        constexpr double gib = 1024.0 * 1024.0 * 1024.0;
        const double bytes = options.moe_gpu_cache_gb * gib;
        if (bytes < 1.0 || bytes > static_cast<double>(std::numeric_limits<int64_t>::max()) ||
            static_cast<long double>(bytes) > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("--moe-gpu-cache-gb is outside the supported range");
        config.cache_bytes = static_cast<std::size_t>(bytes);
        config.drop_file_cache = true;
    }
}

tyloi::models::LoadOptions model_options(const CudaLoadConfig& config, std::int64_t context,
        bool components, bool transposed) {
    return {.context = context, .gpu_layers = config.n_gpu_layers,
        .expert_parallel = config.expert_parallel.enabled(), .recurrent_transposed = transposed,
        .preallocate = cuda::options::kv_vram_preallocated(),
        .vision_resident = cuda::options::vision_resident(),
        .mirror_scalars = config.tensor_parallel_mirror_linear_attention_scalars,
        .yarn = config.yarn, .cpu_offload_layers = config.dsv4_cpu_offload_layers,
        .engram_cache_rows = config.deepseek_v41_engram_cache_rows,
        .host_tokens = config.cache.pipeline, .optional_components = components,
        .kv_quantization = cuda_kv_quantization()};
}

bool graph_allowed(const CudaLoadConfig& config, Executor& executor,
        const tyloi::weights::Cache& weights, const tyloi::models::Info& info) {
    const auto graph_weights = info.graph_weights == models::GraphWeights::resident
        ? tyloi::weights::Cache::Graph::legacy : tyloi::weights::Cache::Graph::native;
    return info.decode_graph && config.dense_cpu_layer_count == 0 && config.dsv4_cpu_offload_layers.empty() &&
        !executor.attention_options().kernels.decode_aten &&
        model_parallel_cuda_graph_enabled(config, executor, weights) &&
        weights.graph_safe(graph_weights, moe_parallel_config(config).enabled());
}

void configure_generation(tyloi::engine::Config& runtime, const CudaLoadConfig& config,
        Executor& executor, const tyloi::weights::Cache& weights, const tyloi::models::Info& info) {
    runtime.profile_decode = config.profile_decode;
    runtime.profile_mtp = config.profile_mtp;
    runtime.trace_moe = config.trace_moe;
    runtime.prefetch = config.prefetch;
    runtime.devices = cuda_devices(config);
    runtime.online_budget = config.layer_placement.devices.size() <= 1 &&
        !config.tensor_parallel.enabled() && !config.expert_parallel.enabled();
    runtime.continuous_batch.preallocate = cuda::options::kv_vram_preallocated();
    runtime.continuous_batch.cpu_offload = config.dense_cpu_layer_count != 0 ||
        !config.dsv4_cpu_offload_layers.empty();
    runtime.continuous_batch.cached_slots = config.weights.min_slots;
    const bool allowed = graph_allowed(config, executor, weights, info);
    runtime.decode_graph.enabled &= allowed;
    runtime.continuous_batch.cuda_graph &= allowed;
}

void configure_placement(CudaLoadConfig& config, Collectives& collectives, int primary) {
    collectives.reset();
    config.layer_placement.layer_devices.clear();

    int available = 0;
    TYLOINFER_CUDA_CHECK(cudaGetDeviceCount(&available));
    if (primary < 0 || primary >= available)
        throw std::invalid_argument("CUDA primary device is unavailable");
    validate_parallel_config(
        config.tensor_parallel,
        "--tensor-parallel", "--tensor-split", available,
        config.tensor_parallel.allow_duplicate_devices);
    validate_parallel_config(
        config.expert_parallel,
        "--expert-parallel", "--expert-split", available,
        config.expert_parallel.allow_duplicate_devices);

    if (config.tensor_parallel.enabled() &&
            config.expert_parallel.enabled() &&
            config.tensor_parallel.devices !=
                config.expert_parallel.devices) {
        throw std::runtime_error(
            "combined tensor and expert parallelism requires the same "
            "ordered CUDA device group");
    }
    if (!config.tensor_parallel.enabled() &&
            !config.expert_parallel.enabled()) {
        TYLOINFER_CUDA_CHECK(cudaSetDevice(primary));
    } else {
        const auto& group = config.tensor_parallel.enabled()
            ? config.tensor_parallel : config.expert_parallel;
        Collectives::enable_peer_access(group.devices);
        collectives.configure(
            group.devices, group.allow_duplicate_devices);
        TYLOINFER_CUDA_CHECK(cudaSetDevice(group.primary_device()));
        print_parallel_config(
            "tensor_parallel", config.tensor_parallel);
        print_parallel_config(
            "expert_parallel", config.expert_parallel);
        std::cerr << "model_parallel ranks=" << group.devices.size()
                  << " collective_backend="
                  << (collectives.enabled() ? "nccl" : "serial")
                  << '\n';
    }

    auto& placement = config.layer_placement;
    if (placement.devices.empty()) {
        if (!placement.split.empty()) {
            throw std::runtime_error(
                "--layer-split requires --layer-parallel");
        }
        placement.devices = {model_parallel_enabled(config) ? model_parallel_primary_device(config) : primary};
        return;
    }
    if (config.tensor_parallel.enabled() ||
            config.expert_parallel.enabled()) {
        throw std::runtime_error(
            "--layer-parallel cannot be combined with tensor/expert parallelism");
    }

    const ParallelConfig layer_parallel{
        placement.devices, placement.split, false};
    validate_parallel_config(
        layer_parallel,
        "--layer-parallel", "--layer-split", available, false);
    TYLOINFER_CUDA_CHECK(cudaSetDevice(placement.primary_device()));
    print_parallel_config("layer_parallel", layer_parallel);
}

std::vector<tyloi::Device> cuda_devices(const CudaLoadConfig& config) {
    std::set<int> devices(config.layer_placement.devices.begin(), config.layer_placement.devices.end());
    devices.insert(config.tensor_parallel.devices.begin(), config.tensor_parallel.devices.end());
    devices.insert(config.expert_parallel.devices.begin(), config.expert_parallel.devices.end());
    devices.insert(config.layer_placement.primary_device());
    std::vector<tyloi::Device> result;
    result.reserve(devices.size());
    for (const int device : devices) result.emplace_back(tyloi::kCUDA, device);
    return result;
}

bool model_parallel_enabled(const CudaLoadConfig& config) {
    return config.tensor_parallel.enabled() || config.expert_parallel.enabled();
}

const ParallelConfig& model_parallel_config(const CudaLoadConfig& config) {
    return config.tensor_parallel.enabled() ? config.tensor_parallel : config.expert_parallel;
}

const ParallelConfig& moe_parallel_config(const CudaLoadConfig& config) {
    return config.expert_parallel.enabled() ? config.expert_parallel : config.tensor_parallel;
}

int model_parallel_primary_device(const CudaLoadConfig& config) {
    if (!config.tensor_parallel.devices.empty()) return config.tensor_parallel.primary_device();
    if (!config.expert_parallel.devices.empty()) return config.expert_parallel.primary_device();
    return config.layer_placement.primary_device();
}

bool model_parallel_cuda_graph_enabled(const CudaLoadConfig& config,
        const Executor& executor, const tyloi::weights::Cache& weights) {
    if (!weights.graph_safe(tyloi::weights::Cache::Graph::model_parallel,
            moe_parallel_config(config).enabled())) return false;
    if (!config.tensor_parallel.enabled() && !config.expert_parallel.enabled()) return true;
    const bool enabled = config.expert_parallel.enabled() && !config.tensor_parallel.enabled()
        ? config.expert_parallel_cuda_graph : config.tensor_parallel_cuda_graph;
    return executor.collectives().enabled() && enabled;
}

} // namespace tyloi::cuda
