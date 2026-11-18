#include "model.h"
#include "weight_quant.h"

#include "engine/backends/cuda/kernels/attention/dflash2/attention.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "engine/weights/source/safetensors.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tyloi::models::dflash2 {
namespace {

Config read_config(const std::filesystem::path& directory) {
    std::ifstream input(directory / "config.json");
    nlohmann::json json;
    if (!input || !(input >> json)) throw std::invalid_argument("DFlash2 config is unavailable");
    const auto& d = json.at("dflash_config");
    Config c;
    c.hidden = json.at("hidden_size"); c.vocabulary = json.at("vocab_size");
    c.layers = json.at("num_hidden_layers"); c.intermediate = json.at("intermediate_size");
    c.heads = json.at("num_attention_heads"); c.kv_heads = json.at("num_key_value_heads");
    c.dimension = json.at("head_dim"); c.window = json.at("sliding_window");
    c.block = d.at("block_size"); c.group = d.at("conv_group_size");
    c.rank = d.at("selector_rank"); c.candidates = d.at("selector_top_k");
    c.mask = d.at("mask_token_id"); c.target_layers = d.at("target_layer_ids").get<std::vector<std::int64_t>>();
    c.theta = json.at("rope_parameters").at("rope_theta"); c.epsilon = json.at("rms_norm_eps");
    if (c.hidden <= 0 || c.hidden % 32 || c.vocabulary <= 0 || c.layers <= 0 || c.intermediate <= 0 ||
        c.intermediate % 32 || c.heads <= 0 || c.kv_heads <= 0 || c.heads % c.kv_heads || c.dimension <= 0 ||
        c.dimension > 512 || c.dimension % 2 || std::int64_t(c.heads) * c.dimension % 32 ||
        c.window < 1 || c.block < 2 || c.group <= 0 || c.hidden % c.group || d.at("conv_kernel_size") != 2 ||
        std::int64_t(c.window) + c.block + 126 > std::numeric_limits<int>::max() ||
        std::int64_t(c.heads) * c.block + 31 > std::numeric_limits<int>::max() ||
        c.rank <= 0 || c.candidates < 1 || c.candidates > std::min(64, c.vocabulary) || c.mask < 0 || c.mask >= c.vocabulary ||
        c.target_layers.empty() || c.target_layers.front() < 0 ||
        !std::is_sorted(c.target_layers.begin(), c.target_layers.end()) ||
        std::adjacent_find(c.target_layers.begin(), c.target_layers.end()) != c.target_layers.end() ||
        !std::isfinite(c.theta) || c.theta <= 0 || !std::isfinite(c.epsilon) || c.epsilon <= 0)
        throw std::invalid_argument("DFlash2 checkpoint geometry is invalid");
    return c;
}

struct Loader {
    Safetensors source;
    Device device;
    bool q4 = false;

    Loader(const std::filesystem::path& directory, Device selected) : source(directory), device(selected) {
        const auto* value = cuda::env_lookup("TYLOINFER_DFLASH2_WEIGHT_TYPE");
        if (value && value[0]) {
            if (std::strcmp(value, "q4_0") == 0) q4 = true;
            else if (std::strcmp(value, "q8_0") != 0)
                throw std::invalid_argument("TYLOINFER_DFLASH2_WEIGHT_TYPE must be q4_0 or q8_0");
        }
        std::cerr << "runtime_dflash2_linear_weights format=" << (q4 ? "Q4_0" : "Q8_0") << '\n';
    }

    DenseTensor host(const std::string& name, const std::vector<std::int64_t>& shape) const {
        const auto it = source.records().find(name);
        if (it == source.records().end() || !it->second.metadata.shape || *it->second.metadata.shape != shape)
            throw std::invalid_argument("DFlash2 tensor geometry differs: " + name);
        const auto& record = it->second;
        const auto& dtype = record.metadata.dtype;
        if (dtype != "BF16" && dtype != "F16" && dtype != "F32")
            throw std::invalid_argument("DFlash2 tensor precision differs: " + name);
        const auto type = dtype == "BF16" ? kBFloat16 : dtype == "F16" ? kFloat16 : kFloat32;
        auto result = cuda::empty(shape, TensorOptions().device(kCPU).dtype(type));
        if (result.nbytes() != record.metadata.nbytes)
            throw std::invalid_argument("DFlash2 tensor payload differs: " + name);
        source.read(record.location.shard, record.location.offset,
            static_cast<std::byte*>(result.data_ptr()), result.nbytes());
        return result;
    }

    DenseTensor dense(const std::string& name, const std::vector<std::int64_t>& shape, bool codebook = false) const {
        return host(name, shape).to(codebook ? kBFloat16 : kFloat32).to(device);
    }

    Matrix linear(const std::string& name, std::int64_t rows, std::int64_t columns) const {
        constexpr auto max = std::numeric_limits<int>::max();
        if (rows <= 0 || rows > max - 31 || columns <= 0 || columns > max - 511 || columns % 32)
            throw std::invalid_argument("DFlash2 tiled linear requires a valid Q8_0 or Q4_0 matrix");
        auto input = host(name + ".weight", {rows, columns});
        auto packed = (q4 ? cuda::dflash2::pack_q4_weight_cpu(input) : cuda::dflash2::pack_q8_weight_cpu(input)).to(device);
        const auto padded_rows = (rows + 31) / 32 * 32;
        const auto padded_columns = (columns + 511) / 512 * 512;
        const auto stride = gguf::row_bytes(q4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0, padded_columns);
        if (stride > std::uint64_t(std::numeric_limits<std::int64_t>::max() / padded_rows))
            throw std::invalid_argument("DFlash2 tiled linear size overflows");
        auto tiles = cuda::empty({std::int64_t(padded_rows * stride)}, TensorOptions().device(device).dtype(kUInt8));
        const auto stream = mfq_current_cuda_stream();
        packed.record_stream(reinterpret_cast<std::uintptr_t>(stream));
        tiles.record_stream(reinterpret_cast<std::uintptr_t>(stream));
        if (q4) cuda::dflash2::q4_tile(packed.data_ptr(), tiles.data_ptr(), int(rows), int(columns), stream);
        else cuda::dflash2::q8_tile(packed.data_ptr(), tiles.data_ptr(), int(rows), int(columns), stream);
        return {std::move(tiles), int(rows), int(columns), int(padded_rows), q4};
    }
};

} // namespace

std::unique_ptr<Draft> load(const CausalModel& target, const std::filesystem::path& directory,
    Device device, std::int64_t hidden, std::int64_t vocabulary, std::int64_t target_layers) {
    auto config = read_config(directory);
    if (config.hidden != hidden || config.vocabulary != vocabulary || config.target_layers.back() >= target_layers)
        throw std::invalid_argument("DFlash2 checkpoint is incompatible with the target");
    if (!device.is_cuda()) throw std::invalid_argument("DFlash2 requires a CUDA device");
    if (config.target_layers.size() > std::size_t(std::numeric_limits<int>::max() / config.hidden))
        throw std::invalid_argument("DFlash2 checkpoint geometry is invalid");
    const MfqCudaGuard guard(device);
    const cuda::ResidentAllocationScope resident;
    Loader loader(directory, device);
    auto model = std::make_unique<Draft>(target, config, device);
    const auto* cache_setting = cuda::env_lookup("TYLOINFER_DFLASH2_KV_BF16");
    model->compact_kv = cache_setting && cache_setting[0] == '1';
    model->context_projection = loader.linear("fc", config.hidden, std::int64_t(config.hidden) * config.target_layers.size());
    model->selector_projection = loader.linear("candidate_selector.hidden_projection", config.rank, config.hidden);
    model->hidden_norm = loader.dense("hidden_norm.weight", {config.hidden});
    model->final_norm = loader.dense("norm.weight", {config.hidden});
    model->predecessor = loader.dense("candidate_selector.predecessor_codebook", {config.vocabulary, config.rank}, true);
    const auto* stage = cuda::env_lookup("TYLOINFER_DFLASH2_STAGE_SUCCESSOR");
    if (stage && stage[0] == '1')
        model->successor = loader.host("candidate_selector.successor_codebook", {config.vocabulary, config.rank}).to(kBFloat16);
    else
        model->successor = loader.dense("candidate_selector.successor_codebook", {config.vocabulary, config.rank}, true);
    if (loader.source.find("mask_embedding")) model->mask_embedding = loader.dense("mask_embedding", {config.hidden});
    model->layers.reserve(config.layers);
    for (int index = 0; index < config.layers; ++index) {
        const auto p = "layers." + std::to_string(index) + ".";
        Layer layer;
        layer.input_norm = loader.dense(p + "input_layernorm.weight", {config.hidden});
        layer.attention_base = loader.dense(p + "attention_conv.base_kernel", {2, 2, config.hidden});
        layer.q_norm = loader.dense(p + "self_attn.q_norm.weight", {config.dimension});
        layer.k_norm = loader.dense(p + "self_attn.k_norm.weight", {config.dimension});
        layer.post_norm = loader.dense(p + "post_attention_layernorm.weight", {config.hidden});
        layer.mlp_base = loader.dense(p + "mlp_conv.base_kernel", {2, 2, config.hidden});
        layer.dynamic_attention = loader.linear(p + "attention_conv.kernel_projection", std::int64_t(4) * config.hidden / config.group, config.hidden);
        layer.q = loader.linear(p + "self_attn.q_proj", std::int64_t(config.heads) * config.dimension, config.hidden);
        layer.k = loader.linear(p + "self_attn.k_proj", std::int64_t(config.kv_heads) * config.dimension, config.hidden);
        layer.v = loader.linear(p + "self_attn.v_proj", std::int64_t(config.kv_heads) * config.dimension, config.hidden);
        layer.output = loader.linear(p + "self_attn.o_proj", config.hidden, std::int64_t(config.heads) * config.dimension);
        layer.dynamic_mlp = loader.linear(p + "mlp_conv.kernel_projection", std::int64_t(4) * config.hidden / config.group, config.hidden);
        layer.gate = loader.linear(p + "mlp.gate_proj", config.intermediate, config.hidden);
        layer.up = loader.linear(p + "mlp.up_proj", config.intermediate, config.hidden);
        layer.down = loader.linear(p + "mlp.down_proj", config.hidden, config.intermediate);
        model->layers.push_back(std::move(layer));
    }
    std::ostringstream identity;
    identity << "-dflash2-" << (loader.q4 ? "q4_0" : "q8_0") << (model->compact_kv ? "-bf16" : "-fp32");
    auto files = loader.source.paths();
    std::sort(files.begin(), files.end());
    files.push_back(loader.source.root() / "config.json");
    for (const auto& file : files) {
        const auto path = std::filesystem::canonical(file);
        const auto stamp = std::filesystem::last_write_time(path);
        identity << '\n' << path.string() << ':' << std::filesystem::file_size(path) << ':'
                 << std::chrono::duration_cast<std::chrono::nanoseconds>(stamp.time_since_epoch()).count();
    }
    model->cache_key = identity.str();
    return model;
}

} // namespace tyloi::models::dflash2
