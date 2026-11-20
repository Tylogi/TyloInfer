#include "models/lfm3/engine.h"
#include "models/lfm3/layer_weights.h"
#include "models/lfm3/request.h"
#include "mlx_gguf.h"
#include <mlx/backend/metal/metal.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <variant>

namespace mfq::models::lfm3::d1 {
namespace {
namespace mx = mlx::core;
using mx::array;
using metal::MlxGgufWeight;
using Clock = std::chrono::steady_clock;
constexpr auto backend_name = "native-metal-mlx-lfm3";

double elapsed_ms(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

struct HostTensor {
    HostWeight payload;
    mx::Shape shape;

    bool packed() const { return !payload.packed.empty(); }
    std::size_t bytes() const { return payload.packed.size() + payload.values.size() * sizeof(std::uint16_t); }
};

using Weight = std::variant<array, MlxGgufWeight>;
using HostLayer = std::map<std::string, HostTensor>;
using DeviceLayer = std::map<std::string, Weight>;

array bf16_bits(const std::vector<std::uint16_t>& bits, const mx::Shape& shape) {
    return mx::view(array(bits.data(), shape, mx::uint16), mx::bfloat16);
}

Weight device_weight(const HostTensor& host) {
    if (host.packed()) return MlxGgufWeight::from_bytes("GGUF_Q8_0", host.payload.packed, host.shape, false);
    return bf16_bits(host.payload.values, host.shape);
}

void fuse_rows(HostLayer& weights, const char* name, std::initializer_list<const char*> sources) {
    const auto& first = weights.at(*sources.begin());
    HostTensor fused{{}, {0, first.shape[1]}};
    std::size_t bytes = 0;
    for (const auto* key : sources) {
        const auto& source = weights.at(key);
        if (source.packed() != first.packed() || source.shape[1] != first.shape[1])
            throw std::invalid_argument("incompatible d1 fused projections");
        fused.shape[0] += source.shape[0];
        bytes += source.bytes();
    }
    if (first.packed()) fused.payload.packed.reserve(bytes);
    else fused.payload.values.reserve(bytes / sizeof(std::uint16_t));
    for (const auto* key : sources) {
        auto source = weights.extract(key);
        auto& payload = source.mapped().payload;
        fused.payload.packed.insert(fused.payload.packed.end(), payload.packed.begin(), payload.packed.end());
        fused.payload.values.insert(fused.payload.values.end(), payload.values.begin(), payload.values.end());
    }
    weights.emplace(name, std::move(fused));
}

DeviceLayer upload(const HostLayer& weights) {
    DeviceLayer device;
    for (const auto& [name, weight] : weights) device.emplace(name, device_weight(weight));
    return device;
}

array fp32(const array& value) { return mx::astype(value, mx::float32); }
array rounded(const array& value) { return mx::astype(value, mx::bfloat16); }

array norm(const array& x, const array& weight, float epsilon) {
    auto f = fp32(x);
    auto scale = mx::rsqrt(mx::mean(mx::square(f), -1, true) + epsilon);
    return rounded(fp32(rounded(f * scale)) * fp32(weight));
}

array columns(const array& x, int first, int count) {
    return mx::slice(x, {0, first}, {x.shape(0), first + count});
}

array host_ids(const std::vector<int>& ids) {
    return array(ids.data(), {static_cast<int>(ids.size())}, mx::int32);
}

} // namespace

struct Engine::Impl {
    Config config;
    bool fused, streaming;
    std::string format;
    std::optional<HostTensor> host_embedding;
    std::optional<Weight> resident_embedding;
    std::optional<array> output_norm;
    std::vector<HostLayer> host_layers;
    std::vector<DeviceLayer> device_layers;
    DeviceLayer active_streamed_layer;
    const DeviceLayer* active = nullptr;
    const Request* request = nullptr;
    double load_ms = 0;
    double conversion_ms = 0;
    std::size_t resident_packed = 0, resident_bf16 = 0, streamed_bytes = 0, max_layer_bytes = 0;
    nlohmann::json manifest = nlohmann::json::array();
    std::uint64_t packed_calls = 0, bf16_calls = 0, packed_embedding_rows = 0;

    Impl(Input& input, bool fuse, bool stream)
        : config(input.config), fused(fuse), streaming(stream), format(input.format) {
        if (const char* library = std::getenv("TYLOINFER_MLX_METALLIB"); library && *library)
            mx::metal::set_metallib_path(library);
        mx::set_default_device(mx::Device::gpu);
        mx::set_default_stream(mx::new_thread_unsafe_stream(mx::Device::gpu));
        const auto start = Clock::now();
        const auto load = [&](const std::string& key, const std::vector<std::int64_t>& shape) {
            return HostTensor{input.load("model.language_model." + key, shape),
                mx::Shape(shape.begin(), shape.end())};
        };
        auto embed = load("embed_tokens.weight", {config.vocabulary, config.hidden});
        auto final_norm = load("embedding_norm.weight", {config.hidden});
        if (streaming) {
            record("embed_tokens.weight", embed, true);
            host_embedding = std::move(embed);
        } else {
            resident_embedding = device_weight(embed);
            record("embed_tokens.weight", embed, false, &*resident_embedding);
        }
        record("embedding_norm.weight", final_norm, false);
        output_norm = bf16_bits(final_norm.payload.values, final_norm.shape);

        for (std::size_t layer = 0; layer < config.layers.size(); ++layer) {
            const auto prefix = "layers." + std::to_string(layer) + ".";
            HostLayer weights;
            load_layer_weights(config, layer, [&](const std::string& key, const std::vector<std::int64_t>& shape) {
                weights.emplace(key, load(prefix + key, shape));
            });
            if (fused) {
                fuse_rows(weights, "feed_forward.fused", {"feed_forward.w1.weight", "feed_forward.w3.weight"});
                if (config.layers[layer] != "conv")
                    fuse_rows(weights, "self_attn.fused", {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight", "self_attn.gate_proj.weight"});
            }
            if (streaming) {
                std::size_t layer_bytes = 0;
                for (const auto& [name, weight] : weights) {
                    record(prefix + name, weight, true);
                    layer_bytes += weight.bytes();
                }
                max_layer_bytes = std::max(max_layer_bytes, layer_bytes);
                host_layers.push_back(std::move(weights));
            } else {
                auto device = upload(weights);
                for (const auto& [name, weight] : weights)
                    record(prefix + name, weight, false, &device.at(name));
                device_layers.push_back(std::move(device));
            }
        }
        mx::synchronize();
        load_ms = elapsed_ms(start);
        conversion_ms = input.conversion_ms;
    }

    void record(const std::string& name, const HostTensor& weight, bool host, const Weight* device = nullptr) {
        auto bytes = weight.bytes();
        if (device && weight.packed())
            bytes = std::get<MlxGgufWeight>(*device).packed_nbytes();
        if (host) streamed_bytes += bytes;
        else if (weight.packed()) resident_packed += bytes;
        else resident_bf16 += bytes;
        const bool matrix = weight.shape.size() == 2;
        manifest.push_back({{"name", name}, {"storage", weight.packed() ? "Q8_0" : "BF16"},
            {"rows", matrix ? weight.shape[0] : 0}, {"columns", matrix ? weight.shape[1] : 0},
            {"bytes", bytes}, {"placement", host ? "host" : "device"}});
    }

    const array& floating(const std::string& key) const { return std::get<array>(active->at(key)); }

    array linear(const array& x, const Weight& weight) {
        if (const auto* packed = std::get_if<MlxGgufWeight>(&weight)) {
            ++packed_calls;
            return packed->matmul_q8_bf16(x);
        }
        ++bf16_calls;
        return mx::matmul(x, mx::transpose(std::get<array>(weight)));
    }

    array project(const array& x, const char* key) { return linear(x, active->at(key)); }

    array embed(const std::vector<int>& ids) {
        if (!streaming) {
            if (const auto* packed = std::get_if<MlxGgufWeight>(&*resident_embedding)) {
                packed_embedding_rows += ids.size();
                return packed->embedding(host_ids(ids), mx::bfloat16);
            }
            return mx::take(std::get<array>(*resident_embedding), host_ids(ids), 0);
        }
        const auto& host = *host_embedding;
        const auto hidden = std::size_t(config.hidden);
        if (!host.packed()) {
            std::vector<std::uint16_t> rows;
            rows.reserve(ids.size() * hidden);
            for (int id : ids) {
                const auto start = host.payload.values.begin() + std::size_t(id) * hidden;
                rows.insert(rows.end(), start, start + hidden);
            }
            return bf16_bits(rows, {int(ids.size()), config.hidden});
        }
        const auto stride = host.payload.packed.size() / std::size_t(config.vocabulary);
        std::vector<std::uint8_t> rows;
        rows.reserve(ids.size() * stride);
        for (int id : ids) {
            const auto start = host.payload.packed.begin() + std::size_t(id) * stride;
            rows.insert(rows.end(), start, start + stride);
        }
        packed_embedding_rows += ids.size();
        return MlxGgufWeight::from_bytes("GGUF_Q8_0", rows, {int(ids.size()), config.hidden}, false)
            .embedding(mx::arange(int(ids.size()), mx::int32), mx::bfloat16);
    }

    struct LayerLease {
        Impl& engine;
        ~LayerLease() {
            engine.active = nullptr;
            engine.active_streamed_layer.clear();
        }
    };

    LayerLease enter_layer(std::size_t layer) {
        if (streaming) {
            active_streamed_layer = upload(host_layers.at(layer));
            active = &active_streamed_layer;
        } else {
            active = &device_layers.at(layer);
        }
        return {*this};
    }

    void finish_layer(array& x) { x.eval(); }

    array normalize(const array& x, std::size_t, const char* name) {
        return norm(x, floating(std::string(name) + ".weight"), config.epsilon);
    }
    array finalize(const array& x) { return norm(x, *output_norm, config.epsilon); }
    array add(const array& x, const array& y) { return rounded(fp32(x) + fp32(y)); }

    array convolve(const array& x, std::size_t) {
        const int hidden = config.hidden, tokens = x.shape(0), kernel = config.convolution;
        auto projected = project(x, "conv.in_proj.weight");
        auto signal = fp32(rounded(fp32(columns(projected, 0, hidden)) * fp32(columns(projected, 2 * hidden, hidden))));
        auto padded = mx::concatenate({mx::zeros({kernel - 1, hidden}, mx::float32), signal}, 0);
        auto weights = mx::reshape(fp32(floating("conv.conv.weight")), {hidden, kernel});
        auto mix = mx::zeros({tokens, hidden}, mx::float32);
        for (int tap = 0; tap < kernel; ++tap)
            mix = mix + mx::slice(padded, {tap, 0}, {tap + tokens, hidden})
                * mx::reshape(columns(weights, tap, 1), {1, hidden});
        auto gated = rounded(fp32(rounded(mix)) * fp32(columns(projected, hidden, hidden)));
        return project(gated, "conv.out_proj.weight");
    }

    array feed_forward(const array& x, std::size_t) {
        const int width = config.intermediate;
        const auto activate = [&](const array& gate, const array& up) {
            auto g = fp32(gate);
            return project(rounded(fp32(rounded(g * mx::sigmoid(g))) * fp32(up)), "feed_forward.w2.weight");
        };
        if (!fused) return activate(project(x, "feed_forward.w1.weight"), project(x, "feed_forward.w3.weight"));
        auto projection = project(x, "feed_forward.fused");
        return activate(columns(projection, 0, width), columns(projection, width, width));
    }

    array attention(const array& q, const array& k, const array& v, const array& gate, bool sliding) {
        const int tokens = q.shape(0), dim = config.head_dim, half = dim / 2;
        const double theta = sliding ? config.sliding_theta : config.full_theta;
        const double factor = sliding ? config.sliding_factor : config.full_factor;
        const int rotated = int(dim * factor) / 2;
        std::vector<float> inverse(half);
        for (int i = 0; i < rotated; ++i)
            inverse[i] = 1.0f / std::pow(float(theta), float(i) / float(half));
        auto positions = mx::arange(tokens, mx::int32);
        auto phase = mx::reshape(fp32(positions), {tokens, 1}) * array(inverse.data(), {1, half}, mx::float32);
        auto cosine = fp32(rounded(mx::cos(phase))), sine = fp32(rounded(mx::sin(phase)));
        const auto qk = [&](const array& projected, const char* weights, int heads) {
            auto value = norm(mx::reshape(projected, {tokens, heads, dim}), floating(weights), config.epsilon);
            auto f = fp32(mx::transpose(value, {1, 0, 2}));
            auto a = mx::slice(f, {0, 0, 0}, {heads, tokens, half});
            auto b = mx::slice(f, {0, 0, half}, {heads, tokens, dim});
            return fp32(rounded(mx::concatenate({a * cosine - b * sine, b * cosine + a * sine}, 2)));
        };
        auto queries = qk(q, "self_attn.q_norm.weight", config.heads);
        auto keys = qk(k, "self_attn.k_norm.weight", config.kv_heads);
        auto values = fp32(mx::transpose(mx::reshape(v, {tokens, config.kv_heads, dim}), {1, 0, 2}));
        const int groups_per_head = config.heads / config.kv_heads;
        if (groups_per_head > 1) {
            keys = mx::repeat(keys, groups_per_head, 0);
            values = mx::repeat(values, groups_per_head, 0);
        }
        keys = mx::transpose(keys, {0, 2, 1});
        auto group = host_ids(request->groups);
        auto key_pos = mx::reshape(positions, {1, tokens});
        auto key_group = mx::reshape(group, {1, tokens});
        auto state = array(request->state_tokens, mx::int32);
        std::vector<array> tiles;
        constexpr int tile = 256;
        for (int start = 0; start < tokens; start += tile) {
            const int end = std::min(start + tile, tokens), count = end - start;
            auto query_pos = mx::reshape(mx::slice(positions, {start}, {end}), {count, 1});
            auto query_group = mx::reshape(mx::slice(group, {start}, {end}), {count, 1});
            auto visible = sliding
                ? mx::logical_and(mx::less_equal(key_pos, query_pos), mx::greater(key_pos, query_pos - config.window))
                : mx::logical_or(mx::logical_or(mx::less(query_pos, state),
                    mx::less(key_pos, state)), mx::equal(key_group, query_group));
            auto row_q = mx::slice(queries, {0, start, 0}, {config.heads, end, dim});
            auto scores = mx::matmul(row_q, keys) * (1.0f / std::sqrt(float(dim)));
            scores = mx::where(visible, scores, array(-std::numeric_limits<float>::infinity()));
            auto tile_output = rounded(mx::matmul(mx::softmax(scores, -1, true), values));
            tile_output.eval();
            tiles.push_back(std::move(tile_output));
        }
        auto mixed = mx::transpose(mx::concatenate(tiles, 1), {1, 0, 2});
        auto gates = fp32(rounded(mx::sigmoid(fp32(gate))));
        return project(mx::reshape(rounded(fp32(mixed) * mx::expand_dims(gates, -1)), {tokens, config.heads * dim}),
            "self_attn.o_proj.weight");
    }

    array attend(const array& x, std::size_t layer) {
        const bool sliding = config.layers[layer] == "sliding_attention";
        if (!fused) return attention(project(x, "self_attn.q_proj.weight"), project(x, "self_attn.k_proj.weight"),
            project(x, "self_attn.v_proj.weight"), project(x, "self_attn.gate_proj.weight"), sliding);
        auto projection = project(x, "self_attn.fused");
        const int q = config.heads * config.head_dim, k = config.kv_heads * config.head_dim;
        return attention(columns(projection, 0, q), columns(projection, q, k), columns(projection, q + k, k),
            columns(projection, q + 2 * k, config.heads), sliding);
    }

    array run(const Request& input) {
        struct ResetRequest {
            const Request*& current;
            ~ResetRequest() { current = nullptr; }
        } reset{request};
        request = &input;
        packed_calls = bf16_calls = packed_embedding_rows = 0;
        auto hidden = forward(config, *this, embed(input.tokens));
        auto slots = mx::take(hidden, host_ids(input.slots), 0);
        auto candidates = embed(input.candidates);
        // Keep the small candidate head in FP32 so BF16 logit rounding
        // does not move a calibrated cumulative probability across 0.5.
        return mx::matmul(fp32(slots), mx::transpose(fp32(candidates)));
    }
};

Engine::Engine(Input& input, bool fused, bool streaming)
    : impl_(std::make_unique<Impl>(input, fused, streaming)) {}
Engine::~Engine() = default;

nlohmann::json Engine::ready() const {
    const auto& m = *impl_;
    return {{"ready", true}, {"backend", backend_name}, {"native_runtime", true}, {"weight_format", m.format},
        {"fused_projections", m.fused}, {"load_ms", m.load_ms}, {"weight_conversion_ms", m.conversion_ms},
        {"weight_placement", m.streaming ? "stream" : "vram"}, {"resident_packed_weight_bytes", m.resident_packed},
        {"resident_bf16_weight_bytes", m.resident_bf16}, {"streamed_weight_bytes", m.streamed_bytes},
        {"layer_buffer_bytes", m.max_layer_bytes}, {"layer_buffer_count", m.streaming ? 1 : 0},
        {"full_matrix_dequant_cache_bytes", 0}, {"tensors", m.manifest}};
}

nlohmann::json Engine::evaluate(const nlohmann::json& document) {
    auto& m = *impl_;
    const auto input = Request::parse(document, m.config);
    std::vector<double> timings;
    const auto wall_start = Clock::now();
    std::optional<array> logits;
    for (int iteration = -input.warmup; iteration < input.iterations; ++iteration) {
        const auto begin = Clock::now();
        logits = m.run(input);
        logits->eval();
        if (iteration >= 0) timings.push_back(elapsed_ms(begin));
    }
    auto contiguous = mx::contiguous(*logits);
    contiguous.eval();
    const auto* result = contiguous.data<float>();
    nlohmann::json rows = nlohmann::json::array();
    for (std::size_t row = 0; row < input.slots.size(); ++row) {
        auto values = nlohmann::json::array();
        for (std::size_t column = 0; column < input.candidates.size(); ++column) {
            const auto value = result[row * input.candidates.size() + column];
            if (!std::isfinite(value)) throw std::runtime_error("non-finite native d1 logit");
            values.push_back(value);
        }
        rows.push_back(std::move(values));
    }
    auto sorted = timings;
    std::sort(sorted.begin(), sorted.end());
    return {{"backend", backend_name}, {"native_runtime", true}, {"logits", std::move(rows)},
        {"candidate_ids", input.candidates}, {"fused_projections", m.fused}, {"weight_format", m.format},
        {"output_tokens", 0}, {"forward_wall_ms", timings}, {"forward_wall_median_ms", sorted[sorted.size() / 2]},
        {"benchmark_wall_ms", elapsed_ms(wall_start)}, {"packed_gemm_calls_per_forward", m.packed_calls},
        {"packed_embedding_rows_per_forward", m.packed_embedding_rows}, {"bf16_gemm_calls_per_forward", m.bf16_calls},
        {"fp32_candidate_gemm_calls_per_forward", 1}};
}

} // namespace mfq::models::lfm3::d1
