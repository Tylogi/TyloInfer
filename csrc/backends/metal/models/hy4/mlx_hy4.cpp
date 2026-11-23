#include "mlx_hy4.h"
#include "mlx_gguf_prefill.h"
#include "mlx_moe_ops.h"

#include "mlx_gguf_layer_tensors.h"
#include "mlx_hyper_connection.h"
#include "mlx_kernel_prepare.h"
#include "mlx_sparse_attention.h"
#include "mlx_weight_residency.h"
#include "nlohmann/json.hpp"
#include "tyloinfer/host_parallel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mlx/io.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mfq::metal {
namespace {
using namespace mlx::core;
using json = nlohmann::json;
constexpr auto engram_weight = "model.associative_memory.embedding.weight";

array axis_slice(const array& input, int axis, int start, int stop) {
    if (axis < 0) axis += static_cast<int>(input.ndim());
    Shape lower(input.ndim(), 0), upper = input.shape();
    lower[axis] = start;
    upper[axis] = stop;
    return slice(input, lower, upper);
}

array rms(const array& input, float eps) {
    return fast::rms_norm(input, std::nullopt, eps);
}

struct RotaryPhases {
    array cosine, sine;
    array operator()(const array& values) const { return mlx_rope_adjacent(values, cosine, sine); }
};

RotaryPhases rotary_phases(const array& positions, int dim, float base) {
    const auto inverse = exp(astype(arange(0, dim, 2, int32), float32) * (-std::log(base) / dim));
    const auto angle = expand_dims(astype(positions, float32), -1) * inverse;
    return {cos(angle), sin(angle)};
}

struct Geometry {
    int context, hidden, vocab, layers, dense_layers, streams;
    int q_rank, kv_rank, q_dim, v_dim, rotary, window;
    int index_heads, index_width, index_budget, group_size, experts, routes;
    float eps, gate_eps, post_scale, rotary_base, window_rotary_base, expert_scale;
    std::vector<int> heads;
    std::vector<bool> sliding;
    std::vector<float> clamp;
    MlxMlaKvFormat indexed_kv;

    Dtype index_pool_dtype() const noexcept { return indexed_kv.quantization.enabled() ? float16 : float32; }
};

Geometry geometry(const json& metadata, int context) {
    const auto field = [&](const char* key) -> const json& {
        return metadata.at(std::string("hy_v4_flash.") + key);
    };
    const auto integer = [&](const char* key) { return field(key).get<int>(); };
    const auto number = [&](const char* key) { return field(key).get<float>(); };
    const int physical = integer("block_count"), nextn = integer("nextn_predict_layers");
    Geometry g{context, integer("embedding_length"), integer("vocab_size"), physical - nextn,
        integer("leading_dense_block_count"), integer("hyper_connection.count"),
        integer("attention.q_lora_rank"), integer("attention.kv_lora_rank"),
        integer("attention.key_length_mla"), integer("attention.value_length_mla"),
        integer("rope.dimension_count"), integer("attention.sliding_window"),
        integer("attention.indexer.head_count"), integer("attention.indexer.key_length"),
        integer("attention.indexer.top_k"), integer("attention.indexer.kpool"),
        integer("expert_count"), integer("expert_used_count"),
        number("attention.layer_norm_rms_epsilon"), number("hyper_connection.epsilon"),
        number("hyper_connection.magnitude"), number("rope.freq_base"), number("rope.freq_base_swa"),
        number("expert_weights_scale"), field("attention.head_count").get<std::vector<int>>(),
        field("attention.sliding_window_pattern").get<std::vector<bool>>(),
        field("swiglu_clamp_exp").get<std::vector<float>>(),
        {integer("attention.kv_lora_rank"), integer("rope.dimension_count"), mlx_kv_quantization()}};
    if (context <= 0 || context > integer("context_length") || g.hidden <= 0 || g.vocab <= 0 ||
        g.layers <= 0 || g.layers > 4096 || g.dense_layers < 0 || g.dense_layers > g.layers ||
        g.streams < 2 || g.streams > 8 || g.q_rank <= 0 || g.kv_rank <= 0 ||
        g.q_dim <= g.rotary || g.v_dim <= 0 || g.rotary <= 0 || g.rotary % 2 ||
        g.window <= 0 || g.index_heads <= 0 || g.index_width < g.rotary ||
        g.group_size <= 1 || g.index_budget / g.group_size < 2 ||
        g.index_budget % g.group_size || g.routes <= 0 || g.routes > g.experts ||
        g.heads.size() < std::size_t(g.layers) || g.sliding.size() < std::size_t(g.layers) ||
        g.clamp.size() < std::size_t(g.layers) ||
        !field("attention.indexer.kpool_select_tail").get<bool>() ||
        integer("expert_gating_func") != 2 || !field("expert_weights_norm").get<bool>()) {
        throw std::runtime_error("invalid Hy4 Flash GGUF geometry");
    }
    return g;
}

MlxHyperConnection ihc(const MlxGgufLayerTensors& tensors, const Geometry& g, const std::string& prefix) {
    auto scale = tensors.dense(prefix + "_scale.weight");
    if (scale.shape() != Shape{2}) throw std::runtime_error("Hy4 iHC scales must have two entries");
    MlxHyperConnectionConfig config;
    config.streams = g.streams;
    config.norm_epsilon = g.eps;
    config.gate_epsilon = g.gate_eps;
    config.post_scale = g.post_scale;
    return MlxHyperConnection(tensors.dense(prefix + "_fn.weight"), tensors.dense(prefix + "_base.weight"),
        config, reshape(broadcast_to(reshape(scale, {2, 1}), {2, g.streams}), {2 * g.streams}));
}

struct FeedForward {
    MlxLinear gate, up, down;
    MlxProjectionBatch projections;
    FeedForward(MlxLinear gate, MlxLinear up, MlxLinear down)
        : gate(std::move(gate)), up(std::move(up)), down(std::move(down)),
          projections({&this->gate, &this->up}) {}
    array operator()(const array& input) const { return down(projections.swiglu(input)); }
};

struct LatentRouter {
    MlxLinear projection;
    bool compact = false;

    explicit LatentRouter(MlxLinear loaded) : projection(std::move(loaded)) {
        const auto* weight = projection.dense_weight_ref();
        if (!weight || weight->dtype() != float32 || weight->shape() != Shape{512, 3072} ||
            default_device() != Device::gpu)
            return;
        auto bf16 = astype(*weight, bfloat16);
        if (!array_equal(astype(bf16, float32), *weight).item<bool>()) return;
        MlxWeightResidency::track(bf16);
        projection = MlxLinear(std::move(bf16));
        compact = true;
    }

    array operator()(const array& input) const {
        if (!compact) return projection(input);
        const auto& weight = *projection.dense_weight_ref();
        if (input.ndim() != 2 || input.shape(1) != 3072 || input.shape(0) < 1 || input.shape(0) > 32 ||
            input.dtype() != bfloat16 || default_device() != Device::gpu)
            return MlxLinear(astype(weight, float32))(input);
        static const auto kernel = [] {
            CompileOptions options;
            options.math_mode = MathMode::Fast;
            return fast::metal_kernel("hy4_f32_router_from_bf16", {"input", "weights"}, {"logits"}, R"METAL(
            {
            #pragma clang fp reassociate(off)
                const uint lane = thread_index_in_simdgroup;
                const uint first = (threadgroup_position_in_grid.x * 4u + simdgroup_index_in_threadgroup) * 2u;
                const uint first_row = TILE == 1 ? 0u : threadgroup_position_in_grid.y * TILE;
                float sum[TILE][2] = {};
                for (uint k = lane * 4u; k < 3072u; k += 128u) {
                    float4 weights_for_output[2];
                    #pragma clang loop unroll(full)
                    for (uint i = 0u; i < 2u; ++i)
                        weights_for_output[i] = float4(*reinterpret_cast<device const bfloat4*>(
                            weights + 3072u * (first + i) + k));
                    #pragma clang loop unroll(full)
                    for (uint r = 0u; r < TILE; ++r) {
                        const uint row = first_row + r;
                        if constexpr (TILE > 1) if (row >= uint(input_shape[0])) continue;
                        const auto activation = float4(*reinterpret_cast<device const bfloat4*>(
                            input + row * 3072u + k));
                        #pragma clang loop unroll(full)
                        for (uint i = 0u; i < 2u; ++i) {
                            const auto w = weights_for_output[i];
                            sum[r][i] = fma(w.x, activation.x, sum[r][i]);
                            sum[r][i] = fma(w.y, activation.y, sum[r][i]);
                            sum[r][i] = fma(w.z, activation.z, sum[r][i]);
                            sum[r][i] = fma(w.w, activation.w, sum[r][i]);
                        }
                    }
                }
                // Keep the reduction order of the FP32 router projection.
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < TILE; ++r) {
                    const uint row = first_row + r;
                    if constexpr (TILE > 1) if (row >= uint(input_shape[0])) continue;
                    #pragma clang loop unroll(full)
                    for (uint i = 0u; i < 2u; ++i) {
                        for (ushort offset = 16u; offset > 0u; offset /= 2u)
                            sum[r][i] += simd_shuffle_down(sum[r][i], offset);
                        if (lane == 0u) logits[row * 512u + first + i] = sum[r][i];
                    }
                }
            }
            )METAL", "", true, false, options);
        }();
        const int rows = input.shape(0), tile = std::min(rows, 4);
        return kernel({contiguous(input), weight}, {{rows, 512}}, {float32},
            {8192, (rows + tile - 1) / tile, 1}, {128, 1, 1}, {{"TILE", tile}},
            std::nullopt, false, {}).front();
    }
};

struct LatentMoe {
    LatentRouter router;
    MlxLinear reduce_input, expand_output;
    array bias;
    std::array<MlxGgufWeight, 3> experts;
    FeedForward shared;
    float limit;

    LatentMoe(const MlxGgufLayerTensors& t, float clamp)
        : router(t.linear("ffn_gate_inp.weight")), reduce_input(t.linear("ffn_routed_down.weight")),
          expand_output(t.linear("ffn_routed_up.weight")), bias(t.dense("exp_probs_b.bias")),
          experts{t.experts("ffn_gate_exps.weight"), t.experts("ffn_up_exps.weight"),
              t.experts("ffn_down_exps.weight")},
          shared(t.linear("ffn_gate_shexp.weight"), t.linear("ffn_up_shexp.weight"),
              t.linear("ffn_down_shexp.weight")), limit(clamp) {}

    array operator()(const array& input, const Geometry& g) const {
        const auto flat = reshape(input, {-1, g.hidden});
        const auto route = moe_sigmoid_topk(router(flat), g.routes, bias, g.expert_scale);
        const auto latent = reduce_input(flat);
        const auto& [gate, up, down] = experts;
        const auto routed = [&] {
            if (gate.input_size() == down.output_size())
                return gguf_routed_ffn(experts, latent, route.ids, route.weights, limit, true);
            const auto activation = gate.routed_swiglu(up, latent, route.ids, nullptr, limit);
            return down.routed_weighted_matmul(activation, route.ids, route.weights);
        }();
        return shared(input) + reshape(expand_output(routed), input.shape());
    }
};

struct IndexPool {
    MlxLinear query, weights, key, value, gate;
    MlxRmsNorm key_norm, pool_norm;
    array positions;
    IndexPool(const MlxGgufLayerTensors& t, const Geometry& g)
        : query(t.linear("indexer.attn_q_b.weight")), weights(t.linear("indexer.proj.weight")),
          key(t.linear("indexer.attn_k.weight")), value(t.linear("indexer_compressor_value.weight")),
          gate(t.linear("indexer_compressor_gate.weight")), key_norm(t.norm("indexer.k_norm.weight", g.eps)),
          pool_norm(t.norm("indexer_compressor_norm.weight", g.eps)),
          positions(t.dense("indexer_compressor_ape.weight")) {}
};

struct EngramProjection {
    MlxLinear kv;
    array bias, lookup_norm, gate, offset, gamma;
    EngramProjection(const MlxGgufLayerTensors& t)
        : kv(t.linear("engram_kv.weight")), bias(t.dense("engram_kv_b.weight")),
          lookup_norm(t.dense("engram_lookup_norm.weight")), gate(t.dense("engram_gate.weight")),
          offset(t.dense("engram_hc_scale.weight")), gamma(t.dense("engram_pre_gamma.weight")) {}
    array operator()(const array& streams, const array& rows, const Geometry& g) const {
        constexpr float eps = 1.1920929e-7f;
        const int tokens = streams.shape(1);
        auto input = reshape(rms(rows, eps) * lookup_norm, {1, tokens, -1});
        auto projected = reshape(kv(input), {1, tokens, g.streams + 1, g.hidden}) + bias;
        auto key = axis_slice(projected, 2, 0, g.streams);
        auto value = axis_slice(projected, 2, g.streams, g.streams + 1);
        auto dot = sum(rms(astype(streams, float32), eps) * rms(key, eps) * gate, -1, true) /
            std::sqrt(float(g.hidden));
        auto signed_root = where(dot < 0, array(-1.0f), array(1.0f)) * sqrt(maximum(abs(dot), array(1e-6f)));
        auto delta = (sigmoid(signed_root) + offset) * rms(value, eps) * gamma;
        return streams + astype(delta, streams.dtype());
    }
};

struct EngramTable {
    ModelSource::TensorReader read;
    int grams, heads, width, pad;
    std::vector<std::uint64_t> multipliers;
    std::vector<int> layers, sizes, offsets, mapping;
    mutable HostParallelPool io;

    EngramTable(const GgufModelSource& source, const json& meta)
        : read(source.mapped_tensor_reader(engram_weight)),
          grams(meta.at("hy_v4_flash.ple.ngram_size").get<int>()),
          heads(meta.at("hy_v4_flash.ple.heads_per_ngram").get<int>()),
          width(meta.at("hy_v4_flash.embedding_length_per_layer_input").get<int>()),
          pad(meta.at("hy_v4_flash.engram.pad_token_id").get<int>()),
          multipliers(meta.at("hy_v4_flash.ple.layer_multipliers").get<std::vector<std::uint64_t>>()),
          layers(meta.at("hy_v4_flash.ple.layers").get<std::vector<int>>()),
          sizes(meta.at("hy_v4_flash.ple.head_vocab_sizes").get<std::vector<int>>()),
          offsets(meta.at("hy_v4_flash.ple.head_offsets").get<std::vector<int>>()),
          mapping(meta.at("hy_v4_flash.engram.token_map").get<std::vector<int>>()) {
        const auto* tensor = source.find_tensor(engram_weight);
        if (!read || !tensor || !tensor->shape || tensor->shape->size() != 2 ||
            tensor->dtype != "F8_E4M3" || tensor->shape->back() != width ||
            grams < 2 || grams > 8 || heads <= 0 || heads > 64 || width <= 0 || width > 1024 ||
            pad < 0 || static_cast<std::size_t>(pad) >= mapping.size() || layers.empty() ||
            multipliers.size() != layers.size() * grams || offsets.size() != sizes.size() ||
            sizes.size() != layers.size() * (grams - 1) * heads) {
            throw std::runtime_error("invalid Hy4 CPU Engram table");
        }
        const auto count = tensor->shape->front();
        for (std::size_t i = 0; i < sizes.size(); ++i)
            if (sizes[i] <= 0 || offsets[i] < 0 || std::int64_t(offsets[i]) + sizes[i] > count)
                throw std::runtime_error("invalid Hy4 Engram bucket range");
    }

    array gather(int layer, const std::vector<std::int32_t>& tokens, std::size_t first) const {
        const auto found = std::find(layers.begin(), layers.end(), layer);
        if (found == layers.end()) throw std::invalid_argument("Hy4 layer has no Engram head");
        const int entry = static_cast<int>(found - layers.begin());
        const int columns = (grams - 1) * heads;
        const int count = static_cast<int>(tokens.size() - first);
        static const auto decode = [] {
            std::array<float, 256> table{};
            for (unsigned bits = 0; bits < table.size(); ++bits) {
                const int exp = (bits >> 3) & 15, man = bits & 7;
                const float mag = exp ? std::ldexp(1.0f + float(man) / 8.0f, exp - 7)
                    : std::ldexp(float(man) / 8.0f, -6);
                table[bits] = exp == 15 && man == 7 ? std::numeric_limits<float>::quiet_NaN() :
                    (bits & 128 ? -mag : mag);
            }
            return table;
        }();
        const auto row_count = std::size_t(count) * columns;
        std::vector<float> rows(row_count * width);
        std::vector<std::uint64_t> addresses(row_count);
        const auto mapped = [&](std::int32_t token) { return mapping.at(token); };
        for (int n = 0; n < count; ++n) {
            const auto position = first + n;
            auto hash = static_cast<std::uint64_t>(mapped(tokens[position])) * multipliers[entry * grams];
            for (int gram = 1; gram < grams; ++gram) {
                const auto previous = position >= static_cast<std::size_t>(gram) ? tokens[position - gram] : pad;
                hash ^= static_cast<std::uint64_t>(mapped(previous)) * multipliers[entry * grams + gram];
                for (int head = 0; head < heads; ++head) {
                    const int column = (gram - 1) * heads + head;
                    const int slot = entry * columns + column;
                    addresses[std::size_t(n) * columns + column] = (hash % sizes[slot] + offsets[slot]) * width;
                }
            }
        }
        std::vector<std::size_t> order;
        if (count >= 16) {
            order.resize(row_count);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](auto a, auto b) { return addresses[a] < addresses[b]; });
        }
        const auto decode_rows = [&](std::int64_t begin, std::int64_t end) {
            std::array<std::byte, 1024> packed;
            auto last_address = std::numeric_limits<std::uint64_t>::max();
            for (auto n = begin; n < end; ++n) {
                const auto index = order.empty() ? static_cast<std::size_t>(n) : order[n];
                const auto address = addresses[index];
                if (address != last_address) {
                    read(address, packed.data(), width);
                    last_address = address;
                }
                auto* output = rows.data() + index * width;
                for (int dim = 0; dim < width; ++dim)
                    output[dim] = decode[std::to_integer<std::uint8_t>(packed[dim])];
            }
        };
        if (order.empty()) decode_rows(0, row_count);
        else io.run(0, row_count, 64, 8, decode_rows);
        return array(rows.begin(), {1, count, columns, width}, float32);
    }
};

struct Layer {
    int heads;
    bool sliding;
    MlxHyperConnection attention_hc, ffn_hc;
    MlxRmsNorm attention_norm, post_attention_norm, ffn_norm, post_ffn_norm, query_norm, latent_norm;
    MlxLinear query_a, query_b, latent, gate, output;
    array key_absorption, value_expansion, sinks;
    std::optional<IndexPool> index;
    std::optional<FeedForward> dense;
    std::optional<LatentMoe> moe;
    std::optional<EngramProjection> engram;
    MlxProjectionBatch projections;

    Layer(const MlxGgufLayerTensors& t, const Geometry& g, int layer, bool has_engram)
        : heads(g.heads.at(layer)), sliding(g.sliding.at(layer)),
          attention_hc(ihc(t, g, "hc_attn")), ffn_hc(ihc(t, g, "hc_ffn")),
          attention_norm(t.norm("attn_norm.weight", g.eps)),
          post_attention_norm(t.norm("post_attention_norm.weight", g.eps)),
          ffn_norm(t.norm("ffn_norm.weight", g.eps)), post_ffn_norm(t.norm("post_ffw_norm.weight", g.eps)),
          query_norm(t.norm("attn_q_a_norm.weight", g.eps)), latent_norm(t.norm("attn_kv_a_norm.weight", g.eps)),
          query_a(t.linear("attn_q_a.weight")), query_b(t.linear("attn_q_b.weight")),
          latent(t.linear("attn_kv_a_mqa.weight")), gate(t.linear("attn_gate.weight")), output(t.linear("attn_output.weight")),
          key_absorption(contiguous(transpose(t.dense("attn_k_b.weight", float16), {0, 2, 1}))),
          value_expansion(contiguous(transpose(t.dense("attn_v_b.weight", float16), {0, 2, 1}))),
          sinks(t.dense("attn_sinks.weight")), projections({&query_a, &latent, &gate}) {
        if (!sliding) index.emplace(t, g);
        if (layer < g.dense_layers) dense.emplace(t.linear("ffn_gate.weight"),
            t.linear("ffn_up.weight"), t.linear("ffn_down.weight"));
        else moe.emplace(t, g.clamp.at(layer));
        if (has_engram) engram.emplace(t);
    }
};
}  // namespace

namespace {
constexpr int selected_query_tile = 16;

struct VerifiedLayerRows {
    std::optional<array> new_window, index_pairs, selections;
};

array trailing_rope(const array& values, const RotaryPhases& phases, int rotary) {
    const int unrotated = values.shape(-1) - rotary;
    if (unrotated == 0) return phases(values);
    return concatenate({axis_slice(values, -1, 0, unrotated),
        phases(axis_slice(values, -1, unrotated, values.shape(-1)))}, -1);
}

array sliding_decode_softmax(const array& scores, const array& sinks) {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return fast::metal_kernel("hy4_sliding_decode_sink_softmax", {"scores", "sinks"}, {"probabilities"}, R"METAL(
            constexpr uint KEYS = 512, GROUPS = 5;
            const uint task = threadgroup_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup, simd = simdgroup_index_in_threadgroup;
            const uint begin = thread_index_in_threadgroup * 4u;
            const auto row = scores + ulong(task) * KEYS;
            threadgroup float peaks[GROUPS], partials[GROUPS];
            float values[4], peak = -INFINITY;
            for (uint i = 0; i < 4u; ++i) {
                const uint column = begin + i;
                const float value = column == 0u ? float(sinks[task % scores_shape[1]]) :
                    column <= KEYS ? float(row[column - 1u]) : -INFINITY;
                values[i] = value;
                peak = max(peak, value);
            }
            peak = simd_max(peak);
            if (lane == 0u) peaks[simd] = peak;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0u) {
                const float largest = simd_max(lane < GROUPS ? peaks[lane] : -INFINITY);
                if (lane == 0u) peaks[0] = largest;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float sum = 0.f;
            for (uint i = 0; i < 4u; ++i) {
                values[i] = metal::fast::exp(values[i] - peaks[0]);
                sum += values[i];
            }
            sum = simd_sum(sum);
            if (lane == 0u) partials[simd] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0u) {
                const float total = simd_sum(lane < GROUPS ? partials[lane] : 0.f);
                if (lane == 0u) partials[0] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inverse = 1.f / partials[0];
            auto out = probabilities + ulong(task) * KEYS;
            for (uint i = 0; i < 4u; ++i) {
                const uint column = begin + i;
                if (column > 0u && column <= KEYS) out[column - 1u] = half(values[i] * inverse);
            }
            )METAL", "", true, false, options);
    }();
    constexpr int threads = 5 * 32;
    return kernel({scores, astype(sinks, float16)}, {scores.shape()}, {float16},
        {scores.shape(0) * scores.shape(1) * threads, 1, 1}, {threads, 1, 1},
        {}, std::nullopt, false, {}).front();
}

array latent_attention(const array& query, const array& rows, const array& sinks,
    float scale, int value_dim, const std::optional<array>& mask) {
    if (query.shape(0) == 2 && query.shape(2) == 1 && rows.shape(1) == 512 &&
        rows.shape(2) == 576 && rows.dtype() == float16 && value_dim == 512 && !mask &&
        default_device() == Device::gpu) {
        const int batch = query.shape(0), heads = query.shape(1), count = rows.shape(1);
        const auto q = astype(query, float16) * array(scale, float16);
        const auto keys = expand_dims(transpose(rows, {0, 2, 1}), 1);
        const auto logits = reshape(matmul(q, keys), {batch, heads, count});
        const auto probabilities = sliding_decode_softmax(logits, sinks);
        const auto value = axis_slice(rows, -1, 0, value_dim);
        // Tile the probability-by-value projection across heads to reuse the cached values.
        return reshape(matmul(probabilities, value), {batch, 1, heads, value_dim});
    }
    const auto keys = expand_dims(rows, 1);
    const auto values = axis_slice(keys, -1, 0, value_dim);
    const auto q = astype(query, rows.dtype());
    auto result = fast::scaled_dot_product_attention(q, keys, values, scale, "", mask, astype(sinks, q.dtype()));
    return transpose(result, {0, 2, 1, 3});
}

array as_dense_attention(const array& query, const array& rows, const array& sinks,
    int position, int first_key, int window, float scale, int value_dim) {
    std::optional<array> mask;
    if (query.shape(2) > 1) {
        const auto queries = expand_dims(arange(position, position + query.shape(2), 1, int32), -1);
        const auto columns = arange(first_key, first_key + rows.shape(1), 1, int32);
        mask = columns <= queries;
        if (window) mask = logical_and(*mask, columns > queries - window);
    }
    return latent_attention(query, rows, sinks, scale, value_dim, mask);
}

array append_sliding_decode(MlxHy4AttentionCache& cache, const Geometry& g, const array& next, int position) {
    if (!cache.window) cache.window = zeros({1, g.window, next.shape(-1)}, next.dtype());
    cache.window = mlx_circular_cache_append(*cache.window, next, position);
    return axis_slice(*cache.window, 1, 0, std::min(position + 1, g.window));
}

array sliding_attention_rows(const array& window, const array& next, int position, int history) {
    if (!history) return next;
    auto chronological = mlx_circular_cache_history(window, position);
    return concatenate({axis_slice(chronological, 1, chronological.shape(1) - history,
        chronological.shape(1)), next}, 1);
}

array sliding_attention(MlxHy4AttentionCache& cache, const Geometry& g, const array& query,
    const array& next, const array& sinks, int position, float scale, VerifiedLayerRows* verified = nullptr) {
    const int count = next.shape(1);
    if (verified) verified->new_window = next;
    if (count == 1 && !verified)
        return as_dense_attention(query, append_sliding_decode(cache, g, next, position),
            sinks, position, 0, 0, scale, g.kv_rank);
    if (!cache.window) cache.window = zeros({1, g.window, next.shape(-1)}, next.dtype());

    const int history = std::min(position, g.window - 1);
    const auto rows = sliding_attention_rows(*cache.window, next, position, history);
    const int first = position - history;
    constexpr int tile = 256;
    std::vector<array> output;
    for (int start = 0; start < count; start += tile) {
        const int stop = std::min(start + tile, count);
        const int first_key = std::max(first, position + start + 1 - g.window);
        const auto visible = axis_slice(rows, 1, first_key - first, position + stop - first);
        output.push_back(as_dense_attention(axis_slice(query, 2, start, stop), visible, sinks,
            position + start, first_key, g.window, scale, g.kv_rank));
    }
    auto attention = output.size() == 1 ? output.front() : concatenate(output, 1);
    if (!verified) cache.window = mlx_circular_cache_append(*cache.window, next, position, attention);
    return attention;
}

array batch_decode_queries(const array& packed) {
    return expand_dims(transpose(squeeze(packed, 0), {1, 0, 2}), 2);
}

array batch_dense_mla_decode(const array& packed_queries, std::vector<array> visible,
    const array& sinks, float scale, int value_dim) {
    const int batch = static_cast<int>(visible.size());
    std::vector<int> lengths;
    lengths.reserve(batch);
    for (const auto& lane : visible) lengths.push_back(lane.shape(1));
    const auto [shortest, longest] = std::minmax_element(lengths.begin(), lengths.end());
    std::optional<array> mask;
    if (*shortest != *longest) {
        const auto length = array(lengths.begin(), {batch, 1, 1, 1}, int32);
        mask = arange(*longest, int32) < length;
        for (int row = 0; row < batch; ++row)
            if (lengths[row] < *longest)
                visible[row] = pad(visible[row], {1}, {0}, {*longest - lengths[row]});
    }
    auto result = latent_attention(batch_decode_queries(packed_queries), concatenate(visible, 0),
        sinks, scale, value_dim, mask);
    return reshape(result, {1, batch, result.shape(2), result.shape(3)});
}

struct MlaWindow {
    array rows;
    array mask;
};

array uniform_mla_windows(const array& packed_queries, const array& kv, const array& mask,
    const array& sinks, float scale, int value_dim) {
    const int batch = kv.shape(0), heads = packed_queries.shape(1), count = mask.shape(1);
    const auto q = transpose(reshape(packed_queries, {heads, batch, count, packed_queries.shape(-1)}),
        {1, 0, 2, 3});
    const auto result = default_device() == Device::gpu && kv.dtype() == float16
        ? mlx_masked_dense_mla_attention(q, kv, mask, sinks, scale, value_dim)
        : latent_attention(q, kv, sinks, scale, value_dim, expand_dims(mask, 1));
    return reshape(result, {1, packed_queries.shape(2), heads, value_dim});
}

array batch_dense_mla_windows(const array& packed_queries, std::span<const MlaWindow> windows,
    const array& sinks, float scale, int value_dim) {
    const int batch = static_cast<int>(windows.size());
    const int total = packed_queries.shape(2), heads = packed_queries.shape(1);
    const int query_count = windows.front().mask.shape(1);
    bool uniform = true;
    int max_keys = 0;
    for (const auto& window : windows) {
        uniform &= window.mask.shape(1) == query_count;
        max_keys = std::max(max_keys, window.rows.shape(1));
    }
    std::vector<array> rows, masks;
    rows.reserve(batch);
    masks.reserve(batch);
    for (const auto& window : windows) {
        const int key_pad = max_keys - window.rows.shape(1);
        rows.push_back(key_pad ? pad(window.rows, {1}, {0}, {key_pad}) : window.rows);
        masks.push_back(key_pad ? pad(window.mask, {2}, {0}, {key_pad}) : window.mask);
    }
    const auto kv = concatenate(rows, 0);
    if (uniform) return uniform_mla_windows(packed_queries, kv, concatenate(masks, 0), sinks, scale, value_dim);
    const bool fp16_gpu = default_device() == Device::gpu && kv.dtype() == float16;
    std::vector<int> query_lanes;
    query_lanes.reserve(total);
    for (int lane = 0; lane < batch; ++lane)
        query_lanes.insert(query_lanes.end(), windows[lane].mask.shape(1), lane);
    const auto lanes = array(query_lanes.begin(), {total}, int32);
    const auto q = batch_decode_queries(packed_queries);
    const auto mask = reshape(concatenate(masks, 1), {total, 1, max_keys});
    const auto result = fp16_gpu ? mlx_masked_dense_mla_attention(q, kv, mask, sinks, scale, value_dim, lanes)
        : latent_attention(q, take(kv, lanes, 0), sinks, scale, value_dim, expand_dims(mask, 1));
    return reshape(result, {1, total, heads, value_dim});
}

array project_index_pairs(const IndexPool& index, const array& input) {
    const auto key = index.key_norm(index.key(input));
    return concatenate({astype(index.value(key), float32), astype(index.gate(key), float32)}, -1);
}

struct IndexScores {
    array query, weights;
};

IndexScores project_index_scores(const IndexPool& index, const Geometry& g,
    const array& input, const array& low, const RotaryPhases& phases) {
    const float scale = 1.0f / std::sqrt(float(g.index_heads * g.index_width));
    return {trailing_rope(reshape(index.query(low), {1, input.shape(1), g.index_heads, g.index_width}),
        phases, g.rotary), astype(index.weights(input), float32) * scale};
}

void append_index_pool(MlxHy4AttentionCache& cache, const IndexPool& index, const Geometry& g,
    array pairs, const std::optional<RotaryPhases>& pool_phases, VerifiedLayerRows* verified) {
    const int width = g.index_width;
    if (cache.index_tail) pairs = concatenate({*cache.index_tail, pairs}, 1);
    if (verified) verified->index_pairs = pairs;
    const int complete = pairs.shape(1) / g.group_size;
    const int split = complete * g.group_size;
    if (split < pairs.shape(1)) {
        auto tail = axis_slice(pairs, 1, split, pairs.shape(1));
        cache.index_tail = tail + zeros_like(tail);
    } else {
        cache.index_tail.reset();
    }
    if (!complete) return;
    const auto groups = reshape(axis_slice(pairs, 1, 0, split), {1, complete, g.group_size, 2 * width});
    const auto values = axis_slice(groups, -1, 0, width);
    const auto gates = axis_slice(groups, -1, width, 2 * width);
    const auto masses = softmax(gates + index.positions, 2);
    const auto pooled = index.pool_norm(sum(masses * values, 2), float32);
    const auto rotated = astype(trailing_rope(pooled, *pool_phases, g.rotary), g.index_pool_dtype());
    if (cache.offloaded_groups) cache.offloaded_groups->append(rotated);
    else cache.groups->append(rotated);
}

array token_selection(const array& groups, const Geometry& g, int position) {
    const int rows = groups.shape(1);
    const auto offsets = arange(0, g.group_size, 1, int32);
    auto ids = reshape(expand_dims(groups * g.group_size, -1) + offsets,
        {1, rows, groups.shape(2) * g.group_size});
    const auto queries = reshape(arange(position, position + rows, 1, int32), {1, rows, 1});
    ids = where(logical_and(ids >= 0, ids <= queries), ids, array(-1, int32));
    const int extra = (32 - ids.shape(-1) % 32) % 32;
    if (extra) ids = concatenate({ids, full({1, rows, extra}, array(-1, int32), int32)}, -1);
    return ids;
}

constexpr int index_token_threads = 128;

array decode_token_selection(const array& groups, const Geometry& g, int position) {
    static const auto kernel = fast::metal_kernel("hy4_index_decode_token_ids", {"groups", "position"}, {"ids"}, R"METAL(
        const uint index = thread_position_in_grid.x;
        const uint slot = index / GROUP;
        const int group = slot < HISTORY ? groups[slot] : position / GROUP;
        const int row = group * GROUP + int(index % GROUP);
        ids[index] = row <= position ? row : -1;
        )METAL");
    return kernel({groups, array(position, int32)}, {{1, 1, g.index_budget}}, {int32},
        {g.index_budget, 1, 1}, {index_token_threads, 1, 1},
        {{"GROUP", g.group_size}, {"HISTORY", groups.shape(-1)}}, std::nullopt, false, {}).front();
}

constexpr int index_gather_max_batch = 8;

std::pair<array, array> gather_index_batch(std::span<const array> views, std::span<const array> indices) {
    static const auto kernels = [] {
        std::array<fast::CustomKernelFunction, index_gather_max_batch + 1> result;
        for (int batch = 2; batch <= index_gather_max_batch; ++batch) {
            std::vector<std::string> names;
            std::string body = R"METAL(
                const uint token = thread_position_in_grid.x / 32u;
                const uint request = threadgroup_position_in_grid.y;
                const uint lane = thread_index_in_simdgroup;
                const device half* cache;
                int chosen;
                switch (request) {
                )METAL";
            for (int lane = 0; lane < batch; ++lane) {
                const auto suffix = std::to_string(lane);
                names.push_back("cache" + suffix);
                names.push_back("indices" + suffix);
                body += "case " + suffix + ": cache = cache" + suffix + "; chosen = indices" +
                    suffix + "[token]; break;\n";
            }
            body += R"METAL(
                default: return;
                }
                const uint row = request * 2048u + token;
                const auto source = reinterpret_cast<device const uint4*>(cache + ulong(max(chosen, 0)) * 576u);
                auto dest = reinterpret_cast<device uint4*>(data + ulong(row) * 576u);
                for (uint word = lane; word < 72u; word += 32u) dest[word] = source[word];
                if (lane == 0u) mask[row] = half(chosen >= 0 ? 0.0f : -INFINITY);
                )METAL";
            result[batch] = fast::metal_kernel("hy4_decode_gather_index_batch_" + std::to_string(batch),
                names, {"data", "mask"}, body);
        }
        return result;
    }();
    const int batch = static_cast<int>(views.size());
    std::vector<array> inputs;
    inputs.reserve(2 * batch);
    for (int lane = 0; lane < batch; ++lane) {
        inputs.push_back(views[lane]);
        inputs.push_back(indices[lane]);
    }
    auto out = kernels[batch](inputs, {{batch, 1, 2048, 576}, {batch, 1, 2048}}, {float16, float16},
        {2048 * 32, batch, 1}, {128, 1, 1}, {}, std::nullopt, false, {});
    return {std::move(out[0]), std::move(out[1])};
}

array choose_resident_tokens(const array& groups, const Geometry& g,
    const array& query, const array& weights, int position) {
    const int take = query.shape(1);
    const int budget = g.index_budget / g.group_size - 1;
    const int available = std::min(groups.shape(1), (position + take - 1) / g.group_size);
    const bool gpu = default_device() == Device::gpu;
    if (gpu && g.group_size == 4 && g.index_budget == 2048 && take <= 6 &&
        available > budget && available <= 8192) {
        const auto scores = mlx_sparse_indexer_weighted_scores(query, axis_slice(groups, 1, 0, available), weights);
        return mlx_sparse_indexer_token_ids(scores, position);
    }
    const bool direct = gpu && take == 1 && available > budget && g.index_budget % index_token_threads == 0;
    const auto select = [budget](const array& scores) {
        return astype(sort(axis_slice(argpartition(-scores, budget - 1, -1), -1, 0, budget), -1), int32);
    };
    if (direct) {
        // All scored groups end before the current group. Only its final rows need masking.
        const auto scores = mlx_sparse_indexer_weighted_scores(query, axis_slice(groups, 1, 0, available), weights);
        return decode_token_selection(select(scores), g, position);
    }
    const auto current = reshape(floor_divide(arange(position, position + take, 1, int32),
        array(g.group_size)), {1, take, 1});
    const auto all_visible = reshape(arange(0, budget, 1, int32), {1, 1, budget});
    if (available <= budget) {
        const auto ids = where(all_visible < current, all_visible, array(-1, int32));
        return token_selection(concatenate({ids, current}, -1), g, position);
    }
    const auto history = reshape(arange(0, available, 1, int32), {1, 1, available});
    const auto scores = mlx_sparse_indexer_weighted_scores(query, axis_slice(groups, 1, 0, available), weights);
    const auto masked = where(history < current, scores, array(-std::numeric_limits<float>::infinity()));
    auto ids = where(current <= budget, all_visible, select(masked));
    ids = where(ids < current, ids, array(-1, int32));
    return token_selection(concatenate({ids, current}, -1), g, position);
}

array append_indexed_cache(MlxHy4AttentionCache& cache, const IndexPool& index, const Geometry& g,
    const array& latent, const array& index_pairs, const std::optional<RotaryPhases>& pool_phases,
    const std::shared_ptr<QsaKvStore>& offload, VerifiedLayerRows* verified) {
    const auto stored = g.indexed_kv.pack(latent);
    if (!cache.latent && !cache.offloaded_latent) {
        if (offload) {
            cache.offloaded_latent.emplace(offload, stored.shape(-1), stored.dtype(), 1);
            cache.offloaded_groups.emplace(offload, g.index_width, g.index_pool_dtype(), kMlxQsaIndexPriority);
        } else {
            cache.latent.emplace(g.context, stored.shape(-1), stored.dtype());
            cache.groups.emplace(std::max(1, g.context / g.group_size), g.index_width, g.index_pool_dtype());
        }
    }
    if (cache.offloaded_latent) cache.offloaded_latent->append(stored);
    else cache.latent->append(stored);
    append_index_pool(cache, index, g, index_pairs, pool_phases, verified);
    return stored;
}

array indexed_visibility(const array& ids, int keys) {
    const int batch = ids.shape(0), rows = ids.shape(1);
    const auto allowed = ids >= 0;
    const auto visible = put_along_axis(zeros({batch, rows, keys + 1}, bool_),
        where(allowed, ids, array(keys, int32)), allowed, -1);
    return axis_slice(visible, -1, 0, keys);
}

array batch_resident_indexed_mla(const array& query, std::span<const array> caches,
    std::span<const array> selections, const array& sinks, float scale, int value_dim, bool dense) {
    if (dense) {
        const auto& shape = selections.front().shape();
        if (default_device() == Device::gpu &&
            std::all_of(selections.begin(), selections.end(), [&](const auto& ids) { return ids.shape() == shape; })) {
            int keys = 0;
            for (const auto& cache : caches) keys = std::max(keys, cache.shape(1));
            std::vector<array> rows;
            rows.reserve(caches.size());
            for (const auto& cache : caches) {
                const int extra = keys - cache.shape(1);
                rows.push_back(extra ? pad(cache, {1}, {0}, {extra}) : cache);
            }
            const auto mask = indexed_visibility(concatenate(std::vector<array>(selections.begin(), selections.end()), 0), keys);
            return uniform_mla_windows(query, concatenate(rows, 0), mask, sinks, scale, value_dim);
        }
        std::vector<MlaWindow> windows;
        for (std::size_t lane = 0; lane < caches.size(); ++lane) {
            windows.push_back({caches[lane], indexed_visibility(selections[lane], caches[lane].shape(1))});
        }
        return batch_dense_mla_windows(query, windows, sinks, scale, value_dim);
    }

    int selected = 0;
    for (const auto& ids : selections) selected = std::max(selected, ids.shape(-1));
    std::vector<array> rows, masks;
    for (std::size_t lane = 0; lane < caches.size(); ++lane) {
        const auto& raw = selections[lane];
        const int extra = selected - raw.shape(-1);
        const auto ids = extra ? pad(raw, {-1}, {0}, {extra}, array(-1, int32)) : raw;
        const int count = ids.shape(1);
        rows.push_back(reshape(mlx_gather_selected_mla_rows(caches[lane], ids),
            {count, 1, selected, caches[lane].shape(-1)}));
        masks.push_back(reshape(where(ids >= 0, array(0.f), array(-std::numeric_limits<float>::infinity())),
            {count, 1, selected}));
    }
    auto result = mlx_gathered_mla_attention(batch_decode_queries(query), concatenate(rows, 0),
        concatenate(masks, 0), sinks, scale, value_dim, true);
    return reshape(result, {1, query.shape(2), query.shape(1), value_dim});
}

array run_indexed_attention(MlxHy4AttentionCache& cache, const IndexPool& index, const Geometry& g,
    const array& query, const array& latent, const array& index_pairs, const std::optional<IndexScores>& scoring,
    const std::optional<RotaryPhases>& pool_phases, const array& sinks, int position, float scale,
    const std::shared_ptr<QsaKvStore>& offload, VerifiedLayerRows* verified) {
    const int count = latent.shape(1), end = position + count;
    const auto stored = append_indexed_cache(cache, index, g, latent, index_pairs, pool_phases, offload, verified);

    if (end <= g.index_budget) {
        cache.last_indices = reshape(arange(0, end, 1, int32), {1, 1, end});
        const auto rows = cache.offloaded_latent
            ? reshape(cache.offloaded_latent->gather(*cache.last_indices), {1, end, stored.shape(-1)})
            : cache.latent->view();
        return as_dense_attention(query, g.indexed_kv.unpack(rows), sinks, position, 0, 0, scale, g.kv_rank);
    }

    const auto& [iq, head_weights] = scoring.value();
    const auto& quantization = g.indexed_kv.quantization;
    std::optional<array> resident;
    // A bounded packed view can supply every query tile of a prefill chunk.
    if (cache.latent) resident = cache.latent->view();
    else if (!verified && count >= 4 && (end <= 2 * g.index_budget || (count >= 64 && quantization.enabled())))
        resident = cache.offloaded_latent->try_view();
    std::optional<array> dense_rows;
    if (resident && count >= 4 && end <= 2 * g.index_budget)
        dense_rows = g.indexed_kv.unpack(*resident);
    auto pooled = cache.groups ? std::optional<array>(cache.groups->view()) : cache.offloaded_groups->try_view();
    if (pooled && pooled->dtype() != float32) pooled = astype(*pooled, float32);
    const bool dense_prefill = dense_rows && !verified && !quantization.enabled() && default_device() == Device::gpu;
    // Packed recalls can share more SSD blocks across queries within one gather.
    const bool packed_recall = cache.offloaded_latent && end > 2 * g.index_budget &&
        quantization.enabled() && quantization.value_bits() <= 4;
    const int query_tile = dense_prefill ? 64 : (packed_recall ? 32 : selected_query_tile);
    std::vector<array> output;
    for (int start = 0; start < count; start += query_tile) {
        const int stop = std::min(count, start + query_tile);
        const auto q = axis_slice(iq, 1, start, stop);
        const auto weights = axis_slice(head_weights, 1, start, stop);
        const auto ids = pooled ? choose_resident_tokens(*pooled, g, q, weights, position + start)
            : token_selection(cache.offloaded_groups->select(
                  q, position + start, g.group_size, g.index_budget, weights, true), g, position + start);
        if (verified) verified->selections = ids;
        const auto queries = axis_slice(query, 2, start, stop);
        if (dense_rows && stop - start >= 4) {
            // At short contexts a dense boolean mask costs less than staging
            // a separate copy of the selected latent rows for each query.
            output.push_back(latent_attention(queries, *dense_rows, sinks, scale, g.kv_rank,
                expand_dims(indexed_visibility(ids, end), 1)));
        } else {
            const bool allow_remap = stop - start >= 16 && quantization.enabled();
            const auto staged = cache.offloaded_latent ? cache.offloaded_latent->gather_union(
                ids, resident ? &*resident : nullptr, allow_remap) : std::nullopt;
            if (staged && !staged->remapped) {
                output.push_back(latent_attention(queries, g.indexed_kv.unpack(staged->rows), sinks,
                    scale, g.kv_rank, staged->selection));
            } else {
                const auto mask = where(ids >= 0, array(0.0f), array(-std::numeric_limits<float>::infinity()));
                if (staged) {
                    const auto unpacked = g.indexed_kv.unpack(staged->rows);
                    const auto safe = reshape(maximum(staged->selection, array(0, int32)), {-1});
                    const auto rows = reshape(take(unpacked, safe, 1), {1, ids.shape(1), ids.shape(2), unpacked.shape(-1)});
                    output.push_back(mlx_gathered_mla_attention(queries, rows, mask, sinks, scale, g.kv_rank));
                } else if (resident && !quantization.enabled()) {
                    output.push_back(mlx_sparse_selected_mla_attention(queries, *resident, ids, mask, sinks, scale, g.kv_rank));
                } else {
                    const auto packed = [&] {
                        if (!resident) return cache.offloaded_latent->gather(ids);
                        const auto safe = reshape(maximum(ids, array(0, int32)), {-1});
                        return reshape(take(*resident, safe, 1), {1, ids.shape(1), ids.shape(2), stored.shape(-1)});
                    }();
                    output.push_back(mlx_gathered_mla_attention(queries, g.indexed_kv.unpack(packed),
                        mask, sinks, scale, g.kv_rank));
                }
            }
        }
        if (stop == count) cache.last_indices = axis_slice(ids, 1, stop - start - 1, stop - start);
    }
    return output.size() == 1 ? output.front() : concatenate(output, 1);
}

std::optional<std::pair<array, array>> normalize_mla_decode(const array& query, const array& kv,
    const MlxRmsNorm& query_norm, const MlxRmsNorm& latent_norm, const RotaryPhases& phases,
    const Geometry& g, bool packed) {
    const int count = query.shape(1);
    if (g.kv_rank != 512 || g.rotary != 64 || count <= 0 || count > (packed ? 64 : 1) ||
        query.shape() != Shape{1, count, 1024} || kv.shape() != Shape{1, count, 576} ||
        query.dtype() != bfloat16 || kv.dtype() != bfloat16 || query_norm.width() != 1024 ||
        latent_norm.width() != 512 || phases.cosine.size() != std::size_t(count) * 32 ||
        phases.cosine.shape() != phases.sine.shape() ||
        default_device() != Device::gpu)
        return std::nullopt;
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return fast::metal_kernel("hy4_mla_decode_norm_rope",
            {"qa", "kv", "qw", "kw", "cosines", "sines", "epsilon"}, {"low", "latent"}, R"METAL(
            const uint row = PACKED_ROWS ? threadgroup_position_in_grid.x >> 1u : 0u;
            const bool key = (threadgroup_position_in_grid.x & 1u) != 0u;
            const auto query_input = qa + row * 1024u;
            const auto key_input = kv + row * 576u;
            const auto query_output = low + row * 1024u;
            const auto key_output = latent + row * 576u;
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint simd = simdgroup_index_in_threadgroup;
            if (key && tid >= 128u && tid < 192u) {
                const uint rotary_column = tid - 128u;
                const uint pair = rotary_column >> 1u;
                const uint pair_base = 512u + (pair << 1u);
                const float cosine = cosines[row * 32u + pair];
                const float sine = sines[row * 32u + pair];
                const float first = float(key_input[pair_base]);
                const float second = float(key_input[pair_base + 1u]);
                const float result = (rotary_column & 1u) == 0u
                    ? first * cosine - second * sine
                    : first * sine + second * cosine;
                key_output[512u + rotary_column] = half(float(bfloat(result)));
            }
            {
            #pragma clang fp reassociate(off)
                const uint width = key ? 512u : 1024u;
                const uint column = tid * 4u;
                const auto input = key ? key_input : query_input;
                float values[4] = {};
                float total = 0.0f;
                if (column < width) {
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        values[i] = float(input[column + i]);
                        total += values[i] * values[i];
                    }
                }
                total = simd_sum(total);
                threadgroup float groups[8];
                threadgroup float inv_rms;
                if (lane == 0u) groups[simd] = total;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd == 0u) {
                    const float square_sum = simd_sum(lane < 8u ? groups[lane] : 0.0f);
                    if (lane == 0u) inv_rms = precise::rsqrt(square_sum / float(width) + epsilon[key ? 1u : 0u]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (column < width) {
                    const auto weights = key ? kw : qw;
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        const bfloat result = bfloat((values[i] * inv_rms) * weights[column + i]);
                        if (key) key_output[column + i] = half(float(result));
                        else query_output[column + i] = result;
                    }
                }
            }
            )METAL", "", true, false, options);
    }();
    auto outputs = kernel({query, kv, query_norm.weight(), latent_norm.weight(), phases.cosine, phases.sine,
        array({query_norm.eps(), latent_norm.eps()}, float32)}, {query.shape(), kv.shape()}, {bfloat16, float16},
        {count * 512, 1, 1}, {256, 1, 1}, {{"PACKED_ROWS", count > 1}}, std::nullopt, false, {});
    return std::pair{std::move(outputs[0]), std::move(outputs[1])};
}

struct MlaRows {
    array query, latent, query_low, gate;
};

std::optional<array> absorb_mla_decode_query(const array& query, const array& weights,
    const RotaryPhases& phases) {
    const int heads = query.shape(2);
    if (query.shape() != Shape{1, 1, heads, 256} || query.dtype() != bfloat16 ||
        weights.shape() != Shape{heads, 192, 512} || weights.dtype() != float16 ||
        phases.cosine.size() != 32 || phases.sine.shape() != phases.cosine.shape() ||
        default_device() != Device::gpu)
        return std::nullopt;
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return fast::metal_kernel("hy4_decode_absorb_rope", {"q", "w", "cs", "sn"}, {"output"}, R"METAL(
            const uint head = threadgroup_position_in_grid.y;
            const uint stripe = threadgroup_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint subgroup = simdgroup_index_in_threadgroup;
            const auto query = q + head * 256u;
            auto out = output + head * 576u;
            if (stripe == 16u) {
                const uint at = thread_index_in_threadgroup;
                const uint pair = at / 2u;
                const float first = float(query[192u + 2u * pair]);
                const float second = float(query[193u + 2u * pair]);
                const float value = (at & 1u) ? first * sn[pair] + second * cs[pair]
                    : first * cs[pair] - second * sn[pair];
                out[512u + at] = float(bfloat(value));
                return;
            }
            {
            #pragma clang fp reassociate(off)
                const uint column = stripe * 32u + subgroup * 16u + (lane & 3u) * 4u;
                const uint part = lane / 4u;
                const auto weights = w + head * 192u * 512u + column;
                float4 acc(0.0f);
                for (uint k = part * 4u; k < 192u; k += 32u) {
                    #pragma clang loop unroll(full)
                    for (uint i = 0u; i < 4u; ++i) {
                        const float activation = float(half(float(query[k + i])));
                        const auto values = float4(*reinterpret_cast<device const half4*>(weights + (k + i) * 512u));
                        acc = fma(values, float4(activation), acc);
                    }
                }
                for (ushort offset = 16u; offset >= 4u; offset /= 2u)
                    acc += simd_shuffle_down(acc, offset);
                if (part == 0u) {
                    #pragma clang loop unroll(full)
                    for (uint i = 0u; i < 4u; ++i) out[column + i] = float(half(acc[i]));
                }
            }
            )METAL", "", true, false, options);
    }();
    return kernel({query, weights, phases.cosine, phases.sine}, {{1, heads, 1, 576}}, {float32},
        {17 * 64, heads, 1}, {64, 1, 1}, {}, std::nullopt, false, {}).front();
}

array absorb_mla_query(const array& query, const array& weights) {
    const auto source = astype(query, weights.dtype());
    const int heads = source.shape(1);
    if (source.shape(2) != 2 || source.shape(3) != 192 || weights.shape() != Shape{heads, 192, 512} ||
        weights.dtype() != float16 || default_device() != Device::gpu)
        return matmul(source, weights);
    static const auto kernel = fast::metal_kernel("hy4_decode_absorb_query_pair", {"x", "w"}, {"out"}, R"METAL(
        constexpr uint DIM = 192, WIDTH = 512;
        const uint head = threadgroup_position_in_grid.y;
        const uint lane = thread_index_in_simdgroup, subgroup = simdgroup_index_in_threadgroup;
        const uint column = (threadgroup_position_in_grid.x * 32u + lane) * 4u;
        const auto first = x + head * 2u * DIM;
        const auto second = first + DIM;
        const auto weight = w + head * DIM * WIDTH + column;
        float4 first_sum(0.f), second_sum(0.f);
        // Reuse each weight vector for both queries of this head.
        for (uint k = subgroup; k < DIM; k += 4u) {
            const auto v = float4(*reinterpret_cast<device const half4*>(weight + k * WIDTH));
            first_sum = fma(v, float4(first[k]), first_sum);
            second_sum = fma(v, float4(second[k]), second_sum);
        }
        threadgroup float4 partial[4][2][32];
        partial[subgroup][0][lane] = first_sum;
        partial[subgroup][1][lane] = second_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (subgroup == 0u) {
            first_sum = (first_sum + partial[1][0][lane]) + (partial[2][0][lane] + partial[3][0][lane]);
            second_sum = (second_sum + partial[1][1][lane]) + (partial[2][1][lane] + partial[3][1][lane]);
            auto dest = out + head * 2u * WIDTH + column;
            *reinterpret_cast<device half4*>(dest) = half4(first_sum);
            *reinterpret_cast<device half4*>(dest + WIDTH) = half4(second_sum);
        }
        )METAL");
    return kernel({source, weights}, {{1, heads, 2, 512}}, {float16},
        {512, heads, 1}, {128, 1, 1}, {}, std::nullopt, false, {}).front();
}

template <class Attention>
MlaRows project_mla(const Attention& layer, const Geometry& g, const array& in, const RotaryPhases& phases,
    bool packed = false) {
    const int count = in.shape(1), plain = g.q_dim - g.rotary;
    const auto projected = layer.projections(in);
    auto normalized = normalize_mla_decode(projected[0], projected[1], layer.query_norm, layer.latent_norm, phases, g, packed);
    auto low = normalized ? std::move(normalized->first) : layer.query_norm(projected[0]);
    const auto q = reshape(layer.query_b(low), {1, count, layer.heads, g.q_dim});
    auto query = [&] {
        if (count == 1 && plain == 192 && g.rotary == 64)
            if (auto fused = absorb_mla_decode_query(q, layer.key_absorption, phases)) return std::move(*fused);
        const auto no_rope = transpose(axis_slice(q, -1, 0, plain), {0, 2, 1, 3});
        const auto absorbed = absorb_mla_query(no_rope, layer.key_absorption);
        const auto rope_q = transpose(phases(axis_slice(q, -1, plain, g.q_dim)), {0, 2, 1, 3});
        return concatenate({absorbed, astype(rope_q, float32)}, -1);
    }();
    auto latent = normalized ? std::move(normalized->second) : astype(concatenate({
        layer.latent_norm(axis_slice(projected[1], -1, 0, g.kv_rank), float32),
        astype(phases(axis_slice(projected[1], -1, g.kv_rank, g.kv_rank + g.rotary)), float32)}, -1), float16);
    return {std::move(query), std::move(latent), std::move(low), projected[2]};
}

template <class Attention>
array expand_mla(const Attention& layer, const Geometry& g, const array& attention, const array& gate,
    Dtype dtype, bool normalize = true) {
    const auto expanded = matmul(astype(transpose(attention, {0, 2, 1, 3}), layer.value_expansion.dtype()),
        layer.value_expansion);
    const auto gated = reshape(transpose(expanded, {0, 2, 1, 3}),
        {1, attention.shape(1), layer.heads * g.v_dim}) * sigmoid(astype(gate, float32));
    auto output = layer.output(astype(gated, dtype));
    return normalize ? layer.post_attention_norm(output) : output;
}

struct PendingResidual {
    array branch;
    MlxHyperConnectionState mixer;
    const MlxRmsNorm* norm = nullptr;

    array normalized() const { return norm ? (*norm)(branch) : branch; }
};

MlxHyperConnectionState prepare_attention_input(const Layer& layer, const Geometry& g, array& streams,
    const std::optional<array>& engram, const PendingResidual* deferred, bool packed_decode = false) {
    if (engram) {
        if (deferred) streams = MlxHyperConnection::expand(streams, deferred->normalized(), deferred->mixer);
        streams = (*layer.engram)(streams, *engram, g);
    } else if (deferred) {
        auto [updated, gates] = layer.attention_hc.expand_and_collapse(
            streams, deferred->branch, deferred->mixer, &layer.attention_norm, packed_decode, deferred->norm);
        streams = std::move(updated);
        return gates;
    }
    return layer.attention_hc.collapse(streams, &layer.attention_norm, packed_decode);
}

std::pair<array, MlxHyperConnectionState> prepare_layer(const Layer& layer, const Geometry& g, MlxHy4AttentionCache& cache, array streams,
    const std::optional<array>& engram, const RotaryPhases& phases,
    const std::optional<RotaryPhases>& pool_phases, int offset,
    const std::shared_ptr<QsaKvStore>& offload, VerifiedLayerRows* verified,
    std::optional<PendingResidual>* deferred) {
    const auto mix = prepare_attention_input(layer, g, streams, engram,
        deferred && *deferred ? &**deferred : nullptr);
    const auto& in = mix.branch;
    const auto rows = project_mla(layer, g, in, phases);
    const float scale = 1.0f / std::sqrt(float(g.q_dim));
    const auto attention = [&] {
        if (layer.sliding)
            return sliding_attention(cache, g, rows.query, rows.latent, layer.sinks, offset, scale, verified);
        std::optional<IndexScores> scoring;
        if (offset + in.shape(1) > g.index_budget)
            scoring = project_index_scores(*layer.index, g, in, rows.query_low, phases);
        return run_indexed_attention(cache, *layer.index, g, rows.query, rows.latent,
            project_index_pairs(*layer.index, in), scoring, pool_phases, layer.sinks, offset, scale, offload, verified);
    }();
    const auto attn_out = expand_mla(layer, g, attention, rows.gate, in.dtype(), !deferred);
    auto [updated_streams, ffn_mix] = layer.ffn_hc.expand_and_collapse(
        streams, attn_out, mix, &layer.ffn_norm, false, deferred ? &layer.post_attention_norm : nullptr);
    streams = std::move(updated_streams);
    return {std::move(streams), std::move(ffn_mix)};
}

array project_ffn(const Layer& layer, const Geometry& g, const array& input, bool normalize = true) {
    auto output = layer.dense ? (*layer.dense)(input) : (*layer.moe)(input, g);
    return normalize ? layer.post_ffn_norm(output) : output;
}

array run_layer(const Layer& layer, const Geometry& g, MlxHy4AttentionCache& cache, array streams,
    const std::optional<array>& engram, const RotaryPhases& phases,
    const std::optional<RotaryPhases>& pool_phases, int offset,
    const std::shared_ptr<QsaKvStore>& offload, VerifiedLayerRows* verified,
    std::optional<PendingResidual>* deferred) {
    auto [updated, ffn_mix] = prepare_layer(layer, g, cache, std::move(streams),
        engram, phases, pool_phases, offset, offload, verified, deferred);
    auto ffn_out = project_ffn(layer, g, ffn_mix.branch, !deferred);
    if (deferred) {
        deferred->emplace(std::move(ffn_out), std::move(ffn_mix), &layer.post_ffn_norm);
        return updated;
    }
    return MlxHyperConnection::expand(updated, ffn_out, ffn_mix);
}

using DraftTensors = std::unordered_map<std::string, array>;

class DraftWeights {
public:
    DraftWeights(const DraftTensors& tensors, std::string prefix)
        : tensors_(tensors), prefix_(std::move(prefix)) {}
    array tensor(const std::string& name) const {
        const auto found = tensors_.find("model.dfly_layers." + prefix_ + name);
        if (found == tensors_.end() || found->second.dtype() != bfloat16)
            throw std::runtime_error("missing BF16 DFly2 tensor " + prefix_ + name);
        return found->second;
    }
    MlxLinear linear(const std::string& name) const { return MlxLinear(tensor(name + ".weight")); }
    MlxRmsNorm norm(const std::string& name, float eps) const { return MlxRmsNorm(tensor(name + ".weight"), eps); }
    FeedForward ffn() const {
        return FeedForward(linear("gate_proj"), linear("up_proj"), linear("down_proj"));
    }

private:
    const DraftTensors& tensors_;
    std::string prefix_;
};

struct DflyLayer {
    int heads;
    MlxRmsNorm attention_norm, post_attention_norm, query_norm, latent_norm, ffn_norm;
    MlxLinear query_a, query_b, latent, gate, output;
    array key_absorption, value_expansion, sinks;
    FeedForward ffn;
    MlxProjectionBatch projections;

    DflyLayer(const DraftWeights& t, const Geometry& g, const int head_count)
        : heads(head_count),
          attention_norm(t.norm("input_layernorm", g.eps)),
          post_attention_norm(t.norm("self_attn.post_proj_norm", g.eps)),
          query_norm(t.norm("self_attn.q_a_layernorm", g.eps)),
          latent_norm(t.norm("self_attn.kv_a_layernorm", g.eps)),
          ffn_norm(t.norm("post_attention_layernorm", g.eps)),
          query_a(t.linear("self_attn.q_a_proj")), query_b(t.linear("self_attn.q_b_proj")),
          latent(t.linear("self_attn.kv_a_proj_with_mqa")), gate(t.linear("self_attn.linear_gate")),
          output(t.linear("self_attn.o_proj")),
          key_absorption(partition_kv(t, g, heads, false)), value_expansion(partition_kv(t, g, heads, true)),
          sinks(t.tensor("self_attn.learnable_sink_param")),
          ffn(FeedForward(t.linear("mlp.gate_proj"), t.linear("mlp.up_proj"), t.linear("mlp.down_proj"))),
          projections({&query_a, &latent, &gate}) {}

    static array partition_kv(const DraftWeights& t, const Geometry& g, int heads, bool value) {
        const int no_rotary = g.q_dim - g.rotary;
        auto weight = reshape(t.tensor("self_attn.kv_b_proj.weight"), {heads, no_rotary + g.v_dim, g.kv_rank});
        weight = value ? transpose(axis_slice(weight, 1, no_rotary, no_rotary + g.v_dim), {0, 2, 1})
            : axis_slice(weight, 1, 0, no_rotary);
        return contiguous(astype(weight, float16));
    }

    array operator()(const Geometry& g, const array& input, const RotaryPhases& phases,
        const array& history, const array& valid) const {
        const auto rows = project_mla(*this, g, attention_norm(input), phases);
        const auto keys = concatenate({history, rows.latent}, 1);
        const int count = input.shape(1);
        auto relative = arange(0, count, 1, int32);
        auto causal = relative <= expand_dims(relative, 1);
        const auto mask = concatenate({broadcast_to(valid, {count, history.shape(1)}), causal}, 1);
        const float scale = 1.0f / std::sqrt(float(g.q_dim));
        const auto attention = latent_attention(rows.query, keys, sinks, scale, g.kv_rank, mask);
        auto residual = input + expand_mla(*this, g, attention, rows.gate, input.dtype());
        return residual + ffn(ffn_norm(residual));
    }
};

struct DflyDraft {
    MlxRmsNorm embedding_norm, target_norm, correction_hidden_norm, correction_embedding_norm, output_norm;
    MlxLinear fuse;
    array fuse_gate;
    FeedForward correction;
    int max_depth, mask_id, local_window;
    std::vector<int> target_layers, index_layers;
    std::vector<DflyLayer> layers;

    DflyDraft(const DraftTensors& tensors, const json& config, const Geometry& g)
        : embedding_norm(DraftWeights(tensors, "mtp_fusion.").norm("enorm", g.eps)),
          target_norm(DraftWeights(tensors, "mtp_fusion.").norm("hnorm", g.eps)),
          correction_hidden_norm(DraftWeights(tensors, "hidden_correction.").norm("hidden_norm", g.eps)),
          correction_embedding_norm(DraftWeights(tensors, "hidden_correction.").norm("embed_norm", g.eps)),
          output_norm(DraftWeights(tensors, "").norm("final_norm", g.eps)),
          fuse(DraftWeights(tensors, "mtp_fusion.").linear("eh_proj")),
          fuse_gate(DraftWeights(tensors, "mtp_fusion.").tensor("gate")),
          correction(DraftWeights(tensors, "hidden_correction.").ffn()),
          max_depth(config.at("speculative_config").at("num_speculative_tokens").get<int>()),
          mask_id(config.at("dfly_mask_token_id").get<int>()),
          local_window(config.at("dfly_context_local_window").is_null() ? 0 : config.at("dfly_context_local_window").get<int>()),
          target_layers(config.at("dfly_target_kv_layers").get<std::vector<int>>()),
          index_layers(config.at("dfly_target_index_layer").get<std::vector<int>>()) {
        const int count = config.at("dfly_num_hidden_layers").get<int>();
        if (config.at("dfly_mtp_fusion") != "slot0" || !config.at("dfly_enable_hidden_correction").get<bool>() ||
            config.at("dfly_target_index_topk") != g.index_budget ||
            max_depth < 1 || max_depth > kMlxMtpEngineMaximumDraftDepth ||
            mask_id < 0 || mask_id >= g.vocab || local_window < 0 || local_window > g.context ||
            count < 1 || count > 64 || target_layers.size() != std::size_t(count) || index_layers.size() != std::size_t(count))
            throw std::invalid_argument("unsupported DFly2 configuration");
        layers.reserve(count);
        for (int i = 0; i < count; ++i) {
            const int target = target_layers[i], index = index_layers[i];
            if (target < 0 || target >= g.layers || g.sliding[target] ||
                index < 0 || index >= g.layers || g.sliding[index])
                throw std::invalid_argument("DFly2 reuse layer must be a sparse Hy4 target layer");
            const auto t = DraftWeights(tensors, std::to_string(i) + ".");
            const auto head_count = t.tensor("self_attn.learnable_sink_param").size();
            if (head_count == 0 || head_count > 256) throw std::invalid_argument("invalid DFly2 head count");
            layers.emplace_back(t, g, static_cast<int>(head_count));
        }
    }

    void propose(const Geometry& g, const MlxHy4Cache& cache, const MlxEmbedding& embedding,
        const MlxLinear& output, std::int32_t pending, int depth, const MlxMtpTokenSelector& select) const {
        std::vector<std::int32_t> proposal(1 + depth, mask_id);
        proposal.front() = pending;
        const auto ids = array(proposal.begin(), {1, 1 + depth}, int32);
        auto previous = axis_slice(ids, 1, 0, 1);
        const auto embeds = embedding(ids, bfloat16);
        const auto anchor = axis_slice(embeds, 1, 0, 1);
        auto corrected_anchor = anchor + fuse_gate *
            fuse(concatenate({embedding_norm(anchor), target_norm(*cache.last_hidden)}, -1));
        auto hidden = concatenate({corrected_anchor, axis_slice(embeds, 1, 1, depth + 1)}, 1);
        const int position = cache.position;
        const auto phases = rotary_phases(arange(position, position + 1 + depth, 1, int32), g.rotary, g.rotary_base);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto [history, valid] = selected_history(g, cache, i);
            hidden = layers[i](g, hidden, phases, history, valid);
        }
        hidden = output_norm(hidden);
        for (int i = 1; i <= depth; ++i) {
            const auto row = axis_slice(hidden, 1, i, i + 1);
            const auto corrected = correct(row, previous, embedding);
            previous = reshape(select(mlx_last_token_logits(output(corrected), g.vocab)), {1, 1});
        }
    }

    std::vector<array> propose_batch(const Geometry& g, std::span<MlxHy4Cache* const> caches,
        std::span<const MlxMtpTargetRequest> steps, const MlxEmbedding& embedding, const MlxLinear& output) const {
        const int batch = static_cast<int>(caches.size());
        if (batch == 1) {
            std::vector<array> proposed;
            propose(g, *caches.front(), embedding, output, steps.front().pending_token, steps.front().draft_count,
                [&](const array& logits) {
                    auto token = reshape(sample_greedy(logits), {1});
                    proposed.push_back(token);
                    return token;
                });
            return {concatenate(proposed, 0)};
        }
        std::vector<int> offsets{0}, ids, positions;
        std::vector<array> target_hidden;
        int max_drafts = 0;
        for (int lane = 0; lane < batch; ++lane) {
            const auto& step = steps[lane];
            ids.push_back(step.pending_token);
            ids.insert(ids.end(), step.draft_count, mask_id);
            for (int token = 0; token <= step.draft_count; ++token)
                positions.push_back(caches[lane]->position + token);
            offsets.push_back(static_cast<int>(ids.size()));
            target_hidden.push_back(*caches[lane]->last_hidden);
            max_drafts = std::max(max_drafts, step.draft_count);
        }
        const int total = offsets.back();
        const auto embedded = embedding(array(ids.begin(), {1, total}, int32), bfloat16);
        const auto anchors = take(embedded, array(offsets.begin(), {batch}, int32), 1);
        const auto fused = anchors + fuse_gate *
            fuse(concatenate({embedding_norm(anchors), target_norm(concatenate(target_hidden, 1))}, -1));
        std::vector<array> inputs;
        for (int lane = 0; lane < batch; ++lane) {
            inputs.push_back(axis_slice(fused, 1, lane, lane + 1));
            inputs.push_back(axis_slice(embedded, 1, offsets[lane] + 1, offsets[lane + 1]));
        }
        auto hidden = concatenate(inputs, 1);
        const auto phases = rotary_phases(array(positions.begin(), {total}, int32), g.rotary, g.rotary_base);
        const float scale = 1.0f / std::sqrt(float(g.q_dim));
        for (std::size_t index = 0; index < layers.size(); ++index) {
            const auto& layer = layers[index];
            const auto rows = project_mla(layer, g, layer.attention_norm(hidden), phases, true);
            std::vector<MlaWindow> windows;
            for (int lane = 0; lane < batch; ++lane) {
                const auto [history, valid] = selected_history(g, *caches[lane], index);
                const auto current = axis_slice(rows.latent, 1, offsets[lane], offsets[lane + 1]);
                const int count = current.shape(1);
                const auto slots = arange(count, int32);
                const auto causal = expand_dims(slots <= expand_dims(slots, 1), 0);
                const auto past = broadcast_to(expand_dims(valid, 1), {1, count, history.shape(1)});
                windows.push_back({concatenate({history, current}, 1), concatenate({past, causal}, -1)});
            }
            const auto attention = batch_dense_mla_windows(rows.query, windows, layer.sinks, scale, g.kv_rank);
            auto residual = hidden + expand_mla(layer, g, attention, rows.gate, hidden.dtype());
            hidden = residual + layer.ffn(layer.ffn_norm(residual));
        }
        hidden = output_norm(hidden);
        std::vector<std::vector<array>> proposed(batch);
        for (int slot = 1; slot <= max_drafts; ++slot) {
            std::vector<int> lanes, rows;
            std::vector<array> prior;
            for (int lane = 0; lane < batch; ++lane) if (slot <= steps[lane].draft_count) {
                lanes.push_back(lane);
                rows.push_back(offsets[lane] + slot);
                prior.push_back(slot == 1 ? array({steps[lane].pending_token}, {1}, int32) : proposed[lane].back());
            }
            const int count = static_cast<int>(lanes.size());
            const auto row = take(hidden, array(rows.begin(), {count}, int32), 1);
            const auto previous = reshape(concatenate(prior, 0), {1, count});
            const auto selected = sample_greedy(reshape(output(correct(row, previous, embedding)), {count, g.vocab}));
            for (int i = 0; i < count; ++i)
                proposed[lanes[i]].push_back(axis_slice(selected, 0, i, i + 1));
        }
        std::vector<array> result;
        for (const auto& lane : proposed) result.push_back(concatenate(lane, 0));
        return result;
    }

private:
    array correct(const array& hidden, const array& previous, const MlxEmbedding& embedding) const {
        const auto prior = correction_embedding_norm(embedding(previous, bfloat16));
        return hidden + correction(concatenate({correction_hidden_norm(hidden), prior}, -1));
    }

    std::pair<array, array> selected_history(const Geometry& g, const MlxHy4Cache& cache, std::size_t layer) const {
        const auto& cached = cache.layers[target_layers[layer]];
        auto selected = *cache.layers[index_layers[layer]].last_indices;
        if (local_window) {
            const int begin = std::max(0, cache.position - local_window);
            selected = where(selected < begin, selected, array(-1, int32));
            const auto local = reshape(arange(begin, cache.position, 1, int32), {1, 1, cache.position - begin});
            selected = concatenate({selected, local}, -1);
        }
        const int keys = selected.shape(-1);
        auto values = cached.offloaded_latent
            ? reshape(cached.offloaded_latent->gather(selected), {1, keys, g.indexed_kv.width()})
            : take(cached.latent->view(), reshape(maximum(selected, array(0, int32)), {-1}), 1);
        return {g.indexed_kv.unpack(values), reshape(selected >= array(0, int32), {1, keys})};
    }
};

array dense_weight(const MfqContainer& model, const std::string& key) {
    return astype(load_dense_array(model.record(key).dtype, model.read(key)), float32);
}

array snapshot_array(const array& source, bool detached) {
    if (!detached) return source;
    auto copy = source + zeros_like(source);
    copy.eval();
    return copy;
}

MlxHy4RowSnapshot snapshot_rows(const std::optional<MlxSequenceCache>& resident,
    const std::optional<MlxQsaIndexOffload>& offloaded, bool detached, std::size_t& bytes) {
    if (offloaded && offloaded->position()) {
        auto snapshot = offloaded->snapshot();
        bytes += snapshot.rows.tail.size() + snapshot.rows.blocks.size() * sizeof(QsaKvStore::BlockPtr);
        return {std::nullopt, std::move(snapshot)};
    }
    if (resident && resident->position()) {
        auto rows = snapshot_array(resident->view(), detached);
        bytes += rows.nbytes();
        return {std::move(rows), std::nullopt};
    }
    return {};
}

void restore_rows(const MlxHy4RowSnapshot& state, int count, int maximum, int width, Dtype dtype,
    int priority, const std::shared_ptr<QsaKvStore>& store, std::optional<MlxSequenceCache>& resident,
    std::optional<MlxQsaIndexOffload>& offloaded) {
    if ((state.resident && state.offloaded) || (count && !state.resident && !state.offloaded) ||
        (state.resident && (state.resident->shape() != Shape{1, count, width} || state.resident->dtype() != dtype)) ||
        (state.offloaded && (state.offloaded->rows.position != static_cast<std::size_t>(count) ||
            state.offloaded->width != width || state.offloaded->dtype != dtype)))
        throw std::invalid_argument("Hy4 prefix sequence geometry mismatch");
    if (store) offloaded.emplace(store, width, dtype, priority);
    else resident.emplace(maximum, width, dtype);
    if (!count) return;
    const auto append = [&](const array& rows) {
        if (offloaded) offloaded->append(rows);
        else resident->append(rows);
    };
    if (state.resident) {
        append(*state.resident);
    } else if (offloaded && store == state.offloaded->rows.store) {
        offloaded->restore(*state.offloaded);
    } else {
        const auto& source = state.offloaded->rows;
        for (int start = 0; start < count;) {
            const int size = std::min(count - start, 256);
            auto rows = array(allocator::malloc(size * source.row_bytes), {1, size, width}, dtype);
            source.read_rows(start, size, rows.data<std::uint8_t>());
            append(rows);
            start += size;
        }
    }
}
}  // namespace

struct MlxHy4Model::Weights {
    Geometry g;
    MlxEmbedding token_embedding;
    MlxLinear output;
    MlxRmsNorm output_norm;
    array head_weight, head_bias, head_scale;
    std::optional<EngramTable> engram;
    std::vector<Layer> layers;
    std::shared_ptr<QsaKvStore> offload;
    std::optional<DflyDraft> draft;

    Weights(const MfqContainer& model, const GgufModelSource& source, const json& meta,
        Geometry config, const std::string& draft_path)
        : g(std::move(config)), token_embedding(MlxEmbedding::load(model, "model.token_embedding.weight")),
          output(MlxLinear::load(model, "model.output.weight")),
          output_norm(dense_weight(model, "model.output_norm.weight"), g.eps),
          head_weight(contiguous(transpose(dense_weight(model, "model.mhc.pre.function")))),
          head_bias(dense_weight(model, "model.mhc.pre.base")), head_scale(dense_weight(model, "model.mhc.pre.scale")),
          offload(mlx_configure_qsa_kv_offload(g.context, std::count(g.sliding.begin(), g.sliding.begin() + g.layers, false),
              1, g.kv_rank + g.rotary, g.index_width, g.group_size)) {
        if (source.find_tensor(engram_weight)) engram.emplace(source, meta);
        layers.reserve(g.layers);
        for (int layer = 0; layer < g.layers; ++layer) {
            const bool has_engram = engram && std::find(engram->layers.begin(), engram->layers.end(), layer) != engram->layers.end();
            layers.emplace_back(MlxGgufLayerTensors(model, source, layer), g, layer, has_engram);
        }
        if (!offload && !g.indexed_kv.quantization.enabled() &&
            g.kv_rank == 512 && g.rotary == 64 && g.index_budget == 2048 && g.context > 2 * g.index_budget &&
            std::any_of(layers.begin(), layers.end(), [](const Layer& layer) { return layer.index && layer.heads == 32; }))
            mlx_prepare_indexed_mla_prefill();
        const int history_groups = std::min(8192, (g.context - 1) / 4);
        if (auto* preparation = MlxKernelPreparation::current();
            preparation && default_device() == Device::gpu && !offload &&
            g.group_size == 4 && g.index_budget == 2048 && history_groups >= 512) {
            preparation->collect([&] {
                for (int width = 512; ; width *= 2) {
                    const int live = std::min(width, history_groups);
                    preparation->add(mlx_sparse_indexer_token_ids(zeros({1, 1, live}, float32), live * 4));
                    if (width >= history_groups) break;
                }
            });
        }
        if (!draft_path.empty()) {
            auto path = std::filesystem::path(draft_path);
            std::ifstream config_file(path.parent_path() / "config.json");
            if (!config_file) throw std::runtime_error("DFly2 requires config.json beside its BF16 safetensors shard");
            const auto config = json::parse(config_file);
            const auto tensors = load_safetensors(draft_path).first;
            std::vector<array> loaded;
            loaded.reserve(tensors.size());
            for (const auto& [name, tensor] : tensors) loaded.push_back(tensor);
            // Deferred safetensors reads use the loader thread's CPU stream.
            // Finish them before the serving worker can build a proposal.
            eval(loaded);
            draft.emplace(tensors, config, g);
            for (const auto& tensor : loaded) MlxWeightResidency::track(tensor);
        }
    }
};

struct MlxHy4Verification {
    int position = 0;
    std::vector<VerifiedLayerRows> layers;
    std::vector<std::int32_t> engram_tokens;
    std::optional<array> hidden;
};

MlxHy4Model MlxHy4Model::load(const MfqContainer& model, int context, const std::string& draft_path) {
    const auto* source = dynamic_cast<const GgufModelSource*>(model.native_source());
    if (!source || source->architecture() != "hy_v4_flash")
        throw std::invalid_argument("Hy4 Flash runtime requires a hy_v4_flash GGUF");
    const auto meta = json::parse(source->metadata().at("gguf_metadata"));
    return MlxHy4Model(std::make_shared<Weights>(model, *source, meta, geometry(meta, context), draft_path));
}

array MlxHy4Model::forward(const array& tokens, MlxHy4Cache& cache, bool last_logit_only) const {
    return forward_impl(tokens, cache, last_logit_only, nullptr);
}

array MlxHy4Model::forward_impl(const array& tokens, MlxHy4Cache& cache, bool last_logit_only,
    Verification* verification) const {
    const auto& w = *weights_;
    const auto& g = w.g;
    if (tokens.ndim() != 2 || tokens.shape(0) != 1 || tokens.shape(1) <= 0 ||
        cache.position < 0 || tokens.shape(1) > g.context - cache.position ||
        (!cache.layers.empty() && cache.layers.size() != std::size_t(g.layers)))
        throw std::invalid_argument("invalid Hy4 tokens or layer cache");
    const int count = tokens.shape(1), offset = cache.position;
    if (cache.layers.empty()) cache.layers.resize(g.layers);
    const auto positions = arange(offset, offset + count, 1, int32);
    const auto indexed_phases = rotary_phases(positions, g.rotary, g.rotary_base);
    const auto window_phases = rotary_phases(positions, g.rotary, g.window_rotary_base);
    // Pool groups use the same absolute positions in every indexed layer.
    const int first_group = offset - offset % g.group_size;
    const int complete = (offset + count - first_group) / g.group_size;
    std::optional<RotaryPhases> pool_phases;
    if (complete) {
        const auto groups = arange(first_group, first_group + complete * g.group_size, g.group_size, int32);
        pool_phases = rotary_phases(groups, g.rotary, g.rotary_base);
    }
    auto hidden = w.token_embedding(tokens, bfloat16);
    auto streams = broadcast_to(expand_dims(hidden, 2), {1, count, g.streams, g.hidden});
    if (verification) {
        verification->position = offset;
        verification->layers.resize(g.layers);
    }

    std::vector<std::int32_t> history = cache.engram_tail;
    const auto first = history.size();
    if (w.engram) {
        auto ids = contiguous(astype(tokens, int32));
        ids.eval();
        const auto* data = ids.data<std::int32_t>();
        history.insert(history.end(), data, data + count);
        if (verification) verification->engram_tokens = history;
    }
    std::optional<PendingResidual> deferred;
    const bool merge_layers = count == 1 && g.streams == 4 && g.hidden == 3072;
    for (int layer = 0; layer < g.layers; ++layer) {
        std::optional<array> rows;
        if (w.layers[layer].engram) rows = w.engram->gather(layer, history, first);
        const auto& phases = g.sliding[layer] ? window_phases : indexed_phases;
        streams = run_layer(w.layers[layer], g, cache.layers[layer], std::move(streams), rows,
            phases, pool_phases, offset, w.offload, verification ? &verification->layers[layer] : nullptr,
            merge_layers ? &deferred : nullptr);
    }
    if (deferred) streams = MlxHyperConnection::expand(streams, deferred->normalized(), deferred->mixer);
    if (w.engram) {
        const auto keep = std::min(history.size(), std::size_t(w.engram->grams - 1));
        cache.engram_tail.assign(history.end() - keep, history.end());
    }
    if (last_logit_only) streams = axis_slice(streams, 1, count - 1, count);
    const auto float_streams = astype(streams, float32);
    const auto flattened = reshape(float_streams, {1, streams.shape(1), g.streams * g.hidden});
    const auto gates = sigmoid(matmul(rms(flattened, g.eps), w.head_weight) * w.head_scale + w.head_bias) + g.gate_eps;
    hidden = w.output_norm(sum(float_streams * expand_dims(gates, -1), 2), bfloat16);
    if (verification) verification->hidden = hidden;
    cache.last_hidden = axis_slice(hidden, 1, hidden.shape(1) - 1, hidden.shape(1));
    cache.position += count;
    return astype(w.output(hidden), float32);
}

array MlxHy4Model::decode_batch(std::span<Cache* const> requests, const array& tokens) const {
    const int batch = static_cast<int>(requests.size());
    if (!batch || tokens.shape() != Shape{batch, 1}) throw std::invalid_argument("invalid Hy4 decode batch");
    std::vector<BatchInput> inputs;
    inputs.reserve(batch);
    for (int row = 0; row < batch; ++row)
        inputs.push_back({requests[row], axis_slice(tokens, 0, row, row + 1)});
    auto outputs = forward_batch(inputs);
    return outputs.size() == 1 ? std::move(outputs.front()) : concatenate(outputs, 0);
}

std::vector<array> MlxHy4Model::forward_batch(std::span<const BatchInput> inputs) const {
    const auto& w = *weights_;
    const auto& g = w.g;
    const int batch = static_cast<int>(inputs.size());
    if (!batch) throw std::invalid_argument("empty Hy4 decode batch");
    std::vector<Cache*> requests;
    std::vector<array> tokens;
    requests.reserve(batch);
    tokens.reserve(batch);
    int total = 0;
    for (const auto& input : inputs) {
        const auto* request = input.cache;
        if (!request || input.tokens.ndim() != 2 || input.tokens.shape(0) != 1 || input.tokens.shape(1) <= 0 ||
            request->position <= 0 || input.tokens.shape(1) > g.context - request->position ||
            request->layers.size() != static_cast<std::size_t>(g.layers) ||
            std::find(requests.begin(), requests.end(), request) != requests.end())
            throw std::invalid_argument("invalid Hy4 decode request state");
        requests.push_back(input.cache);
        tokens.push_back(input.tokens);
        total += input.tokens.shape(1);
    }
    if (batch == 1) {
        const auto& input = inputs.front();
        return {forward_impl(input.tokens, *input.cache, false, input.verification)};
    }
    struct Sequence {
        std::optional<RotaryPhases> pooled;
        std::vector<std::int32_t> history;
        std::size_t first;
        int begin, end;
    };
    auto ids = contiguous(astype(concatenate(tokens, 1), int32));
    if (w.engram) ids.eval();
    auto hidden = w.token_embedding(ids, bfloat16);
    auto streams = broadcast_to(expand_dims(hidden, 2), {1, total, g.streams, g.hidden});
    std::vector<Sequence> sequences;
    std::vector<std::int32_t> positions;
    sequences.reserve(batch);
    positions.reserve(total);
    bool score_index = false, all_scored = true;
    bool singleton_batch = total == batch;
    int max_window = 0, max_end = 0;
    for (const auto& input : inputs) {
        const auto& cache = *input.cache;
        const int count = input.tokens.shape(1), end = cache.position + count;
        const int begin = static_cast<int>(positions.size());
        for (int position = cache.position; position < end; ++position) positions.push_back(position);
        score_index |= end > g.index_budget;
        all_scored &= end > g.index_budget;
        singleton_batch &= !input.verification;
        max_window = std::max(max_window, count);
        max_end = std::max(max_end, end);
        const int first_group = cache.position - cache.position % g.group_size;
        const int complete = (end - first_group) / g.group_size;
        std::optional<RotaryPhases> pooled;
        if (complete)
            pooled = rotary_phases(arange(first_group, first_group + complete * g.group_size,
                g.group_size, int32), g.rotary, g.rotary_base);
        auto history = cache.engram_tail;
        const auto first = history.size();
        if (w.engram) {
            const auto* row = ids.data<std::int32_t>() + begin;
            history.insert(history.end(), row, row + count);
        }
        if (input.verification) {
            input.verification->position = cache.position;
            input.verification->layers.resize(g.layers);
            if (w.engram) input.verification->engram_tokens = history;
        }
        sequences.push_back({std::move(pooled), std::move(history), first, begin, begin + count});
    }

    const auto positions_array = array(positions.begin(), {total}, int32);
    const auto indexed_phases = rotary_phases(positions_array, g.rotary, g.rotary_base);
    const auto sliding_phases = rotary_phases(positions_array, g.rotary, g.window_rotary_base);
    const bool verify_windows = !singleton_batch && max_window <= kMlxMtpEngineMaximumDraftDepth + 1;
    std::vector<array> verify_sliding_masks;
    if (verify_windows) {
        verify_sliding_masks.reserve(batch);
        for (const auto& input : inputs) {
            const int history = std::min(input.cache->position, g.window - 1);
            const int count = input.tokens.shape(1);
            const auto queries = reshape(arange(history, history + count, 1, int32), {1, count, 1});
            const auto keys = arange(history + count, int32);
            verify_sliding_masks.push_back(logical_and(keys <= queries, keys > queries - g.window));
        }
    }
    const float scale = 1.0f / std::sqrt(float(g.q_dim));
    std::optional<PendingResidual> deferred;
    for (int index = 0; index < g.layers; ++index) {
        const auto& layer = w.layers[index];
        std::optional<array> engram;
        if (layer.engram) {
            std::vector<array> rows;
            rows.reserve(batch);
            for (const auto& sequence : sequences)
                rows.push_back(w.engram->gather(index, sequence.history, sequence.first));
            engram = concatenate(rows, 1);
        }
        const auto mix = prepare_attention_input(layer, g, streams, engram,
            deferred ? &*deferred : nullptr, true);
        const auto& in = mix.branch;
        const auto& phases = layer.sliding ? sliding_phases : indexed_phases;
        const auto rows = project_mla(layer, g, in, phases, true);
        const auto pairs = layer.index ? std::optional(project_index_pairs(*layer.index, in)) : std::nullopt;
        const auto scoring = layer.index && score_index
            ? std::optional(project_index_scores(*layer.index, g, in, rows.query_low, phases)) : std::nullopt;
        const auto attention = [&] {
            if (singleton_batch && layer.sliding) {
                std::vector<array> visible;
                visible.reserve(batch);
                for (int row = 0; row < batch; ++row) {
                    auto& request = *requests[row];
                    visible.push_back(append_sliding_decode(request.layers[index], g,
                        axis_slice(rows.latent, 1, row, row + 1), request.position));
                }
                return batch_dense_mla_decode(rows.query, std::move(visible), layer.sinks, scale, g.kv_rank);
            }
            if (singleton_batch && layer.index && !w.offload && !g.indexed_kv.quantization.enabled() &&
                (!score_index || all_scored)) {
                const bool packed_gather = batch <= index_gather_max_batch && scoring && g.index_budget == 2048 &&
                    rows.latent.shape(-1) == 576 && rows.latent.dtype() == float16 && default_device() == Device::gpu;
                std::vector<array> visible, masks, indices;
                visible.reserve(batch);
                (packed_gather ? indices : masks).reserve(batch);
                for (int row = 0; row < batch; ++row) {
                    auto& request = *requests[row];
                    auto& cache = request.layers[index];
                    append_indexed_cache(cache, *layer.index, g, axis_slice(rows.latent, 1, row, row + 1),
                        axis_slice(*pairs, 1, row, row + 1), sequences[row].pooled, nullptr, nullptr);
                    auto view = cache.latent->view();
                    if (scoring) {
                        auto selected = choose_resident_tokens(cache.groups->view(), g,
                            axis_slice(scoring->query, 1, row, row + 1),
                            axis_slice(scoring->weights, 1, row, row + 1), request.position);
                        if (packed_gather) {
                            visible.push_back(std::move(view));
                            indices.push_back(selected);
                        } else {
                            masks.push_back(where(selected >= 0, array(0.0f, float16),
                                array(-std::numeric_limits<float>::infinity(), float16)));
                            visible.push_back(mlx_gather_selected_mla_rows(view, selected));
                        }
                        cache.last_indices = std::move(selected);
                    } else {
                        const int end = request.position + 1;
                        cache.last_indices = reshape(arange(0, end, 1, int32), {1, 1, end});
                        visible.push_back(g.indexed_kv.unpack(view));
                    }
                }
                if (!scoring)
                    return batch_dense_mla_decode(rows.query, std::move(visible), layer.sinks, scale, g.kv_rank);
                const auto [data, mask] = packed_gather ? gather_index_batch(visible, indices) :
                    std::pair{concatenate(visible, 0), concatenate(masks, 0)};
                auto result = mlx_gathered_mla_attention(batch_decode_queries(rows.query), data,
                    mask, layer.sinks, scale, g.kv_rank);
                return reshape(result, {1, batch, result.shape(2), result.shape(3)});
            }
            if (verify_windows && layer.sliding) {
                std::vector<MlaWindow> windows;
                std::vector<array> updates;
                for (int row = 0; row < batch; ++row) {
                    const auto& sequence = sequences[row];
                    auto& request = *requests[row];
                    auto& cache = request.layers[index];
                    auto next = axis_slice(rows.latent, 1, sequence.begin, sequence.end);
                    if (!cache.window) cache.window = zeros({1, g.window, next.shape(-1)}, next.dtype());
                    if (inputs[row].verification)
                        inputs[row].verification->layers[index].new_window = next;
                    const int history = std::min(request.position, g.window - 1);
                    auto visible = sliding_attention_rows(*cache.window, next, request.position, history);
                    windows.push_back({std::move(visible), verify_sliding_masks[row]});
                    updates.push_back(std::move(next));
                }
                auto result = batch_dense_mla_windows(rows.query, windows, layer.sinks, scale, g.kv_rank);
                for (int row = 0; row < batch; ++row) {
                    if (inputs[row].verification) continue;
                    auto& request = *requests[row];
                    auto& cache = request.layers[index];
                    cache.window = mlx_circular_cache_append(*cache.window, updates[row], request.position,
                        axis_slice(result, 1, sequences[row].begin, sequences[row].end));
                }
                return result;
            }
            if (verify_windows && layer.index && !w.offload && !g.indexed_kv.quantization.enabled()) {
                std::vector<array> visible, selections;
                for (int row = 0; row < batch; ++row) {
                    const auto& sequence = sequences[row];
                    auto& request = *requests[row];
                    auto& cache = request.layers[index];
                    auto* verified = inputs[row].verification ? &inputs[row].verification->layers[index] : nullptr;
                    const auto latent = axis_slice(rows.latent, 1, sequence.begin, sequence.end);
                    append_indexed_cache(cache, *layer.index, g, latent,
                        axis_slice(*pairs, 1, sequence.begin, sequence.end), sequence.pooled, nullptr, verified);
                    const int count = latent.shape(1), end = request.position + count;
                    if (end <= g.index_budget) {
                        cache.last_indices = reshape(arange(end, int32), {1, 1, end});
                        const auto queries = reshape(arange(request.position, end, 1, int32), {1, count, 1});
                        selections.push_back(where(*cache.last_indices <= queries, *cache.last_indices, array(-1)));
                    } else {
                        auto ids = choose_resident_tokens(astype(cache.groups->view(), float32), g,
                            axis_slice(scoring->query, 1, sequence.begin, sequence.end),
                            axis_slice(scoring->weights, 1, sequence.begin, sequence.end), request.position);
                        if (verified) verified->selections = ids;
                        cache.last_indices = axis_slice(ids, 1, count - 1, count);
                        selections.push_back(std::move(ids));
                    }
                    visible.push_back(cache.latent->view());
                }
                return batch_resident_indexed_mla(rows.query, visible, selections, layer.sinks,
                    scale, g.kv_rank, max_end <= 2 * g.index_budget);
            }
            std::vector<array> attended;
            attended.reserve(batch);
            // The projections use packed rows. Each cache owns its own key positions and rollback window.
            for (int row = 0; row < batch; ++row) {
                const auto& sequence = sequences[row];
                auto& request = *requests[row];
                auto& cache = request.layers[index];
                auto* verified = inputs[row].verification ? &inputs[row].verification->layers[index] : nullptr;
                const auto query = axis_slice(rows.query, 2, sequence.begin, sequence.end);
                const auto latent = axis_slice(rows.latent, 1, sequence.begin, sequence.end);
                if (layer.sliding) {
                    attended.push_back(sliding_attention(cache, g, query, latent, layer.sinks,
                        request.position, scale, verified));
                } else {
                    std::optional<IndexScores> lane_scores;
                    if (scoring && request.position + latent.shape(1) > g.index_budget)
                        lane_scores.emplace(axis_slice(scoring->query, 1, sequence.begin, sequence.end),
                            axis_slice(scoring->weights, 1, sequence.begin, sequence.end));
                    attended.push_back(run_indexed_attention(cache, *layer.index, g, query, latent,
                        axis_slice(*pairs, 1, sequence.begin, sequence.end), lane_scores,
                        sequence.pooled, layer.sinks, request.position, scale, w.offload, verified));
                }
            }
            return concatenate(attended, 1);
        }();
        const auto attn_out = expand_mla(layer, g, attention, rows.gate, in.dtype());
        auto [updated, ffn_mix] = layer.ffn_hc.expand_and_collapse(streams, attn_out, mix, &layer.ffn_norm, true);
        streams = std::move(updated);
        auto output = project_ffn(layer, g, ffn_mix.branch);
        deferred.emplace(std::move(output), std::move(ffn_mix));
    }
    if (deferred) streams = MlxHyperConnection::expand(streams, deferred->branch, deferred->mixer);
    streams = astype(streams, float32);
    const auto flattened = reshape(streams, {1, total, g.streams * g.hidden});
    const auto gates = sigmoid(matmul(rms(flattened, g.eps), w.head_weight) * w.head_scale + w.head_bias) + g.gate_eps;
    hidden = w.output_norm(sum(streams * expand_dims(gates, -1), 2), bfloat16);
    const auto logits = astype(w.output(hidden), float32);
    std::vector<array> outputs;
    outputs.reserve(batch);
    for (int row = 0; row < batch; ++row) {
        auto& cache = *requests[row];
        const auto& sequence = sequences[row];
        if (w.engram) {
            const auto keep = std::min(sequence.history.size(), std::size_t(w.engram->grams - 1));
            cache.engram_tail.assign(sequence.history.end() - keep, sequence.history.end());
        }
        if (inputs[row].verification)
            inputs[row].verification->hidden = axis_slice(hidden, 1, sequence.begin, sequence.end);
        cache.last_hidden = axis_slice(hidden, 1, sequence.end - 1, sequence.end);
        cache.position += sequence.end - sequence.begin;
        outputs.push_back(axis_slice(logits, 1, sequence.begin, sequence.end));
    }
    return outputs;
}

int MlxHy4Model::context() const noexcept { return weights_->g.context; }
int MlxHy4Model::vocabulary() const noexcept { return weights_->g.vocab; }
bool MlxHy4Model::supports_mtp() const noexcept { return weights_->draft.has_value(); }

std::optional<QsaKvStoreStats> MlxHy4Model::qsa_kv_offload_stats() const {
    if (weights_->offload) return weights_->offload->stats();
    return std::nullopt;
}

std::size_t MlxHy4Model::kv_cache_bytes(const Cache& cache) const noexcept {
    std::size_t bytes = cache.engram_tail.size() * sizeof(std::int32_t);
    if (cache.last_hidden) bytes += cache.last_hidden->nbytes();
    for (const auto& layer : cache.layers) {
        if (layer.window) bytes += layer.window->nbytes();
        if (layer.index_tail) bytes += layer.index_tail->nbytes();
        if (layer.last_indices) bytes += layer.last_indices->nbytes();
        if (layer.latent) bytes += layer.latent->storage_bytes();
        if (layer.groups) bytes += layer.groups->storage_bytes();
    }
    return bytes;
}

int MlxHy4Model::generate_speculative(Cache& cache, MlxMtpEngineRequest request,
    const MlxPrefixCacheHooks& prefix, MlxMtpGenerationStats& stats, std::optional<array>& last_logits) const {
    const auto& w = *weights_;
    const auto& g = w.g;
    const int input_end = cache.position;
    std::optional<std::int32_t> last_emitted;
    request.callback = [callback = std::move(request.callback), &last_emitted](std::int64_t token) {
        last_emitted = static_cast<std::int32_t>(token);
        return !callback || callback(token);
    };
    auto callbacks = batch_mtp_callbacks(cache, last_logits);
    callbacks.verify_target = [&](std::int32_t pending, const array& drafts, int count) {
        const std::array states{&cache};
        const std::array steps{MlxMtpTargetRequest{pending, drafts, count}};
        const std::array logits{&last_logits};
        return std::move(verify_batch(states, steps, logits).front());
    };
    if (prefix) {
        callbacks.draft_limit = [&](int position, int depth) { return prefix.draft_limit(position, depth); };
        callbacks.committed_target = [&](const MlxMtpDraftContext&) { prefix.checkpoint(cache.position); };
    }
    const int generated = run_mlx_mtp_generation(std::move(request), callbacks, stats);
    if (last_emitted && cache.position < input_end + generated && prefix.wants(cache.position + 1, true)) {
        last_logits = mlx_last_token_logits(forward(array({*last_emitted}, {1, 1}, int32), cache, true), g.vocab);
        last_logits->eval();
    }
    prefix.checkpoint(cache.position, true);
    return generated;
}

void MlxHy4Model::resolve_verification(Cache& cache, const Verification& trace, const array& logits,
    int accepted, int proposed, std::optional<array>& last_logits) const {
    const auto& w = *weights_;
    const auto& g = w.g;
    const int keep = accepted + 1, position = trace.position + keep;
    const bool rejected = accepted != proposed;
    for (int i = 0; i < g.layers; ++i) {
        auto& layer = cache.layers[i];
        const auto& saved = trace.layers[i];
        if (g.sliding[i]) {
            // Commit after target verification has finished reading the original window.
            const auto rows = axis_slice(*saved.new_window, 1, 0, keep);
            layer.window = mlx_circular_cache_append(*layer.window, rows, trace.position, logits);
            continue;
        }
        if (!rejected) continue;
        if (layer.offloaded_latent) {
            layer.offloaded_latent->trim(position);
            layer.offloaded_groups->trim(position / g.group_size);
        } else {
            layer.latent->trim(proposed - accepted);
            layer.groups->trim(layer.groups->position() - position / g.group_size);
        }
        const int owed = position % g.group_size;
        if (owed) {
            const int end = trace.position % g.group_size + keep;
            layer.index_tail = axis_slice(*saved.index_pairs, 1, end - owed, end);
        } else {
            layer.index_tail.reset();
        }
        layer.last_indices = saved.selections
            ? axis_slice(*saved.selections, 1, accepted, keep)
            : reshape(arange(0, position, 1, int32), {1, 1, position});
    }
    if (rejected) {
        if (w.engram) {
            const auto end = trace.engram_tokens.end() - (proposed - accepted);
            const int length = std::min(position, w.engram->grams - 1);
            cache.engram_tail.assign(end - length, end);
        }
        cache.last_hidden = axis_slice(*trace.hidden, 1, accepted, keep);
        cache.position = position;
    }
    last_logits = axis_slice(logits, 0, accepted, keep);
}

MlxMtpEngineCallbacks MlxHy4Model::batch_mtp_callbacks(Cache& cache,
    std::optional<array>& last_logits) const {
    const auto& w = *weights_;
    if (!w.draft) throw std::logic_error("Hy4 does not have a loaded draft");
    MlxMtpEngineCallbacks callbacks;
    callbacks.predictor = MlxMtpPredictorDescriptor::recurrent(w.draft->max_depth);
    callbacks.target_cache_position = [&cache] { return cache.position; };
    callbacks.prepare_draft = [this, &cache](const MlxMtpDraftContext& context, const MlxMtpTokenSelector& select) {
        const auto& w = *weights_;
        if (context.requested_depth)
            w.draft->propose(w.g, cache, w.token_embedding, w.output, context.pending_token,
                context.requested_depth, select);
    };
    callbacks.resolve_target = [this, &cache, &last_logits](int accepted, int proposed) {
        if (!proposed) return;
        resolve_verification(cache, *cache.pending_verification, *cache.pending_logits,
            accepted, proposed, last_logits);
        cache.pending_verification.reset();
        cache.pending_logits.reset();
    };
    return callbacks;
}

std::vector<array> MlxHy4Model::draft_batch(std::span<Cache* const> requests,
    std::span<const MlxMtpTargetRequest> steps) const {
    const auto& w = *weights_;
    if (!w.draft || requests.empty() || requests.size() != steps.size())
        throw std::invalid_argument("invalid Hy4 draft batch");
    for (std::size_t lane = 0; lane < steps.size(); ++lane) {
        const auto& step = steps[lane];
        const auto* cache = requests[lane];
        if (!cache || !cache->last_hidden || cache->layers.size() != static_cast<std::size_t>(w.g.layers) ||
            step.kind != MlxMtpTargetRequest::Kind::draft_window ||
            step.draft_count <= 0 || step.draft_count > w.draft->max_depth ||
            step.pending_token < 0 || step.pending_token >= w.g.vocab ||
            cache->position <= 0 || cache->position > w.g.context - step.draft_count - 1)
            throw std::invalid_argument("invalid Hy4 draft window");
    }
    return w.draft->propose_batch(w.g, requests, steps, w.token_embedding, w.output);
}

std::vector<MlxMtpTargetBatch> MlxHy4Model::verify_batch(std::span<Cache* const> requests,
    std::span<const MlxMtpTargetRequest> steps, std::span<std::optional<array>* const> last_logits) const {
    if (requests.size() != steps.size() || requests.size() != last_logits.size())
        throw std::logic_error("Hy4 target batch has inconsistent lanes");
    std::vector<BatchInput> inputs;
    inputs.reserve(requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        auto& request = *requests[i];
        const auto& work = steps[i];
        if (request.pending_verification || work.kind != MlxMtpTargetRequest::Kind::target)
            throw std::logic_error("Hy4 target batch contains an unresolved step");
        if (work.draft_count) request.pending_verification = std::make_shared<Verification>();
        inputs.push_back({&request, mlx_mtp_verification_ids(work.pending_token, work.draft_tokens, work.draft_count),
            request.pending_verification.get()});
    }
    const auto outputs = forward_batch(inputs);
    std::vector<MlxMtpTargetBatch> results;
    results.reserve(requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        auto& request = *requests[i];
        auto logits = reshape(outputs[i], {steps[i].draft_count + 1, vocabulary()});
        if (request.pending_verification) request.pending_logits = logits;
        else *last_logits[i] = logits;
        results.push_back({std::move(logits), request.pending_verification
            ? *request.pending_verification->hidden : *request.last_hidden});
    }
    return results;
}

void MlxHy4Model::discard_batch_verification(Cache& cache, std::optional<array>& logits) const {
    if (!cache.pending_verification || !cache.pending_logits) return;
    resolve_verification(cache, *cache.pending_verification, *cache.pending_logits, 0,
        cache.pending_logits->shape(0) - 1, logits);
    cache.pending_verification.reset();
    cache.pending_logits.reset();
}

MlxHy4TextSessionState MlxHy4Model::capture(const Cache& cache,
    const std::vector<std::int64_t>& tokens, bool detached) const {
    const auto& g = weights_->g;
    if (cache.position <= 0 || tokens.size() != static_cast<std::size_t>(cache.position) ||
        cache.layers.size() != static_cast<std::size_t>(g.layers) || !cache.last_hidden)
        throw std::invalid_argument("Hy4 prefix tokens do not match the model position");
    TextSessionState state;
    state.tokens = tokens;
    state.window = g.window;
    state.group_size = g.group_size;
    state.bytes = tokens.size() * sizeof(tokens.front());
    const auto copy = [&](const std::optional<array>& value) -> std::optional<array> {
        if (!value) return {};
        state.bytes += value->nbytes();
        return snapshot_array(*value, detached);
    };
    for (const auto& source : cache.layers) {
        MlxHy4LayerSnapshot layer;
        if (source.window) {
            layer.window = copy(mlx_circular_cache_history(*source.window, cache.position));
        } else {
            layer.latent = snapshot_rows(source.latent, source.offloaded_latent, detached, state.bytes);
            layer.groups = snapshot_rows(source.groups, source.offloaded_groups, detached, state.bytes);
            layer.index_tail = copy(source.index_tail);
            layer.last_indices = copy(source.last_indices);
        }
        state.layers.push_back(std::move(layer));
    }
    state.last_hidden = copy(cache.last_hidden);
    return state;
}

MlxHy4Cache MlxHy4Model::restore(const TextSessionState& state) const {
    const auto& w = *weights_;
    const auto& g = w.g;
    if (state.tokens.empty() || state.tokens.size() > static_cast<std::size_t>(g.context) ||
        state.window != g.window || state.group_size != g.group_size ||
        state.layers.size() != static_cast<std::size_t>(g.layers) || !state.last_hidden ||
        state.last_hidden->shape() != Shape{1, 1, g.hidden} || state.last_hidden->dtype() != bfloat16)
        throw std::invalid_argument("Hy4 prefix cache does not match the loaded model");
    Cache restored;
    restored.position = static_cast<int>(state.tokens.size());
    restored.last_hidden = state.last_hidden;
    const int width = g.kv_rank + g.rotary, position = restored.position;
    const int length = std::min(g.window, position), owed = position % g.group_size;
    for (int i = 0; i < g.layers; ++i) {
        const auto& source = state.layers[i];
        MlxHy4AttentionCache layer;
        if (g.sliding[i]) {
            if (!source.window || source.window->shape() != Shape{1, length, width} || source.window->dtype() != float16)
                throw std::invalid_argument("Hy4 prefix sliding-window geometry mismatch");
            layer.window = mlx_circular_cache_append(zeros({1, g.window, width}, float16), *source.window, position - length);
        } else {
            if (source.window || !source.last_indices || source.last_indices->ndim() != 3 ||
                source.last_indices->shape(0) != 1 || source.last_indices->shape(1) != 1 ||
                source.last_indices->shape(2) > g.index_budget || source.last_indices->dtype() != int32 ||
                bool(source.index_tail) != bool(owed) || (source.index_tail &&
                    (source.index_tail->shape() != Shape{1, owed, 2 * g.index_width} || source.index_tail->dtype() != float32)))
                throw std::invalid_argument("Hy4 prefix IndexPool state mismatch");
            restore_rows(source.latent, position, g.context, g.indexed_kv.width(), g.indexed_kv.dtype(), 1, w.offload,
                layer.latent, layer.offloaded_latent);
            restore_rows(source.groups, position / g.group_size, std::max(1, g.context / g.group_size),
                g.index_width, g.index_pool_dtype(), kMlxQsaIndexPriority, w.offload, layer.groups, layer.offloaded_groups);
            layer.index_tail = source.index_tail;
            layer.last_indices = source.last_indices;
        }
        restored.layers.push_back(std::move(layer));
    }
    if (w.engram) {
        const int length = std::min(position, w.engram->grams - 1);
        for (auto token = state.tokens.end() - length; token != state.tokens.end(); ++token) {
            if (*token < 0 || *token >= g.vocab) throw std::invalid_argument("Hy4 prefix Engram token outside vocabulary");
            restored.engram_tail.push_back(static_cast<std::int32_t>(*token));
        }
    }
    return restored;
}

}  // namespace mfq::metal
