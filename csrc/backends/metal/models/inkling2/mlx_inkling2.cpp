#include "mlx_inkling2.h"
#include "mlx_moe_ops.h"

#include "mfq_container.h"
#include "mlx_gguf_layer_tensors.h"
#include "mlx_detached_copy.h"
#include "mlx_staged_attention.h"
#include "mlx_tensor.h"
#include "nlohmann/json.hpp"
#include "tyloinfer/gguf_model_source.h"
#include "tyloinfer/layer_loop.h"

#include <algorithm>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mfq::metal {
namespace {
using namespace mlx::core;
using json = nlohmann::json;

array time_slice(const array& input, int begin, int end) {
    Shape from(input.ndim(), 0);
    auto to = input.shape();
    from[1] = begin;
    to[1] = end;
    return slice(input, from, to);
}

array rms(const array& x, float eps = 1e-6f) {
    return fast::rms_norm(x, std::nullopt, eps);
}

array depth_mix(const array& history, const array& current, const array& query) {
    const auto normalized = rms(astype(current, float32));
    const int hidden = current.shape(-1), depth = history.shape(2) + 1;
    auto rows = reshape(astype(query, float32), {-1, hidden});
    const auto threads = std::int64_t(current.shape(0)) * current.shape(1) * rows.shape(0) * 256;
    if (default_device() == Device::gpu && current.shape(1) <= 8 && depth <= 16 &&
        threads <= std::numeric_limits<int>::max() &&
        (current.dtype() == float16 || current.dtype() == bfloat16)) {
        static const auto kernel = [] {
            CompileOptions options;
            options.math_mode = MathMode::Safe;
            return fast::metal_kernel("tyloinfer_inkling_depth_mix",
                {"history", "current", "queries"}, {"mixed"}, R"MSL(
                const uint row = threadgroup_position_in_grid.x;
                const uint token = row / QUERIES;
                const uint query = row % QUERIES;
                const uint tid = thread_index_in_threadgroup;
                const uint lane = thread_index_in_simdgroup;
                const uint group = simdgroup_index_in_threadgroup;
                threadgroup float partial[DEPTH][8];
                threadgroup float weights[DEPTH];
                const auto state = [&](uint depth) {
                    return depth == DEPTH - 1 ? current + ulong(token) * WIDTH
                        : history + (ulong(token) * (DEPTH - 1) + depth) * WIDTH;
                };
                for (uint depth = 0; depth < DEPTH; ++depth) {
                    float sum = 0.f;
                    const auto v = state(depth);
                    const auto q = queries + query * WIDTH;
                    for (uint i = tid; i < WIDTH; i += 256)
                        sum = metal::fma(q[i], v[i], sum);
                    sum = simd_sum(sum);
                    if (lane == 0) partial[depth][group] = sum;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    float maximum = -INFINITY;
                    for (uint depth = 0; depth < DEPTH; ++depth) {
                        float dot = 0.f;
                        for (uint group = 0; group < 8; ++group) dot += partial[depth][group];
                        weights[depth] = dot / metal::precise::sqrt(float(WIDTH));
                        maximum = max(maximum, weights[depth]);
                    }
                    float norm = 0.f;
                    for (uint depth = 0; depth < DEPTH; ++depth) {
                        weights[depth] = metal::exp(weights[depth] - maximum);
                        norm += weights[depth];
                    }
                    for (uint depth = 0; depth < DEPTH; ++depth) weights[depth] /= norm;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = tid; i < WIDTH; i += 256) {
                    float value = 0.f;
                    for (uint depth = 0; depth < DEPTH; ++depth)
                        value = metal::fma(weights[depth], state(depth)[i], value);
                    mixed[ulong(row) * WIDTH + i] = T(value);
                }
                )MSL", "", true, false, options);
        }();
        const Shape shape{current.shape(0), current.shape(1), rows.shape(0), hidden};
        return kernel({depth > 1 ? history : normalized, normalized, rows}, {shape}, {current.dtype()},
            {static_cast<int>(threads), 1, 1}, {256, 1, 1},
            {{"T", current.dtype()}, {"DEPTH", depth}, {"WIDTH", hidden}, {"QUERIES", rows.shape(0)}},
            std::nullopt, false, {}).front();
    }
    const auto last = expand_dims(normalized, -2);
    const auto values = depth > 1 ? concatenate({history, last}, -2) : last;
    auto score = matmul(rows, transpose(values, {0, 1, 3, 2})) / std::sqrt(float(hidden));
    return astype(matmul(softmax(score, -1, true), values), current.dtype());
}

array clip_norm(array x, const std::optional<array>& scale, float threshold) {
    x = astype(x, float32);
    if (scale) x = x * *scale;
    auto length = sqrt(sum(square(x), -1, true)) + array(1e-6f);
    return x * minimum(array(1.0f), array(threshold) / length);
}

struct Controller {
    MlxLinear down;
    MlxProjectionBatch gates;

    Controller(MlxLinear down, MlxLinear gain, MlxLinear forget)
        : down(std::move(down)), gates({&gain, &forget}) {}

    array operator()(const array& current, const array& message) const {
        const auto inputs = reshape(rms(stack({current, message}, -2)),
            {current.shape(0), current.shape(1), current.shape(2) * 2});
        const auto projected = gates(down(inputs));
        static const auto blend = compile([](const std::vector<array>& values) {
            return std::vector<array>{values[0] * (array(1.0f, values[0].dtype()) - tanh(values[3])) +
                values[1] * tanh(values[2])};
        });
        return blend({current, message, projected[0], projected[1]}).front();
    }
};

array short_convolution(const array& input, const array& weight,
    std::optional<array>& state, const std::optional<array>& dynamic = std::nullopt,
    const array* residual = nullptr) {
    const int batch = input.shape(0), tokens = input.shape(1), channels = input.shape(2);
    if (weight.ndim() != 2 || weight.shape(0) != channels || weight.shape(1) <= 0)
        throw std::invalid_argument("Inkling short convolution filter dimensions disagree");
    const int taps = weight.shape(1);
    const Shape state_shape{batch, taps - 1, channels};
    const int groups = dynamic ? dynamic->shape(-1) / taps : 1;
    if ((state && state->shape() != state_shape) || (residual && residual->shape() != input.shape()) ||
        (dynamic && (groups < 1 || channels % groups ||
        dynamic->shape() != Shape{batch, tokens, groups * taps})))
        throw std::invalid_argument("Inkling short convolution state or group dimensions disagree");
    const auto previous = state ? *state : zeros(state_shape, input.dtype());
    if (default_device() == Device::gpu && taps >= 2 && taps <= 16 &&
        (input.dtype() == float16 || input.dtype() == bfloat16 || input.dtype() == float32)) {
        static const auto kernel = [] {
            CompileOptions options;
            options.math_mode = MathMode::Safe;
            return fast::metal_kernel("tyloinfer_inkling_short_convolution",
                {"source", "filter", "previous", "adjust", "residual"}, {"output", "next_state"}, R"MSL(
                const uint channel = thread_position_in_grid.x;
                const uint tokens = uint(source_shape[1]);
                const uint channels = uint(source_shape[2]);
                if (channel >= channels) return;
                const uint rows = metal::max(tokens, uint(TAPS - 1));
                const uint batch = thread_position_in_grid.y / rows;
                const uint row = thread_position_in_grid.y % rows;
                const ulong source_base = ulong(batch) * tokens * channels + channel;
                const ulong state_base = ulong(batch) * (TAPS - 1) * channels + channel;
                if (row < tokens) {
                    float value = float(source[source_base + ulong(row) * channels]);
                    for (uint tap = 0u; tap < TAPS; ++tap) {
                        const uint at = row + tap;
                        const float sample = at < TAPS - 1
                            ? float(previous[state_base + ulong(at) * channels])
                            : float(source[source_base + ulong(at - (TAPS - 1)) * channels]);
                        value += sample * filter[ulong(channel) * TAPS + tap];
                        if (DYNAMIC) {
                            const uint group = channel / (channels / GROUPS);
                            const ulong offset = ((ulong(batch) * tokens + row) * GROUPS + group) * TAPS;
                            value += sample * float(adjust[offset + (TAPS - 1 - tap)]);
                        }
                    }
                    const ulong at = source_base + ulong(row) * channels;
                    const T convolved = T(value / metal::precise::sqrt(float((DYNAMIC ? 2 : 1) * TAPS + 1)));
                    if constexpr (ADD_RESIDUAL) output[at] = R(float(convolved) + float(residual[at]));
                    else output[at] = R(convolved);
                }
                if (row < TAPS - 1) {
                    const uint at = tokens + row;
                    next_state[state_base + ulong(row) * channels] = at < TAPS - 1
                        ? previous[state_base + ulong(at) * channels]
                        : source[source_base + ulong(at - (TAPS - 1)) * channels];
                }
                )MSL", "", true, false, options);
        }();
        const auto output_dtype = residual ? promote_types(input.dtype(), residual->dtype()) : input.dtype();
        auto outputs = kernel({input, weight, previous, dynamic ? *dynamic : zeros({2}, float32), residual ? *residual : input},
            {input.shape(), state_shape}, {output_dtype, input.dtype()},
            {channels, batch * std::max(tokens, taps - 1), 1}, {std::min(channels, 256), 1, 1},
            {{"T", input.dtype()}, {"R", output_dtype}, {"TAPS", taps}, {"DYNAMIC", int(dynamic.has_value())}, {"GROUPS", groups},
                {"ADD_RESIDUAL", residual != nullptr}},
            std::nullopt, false, {});
        state = std::move(outputs.at(1));
        return std::move(outputs.at(0));
    }
    const auto history = concatenate({previous, input}, 1);
    auto output = astype(input, float32);
    const auto adjustment = dynamic ?
        std::optional<array>(reshape(astype(*dynamic, float32), {batch, tokens, groups, taps})) : std::nullopt;
    for (int tap = 0; tap < taps; ++tap) {
        const auto sample = astype(time_slice(history, tap, tap + tokens), float32);
        const auto coefficient = reshape(slice(weight, {0, tap}, {channels, tap + 1}), {1, 1, channels});
        output = output + sample * coefficient;
        if (adjustment) {
            const int lag = taps - 1 - tap;
            auto dynamic_tap = slice(*adjustment, {0, 0, 0, lag}, {batch, tokens, groups, lag + 1});
            auto term = reshape(sample, {batch, tokens, groups, channels / groups}) * dynamic_tap;
            output = output + reshape(term, input.shape());
        }
    }
    state = contiguous(time_slice(history, tokens, tokens + taps - 1));
    output = astype(output / std::sqrt(float((dynamic ? 2 : 1) * taps + 1)), input.dtype());
    return residual ? *residual + output : output;
}

struct Geometry {
    int context, hidden, vocab, heads, kv_heads, dim, relative, extent, window;
    int block, loop_start, loop_end, experts, routes, log_floor;
    float eps, q_clip, r_clip, log_alpha, expert_scale, situ, situ_linear, logit_cap;
    LayerLoop loop;
    std::vector<bool> sliding;
};

Geometry geometry(const GgufModelSource& source, int context) {
    const auto meta = json::parse(source.metadata().at("gguf_metadata"));
    const auto config = json::parse(source.model_config_json());
    const auto integer = [&](const char* name) { return meta.at(std::string("inkling2_loop.") + name).get<int>(); };
    const auto number = [&](const char* name) { return meta.at(std::string("inkling2_loop.") + name).get<float>(); };
    const auto head_count = [&](const char* field) {
        const auto& values = meta.at(std::string("inkling2_loop.") + field);
        const auto count = values.is_array() ? values.at(0).get<int>() : values.get<int>();
        if (values.is_array() && std::any_of(values.begin(), values.end(),
            [&](const json& n) { return n.get<int>() != count; }))
            throw std::runtime_error("Inkling GGUF requires uniform head counts");
        return count;
    };
    const int layers = integer("block_count");
    const int begin = integer("loop.start"), end = integer("loop.end"), count = integer("loop.count");
    Geometry g{context, integer("embedding_length"), integer("vocab_size_unpadded"),
        head_count("attention.head_count"), head_count("attention.head_count_kv"),
        integer("attention.key_length"), integer("attention.relative_length"),
        integer("attention.relative_extent"), integer("attention.sliding_window"),
        integer("attn_res.block_size"), begin, end, integer("expert_count"),
        integer("expert_used_count"), integer("attention.log_scale_floor"),
        number("attention.layer_norm_rms_epsilon"), number("attention.qk_clip"),
        number("attention.relative_clip"), number("attention.log_scale_alpha"),
        number("expert_weights_scale"), number("activation.situ_beta"),
        number("activation.situ_linear_beta"), number("final_logit_softcapping"),
        LayerLoop(layers, begin, end, count), meta.at("inkling2_loop.attention.sliding_window_pattern").get<std::vector<bool>>()};
    if (context <= 0 || context > integer("context_length") || g.hidden <= 0 || g.vocab <= 0 ||
        g.heads <= 0 || g.kv_heads <= 0 || g.heads % g.kv_heads || g.dim != integer("attention.value_length") ||
        g.dim <= 0 || g.relative <= 0 || g.extent <= 0 || g.window <= 0 || g.block <= 0 ||
        g.loop_start <= 0 || g.loop_start % g.block || g.loop_end % g.block ||
        g.routes <= 0 || g.routes > g.experts || g.situ <= 0 || g.situ_linear <= 0 ||
        g.log_floor <= 0 || g.sliding.size() != std::size_t(layers) ||
        config.at("num_hidden_visits") != g.loop.visits()) {
        throw std::runtime_error("invalid Inkling loop geometry");
    }
    return g;
}


struct Layer {
    bool sliding;
    std::array<MlxRmsNorm, 3> input_norms;
    MlxRmsNorm ffn_norm;
    MlxLinear query, key, value, rel_query, sinks, attention_gate, attention_out, router;
    std::optional<MlxLinear> dynamic_k, dynamic_v;
    std::array<array, 4> shortconv;
    array query_clip, relative_clip, relative_table, ffn_res_query;
    std::optional<array> attn_res_query, router_bias, message_query;
    MlxGgufWeight gate_up, down;
    std::vector<Controller> controls;
    MlxProjectionBatch query_projections;

    Layer(const MlxGgufLayerTensors& t, const Geometry& g, int physical)
        : sliding(g.sliding.at(physical)),
          input_norms{t.norm(sliding ? "attn_norm.weight" : "attn_q_inp_norm.weight", g.eps),
              t.norm(sliding ? "attn_norm.weight" : "attn_k_inp_norm.weight", g.eps),
              t.norm(sliding ? "attn_norm.weight" : "attn_v_inp_norm.weight", g.eps)},
          ffn_norm(t.norm("ffn_norm.weight", g.eps)),
          query(t.linear("attn_q.weight")), key(t.linear("attn_k.weight")), value(t.linear("attn_v.weight")),
          rel_query(t.linear("attn_rel.weight")), sinks(t.linear("attn_sink_proj.weight")),
          attention_gate(t.linear("attn_gate.weight")), attention_out(t.linear("attn_output.weight")),
          router(t.linear("ffn_gate_inp.weight")),
          shortconv{t.dense("attn_k_conv.weight"), t.dense("attn_v_conv.weight"),
              t.dense("attn_post_conv.weight"), t.dense("ffn_post_conv.weight")},
          query_clip(t.dense("attn_q_clip.weight")), relative_clip(t.dense("attn_rel_clip.weight")),
          relative_table(contiguous(transpose(t.dense("attn_rel_bias.weight")))),
          ffn_res_query(t.dense("ffn_res_query.weight")),
          gate_up(t.experts("ffn_gate_up_exps.weight")), down(t.experts("ffn_down_exps.weight")),
          query_projections({&query, &rel_query, &sinks, &attention_gate}) {
        if (!sliding) {
            dynamic_k = t.linear("attn_k_conv_proj.weight");
            dynamic_v = t.linear("attn_v_conv_proj.weight");
        }
        if (physical) attn_res_query = t.dense("attn_res_query.weight");
        if (t.has("exp_probs_b.bias")) router_bias = t.dense("exp_probs_b.bias");
        if (physical >= g.loop_start && physical < g.loop_end) {
            message_query = t.dense("loop_msg_query.weight");
            for (const std::string component : {"q", "k", "v", "ffn"}) {
                const auto name = "loop_" + component + "_ctrl_";
                controls.push_back({t.linear(name + "down.weight"), t.linear(name + "gain.weight"),
                    t.linear(name + "forget.weight")});
            }
        }
    }

    array mixture(const array& input, const Geometry& g) const {
        const auto dims = input.shape();
        const auto flat = reshape(input, {-1, g.hidden});
        const auto route = moe_sigmoid_topk(router(flat), g.routes, router_bias, g.expert_scale);
        static const auto activate = compile([](const std::vector<array>& arguments) {
            const auto& projected = arguments[0];
            const auto parts = split(astype(projected, float32), Shape{projected.shape(-1) / 2}, -1);
            auto values = tanh(parts[0] / arguments[1]) * sigmoid(parts[0]) *
                tanh(parts[1] / arguments[2]) * (arguments[1] * arguments[2]);
            return std::vector<array>{astype(values, projected.dtype())};
        });
        const auto projected = gate_up.routed_matmul(flat, route.ids, true);
        const auto activated = activate({projected, array(g.situ), array(g.situ_linear)}).front();
        return reshape(down.routed_weighted_matmul(activated, route.ids, route.weights), dims);
    }
};

}  // namespace

namespace {
array mix_stream(const array& mixed, int stream) {
    return squeeze(slice(mixed, {0, 0, stream, 0},
        {mixed.shape(0), mixed.shape(1), stream + 1, mixed.shape(3)}), 2);
}

array attention_rows(const array& query, const array& key, const array& value,
    const array& table, const array& key_pos, int offset, int window, float scale) {
    const int batch = query.shape(0), heads = query.shape(1), rows = query.shape(2);
    const int keys = key.shape(2);
    const auto sinks = zeros({heads}, query.dtype());
    const auto bias = contiguous(table);
    // Bound the gathered relative-position bias independently of cache length.
    const int tile = std::clamp((1 << 18) / std::min(keys, window ? window : keys), 1, 128);
    std::vector<array> chunks;
    for (int first = 0; first < rows; first += tile) {
        const int last = std::min(rows, first + tile);
        const int key_start = window && rows > 1 ?
            std::max(0, keys - rows + first - window + 1) : 0;
        const int key_end = window && rows > 1 ? keys - rows + last : keys;
        const auto selected_positions = slice(key_pos, {key_start}, {key_end});
        auto scores = mlx_attention_relative_bias(bias, selected_positions, first, last - first,
            offset + first, window, query.dtype());
        const auto q = slice(query, {0, 0, first, 0}, {batch, heads, last, query.shape(3)});
        const auto k = slice(key, {0, 0, key_start, 0}, {key.shape(0), key.shape(1), key_end, key.shape(3)});
        const auto v = slice(value, {0, 0, key_start, 0},
            {value.shape(0), value.shape(1), key_end, value.shape(3)});
        chunks.push_back(fast::scaled_dot_product_attention(q, k, v, scale, "", std::move(scores), sinks));
    }
    return chunks.size() == 1 ? chunks.front() : concatenate(chunks, 2);
}

array apply_layer(const Layer& layer, const Geometry& g, int physical, int offset,
    const array& hidden, array& blocks, const std::optional<array>& messages,
    MlxInkling2VisitCache& cache) {
    const int batch = hidden.shape(0), rows = hidden.shape(1);
    const auto dtype = hidden.dtype();
    const auto attention_input = layer.attn_res_query ?
        depth_mix(blocks, hidden, *layer.attn_res_query) : expand_dims(hidden, 2);
    auto q = mix_stream(attention_input, 0);
    auto k = layer.sliding ? q : mix_stream(attention_input, 1);
    auto v = layer.sliding ? q : mix_stream(attention_input, 2);
    if (messages) {
        q = layer.controls[0](q, mix_stream(*messages, 0));
        k = layer.controls[1](k, mix_stream(*messages, 1));
        v = layer.controls[2](v, mix_stream(*messages, 2));
    }
    q = layer.input_norms[0](q);
    k = layer.sliding && !messages ? q : layer.input_norms[1](k);
    v = layer.sliding && !messages ? q : layer.input_norms[2](v);
    auto projections = layer.query_projections(q);
    auto next_k = layer.key(k), next_v = layer.value(v);
    const auto k_dynamic = layer.dynamic_k ? std::optional<array>((*layer.dynamic_k)(k)) : std::nullopt;
    const auto v_dynamic = layer.dynamic_v ? std::optional<array>((*layer.dynamic_v)(v)) : std::nullopt;
    next_k = short_convolution(next_k, layer.shortconv[0], cache.convolution[0], k_dynamic);
    next_v = short_convolution(next_v, layer.shortconv[1], cache.convolution[1], v_dynamic);
    q = clip_norm(reshape(projections[0], {batch, rows, g.heads, g.dim}), layer.query_clip, g.q_clip);
    auto r = clip_norm(reshape(projections[1], {batch, rows, g.heads, g.relative}), layer.relative_clip, g.r_clip);
    next_k = astype(clip_norm(reshape(next_k, {batch, rows, g.kv_heads, g.dim}), std::nullopt, g.q_clip), dtype);
    next_v = rms(reshape(next_v, {batch, rows, g.kv_heads, g.dim}), g.eps);
    if (!layer.sliding) {
        const auto position = astype(arange(offset + 1, offset + rows + 1, 1, int32), float32);
        auto scale = reshape(array(1.0f) + log(maximum(position / float(g.log_floor), array(1.0f))) * g.log_alpha,
            {1, rows, 1, 1});
        q = q * scale;
        r = r * scale;
    }
    auto relative = matmul(r, layer.relative_table);
    relative = concatenate({relative, zeros({batch, rows, g.heads, 1}, float32)}, -1);
    relative = transpose(relative - reshape(astype(projections[2], float32), {batch, rows, g.heads, 1}),
        {0, 2, 1, 3});

    array keys = next_k, values = next_v, positions = arange(offset, offset + rows, 1, int32);
    std::optional<array> appended;
    if (layer.sliding) {
        const int width = g.kv_heads * g.dim;
        appended = concatenate({reshape(next_k, {batch, rows, width}),
            reshape(astype(next_v, dtype), {batch, rows, width})}, -1);
        if (!cache.window) cache.window.emplace(zeros({batch, g.window, width * 2}, dtype));
        array joined = *appended;
        if (rows == 1) {
            cache.window = mlx_circular_cache_append(*cache.window, *appended, offset);
            const auto count = std::min(g.window, offset + 1);
            joined = time_slice(*cache.window, 0, count);
            positions = arange(count, int32);
            if (offset + 1 > g.window) {
                const auto tail = offset + 1 - g.window;
                positions = array(tail) + remainder(positions + array(g.window - (tail % g.window)), array(g.window));
            }
        } else if (offset > 0) {
            joined = concatenate({mlx_circular_cache_history(*cache.window, offset), *appended}, 1);
            positions = arange(offset + rows - joined.shape(1), offset + rows, 1, int32);
        }
        auto kv = split(joined, Shape{width}, -1);
        keys = transpose(reshape(kv[0], {batch, joined.shape(1), g.kv_heads, g.dim}), {0, 2, 1, 3});
        values = transpose(reshape(kv[1], {batch, joined.shape(1), g.kv_heads, g.dim}), {0, 2, 1, 3});
    } else {
        if (!cache.global) cache.global.emplace(batch, g.kv_heads, g.context, g.dim, std::min(g.context, 16), dtype);
        auto kv = cache.global->append(transpose(next_k, {0, 2, 1, 3}),
            transpose(astype(next_v, dtype), {0, 2, 1, 3}));
        keys = std::move(kv.first);
        values = std::move(kv.second);
        positions = arange(offset + rows, int32);
    }
    q = transpose(astype(q, dtype), {0, 2, 1, 3});
    auto attended = attention_rows(q, keys, values, relative, positions, offset, layer.sliding ? g.window : 0,
        1.0f / std::sqrt(float(g.dim)));
    if (appended && rows > 1) {
        cache.window = mlx_circular_cache_append(*cache.window, *appended, offset, attended);
        attended = depends({attended}, {*cache.window}).front();
    }
    attended = reshape(transpose(attended, {0, 2, 1, 3}), {batch, rows, g.heads * g.dim});
    auto branch = layer.attention_out(attended * sigmoid(projections[3]));
    const bool starts_block = physical % g.block == 0;
    branch = short_convolution(branch, layer.shortconv[2], cache.convolution[2], std::nullopt,
        starts_block ? nullptr : &hidden);
    if (starts_block) {
        auto block = expand_dims(rms(astype(hidden, float32)), -2);
        blocks = blocks.shape(2) ? concatenate({blocks, block}, -2) : block;
    }
    auto ffn = mix_stream(depth_mix(blocks, branch, layer.ffn_res_query), 0);
    if (messages) ffn = layer.controls[3](ffn, mix_stream(*messages, 3));
    ffn = layer.mixture(layer.ffn_norm(ffn), g);
    return short_convolution(ffn, layer.shortconv[3], cache.convolution[3], std::nullopt, &branch);
}
}  // namespace

struct MlxInkling2Model::Weights {
    Geometry geometry;
    MlxEmbedding embedding;
    MlxRmsNorm embedding_norm, output_norm;
    MlxLinear output;
    array output_query;
    std::vector<Layer> layers;

    Weights(const MfqContainer& model, const GgufModelSource& source, Geometry g)
        : geometry(std::move(g)),
          embedding(MlxEmbedding::load(model, "model.token_embedding.weight")),
          embedding_norm(load_dense_array(model.record("token_embd_norm.weight").dtype,
              model.read("token_embd_norm.weight")), geometry.eps),
          output_norm(load_dense_array(model.record("model.output_norm.weight").dtype,
              model.read("model.output_norm.weight")), geometry.eps),
          output(MlxLinear::load(model, "model.output.weight")),
          output_query(load_dense_array(model.record("output_res_query.weight").dtype,
              model.read("output_res_query.weight"))) {
        layers.reserve(geometry.loop.weight_layers());
        for (int i = 0; i < geometry.loop.weight_layers(); ++i)
            layers.emplace_back(MlxGgufLayerTensors(model, source, i), geometry, i);
    }
};

MlxInkling2Model MlxInkling2Model::load(const MfqContainer& model, int context) {
    const auto* source = dynamic_cast<const GgufModelSource*>(model.native_source());
    if (!source || source->architecture() != "inkling2_loop")
        throw std::invalid_argument("Inkling runtime requires an inkling2_loop GGUF");
    return MlxInkling2Model(std::make_shared<Weights>(model, *source, geometry(*source, context)));
}

array MlxInkling2Model::forward(const array& tokens, MlxInkling2Cache& cache, bool last_logit_only) const {
    const auto& g = weights_->geometry;
    if (tokens.ndim() != 2 || tokens.shape(0) <= 0 || tokens.shape(1) <= 0 ||
        cache.position < 0 || tokens.shape(1) > g.context - cache.position ||
        (cache.batch && cache.batch != tokens.shape(0)) ||
        (!cache.visits.empty() && cache.visits.size() != std::size_t(g.loop.visits()))) {
        throw std::invalid_argument("invalid Inkling tokens or visit cache");
    }
    const int count = tokens.shape(1);
    cache.batch = tokens.shape(0);
    cache.visits.resize(g.loop.visits());
    auto hidden = weights_->embedding_norm(weights_->embedding(tokens, bfloat16));
    std::optional<array> prelude, previous_output;
    auto blocks = zeros({tokens.shape(0), count, 0, g.hidden}, float32);
    auto prelude_blocks = blocks, previous_blocks = blocks;
    for (int visit = 0; visit < g.loop.visits(); ++visit) {
        const int physical = g.loop.weight_layer(visit);
        if (physical == g.loop_start) {
            if (!prelude) {
                prelude = hidden;
                prelude_blocks = blocks;
            } else {
                previous_blocks = slice(blocks, {0, 0, prelude_blocks.shape(2) + 1, 0}, blocks.shape());
                previous_output = hidden;
                hidden = *prelude;
                blocks = prelude_blocks;
            }
        }
        const auto& layer = weights_->layers[physical];
        auto messages = previous_output && layer.message_query ?
            std::optional<array>(depth_mix(previous_blocks, *previous_output, *layer.message_query)) : std::nullopt;
        hidden = apply_layer(layer, g, physical, cache.position, hidden, blocks, messages, cache.visits[visit]);
    }
    if (last_logit_only) {
        hidden = time_slice(hidden, count - 1, count);
        blocks = time_slice(blocks, count - 1, count);
    }
    hidden = mix_stream(depth_mix(blocks, hidden, weights_->output_query), 0);
    auto logits = astype(weights_->output(weights_->output_norm(hidden)), float32);
    if (g.logit_cap > 0) logits = tanh(logits / g.logit_cap) * g.logit_cap;
    cache.position += count;
    return slice(logits, {0, 0, 0}, {logits.shape(0), logits.shape(1), g.vocab});
}

MlxInkling2TextSessionState MlxInkling2Model::capture(const Cache& state,
    const std::vector<std::int64_t>& tokens, bool detached) const {
    const auto& g = weights_->geometry;
    if (state.batch != 1 || state.position <= 0 || state.position != static_cast<int>(tokens.size()) ||
        state.visits.size() != static_cast<std::size_t>(g.loop.visits()))
        throw std::invalid_argument("Inkling prefix does not match the live visit state");
    MlxInkling2TextSessionState result;
    result.tokens = tokens;
    result.window = g.window;
    result.visits.reserve(state.visits.size());
    for (const auto& visit : state.visits) {
        std::optional<array> window;
        if (visit.window) {
            window = mlx_circular_cache_history(*visit.window, state.position);
            if (detached) window = detached_copy(*window);
            window->eval();
            result.bytes += window->nbytes();
        }
        auto global = visit.global ? std::optional<MlxKvCacheSnapshot>(visit.global->snapshot(detached)) : std::nullopt;
        if (global) result.bytes += global->nbytes();
        std::array<array, 4> convolution{
            *visit.convolution[0], *visit.convolution[1], *visit.convolution[2], *visit.convolution[3]};
        for (auto& history : convolution) {
            history.eval();
            result.bytes += history.nbytes();
        }
        result.visits.push_back({std::move(window), std::move(global), std::move(convolution)});
    }
    return result;
}

MlxInkling2Cache MlxInkling2Model::restore(const TextSessionState& state) const {
    const auto& g = weights_->geometry;
    if (state.tokens.empty() || state.tokens.size() > static_cast<std::size_t>(g.context) ||
        state.visits.size() != static_cast<std::size_t>(g.loop.visits()) || state.window != g.window)
        throw std::invalid_argument("invalid Inkling prefix geometry");
    Cache result;
    result.position = static_cast<int>(state.tokens.size());
    result.batch = 1;
    result.visits.resize(state.visits.size());
    for (std::size_t i = 0; i < state.visits.size(); ++i) {
        const auto& source = state.visits[i];
        auto& target = result.visits[i];
        const int physical = g.loop.weight_layer(i);
        const auto& layer = weights_->layers[physical];
        if (layer.sliding) {
            if (source.global || !source.window ||
                source.window->shape() != Shape{1, std::min(g.window, result.position), 2 * g.kv_heads * g.dim})
                throw std::invalid_argument("invalid Inkling sliding prefix");
            target.window = mlx_circular_cache_append(
                zeros({1, g.window, 2 * g.kv_heads * g.dim}, source.window->dtype()),
                *source.window, result.position - source.window->shape(1));
        } else {
            if (source.window || !source.global || source.global->position != result.position)
                throw std::invalid_argument("invalid Inkling full-attention prefix");
            target.global.emplace(1, g.kv_heads, g.context, g.dim, std::min(g.context, 16), source.global->dtype);
            target.global->restore_snapshot(*source.global);
        }
        for (std::size_t index = 0; index < source.convolution.size(); ++index) {
            const auto& weight = layer.shortconv[index];
            if (source.convolution[index].shape() != Shape{1, weight.shape(1) - 1, weight.shape(0)})
                throw std::invalid_argument("invalid Inkling short-convolution prefix");
            target.convolution[index] = source.convolution[index];
        }
    }
    return result;
}

int MlxInkling2Model::context() const noexcept { return weights_->geometry.context; }
int MlxInkling2Model::vocabulary() const noexcept { return weights_->geometry.vocab; }

}  // namespace mfq::metal
