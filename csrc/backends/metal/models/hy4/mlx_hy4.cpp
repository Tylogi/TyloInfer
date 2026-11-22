#include "mlx_hy4.h"
#include "mlx_gguf_prefill.h"
#include "mlx_moe_ops.h"

#include "mlx_gguf_layer_tensors.h"
#include "mlx_hyper_connection.h"
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

array rotate(const array& x, const array& positions, float base) {
    const int dim = x.shape(-1);
    auto inverse = exp(astype(arange(0, dim, 2, int32), float32) * (-std::log(base) / dim));
    const auto angle = expand_dims(astype(positions, float32), -1) * inverse;
    return mlx_rope_adjacent(x, cos(angle), sin(angle));
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

struct LatentMoe {
    MlxLinear router, reduce_input, expand_output;
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
        : read(source.mapped_tensor_reader("per_layer_token_embd.weight")),
          grams(meta.at("hy_v4_flash.ple.ngram_size").get<int>()),
          heads(meta.at("hy_v4_flash.ple.heads_per_ngram").get<int>()),
          width(meta.at("hy_v4_flash.embedding_length_per_layer_input").get<int>()),
          pad(meta.at("hy_v4_flash.engram.pad_token_id").get<int>()),
          multipliers(meta.at("hy_v4_flash.ple.layer_multipliers").get<std::vector<std::uint64_t>>()),
          layers(meta.at("hy_v4_flash.ple.layers").get<std::vector<int>>()),
          sizes(meta.at("hy_v4_flash.ple.head_vocab_sizes").get<std::vector<int>>()),
          offsets(meta.at("hy_v4_flash.ple.head_offsets").get<std::vector<int>>()),
          mapping(meta.at("hy_v4_flash.engram.token_map").get<std::vector<int>>()) {
        const auto* tensor = source.find_tensor("per_layer_token_embd.weight");
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
    std::optional<array> old_window, new_window, index_pairs, selections;
};

array trailing_rope(const array& values, const array& positions, int rotary, float base) {
    const int unrotated = values.shape(-1) - rotary;
    if (unrotated == 0) return rotate(values, positions, base);
    return concatenate({axis_slice(values, -1, 0, unrotated),
        rotate(axis_slice(values, -1, unrotated, values.shape(-1)), positions, base)}, -1);
}

array latent_attention(const array& query, const array& rows, const array& sinks,
    float scale, int value_dim, const std::optional<array>& mask) {
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

array sliding_attention(MlxHy4AttentionCache& cache, const Geometry& g, const array& query,
    const array& next, const array& sinks, int position, float scale) {
    const int width = next.shape(-1), count = next.shape(1);
    if (!cache.window) cache.window = zeros({1, g.window, width}, next.dtype());
    if (count == 1) {
        cache.window = mlx_circular_cache_append(*cache.window, next, position);
        const auto visible = axis_slice(*cache.window, 1, 0, std::min(position + 1, g.window));
        return as_dense_attention(query, visible, sinks, position, 0, 0, scale, g.kv_rank);
    }

    const int history = std::min(position, g.window - 1);
    auto rows = next;
    if (history) {
        auto chronological = mlx_circular_cache_history(*cache.window, position);
        rows = concatenate({axis_slice(chronological, 1, chronological.shape(1) - history,
            chronological.shape(1)), next}, 1);
    }
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
    cache.window = mlx_circular_cache_append(*cache.window, next, position, attention);
    return attention;
}

void append_index_pool(MlxHy4AttentionCache& cache, const IndexPool& index, const Geometry& g,
    const array& input, int position, VerifiedLayerRows* verified) {
    const int width = g.index_width;
    const auto key = index.key_norm(index.key(input));
    auto pairs = concatenate({astype(index.value(key), float32), astype(index.gate(key), float32)}, -1);
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
    const int first = position - position % g.group_size;
    const auto positions = arange(first, first + split, g.group_size, int32);
    const auto rotated = astype(trailing_rope(pooled, positions, g.rotary, g.rotary_base), g.index_pool_dtype());
    if (cache.offloaded_groups) cache.offloaded_groups->append(rotated);
    else cache.groups->append(rotated);
}

array choose_resident_groups(const array& groups, const Geometry& g,
    const array& query, const array& weights, int position) {
    const int take = query.shape(1);
    const int budget = g.index_budget / g.group_size - 1;
    const auto current = reshape(floor_divide(arange(position, position + take, 1, int32),
        array(g.group_size)), {1, take, 1});
    const auto all_visible = reshape(arange(0, budget, 1, int32), {1, 1, budget});
    const int available = std::min(groups.shape(1), (position + take - 1) / g.group_size);
    if (available <= budget)
        return concatenate({where(all_visible < current, all_visible, array(-1, int32)), current}, -1);
    const auto history = reshape(arange(0, available, 1, int32), {1, 1, available});
    auto scores = mlx_sparse_indexer_weighted_scores(query, axis_slice(groups, 1, 0, available), weights);
    scores = where(history < current, scores, array(-std::numeric_limits<float>::infinity()));
    auto ids = astype(sort(axis_slice(argpartition(-scores, budget - 1, -1), -1, 0, budget), -1), int32);
    ids = where(current <= budget, all_visible, ids);
    return concatenate({where(ids < current, ids, array(-1, int32)), current}, -1);
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

array run_indexed_attention(MlxHy4AttentionCache& cache, const IndexPool& index, const Geometry& g,
    const array& query, const array& latent, const array& input, const array& query_low,
    const array& positions, const array& sinks, int position, float scale,
    const std::shared_ptr<QsaKvStore>& offload, VerifiedLayerRows* verified) {
    const int count = latent.shape(1), end = position + count;
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
    append_index_pool(cache, index, g, input, position, verified);

    if (end <= g.index_budget) {
        cache.last_indices = reshape(arange(0, end, 1, int32), {1, 1, end});
        const auto rows = cache.offloaded_latent
            ? reshape(cache.offloaded_latent->gather(*cache.last_indices), {1, end, stored.shape(-1)})
            : cache.latent->view();
        return as_dense_attention(query, g.indexed_kv.unpack(rows), sinks, position, 0, 0, scale, g.kv_rank);
    }

    const float weight_scale = 1.0f / std::sqrt(float(g.index_heads * g.index_width));
    const auto iq = trailing_rope(reshape(index.query(query_low), {1, count, g.index_heads, g.index_width}),
        positions, g.rotary, g.rotary_base);
    const auto head_weights = astype(index.weights(input), float32) * weight_scale;
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
    // Packed recalls can share more SSD blocks across queries within one gather.
    const int query_tile = cache.offloaded_latent && end > 2 * g.index_budget &&
        quantization.enabled() && quantization.value_bits() <= 4
        ? 32 : selected_query_tile;
    std::vector<array> output;
    for (int start = 0; start < count; start += query_tile) {
        const int stop = std::min(count, start + query_tile);
        const auto q = axis_slice(iq, 1, start, stop);
        const auto weights = axis_slice(head_weights, 1, start, stop);
        const auto groups = pooled ? choose_resident_groups(*pooled, g, q, weights, position + start)
            : cache.offloaded_groups->select(q, position + start, g.group_size, g.index_budget, weights, true);
        const auto ids = token_selection(groups, g, position + start);
        if (verified) verified->selections = ids;
        const auto queries = axis_slice(query, 2, start, stop);
        if (dense_rows && stop - start >= 4) {
            // At short contexts a dense boolean mask costs less than staging
            // a separate copy of the selected latent rows for each query.
            const auto selected = where(ids >= 0, ids, array(end, int32));
            auto visible = put_along_axis(zeros({1, stop - start, end + 1}, bool_), selected, ids >= 0, -1);
            output.push_back(latent_attention(queries, *dense_rows, sinks, scale, g.kv_rank,
                expand_dims(axis_slice(visible, -1, 0, end), 1)));
        } else {
            const auto staged = cache.offloaded_latent && !verified ?
                cache.offloaded_latent->gather_union(ids, resident ? &*resident : nullptr) : std::nullopt;
            if (staged) {
                output.push_back(latent_attention(queries, g.indexed_kv.unpack(staged->rows), sinks,
                    scale, g.kv_rank, staged->mask));
            } else {
                const auto mask = where(ids >= 0, array(0.0f), array(-std::numeric_limits<float>::infinity()));
                if (resident && !quantization.enabled()) {
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

struct MlaRows {
    array query, latent, query_low, gate;
};

template <class Attention>
MlaRows project_mla(const Attention& layer, const Geometry& g, const array& in, const array& positions, float base) {
    const int count = in.shape(1), plain = g.q_dim - g.rotary;
    const auto projected = layer.projections(in);
    auto low = layer.query_norm(projected[0]);
    const auto q = reshape(layer.query_b(low), {1, count, layer.heads, g.q_dim});
    const auto no_rope = transpose(axis_slice(q, -1, 0, plain), {0, 2, 1, 3});
    const auto absorbed = matmul(astype(no_rope, layer.key_absorption.dtype()), layer.key_absorption);
    const auto rope_q = transpose(rotate(axis_slice(q, -1, plain, g.q_dim), positions, base), {0, 2, 1, 3});
    auto query = concatenate({absorbed, astype(rope_q, float32)}, -1);
    const auto latent = concatenate({layer.latent_norm(axis_slice(projected[1], -1, 0, g.kv_rank), float32),
        astype(rotate(axis_slice(projected[1], -1, g.kv_rank, g.kv_rank + g.rotary), positions, base), float32)}, -1);
    return {std::move(query), astype(latent, float16), std::move(low), projected[2]};
}

template <class Attention>
array expand_mla(const Attention& layer, const Geometry& g, const array& attention, const array& gate, Dtype dtype) {
    const auto expanded = matmul(astype(transpose(attention, {0, 2, 1, 3}), layer.value_expansion.dtype()),
        layer.value_expansion);
    const auto gated = reshape(transpose(expanded, {0, 2, 1, 3}),
        {1, attention.shape(1), layer.heads * g.v_dim}) * sigmoid(astype(gate, float32));
    return layer.post_attention_norm(layer.output(astype(gated, dtype)));
}

array run_layer(const Layer& layer, const Geometry& g, MlxHy4AttentionCache& cache, array streams,
    const std::optional<array>& engram, const array& positions, int offset,
    const std::shared_ptr<QsaKvStore>& offload, VerifiedLayerRows* verified) {
    if (engram) streams = (*layer.engram)(streams, *engram, g);
    const auto mix = layer.attention_hc.collapse(streams);
    const auto in = layer.attention_norm(mix.branch);
    const auto rows = project_mla(layer, g, in, positions, layer.sliding ? g.window_rotary_base : g.rotary_base);
    if (verified && layer.sliding) {
        verified->old_window = cache.window;
        verified->new_window = rows.latent;
    }
    const float scale = 1.0f / std::sqrt(float(g.q_dim));
    const auto attention = layer.sliding ? sliding_attention(cache, g, rows.query, rows.latent, layer.sinks, offset, scale)
        : run_indexed_attention(cache, *layer.index, g, rows.query, rows.latent, in, rows.query_low, positions, layer.sinks,
            offset, scale, offload, verified);
    const auto attn_out = expand_mla(layer, g, attention, rows.gate, in.dtype());
    streams = MlxHyperConnection::expand(streams, attn_out, mix);
    const auto ffn_mix = layer.ffn_hc.collapse(streams);
    const auto ffn_in = layer.ffn_norm(ffn_mix.branch);
    const auto ffn = layer.dense ? (*layer.dense)(ffn_in) : (*layer.moe)(ffn_in, g);
    return MlxHyperConnection::expand(streams, layer.post_ffn_norm(ffn), ffn_mix);
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

    array operator()(const Geometry& g, const array& input, const array& positions,
        const array& history, const array& valid) const {
        const auto rows = project_mla(*this, g, attention_norm(input), positions, g.rotary_base);
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
        const auto positions = arange(position, position + 1 + depth, 1, int32);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& cached = cache.layers[target_layers[i]];
            auto selected = *cache.layers[index_layers[i]].last_indices;
            if (local_window) {
                const int begin = std::max(0, position - local_window);
                selected = where(selected < begin, selected, array(-1, int32));
                const auto local = reshape(arange(begin, position, 1, int32), {1, 1, position - begin});
                selected = concatenate({selected, local}, -1);
            }
            const auto shape = Shape{1, selected.shape(-1), g.indexed_kv.width()};
            auto history = cached.offloaded_latent ? reshape(cached.offloaded_latent->gather(selected), shape)
                : take(cached.latent->view(), reshape(maximum(selected, array(0, int32)), {-1}), 1);
            auto valid = reshape(selected >= array(0, int32), {1, selected.shape(-1)});
            hidden = layers[i](g, hidden, positions, g.indexed_kv.unpack(history), valid);
        }
        hidden = output_norm(hidden);
        for (int i = 1; i <= depth; ++i) {
            const auto row = axis_slice(hidden, 1, i, i + 1);
            const auto prior = correction_embedding_norm(embedding(previous, bfloat16));
            const auto corrected = row + correction(concatenate({correction_hidden_norm(row), prior}, -1));
            previous = reshape(select(mlx_last_token_logits(output(corrected), g.vocab)), {1, 1});
        }
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
          head_weight(contiguous(transpose(dense_weight(model, "output_hc_fn.weight")))),
          head_bias(dense_weight(model, "output_hc_base.weight")), head_scale(dense_weight(model, "output_hc_scale.weight")),
          offload(mlx_configure_qsa_kv_offload(g.context, std::count(g.sliding.begin(), g.sliding.begin() + g.layers, false),
              1, g.kv_rank + g.rotary, g.index_width, g.group_size)) {
        if (source.find_tensor("per_layer_token_embd.weight")) engram.emplace(source, meta);
        layers.reserve(g.layers);
        for (int layer = 0; layer < g.layers; ++layer) {
            const bool has_engram = engram && std::find(engram->layers.begin(), engram->layers.end(), layer) != engram->layers.end();
            layers.emplace_back(MlxGgufLayerTensors(model, source, layer), g, layer, has_engram);
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

struct MlxHy4Model::Verification {
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
    for (int layer = 0; layer < g.layers; ++layer) {
        std::optional<array> rows;
        if (w.layers[layer].engram) rows = w.engram->gather(layer, history, first);
        streams = run_layer(w.layers[layer], g, cache.layers[layer], std::move(streams), rows,
            positions, offset, w.offload, verification ? &verification->layers[layer] : nullptr);
    }
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
    const auto& draft = *w.draft;
    const auto& g = w.g;
    std::optional<Verification> verification;
    std::optional<array> logits;
    const int input_end = cache.position;
    std::optional<std::int32_t> last_emitted;
    request.callback = [callback = std::move(request.callback), &last_emitted](std::int64_t token) {
        last_emitted = static_cast<std::int32_t>(token);
        return !callback || callback(token);
    };
    MlxMtpEngineCallbacks callbacks;
    callbacks.predictor = MlxMtpPredictorDescriptor::recurrent(draft.max_depth);
    callbacks.target_cache_position = [&] { return cache.position; };
    callbacks.prepare_draft = [&](const MlxMtpDraftContext& context, const MlxMtpTokenSelector& select) {
        if (context.requested_depth)
            draft.propose(g, cache, w.token_embedding, w.output, context.pending_token,
                context.requested_depth, select);
    };
    callbacks.verify_target = [&](std::int32_t pending, const array& drafts, int count) {
        auto ids = mlx_mtp_verification_ids(pending, drafts, count);
        if (count) verification.emplace();
        auto target = reshape(forward_impl(ids, cache, false, verification ? &*verification : nullptr),
            {count + 1, g.vocab});
        if (count) logits = target;
        else last_logits = target;
        return MlxMtpTargetBatch{std::move(target), verification ? *verification->hidden : *cache.last_hidden};
    };
    callbacks.resolve_target = [&](int accepted, int proposed) {
        if (!proposed) return;
        const auto& trace = *verification;
        const int keep = accepted + 1, position = trace.position + keep;
        if (accepted != proposed) {
            for (int i = 0; i < g.layers; ++i) {
                auto& layer = cache.layers[i];
                const auto& saved = trace.layers[i];
                if (g.sliding[i]) {
                    const auto new_rows = axis_slice(*saved.new_window, 1, 0, keep);
                    layer.window = mlx_circular_cache_append(*saved.old_window, new_rows, trace.position);
                    continue;
                }
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
            if (w.engram) {
                const auto end = trace.engram_tokens.end() - (proposed - accepted);
                const int length = std::min(position, w.engram->grams - 1);
                cache.engram_tail.assign(end - length, end);
            }
            cache.last_hidden = axis_slice(*trace.hidden, 1, accepted, keep);
            cache.position = position;
        }
        last_logits = axis_slice(*logits, 0, accepted, keep);
        verification.reset();
        logits.reset();
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
