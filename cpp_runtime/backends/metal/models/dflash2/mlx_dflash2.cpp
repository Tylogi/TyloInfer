#include "mlx_dflash2.h"
#include "hf_safetensors_store.h"
#include "mlx_cached_kernel.h"
#include "mlx_tensor.h"
#include "mlx_weight_residency.h"
#include "mlx_kernel_prepare.h"
#include "mtp_policy.h"
#include "../flash_next/mlx_qwen4_causal_lm.h"
#include "../qwen35/mlx_qwen35_causal_lm.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <nlohmann/json.hpp>

namespace mfq::metal {
namespace {
using namespace mlx::core;

array normalized(const array& value, const array& weight, float epsilon) {
    return fast::rms_norm(astype(value, bfloat16), weight, epsilon);
}

array rotated(const array& value, int dimension, float theta, int offset) {
    return fast::rope(value, dimension, false, theta, 1.f, offset);
}

array dynamic_convolution(const array& input, const array& dynamic, const array& base,
    const array& residual, int group, bool finish) {
    static const auto kernel = cached_metal_kernel("tinfer_dflash2_dynamic_convolution",
        {"input", "dynamic", "base", "residual", "params"}, {"output"}, R"METAL(
uint index = thread_position_in_grid.x;
uint hidden = params[0], rows = params[1], groups = hidden / params[2], stage = params[3];
if (index >= hidden * rows) return;
uint row = index / hidden, channel = index % hidden, conv = channel / params[2];
uint dynamic_base = row * groups * 4u + stage * 2u * groups;
uint weight_base = stage * 2u * hidden;
float result = float(input[index]) * (float(base[weight_base + channel]) + float(dynamic[dynamic_base + conv]));
if (row % params[4] != 0u)
    result += float(input[index - hidden]) *
        (float(base[weight_base + hidden + channel]) + float(dynamic[dynamic_base + groups + conv]));
if (stage != 0u) result += float(residual[index]);
output[index] = bfloat(result);
)METAL");
    const int hidden = input.shape(2), rows = input.shape(0) * input.shape(1);
    return kernel({input, dynamic, base, residual,
        array({uint32_t(hidden), uint32_t(rows), uint32_t(group), uint32_t(finish), uint32_t(input.shape(1))})},
        {input.shape()}, {bfloat16}, {hidden * rows, 1, 1}, {256, 1, 1}, {}, std::nullopt, false, {})[0];
}

std::pair<array, array> candidates(const array& logits, int rows, int vocabulary) {
    static const auto kernel = cached_metal_kernel("tinfer_dflash2_top16",
        {"values", "ids", "params"}, {"scores", "tokens"}, R"METAL(
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const uint shards = params[1], row = threadgroup_position_in_grid.x / (FIRST ? shards : 1u);
const uint shard = FIRST ? threadgroup_position_in_grid.x % shards : 0u;
const uint width = FIRST ? params[0] : shards * 16u;
const uint begin = FIRST ? width * shard / shards : 0u;
const uint end = FIRST ? width * (shard + 1u) / shards : width;
float best[16]; uint best_id[16];
for (uint rank = 0u; rank < 16u; ++rank) { best[rank] = -INFINITY; best_id[rank] = 0xffffffffu; }
for (uint position = begin + tid; position < end; position += 128u) {
    const ulong at = ulong(row) * width + position;
    float score = float(values[at]); uint id = FIRST ? position : ids[at];
    if (score < best[15] || (score == best[15] && id >= best_id[15])) continue;
    for (uint rank = 0u; rank < 16u; ++rank) {
        if (score > best[rank] || (score == best[rank] && id < best_id[rank])) {
            float displaced = best[rank]; best[rank] = score; score = displaced;
            uint displaced_id = best_id[rank]; best_id[rank] = id; id = displaced_id;
        }
    }
}
threadgroup float group_score[4]; threadgroup uint group_id[4];
for (uint rank = 0u; rank < 16u; ++rank) {
    float score = simd_max(best[0]);
    uint id = simd_min(best[0] == score ? best_id[0] : 0xffffffffu);
    if (lane == 0u) { group_score[sg] = score; group_id[sg] = id; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    score = group_score[0]; id = group_id[0];
    for (uint other = 1u; other < 4u; ++other)
        if (group_score[other] > score || (group_score[other] == score && group_id[other] < id)) {
            score = group_score[other]; id = group_id[other];
        }
    if (tid == 0u) {
        const ulong at = (ulong(row) * (FIRST ? shards : 1u) + shard) * 16u + rank;
        scores[at] = score; tokens[at] = id;
    }
    if (best_id[0] == id) {
        for (uint local = 0u; local < 15u; ++local) { best[local] = best[local + 1u]; best_id[local] = best_id[local + 1u]; }
        best[15] = -INFINITY; best_id[15] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
)METAL");
    const int shards = std::min(8, std::max(1, (vocabulary + 255) / 256));
    const auto params = array({uint32_t(vocabulary), uint32_t(shards)});
    auto partial = kernel({logits, zeros({1}, uint32), params},
        {{rows, shards, 16}, {rows, shards, 16}}, {float32, uint32},
        {rows * shards * 128, 1, 1}, {128, 1, 1}, {{"FIRST", 1}}, std::nullopt, false, {});
    auto result = kernel({partial[0], partial[1], params}, {{rows, 16}, {rows, 16}}, {float32, uint32},
        {rows * 128, 1, 1}, {128, 1, 1}, {{"FIRST", 0}}, std::nullopt, false, {});
    return {std::move(result[0]), std::move(result[1])};
}

MlxDFlash2Proposal select_proposals(const array& logits, const array& hidden,
    const array& predecessor, const array& successor, const array& anchors,
    const array& offsets, int maximum_depth, const MlxDFlash2Config& config) {
    const auto batch = anchors.shape(0);
    auto [unary, ids] = candidates(contiguous(logits), logits.size() / config.vocabulary, config.vocabulary);
    static const auto kernel = cached_metal_kernel("tinfer_dflash2_select_chain",
        {"unary", "ids", "hidden", "predecessor", "successor", "anchors", "offsets", "params"}, {"tokens", "confidence"}, R"METAL(
uint lane = thread_index_in_simdgroup, candidate = simdgroup_index_in_threadgroup;
const uint request = threadgroup_position_in_grid.x;
const uint stride = params[0], vocab = params[1];
const uint start = uint(offsets[request]), steps = uint(offsets[request + 1u]) - start;
const uint output_base = request * stride;
threadgroup float scores[16]; threadgroup uint previous;
if (thread_index_in_threadgroup == 0u) previous = uint(anchors[request]);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint row = 0u; row < steps; ++row) {
    const uint at = (start + row) * 16u;
    uint id = ids[at + candidate];
    uint safe_id = min(id, vocab - 1u), safe_previous = min(previous, vocab - 1u);
    float edge = 0.f;
    for (uint column = lane; column < uint(RANK); column += 32u)
        edge += (float(predecessor[ulong(safe_previous) * RANK + column]) *
            float(hidden[ulong(start + row) * RANK + column])) * float(successor[ulong(safe_id) * RANK + column]);
    edge = simd_sum(edge);
    if (lane == 0u) scores[candidate] = unary[at + candidate] + edge;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index_in_threadgroup == 0u) {
        uint best = 0u;
        for (uint i = 1u; i < 16u; ++i) if (scores[i] > scores[best]) best = i;
        float mass = 0.f;
        for (uint i = 0u; i < 16u; ++i) mass += exp(scores[i] - scores[best]);
        previous = ids[at + best];
        tokens[output_base + row] = int(previous); confidence[output_base + row] = 1.f / mass;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (thread_index_in_threadgroup == 0u) for (uint row = steps; row < stride; ++row) {
    tokens[output_base + row] = -1;
    confidence[output_base + row] = 0.f;
}
)METAL");
    auto output = kernel({unary, ids, contiguous(hidden), predecessor, successor, anchors, offsets,
        array({uint32_t(maximum_depth), uint32_t(config.vocabulary)})},
        {{batch, maximum_depth}, {batch, maximum_depth}}, {int32, float32},
        {batch * 512, 1, 1}, {512, 1, 1}, {{"RANK", config.selector_rank}}, std::nullopt, false, {});
    return {std::move(output[0]), std::move(output[1])};
}

struct Loader {
    HfSafetensorStore store;
    std::size_t bytes = 0;
    std::vector<array> allocations;
    explicit Loader(const std::filesystem::path& directory) : store(directory, false) {}
    array dense(const std::string& name, const Shape& shape, bool retain = true) {
        const auto& record = store.tensor(name);
        if (record.dtype != "BF16" || std::vector<std::int64_t>(shape.begin(), shape.end()) != record.shape)
            throw std::invalid_argument("DFlash2 tensor shape or precision differs: " + name);
        auto value = array(allocator::malloc(record.nbytes), shape, bfloat16);
        store.read_tensor(record, {reinterpret_cast<std::byte*>(value.data<std::uint8_t>()), std::size_t(record.nbytes)});
        if (retain) { bytes += value.nbytes(); MlxWeightResidency::track(value); allocations.push_back(value); }
        return value;
    }
    MlxLinear linear(const std::string& name, int rows, int columns) {
        auto source = dense(name + ".weight", {rows, columns}, false);
        auto quantized = quantize(source, 64, 4, "affine");
        eval(quantized);
        for (const auto& value : quantized) {
            bytes += value.nbytes(); MlxWeightResidency::track(value); allocations.push_back(value);
        }
        MlxLinear result(MlxAffineWeight::from_arrays(quantized[0], quantized[1], quantized[2], 4, 64));
        bytes += result.affine_weight_ref()->packed_nbytes() - quantized[0].nbytes() - quantized[1].nbytes() - quantized[2].nbytes();
        return result;
    }
};

struct Layer {
    array input_norm, attention_base, q_norm, k_norm, post_norm, mlp_base;
    MlxLinear dynamic_attention, q, k, v, output, dynamic_mlp, gate, up, down;
    MlxProjectionBatch qkv, kv, ffn;
    Layer(Loader& loader, const MlxDFlash2Config& c, int index)
        : input_norm(loader.dense("layers." + std::to_string(index) + ".input_layernorm.weight", {c.hidden})),
          attention_base(loader.dense("layers." + std::to_string(index) + ".attention_conv.base_kernel", {2, 2, c.hidden})),
          q_norm(loader.dense("layers." + std::to_string(index) + ".self_attn.q_norm.weight", {c.head_dimension})),
          k_norm(loader.dense("layers." + std::to_string(index) + ".self_attn.k_norm.weight", {c.head_dimension})),
          post_norm(loader.dense("layers." + std::to_string(index) + ".post_attention_layernorm.weight", {c.hidden})),
          mlp_base(loader.dense("layers." + std::to_string(index) + ".mlp_conv.base_kernel", {2, 2, c.hidden})),
          dynamic_attention(loader.linear("layers." + std::to_string(index) + ".attention_conv.kernel_projection", 4 * c.hidden / c.convolution_group, c.hidden)),
          q(loader.linear("layers." + std::to_string(index) + ".self_attn.q_proj", c.query_heads * c.head_dimension, c.hidden)),
          k(loader.linear("layers." + std::to_string(index) + ".self_attn.k_proj", c.kv_heads * c.head_dimension, c.hidden)),
          v(loader.linear("layers." + std::to_string(index) + ".self_attn.v_proj", c.kv_heads * c.head_dimension, c.hidden)),
          output(loader.linear("layers." + std::to_string(index) + ".self_attn.o_proj", c.hidden, c.query_heads * c.head_dimension)),
          dynamic_mlp(loader.linear("layers." + std::to_string(index) + ".mlp_conv.kernel_projection", 4 * c.hidden / c.convolution_group, c.hidden)),
          gate(loader.linear("layers." + std::to_string(index) + ".mlp.gate_proj", c.intermediate, c.hidden)),
          up(loader.linear("layers." + std::to_string(index) + ".mlp.up_proj", c.intermediate, c.hidden)),
          down(loader.linear("layers." + std::to_string(index) + ".mlp.down_proj", c.hidden, c.intermediate)),
          qkv({&q, &k, &v}), kv({&k, &v}), ffn({&gate, &up}) {}
};

}

MlxDFlash2Config MlxDFlash2Config::load(const std::filesystem::path& directory) {
    std::ifstream input(directory / "config.json");
    nlohmann::json json;
    if (!input || !(input >> json)) throw std::invalid_argument("DFlash2 config is unavailable");
    const auto& d = json.at("dflash_config");
    MlxDFlash2Config c;
    c.hidden = json.at("hidden_size"); c.vocabulary = json.at("vocab_size");
    c.layers = json.at("num_hidden_layers"); c.intermediate = json.at("intermediate_size");
    c.query_heads = json.at("num_attention_heads"); c.kv_heads = json.at("num_key_value_heads");
    c.head_dimension = json.at("head_dim"); c.window = json.at("sliding_window");
    c.block_size = d.at("block_size"); c.convolution_group = d.at("conv_group_size");
    c.selector_rank = d.at("selector_rank"); c.candidates = d.at("selector_top_k");
    c.mask_token = d.at("mask_token_id"); c.target_layers = d.at("target_layer_ids").get<std::vector<int>>();
    c.rope_theta = json.at("rope_parameters").at("rope_theta"); c.epsilon = json.at("rms_norm_eps");
    if (c.hidden <= 0 || c.hidden % 64 || c.vocabulary < 16 || c.layers < 1 || c.layers > 64 ||
        c.intermediate <= 0 || c.intermediate % 64 || c.query_heads <= 0 || c.kv_heads <= 0 ||
        c.query_heads % c.kv_heads || c.head_dimension <= 0 || c.head_dimension % 2 ||
        (std::int64_t(c.query_heads) * c.head_dimension) % 64 || c.window <= 0 ||
        c.block_size != 8 || c.candidates != 16 || c.convolution_group <= 0 ||
        c.hidden % c.convolution_group ||
        c.selector_rank <= 0 || c.selector_rank % 32 || c.selector_rank > 256 ||
        d.at("conv_kernel_size").get<int>() != 2 || c.target_layers.empty() ||
        c.target_layers.front() < 0 || c.target_layers.size() > 32 ||
        std::adjacent_find(c.target_layers.begin(), c.target_layers.end(), std::greater_equal<int>()) != c.target_layers.end() ||
        c.mask_token < 0 || c.mask_token >= c.vocabulary || json.value("hidden_act", "silu") != "silu" ||
        std::int64_t(c.hidden) * c.target_layers.size() > std::numeric_limits<int>::max() ||
        std::int64_t(c.hidden) * c.block_size > std::numeric_limits<int>::max() ||
        std::int64_t(c.query_heads) * c.head_dimension > std::numeric_limits<int>::max() ||
        !std::isfinite(c.rope_theta) || c.rope_theta <= 0 || !std::isfinite(c.epsilon) || c.epsilon <= 0)
        throw std::invalid_argument("unsupported DFlash2 checkpoint geometry");
    return c;
}

struct MlxDFlash2::Impl {
    MlxDFlash2Config config;
    std::vector<std::unique_ptr<Layer>> layers;
    MlxLinear context_projection, selector_projection;
    array hidden_norm, final_norm, predecessor, successor;
    std::optional<array> mask_embedding;
    std::vector<array> allocations;
    std::size_t bytes = 0;
    Impl(Loader& loader, MlxDFlash2Config c)
        : config(std::move(c)),
          context_projection(loader.linear("fc", config.hidden, config.hidden * int(config.target_layers.size()))),
          selector_projection(loader.linear("candidate_selector.hidden_projection", config.selector_rank, config.hidden)),
          hidden_norm(loader.dense("hidden_norm.weight", {config.hidden})),
          final_norm(loader.dense("norm.weight", {config.hidden})),
          predecessor(loader.dense("candidate_selector.predecessor_codebook", {config.vocabulary, config.selector_rank})),
          successor(loader.dense("candidate_selector.successor_codebook", {config.vocabulary, config.selector_rank})) {
        for (int index = 0; index < config.layers; ++index) layers.push_back(std::make_unique<Layer>(loader, config, index));
        if (loader.store.tensors().contains("mask_embedding")) mask_embedding = loader.dense("mask_embedding", {config.hidden});
        bytes = loader.bytes;
        allocations = std::move(loader.allocations);
    }
    void validate_context(const MlxDFlash2Context& context) const {
        if (context.owner != this || context.position < 0 ||
            context.position > std::numeric_limits<int>::max() - config.block_size ||
            context.layers.size() != std::size_t(config.layers) || !context.window ||
            context.window->shape() != Shape{1, config.window, config.hidden} || context.window->dtype() != bfloat16)
            throw std::invalid_argument("DFlash2 context state differs from checkpoint");
        for (const auto& layer : context.layers) for (const auto* value : {&layer.keys, &layer.values})
            if (!*value || (*value)->shape() != Shape{1, config.kv_heads, config.window, config.head_dimension} ||
                (*value)->dtype() != bfloat16)
                throw std::invalid_argument("DFlash2 context cache geometry differs from checkpoint");
    }
};

MlxDFlash2 MlxDFlash2::load(const std::filesystem::path& directory) {
    auto config = MlxDFlash2Config::load(directory);
    Loader loader(directory);
    return MlxDFlash2(std::make_shared<Impl>(loader, std::move(config)));
}
const MlxDFlash2Config& MlxDFlash2::config() const noexcept { return impl_->config; }
std::size_t MlxDFlash2::weight_bytes() const noexcept { return impl_->bytes; }
void MlxDFlash2::track_weights() const {
    for (const auto& value : impl_->allocations) MlxWeightResidency::track(value);
}
std::size_t MlxDFlash2Context::bytes() const noexcept {
    std::size_t result = window ? window->nbytes() : 0;
    for (const auto& layer : layers) {
        if (layer.keys) result += layer.keys->nbytes();
        if (layer.values) result += layer.values->nbytes();
    }
    return result;
}

void MlxDFlash2::reset(MlxDFlash2Context& context) const {
    const auto& c = config();
    MlxDFlash2Context next;
    next.owner = impl_.get();
    next.window = zeros({1, c.window, c.hidden}, bfloat16);
    for (int index = 0; index < c.layers; ++index) next.layers.push_back({
        zeros({1, c.kv_heads, c.window, c.head_dimension}, bfloat16),
        zeros({1, c.kv_heads, c.window, c.head_dimension}, bfloat16)});
    context = std::move(next);
}

void MlxDFlash2::append_context(MlxDFlash2Context& context, const array& target_hidden, int start_position) const {
    const ContextCommit commit{&context, target_hidden, start_position};
    append_context_batch({&commit, 1});
}

void MlxDFlash2::append_context_batch(std::span<const ContextCommit> commits) const {
    if (commits.empty()) return;
    const auto& c = config();
    struct Lane {
        int begin, retained, position, end_position;
        array slots, index;
        std::vector<MlxDFlash2Context::Layer> layers;
    };
    std::vector<Lane> lanes;
    std::vector<array> features, outputs;
    std::unordered_set<MlxDFlash2Context*> seen;
    int begin = 0;
    for (const auto& [context, hidden, start] : commits) {
        if (!context || !seen.insert(context).second) throw std::invalid_argument("invalid DFlash2 context lane");
        impl_->validate_context(*context);
        if (hidden.ndim() != 3 || hidden.shape(0) != 1 || hidden.shape(1) < 1 ||
            hidden.shape(2) != c.hidden * int(c.target_layers.size()) || start != context->position ||
            hidden.shape(1) > std::numeric_limits<int>::max() - start)
            throw std::invalid_argument("DFlash2 context position or hidden features differ");
        const int count = hidden.shape(1), retained = std::min(count, c.window), offset = count - retained;
        features.push_back(astype(slice(hidden, {0, offset, 0}, {1, count, hidden.shape(2)}), bfloat16));
        auto slots = remainder(arange(start + offset, start + count, int32), array(c.window, int32));
        auto index = broadcast_to(reshape(slots, {1, 1, retained, 1}), {1, c.kv_heads, retained, c.head_dimension});
        lanes.push_back({begin, retained, start + offset, start + count, std::move(slots), std::move(index), {}});
        if (begin > std::numeric_limits<int>::max() - retained)
            throw std::invalid_argument("DFlash2 context batch is too large");
        begin += retained;
    }
    auto hidden = normalized(impl_->context_projection(features.size() == 1 ? features[0] : concatenate(features, 1)),
        impl_->hidden_norm, c.epsilon);
    const auto part = [](const array& packed, const Lane& lane) {
        if (!lane.begin && lane.retained == packed.shape(1)) return packed;
        return slice(packed, {0, lane.begin, 0}, {1, lane.begin + lane.retained, packed.shape(2)});
    };
    for (int layer = 0; layer < c.layers; ++layer) {
        auto projected = impl_->layers[layer]->kv(hidden);
        for (std::size_t i = 0; i < lanes.size(); ++i) {
            auto& lane = lanes[i];
            const auto& state = commits[i].context->layers[layer];
            const Shape shape{1, lane.retained, c.kv_heads, c.head_dimension};
            auto key = transpose(reshape(part(projected[0], lane), shape), {0, 2, 1, 3});
            key = rotated(normalized(key, impl_->layers[layer]->k_norm, c.epsilon),
                c.head_dimension, c.rope_theta, lane.position);
            auto value = transpose(reshape(part(projected[1], lane), shape), {0, 2, 1, 3});
            auto keys = put_along_axis(*state.keys, lane.index, key, 2);
            auto values = put_along_axis(*state.values, lane.index, value, 2);
            outputs.push_back(keys);
            outputs.push_back(values);
            lane.layers.push_back({std::move(keys), std::move(values)});
        }
    }
    std::vector<array> windows;
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        const auto& lane = lanes[i];
        auto index = broadcast_to(reshape(lane.slots, {1, lane.retained, 1}), {1, lane.retained, c.hidden});
        windows.push_back(put_along_axis(*commits[i].context->window, index, part(hidden, lane), 1));
        outputs.push_back(windows.back());
    }
    if (!MlxKernelPreparation::collecting()) async_eval(outputs);
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        auto& context = *commits[i].context;
        context.layers = std::move(lanes[i].layers);
        context.window = std::move(windows[i]);
        context.position = lanes[i].end_position;
    }
}

MlxDFlash2Proposal MlxDFlash2::propose(const MlxDFlash2Context& context, int anchor,
    const array& embeddings, const std::function<array(const array&)>& output_head,
    const std::function<void(std::string_view, const array&)>& trace) const {
    const MlxDFlash2Context* lane = &context;
    const auto token = static_cast<std::int32_t>(anchor);
    auto result = propose_batch({&lane, 1}, {&token, 1}, embeddings, output_head, {}, trace);
    return {reshape(result.tokens, {-1}), reshape(result.confidence, {-1})};
}

struct MlxDFlash2::ProposalLattice {
    array logits, selector;
    std::vector<int> offsets;
    int maximum_depth;
};

MlxDFlash2::ProposalLattice MlxDFlash2::proposal_lattice(std::span<const MlxDFlash2Context* const> contexts,
    std::span<const std::int32_t> anchors, const array& embeddings,
    const std::function<array(const array&)>& output_head,
    std::span<const int> depths,
    const std::function<void(std::string_view, const array&)>& trace) const {
    const auto& c = config();
    if (contexts.empty() || contexts.size() != anchors.size() ||
        contexts.size() > std::numeric_limits<int>::max() /
            (c.block_size * std::int64_t(std::max(c.hidden, 1024))) ||
        (!depths.empty() && depths.size() != contexts.size()) || !output_head)
        throw std::invalid_argument("invalid DFlash2 proposal batch");
    const int batch = static_cast<int>(contexts.size());
    if (embeddings.shape() != Shape{batch, c.block_size, c.hidden})
        throw std::invalid_argument("invalid DFlash2 proposal embedding geometry");

    std::vector<int> positions;
    std::vector<int> row_offsets{0};
    int maximum_depth = 0;
    std::vector<array> masks;
    auto slots = arange(c.window, int32);
    const auto proposal_mask = ones({c.block_size, c.block_size}, bool_);
    for (int lane = 0; lane < batch; ++lane) {
        if (!contexts[lane] || anchors[lane] < 0 || anchors[lane] >= c.vocabulary)
            throw std::invalid_argument("invalid DFlash2 proposal lane");
        const auto& context = *contexts[lane];
        impl_->validate_context(context);
        const int depth = depths.empty() ? c.block_size - 1 : depths[lane];
        if (depth < 1 || depth >= c.block_size) throw std::invalid_argument("invalid DFlash2 proposal depth");
        row_offsets.push_back(row_offsets.back() + depth);
        maximum_depth = std::max(maximum_depth, depth);
        positions.push_back(context.position);
        auto logical = context.position < c.window ? slots : array(context.position - c.window, int32) +
            remainder(slots - array(context.position % c.window, int32) + array(c.window, int32), array(c.window, int32));
        auto lower = reshape(maximum(arange(c.block_size, int32) + array(context.position - (c.window - 1), int32),
            array(0, int32)), {c.block_size, 1});
        auto past_mask = (expand_dims(logical, 0) >= lower) & (expand_dims(logical, 0) < array(context.position, int32));
        masks.push_back(reshape(concatenate({past_mask, proposal_mask}, 1),
            {1, 1, c.block_size, c.window + c.block_size}));
    }
    auto mask = batch == 1 ? masks.front() : concatenate(masks, 0);
    auto offsets = array(positions.begin(), {batch}, int32);
    auto hidden = astype(embeddings, bfloat16);
    if (impl_->mask_embedding) hidden = concatenate({slice(hidden, {0, 0, 0}, {batch, 1, c.hidden}),
        broadcast_to(reshape(*impl_->mask_embedding, {1, 1, c.hidden}), {batch, c.block_size - 1, c.hidden})}, 1);
    for (int index = 0; index < c.layers; ++index) {
        const auto& layer = *impl_->layers[index];
        auto input = normalized(hidden, layer.input_norm, c.epsilon);
        auto dynamic = layer.dynamic_attention(input);
        auto convolved = dynamic_convolution(input, dynamic, layer.attention_base, hidden, c.convolution_group, false);
        auto qkv = layer.qkv(convolved);
        auto query = transpose(reshape(qkv[0], {batch, c.block_size, c.query_heads, c.head_dimension}), {0, 2, 1, 3});
        auto key = transpose(reshape(qkv[1], {batch, c.block_size, c.kv_heads, c.head_dimension}), {0, 2, 1, 3});
        auto value = transpose(reshape(qkv[2], {batch, c.block_size, c.kv_heads, c.head_dimension}), {0, 2, 1, 3});
        query = normalized(query, layer.q_norm, c.epsilon);
        key = normalized(key, layer.k_norm, c.epsilon);
        if (batch == 1) {
            query = rotated(query, c.head_dimension, c.rope_theta, positions.front());
            key = rotated(key, c.head_dimension, c.rope_theta, positions.front());
        } else {
            query = fast::rope(query, c.head_dimension, false, c.rope_theta, 1.f, offsets);
            key = fast::rope(key, c.head_dimension, false, c.rope_theta, 1.f, offsets);
        }
        std::vector<array> keys, values;
        for (const auto* context : contexts) {
            keys.push_back(*context->layers[index].keys);
            values.push_back(*context->layers[index].values);
        }
        auto attended = fast::scaled_dot_product_attention(query,
            concatenate({batch == 1 ? keys.front() : concatenate(keys, 0), key}, 2),
            concatenate({batch == 1 ? values.front() : concatenate(values, 0), value}, 2),
            1.f / std::sqrt(float(c.head_dimension)), "", mask);
        auto projected = layer.output(reshape(transpose(attended, {0, 2, 1, 3}),
            {batch, c.block_size, c.query_heads * c.head_dimension}));
        auto residual = dynamic_convolution(projected, dynamic, layer.attention_base, hidden, c.convolution_group, true);
        input = normalized(residual, layer.post_norm, c.epsilon);
        dynamic = layer.dynamic_mlp(input);
        convolved = dynamic_convolution(input, dynamic, layer.mlp_base, residual, c.convolution_group, false);
        projected = layer.down(layer.ffn.swiglu(convolved));
        hidden = dynamic_convolution(projected, dynamic, layer.mlp_base, residual, c.convolution_group, true);
        if (trace) trace("layer_" + std::to_string(index), hidden);
    }
    hidden = normalized(hidden, impl_->final_norm, c.epsilon);
    const auto pack = [&](const array& values) {
        if (depths.empty()) return slice(values, {0, 1, 0}, {batch, c.block_size, values.shape(2)});
        std::vector<array> rows;
        for (int lane = 0; lane < batch; ++lane) rows.push_back(slice(values,
            {lane, 1, 0}, {lane + 1, 1 + depths[lane], values.shape(2)}));
        return batch == 1 ? rows.front() : concatenate(rows, 1);
    };
    auto head_hidden = depths.empty() ? hidden : pack(hidden);
    auto logits = output_head(head_hidden);
    if (logits.shape() != Shape{head_hidden.shape(0), head_hidden.shape(1), c.vocabulary})
        throw std::invalid_argument("DFlash2 target output head vocabulary differs");
    auto selector = impl_->selector_projection(head_hidden);
    if (trace) { trace("final_hidden", head_hidden); trace("logits", logits); trace("selector_hidden", selector); }
    return {depths.empty() ? pack(logits) : logits, depths.empty() ? pack(selector) : selector,
        std::move(row_offsets), maximum_depth};
}

MlxDFlash2Proposal MlxDFlash2::propose_batch(std::span<const MlxDFlash2Context* const> contexts,
    std::span<const std::int32_t> anchors, const array& embeddings,
    const std::function<array(const array&)>& output_head, std::span<const int> depths,
    const std::function<void(std::string_view, const array&)>& trace) const {
    auto lattice = proposal_lattice(contexts, anchors, embeddings, output_head, depths, trace);
    return select_proposals(lattice.logits, lattice.selector, impl_->predecessor, impl_->successor,
        array(anchors.begin(), {static_cast<int>(anchors.size())}, int32),
        array(lattice.offsets.begin(), {static_cast<int>(lattice.offsets.size())}, int32),
        lattice.maximum_depth, config());
}

std::vector<MlxDFlash2Tree> MlxDFlash2::propose_tree_batch(
    std::span<const MlxDFlash2Context* const> contexts, std::span<const std::int32_t> anchors,
    const array& embeddings, const std::function<array(const array&)>& output_head,
    std::span<const MlxDFlash2TreeInput> inputs) const {
    if (inputs.size() != contexts.size() || inputs.size() != anchors.size())
        throw std::invalid_argument("DFlash2 tree batch geometry disagrees");
    constexpr int choices = 16;
    const int vocab = config().vocabulary;
    std::vector<int> depths;
    std::vector<std::int32_t> previous_rows, first_anchors;
    for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
        const auto& input = inputs[lane];
        if (input.rows < 2 || input.rows > 16 || input.depth < 1 || input.depth > 7 ||
            input.lookup.size() > static_cast<std::size_t>(input.depth) ||
            input.lookup.size() != input.lookup_rates.size() ||
            std::any_of(input.lookup.begin(), input.lookup.end(), [vocab](auto id) { return id < 0 || id >= vocab; }) ||
            std::any_of(input.lookup_rates.begin(), input.lookup_rates.end(), [](const auto& rates) {
                return std::any_of(rates.begin(), rates.end(), [](float p) { return !std::isfinite(p) || p < 0.f || p > 1.f; });
            }))
            throw std::invalid_argument("invalid DFlash2 tree bounds or lookup chain");
        depths.push_back(input.depth);
        for (int depth = 0; depth < input.depth; ++depth) {
            previous_rows.push_back(depth ? static_cast<int>(previous_rows.size()) - 1 : -1);
            first_anchors.push_back(anchors[lane]);
        }
    }
    auto draft = proposal_lattice(contexts, anchors, embeddings, output_head, depths, {});
    const int rows = draft.offsets.back();
    auto [unary, ids] = candidates(contiguous(draft.logits), rows, vocab);
    static const auto kernel = cached_metal_kernel("tinfer_dflash2_tree_edges",
        {"unary", "ids", "features", "predecessor", "successor", "previous", "anchors", "params"},
        {"edges"}, R"METAL(
const uint pair = threadgroup_position_in_grid.x * 4u + simdgroup_index_in_threadgroup;
const uint row = pair / 256u, candidate = pair % 16u, parent = (pair / 16u) % 16u;
const uint lane = thread_index_in_simdgroup;
if (row >= params[0]) return;
const int earlier = previous[row];
const int from = earlier >= 0 ? int(ids[16u * uint(earlier) + parent]) : parent == 0 ? anchors[row] : -1;
const uint to = ids[16u * row + candidate];
if (from < 0 || to >= params[1]) {
    if (lane == 0u) edges[pair] = -INFINITY;
    return;
}
float edge = 0.f;
for (uint c = lane; c < uint(RANK); c += 32u)
    edge += (float(predecessor[ulong(from) * RANK + c]) * float(features[ulong(row) * RANK + c])) *
        float(successor[ulong(to) * RANK + c]);
edge = simd_sum(edge);
if (lane == 0u) edges[pair] = unary[16u * row + candidate] + edge;
)METAL");
    const int groups = (rows * choices * choices + 3) / 4;
    auto edges = kernel({unary, ids, contiguous(draft.selector), impl_->predecessor, impl_->successor,
        array(previous_rows.begin(), {rows}, int32), array(first_anchors.begin(), {rows}, int32),
        array({static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(vocab)})},
        {{rows, choices, choices}}, {float32}, {groups * 128, 1, 1}, {128, 1, 1},
        {{"RANK", config().selector_rank}}, std::nullopt, false, {})[0];
    eval(ids, edges);

    // Greedy portion of Splash's draft_select_tree. Lookup tokens and draft
    // candidates enter independently. A draft can join a lookup node once.
    struct Candidate {
        float score;
        int order, parent, position, slot;
        bool operator<(const Candidate& other) const {
            return score != other.score ? score < other.score : order > other.order;
        }
    };
    const auto* candidate_ids = ids.data<std::uint32_t>();
    const auto* scores = edges.data<float>();
    constexpr float intercept[] = {0.f, -.35f, -.8f, -.75f, -.75f, -.85f, -.45f};
    std::vector<MlxDFlash2Tree> result;
    for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
        const auto& input = inputs[lane];
        const int begin = draft.offsets[lane], chain = static_cast<int>(input.lookup.size());
        auto& tree = result.emplace_back();
        tree.tokens.push_back(anchors[lane]);
        tree.parents.push_back(-1);
        tree.reach.push_back(1.f);
        std::array<bool, 16> on_lookup{true};
        std::priority_queue<Candidate> frontier;
        int order = 0;
        const auto lookup_rate = [&](int position) {
            const auto* options = candidate_ids + (begin + position) * choices;
            const auto* found = std::find(options, options + choices, static_cast<std::uint32_t>(input.lookup[position]));
            return input.lookup_rates[position][found == options ? 0 : found == options + choices ? 2 : 1];
        };
        const auto expand = [&](int parent, int position, int predecessor, float score) {
            const int row = begin + position;
            const auto* options = candidate_ids + row * choices;
            const auto* logits = scores + (row * choices + predecessor) * choices;
            const auto lookup = on_lookup[parent] && position < chain ? input.lookup[position] : -1;
            const float reserved = lookup >= 0 ? lookup_rate(position) : 0.f;
            const float best = *std::max_element(logits, logits + choices);
            std::array<float, choices> weights;
            float total = 0.f;
            for (int i = 0; i < choices; ++i) {
                weights[i] = std::exp(logits[i] - best);
                total += weights[i];
            }
            float left = 1.f - reserved, bound = score;
            for (int child = 0; child < 4 && total > 0.f; ++child) {
                auto ranked = std::max_element(weights.begin(), weights.end());
                if (*ranked == 0.f) break;
                const int slot = static_cast<int>(ranked - weights.begin());
                const float q = std::clamp(*ranked / total, 1.e-6f, 1.f - 1.e-6f);
                *ranked = 0.f;
                const bool joins = static_cast<int>(options[slot]) == lookup;
                const float cap = left + (joins ? reserved : 0.f);
                const float p = std::min(cap, 1.f / (1.f + std::exp(-(.7f * std::log(q / (1.f - q)) + intercept[position]))));
                left = cap - (joins ? std::max(p, reserved) : p);
                bound = std::min(bound, score * p);
                if (bound > 0.f) frontier.push({bound, order++, parent, position, slot});
            }
        };
        expand(0, 0, 0, 1.f);
        if (chain) frontier.push({lookup_rate(0), order++, 0, 0, -1});
        while (!frontier.empty() && tree.tokens.size() < static_cast<std::size_t>(input.rows)) {
            const auto node = frontier.top();
            frontier.pop();
            const bool lookup = node.slot < 0;
            const std::int64_t token = lookup ? input.lookup[node.position] :
                candidate_ids[(begin + node.position) * choices + node.slot];
            int row = 1;
            for (; row < static_cast<int>(tree.tokens.size()); ++row)
                if (tree.parents[row] == node.parent && tree.tokens[row] == token) break;
            if (row == static_cast<int>(tree.tokens.size())) {
                tree.tokens.push_back(token);
                tree.parents.push_back(node.parent);
                tree.reach.push_back(node.score);
                on_lookup[row] = on_lookup[node.parent] && node.position < chain && token == input.lookup[node.position];
            }
            const int next = node.position + 1;
            if (next >= input.depth) continue;
            if (!lookup) expand(row, next, node.slot, node.score);
            else if (next < chain) frontier.push({node.score * lookup_rate(next), order++, row, next, -1});
        }
    }
    return result;
}

void MlxDFlash2::prepare(const std::function<array(const array&)>& output_head) const {
    if (auto* preparation = MlxKernelPreparation::current()) preparation->collect([&] {
        const auto& c = config();
        MlxDFlash2Context context;
        reset(context);
        append_context(context, zeros({1, c.block_size, c.hidden * int(c.target_layers.size())}, bfloat16), 0);
        auto proposal = propose(context, 0, zeros({1, c.block_size, c.hidden}, bfloat16), output_head);
        preparation->add(proposal.tokens); preparation->add(proposal.confidence);
        for (const auto dtype : {float16, bfloat16, float32})
            preparation->add(sample_greedy(zeros({c.block_size, c.vocabulary}, dtype)));
    });
}

std::int32_t MlxDFlash2::generate_greedy(MlxQwen4CausalLm& target,
    const std::vector<std::int64_t>& prompt, std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    int prefill_chunk_size, std::optional<int> stop_token, MlxDFlash2GenerationStats* stats, int maximum_drafts) const {
    return generate(target, prompt, MlxSamplingParams{}, max_tokens, callback, prefill_callback,
        prefill_chunk_size, stop_token, stats, maximum_drafts);
}

namespace {

struct Qwen4DFlashTarget {
    MlxQwen4CausalLm& model;
    std::shared_ptr<MlxQwen4RequestCache> cache;
    explicit Qwen4DFlashTarget(MlxQwen4CausalLm& target) : model(target) {}
    void begin() { cache = model.make_request_cache(); }
    const auto& config() const { return model.config(); }
    auto layer_count() const { return model.layer_count(); }
    int context_capacity() const { return model.context_capacity(); }
    void set_hidden_capture_layers(std::vector<int> layers) { model.set_hidden_capture_layers(std::move(layers)); }
    auto take_hidden_capture() { return model.take_hidden_capture(); }
    auto embed_token_ids(const array& ids) { return model.embed_token_ids(ids); }
    auto project_hidden_logits(const array& hidden) { return model.project_hidden_logits(hidden); }
    auto prefill(const std::vector<std::int64_t>& tokens) { return model.prefill_request(*cache, tokens, false); }
    auto decode(int token) { return model.decode_requests({cache}, {token}); }
    auto verify(const std::vector<std::int64_t>& tokens) { return model.verify_requests({cache}, {tokens}).first; }
    void resolve(int accepted) { model.resolve_request(*cache, accepted); }
    int position() const { return model.request_cache_position(*cache); }
};

struct Qwen35DFlashTarget {
    MlxQwen35CausalLm& model;
    int drafts = 0;
    explicit Qwen35DFlashTarget(MlxQwen35CausalLm& target) : model(target) {}
    void begin() { model.reset_cache(1); }
    const auto& config() const { return model.config(); }
    auto layer_count() const { return model.layer_count(); }
    int context_capacity() const { return model.context_capacity(); }
    void set_hidden_capture_layers(std::vector<int> layers) { model.set_hidden_capture_layers(std::move(layers)); }
    auto take_hidden_capture() { return model.take_hidden_capture(); }
    auto embed_token_ids(const array& ids) { return model.embed_tokens(ids); }
    auto project_hidden_logits(const array& hidden) { return model.project_hidden_logits(hidden); }
    array ids(const std::vector<std::int64_t>& tokens) const {
        return array(tokens.begin(), {1, int(tokens.size())}, int32);
    }
    array prefill(const std::vector<std::int64_t>& tokens) {
        return reshape(model.prefill_logits(ids(tokens)), {-1, int(config().vocab_size)});
    }
    array decode(int token) { return reshape(model.forward(ids({token}), true), {-1, int(config().vocab_size)}); }
    array verify(const std::vector<std::int64_t>& tokens) {
        drafts = int(tokens.size()) - 1;
        return reshape(model.verify_tokens(ids(tokens)), {-1, int(config().vocab_size)});
    }
    array verify(const array& tokens) {
        drafts = tokens.shape(1) - 1;
        return reshape(model.verify_tokens(tokens), {-1, int(config().vocab_size)});
    }
    void resolve(int accepted) { model.resolve_verified_tokens(accepted, drafts); }
    int position() const { return model.cache_position(); }
};

template<class Target>
std::int32_t generate_target(const MlxDFlash2& draft, Target& target,
    const std::vector<std::int64_t>& prompt, const MlxSamplingParams& sampling, std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    int prefill_chunk_size, std::optional<int> stop_token, MlxDFlash2GenerationStats* stats, int maximum_drafts) {
    const auto& c = draft.config();
    if (prompt.empty() || prompt.size() > std::size_t(target.context_capacity()) || max_tokens < 0 ||
        prefill_chunk_size < 1 || maximum_drafts < 1 || maximum_drafts >= c.block_size ||
        target.config().hidden_size != c.hidden || target.config().vocab_size != c.vocabulary ||
        c.target_layers.back() >= target.layer_count() ||
        (stop_token && (*stop_token < 0 || *stop_token >= c.vocabulary)))
        throw std::invalid_argument("DFlash2 generation target, prompt or limit differs");
    for (auto id : prompt) if (id < 0 || id >= c.vocabulary)
        throw std::invalid_argument("DFlash2 prompt token is outside target vocabulary");
    if (sampling.has_penalties()) throw std::invalid_argument("DFlash2 does not support sampling penalties");
    MlxSampler sampler(sampling);
    const auto select = [&](const array& logits) { return reshape(astype(sampler.sample(logits), int32), {-1}); };
    if (stats) *stats = {};
    if (!max_tokens) return 0;
    target.set_hidden_capture_layers(c.target_layers);
    struct CaptureReset {
        Target& target;
        ~CaptureReset() { target.set_hidden_capture_layers({}); }
    } reset_capture{target};
    MlxDFlash2Context context;
    draft.reset(context);
    std::optional<array> logits;
    using Clock = std::chrono::steady_clock;
    for (std::size_t start = 0; start < prompt.size();) {
        const auto count = std::min<std::size_t>(prefill_chunk_size, prompt.size() - start);
        const auto began = Clock::now();
        if (!start) target.begin();
        logits = target.prefill({prompt.begin() + start, prompt.begin() + start + count});
        auto features = target.take_hidden_capture();
        if (!features) throw std::runtime_error("DFlash2 target omitted prefill features");
        draft.append_context(context, *features, int(start));
        logits->eval();
        const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - began).count();
        if (stats) stats->prefill_ms += elapsed;
        if (prefill_callback) prefill_callback(count, elapsed);
        start += count;
    }
    const int limit = std::min(max_tokens, target.context_capacity() - int(prompt.size()) + 1);
    int pending = select(*logits).template item<std::int32_t>();
    int generated = 0;
    bool running = true;
    const auto emit = [&](int token) {
        if (stop_token && token == *stop_token) { running = false; return false; }
        ++generated;
        running = !callback || callback(token);
        return running;
    };
    emit(pending);
    if (stats) stats->standard_tokens = generated;
    while (running && generated < limit) {
        const int remaining = limit - generated;
        if (remaining == 1) {
            const auto started = Clock::now();
            auto value = target.decode(pending);
            const auto token = select(value).template item<std::int32_t>();
            if (stats) stats->target_ms += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            emit(token);
            if (stats) ++stats->standard_tokens;
            break;
        }
        const int position = target.position();
        if (position != context.position) throw std::logic_error("DFlash2 target and draft positions diverged");
        std::vector<std::int32_t> input(c.block_size, c.mask_token);
        input[0] = pending;
        const auto cycle_started = Clock::now();
        auto proposal = draft.propose(context, pending, target.embed_token_ids(array(input.begin(), {1, c.block_size}, int32)),
            [&](const array& hidden) { return target.project_hidden_logits(hidden); });
        constexpr bool device_verification = requires(const array& tokens) { target.verify(tokens); };
        if constexpr (device_verification) async_eval({proposal.tokens});
        else proposal.tokens.eval();
        if (stats) stats->head_ms += std::chrono::duration<double, std::milli>(Clock::now() - cycle_started).count();
        const int depth = std::min(maximum_drafts, remaining - 1);
        std::vector<std::int64_t> teachers{pending};
        const auto read_teachers = [&] {
            for (int i = 0; i < depth; ++i) {
                const int id = proposal.tokens.template data<std::int32_t>()[i];
                if (id < 0 || id >= c.vocabulary) throw std::runtime_error("DFlash2 proposed an invalid token");
                teachers.push_back(id);
            }
        };
        if constexpr (!device_verification) read_teachers();
        const auto verify_started = Clock::now();
        auto verified = [&] {
            if constexpr (device_verification) {
                auto selected = slice(proposal.tokens, {0}, {depth});
                auto ids = concatenate({full({1}, pending, int32), selected});
                return target.verify(reshape(ids, {1, depth + 1}));
            } else return target.verify(teachers);
        }();
        int retained = 0;
        struct Resolve {
            Target& target;
            int& retained;
            bool active = true;
            ~Resolve() { if (active) try { target.resolve(retained); } catch (...) {} }
        } transaction{target, retained};
        auto predictions = select(verified);
        if constexpr (device_verification) eval({predictions, proposal.tokens});
        else predictions.eval();
        if constexpr (device_verification) read_teachers();
        if (stats) stats->target_ms += std::chrono::duration<double, std::milli>(Clock::now() - verify_started).count();
        auto features = target.take_hidden_capture();
        if (!features) throw std::runtime_error("DFlash2 target omitted verification features");
        const auto decision = mfq::engine::mtp::verify_greedy(
            std::span<const std::int32_t>(proposal.tokens.template data<std::int32_t>(), depth),
            std::span<const std::int32_t>(predictions.template data<std::int32_t>(), depth + 1));
        const int matched = int(decision.accepted_drafts);
        if (stats) {
            ++stats->cycles; stats->proposed += c.block_size - 1;
            stats->verified += depth; stats->accepted += matched;
            ++stats->depth_cycles[depth];
            for (int i = 0; i < depth; ++i) ++stats->position_verified[i];
            for (int i = 0; i < matched; ++i) ++stats->position_accepted[i];
        }
        for (int i = 0; running && i < matched; ++i) {
            if (stop_token && teachers[i + 1] == *stop_token) { running = false; break; }
            ++retained;
            emit(teachers[i + 1]);
        }
        if (running) {
            pending = decision.next_token;
            emit(pending);
        }
        const auto resolve_started = Clock::now();
        target.resolve(retained);
        if (stats) stats->rollback_ms += std::chrono::duration<double, std::milli>(Clock::now() - resolve_started).count();
        transaction.active = false;
        if (running && generated < limit) {
            const auto context_started = Clock::now();
            draft.append_context(context, slice(*features, {0, 0, 0}, {1, retained + 1, features->shape(2)}), position);
            if (stats) stats->head_ms += std::chrono::duration<double, std::milli>(Clock::now() - context_started).count();
        }
        if (stats) stats->measured_depth_ms[depth] += std::chrono::duration<double, std::milli>(Clock::now() - cycle_started).count();
    }
    return generated;
}

}

std::int32_t MlxDFlash2::generate(MlxQwen4CausalLm& target,
    const std::vector<std::int64_t>& prompt, const MlxSamplingParams& sampling, std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    int prefill_chunk_size, std::optional<int> stop_token, MlxDFlash2GenerationStats* stats, int maximum_drafts) const {
    Qwen4DFlashTarget adapter(target);
    return generate_target(*this, adapter, prompt, sampling, max_tokens, callback, prefill_callback,
        prefill_chunk_size, stop_token, stats, maximum_drafts);
}

std::int32_t MlxDFlash2::generate(MlxQwen35CausalLm& target,
    const std::vector<std::int64_t>& prompt, const MlxSamplingParams& sampling, std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    int prefill_chunk_size, std::optional<int> stop_token, MlxDFlash2GenerationStats* stats, int maximum_drafts) const {
    Qwen35DFlashTarget adapter(target);
    return generate_target(*this, adapter, prompt, sampling, max_tokens, callback, prefill_callback,
        prefill_chunk_size, stop_token, stats, maximum_drafts);
}

}
