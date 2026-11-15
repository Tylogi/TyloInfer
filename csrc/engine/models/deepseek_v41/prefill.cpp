#include "env.h"
#include "ops.h"

#include <array>
#include <cstring>
#include <exception>

namespace tyloi::cuda::deepseek_v41 {
namespace {
using DenseTensor = ::tyloi::DenseTensor;
using HashBatch = tyloi::models::deepseek_v41::EngramHashBatch;

DenseTensor append_tail(const DenseTensor& previous, const DenseTensor& current, int64_t window) {
    if (current.size(1) >= window)
        return current.narrow(1, current.size(1) - window, window).clone();
    auto combined = previous.defined() ? ::tyloi::cuda::cat({previous, current}, 1) : current;
    const auto count = std::min(window, combined.size(1));
    return combined.narrow(1, combined.size(1) - count, count).clone();
}

HashBatch append_hash_tail(const HashBatch& previous, const HashBatch& current, int window) {
    HashBatch result;
    result.batch = current.batch;
    result.layers = current.layers;
    result.columns = current.columns;
    result.tokens = std::min(window, previous.tokens + current.tokens);
    result.values.resize(std::size_t(result.layers) * result.batch * result.tokens * result.columns);
    const int start = previous.tokens + current.tokens - result.tokens;
    for (int layer = 0; layer < result.layers; ++layer) for (int batch = 0; batch < result.batch; ++batch) {
        auto* output = result.values.data() +
            ((std::size_t(layer) * result.batch + batch) * result.tokens) * result.columns;
        for (int token = 0; token < result.tokens; ++token) {
            const int position = start + token;
            const auto* input = position < previous.tokens
                ? previous.layer_data(layer) + (std::size_t(batch) * previous.tokens + position) * result.columns
                : current.layer_data(layer) + (std::size_t(batch) * current.tokens + position - previous.tokens) * result.columns;
            std::memcpy(output + std::size_t(token) * result.columns, input,
                        std::size_t(result.columns) * sizeof(std::int64_t));
        }
    }
    return result;
}
DenseTensor forward_layer(const Model& model, State& state, const DenseTensor& ids, int64_t layer,
                          DenseTensor hidden, DenseTensor positions, DenseTensor host_ids) {
    const auto* profile_layer=tyloi::cuda::env_lookup("TYLOINFER_PROFILE_DSV41_LAYER");
    const bool selected=profile_layer && layer==std::atoi(profile_layer);
    tyloi::ops::ProfileCapture capture(selected);
    auto& block=*model.blocks.at(layer);
    MfqCudaGuard guard(block.cuda_device);
    auto local_ids = block.cpu_offloaded
        ? host_ids
        : tyloi::ops::copy(ids, ::tyloi::Device(::tyloi::kCUDA, block.cuda_device)).dense();
    auto local_pos = block.cpu_offloaded
        ? positions.to(::tyloi::kCPU).contiguous()
        : tyloi::ops::copy(positions, ::tyloi::Device(::tyloi::kCUDA, block.cuda_device)).dense();
    block.set_token_ids(*state.shared, local_ids);
    hidden = block.cpu_offloaded
        ? hidden.to(::tyloi::kCPU).contiguous()
        : tyloi::ops::copy(hidden, ::tyloi::Device(::tyloi::kCUDA, block.cuda_device)).dense();
    hidden = block.forward(*state.shared, std::move(hidden), local_pos, state.cache_pos,
                           mfq_nullopt, mfq_nullopt, 0);
    if(selected)capture.report("deepseek_v41.layer."+std::to_string(layer));
    return hidden;
}
}


tyloi::models::PrefillPlan Model::prefill_plan(
        const tyloi::models::State& value, const tyloi::models::PrefillOptions& policy) const {
    const auto& state = dynamic_cast<const State&>(value);
    TYLOINFER_RUNTIME_CHECK(state.belongs_to(*this), "DeepSeek-V4.1 state belongs to a different model");
    if (policy.transformed || policy.tokens <= 0 || policy.compute <= 0 || state.decode_position_delta != 0 ||
        config.causal_encoder_layers <= 0 || config.causal_encoder_layers >= num_hidden_layers())
        return {};
    // CED encodes bounded chunks but replays one local decoder window.
    return {true, policy.retained ? std::min(policy.compute, policy.tokens) : policy.tokens};
}

tyloi::StepSequence<tyloi::models::Prefill> Model::prefill(
        tyloi::models::Input input, tyloi::models::State& value,
        const tyloi::models::ForwardOptions&, const tyloi::models::PrefillOptions& policy) const {
    auto& state = dynamic_cast<State&>(value);
    const auto split = config.causal_encoder_layers;
    if (!prefill_plan(state, policy).layered) co_return;
    TYLOINFER_RUNTIME_CHECK(input.ids && input.ids.dim() == 2,
        "DeepSeek-V4.1 encoder prefill requires [batch,tokens] IDs");
    const auto batch = input.ids.size(0), tokens = input.ids.size(1);
    const auto compute = std::min<std::int64_t>(policy.compute, 8192);
    TYLOINFER_RUNTIME_CHECK(state.shared && batch > 0 && tokens > 0 && compute > 0 &&
        state.speculative_start < 0 && state.cache_pos >= 0 &&
        state.cache_pos <= max_position_embeddings() && tokens <= max_position_embeddings() - state.cache_pos,
        "invalid DeepSeek-V4.1 encoder prefill request");
    const auto previous_batch = state.batch;
    if (state.batch != batch || state.cache_pos == 0) state.reset(batch);
    const auto begin = state.cache_pos;
    TYLOINFER_RUNTIME_CHECK(begin <= max_position_embeddings() && tokens <= max_position_embeddings() - begin,
        "invalid DeepSeek-V4.1 encoder prefill request");
    const auto end = begin + tokens;
    struct Abort {
        State& state;
        std::int64_t batch;
        bool complete = false;
        ~Abort() noexcept {
            if (!complete) {
                try { state.reset(batch); }
                catch (...) { std::terminate(); }
            }
        }
    } abort{state, batch};
    if (previous_batch != batch || begin == 0) {
        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, state.allocation_growth()};
        state.allocate();
    }
    const Device primary(kCUDA, primary_device);
    if (input.ids.device() != primary || input.ids.scalar_type() != kInt64 || !input.ids.is_contiguous()) {
        const std::array<std::int64_t, 2> shape{batch, tokens};
        const auto bytes = make_contiguous_view(nullptr, shape, kInt64, primary).nbytes();
        std::vector<std::pair<Device, std::size_t>> growth{{primary, bytes}};
        if (input.ids.scalar_type() != kInt64 && input.ids.device() != primary)
            growth.emplace_back(input.ids.device(), bytes);
        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, std::move(growth)};
    }
    auto ids = device_ids(input.ids.dense());
    if (tokens > config.sliding_window) {
        tyloi::models::Prefill phase;
        phase.phase = tyloi::models::Prefill::Phase::encode;
        phase.first = 0; phase.last = static_cast<int>(split);
        co_yield phase;
    }

    DenseTensor tail_hidden, tail_pre, tail_ids;
    HashBatch tail_hashes;
    const bool capture_raw = input.raw_hidden != nullptr;
    std::vector<DenseTensor> tail_targets(config.dspark_target_layer_ids.size());
    const auto* profile = tyloi::cuda::env_lookup("TYLOINFER_PROFILE_SSD_EXPERTS");
    const bool trace = profile && profile[0] == '1';
    for (std::int64_t offset = 0; offset < tokens; offset += compute) {
        const auto count = std::min(compute, tokens - offset);
        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr,
            state.ced_growth(state.cache_pos, count, 0, split)};
        auto chunk_ids = ids.narrow(1, offset, count).contiguous();
        auto positions = ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + count,
            ::tyloi::TensorOptions().device(primary).dtype(::tyloi::kInt64));
        DenseTensor host_ids;
        if (needs_host_tokens) host_ids = chunk_ids.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        auto hidden = prepare_hidden(embed_tokens(chunk_ids), batch, count);
        state.begin_forward(capture_raw);
        for (std::int64_t layer = 0; layer < split; ++layer) {
            if (trace) std::cerr << "runtime_deepseek_v41_prefill phase=encoder layer=" << layer << " tokens=" << count << '\n';
            hidden = forward_layer(*this, state, chunk_ids, layer, std::move(hidden), positions, host_ids);
            co_yield tyloi::StepState::advanced;
        }
        for (std::int64_t layer = split; layer < num_hidden_layers(); ++layer) {
            auto& block = *blocks.at(layer);
            if (!block.kv_source()) continue;
            auto source = deepseek_v41_hc_input_cuda(hidden, state.shared->previous_pre,
                block.attention_norm.dense(), config.rms_eps);
            tyloi::ops::measure("deepseek_v41.ced.global_kv", [&] {
                block.publish_compressed_source(*state.shared, source, state.cache_pos);
                return source;
            });
            state.shared->layers.at(static_cast<std::size_t>(layer)).prefilled_global_end = state.cache_pos + count;
        }
        tail_hidden = append_tail(tail_hidden, hidden, config.sliding_window);
        tail_pre = append_tail(tail_pre, state.shared->previous_pre, config.sliding_window);
        tail_ids = append_tail(tail_ids, chunk_ids, config.sliding_window);
        tail_hashes = append_hash_tail(tail_hashes, state.shared->engram_hashes, int(config.sliding_window));
        if (capture_raw) {
            for (std::size_t slot = 0; slot < tail_targets.size(); ++slot) {
                const auto& target = state.shared->dspark_target_hiddens[slot];
                if (target.defined()) tail_targets[slot] = append_tail(tail_targets[slot], target, config.sliding_window);
            }
        }
        state.cache_pos += count;
    }

    tyloi::models::Prefill decode;
    decode.phase = tyloi::models::Prefill::Phase::decode;
    co_yield decode;
    const auto replay_tokens = tail_ids.size(1), replay_begin = end - replay_tokens;
    co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr,
        state.ced_growth(replay_begin, replay_tokens, split, num_hidden_layers())};
    const bool truncated_local = replay_begin > begin;
    state.cache_pos = replay_begin;
    state.shared->batch = batch;
    state.shared->tokens = replay_tokens;
    state.shared->cache_position = replay_begin;
    state.shared->previous_pre = std::move(tail_pre);
    state.shared->engram_hashes = std::move(tail_hashes);
    if (capture_raw) state.shared->dspark_target_hiddens = std::move(tail_targets);
    auto positions = ::tyloi::cuda::arange(replay_begin, end,
        ::tyloi::TensorOptions().device(primary).dtype(::tyloi::kInt64));
    DenseTensor host_ids;
    if (needs_host_tokens) host_ids = tail_ids.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
    for (std::int64_t layer = split; layer < num_hidden_layers(); ++layer) {
        if (trace) std::cerr << "runtime_deepseek_v41_prefill phase=decoder layer=" << layer << " tokens=" << replay_tokens << '\n';
        state.shared->layers.at(static_cast<std::size_t>(layer)).position = replay_begin;
        state.shared->layers.at(static_cast<std::size_t>(layer)).replay_local_window = truncated_local;
        tail_hidden = forward_layer(*this, state, tail_ids, layer, std::move(tail_hidden), positions, host_ids);
        co_yield tyloi::StepState::advanced;
    }
    state.cache_pos = end;
    std::cerr << "runtime_deepseek_v41_ced_prefill tokens=" << tokens << " encoder_layers=" << split
              << " decoder_tokens=" << replay_tokens << " compute_chunk=" << compute << '\n';
    tyloi::models::Prefill result;
    result.hidden = finalize_hidden(state,
        tyloi::ops::copy(tail_hidden, ::tyloi::Device(::tyloi::kCUDA, primary_device)).dense(),
        batch, replay_tokens);
    if (capture_raw) {
        for (auto& target : state.shared->dspark_target_hiddens)
            if (target.defined() && target.device() != primary)
                target = tyloi::ops::copy(target, primary).dense();
        result.raw = state.shared->dspark_target_hidden();
        if (policy.raw_host) result.raw = tyloi::ops::copy(result.raw, Device(kCPU));
        *input.raw_hidden = result.raw;
    }
    abort.complete = true;
    co_yield result;
}

} // namespace tyloi::cuda::deepseek_v41
