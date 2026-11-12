#include "engine/generation/mtp/generation.h"
#include "engine/generation/mtp/sequence.h"
#include "engine/generation/mtp/sampling.h"
#include "engine/generation/mtp/state.h"
#include "engine/generation/sampling.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/mtp.h"
#include "engine/models/source.h"
#include "engine/weights/cache.h"
#include "engine/budget.h"
#include "engine/executor.h"
#include "engine/ops.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi::engine {
namespace {
struct MtpOps {
    using Tensor = ::tyloi::Tensor;
    const models::CausalModel& model;
    models::State& state;
    Executor& executor;
    Executor& predictor_execution;
    weights::Cache& weights;
    Budget& budget;
    const ModelSource* source;
    models::Info info;
    models::Mtp& mtp;
    models::MtpState& predictor;
    mtp::State& mtp_state;
    const models::Prepared* prepared;
    Device device;
    models::ForwardOptions options;
    bool profile;
    Tensor input_ids, counts, known_ids_host, known_ids_device;
    Tensor projected_predictor_hidden, projected_predictor_logits;
    int32_t projected_predictor_greedy = -1;
    std::optional<Sampler> sampler_;
    bool penalties = false;
    std::uint64_t target_graph_captures = 0, target_transfer_bytes = 0;
    std::array<double, 6> target_profile{};
    weights::Cache::Routes target_routes;

    std::array<double, 6> target_costs() const {
        std::array<double, 6> result{};
        for (const auto& [name, value] : weights.metrics()) {
            if (name == "expert_route_wait_ms") result[0] = value;
            else if (name == "expert_host_fetch_ms") result[1] = value;
            else if (name == "expert_host_copy_ms") result[2] = value;
            else if (name == "ssd_expert_read_wait_ms") result[3] = value;
        }
        if (source) {
            const auto io = source->file_read_stats();
            result[4] = double(io.read_nanoseconds) / 1e6;
            result[5] = double(io.physical_bytes) / 1048576.0;
        }
        return result;
    }
    void begin_target_routes(int rows) { target_routes.begin(weights, rows); }
    void resolve_target_routes(int rows, bool cancelled) { target_routes.resolve(rows, cancelled); }
    bool take_target_maintenance() {
        const auto captures = weights.activity().graph_captures;
        const bool changed = captures != target_graph_captures;
        target_graph_captures = captures;
        return changed;
    }
    bool retains_partial_target_prefix(int accepted) const {
        return state.retains_partial_prefix(accepted) || mtp.retains_partial_target_prefix();
    }
    void rollback_target(std::int64_t accepted) { state.rollback_speculative(accepted); }
    void initialize(const std::vector<int64_t>& prompt, const MfqSamplingParams& sampling) {
        input_ids = ids_for(prompt);
        sampler_.emplace(sampling, device);
        penalties = sampler().has_penalties();
        if (penalties) {
            counts = ops::zeros({info.vocab_size}, TensorOptions().device(device).dtype(kInt32));
            ops::add_token_counts(counts, input_ids);
        }
    }
    Sampler& sampler() { return *sampler_; }
    static void add_counts(Tensor counts, const Tensor& ids) { ops::add_token_counts(counts, ids); }
    static bool defined(const Tensor& value) { return value.defined(); }
    static int64_t size(const Tensor& value, int axis) { return value.size(axis); }
    static int rank(const Tensor& value) { return value.dim(); }
    Tensor clone(const Tensor& value) const { return value.is_cpu() ? ops::copy(value, device) : ops::clone(value); }
    static Tensor slice(const Tensor& value, int axis, int64_t offset, int64_t length) {
        return ops::contiguous(ops::slice(value, axis, offset, offset + length));
    }
    static Tensor reshape(Tensor value, std::initializer_list<int64_t> shape) { return ops::reshape(value, shape); }
    static Tensor concatenate(const std::vector<Tensor>& values, int axis) {
        if (std::any_of(values.begin(), values.end(), [](const auto& value) { return value.is_cpu(); })) {
            auto host = values;
            for (auto& value : host) value = ops::copy(value, Device(kCPU));
            return ops::contiguous(ops::cat(host, axis));
        }
        return ops::contiguous(ops::cat(values, axis));
    }
    Tensor forward(models::Input input) {
        budget.reserve(state.growth(input, options));
        auto scope = executor.enter();
        auto hidden = model.forward(std::move(input), state, options);
        if (!hidden.defined()) throw std::runtime_error("model forward returned no hidden tensor");
        return hidden;
    }
    Tensor target_forward(Tensor ids, Tensor* raw = nullptr, int64_t confirmed = 0) {
        target_transfer_bytes = 0;
        const auto before = weights.activity().streamed_bytes;
        const auto costs = profile ? target_costs() : std::array<double, 6>{};
        models::Input input;
        input.ids = std::move(ids); input.raw_hidden = raw; input.confirmed = confirmed;
        auto hidden = forward(std::move(input));
        target_transfer_bytes = weights.activity().streamed_bytes - before;
        if (profile) {
            const auto after = target_costs();
            for (std::size_t i = 0; i < after.size(); ++i) target_profile[i] = after[i] - costs[i];
        }
        return hidden;
    }
    Tensor target_suffix(Tensor ids, Tensor* raw) {
        if (!model.supports_suffix_speculation()) throw std::runtime_error("model does not support speculative suffix forwarding");
        const auto before = weights.activity().streamed_bytes;
        const auto costs = profile ? target_costs() : std::array<double, 6>{};
        models::Input input;
        input.ids = std::move(ids); input.raw_hidden = raw; input.suffix = true;
        budget.reserve(state.growth(input, options));
        state.begin_speculative_suffix(input.shape().second);
        auto scope = executor.enter();
        auto hidden = model.forward(std::move(input), state, options);
        if (!hidden.defined()) throw std::runtime_error("model forward returned no hidden tensor");
        target_transfer_bytes += weights.activity().streamed_bytes - before;
        if (profile) {
            const auto after = target_costs();
            for (std::size_t i = 0; i < after.size(); ++i) target_profile[i] += after[i] - costs[i];
        }
        return hidden;
    }
    struct PrefillResult { Tensor hidden, raw; double elapsed; };
    PrefillResult prefill(PrefillChunk chunk) {
        auto timer = executor.timer();
        Tensor raw;
        auto ids = slice(input_ids, 1, chunk.offset, chunk.count);
        Tensor hidden;
        if (prepared && prepared->transformed()) {
            models::Input input;
            input.ids = std::move(ids);
            input.embeddings = slice(prepared->embeddings, 1, chunk.offset, chunk.count);
            if (prepared->positions) input.positions = slice(prepared->positions, -1, chunk.offset, chunk.count);
            input.raw_hidden = &raw; input.advance = true;
            hidden = forward(std::move(input));
        } else hidden = target_forward(std::move(ids), &raw);
        timer->stop();
        return {std::move(hidden), std::move(raw), timer->elapsed_ms()};
    }
    Tensor ids_for(std::vector<int64_t> tokens) {
        known_ids_host = ops::reshape(ops::tensor(tokens, Device(kCPU)), {1, -1});
        known_ids_device = ops::copy(known_ids_host, device);
        return known_ids_device;
    }
    Tensor decode_positions(int64_t cache_start, int64_t tokens) {
        if (tokens <= 0) throw std::runtime_error("MTP position span must be positive");
        const bool transformed = prepared && prepared->transformed();
        const int64_t start = cache_start + (transformed ? prepared->decode_position_delta : 0);
        if (start < 0 || start + tokens > info.max_context) throw std::runtime_error("MTP position span exceeds context capacity");
        auto result = ops::arange(start, start + tokens, device);
        if (transformed && prepared->positions && prepared->positions.dim() == 2 && prepared->positions.size(0) == 3)
            return ops::contiguous(ops::expand(ops::reshape(result, {1, tokens}), {3, tokens}));
        return result;
    }
    models::MtpStep predictor_step(Tensor hidden, Tensor ids, Tensor positions) {
        auto predictor_options = options;
        predictor_options.kv_capacity = ops::kv_capacity(device);
        predictor_options.serial_experts = false;
        models::MtpStep result;
        {
            auto scope = ops::dense_scope(&predictor_execution, device);
            result = mtp.step(predictor, hidden, ids, positions, predictor_options);
        }
        projected_predictor_logits = result.sample_logits;
        projected_predictor_greedy = result.greedy_token;
        if (projected_predictor_logits.defined())
            projected_predictor_hidden = slice(result.sample_hidden, 1, result.sample_hidden.size(1) - 1, 1);
        else projected_predictor_hidden = {};
        return result;
    }
    int64_t predictor_prefill_chunk_size(int64_t requested) const { return std::min<int64_t>(512, requested); }
    double end_predictor_prefill() { weights.prepare(weights::Cache::Stage::decode, device, source); return 0; }
    double predictor_prefill(Tensor hidden, Tensor ids, Tensor positions) {
        auto timer = executor.timer();
        hidden = ops::copy(hidden, device);
        (void)predictor_step(std::move(hidden), std::move(ids), std::move(positions));
        timer->stop();
        return timer->elapsed_ms();
    }
    void trace_prompt_prefill(double target_ms, double predictor_ms, double workspace_ms) const {
        if (!mtp.teacher_forced_prompt_prime()) weights.prepare(weights::Cache::Stage::decode, device, source);
        std::cerr << "runtime_mtp_prefill target_ms=" << target_ms << " predictor_ms=" << predictor_ms
                  << " predictor_workspace_ms=" << workspace_ms << " streamed=0" << std::endl;
    }
    void trace_cycle(uint64_t cycle, int depth, int accepted, double draft_ms, double verify_ms, double commit_ms) {
        if (profile || (cycle <= 32 && draft_ms + verify_ms + commit_ms >= 55.0)) {
            std::cerr << (profile ? "runtime_mtp_cycle_profile cycle=" : "runtime_mtp_slow_cycle cycle=") << cycle
                      << " depth=" << depth << " accepted=" << accepted << " draft_ms=" << draft_ms
                      << " verify_ms=" << verify_ms << " commit_ms=" << commit_ms
                      << " target_transfer_mb=" << target_transfer_bytes / 1048576.0;
            if (profile) std::cerr << " target_route_ms=" << target_profile[0] << " target_fetch_ms=" << target_profile[1]
                << " target_host_copy_ms=" << target_profile[2] << " target_ssd_wait_ms=" << target_profile[3]
                << " target_parallel_read_ms=" << target_profile[4] << " target_physical_read_mb=" << target_profile[5];
            std::cerr << '\n';
        }
    }
    static bool same_data(const Tensor& a, const Tensor& b) {
        if (!a || !b || a.numel() != b.numel()) return false;
        const auto x = a.buffers(), y = b.buffers();
        return x.size() == 1 && y.size() == 1 && x[0] && y[0] && x[0]->data_ptr() == y[0]->data_ptr();
    }
    static bool same_view(const Tensor& a, const Tensor& b) {
        if (!same_data(a, b) || a.dim() != b.dim()) return false;
        for (int i = 0; i < a.dim(); ++i) if (a.size(i) != b.size(i)) return false;
        return true;
    }
    int32_t sample_normal(Tensor logits, Tensor token_counts) {
        if (sampler().greedy() && !penalties && projected_predictor_greedy >= 0 && same_data(logits, projected_predictor_logits))
            return projected_predictor_greedy;
        if (penalties) logits = ops::clone(logits);
        return static_cast<int32_t>(ops::item(sample_logits(sampler(), std::move(logits), token_counts)));
    }
    std::vector<int32_t> greedy_tokens(Tensor logits) {
        if (penalties || !sampler().greedy() || logits.dim() != 2) throw std::runtime_error("batched greedy MTP sampling geometry is invalid");
        const auto tokens = ops::read_integers(ops::sample_greedy(ops::contiguous(logits)));
        return {tokens.begin(), tokens.end()};
    }
    int32_t sample_constrained(Tensor logits, Tensor token_counts, const MfqTokenConstraintPtr& constraint) {
        if (!constraint && sampler().greedy() && !penalties) return sample_normal(logits, token_counts);
        if (penalties) logits = ops::clone(logits);
        auto cursor = constraint ? constraint->clone() : MfqTokenConstraintPtr{};
        return static_cast<int32_t>(ops::item(sample_logits(sampler(), std::move(logits), token_counts, cursor)));
    }
    Tensor probability_logits(Tensor logits, Tensor token_counts, const MfqSamplingParams& parameters) {
        logits = ops::reshape(ops::contiguous(logits), {1, -1});
        if (penalties) {
            logits = ops::clone(logits);
            ops::apply_penalties(logits, token_counts, parameters.presence_penalty, parameters.frequency_penalty, parameters.repetition_penalty);
        }
        return logits;
    }
    std::vector<float> probabilities(Tensor logits, Tensor token_counts, const MfqSamplingParams& parameters) {
        return mtp::probabilities(probability_logits(std::move(logits), std::move(token_counts), parameters), parameters);
    }
    mtp::CompactDistribution compact_probabilities(Tensor logits, Tensor token_counts, const MfqSamplingParams& parameters) {
        if (parameters.top_k <= 0 || parameters.top_k > 64 || !(parameters.temperature > 0.0)) throw std::runtime_error("compact MTP sampling requires top-k 1-64");
        return std::move(mtp::compact_rows(probability_logits(std::move(logits), std::move(token_counts), parameters), parameters).front());
    }
    std::vector<mtp::CompactDistribution> compact_probability_rows(Tensor logits, const MfqSamplingParams& parameters) {
        if (penalties || logits.dim() != 2 || parameters.top_k <= 0 || parameters.top_k > 64 || !(parameters.temperature > 0.0))
            throw std::runtime_error("batched compact MTP sampling geometry is invalid");
        return mtp::compact_rows(ops::contiguous(logits), parameters);
    }
    Tensor logits_for(Tensor normalized) {
        if (same_view(projected_predictor_hidden, normalized)) return projected_predictor_logits;
        auto input = ops::contiguous(mtp.preserve_output_dtype() ? normalized : ops::cast(normalized, kFloat16));
        auto scope = ops::dense_scope(nullptr, device);
        auto result = model.logits(input, true);
        if (!result.defined()) throw std::runtime_error("model logits returned no tensor");
        return result;
    }
};
}

Generation generate_mtp(const models::CausalModel& model, models::State& state,
    Executor& executor, Executor& predictor_execution, weights::Cache& weights, Budget& budget,
    const ModelSource* source, mtp::State& mtp_state, InferenceRequest& request, InferenceOutput& output,
    std::int64_t prefill_chunk_size, const models::Prepared* prepared, std::size_t reused_tokens,
    Tensor restored_last_hidden, Tensor* session_last_hidden, std::size_t stable_prefix_tokens,
    Device device, models::ForwardOptions options, bool profile) {
    auto* predictor = model.predictor();
    if (!model.supports_speculation() || !predictor) throw std::invalid_argument("this model has no speculative predictor");
    if (!state.predictor || !state.predictor->belongs_to(*predictor))
        throw std::invalid_argument("predictor state is unavailable or belongs to another model");
    if (prepared && prepared->image_mask && prepared->image_mask.defined())
        throw std::invalid_argument("MTP image participation masks are unsupported");
    MtpOps ops{model, state, executor, predictor_execution, weights, budget, source, model.info(),
        *predictor, *state.predictor, mtp_state, prepared, device, options, profile};
    ops.target_graph_captures = weights.activity().graph_captures;
    return speculative_sequence(std::move(ops), request, output, prefill_chunk_size, reused_tokens,
        std::move(restored_last_hidden), session_last_hidden, stable_prefix_tokens);
}
}
