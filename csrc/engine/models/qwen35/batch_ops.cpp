#include "engine/models/qwen35/batch.h"
#include "engine/models/common/check.h"
#include "engine/ops/ops.h"
#include "engine/backends/cuda/executor/graph/window.h"
#include "engine/backends/cuda/executor/graph/decode.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/attention/scratch.h"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::models::qwen35 {
namespace {

void add(BatchGrowth& result, const BatchGrowth& value) {
    for (const auto& [device, bytes] : value) {
        auto it = std::find_if(result.begin(), result.end(), [&](const auto& item) { return item.first == device; });
        if (it == result.end()) result.emplace_back(device, bytes);
        else {
            if (it->second > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::overflow_error("Qwen batch growth overflow");
            it->second += bytes;
        }
    }
}

class PublicBatch;

class Request final : public models::BatchState {
public:
    Request(PublicBatch& owner, std::int64_t prompt) : owner_(owner), prompt_(prompt) {}
    std::int64_t position() const override { return position_; }
    std::size_t resident_bytes() const override;
    BatchGrowth initial_growth_bytes() const override;
    void initialize() override;
    BatchGrowth prefill_growth_bytes(std::int64_t tokens) const override;
    std::vector<Tensor> storage() const override;
    PublicBatch& owner_;
    std::int64_t prompt_, position_ = 0;
    RequestState cache_;
};

class PublicBatch final : public models::Batch {
    struct GraphPlan final : Plan { std::int64_t length = 0; };
public:
    PublicBatch(const Model& model, BatchConfig config)
        : model_(model), config_(std::move(config)), options_{config_.slots, config_.paged,
            config_.preallocate, config_.cpu_offload, config_.cached_slots} {
        require_model(config_.slots > 0 && config_.slots <= 32,
            "Qwen continuous batching supports 1..32 requests");
    }
    BatchGrowth initial_growth_bytes() const override { return Batch::construction_growth(model_, options_); }
    void initialize() override { state_ = std::make_unique<Batch>(model_, model_.recurrent_transposed, options_); }
    std::unique_ptr<models::BatchState> make_state(std::int64_t prompt) override {
        require_model(state_ && prompt > 0, "invalid Qwen batch prompt");
        return std::make_unique<Request>(*this, prompt);
    }
    BatchGrowth activate_growth(const models::BatchState& value) const override {
        return state_->activate_growth(request(value).cache_, config_.forward);
    }
    void activate(models::BatchState& value) override { state_->activate(request(value).cache_); }
    void release(models::BatchState& value) override { clear(); state_->release(request(value).cache_); }
    void resume(std::int64_t position) override { state_->resume_decode(position); }
    void retire(std::int64_t position) override { state_->finish_retire(position); }
    void recover() override { state_->finish_retire(0); }
    BatchGrowth decode_growth(std::span<const BatchRow> rows) const override;
    void prepare_decode(std::span<const BatchRow> rows) override;
    std::size_t prefill_row_bytes() const override {
        return std::size_t(model_.metadata.hidden_size) * model_.metadata.hc_mult * sizeof(float) * 8;
    }
    StepSequence<std::vector<Tensor>> prefill(std::vector<models::BatchState*> states,
        std::vector<Tensor> ids) override;
    Tensor decode(std::span<const BatchRow> rows, Tensor ids, const Tensor& host_ids) override;
    StepSequence<std::unique_ptr<Plan>> plan(std::span<const BatchRow>, bool) override;
    StepSequence<std::optional<Tensor>> capture(std::span<const BatchRow>, const Tensor&,
        const Tensor&, bool, const Plan&) override;
    void clear() override;
    engine::Metrics metrics() const override;

    Request& request(models::BatchState& value) const {
        auto* result = dynamic_cast<Request*>(&value);
        require_model(result && &result->owner_ == this, "Qwen batch state belongs to another batch");
        return *result;
    }
    const Request& request(const models::BatchState& value) const {
        auto* result = dynamic_cast<const Request*>(&value);
        require_model(result && &result->owner_ == this, "Qwen batch state belongs to another batch");
        return *result;
    }
    const Model& model_;
    BatchConfig config_;
    Batch::Options options_;
    std::unique_ptr<Batch> state_;
    std::unique_ptr<cuda::DecodeStream> stream_;
    std::unique_ptr<cuda::DecodeWindow> window_;
    Tensor graph_metadata_, graph_output_;
    std::int64_t graph_length_ = 0;
    std::int64_t captures_ = 0, replays_ = 0;
};

std::size_t Request::resident_bytes() const {
    std::size_t result = 0;
    if (cache_.initialized()) for (const auto& tensor : cache_.state().storage()) {
        const auto bytes = std::size_t(tensor.numel()) * tensor.element_size();
        if (result > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("Qwen batch resident size overflow");
        result += bytes;
    }
    return result;
}
BatchGrowth Request::initial_growth_bytes() const { return owner_.state_->request_growth(cache_, prompt_); }
void Request::initialize() { owner_.state_->prepare_prefill(cache_, 0, prompt_); }
BatchGrowth Request::prefill_growth_bytes(std::int64_t tokens) const {
    if (!cache_.initialized()) return {};
    auto result = cache_.state().growth_for(1, tokens, owner_.config_.forward, owner_.options_.paged);
    add(result, owner_.state_->request_growth(cache_, prompt_));
    return result;
}
std::vector<Tensor> Request::storage() const {
    return cache_.initialized() ? cache_.state().storage() : owner_.state_->state().storage();
}

BatchGrowth PublicBatch::decode_growth(std::span<const BatchRow> rows) const {
    std::vector<std::pair<const RequestState*, std::int64_t>> requests;
    std::size_t paused = 0;
    for (const auto& row : rows) {
        const auto& item = request(*row.state);
        requests.emplace_back(&item.cache_, row.position + 1);
        if (!row.eligible) ++paused;
    }
    auto result = state_->decode_growth(requests);
    if (paused) for (const auto& layer : state_->state().layers) {
        if (const auto* linear = std::get_if<LinearState>(&layer); linear && linear->mixer.defined()) {
            for (const auto& tensor : {linear->mixer.view().convolution, linear->mixer.view().recurrent}) {
                const auto bytes = std::size_t(tensor.numel()) / config_.slots * paused * tensor.element_size();
                add(result, {{tensor.device(), bytes * 2}});
            }
        }
    }
    return result;
}

void PublicBatch::prepare_decode(std::span<const BatchRow> rows) {
    std::int64_t maximum = 0;
    for (const auto& row : rows) {
        auto& item = request(*row.state);
        require_model(item.position_ == row.position, "Qwen batch position mismatch");
        state_->ensure_decode_tokens(item.cache_, row.position + 1);
        maximum = std::max(maximum, row.position);
    }
    state_->prepare_decode(maximum);
}

StepSequence<std::vector<Tensor>> PublicBatch::prefill(std::vector<models::BatchState*> states,
        std::vector<Tensor> ids) {
    require_model(states.size() == 1 && ids.size() == 1, "Qwen batch prefill requires one request");
    auto& item = request(*states.front());
    require_model(item.cache_.initialized() && item.position_ == item.cache_.state().position(),
        "Qwen batch prefill state mismatch");
    state_->prepare_prefill(item.cache_, item.position_, item.prompt_);
    models::Input input;
    input.ids = std::move(ids.front());
    auto output = model_.forward(std::move(input), item.cache_.state(), config_.forward);
    item.position_ = item.cache_.state().position();
    co_yield std::vector<Tensor>{std::move(output)};
}

Tensor PublicBatch::decode(std::span<const BatchRow> rows, Tensor, const Tensor&) {
    const auto count = std::int64_t(config_.slots);
    std::vector<std::int64_t> values(std::size_t(3 * count), 0);
    std::fill(values.begin() + 2 * count, values.end(), 1);
    std::vector<std::int32_t> paused;
    std::int64_t maximum = 0;
    for (const auto& row : rows) {
        auto& item = request(*row.state);
        const auto slot = item.cache_.slot();
        require_model(slot >= 0 && slot < count, "Qwen batch lost its stable slot");
        values[slot] = row.token;
        values[count + slot] = row.position;
        values[2 * count + slot] = row.position + 1;
        maximum = std::max(maximum, row.position + 1);
        if (!row.eligible) paused.push_back(slot);
    }
    auto metadata = ops::reshape(ops::tensor(values, model_.device), {3, count});
    models::Input input;
    input.ids = ops::reshape(ops::slice(metadata, 0, 0, 1), {count, 1});
    input.positions = ops::reshape(ops::slice(metadata, 0, 1, 2), {count, 1});
    input.cache_positions = input.positions;
    input.lengths = ops::reshape(ops::slice(metadata, 0, 2, 3), {count});
    auto saved = state_->capture_recurrent_slots(paused);
    auto forward = config_.forward;
    forward.serial_experts = state_->has_cached_moe() && count > 1;
    auto hidden = model_.forward(std::move(input), state_->state(), forward);
    auto logits = model_.logits(ops::contiguous(ops::select(hidden, 1, -1)));
    state_->finish_decode(maximum);
    state_->restore_recurrent_slots(paused, saved);
    std::vector<Tensor> selected;
    for (const auto& row : rows) if (row.eligible) {
        auto& item = request(*row.state);
        selected.push_back(ops::slice(logits, 0, item.cache_.slot(), item.cache_.slot() + 1));
        ++item.position_;
    }
    return ops::cat(selected, 0);
}

StepSequence<std::unique_ptr<models::Batch::Plan>> PublicBatch::plan(
        std::span<const BatchRow> rows, bool greedy) {
    if (!config_.graph || !greedy || config_.slots < 2 || state_->has_cached_moe() || rows.empty() ||
        !mfq_cuda_graph_capture_supported()) co_return;
    std::int64_t requested = 0, remaining = std::numeric_limits<std::int64_t>::max();
    for (const auto& row : rows) {
        requested = std::max(requested, row.position + row.remaining);
        remaining = std::min(remaining, row.remaining);
    }
    const auto length = ::decode_graph_bucket(requested, model_.metadata.max_position_embeddings);
    const auto owners = state_->state().storage();
    const bool reuse = window_ && graph_length_ == length && window_->matches(owners);
    if (!reuse && remaining < config_.minimum) co_return;
    co_yield GraphPhase::ready;
    auto result = std::make_unique<GraphPlan>();
    result->length = length;
    result->rebuild = !reuse;
    if (!reuse) {
        result->growth = cuda::DecodeWindow::fixed_growth(model_.device, false, false, 0);
        add(result->growth, {{model_.device, 3 * config_.slots * sizeof(std::int64_t)}});
        add(result->growth, {{Device(kCPU), 3 * config_.slots * sizeof(std::int64_t)}});
        for (const auto* tensor : state_->state().graph_state())
            add(result->growth, {{tensor->device(), tensor->nbytes()}});
    }
    co_yield std::unique_ptr<models::Batch::Plan>(std::move(result));
}

StepSequence<std::optional<Tensor>> PublicBatch::capture(std::span<const BatchRow> rows,
        const Tensor&, const Tensor&, bool, const models::Batch::Plan& value) {
    const auto& plan = static_cast<const GraphPlan&>(value);
    bool prepared = false;
    if (plan.rebuild) co_yield GraphRequest{GraphPhase::prepare, &prepared};
    if (!stream_) stream_ = std::make_unique<cuda::DecodeStream>();
    auto stream_scope = stream_->enter();
    const auto count = std::int64_t(config_.slots);
    if (plan.rebuild) {
        window_.reset();
        graph_metadata_ = {}; graph_output_ = {};
        window_ = std::make_unique<cuda::DecodeWindow>(false, true);
        if (!prepared) window_->prepare_pool();
        graph_metadata_ = window_->buffer({3, count}, TensorOptions().device(model_.device).dtype(kInt64));
        graph_length_ = plan.length;
    }
    std::vector<std::int64_t> values(std::size_t(3 * count), 0);
    std::fill(values.begin() + 2 * count, values.end(), 1);
    std::vector<std::int32_t> paused;
    std::int64_t maximum = 0;
    for (const auto& row : rows) {
        const auto& item = request(*row.state);
        const auto slot = item.cache_.slot();
        require_model(slot >= 0 && slot < count, "Qwen batch lost its stable slot");
        values[slot] = row.token;
        values[count + slot] = row.position;
        values[2 * count + slot] = row.position + 1;
        maximum = std::max(maximum, row.position + 1);
        if (!row.eligible) paused.push_back(slot);
    }
    auto host = ops::reshape(ops::tensor(values, Device(kCPU)), {3, count});
    ops::copy_into(graph_metadata_, host);
    auto paused_state = state_->capture_recurrent_slots(paused);
    if (!window_->valid()) {
        struct Saved { Tensor* target; Tensor copy; };
        std::vector<Saved> saved;
        for (auto* tensor : state_->state().graph_state()) saved.push_back({tensor, ops::clone(*tensor)});
        try {
            auto recording = window_->begin_capture();
            while (recording->next()) {
                std::exception_ptr failure;
                bool warm = false;
                try {
                    auto policy = cuda::Executor::current_ffn();
                    policy.serial = true; policy.projection_major = true;
                    auto execution = cuda::Executor::current().enter(policy);
                    models::Input input;
                    input.ids = ops::reshape(ops::slice(graph_metadata_, 0, 0, 1), {count, 1});
                    input.positions = ops::reshape(ops::slice(graph_metadata_, 0, 1, 2), {count, 1});
                    input.cache_positions = input.positions;
                    input.lengths = ops::reshape(ops::slice(graph_metadata_, 0, 2, 3), {count});
                    input.kv_length = plan.length;
                    input.attention_parts = ::decode_graph_attention_parts(plan.length, cuda::AttentionScratch::kMaxParts);
                    auto hidden = model_.forward(std::move(input), state_->state(), config_.forward);
                    auto logits = model_.logits(ops::contiguous(ops::select(hidden, 1, -1)));
                    graph_output_ = ops::reshape(ops::sample_greedy(ops::contiguous(logits)), {count, 1});
                    warm = recording->finish();
                } catch (...) { failure = std::current_exception(); }
                if (failure) {
                    if (!recording->retry(failure)) std::rethrow_exception(failure);
                    warm = true;
                }
                for (const auto& item : saved) ops::copy_into(*item.target, item.copy);
                if (warm) {
                    graph_output_ = {};
                    recording->release_warm_outputs();
                    recording->pause();
                    stream_scope.reset();
                    co_yield GraphRequest{GraphPhase::admit, nullptr, recording->growth()};
                    stream_scope = stream_->enter();
                    recording->resume();
                }
            }
            recording->complete();
            window_->retain_result(graph_output_);
            window_->retain(state_->state().storage());
            ++captures_;
        } catch (...) {
            for (const auto& item : saved) ops::copy_into(*item.target, item.copy);
            clear();
            throw;
        }
    }
    window_->run();
    ++replays_;
    state_->finish_decode(maximum);
    state_->restore_recurrent_slots(paused, paused_state);
    std::vector<Tensor> selected;
    for (const auto& row : rows) if (row.eligible) {
        auto& item = request(*row.state);
        selected.push_back(ops::slice(graph_output_, 0, item.cache_.slot(), item.cache_.slot() + 1));
        ++item.position_;
    }
    auto result = ops::cat(selected, 0);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
    stream_scope.reset();
    co_yield std::optional<Tensor>(std::move(result));
}

void PublicBatch::clear() {
    window_.reset();
    graph_metadata_ = {}; graph_output_ = {}; graph_length_ = 0;
}

engine::Metrics PublicBatch::metrics() const {
    const auto stats = state_->paged_kv_stats();
    return {{"continuous_batching_stable_slot_releases", double(state_->slot_releases())},
        {"continuous_batching_cuda_graph_captures", double(captures_)},
        {"continuous_batching_cuda_graph_replays", double(replays_)},
        {"continuous_batching_moe", double(state_->has_moe())},
        {"continuous_batching_moe_cached_row_serial", double(state_->has_cached_moe())},
        {"continuous_batching_paged_kv", double(state_->paged_kv_enabled())},
        {"paged_kv_page_size", double(stats.page_size)}, {"paged_kv_live_pages", double(stats.live_pages)},
        {"paged_kv_peak_live_pages", double(stats.peak_live_pages)},
        {"paged_kv_capacity_pages", double(stats.capacity_pages)},
        {"paged_kv_reserved_bytes", double(stats.reserved_bytes)},
        {"paged_kv_page_allocations", double(stats.allocations)}, {"paged_kv_page_reuses", double(stats.reuses)},
        {"paged_kv_page_releases", double(stats.releases)}, {"paged_kv_table_updates", double(stats.table_updates)}};
}

} // namespace

std::unique_ptr<models::Batch> Model::make_batch(const BatchConfig& config) const {
    return std::make_unique<PublicBatch>(*this, config);
}
} // namespace tyloi::models::qwen35
