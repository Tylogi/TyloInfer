#pragma once

#include "engine/models/common/recurrent_attention.h"
#include "engine/models/qwen4_exp/ngram.h"
#include "engine/ops.h"
#include "engine/ops/ple.h"
#include "engine/weights/row_stage.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace tyloi::models::qwen4_exp {

// Hashing is the upstream CPU uint64 algorithm, including EOS boundaries.
class NgramEmbedding {
public:
    using Context = NgramContext;
    struct State {
        State() = default;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        State(State&&) = default;
        State& operator=(State&&) = default;
        Context context;
        std::shared_ptr<weights::RowReader> reader;
        bool hash_ready = false;
        int cached_tokens = 0;
        std::array<int64_t, 8> cached_ids{};
        NgramHashes cached_hashes;
    };

    NgramEmbedding(std::vector<Tensor> shards, int64_t rows, int64_t dimension, int64_t ngram,
        int64_t heads_per_ngram, int64_t eos, std::vector<int64_t> multipliers,
        std::vector<int64_t> offsets, std::vector<int64_t> vocab)
        : shards_(std::move(shards)), rows_(rows), dimension_(dimension), ngram_(ngram),
          heads_(heads_per_ngram), eos_(eos), multipliers_(std::move(multipliers)),
          offsets_(std::move(offsets)), vocab_(std::move(vocab)),
          source_(weights::RowReader::source(shards_, rows)) {
        require_model(!shards_.empty() && rows_ > 0 && dimension_ > 0 && ngram_ > 1 && heads_ > 0 &&
            multipliers_.size() == size_t(ngram_) && offsets_.size() == size_t((ngram_ - 1) * heads_) &&
            vocab_.size() == offsets_.size(), "invalid Qwen4 ngram metadata geometry");
        for (size_t h = 0; h < vocab_.size(); ++h)
            require_model(vocab_[h] > 0 && offsets_[h] >= 0 &&
                offsets_[h] <= int64_t(shards_.size()) * rows_ - vocab_[h],
                "Qwen4 ngram vocabulary lies outside embedding shards");
    }
    void reset(State& state) const { state.context = {}; state.hash_ready = false; }
    bool graph_supported() const { return bool(source_); }
    std::pair<std::size_t, std::size_t> context_growth_bytes(const State& state, int64_t batch, int64_t tokens) const {
        require_model(batch > 0 && tokens > 0 && ngram_ > 1 &&
            tokens <= std::numeric_limits<int64_t>::max() - (ngram_ - 1), "invalid ngram growth geometry");
        const std::array<int64_t, 2> next{batch, ngram_ - 1};
        const std::array<int64_t, 2> history{batch, ngram_ - 1 + tokens};
        const auto retained = make_contiguous_view(nullptr, next, kInt64, {}).nbytes();
        auto temporary = make_contiguous_view(nullptr, history, kInt64, {}).nbytes();
        if (state.context.tokens.size() > (std::numeric_limits<std::size_t>::max() - temporary) / sizeof(int64_t))
            throw std::overflow_error("ngram context growth overflow");
        temporary += state.context.tokens.size() * sizeof(int64_t);
        return {retained, temporary};
    }
    const Context& context(const State& state) const { return state.context; }
    void restore(State& state, Context context) const { state.context = std::move(context); state.hash_ready = false; }
    Context prefix_context(const State& state, const Tensor& host_ids, int64_t count) const {
        require_model(host_ids.is_cpu() && host_ids.dim() == 2 && count > 0 && count <= host_ids.size(1),
            "PLE prefix requires host token IDs");
        const auto host = ops::read_integers(ops::contiguous(ops::slice(host_ids, 1, 0, count)));
        return ngram_hashes(host.data(), host_ids.size(0), count, ngram_, heads_, eos_,
            multipliers_, offsets_, vocab_, state.context, true).next;
    }
    void prefetch(State& state, const Tensor& host_ids, bool early_prefix = false) const {
        if (!source_) return;
        require_model(host_ids.is_cpu() && host_ids.dim() == 2, "PLE prefetch requires host token IDs");
        if (!state.reader) state.reader = std::make_shared<weights::RowReader>(source_);
        const auto host = ops::read_integers(host_ids);
        const auto b = host_ids.size(0), t = host_ids.size(1);
        if (cacheable_hashes(b, t)) {
            const auto& hashes = graph_hashes(state, host);
            state.reader->prefetch(hashes.ids, {b, t, (ngram_ - 1) * heads_}, early_prefix);
        } else {
            auto hashes = ngram_hashes(host.data(), b, t, ngram_, heads_, eos_,
                multipliers_, offsets_, vocab_, state.context, true);
            state.reader->prefetch(hashes.ids, {b, t, (ngram_ - 1) * heads_}, early_prefix);
        }
    }
    std::shared_ptr<RowStage> graph_stage(State& state, int64_t b, int64_t t, const Device& device) const {
        require_model(bool(source_), "PLE graph requires prepared range rows");
        if (!state.reader) state.reader = std::make_shared<weights::RowReader>(source_);
        return state.reader->stage({b, t, (ngram_ - 1) * heads_}, device);
    }
    std::vector<std::pair<Device, std::size_t>> graph_stage_growth(int64_t b, int64_t t, const Device& device) const {
        require_model(bool(source_), "PLE graph requires prepared range rows");
        return weights::RowReader::stage_growth(source_, {b, t, (ngram_ - 1) * heads_}, device);
    }
    void upload_graph(State& state, const Tensor& host_ids, RowStage& stage) const {
        require_model(graph_supported() && host_ids.is_cpu() && host_ids.dim() == 2,
            "PLE graph requires host token IDs");
        const auto host = ops::read_integers(host_ids);
        const auto b = host_ids.size(0), t = host_ids.size(1);
        if (cacheable_hashes(b, t)) {
            auto& hashes = graph_hashes(state, host);
            stage.upload(hashes.ids, {b, t, (ngram_ - 1) * heads_});
            restore(state, std::move(hashes.next));
        } else {
            auto hashes = ngram_hashes(host.data(), b, t, ngram_, heads_, eos_,
                multipliers_, offsets_, vocab_, state.context, true);
            stage.upload(hashes.ids, {b, t, (ngram_ - 1) * heads_});
            restore(state, std::move(hashes.next));
        }
    }
    Tensor forward(State& state, const Tensor& ids, bool cache, Tensor* hashed_ids = nullptr,
                   const Tensor& host_ids = {}) const {
        require_model(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0,
            "Qwen4 ngram IDs must be [B,T]");
        const auto b = ids.size(0), t = ids.size(1), nh = (ngram_ - 1) * heads_;
        if (host_ids) require_model(host_ids.is_cpu() && host_ids.dim() == 2 &&
            host_ids.size(0) == b && host_ids.size(1) == t, "PLE host token mirror differs");
        const auto host = ops::read_integers(host_ids ? host_ids : ids);
        auto hashes = ngram_hashes(host.data(), b, t, ngram_, heads_, eos_,
            multipliers_, offsets_, vocab_, state.context, cache);
        const auto& global = hashes.ids;
        if (hashed_ids) *hashed_ids = ops::reshape(ops::tensor(global, ids.device()), {b, t, nh});
        Tensor result;
        if (source_) {
            if (!state.reader) state.reader = std::make_shared<weights::RowReader>(source_);
            result = state.reader->collect(global, {b, t, nh}, ids.device());
        } else {
            std::set<int64_t> used;
            for (auto id : global) used.insert(id / rows_);
            for (auto shard : used) {
                std::vector<int64_t> local(global.size());
                std::vector<float> mask(global.size());
                for (size_t i = 0; i < global.size(); ++i) {
                    if (global[i] / rows_ == shard) {
                        local[i] = global[i] - shard * rows_;
                        mask[i] = 1.f;
                    }
                }
                auto indices = ops::reshape(ops::tensor(local, ids.device()), {b, t, nh});
                auto values = ops::embedding(indices, shards_.at(shard));
                require_model(values.dim() == 4 && values.size(0) == b && values.size(1) == t &&
                    values.size(2) == nh && values.size(3) == dimension_, "Qwen4 ngram embedding shape mismatch");
                auto select = ops::reshape(ops::tensor(mask, ids.device()), {b, t, nh, 1});
                values = ops::multiply(values, ops::cast(select, values.scalar_type()));
                result = result ? ops::add(result, values) : values;
            }
        }
        if (cache) restore(state, std::move(hashes.next));
        return ops::reshape(result, {b, t, nh * dimension_});
    }

private:
    std::vector<Tensor> shards_;
    int64_t rows_, dimension_, ngram_, heads_, eos_;
    std::vector<int64_t> multipliers_, offsets_, vocab_;
    std::shared_ptr<const weights::RowReader::Source> source_;
    bool cacheable_hashes(int64_t b, int64_t t) const { return b == 1 && t > 0 && t <= 8; }
    NgramHashes& graph_hashes(State& state, const std::vector<int64_t>& ids) const {
        const auto tokens = ids.size();
        if (state.hash_ready && state.cached_tokens == int(tokens) &&
            std::equal(ids.begin(), ids.end(), state.cached_ids.begin())) return state.cached_hashes;
        state.cached_hashes = ngram_hashes(ids.data(), 1, tokens, ngram_, heads_, eos_,
            multipliers_, offsets_, vocab_, state.context, true);
        std::copy(ids.begin(), ids.end(), state.cached_ids.begin());
        state.cached_tokens = int(tokens); state.hash_ready = true;
        return state.cached_hashes;
    }
};

using PleWeights = ops::PleWeights;
class Ple {
public:
    struct State {
        NgramEmbedding::State embedding;
        Tensor conv, saved_conv;
        NgramEmbedding::Context saved_context;
        std::vector<Tensor> saved_prefix_conv;
        std::vector<NgramEmbedding::Context> saved_prefix_context;
        int64_t saved_prefix_position = -1;
        bool saved = false;
    };
    Ple(NgramEmbedding embedding, PleWeights weights, int64_t hidden, int64_t streams,
        int64_t dilation, double eps)
        : embedding_(std::move(embedding)), w_(std::move(weights)), hidden_(hidden),
          streams_(streams), dilation_(dilation), eps_(eps) {}
    void reset(State& state, bool retain_storage = false) const {
        embedding_.reset(state.embedding);
        if (retain_storage && state.conv) ops::fill(state.conv, 0); else state.conv = {};
        commit(state);
    }
    void commit(State& state) const {
        state.saved = false; state.saved_conv = {}; state.saved_context = {};
        state.saved_prefix_conv.clear(); state.saved_prefix_context.clear(); state.saved_prefix_position = -1;
    }
    void rollback(State& state, int64_t keep = -1) const {
        require_model(state.saved, "Qwen4 PLE has no speculative checkpoint");
        if (keep >= 0 && state.saved_prefix_position >= 0) {
            const auto index = keep - state.saved_prefix_position;
            require_model(index >= 0 && index < int64_t(state.saved_prefix_conv.size()),
                "Qwen4 PLE prefix checkpoint is out of range");
            state.saved_conv = state.saved_prefix_conv[index]; state.saved_context = state.saved_prefix_context[index];
        }
        if (state.conv) ops::copy_into(state.conv, state.saved_conv); else state.conv = state.saved_conv;
        embedding_.restore(state.embedding, state.saved_context);
        commit(state);
    }
    const Tensor& conv_state(const State& state) const { return state.conv; }
    std::vector<Tensor*> graph_state(State& state) const { return {&state.conv}; }
    std::pair<std::size_t, std::size_t> ordinary_growth_bytes(const State& state, int64_t batch, int64_t tokens) const {
        if (state.conv) require_model(state.conv.size(0) == batch, "PLE batch changed");
        const auto kernel = w_.conv.size(-1);
        require_model(batch > 0 && tokens > 0 && kernel > 0 && dilation_ > 0 &&
            kernel - 1 <= std::numeric_limits<int64_t>::max() / dilation_, "invalid PLE growth geometry");
        const auto history = (kernel - 1) * dilation_;
        require_model(tokens <= std::numeric_limits<int64_t>::max() - history, "PLE growth overflow");
        const std::array<int64_t, 4> joined{batch, history + tokens, streams_, hidden_};
        const std::array<int64_t, 4> conv{batch, history, streams_, hidden_};
        const auto a = make_contiguous_view(nullptr, joined, kFloat32, w_.conv.device()).nbytes();
        const auto b = make_contiguous_view(nullptr, conv, kFloat32, w_.conv.device()).nbytes();
        if (batch > 1 && a > std::numeric_limits<std::size_t>::max() - b) throw std::overflow_error("PLE growth overflow");
        if (state.conv) return {0, a + (batch > 1 ? b : 0)};
        return batch == 1 ? std::pair{a, b} : std::pair{b, a + b};
    }
    std::size_t checkpoint_growth_bytes(const State& state, int64_t batch, int64_t tokens, int64_t confirmed) const {
        if (tokens <= 8) require_model(bool(state.conv), "PLE verification requires committed convolution state");
        const auto kernel = w_.conv.size(-1);
        require_model(kernel > 0 && dilation_ > 0 && kernel - 1 <= std::numeric_limits<int64_t>::max() / dilation_,
            "invalid PLE checkpoint geometry");
        const auto count = tokens <= 8 ? tokens - confirmed : 1;
        const std::array<int64_t, 4> shape{batch, (kernel - 1) * dilation_, streams_, hidden_};
        const auto bytes = make_contiguous_view(nullptr, shape, kFloat32, w_.conv.device()).nbytes();
        if (bytes && std::size_t(count) > std::numeric_limits<std::size_t>::max() / bytes)
            throw std::overflow_error("PLE checkpoint growth overflow");
        return bytes * std::size_t(count);
    }
    std::pair<std::size_t, std::size_t> context_growth_bytes(const State& state, int64_t batch, int64_t tokens) const {
        return embedding_.context_growth_bytes(state.embedding, batch, tokens);
    }
    bool graph_supported() const { return embedding_.graph_supported(); }
    const NgramEmbedding::Context& session_context(const State& state) const { return embedding_.context(state.embedding); }
    void restore_graph_context(State& state, NgramEmbedding::Context context) const { embedding_.restore(state.embedding, std::move(context)); }
    void restore_session(State& state, const Tensor& convolution, NgramEmbedding::Context context, bool retain_storage = false) const {
        require_model(bool(convolution), "missing Qwen4 PLE session state");
        if (retain_storage) {
            require_model(state.conv && state.conv.dim() == convolution.dim(), "Qwen4 PLE graph session geometry changed");
            for (int64_t i = 0; i < convolution.dim(); ++i)
                require_model(state.conv.size(i) == convolution.size(i), "Qwen4 PLE graph session geometry changed");
            auto prepared = convolution.device() == state.conv.device() ? convolution : ops::copy(convolution, state.conv.device());
            commit(state); ops::copy_into(state.conv, ops::contiguous(prepared));
        } else { reset(state); state.conv = ops::clone(convolution); }
        embedding_.restore(state.embedding, std::move(context));
    }
    std::shared_ptr<RowStage> graph_stage(State& state, int64_t tokens, const Device& device) const {
        return embedding_.graph_stage(state.embedding, 1, tokens, device);
    }
    std::vector<std::pair<Device, std::size_t>> graph_stage_growth(int64_t tokens, const Device& device) const {
        return embedding_.graph_stage_growth(1, tokens, device);
    }
    void upload_graph(State& state, const Tensor& host_ids, RowStage& stage) const { embedding_.upload_graph(state.embedding, host_ids, stage); }
    NgramEmbedding::Context graph_prefix_context(const State& state, const Tensor& host_ids, int64_t count) const {
        return embedding_.prefix_context(state.embedding, host_ids, count);
    }
    void bind_graph_checkpoint(State& state, const std::vector<Tensor>& conv,
        std::vector<NgramEmbedding::Context> context, int64_t position) const {
        require_model(!state.saved, "resolve previous PLE checkpoint");
        require_model(!conv.empty() && conv.size() == context.size(), "invalid PLE prefix checkpoints");
        state.saved = true; state.saved_conv = conv.front(); state.saved_context = context.front();
        state.saved_prefix_conv = conv; state.saved_prefix_context = std::move(context); state.saved_prefix_position = position;
    }
    Tensor forward_graph(State& state, const Tensor& x, const Tensor& ids,
        const Tensor& rows, int64_t confirmed, std::vector<Tensor>& checkpoint_conv) const {
        if (!confirmed) return forward_chunk(state, x, ids, true, {}, rows);
        auto prefix = forward_chunk(state, ops::slice(x, 1, 0, confirmed), ops::slice(ids, 1, 0, confirmed),
            true, {}, ops::slice(rows, 1, 0, confirmed));
        ops::copy_into(checkpoint_conv.front(), state.conv);
        auto suffix = forward_chunk(state, ops::slice(x, 1, confirmed, x.size(1)),
            ops::slice(ids, 1, confirmed, ids.size(1)), true, {}, ops::slice(rows, 1, confirmed, rows.size(1)),
            &checkpoint_conv, 1);
        const std::array<Tensor, 2> parts{prefix, suffix};
        return ops::cat(parts, 1);
    }
    void prefetch_tokens(State& state, const Tensor& ids, bool early_prefix = false) const { embedding_.prefetch(state.embedding, ids, early_prefix); }
    Tensor forward(State& state, const Tensor& x, const Tensor& ids, bool cache,
        int64_t confirmed = 0, const Tensor& host_ids = {}, int64_t cache_position = -1) const {
        require_model(x.dim() == 3 && ids.dim() == 2 && x.size(0) == ids.size(0) &&
            x.size(1) == ids.size(1) && x.size(2) == hidden_ * streams_ && confirmed >= 0 &&
            confirmed <= x.size(1) && (!confirmed || cache), "invalid Qwen4 PLE input");
        if (cache && cache_position >= 0 && confirmed > 0 && confirmed < x.size(1) && x.size(1) <= 8) {
            require_model(!state.saved && state.conv, "PLE verification requires committed convolution state");
            auto host = host_ids ? host_ids : ops::cast(ids, kInt64, Device(kCPU));
            std::vector<Tensor> conv;
            std::vector<NgramEmbedding::Context> context;
            for (int64_t i = confirmed; i < x.size(1); ++i) {
                std::vector<int64_t> shape;
                for (int64_t d = 0; d < state.conv.dim(); ++d) shape.push_back(state.conv.size(d));
                conv.push_back(ops::empty(shape, state.conv.options()));
                context.push_back(embedding_.prefix_context(state.embedding, host, i));
            }
            auto rows = embedding_.forward(state.embedding, ids, true, nullptr, host);
            auto result = forward_graph(state, x, ids, rows, confirmed, conv);
            bind_graph_checkpoint(state, conv, std::move(context), cache_position + confirmed);
            return result;
        }
        return models::recurrent_window(x.size(1), cache, confirmed, state.saved,
            [&](int64_t start, int64_t count) {
                return forward_chunk(state, ops::slice(x, 1, start, start + count),
                    ops::slice(ids, 1, start, start + count), cache,
                    host_ids ? ops::slice(host_ids, 1, start, start + count) : Tensor{});
            },
            [&] { state.saved = true; state.saved_conv = ops::clone(state.conv);
                  state.saved_context = embedding_.context(state.embedding); },
            [&] { rollback(state); },
            [](Tensor first, Tensor second) { const std::array<Tensor, 2> parts{first, second}; return ops::cat(parts, 1); });
    }
    Tensor forward_chunk(State& state, const Tensor& x, const Tensor& ids, bool cache,
        const Tensor& host_ids = {}, const Tensor& graph_embeddings = {},
        std::vector<Tensor>* checkpoints = nullptr, std::size_t checkpoint_offset = 0) const {
        auto context = embedding_.context(state.embedding);
        try {
            auto rows = graph_embeddings ? graph_embeddings : embedding_.forward(state.embedding, ids, cache, nullptr, host_ids);
            return ops::ple(x, rows, w_, state.conv, cache, hidden_, streams_, dilation_, eps_, checkpoints, checkpoint_offset);
        } catch (...) {
            embedding_.restore(state.embedding, std::move(context));
            throw;
        }
    }
private:
    NgramEmbedding embedding_;
    PleWeights w_;
    int64_t hidden_, streams_, dilation_;
    double eps_;
};

} // namespace tyloi::models::qwen4_exp
