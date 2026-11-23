#pragma once

#include "mlx_qsa_kv_offload.h"
#include "mlx_text_backbone_runtime.h"
#include "mlx_transformer.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace mfq::metal {

class MfqContainer;

struct MlxHy4AttentionCache {
    std::optional<mlx::core::array> window, index_tail;
    std::optional<MlxSequenceCache> latent, groups;
    std::optional<MlxQsaIndexOffload> offloaded_latent, offloaded_groups;
    std::optional<mlx::core::array> last_indices;
};

struct MlxHy4Verification;

struct MlxHy4Cache {
    int position = 0;
    std::vector<MlxHy4AttentionCache> layers;
    std::vector<std::int32_t> engram_tail;
    std::optional<mlx::core::array> last_hidden;
    std::shared_ptr<MlxHy4Verification> pending_verification;
    std::optional<mlx::core::array> pending_logits;
};

struct MlxHy4RowSnapshot {
    std::optional<mlx::core::array> resident;
    std::optional<MlxQsaIndexSnapshot> offloaded;
};

struct MlxHy4LayerSnapshot {
    std::optional<mlx::core::array> window, index_tail, last_indices;
    MlxHy4RowSnapshot latent, groups;
};

struct MlxHy4TextSessionState {
    std::vector<std::int64_t> tokens;
    std::vector<MlxHy4LayerSnapshot> layers;
    int window = 0, group_size = 0;
    std::optional<mlx::core::array> last_hidden, last_logits;
    std::size_t bytes = 0;
};

class MlxHy4Model {
public:
    using Cache = MlxHy4Cache;
    using TextSessionState = MlxHy4TextSessionState;
    static MlxHy4Model load(const MfqContainer& model, int context, const std::string& draft_path = {});

    mlx::core::array forward(const mlx::core::array& tokens, MlxHy4Cache& cache,
        bool last_logit_only = false) const;
    mlx::core::array decode_batch(std::span<Cache* const> requests, const mlx::core::array& tokens) const;
    int context() const noexcept;
    int vocabulary() const noexcept;
    TextSessionState capture(const Cache& cache, const std::vector<std::int64_t>& tokens, bool detached) const;
    Cache restore(const TextSessionState& state) const;
    std::optional<QsaKvStoreStats> qsa_kv_offload_stats() const;
    std::size_t kv_cache_bytes(const Cache& cache) const noexcept;
    bool supports_mtp() const noexcept;
    MlxMtpEngineCallbacks batch_mtp_callbacks(Cache& cache, std::optional<mlx::core::array>& last_logits) const;
    std::vector<mlx::core::array> draft_batch(std::span<Cache* const> requests,
        std::span<const MlxMtpTargetRequest> steps) const;
    std::vector<MlxMtpTargetBatch> verify_batch(std::span<Cache* const> requests,
        std::span<const MlxMtpTargetRequest> steps,
        std::span<std::optional<mlx::core::array>* const> logits) const;
    void discard_batch_verification(Cache& cache, std::optional<mlx::core::array>& logits) const;
    int generate_speculative(Cache& cache, MlxMtpEngineRequest request,
        const MlxPrefixCacheHooks& prefix, MlxMtpGenerationStats& stats,
        std::optional<mlx::core::array>& last_logits) const;

private:
    struct Weights;
    using Verification = MlxHy4Verification;
    struct BatchInput {
        Cache* cache;
        mlx::core::array tokens;
        Verification* verification = nullptr;
    };
    std::vector<mlx::core::array> forward_batch(std::span<const BatchInput> inputs) const;
    void resolve_verification(Cache& cache, const Verification& verification,
        const mlx::core::array& logits, int accepted, int proposed,
        std::optional<mlx::core::array>& last_logits) const;
    mlx::core::array forward_impl(const mlx::core::array& tokens, Cache& cache,
        bool last_logit_only, Verification* verification) const;
    explicit MlxHy4Model(std::shared_ptr<const Weights> weights) : weights_(std::move(weights)) {}
    std::shared_ptr<const Weights> weights_;
};

using MlxHy4Runtime = MlxTextBackboneRuntime<MlxHy4Model>;

}  // namespace mfq::metal
