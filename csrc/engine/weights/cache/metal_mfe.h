#pragma once

#include "engine/backends/metal/ops/mfq/mlx_moe.h"

namespace tyloi::metal {

struct MlxMfeCachedProjection;

struct MlxMfeProjectionInfo {
    int experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<std::int32_t> available_experts;
};

struct MlxMfeCacheReuseStats {
    std::size_t expert_loads = 0;
    std::size_t expert_hits = 0;
    std::size_t packed_hits = 0;
    std::size_t load_waits = 0;
    std::size_t bytes_read = 0;
    std::size_t read_calls = 0;
    std::size_t evictions = 0;
    double io_seconds = 0;
    double wait_seconds = 0;
    double prepare_seconds = 0;
    double view_seconds = 0;
    std::size_t cached_route_hits = 0;
    std::size_t cached_route_misses = 0;
    std::size_t overlapped_read_batches = 0;
    std::size_t overlapped_read_submissions = 0;
    std::size_t parallel_body_read_batches = 0;
};

// Optional bounded per-expert residency for MFE records.
//
// This is an offload policy, not the MFE container itself. Full-resident
// MlxMfeWeight is the default model path. The cache is constructed only
// after an explicit disk-offload request. Apple Silicon has unified memory,
// so there is no separate CUDA-style CPU-RAM versus device-RAM mode. Disk
// backing uses read_range(); the LRU limit accounts for packed expert bytes
// staged in unified memory for active dispatches.
class MlxMfeOffloadCache {
public:
    MlxMfeOffloadCache(
        const tyloi::source::Source& source,
        std::size_t cache_limit_bytes,
        int experts = 0);
    MlxMfeOffloadCache(
        const MfqContainer& model,
        std::size_t cache_limit_bytes,
        int experts = 0);
    virtual ~MlxMfeOffloadCache();

    MlxMfeOffloadCache(
        const MlxMfeOffloadCache&) = delete;
    MlxMfeOffloadCache& operator=(
        const MlxMfeOffloadCache&) = delete;

    // Returns false only for a valid non-streamable representation (for
    // example legacy NINTv1, dense cohorts, or unsupported VQ layouts).
    // Malformed streamable records still raise.
    virtual bool can_offload(const std::string& name);
    virtual bool can_stream(const std::string& name) {
        return can_offload(name);
    }
    virtual bool can_group_mfe(const std::string& name);
    virtual void activate_record(const std::string& name);
    virtual void prepare_projection(
        const std::string& name, bool shared_input, bool fused_swiglu = false);
    virtual void prepare_swiglu_pair(
        const std::string& gate, const std::string& up, int routes);
    virtual std::size_t backing_payload_bytes() const;

    virtual MlxMfeProjectionInfo projection_info(
        const std::string& name);
    virtual std::vector<std::uint8_t> availability(
        const std::string& name);

    // Materialize only the requested experts from a native MFE record and
    // return one ordinary heterogeneous execution weight whose local expert
    // order matches active_experts. NINTv2, NVQ1-S/L, NVQ-JSC, MXFP4, and aligned
    // MXFP8 reuse their existing packed kernels after paging.
    virtual MlxMfeWeight grouped_mfe(
        const std::string& name,
        const std::vector<std::int32_t>& active_experts,
        std::span<const std::uint32_t> activation_counts = {},
        bool prefill_scan = false);

    virtual std::vector<MlxMfeWeight> grouped_mfe(
        std::span<const std::string> names,
        const std::vector<std::int32_t>& active_experts,
        std::span<const std::uint32_t> activation_counts = {},
        bool prefill_scan = false,
        const std::function<void(const std::vector<MlxMfeWeight>&)>& prefix_ready = {});

    virtual std::optional<std::vector<MlxMfeWeight>> layer_mfe(
        std::span<const std::string> names,
        std::span<const std::int32_t> active_experts = {});

    virtual std::shared_ptr<const MlxMfeCachedProjection> cached_projection(const std::string& name);
    virtual bool resolve_cached_projections(std::span<const std::shared_ptr<const MlxMfeCachedProjection>> views,
        std::span<const std::int32_t> experts);

    virtual std::size_t cache_limit_bytes() const noexcept;
    virtual std::size_t resident_packed_bytes() const;
    virtual std::size_t cached_expert_count() const;
    virtual MlxMfeCacheReuseStats reuse_stats() const;
    virtual std::size_t set_cache_limit(std::size_t bytes);
    virtual void set_prefill_priority(std::span<const std::string> names);

    // Atomically forget one parsed projection and all of its resident
    // experts.  Already returned routed weights retain their own Metal
    // arrays and remain executable.
    virtual void discard_record(
        const std::string& name) noexcept;
    virtual void clear();

private:
    MlxMfeOffloadCache(const tyloi::source::Source& source, std::size_t limit, int experts,
        std::function<void(const std::string&)> prepared);
    MlxMfeOffloadCache(std::shared_ptr<MfqContainer> model, std::size_t limit, int experts);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct MlxMfeCachedProjection {
    MlxMfeWeight weight;
    mlx::core::array expert_map;
    std::vector<std::int32_t> host_expert_map;
    std::string name;
    std::size_t metadata_bytes = 0;
};

struct MlxMfeCachedRoute {
    std::shared_ptr<MlxMfeOffloadCache> cache;
    std::vector<std::shared_ptr<const MlxMfeCachedProjection>> projections;
    std::optional<mlx::core::array> ids;

    mlx::core::array forward(const mlx::core::array& source,
        const mlx::core::array& routes, float swiglu_limit = 0.0f,
        const mlx::core::array* reduce_weights = nullptr);
    std::span<const std::int32_t> host_ids();
    bool all_resident();
    bool resolve();
};

} // namespace tyloi::metal
