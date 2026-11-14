#pragma once

#include "mlx_tq.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tyloi::source { class Source; }
namespace tyloi::metal {
namespace tq_cache_detail { struct LegacyPlan; }

struct MlxTqExpertWeights {
    MlxTqWeight gate, up, down;
    void prepare(int top_k, float limit = 0.f, bool gelu = false) const;
    std::optional<mlx::core::array> reuse_plan(const mlx::core::array& ids) const;
    mlx::core::array routed_ffn(const mlx::core::array& input, const mlx::core::array& ids,
        const mlx::core::array& weights, float limit = 0.f, bool gelu = false,
        const mlx::core::array* reuse = nullptr) const;
};

struct MlxTqExpertCacheStats {
    std::uint64_t requests = 0, hits = 0, misses = 0, evictions = 0;
    std::uint64_t bytes_read = 0, read_calls = 0;
    std::size_t cached_experts = 0, cached_bytes = 0, allocated_bytes = 0;
    double io_seconds = 0, route_sync_seconds = 0, prepare_seconds = 0, graph_seconds = 0;
    std::size_t metadata_bytes = 0;
    std::uint64_t overlapped_read_batches = 0;
    std::size_t pinned_bytes = 0;
    std::uint64_t prefill_read_batches = 0, prefill_bytes_read = 0;
};

class MlxTqExpertCache {
public:
    class Plan {
    public:
        ~Plan();
        Plan(Plan&&) noexcept;
        Plan& operator=(Plan&&) noexcept;
        Plan(const Plan&) = delete;
        Plan& operator=(const Plan&) = delete;
        std::size_t minimum() const noexcept;
        std::size_t maximum() const noexcept;
    private:
        friend class MlxTqExpertCache;
        struct Data;
        explicit Plan(std::unique_ptr<Data> data);
        std::unique_ptr<Data> data_;
    };
    static Plan plan(const source::Source& source, const std::vector<std::string>& prefixes,
        int io_workers = 8);
    MlxTqExpertCache(Plan plan, std::size_t budget);
    MlxTqExpertCache(const MfqContainer& model, const std::vector<std::string>& prefixes,
        std::size_t budget, int io_workers = 8);
    MlxTqExpertCache(const source::Source& model, const std::vector<std::string>& prefixes,
        std::size_t budget, int io_workers = 8);
    std::size_t minimum() const;
    std::size_t maximum() const;
    mlx::core::array apply(std::size_t layer, const mlx::core::array& ids,
        const std::function<mlx::core::array(const MlxTqExpertWeights&)>& operation);
    mlx::core::array apply(std::size_t layer, const mlx::core::array& ids,
        const std::function<mlx::core::array(const MlxTqExpertWeights&,
            const mlx::core::array*)>& operation);
    mlx::core::array apply(std::size_t layer, const mlx::core::array& ids,
        const std::function<mlx::core::array(const MlxTqExpertWeights&,
            const mlx::core::array*)>& gate_up,
        const std::function<mlx::core::array(const MlxTqExpertWeights&,
            const mlx::core::array&, const mlx::core::array*)>& down,
        const mlx::core::array* parallel = nullptr);
    mlx::core::array apply_ffn(std::size_t layer, const mlx::core::array& input,
        const mlx::core::array& ids, const mlx::core::array& weights,
        float limit = 0.f, bool gelu = false, const mlx::core::array* parallel = nullptr);
    MlxTqExpertCacheStats stats() const;
    std::size_t limit_bytes() const noexcept;
    std::size_t backing_bytes() const noexcept;
    std::size_t resident_bytes() const noexcept;
    std::size_t set_limit(std::size_t bytes);
    std::size_t trim(std::size_t bytes);
    void validate_layer(std::size_t layer, int experts, int input, int intermediate) const;
    void prepare_layer(std::size_t layer, int top_k, float limit = 0.f, bool gelu = false) const;
    void preload();
    void drain();
    void clear();
    void prefill_lookahead(std::size_t layer, std::optional<std::size_t> next);

private:
    struct Impl;
    void load_legacy(tq_cache_detail::LegacyPlan plan, std::size_t budget);
    std::shared_ptr<Impl> impl_;
};

} // namespace tyloi::metal
