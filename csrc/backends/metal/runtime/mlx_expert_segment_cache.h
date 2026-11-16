#pragma once

#include "mlx_ssd_expert_cache.h"
#include "engine/weights/moe/residency/segments.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tyloi { struct TensorStorage; }
namespace tyloi::metal { class ResidentAllocations; }
namespace tyloi::source { class Source; }

namespace tyloi::metal {
class MfqContainer;

class MlxExpertSegmentCache {
public:
    class Plan {
    public:
        struct Impl;
        Plan(Plan&&) noexcept;
        Plan& operator=(Plan&&) noexcept;
        ~Plan();
        Plan(const Plan&) = delete;
        Plan& operator=(const Plan&) = delete;
        std::size_t minimum() const noexcept;
        std::size_t capacity() const noexcept;
    private:
        explicit Plan(std::unique_ptr<Impl>);
        std::unique_ptr<Impl> impl_;
        void reserve_metadata(std::size_t bytes);
        bool reserve_capacity_ = true;
        friend class MlxGgufExpertCache;
        friend class MlxExpertSegmentCache;
    };
    static Plan plan(const tyloi::source::Source& source, std::vector<tyloi::weights::Segments> layers);
    MlxExpertSegmentCache(Plan plan, std::size_t budget, int workers = 8, bool prefill_read_ahead = false);
    MlxExpertSegmentCache(const MfqContainer& model, std::vector<tyloi::weights::Segments> layers,
        std::size_t budget, int workers = 8, bool prefill_read_ahead = false);
    MlxExpertSegmentCache(const tyloi::source::Source& source, std::vector<tyloi::weights::Segments> layers,
        std::size_t budget, int workers = 8, bool prefill_read_ahead = false);
    mlx::core::array apply(std::size_t layer, const mlx::core::array& ids,
        std::size_t first_stage_fields, const std::function<mlx::core::array()>& first_stage,
        const std::function<mlx::core::array(const mlx::core::array&, bool)>& second_stage,
        const mlx::core::array* parallel = nullptr);
    // Called while apply owns the cache lock, before publishing lazy weight metadata.
    void record_metadata(const std::vector<std::shared_ptr<tyloi::TensorStorage>>& storage);
    const std::vector<mlx::core::array>& address_tables(std::size_t layer) const;
    const std::shared_ptr<tyloi::metal::ResidentAllocations>& residency() const noexcept;
    void preload();
    void drain();
    void set_prefill_next_rows(std::size_t layer, std::size_t rows);
    std::size_t limit_bytes() const noexcept;
    std::size_t backing_bytes() const noexcept;
    std::size_t resident_bytes() const noexcept;
    std::size_t set_limit(std::size_t bytes);
    MlxSsdExpertCacheStats stats() const;
    void clear();
private:
    static Plan plan_legacy(const MfqContainer& model, std::vector<tyloi::weights::Segments> layers);
    friend class MlxGgufExpertCache;
    friend class MlxAffineExpertCache;
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
} // namespace tyloi::metal
