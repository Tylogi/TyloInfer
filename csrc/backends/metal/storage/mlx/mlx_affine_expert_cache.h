#pragma once

#include "mlx_packed_expert_cache.h"
#include <memory>
#include <string>
#include <vector>

namespace tyloi::source { class Source; }
namespace tyloi::metal {
class MfqContainer;
class MlxAffineExpertCache final : public MlxPackedExpertCache {
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
        friend class MlxAffineExpertCache;
    };
    static Plan plan(const tyloi::source::Source& source, const std::vector<std::string>& prefixes);
    MlxAffineExpertCache(Plan plan, std::size_t budget, int workers = 8);
    MlxAffineExpertCache(const MfqContainer& model, const std::vector<std::string>& prefixes,
        std::size_t budget, int workers = 8);
    MlxAffineExpertCache(const tyloi::source::Source& source, const std::vector<std::string>& prefixes,
        std::size_t budget, int workers = 8);
    mlx::core::array apply_ffn(std::size_t layer, const mlx::core::array& input,
        const mlx::core::array& ids, const mlx::core::array& weights,
        const mlx::core::array* parallel = nullptr) override;
    void validate_layer(std::size_t layer, int experts, int input, int intermediate) const override;
    void prepare_layer(std::size_t layer, int top_k) const override;
    void preload() override;
    void drain() override;
    bool preserves_activation_dtype() const noexcept override { return true; }
    std::size_t limit_bytes() const noexcept override;
    std::size_t backing_bytes() const noexcept override;
    std::size_t resident_bytes() const noexcept override;
    std::size_t set_limit(std::size_t bytes) override;
    MlxSsdExpertCacheStats stats() const override;
    void clear() override;
private:
    static Plan plan_legacy(const MfqContainer& model, const std::vector<std::string>& prefixes);
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
} // namespace tyloi::metal
