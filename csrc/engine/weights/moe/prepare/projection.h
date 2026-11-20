#pragma once

#include "engine/weights/moe/host/cache.h"
#include "engine/weights/moe/host/dispatch_plan.h"
#include "engine/tensor/encoding/ggml.h"

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <unordered_set>

class MoeCachedSource;
struct MoeRoutePlan;
namespace tyloi { class GgmlExpertBank; class MfeTensor; }
namespace tyloi::cuda { class Event; }

namespace tyloi::weights {

class GgmlExpertCache;
enum class ExpertProjection { project, prequantized, glu, clamp };

struct ExpertRows {
    struct Group { std::int64_t expert, begin, count; };
    std::int64_t tokens, routes, experts;
    DenseTensor inputs, slots;
    std::vector<Group> groups;
    DenseTensor host_ids;
    DenseTensor device_ids, device_groups;
};

struct PreparedExperts {
    enum class Mode { gpu, cpu, hybrid, staged } mode = Mode::gpu;
    std::shared_ptr<const MfeTensor> weight;
    std::vector<GgmlQuantizedTensor> ggml;
    std::unordered_map<int, MoeHostExpertCache::Lease> cpu;
    MoeDispatchPlan dispatch;
    Device device;
    bool complete = false;
    bool quantized = false;

    PreparedExperts() = default;
    ~PreparedExperts() = default;
    PreparedExperts(PreparedExperts&&) noexcept = default;
    PreparedExperts& operator=(PreparedExperts&&) noexcept = default;
    PreparedExperts(const PreparedExperts&) = delete;
    PreparedExperts& operator=(const PreparedExperts&) = delete;

    // Order the consuming stream after any uploads without blocking the host.
    void wait() const;
    // Register the consuming stream. Storage fences its last use when released.
    void retire();
    void record_cpu(std::size_t projections);
    void record_hybrid();

private:
    friend class GgmlExpertCache;
    friend PreparedExperts prepare_cached_projection(const MfeTensor&, const MoeRoutePlan&,
        Device, ExpertProjection, bool);
    std::shared_ptr<MoeCachedSource> source_;
    std::unordered_set<std::shared_ptr<cuda::Event>> ready_;
    bool retired_ = false;
};

PreparedExperts prepare_cached_projection(const MfeTensor& weight, const MoeRoutePlan& route,
    Device input_device, ExpertProjection projection, bool heterogeneous);
ExpertRows group_expert_rows(const DenseTensor& ids, std::int64_t experts, bool device_resident = false);
PreparedExperts prepare_cached_projection(const GgmlExpertBank& bank,
    std::span<const ExpertRows::Group> selected);
PreparedExperts prepare_indexed_projection(const GgmlExpertBank& bank,
    std::span<const std::int64_t> selected);
void observe_ggml_routes(std::span<const GgmlExpertBank* const> banks,
    const DenseTensor& host_ids, int row_offset, int row_total, bool prefill);

} // namespace tyloi::weights
