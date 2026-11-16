#pragma once

#include "engine/weights/moe/host/cache.h"
#include "engine/weights/moe/host/dispatch_plan.h"

#include <cstddef>
#include <memory>
#include <unordered_map>

class MoeCachedSource;
struct MoeRoutePlan;
namespace tyloi { class MfeTensor; }

namespace tyloi::weights {

enum class ExpertProjection { project, prequantized, glu, clamp };

struct PreparedExperts {
    enum class Mode { gpu, cpu, hybrid, staged } mode = Mode::gpu;
    std::shared_ptr<const MfeTensor> weight;
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

    // Record after the last GPU launch; this does not wait for device completion.
    void retire();
    void record_cpu(std::size_t projections);
    void record_hybrid();

private:
    friend PreparedExperts prepare_cached_projection(const MfeTensor&, const MoeRoutePlan&,
        Device, ExpertProjection, bool);
    std::shared_ptr<MoeCachedSource> source_;
    bool retired_ = false;
};

PreparedExperts prepare_cached_projection(const MfeTensor& weight, const MoeRoutePlan& route,
    Device input_device, ExpertProjection projection, bool heterogeneous);

} // namespace tyloi::weights
