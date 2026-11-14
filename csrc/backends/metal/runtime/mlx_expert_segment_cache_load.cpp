#include "mlx_expert_segment_cache_internal.h"
#include "mfq_container.h"

namespace tyloi::metal {
MlxExpertSegmentCache::Plan MlxExpertSegmentCache::plan_legacy(
        const MfqContainer& model, std::vector<tyloi::weights::Segments> layers) {
    Plan result(segment_make_plan(std::move(layers),
        [source = std::make_shared<MfqContainer>(model)](const std::string& name) {
            return SegmentPlan::Reader{source->record(name).nbytes, [source, name](auto offset, auto targets) {
                source->readv_range(name, offset, targets);
            }};
        }));
    result.reserve_capacity_ = false;
    return result;
}
MlxExpertSegmentCache::MlxExpertSegmentCache(const MfqContainer& model,
        std::vector<tyloi::weights::Segments> layers, std::size_t budget, int workers, bool read_ahead)
    : MlxExpertSegmentCache(plan_legacy(model, std::move(layers)), budget, workers, read_ahead) {}
} // namespace tyloi::metal
