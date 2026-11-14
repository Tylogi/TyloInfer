#include "mlx_affine_expert_cache_internal.h"
#include "mfq_container.h"

namespace tyloi::metal {
namespace affine_cache_detail {
const std::string& dtype(const MfqContainer& source, const std::string& name) { return source.record(name).dtype; }
std::uint64_t extent(const MfqContainer& source, const std::string& name) { return source.record(name).nbytes; }
std::vector<std::byte> read_prefix(const MfqContainer& source, const std::string& name, std::size_t size) {
    std::vector<std::byte> bytes(size);
    source.read_range_into(name, 0, bytes);
    return bytes;
}
void prepared(const MfqContainer& source, const std::string& name) { source.record_prepared(name); }
} // namespace affine_cache_detail

MlxAffineExpertCache::Plan MlxAffineExpertCache::plan_legacy(
        const MfqContainer& model, const std::vector<std::string>& prefixes) {
    auto description = affine_cache_detail::describe(model, prefixes);
    auto segments = MlxExpertSegmentCache::plan_legacy(model, std::move(description.segments));
    return Plan(std::make_unique<Plan::Impl>(std::move(description.layers), std::move(segments)));
}
MlxAffineExpertCache::MlxAffineExpertCache(const MfqContainer& model,
        const std::vector<std::string>& prefixes, std::size_t budget, int workers)
    : MlxAffineExpertCache(plan_legacy(model, prefixes), budget, workers) {}
} // namespace tyloi::metal
