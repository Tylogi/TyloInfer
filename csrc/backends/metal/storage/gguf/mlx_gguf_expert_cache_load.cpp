#include "mlx_gguf_expert_cache_internal.h"
#include "mfq_container.h"

namespace tyloi::metal {
namespace gguf_cache_detail {
const Metadata& metadata(const MfqContainer& source) { return source.header().extra_json; }
const std::string& dtype(const MfqContainer& source, const std::string& name) { return source.record(name).dtype; }
std::uint64_t extent(const MfqContainer& source, const std::string& name) { return source.record(name).nbytes; }
std::vector<std::int64_t> shape(const MfqContainer& source, const std::string& name, nlohmann::json&) {
    return source.tensor_shape(name);
}
void prepared(const MfqContainer& source, const std::string& name) { source.record_prepared(name); }
std::optional<Projection::Rotation> rotation(const MfqContainer&, const nlohmann::json& layouts, const std::string& name) {
    if (!layouts.contains(name)) return std::nullopt;
    const auto& entry = layouts.at(name);
    return Projection::Rotation{entry.at("block_size").get<std::int64_t>(), entry.at("signs").get<std::vector<float>>()};
}
bool normalized_rotations(const MfqContainer&) { return false; }
} // namespace gguf_cache_detail

MlxGgufExpertCache::Plan MlxGgufExpertCache::plan_legacy(
        const MfqContainer& model, const std::vector<std::string>& prefixes) {
    auto description = gguf_cache_detail::describe(model, prefixes);
    auto segments = MlxExpertSegmentCache::plan_legacy(model, std::move(description.segments));
    segments.reserve_metadata(gguf_cache_detail::metadata_reservation(description.layers));
    return Plan(std::make_unique<Plan::Impl>(std::move(description.layers), std::move(segments)));
}
MlxGgufExpertCache::MlxGgufExpertCache(const MfqContainer& model,
        const std::vector<std::string>& prefixes, std::size_t budget, int workers)
    : MlxGgufExpertCache(plan_legacy(model, prefixes), budget, workers) {}
} // namespace tyloi::metal
