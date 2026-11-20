#include "engine/weights/cache/cache.h"
#include "engine/memory/host.h"
#include "engine/weights/cache/workspace.h"
#include "engine/memory/budget.h"
#include "engine/backends/metal/ops/moe.h"
#include "engine/weights/encoding/mx/affine.h"
#include "engine/weights/memory.h"
#include "engine/weights/moe/experts.h"
#include "engine/weights/source/read.h"
#include "engine/weights/tq/source.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/tensor/encoding/metal_experts.h"
#include "engine/weights/cache/metal_mfe.h"
#include "engine/backends/metal/ops/mfq/mlx_moe.h"
#include "engine/backends/metal/ops/mfq/mlx_nint.h"
#endif
#include "mlx_tq_expert_cache.h"
#include "mlx_gguf_expert_cache.h"
#include "mlx_affine_expert_cache.h"

#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace tyloi::weights {

#if TYLOINFER_ENABLE_MFQ_COMPAT
// Private policy storage. MFE is a residency policy, not a numeric ExpertBank.
using MetalCachePolicy = std::variant<
    metal::ExpertCache,
    std::shared_ptr<metal::MlxMfeOffloadCache>>;
#else
using MetalCachePolicy = metal::ExpertCache;
#endif

struct Resident::Impl {};
Resident::Resident() : impl_(std::make_unique<Impl>()) {}
Resident::~Resident() = default;

struct Cache::Resources {
    struct Group {
        MetalCachePolicy cache;
        std::vector<std::string> prefixes;
        std::size_t minimum = 0, maximum = 0, limit = 0;
    };
    std::vector<Group> groups;
    const source::Source* source = nullptr;
    bool ready = false;
    std::optional<std::size_t> configured;
    CachePolicy policy;
    bool preload = false;
    engine::Budget* budget = nullptr;
};

namespace {
std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("Metal expert cache budget overflow");
    return a + b;
}
#if TYLOINFER_ENABLE_MFQ_COMPAT
bool mfe_policy(const MetalCachePolicy& cache) {
    return std::holds_alternative<std::shared_ptr<metal::MlxMfeOffloadCache>>(cache);
}
#endif
template <typename Visitor>
decltype(auto) visit_cache(const MetalCachePolicy& cache, Visitor&& visitor) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return std::visit([&](const auto& value) -> decltype(auto) {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, std::shared_ptr<metal::MlxMfeOffloadCache>>)
            return visitor(value);
        else
            return std::visit([&](const auto& item) -> decltype(auto) { return visitor(item); }, value);
    }, cache);
#else
    return std::visit(std::forward<Visitor>(visitor), cache);
#endif
}
std::size_t limit(const MetalCachePolicy& cache) {
    return visit_cache(cache, [](const auto& value) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, std::shared_ptr<metal::MlxMfeOffloadCache>>)
            return value->cache_limit_bytes();
        else
#endif
            return value->limit_bytes();
    });
}
std::size_t resident(const MetalCachePolicy& cache) {
    return visit_cache(cache, [](const auto& value) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, std::shared_ptr<metal::MlxMfeOffloadCache>>)
            return value->resident_packed_bytes();
        else
#endif
            return value->resident_bytes();
    });
}
std::size_t set_limit(const MetalCachePolicy& cache, std::size_t bytes) {
    return visit_cache(cache, [&](const auto& value) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, std::shared_ptr<metal::MlxMfeOffloadCache>>)
            return value->set_cache_limit(bytes);
        else
#endif
            return value->set_limit(bytes);
    });
}
void clear(const MetalCachePolicy& cache) {
    visit_cache(cache, [](const auto& value) { value->clear(); });
}
void preload_cache(const MetalCachePolicy& cache) {
    visit_cache(cache, [](const auto& value) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, std::shared_ptr<metal::MlxMfeOffloadCache>>)
            throw std::invalid_argument("Metal native MFE expert preload is unsupported");
        else
#endif
            value->preload();
    });
}
void drain_cache(const MetalCachePolicy& cache) {
    visit_cache(cache, [](const auto& value) { value->drain(); });
}
void require_metal(Device device) {
    if (device != Device(DeviceType::metal))
        throw std::invalid_argument("Metal cache requires Metal device 0");
}
} // namespace

void prepare_experts(std::span<const Tensor> gate_up, const Tensor& down,
                     std::int64_t top_k, double limit) {
    if ((gate_up.size() != 1 && gate_up.size() != 2) || !down || down.dim() != 3 ||
        top_k <= 0 || !std::isfinite(limit) || limit < 0)
        throw std::invalid_argument("incomplete Metal routed expert bank");
    for (const auto& bank : gate_up)
        if (!bank || bank.dim() != 3 || !bank.is_metal() ||
            bank.size(0) != down.size(0))
            throw std::invalid_argument("invalid Metal routed expert bank");
    if (!down.is_metal() || down.size(1) <= 0 || down.size(2) <= 0 ||
        (gate_up.size() == 1 ? gate_up[0].size(1) != 2 * down.size(2) :
         gate_up[0].size(1) != down.size(2) || gate_up[1].size(1) != down.size(2)) ||
        gate_up[0].size(2) <= 0)
        throw std::invalid_argument("invalid Metal routed expert bank geometry");
    const auto* gate = gate_up.front().representation<metal::ExpertBank>();
    const auto* up = gate_up.size() == 2 ? gate_up.back().representation<metal::ExpertBank>() : nullptr;
    const auto* down_bank = down.representation<metal::ExpertBank>();
    if (!gate && !up && !down_bank) return;
    if (!gate || !up || !down_bank || gate->role() != metal::ExpertRole::gate ||
        up->role() != metal::ExpertRole::up || down_bank->role() != metal::ExpertRole::down ||
        gate->cache() != up->cache() || gate->cache() != down_bank->cache() ||
        gate->layer() != up->layer() || gate->layer() != down_bank->layer())
        throw std::invalid_argument("incompatible Metal expert cache banks");
    std::visit([&](const auto& cache) {
        cache->validate_layer(gate->layer(), static_cast<int>(down.size(0)),
                              static_cast<int>(gate_up.front().size(2)),
                              static_cast<int>(down.size(2)));
        cache->prepare_layer(gate->layer(), static_cast<int>(top_k),
                             static_cast<float>(limit));
    }, gate->cache());
}

Cache::Cache() : data_(std::make_unique<Resources>()) {}
Cache::~Cache() {
    try { retire(); } catch (...) { std::terminate(); }
}
void Cache::begin_load() { retire(); }
void Cache::configure(std::size_t bytes, const CachePolicy& policy, bool preload,
                      const std::string& profile_path) {
    if (!bytes) throw std::invalid_argument("Metal expert cache budget must be nonzero");
    if (!profile_path.empty()) throw std::invalid_argument("Metal expert cache profiles are not supported");
    if (!data_->groups.empty()) throw std::logic_error("cannot configure an active Metal expert cache");
    data_->configured = bytes;
    data_->policy = policy;
    data_->preload = preload;
}
std::size_t Cache::budget() const {
    if (data_->configured) return *data_->configured;
    std::size_t total = 0;
    for (const auto& group : data_->groups) total = add(total, group.limit);
    return total;
}
std::size_t Cache::minimum() const {
    std::size_t total = 0;
    for (const auto& group : data_->groups) total = add(total, group.minimum);
    return total;
}
bool Cache::empty() const { return data_->groups.empty(); }
std::vector<std::pair<std::string, double>> Cache::metrics() const {
    std::size_t bytes = 0;
    std::uint64_t hits = 0, misses = 0, evictions = 0, read = 0, calls = 0;
    for (const auto& group : data_->groups) {
        bytes = add(bytes, resident(group.cache));
        visit_cache(group.cache, [&](const auto& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
            using Value = std::decay_t<decltype(cache)>;
            if constexpr (std::is_same_v<Value, std::shared_ptr<metal::MlxMfeOffloadCache>>) {
                const auto stats = cache->reuse_stats();
                hits += stats.expert_hits;
                misses += stats.expert_loads;
                evictions += stats.evictions;
                read += stats.bytes_read;
                calls += stats.read_calls;
            } else
#endif
            {
                const auto stats = cache->stats();
                hits += stats.hits; misses += stats.misses; evictions += stats.evictions;
                read += stats.bytes_read; calls += stats.read_calls;
            }
        });
    }
    return {{"resident_expert_cache_bytes", double(bytes)}, {"expert_cache_budget_bytes", double(budget())},
            {"expert_cache_hits", double(hits)}, {"expert_cache_misses", double(misses)},
            {"expert_cache_evictions", double(evictions)}, {"ssd_expert_bytes_read", double(read)},
            {"ssd_expert_read_count", double(calls)}};
}
void Cache::report(std::ostream& output) const {
    for (const auto& [name, value] : metrics()) output << name << '=' << value << ' ';
    output << '\n';
}
Cache::Activity Cache::activity() const {
    std::uint64_t bytes = 0;
    for (const auto& group : data_->groups)
        visit_cache(group.cache, [&](const auto& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
            using Value = std::decay_t<decltype(cache)>;
            if constexpr (std::is_same_v<Value, std::shared_ptr<metal::MlxMfeOffloadCache>>)
                bytes += cache->reuse_stats().bytes_read;
            else
                bytes += cache->stats().bytes_read;
#else
            bytes += cache->stats().bytes_read;
#endif
        });
    return {bytes, 0};
}
Cache::Routing Cache::routing() const { return {}; }
bool Cache::legacy_experts_resident() const { return false; }
bool Cache::needs_finalize() const { return false; }
void Cache::finalize(std::size_t, std::size_t) {}
Cache::PrefillBudget Cache::prefill_budget(bool) const { return {}; }

void Cache::resize(std::size_t bytes) {
    if (data_->groups.empty()) { data_->configured = bytes; return; }
    auto minimum_bytes = minimum();
    if (bytes < minimum_bytes) throw std::invalid_argument("Metal expert cache budget is below its minimum");
    std::vector<std::size_t> next;
    next.reserve(data_->groups.size());
    for (const auto& group : data_->groups) next.push_back(group.minimum);
    auto remaining = bytes - minimum_bytes;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto has_mfe = std::any_of(data_->groups.begin(), data_->groups.end(),
        [](const auto& group) { return mfe_policy(group.cache); });
#else
    constexpr bool has_mfe = false;
#endif
    if (has_mfe) {
        // MFE has no truthful static maximum. Fixed pools keep their real
        // minimum; the remaining configured budget is its physical ceiling.
        for (std::size_t i = 0; i < data_->groups.size(); ++i) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
            if (mfe_policy(data_->groups[i].cache)) {
                next[i] = add(next[i], remaining);
                remaining = 0;
                break;
            }
#endif
        }
    } else {
        std::size_t total_room = 0;
        for (const auto& group : data_->groups)
            total_room = add(total_room, group.maximum - group.minimum);
        const auto distributable = std::min(remaining, total_room);
        if (total_room) for (std::size_t i = 0; i < data_->groups.size(); ++i) {
            const auto room = data_->groups[i].maximum - next[i];
            const auto grant = std::min(room, static_cast<std::size_t>(
                static_cast<long double>(distributable) * room / total_room));
            next[i] += grant;
            remaining -= grant;
        }
        for (std::size_t i = 0; i < data_->groups.size() && remaining; ++i) {
            const auto room = data_->groups[i].maximum - next[i];
            const auto grant = std::min(room, remaining);
            next[i] += grant;
            remaining -= grant;
        }
    }
    std::vector<std::size_t> previous;
    previous.reserve(data_->groups.size());
    for (const auto& group : data_->groups) previous.push_back(group.limit);
    try {
        for (std::size_t i = 0; i < next.size(); ++i)
            if (next[i] < previous[i]) set_limit(data_->groups[i].cache, next[i]);
        for (std::size_t i = 0; i < next.size(); ++i)
            if (next[i] > previous[i]) set_limit(data_->groups[i].cache, next[i]);
    } catch (...) {
        const auto failure = std::current_exception();
        for (std::size_t i = 0; i < previous.size(); ++i)
            try { if (limit(data_->groups[i].cache) != previous[i]) set_limit(data_->groups[i].cache, previous[i]); }
            catch (...) {}
        for (auto& group : data_->groups) group.limit = limit(group.cache);
        std::rethrow_exception(failure);
    }
    for (std::size_t i = 0; i < next.size(); ++i) {
        data_->groups[i].limit = next[i];
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if (mfe_policy(data_->groups[i].cache)) data_->groups[i].maximum = next[i];
#endif
    }
    data_->configured = bytes;
}
std::size_t Cache::reserve(std::size_t bytes) {
    reclaim_host(bytes);
    std::size_t before = 0;
    for (const auto& group : data_->groups) before = add(before, resident(group.cache));
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto has_mfe = std::any_of(data_->groups.begin(), data_->groups.end(),
        [](const auto& group) { return mfe_policy(group.cache); });
    if (has_mfe) {
        if (bytes) for (auto& group : data_->groups) {
            if (!mfe_policy(group.cache) || !group.limit) continue;
            const auto ceiling = group.limit;
            const auto current = resident(group.cache);
            const auto target = current > bytes ? std::min(ceiling, current - bytes) : 0;
            try {
                set_limit(group.cache, target);
                set_limit(group.cache, ceiling);
            } catch (...) {
                try { set_limit(group.cache, ceiling); } catch (...) {}
                throw;
            }
            break;
        }
        std::size_t after = 0;
        for (const auto& group : data_->groups) after = add(after, resident(group.cache));
        // resize() already keeps fixed pools at their minimum while MFE is active.
        return before > after ? before - after : 0;
    }
#endif
    std::size_t current = 0;
    for (const auto& group : data_->groups) current = add(current, group.limit);
    if (current <= minimum()) return 0;
    const auto configured = data_->configured;
    resize(current - std::min(bytes, current - minimum()));
    data_->configured = configured;
    std::size_t after = 0;
    for (const auto& group : data_->groups) after = add(after, resident(group.cache));
    return before > after ? before - after : 0;
}
std::size_t Cache::reclaim_host_duplicates(std::size_t) { return 0; }
std::size_t Cache::reserve_host_capacity(std::size_t bytes) { return reserve(bytes); }
void Cache::share_host(std::size_t) {}
void Cache::workspace(std::size_t bytes) {
    if (data_->budget) data_->budget->reclaim(bytes);
}
void Cache::release_workspace(std::size_t bytes, const char*) {
    if (bytes != std::numeric_limits<std::size_t>::max()) (void)reserve(bytes);
}
void Cache::clear_workspace() noexcept {}
void Cache::drain() const {
    if (data_->groups.empty()) return;
    for (const auto& group : data_->groups)
        drain_cache(group.cache);
    mlx::core::synchronize();
}
void Cache::finish_exchange() const { drain(); }
void Cache::admit_routes(Device) {}
void Cache::retire_prefill() { drain(); }
void Cache::discard_prefill() {}
void Cache::settle() { drain(); }
void Cache::request(bool) { drain(); }
std::size_t Cache::storage_workspace_reserve() const { return 0; }
bool Cache::lend_decode_storage(std::size_t) { return false; }
void Cache::prepare(Stage, Device device, const source::Source*) { require_metal(device); }
void Cache::trace(const char*, bool wait, const source::Source*) { if (wait) drain(); }
void Cache::bind(engine::Budget& budget, Device device, bool) { require_metal(device); data_->budget = &budget; }
void Cache::unbind() noexcept { data_->budget = nullptr; }
void Cache::retire_decode() { drain(); }
void Cache::retire() {
    drain();
    for (const auto& group : data_->groups) clear(group.cache);
    data_->groups.clear();
    data_->source = nullptr;
    data_->ready = false;
    data_->configured.reset();
    data_->budget = nullptr;
}
void Cache::reset() noexcept {
    try { retire(); }
    catch (...) { /* Keep groups alive until an explicit drain or destruction. */ }
}

// Metal expert caches own their pooled storage and do not lend it as KV or
// prefill workspace. An empty region/loan reports that no storage was lent.
struct KvRegion::Impl {};
KvRegion::KvRegion() noexcept = default;
KvRegion::KvRegion(KvRegion&&) noexcept = default;
KvRegion& KvRegion::operator=(KvRegion&&) = default;
KvRegion::~KvRegion() noexcept = default;
void KvRegion::reset() { impl_.reset(); }
void KvRegion::advance(std::int64_t) {}
void KvRegion::protect(std::int64_t) {}
std::size_t KvRegion::bytes() const { return 0; }
void Cache::prepare_kv() {}
KvRegion Cache::lend_kv(const Tensor&, std::int64_t, std::size_t, bool, std::int64_t) { return {}; }
KvRegion Cache::lend_kv(const Tensor&) { return {}; }
void Cache::commit_kv() {}
void Cache::protect_kv(std::int64_t) {}
Cache::KvMetrics Cache::kv_metrics() const { return {}; }

struct Loan::Impl { Cache& cache; };
Loan::Loan(Cache& cache) { cache.lend(*this); }
Loan::Loan(Cache& cache, std::size_t bytes) : Loan(cache) { begin(bytes); }
Loan::~Loan() = default;
void Cache::lend(Loan& loan) { loan.impl_ = std::make_unique<Loan::Impl>(Loan::Impl{*this}); }
std::size_t Loan::begin(std::size_t bytes) { impl_->cache.workspace(bytes); return 0; }
std::size_t Loan::predictor(std::size_t bytes) { impl_->cache.workspace(bytes); return 0; }
std::size_t Loan::transfer_bytes() const { return 0; }
void Loan::suspend() {}
void Loan::finish() {}
void Loan::finish(Tensor& output) {
    Tensor* slot = &output;
    finish(std::span<Tensor* const>(&slot, 1));
}
void Loan::finish(std::span<Tensor* const> outputs) {
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (!outputs[i]) throw std::invalid_argument("loan output slot is null");
        for (std::size_t j = 0; j < i; ++j)
            if (outputs[i] == outputs[j]) throw std::invalid_argument("duplicate loan output slot");
    }
}

struct Cache::Routes::Impl {};
Cache::Routes::Routes() noexcept = default;
Cache::Routes::Routes(Routes&&) noexcept = default;
Cache::Routes::~Routes() = default;
void Cache::Routes::begin(Cache&, int) {}
void Cache::Routes::begin_prefill(Cache&, int) {}
bool Cache::Routes::begin_prefill(Cache&, std::span<const int>) { return false; }
void Cache::Routes::detach_prefill() noexcept {}
void Cache::Routes::finish_prefill(bool) {}
void Cache::Routes::finish_prefill(std::span<const std::uint8_t>) {}
void Cache::Routes::resolve(int, bool) {}
void Cache::Routes::resolve(std::span<const std::uint8_t>) {}

struct Cache::Warmup::Impl { Cache& cache; };
Cache::Warmup::Warmup(Cache& cache, bool) : impl_(std::make_unique<Impl>(Impl{cache})) {}
Cache::Warmup::~Warmup() = default;
void Cache::Warmup::drain() { impl_->cache.drain(); }
void Cache::Warmup::release(Device device, bool) { require_metal(device); drain(); }

struct Cache::Priority::Impl {};
Cache::Priority::Priority(Cache&, int, int) : impl_(std::make_unique<Impl>()) {}
Cache::Priority::~Priority() = default;
void Cache::Priority::finish() {}

Tensor Cache::experts(const source::Source& source, const std::string& name,
                            int, const std::string& role) {
    const auto& record = require_tensor(source, name);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto paged_mfe = data_->configured.has_value() &&
        (record.dtype == "MFE" || metal::is_nint_dtype(record.dtype));
    if (paged_mfe && data_->preload)
        throw std::invalid_argument("Metal native MFE expert preload is unsupported");
    if (!data_->configured && (record.dtype == "MFE" || metal::is_nint_dtype(record.dtype))) {
        // Without an explicit cache budget, preserve the existing full-resident path.
        const auto blob = read_tensor(source, name);
        auto weight = record.dtype == "MFE" ? metal::MlxMfeWeight::from_blob(blob)
                                             : metal::MlxMfeWeight::from_nint_blob(blob);
        auto value = std::make_shared<encoding::MetalExperts>(std::move(weight));
        if (record.shape && (*record.shape != std::vector<std::int64_t>{value->size(0), value->size(1), value->size(2)}))
            throw std::invalid_argument("Metal expert bank shape differs from payload: " + name);
        observe_weight(value);
        return value;
    }
#else
    constexpr bool paged_mfe = false;
#endif
    using metal::ExpertRole;
    ExpertRole kind = ExpertRole::gate;
    std::string prefix;
    if (!paged_mfe) {
        if (role == "gate") kind = ExpertRole::gate;
        else if (role == "up") kind = ExpertRole::up;
        else if (role == "down") kind = ExpertRole::down;
        else throw std::invalid_argument("Metal cached experts require separate gate, up and down banks");
        const auto suffix = std::string(".mlp.experts.") + role + ".weight";
        if (name.size() <= suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix))
            throw std::invalid_argument("unexpected Metal expert bank name: " + name);
        prefix = name.substr(0, name.size() - suffix.size());
    }
    if (data_->source && data_->source != &source)
        throw std::invalid_argument("Metal expert cache belongs to a different source");
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::shared_ptr<metal::MlxMfeOffloadCache> requested_mfe;
    bool attach_mfe = false;
    if (paged_mfe) {
        for (const auto& group : data_->groups)
            if (const auto* value = std::get_if<std::shared_ptr<metal::MlxMfeOffloadCache>>(&group.cache)) {
                requested_mfe = *value;
                break;
            }
        if (!requested_mfe) {
            requested_mfe = std::make_shared<metal::MlxMfeOffloadCache>(source, 0);
            attach_mfe = data_->source != nullptr;
        }
    }
#endif

    if (!data_->source) {
        enum class Format { tq, gguf, affine };
        const auto classify = [](const std::string& dtype) -> std::optional<Format> {
            if (dtype == "TQ") return Format::tq;
            if (dtype.compare(0, 5, "GGUF_") == 0) return Format::gguf;
            if (dtype == "MLX-AFFINE") return Format::affine;
            return std::nullopt;
        };
        std::map<Format, std::vector<std::string>> prefixes;
#if TYLOINFER_ENABLE_MFQ_COMPAT
        const auto has_mfe = bool(requested_mfe);
#else
        constexpr bool has_mfe = false;
#endif
        const std::string gate_suffix = ".mlp.experts.gate.weight";
        for (const auto& tensor : source.tensors()) {
            if (tensor.name.size() <= gate_suffix.size() ||
                tensor.name.compare(tensor.name.size() - gate_suffix.size(), gate_suffix.size(), gate_suffix)) continue;
            if (tensor.dtype == "F32" || tensor.dtype == "F16" || tensor.dtype == "BF16") continue;
            const auto base = tensor.name.substr(0, tensor.name.size() - gate_suffix.size());
            const auto format = classify(tensor.dtype);
            if (!format) continue;
            const auto& up = require_tensor(source, base + ".mlp.experts.up.weight");
            const auto& down = require_tensor(source, base + ".mlp.experts.down.weight");
            if (classify(up.dtype) != format || classify(down.dtype) != format)
                throw std::invalid_argument("Metal expert banks use incompatible formats: " + base);
            prefixes[*format].push_back(base);
        }
        using TqPlan = metal::MlxTqExpertCache::Plan;
        using GgufPlan = metal::MlxGgmlExpertCache::Plan;
        using AffinePlan = metal::MlxAffineExpertCache::Plan;
        struct Pending {
            Format format;
            std::vector<std::string> prefixes;
            std::size_t minimum, maximum, limit;
            std::variant<TqPlan, GgufPlan, AffinePlan> plan;
        };
        std::vector<Pending> pending;
        std::size_t total_minimum = 0, total_maximum = 0;
        for (auto& [format, names] : prefixes) {
            std::sort(names.begin(), names.end());
            if (format == Format::tq) {
                auto plan = metal::MlxTqExpertCache::plan(source, names, data_->policy.io_workers);
                const auto low = plan.minimum(), high = plan.maximum();
                pending.push_back({format, std::move(names), low, high, low, std::move(plan)});
            } else if (format == Format::gguf) {
                auto plan = metal::MlxGgmlExpertCache::plan(source, names);
                const auto low = plan.minimum(), high = plan.capacity();
                pending.push_back({format, std::move(names), low, high, low, std::move(plan)});
            } else {
                auto plan = metal::MlxAffineExpertCache::plan(source, names);
                const auto low = plan.minimum(), high = plan.capacity();
                pending.push_back({format, std::move(names), low, high, low, std::move(plan)});
            }
            total_minimum = add(total_minimum, pending.back().minimum);
            total_maximum = add(total_maximum, pending.back().maximum);
        }
        const auto budget = data_->configured.value_or(total_maximum);
        if (budget < total_minimum) throw std::invalid_argument("Metal expert cache budget is below its minimum");
        if (!has_mfe) {
            auto remaining = budget - total_minimum;
            const auto room = total_maximum - total_minimum;
            const auto distributable = std::min(remaining, room);
            if (room) for (auto& group : pending) {
                const auto available = group.maximum - group.minimum;
                const auto grant = std::min(available, static_cast<std::size_t>(
                    static_cast<long double>(distributable) * available / room));
                group.limit += grant;
                remaining -= grant;
            }
            for (auto& group : pending) {
                const auto grant = std::min(remaining, group.maximum - group.limit);
                group.limit += grant;
                remaining -= grant;
            }
        }
        std::vector<Resources::Group> prepared;
        prepared.reserve(pending.size() + (has_mfe ? 1 : 0));
        for (auto& item : pending) {
            metal::ExpertCache cache;
            if (item.format == Format::tq) {
                // TQ cache reports its own metadata and pooled storage. Keep
                // those buffers out of the ordinary model-weight ledger.
                Memory cache_memory;
                LoadScope cache_scope(cache_memory);
                auto tq = std::make_shared<metal::MlxTqExpertCache>(std::move(std::get<TqPlan>(item.plan)), item.limit);
                item.minimum = tq->minimum();
                item.maximum = tq->maximum();
                cache = std::move(tq);
            } else if (item.format == Format::gguf)
                cache = std::shared_ptr<metal::MlxPackedExpertCache>(std::make_shared<metal::MlxGgmlExpertCache>(
                    std::move(std::get<GgufPlan>(item.plan)), item.limit, data_->policy.io_workers));
            else
                cache = std::shared_ptr<metal::MlxPackedExpertCache>(std::make_shared<metal::MlxAffineExpertCache>(
                    std::move(std::get<AffinePlan>(item.plan)), item.limit, data_->policy.io_workers));
            prepared.push_back({std::move(cache), std::move(item.prefixes), item.minimum, item.maximum, item.limit});
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if (has_mfe) {
            prepared.push_back({requested_mfe, {}, 0, budget, 0});
        }
#endif
        data_->groups = std::move(prepared);
        data_->source = &source;
        const auto configured = data_->configured;
        resize(budget);
        data_->configured = configured;
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (attach_mfe) {
        const auto budget = data_->configured.value();
        data_->groups.push_back({requested_mfe, {}, 0, budget, 0});
        resize(budget);
    }
#endif
    if (!data_->ready) {
        std::size_t full = 0;
        for (const auto& group : data_->groups) full = add(full, group.maximum);
        if (data_->preload && budget() < full)
            throw std::invalid_argument("Metal expert preload requires the complete cache budget");
        for (const auto& group : data_->groups) {
            if (data_->preload) preload_cache(group.cache);
            else if (group.limit >= group.maximum) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
                if (const auto* native = std::get_if<metal::ExpertCache>(&group.cache))
                    if (const auto* tq = std::get_if<std::shared_ptr<metal::MlxTqExpertCache>>(native))
                        (*tq)->preload();
#else
                if (const auto* tq = std::get_if<std::shared_ptr<metal::MlxTqExpertCache>>(&group.cache))
                    (*tq)->preload();
#endif
            }
        }
        data_->ready = true;
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (paged_mfe) {
        if (!requested_mfe) throw std::invalid_argument("Metal cache has no MFE policy for: " + name);
        if (!requested_mfe->can_offload(name))
            throw std::invalid_argument("Metal bounded MFE/NINT record is not streamable: " + name);
        requested_mfe->activate_record(name);
        const auto info = requested_mfe->projection_info(name);
        const std::array<std::int64_t, 3> shape{
            info.experts, info.out_per_expert, info.neuron_len};
        if (record.shape && *record.shape != std::vector<std::int64_t>{shape[0], shape[1], shape[2]})
            throw std::invalid_argument("Metal expert bank shape differs from MFE header: " + name);
        return std::make_shared<encoding::MetalExperts>(requested_mfe, name, shape);
    }
#endif
    std::vector<std::int64_t> shape;
    if (record.dtype == "TQ") shape = tq_shape(source, name);
    else if (record.shape) shape = *record.shape;
    else if (record.dtype == "MLX-AFFINE") {
        const auto header = source.read(name, 0, 8);
        const auto bytes = MlxAffineLayout::prefix_size(header.data(), header.size());
        const auto full = source.read(name, 0, bytes);
        shape = MlxAffineLayout::parse(full.data(), full.size()).shape;
    }
    if (shape.size() != 3) throw std::invalid_argument("Metal expert bank requires rank-3 geometry: " + name);
    for (const auto& group : data_->groups) {
        const auto found = std::lower_bound(group.prefixes.begin(), group.prefixes.end(), prefix);
        if (found == group.prefixes.end() || *found != prefix) continue;
#if TYLOINFER_ENABLE_MFQ_COMPAT
        const auto* cache = std::get_if<metal::ExpertCache>(&group.cache);
        if (!cache) throw std::invalid_argument("Metal expert cache bank uses a non-numeric policy: " + name);
        return std::make_shared<metal::ExpertBank>(*cache,
#else
        return std::make_shared<metal::ExpertBank>(group.cache,
#endif
            static_cast<std::size_t>(found - group.prefixes.begin()), kind,
            std::array<std::int64_t, 3>{shape[0], shape[1], shape[2]});
    }
    throw std::invalid_argument("Metal expert cache has no split bank: " + name);
}

} // namespace tyloi::weights
