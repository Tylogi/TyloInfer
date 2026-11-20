#include "engine/weights/cache/resources.h"
#include "engine/weights/cache/ggml.h"
#include "engine/memory/host.h"

#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/tq/resources.h"
#include "engine/weights/moe/residency/expert_cache.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/moe/residency/expert_cache_internal.h"
#endif
#include "engine/weights/source/source.h"
#include "engine/weights/tq/cache.h"
#include "env.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace tyloi::weights {

struct Cache::Routes::Impl {
    std::shared_ptr<MoeExpertCache> cache;
    std::shared_ptr<TqCache> tq;
    std::shared_ptr<GgmlExpertCache> ggml;
    std::shared_ptr<GgmlExpertCache::Prefill> prompt;
    int rows = 0;

    void discard() noexcept {
        discard_moe_speculative_routes(cache);
        if (tq) tq->discard_routes();
        if (ggml) {
            if (prompt) ggml->detach_prefill(*prompt);
            else ggml->discard_routes();
        }
        cache.reset();
        tq.reset();
        ggml.reset();
        rows = 0;
        prompt.reset();
    }
};

Cache::Routes::Routes() noexcept = default;
Cache::Routes::Routes(Routes&&) noexcept = default;
Cache::Routes::~Routes() {
    if (impl_) impl_->discard();
}

void Cache::Routes::begin(Cache& owner, int rows) {
    if (!impl_) impl_ = std::make_unique<Impl>();
    if (impl_->cache || impl_->tq || impl_->ggml)
        throw std::logic_error("speculative routes already pending");
    auto cache = owner.data_->moe;
    auto tq = owner.data_->tq;
    auto ggml = owner.data_->ggml;
    bool tq_started = false;
    try {
        begin_moe_speculative_routes(cache, rows);
        if (tq) {
            tq->begin_routes(rows);
            tq_started = true;
        }
        if (ggml) ggml->begin_routes(rows);
    } catch (...) {
        if (tq_started) tq->discard_routes();
        discard_moe_speculative_routes(cache);
        throw;
    }
    impl_->cache = std::move(cache);
    impl_->tq = std::move(tq);
    impl_->ggml = std::move(ggml);
    impl_->rows = rows;
}

void Cache::Routes::begin_prefill(Cache& owner, int rows) {
    if (rows < 0) throw std::invalid_argument("invalid prefill row count");
    begin_prefill(owner, rows ? std::span<const int>{&rows, 1} : std::span<const int>{});
}

bool Cache::Routes::begin_prefill(Cache& owner, std::span<const int> request_rows) {
    const auto& cache = owner.data_->ggml;
    if (!cache || !cache->has_cached_sources()) return false;
    if (impl_) throw std::logic_error("route journal already started");
    impl_ = std::make_unique<Impl>();
    impl_->prompt = cache->begin_prefill(request_rows);
    impl_->ggml = cache;
    return true;
}

void Cache::Routes::detach_prefill() noexcept {
    if (impl_ && impl_->prompt) impl_->ggml->detach_prefill(*impl_->prompt);
}

void Cache::Routes::finish_prefill(bool complete) {
    if (!impl_ || !impl_->prompt) return;
    const std::uint8_t keep = complete;
    finish_prefill(std::span<const std::uint8_t>{&keep, 1});
}

void Cache::Routes::finish_prefill(std::span<const std::uint8_t> accepted) {
    if (!impl_ || !impl_->prompt) return;
    detach_prefill();
    impl_->ggml->finish_prefill(*impl_->prompt, accepted);
    impl_.reset();
}

void Cache::Routes::resolve(int rows, bool cancelled) {
    if (!impl_ || (!impl_->cache && !impl_->tq && !impl_->ggml)) return;
    if (cancelled) {
        impl_->discard();
        return;
    }
    try {
        if (rows < 0 || rows > impl_->rows)
            throw std::invalid_argument("committed route prefix is out of range");
        std::vector<std::uint8_t> keep(std::size_t(impl_->rows), std::uint8_t(0));
        std::fill(keep.begin(), keep.begin() + rows, std::uint8_t(1));
        commit(keep, true);
    } catch (...) {
        impl_->discard();
        throw;
    }
}

void Cache::Routes::resolve(std::span<const std::uint8_t> keep) {
    commit(keep, false);
}

void Cache::Routes::commit(std::span<const std::uint8_t> keep, bool single_sequence) {
    if (!impl_ || (!impl_->cache && !impl_->tq && !impl_->ggml)) return;
    try {
        if (keep.size() != std::size_t(impl_->rows))
            throw std::invalid_argument("speculative route mask row count is out of range");
        commit_moe_speculative_routes(impl_->cache, keep, single_sequence);
        if (impl_->tq) impl_->tq->commit_routes(keep, single_sequence);
        if (impl_->ggml) impl_->ggml->commit_routes(keep, single_sequence);
    } catch (...) {
        impl_->discard();
        throw;
    }
    impl_->cache.reset();
    impl_->tq.reset();
    impl_->ggml.reset();
    impl_->rows = 0;
}

struct Cache::Warmup::Impl {
    std::shared_ptr<MoeExpertCache> cache;
    std::shared_ptr<TqCache> tq;
    std::shared_ptr<GgmlExpertCache> ggml;
    bool suspended = false;
    bool previous_residency = false;
    bool previous_learning = false;
    bool previous_tq_learning = false;
    bool previous_ggml_learning = false;
};

Cache::Warmup::Warmup(Cache& owner, bool suspend_residency) : impl_(std::make_unique<Impl>()) {
    impl_->cache = owner.data_->moe;
    impl_->tq = owner.data_->tq;
    impl_->ggml = owner.data_->ggml;
    impl_->suspended = suspend_residency;
    if (suspend_residency) impl_->previous_residency = suspend_moe_residency(impl_->cache, true);
    impl_->previous_learning = set_moe_reuse_learning(impl_->cache, false);
    try {
        if (impl_->tq) impl_->previous_tq_learning = impl_->tq->set_learning(false);
        if (impl_->ggml) impl_->previous_ggml_learning = impl_->ggml->set_learning(false);
    } catch (...) {
        if (impl_->tq) impl_->tq->set_learning(impl_->previous_tq_learning);
        set_moe_reuse_learning(impl_->cache, impl_->previous_learning);
        if (impl_->suspended) suspend_moe_residency(impl_->cache, impl_->previous_residency);
        throw;
    }
}

Cache::Warmup::~Warmup() {
    if (impl_->ggml) impl_->ggml->set_learning(impl_->previous_ggml_learning);
    if (impl_->tq) impl_->tq->set_learning(impl_->previous_tq_learning);
    set_moe_reuse_learning(impl_->cache, impl_->previous_learning);
    if (impl_->suspended) suspend_moe_residency(impl_->cache, impl_->previous_residency);
}

void Cache::Warmup::drain() {
    finish_moe_expert_exchanges(impl_->cache);
}

void Cache::Warmup::release(Device device, bool decode) {
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("warmup requires a CUDA device");
    if (decode) release_moe_decode_buffers(impl_->cache);
    else release_moe_prefill_buffers(impl_->cache);
    cuda::default_context(device.index)->reclaim_idle_memory();
}

void Cache::begin_load() {
    reset();
    auto& executor = cuda::Executor::current();
    data_->exec = executor.tq();
    data_->host_bytes = cuda::env_size("TYLOINFER_MOE_HOST_CACHE_BYTES", 0);
    data_->host_limit = cuda::env_size("TYLOINFER_MOE_HOST_PHYSICAL_BYTES", 0);
}

void Cache::configure(std::size_t bytes, const CachePolicy& policy, bool preload,
        const std::string& profile_path) {
    data_->device_bytes = bytes;
    const auto& options = cuda::Executor::current().options();
    data_->moe = make_moe_expert_cache(static_cast<std::int64_t>(bytes), MoeCacheOptions{
        .policy = policy,
        .host_bytes = data_->host_bytes,
        .host_limit = data_->host_limit,
        .preload = preload,
        .heterogeneous = options.projection.moe.heterogeneous});
    if (!profile_path.empty() && data_->moe)
        set_moe_expert_cache_profile(*data_->moe, tyloi::load_moe_cache_profile(profile_path));
}

std::vector<std::pair<std::string, double>> Cache::metrics() const {
    auto result = moe_expert_cache_has_sources(data_->moe) ? moe_expert_memory_metrics(data_->moe)
        : std::vector<std::pair<std::string, double>>{};
    const auto merge = [&](auto counters) {
        for (auto& [name, value] : counters) {
            const auto match = std::find_if(result.begin(), result.end(), [&](const auto& entry) { return entry.first == name; });
            if (match == result.end()) result.emplace_back(std::move(name), value);
            else if (name.ends_with("_enabled")) match->second = std::max(match->second, value);
            else match->second += value;
        }
    };
    if (data_->tq) merge(data_->tq->metrics());
    if (data_->ggml) merge(data_->ggml->metrics());
    append_workspace_metrics(result);
    return result;
}

void Cache::report(std::ostream& output) const {
    if (data_->moe) print_moe_expert_cache_stats(data_->moe, output);
}

Cache::Activity Cache::activity() const {
    const auto gguf = data_->ggml ? data_->ggml->streamed_bytes() : 0;
    const auto tq = data_->tq ? data_->tq->streamed_bytes() : 0;
    return {gguf + tq + moe_expert_streamed_bytes(data_->moe), moe_expert_graph_captures(data_->moe)};
}

std::size_t Cache::budget() const {
    return (data_->tq ? data_->tq->budget() : 0) +
        (data_->ggml ? data_->ggml->budget() : 0) +
        (moe_expert_cache_has_sources(data_->moe) ? moe_expert_cache_budget_bytes(data_->moe) : 0);
}

std::size_t Cache::minimum() const {
    return moe_expert_cache_minimum_bytes(data_->moe);
}

bool Cache::empty() const {
    return !data_->moe && !data_->tq && !data_->ggml;
}

Cache::Routing Cache::routing() const {
    return {moe_expert_cache_has_sources(data_->moe), bool(data_->tq),
        data_->ggml && data_->ggml->host_routes_required(),
        moe_expert_cache_ssd_experts(data_->moe)};
}

bool Cache::legacy_experts_resident() const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto& cache = data_->moe;
    return cache && cache->has_sources() && cache->finalized() &&
        cache->complete_residency() && cache->budget_bytes() > 0;
#else
    return false;
#endif
}

Cache::PrefillBudget Cache::prefill_budget(bool allow_borrow) const {
    const auto budget = moe_prefill_workspace_budget(data_->moe, allow_borrow);
    const auto tq_bytes = allow_borrow && data_->tq && !moe_expert_cache_has_sources(data_->moe)
        ? data_->tq->lendable_bytes(mfq_current_cuda_device()) : 0;
    const auto reserve = moe_expert_cache_ssd_experts(data_->moe)
        ? std::max(std::size_t(1) << 30, cuda::options::workspace_reserve_bytes())
        : std::size_t(1) << 30;
    return {reserve, budget.lendable_bytes + tq_bytes, budget.transfer_bytes};
}

void Cache::workspace(std::size_t bytes) {
    if (data_->ggml) data_->ggml->reclaim_device(Device(kCUDA, mfq_current_cuda_device()), bytes);
    if (data_->tq) {
        int device = 0;
        TYLOINFER_CUDA_CHECK(cudaGetDevice(&device));
        data_->tq->reclaim(device, bytes);
    }
    reserve_moe_workspace(data_->moe, bytes);
}

void Cache::request(bool retire_decode) {
    finish_exchange();
    if (retire_decode) release_moe_decode_buffers(data_->moe);
    else retire_prefill();
}

std::size_t Cache::storage_workspace_reserve() const {
    return cuda::options::workspace_reserve_bytes();
}

bool Cache::lend_decode_storage(std::size_t workspace) {
    return lend_moe_decode_cache_for_prefill(data_->moe, workspace);
}

void Cache::prepare(Stage stage, Device device, const source::Source* source) {
    if (stage == Stage::snapshot) {
        retire_prefill();
        return;
    }
    if (data_->ggml) data_->ggml->reclaim_device(device, cuda::options::workspace_reserve_bytes());
    if (data_->tq) {
        const auto reserve=cuda::options::workspace_reserve_bytes();
        if (stage == Stage::loaded)
            data_->tq->preload(device.index, reserve);
        else if (stage == Stage::decode)
            data_->tq->prepare_decode(device.index, reserve);
    }
    if (stage == Stage::decode && !data_->moe) return;
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("budget requires a CUDA device");
    prepare_moe_decode_residency(data_->moe);
    cuda::default_context(device.index)->reclaim_idle_memory();
    restore_workspace(cuda::options::workspace_reserve_bytes(), source);
}

void Cache::resize(std::size_t bytes) {
    const bool legacy = moe_expert_cache_has_sources(data_->moe);
    const bool tq = data_->tq && data_->tq->has_cached_sources();
    const bool gguf = data_->ggml && data_->ggml->has_cached_sources();
    const auto required = minimum();
    if (bytes < required || bytes > std::size_t(INT64_MAX))
        throw std::invalid_argument("invalid routed weight cache budget");
    const auto groups = std::size_t(legacy) + std::size_t(tq) + std::size_t(gguf);
    const auto share = groups ? (bytes - required) / groups : 0;
    auto extra = groups ? (bytes - required) % groups : 0;
    const auto grant = [&](bool active) {
        if (!active) return std::size_t{0};
        if (!extra) return share;
        --extra;
        return share + 1;
    };
    const auto old_tq = data_->tq ? data_->tq->budget() : 0;
    const auto old_gguf = data_->ggml ? data_->ggml->budget() : 0;
    const auto next_tq = grant(tq);
    const auto next_gguf = grant(gguf);
    const auto next_legacy = grant(legacy);

    // Shrink before admitting the new MFE arenas so growth fits the aggregate ceiling.
    if (data_->tq && old_tq > next_tq) data_->tq->resize(next_tq);
    if (data_->ggml && old_gguf > next_gguf) data_->ggml->resize(next_gguf);
    try {
        if (legacy) resize_moe_expert_cache(data_->moe, required + next_legacy);
    } catch (...) {
        if (data_->tq && old_tq > next_tq) data_->tq->resize(old_tq);
        if (data_->ggml && old_gguf > next_gguf) data_->ggml->resize(old_gguf);
        throw;
    }
    if (data_->tq && old_tq < next_tq) data_->tq->resize(next_tq);
    if (data_->ggml && old_gguf < next_gguf) data_->ggml->resize(next_gguf);
    data_->device_bytes = bytes;
}

std::size_t Cache::reserve(std::size_t bytes) {
    auto reclaimed = reclaim_host(bytes, data_->host_limit);
    if (data_->tq) reclaimed += data_->tq->reserve(bytes);
    return reclaimed + reclaim_moe_optional_host(data_->moe, bytes);
}

std::size_t Cache::reclaim_host_duplicates(std::size_t bytes) {
    std::size_t reclaimed = 0;
    if (data_->tq) reclaimed = data_->tq->reclaim_duplicates(bytes);
    if (reclaimed < bytes && data_->ggml) reclaimed += data_->ggml->reclaim_duplicates(bytes - reclaimed);
    if (reclaimed < bytes) reclaimed += reclaim_moe_expert_duplicates(data_->moe, bytes - reclaimed);
    return reclaimed;
}

std::size_t Cache::reserve_host_capacity(std::size_t bytes) {
    const auto reclaimed = reclaim_host(bytes, data_->host_limit);
    return reclaimed + reserve_moe_expert_host(data_->moe, bytes);
}

void Cache::share_host(std::size_t bytes) {
    if (data_->ggml) data_->ggml->share_host(bytes);
    if (data_->tq) data_->tq->share_host(bytes);
    else share_moe_expert_host(data_->moe, bytes);
}

void Cache::drain() const {
    drain_moe_expert_cache(data_->moe);
}

void Cache::finish_exchange() const {
    finish_moe_expert_exchanges(data_->moe);
}

void Cache::admit_routes(Device device) {
    if (data_->tq && device.is_cuda())
        data_->tq->admit_routes(device.index, cuda::options::workspace_reserve_bytes());
    if (data_->ggml && device.is_cuda())
        data_->ggml->admit_routes(device, cuda::options::workspace_reserve_bytes());
}

void Cache::retire_prefill() {
    release_moe_prefill_buffers(data_->moe);
}

void Cache::discard_prefill() {
    discard_moe_prefill_routes(data_->moe);
}

void Cache::settle() {
    release_workspace();
    finish_exchange();
}

void Cache::trace(const char* boundary, bool wait, const source::Source* source) {
    std::cerr << "moe_request_cache_stats boundary=" << boundary << '\n';
    if (wait) print_moe_expert_cache_stats(data_->moe, std::cerr);
    std::cerr << "moe_phase_metrics boundary=" << boundary;
    for (const auto& [name, value] : metrics())
        if (name == "ssd_expert_read_wait_ms" || name == "expert_route_wait_ms" ||
            name == "expert_host_fetch_ms" || name == "expert_host_copy_ms" ||
            name == "expert_transfer_cache_prepare_ms" || name == "ssd_expert_bytes_read" ||
            name == "ssd_expert_read_count" || name == "ram_expert_cache_hits" ||
            name == "ram_expert_cache_misses" || name == "ram_expert_pcie_read_bytes" ||
            name == "expert_logical_transfer_bytes") std::cerr << ' ' << name << '=' << value;
    std::cerr << '\n';
    if (source) {
        const auto io = source->file_read_stats();
        std::cerr << "runtime_model_io_stats boundary=" << boundary << " calls=" << io.calls
                  << " logical_bytes=" << io.logical_bytes << " physical_bytes=" << io.physical_bytes
                  << " read_ms=" << double(io.read_nanoseconds) / 1e6 << '\n';
    }
}

void Cache::reset() noexcept {
    clear_workspace();
    data_->moe.reset();
    data_->tq.reset();
    data_->ggml.reset();
    data_->device_bytes.reset();
    data_->exec.reset();
    data_->kv.reset();
}

} // namespace tyloi::weights
