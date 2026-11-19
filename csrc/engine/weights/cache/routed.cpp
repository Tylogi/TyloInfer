#include "engine/weights/cache/resources.h"
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
    int rows = 0;

    void discard() noexcept {
        discard_moe_speculative_routes(cache);
        if (tq) tq->discard_routes();
        cache.reset();
        tq.reset();
        rows = 0;
    }
};

Cache::Routes::Routes() noexcept = default;
Cache::Routes::Routes(Routes&&) noexcept = default;
Cache::Routes::~Routes() {
    if (impl_) impl_->discard();
}

void Cache::Routes::begin(Cache& owner, int rows) {
    if (!impl_) impl_ = std::make_unique<Impl>();
    if (impl_->cache || impl_->tq)
        throw std::logic_error("speculative routes already pending");
    auto cache = owner.data_->moe;
    auto tq = owner.data_->tq;
    try {
        begin_moe_speculative_routes(cache, rows);
        if (tq) tq->begin_routes(rows);
    } catch (...) {
        discard_moe_speculative_routes(cache);
        throw;
    }
    impl_->cache = std::move(cache);
    impl_->tq = std::move(tq);
    impl_->rows = rows;
}

void Cache::Routes::resolve(int rows, bool cancelled) {
    if (!impl_ || (!impl_->cache && !impl_->tq)) return;
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
    if (!impl_ || (!impl_->cache && !impl_->tq)) return;
    try {
        if (keep.size() != std::size_t(impl_->rows))
            throw std::invalid_argument("speculative route mask row count is out of range");
        commit_moe_speculative_routes(impl_->cache, keep, single_sequence);
        if (impl_->tq) impl_->tq->commit_routes(keep, single_sequence);
    } catch (...) {
        impl_->discard();
        throw;
    }
    impl_->cache.reset();
    impl_->tq.reset();
    impl_->rows = 0;
}

struct Cache::Warmup::Impl {
    std::shared_ptr<MoeExpertCache> cache;
    std::shared_ptr<TqCache> tq;
    bool suspended = false;
    bool previous_residency = false;
    bool previous_learning = false;
    bool previous_tq_learning = false;
};

Cache::Warmup::Warmup(Cache& owner, bool suspend_residency) : impl_(std::make_unique<Impl>()) {
    impl_->cache = owner.data_->moe;
    impl_->tq = owner.data_->tq;
    impl_->suspended = suspend_residency;
    if (suspend_residency) impl_->previous_residency = suspend_moe_residency(impl_->cache, true);
    impl_->previous_learning = set_moe_reuse_learning(impl_->cache, false);
    try {
        if (impl_->tq) impl_->previous_tq_learning = impl_->tq->set_learning(false);
    } catch (...) {
        set_moe_reuse_learning(impl_->cache, impl_->previous_learning);
        if (impl_->suspended) suspend_moe_residency(impl_->cache, impl_->previous_residency);
        throw;
    }
}

Cache::Warmup::~Warmup() {
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
    auto result = data_->tq ? data_->tq->metrics() : moe_expert_memory_metrics(data_->moe);
    append_workspace_metrics(result);
    return result;
}

void Cache::report(std::ostream& output) const {
    if (data_->moe) print_moe_expert_cache_stats(data_->moe, output);
}

Cache::Activity Cache::activity() const {
    if (data_->tq) return {data_->tq->streamed_bytes(), 0};
    return {moe_expert_streamed_bytes(data_->moe), moe_expert_graph_captures(data_->moe)};
}

std::size_t Cache::budget() const {
    return data_->tq ? data_->tq->budget() : moe_expert_cache_budget_bytes(data_->moe);
}

std::size_t Cache::minimum() const {
    return moe_expert_cache_minimum_bytes(data_->moe);
}

bool Cache::empty() const {
    return !data_->moe && !data_->tq;
}

bool Cache::graph_safe(Graph kind, bool parallel) const {
    if (kind == Graph::model_parallel) return !(data_->tq && parallel);
    return !data_->moe && !(data_->tq && (parallel || kind == Graph::legacy));
}

Cache::Transfer Cache::transfer() const {
    return data_->moe ? Transfer::coordinated : Transfer::automatic;
}

Cache::Decode Cache::decode() const {
    return data_->tq ? Decode::routed : Decode::none;
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

bool Cache::split_prefill(std::int64_t rows) const {
    return !data_->tq || rows <= 8;
}

bool Cache::prepare_graph(Device device) {
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("graph requires a CUDA device");
    if (prepare_moe_graph_pool(data_->moe)) return true;
    if (!data_->tq) return false;
    data_->exec->get(device.index)->prepare_graph();
    return true;
}

Cache::Prefill Cache::prefill_mode(std::size_t tokens, bool resumed) const {
    if (resumed && tokens <= 128 && moe_expert_cache_ssd_experts(data_->moe)) return Prefill::retained;
    if (tokens > 8 && moe_expert_cache_ssd_experts(data_->moe)) return Prefill::streamed;
    return Prefill::regular;
}

Cache::PrefillBudget Cache::prefill_budget(bool allow_borrow) const {
    if(data_->tq)return {std::size_t(1)<<30,allow_borrow?data_->tq->resident_bytes():0,0};
    const auto budget = moe_prefill_workspace_budget(data_->moe, allow_borrow);
    const auto reserve = moe_expert_cache_ssd_experts(data_->moe)
        ? std::max(std::size_t(1) << 30, cuda::options::workspace_reserve_bytes())
        : std::size_t(1) << 30;
    return {reserve, budget.lendable_bytes, budget.transfer_bytes};
}

void Cache::workspace(std::size_t bytes) {
    if (data_->tq) {
        int device = 0;
        TYLOINFER_CUDA_CHECK(cudaGetDevice(&device));
        data_->tq->reclaim(device, bytes);
        return;
    }
    reserve_moe_workspace(data_->moe, bytes);
}

void Cache::request(bool retire_decode) {
    finish_exchange();
    if (retire_decode) release_moe_decode_buffers(data_->moe);
    else retire_prefill();
}

std::optional<std::size_t> Cache::prepare_prefill(Device device, std::size_t tokens,
        std::int64_t limit, Prefill mode, bool retire_decode) {
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("budget requires a CUDA device");
    // TQ lends the concrete workspace requested by the prefill plan.
    if(data_->tq)return std::nullopt;
    if (mode == Prefill::retained) {
        std::cerr << "runtime_streamed_prefill retained_decode=1 tokens=" << tokens
                  << " chunk=" << std::min<std::int64_t>(4, limit) << '\n';
        return std::nullopt;
    }
    if (mode == Prefill::streamed) {
        const bool retained = lend_moe_decode_cache_for_prefill(data_->moe, cuda::options::workspace_reserve_bytes());
        if (!retained && retire_decode) {
            release_moe_decode_buffers(data_->moe);
            cuda::default_context(device.index)->reclaim_idle_memory();
        }
    }
    if (tokens > 1024 || moe_expert_cache_ssd_experts(data_->moe))
        return cuda::options::workspace_reserve_bytes();
    return std::nullopt;
}

void Cache::prepare(Stage stage, Device device, const source::Source* source) {
    if (stage == Stage::snapshot) {
        retire_prefill();
        return;
    }
    if (data_->tq) {
        const auto reserve=cuda::options::workspace_reserve_bytes();
        if (stage == Stage::loaded)
            data_->tq->preload(device.index, reserve);
        else if (stage == Stage::decode)
            data_->tq->prepare_decode(device.index, reserve);
        return;
    }
    if (stage == Stage::decode && !data_->moe) return;
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("budget requires a CUDA device");
    prepare_moe_decode_residency(data_->moe);
    cuda::default_context(device.index)->reclaim_idle_memory();
    restore_workspace(cuda::options::workspace_reserve_bytes(), source);
}

void Cache::resize(std::size_t bytes) {
    if (data_->tq) data_->tq->resize(bytes);
    else resize_moe_expert_cache(data_->moe, bytes);
}

std::size_t Cache::reserve(std::size_t bytes) {
    reclaim_host(bytes, data_->host_limit);
    return data_->tq ? data_->tq->reserve(bytes) : reclaim_moe_optional_host(data_->moe, bytes);
}

std::size_t Cache::reclaim_host_duplicates(std::size_t bytes) {
    return data_->tq ? data_->tq->reclaim_duplicates(bytes) : reclaim_moe_expert_duplicates(data_->moe, bytes);
}

std::size_t Cache::reserve_host_capacity(std::size_t bytes) {
    reclaim_host(bytes, data_->host_limit);
    return data_->tq ? data_->tq->reserve(bytes) : reserve_moe_expert_host(data_->moe, bytes);
}

void Cache::share_host(std::size_t bytes) {
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
    data_->device_bytes.reset();
    data_->exec.reset();
    data_->kv.reset();
}

} // namespace tyloi::weights
