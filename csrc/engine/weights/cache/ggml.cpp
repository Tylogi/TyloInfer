#include "engine/weights/cache/ggml.h"

#include "components/error.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/native/dma_batch.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/weights/moe/prepare/projection.h"
#include "engine/weights/moe/residency/retention.h"
#include "engine/weights/moe/residency/routes.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <unordered_set>

namespace tyloi::weights {
namespace {

std::size_t host_headroom(std::size_t process_limit) {
    try {
        return sample_host_memory().headroom(process_limit);
    } catch (const std::runtime_error&) {
        return 0;
    }
}

} // namespace

struct GgmlExpertCache::Routes {
    struct Bundle {
        std::array<std::uint32_t, 3> sources;
        std::size_t first, count;
    };
    engine::RouteHistory history{{}, engine::RouteHistory::HeatMode::count, engine::expert_usage_retention};
    std::vector<Bundle> bundles;
    std::vector<ExpertRetention> experts;
    std::unordered_set<std::uint64_t> protected_images;
    std::optional<std::size_t> budget;
    std::uint64_t windows = 0;
    bool sampled = false;
};

struct GgmlExpertCache::Prefill {
    struct Sample {
        std::vector<std::int32_t> counts;
        int rows = 0, routes = 0;
    };
    std::vector<int> offsets{0};
    std::vector<Sample> samples;
};

GgmlExpertCache::HostImage::HostImage(DenseTensor value, std::atomic_size_t& counter)
    : tensor(std::move(value)), physical(counter) {
    physical.fetch_add(tensor.nbytes(), std::memory_order_relaxed);
}

GgmlExpertCache::HostImage::~HostImage() {
    const auto bytes = tensor.nbytes();
    tensor = {};
    physical.fetch_sub(bytes, std::memory_order_relaxed);
}

GgmlExpertCache::GgmlExpertCache(std::size_t device_bytes, std::size_t host_bytes, std::size_t host_limit)
    : host_(host_bytes ? host_bytes : host_headroom(host_limit) / 5 * 4),
      device_(device_bytes), host_limit_(host_limit), ordinary_host_budget_(*host_.limit()) {}

GgmlExpertCache::~GgmlExpertCache() {
    for (const auto& upload : staging_) {
        if (!upload.ready) continue;
        try { upload.ready->synchronize(); } catch (const cuda::Error&) {}
    }
}

std::uint32_t GgmlExpertCache::register_source(const std::shared_ptr<const GgmlExpertBank>& bank,
        source::Source::TensorReader read, source::Source::TensorBatchReader read_batch) {
    std::lock_guard lock(mutex_);
    const bool resident = bank->image.packed.defined();
    const auto bytes = resident ? bank->backing_nbytes() :
        static_cast<std::size_t>(bank->expert_bytes * bank->experts);
    auto& total = resident ? permanent_bytes_ : source_bytes_;
    if (sources_.size() == UINT32_MAX || bytes > SIZE_MAX - total)
        throw std::overflow_error("GGML expert sources exceed cache limits");
    sources_.push_back({bank, std::move(read), std::move(read_batch)});
    total += bytes;
    host_routes_required_ |= !bank->resident_addresses.defined();
    return static_cast<std::uint32_t>(sources_.size());
}

void GgmlExpertCache::observe_routes(std::span<const GgmlExpertBank* const> banks, const DenseTensor& ids,
        int row_offset, int row_total, bool prefill) {
    if (banks.size() < 2 || banks.size() > 3 || !ids.defined() || !ids.is_cpu() || ids.scalar_type() != kInt32 ||
        ids.dim() != 2 || !ids.is_contiguous()) throw std::invalid_argument("invalid GGML route sample");
    std::lock_guard lock(mutex_);
    if (!*device_.limit() || (!learning_ && (!routes_ || !routes_->history.pending()))) return;
    if (prefill_ && prefill_->offsets.back() == 0) return;
    if (prefill && !prefill_) return;
    std::array<std::uint32_t, 3> sources{};
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < banks.size(); ++i) {
        const auto* bank = banks[i];
        if (!bank || bank->cache.get() != this || bank->image.packed.defined()) return;
        if (std::find(sources.begin(), sources.begin() + i, bank->source_id) != sources.begin() + i)
            return;
        if (std::size_t(bank->expert_bytes) > SIZE_MAX - bytes)
            throw std::overflow_error("GGML expert group size overflows");
        bytes += bank->expert_bytes;
        sources[i] = bank->source_id;
    }
    if (!routes_) routes_ = std::make_unique<Routes>();
    auto& route = *routes_;
    const auto selected = std::find_if(route.bundles.begin(), route.bundles.end(),
        [&](const auto& group) { return group.sources == sources; });
    const auto bundle = std::size_t(selected - route.bundles.begin());
    if (selected == route.bundles.end()) {
        for (const auto& group : route.bundles)
            for (const auto source : sources)
                if (source && std::find(group.sources.begin(), group.sources.end(), source) != group.sources.end())
                    return;
        const auto count = std::size_t(banks.front()->experts);
        route.bundles.reserve(route.bundles.size() + 1);
        route.experts.reserve(route.experts.size() + count);
        route.history.add_bundle({static_cast<int>(count), true});
        route.bundles.push_back({sources, route.experts.size(), count});
        for (std::size_t expert = 0; expert < count; ++expert)
            route.experts.push_back({(std::uint64_t(sources.front()) << 32) | expert,
                banks.front()->target.index, bytes, 0, false});
    }
    const std::span ids_view{ids.data_ptr<std::int32_t>(), std::size_t(ids.numel())};
    if (prefill_) {
        const auto& offsets = prefill_->offsets;
        const auto expected = offsets.back();
        const auto count = route.bundles[bundle].count;
        prefill_->samples.resize(route.bundles.size());
        auto& sample = prefill_->samples[bundle];
        const auto rows = static_cast<int>(ids.size(0)), width = static_cast<int>(ids.size(1));
        if (sample.rows < 0) return;
        if (row_total != expected || row_offset != sample.rows ||
            rows <= 0 || width <= 0 || width > INT32_MAX / expected ||
            (sample.routes && width != sample.routes) || rows > expected - sample.rows) {
            sample.rows = -1;
            return;
        }
        if (sample.counts.empty()) sample.counts.resize((offsets.size() - 1) * count);
        const int end = row_offset + rows;
        auto request = std::size_t(std::upper_bound(offsets.begin(), offsets.end(), row_offset) - offsets.begin() - 1);
        for (int row = row_offset; row < end; ++request) {
            const auto next = std::min(end, offsets[request + 1]);
            assert(next > row);
            for (const auto id : ids_view.subspan(std::size_t(row - row_offset) * width, std::size_t(next - row) * width)) {
                if (id < 0 || std::size_t(id) >= count) throw std::out_of_range("GGML prompt expert ID");
                ++sample.counts[request * count + id];
            }
            row = next;
        }
        sample.rows = end;
        sample.routes = width;
        return;
    }
    route.history.observe(bundle, ids_view, static_cast<int>(ids.size(0)), false, row_offset, row_total, learning_);
    if (!route.history.pending()) route.sampled = true;
}

std::shared_ptr<GgmlExpertCache::Prefill> GgmlExpertCache::begin_prefill(std::span<const int> request_rows) {
    auto journal = std::make_shared<Prefill>();
    for (const auto rows : request_rows) {
        if (rows <= 0 || rows > INT32_MAX - journal->offsets.back())
            throw std::invalid_argument("invalid GGML prefill row count");
        journal->offsets.push_back(journal->offsets.back() + rows);
    }
    std::lock_guard lock(mutex_);
    if (prefill_ || (routes_ && routes_->history.pending()))
        throw std::logic_error("invalid GGML prompt route journal");
    prefill_ = journal.get();
    return journal;
}

void GgmlExpertCache::detach_prefill(const Prefill& journal) noexcept {
    std::lock_guard lock(mutex_);
    if (prefill_ == &journal) prefill_ = nullptr;
}

void GgmlExpertCache::finish_prefill(Prefill& journal, std::span<const std::uint8_t> accepted) {
    const auto& offsets = journal.offsets;
    if (offsets.size() == 1) return;
    if (accepted.size() != offsets.size() - 1)
        throw std::invalid_argument("GGML prefill acceptance count differs from its requests");
    int rows = 0;
    for (std::size_t i = 0; i < accepted.size(); ++i)
        if (accepted[i]) rows += offsets[i + 1] - offsets[i];
    if (!rows) return;
    std::lock_guard lock(mutex_);
    if (!learning_ || journal.samples.empty()) return;
    auto& route = *routes_;
    const auto first = std::size_t(std::find_if(accepted.begin(), accepted.end(),
        [](std::uint8_t keep) { return keep != 0; }) - accepted.begin());
    assert(first < accepted.size());
    for (std::size_t i = 0; i < journal.samples.size(); ++i) {
        auto& sample = journal.samples[i];
        if (sample.rows != offsets.back()) continue;
        const auto count = route.bundles[i].count;
        const auto total = std::span(sample.counts).subspan(first * count, count);
        for (std::size_t request = first + 1; request < accepted.size(); ++request)
            if (accepted[request]) {
                const auto selected = sample.counts.data() + request * count;
                std::transform(total.begin(), total.end(), selected, total.begin(), std::plus<>{});
            }
        route.history.prefill_counts(i, total, rows, sample.routes);
        route.windows = engine::expert_usage_interval;
    }
}

void GgmlExpertCache::begin_routes(int rows) {
    std::lock_guard lock(mutex_);
    if (!routes_) routes_ = std::make_unique<Routes>();
    if (prefill_) throw std::logic_error("GGML prompt routes are pending");
    routes_->history.begin(rows);
}

void GgmlExpertCache::commit_routes(std::span<const std::uint8_t> keep, bool single_sequence) {
    std::lock_guard lock(mutex_);
    if (routes_ && routes_->history.commit(keep, single_sequence, learning_).heat_routes)
        routes_->sampled = true;
}

void GgmlExpertCache::discard_routes() noexcept {
    std::lock_guard lock(mutex_);
    if (routes_) routes_->history.discard();
}

bool GgmlExpertCache::set_learning(bool enabled) {
    std::lock_guard lock(mutex_);
    return std::exchange(learning_, enabled);
}

void GgmlExpertCache::admit_routes(Device device, std::size_t workspace_reserve) {
    std::unique_lock lock(mutex_);
    if (!routes_ || !learning_ || prefill_ || routes_->history.pending()) return;
    auto& route = *routes_;
    if (std::exchange(route.sampled, false)) ++route.windows;
    const auto budget = *device_.limit();
    if (route.windows < engine::expert_usage_interval && route.budget == budget) return;
    route.history.flush();
    auto items = route.experts;
    for (std::size_t i = 0; i < route.bundles.size(); ++i) {
        const auto& group = route.bundles[i];
        for (std::size_t expert = 0; expert < group.count; ++expert)
            items[group.first + expert].heat = route.history.heat(i, static_cast<int>(expert));
    }
    select_expert_retention(items, budget, route.budget != budget);
    decltype(route.protected_images) retained;
    std::array<std::uint32_t, 3> warm_sources{};
    std::int64_t warm_expert = 0;
    float warm_heat = 0;
    for (const auto& group : route.bundles) {
        for (std::size_t expert = 0; expert < group.count; ++expert) {
            const auto& item = items[group.first + expert];
            if (!item.retained) continue;
            auto missing = group.sources;
            bool incomplete = false;
            for (auto& source : missing) {
                if (!source) continue;
                const auto key = (std::uint64_t(source) << 32) | expert;
                retained.insert(key);
                if (device_.find(key)) source = 0;
                else incomplete = true;
            }
            if (incomplete && item.device == device.index && item.heat > warm_heat) {
                warm_sources = missing;
                warm_expert = static_cast<std::int64_t>(expert);
                warm_heat = item.heat;
            }
        }
    }
    route.protected_images.swap(retained);
    route.experts.swap(items);
    if (route.windows) route.history.decay(engine::expert_usage_decay(route.windows));
    route.windows = 0;
    route.budget = budget;

    std::array<std::shared_ptr<const GgmlExpertBank>, 3> warm_banks;
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < warm_sources.size(); ++i) {
        if (const auto source = warm_sources[i]; source) {
            warm_banks[i] = sources_.at(source - 1).bank.lock();
            if (!warm_banks[i]) return;
            bytes += warm_banks[i]->expert_bytes;
        }
    }
    lock.unlock();
    if (!bytes || bytes > SIZE_MAX - workspace_reserve) return;
    const auto context = cuda::default_context(device.index);
    if (context->available_memory() < bytes + workspace_reserve) return;
    // One group per placement update bounds extra IO on the decode stream.
    try {
        for (const auto& bank : warm_banks)
            if (bank) prepare(*bank, {&warm_expert, 1});
    } catch (const MfqResourceExhausted&) {}
}

PreparedExperts GgmlExpertCache::prepare(const GgmlExpertBank& bank, std::span<const std::int64_t> ids,
        bool graph_upload) {
    if (!bank.defined() || bank.cache.get() != this || bank.expert_bytes <= 0 ||
        std::any_of(ids.begin(), ids.end(), [&](auto id) { return id < 0 || id >= bank.experts; }))
        throw std::out_of_range("GGUF routed expert ID is out of bounds");
    PreparedExperts result;
    result.device = bank.device();
    auto& images = result.ggml;
    images.assign(ids.size(), bank.image);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        auto& image = images[i];
        if (image.correction) {
            image.correction->first = ids[i] * image.out;
            image.correction->rows = image.out;
        }
        if (image.packed.defined())
            image.packed = image.packed.narrow(0, ids[i] * bank.expert_bytes, bank.expert_bytes);
    }
    if (bank.image.packed.defined()) return result;

    const auto key = [&](std::size_t i) {
        return (std::uint64_t(bank.source_id) << 32) | static_cast<std::uint32_t>(ids[i]);
    };
    const auto bytes = static_cast<std::size_t>(bank.expert_bytes);
    const auto use = [&](std::size_t i, const DeviceImage& image) {
        images[i].packed = image.tensor;
        if (image.ready) result.ready_.insert(image.ready);
    };
    std::vector<std::shared_ptr<const HostImage>> host(ids.size());
    std::vector<std::size_t> missing;
    Reader reader;
    {
        std::lock_guard lock(mutex_);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (const auto* image = device_.get(key(i))) {
                ++device_hits_;
                use(i, *image);
            } else if (const auto* image = host_.get(key(i))) {
                ++host_hits_;
                host[i] = *image;
            } else missing.push_back(i);
        }
        if (!missing.empty()) reader = sources_.at(bank.source_id - 1);
    }
    if (!missing.empty()) {
        reclaim_host(missing.size() * bytes, host_limit_);
        std::vector<ReadSpan> reads;
        reads.reserve(missing.size());
        for (const auto i : missing) {
            auto tensor = cuda::empty({bank.expert_bytes}, TensorOptions{}.device(kCPU).dtype(kUInt8));
            reads.push_back({static_cast<std::uint64_t>(ids[i] * bank.expert_bytes),
                tensor.data_ptr<std::byte>(), bytes});
            host[i] = std::make_shared<HostImage>(std::move(tensor), host_physical_);
        }
        if (reader.read_batch && reads.size() > 1) reader.read_batch(reads);
        else for (const auto& read : reads) reader.read(read.offset, read.destination, read.size);

        constexpr auto reserve = std::size_t(2) << 30;
        const bool room = host_headroom(host_limit_) >= reserve;
        std::lock_guard lock(mutex_);
        reads_ += reads.size();
        fetched_ += reads.size() * bytes;
        if (room && bytes <= *host_.limit())
            for (const auto i : missing)
                if (!host_.find(key(i))) host_.put(key(i), host[i], bytes);
    }

    std::vector<std::size_t> uploads;
    {
        std::lock_guard lock(mutex_);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (images[i].packed.defined()) continue;
            if (const auto* image = device_.get(key(i))) use(i, *image);
            else uploads.push_back(i);
        }
    }
    if (uploads.empty()) return result;

    const cuda::DeviceGuard guard(bank.target.index);
    const auto stream = cuda::stream_from_pool(false, bank.target.index);
    const cuda::StreamGuard transfer(stream);
    std::size_t retained = 0;
    {
        std::lock_guard lock(mutex_);
        const auto limit = *device_.limit();
        retained = std::min(uploads.size(), limit / bytes);
        if (!routes_ || routes_->protected_images.empty()) device_.reserve(retained * bytes);
        else {
            const auto& protected_images = routes_->protected_images;
            std::stable_partition(uploads.begin(), uploads.end(),
                [&](auto i) { return protected_images.contains(key(i)); });
            while (device_.used() > limit - retained * bytes) {
                const auto* victim = device_.oldest(protected_images);
                if (!victim) break;
                const auto old_key = *victim;
                device_.evict(old_key);
            }
            retained = std::min(retained, (limit - device_.used()) / bytes);
        }
    }
    std::shared_ptr<cuda::Event> ready;
    const auto packet_bytes = uploads.size() * bytes;
    if (packet_bytes <= (std::size_t(64) << 20)) {
        std::lock_guard lock(staging_mutex_);
        const auto slot = std::find_if(staging_.begin(), staging_.end(), [](const Upload& upload) {
            return !upload.ready || upload.ready->ready();
        });
        if (slot != staging_.end()) {
            if ((!slot->packet.defined() || slot->packet.nbytes() < packet_bytes) &&
                host_headroom(host_limit_) >= packet_bytes) {
                try {
                    slot->packet = cuda::empty({static_cast<std::int64_t>(packet_bytes)},
                        TensorOptions{}.device(kCPU).dtype(kUInt8).pinned_memory(true));
                } catch (const MfqResourceExhausted&) {}
            }
            if (slot->packet.defined() && slot->packet.nbytes() >= packet_bytes) {
                ready = std::make_shared<cuda::Event>();
                try {
                    auto* packet = static_cast<std::byte*>(slot->packet.data_ptr());
                    std::vector<cuda::DmaCopy> copies;
                    copies.reserve(uploads.size());
                    std::size_t offset = 0;
                    for (const auto i : uploads) {
                        // Graph address tables are patched at replay, so these images can change size and address.
                        const cuda::RoutedWeightAllocationScope allocation(graph_upload || copies.size() < retained);
                        auto& target = images[i].packed;
                        target = cuda::empty({bank.expert_bytes}, TensorOptions{}.device(bank.target).dtype(kUInt8));
                        std::memcpy(packet + offset, host[i]->tensor.data_ptr(), bytes);
                        copies.push_back({target.data_ptr(), packet + offset, bytes});
                        offset += bytes;
                    }
                    assert(offset == packet_bytes);
                    const auto* option = cuda::env_lookup("TYLOINFER_MOE_NATIVE_DMA_BATCH");
                    const auto mode = option ? std::clamp(std::atoi(option), 0, 2) : 2;
                    cudaPointerAttributes attributes{};
                    if (mode && copies.size() > 1 && cuda::NativeDmaBatch::available())
                        TYLOINFER_NATIVE_CUDA_CHECK(cudaPointerGetAttributes(&attributes, packet));
                    const auto base = reinterpret_cast<CUdeviceptr>(attributes.devicePointer);
                    if (!base || !cuda::NativeDmaBatch::submit(copies, stream.stream(), mode,
                            [packet, base](const void* source, std::size_t) {
                                return base + (static_cast<const std::byte*>(source) - packet);
                            })) {
                        for (const auto& copy : copies)
                            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(copy.destination, copy.source,
                                copy.bytes, cudaMemcpyHostToDevice, stream.stream()));
                    }
                    ready->record(stream.stream());
                    slot->ready = ready;
                } catch (...) {
                    if (cudaStreamSynchronize(stream.stream()) != cudaSuccess) std::terminate();
                    throw;
                }
            }
        }
    }
    if (ready) result.ready_.insert(ready);
    else {
        for (std::size_t slot = 0; slot < uploads.size(); ++slot) {
            const auto i = uploads[slot];
            const cuda::RoutedWeightAllocationScope allocation(graph_upload || slot < retained);
            images[i].packed = host[i]->tensor.to(bank.target);
        }
    }
    std::lock_guard lock(mutex_);
    transferred_ += uploads.size() * bytes;
    for (std::size_t slot = 0; slot < retained; ++slot) {
        const auto i = uploads[slot];
        if (bytes <= *device_.limit() - device_.used() && !device_.find(key(i)))
            device_.put(key(i), DeviceImage{images[i].packed, ready}, bytes);
    }
    return result;
}

void GgmlExpertCache::resize(std::size_t bytes) {
    std::lock_guard lock(mutex_);
    device_.limit(bytes);
}

void GgmlExpertCache::reclaim_device(Device device, std::size_t available) {
    std::lock_guard lock(mutex_);
    const auto context = cuda::default_context(device.index);
    context->reclaim_idle_memory();
    for (auto it = device_.keys().end(); it != device_.keys().begin();) {
        if (context->available_memory() >= available) break;
        const auto key = *--it;
        if (device_.at(key).tensor.device() != device) continue;
        it = std::next(it);
        device_.evict(key);
        context->reclaim_idle_memory();
    }
}

void GgmlExpertCache::share_host(std::size_t available) {
    std::lock_guard lock(mutex_);
    host_.limit(shared_host_cache_limit(ordinary_host_budget_, host_.used(),
        host_headroom(host_limit_), available));
}

std::size_t GgmlExpertCache::reclaim(std::size_t bytes) {
    std::size_t staged = 0;
    {
        std::lock_guard lock(staging_mutex_);
        for (auto& upload : staging_) {
            if (staged >= bytes) break;
            if (upload.packet.defined() && (!upload.ready || upload.ready->ready())) {
                staged += upload.packet.nbytes();
                upload = {};
            }
        }
    }
    if (staged >= bytes) return staged;
    std::lock_guard lock(mutex_);
    const auto before = host_physical_.load(std::memory_order_relaxed);
    const auto unindexed = host_.trim(bytes - staged);
    const auto after = host_physical_.load(std::memory_order_relaxed);
    return staged + (after < before ? std::min(unindexed, before - after) : 0);
}

std::size_t GgmlExpertCache::reclaim_duplicates(std::size_t bytes) {
    std::lock_guard lock(mutex_);
    std::size_t reclaimed = 0;
    for (auto it = host_.keys().end(); it != host_.keys().begin() && reclaimed < bytes;) {
        const auto key = *--it;
        if (!device_.find(key)) continue;
        it = std::next(it);
        const auto before = host_physical_.load(std::memory_order_relaxed);
        host_.evict(key);
        const auto after = host_physical_.load(std::memory_order_relaxed);
        if (after < before) reclaimed += before - after;
    }
    return reclaimed;
}

std::size_t GgmlExpertCache::budget() const {
    std::lock_guard lock(mutex_);
    return *device_.limit();
}

bool GgmlExpertCache::has_cached_sources() const {
    std::lock_guard lock(mutex_);
    return source_bytes_ != 0;
}

bool GgmlExpertCache::host_routes_required() const {
    std::lock_guard lock(mutex_);
    return host_routes_required_;
}

std::uint64_t GgmlExpertCache::streamed_bytes() const {
    std::lock_guard lock(mutex_);
    return transferred_;
}

std::vector<std::pair<std::string, double>> GgmlExpertCache::metrics() const {
    std::lock_guard lock(mutex_);
    std::size_t coverage = device_.used();
    for (const auto& [key, entry] : host_.entries())
        if (!device_.entries().contains(key)) coverage += entry.bytes;
    const auto cold = source_bytes_ - std::min(coverage, source_bytes_);
    return {{"resident_expert_cache_bytes", double(device_.used())},
        {"expert_cache_budget_bytes", double(*device_.limit())},
        {"gguf_predictor_resident_expert_bytes", double(permanent_bytes_)},
        {"ram_expert_enabled", double(host_.used() != 0)},
        {"ram_expert_payload_bytes", double(host_.used())},
        {"ram_expert_physical_bytes", double(host_physical_.load(std::memory_order_relaxed))},
        {"ram_expert_cache_budget_bytes", double(*host_.limit())},
        {"ram_expert_pcie_read_bytes", double(transferred_)},
        {"ram_expert_cache_hits", double(host_hits_)},
        {"ram_expert_cache_misses", double(reads_)},
        {"ram_expert_cache_evictions", double(host_.evictions())},
        {"ssd_expert_enabled", double(cold != 0)},
        {"ssd_expert_payload_bytes", double(cold)},
        {"ssd_expert_bytes_read", double(fetched_)},
        {"ssd_expert_read_count", double(reads_)},
        {"expert_logical_transfer_bytes", double(transferred_)},
        {"expert_cache_hits", double(device_hits_)},
        {"expert_cache_misses", double(host_hits_) + double(reads_)},
        {"expert_cache_evictions", double(device_.evictions())}};
}

} // namespace tyloi::weights
