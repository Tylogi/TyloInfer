#include "mlx_expert_segment_cache.h"
#include "mlx_expert_segment_cache_internal.h"
#include "mlx_resource_telemetry.h"
#include "engine/backends/metal/native/pool.h"
#include "engine/backends/metal/native/resident_allocations.h"
#include "engine/backends/metal/native/allocation.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/backends/cpu/native/host_parallel.h"
#include "engine/weights/lru.h"
#include "engine/weights/source/source.h"
#include "engine/weights/source/read.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <list>
#include <mutex>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <unistd.h>

namespace tyloi::metal {
namespace {
using namespace mlx::core;
using Clock = std::chrono::steady_clock;
}


using SegmentPlan = MlxExpertSegmentCache::Plan::Impl;
std::size_t segment_metadata_reservation(std::size_t bytes) {
    const auto page = std::size_t(getpagesize());
    if (!bytes || bytes > SIZE_MAX - (page - 1)) throw std::overflow_error("expert metadata extent is too large");
    // MLX rounds requests above a page, then may reuse a cached buffer shorter
    // than min(2 * request, request + 2 * page). Reserve the aligned bound.
    const auto request = bytes > page ? (bytes + page - 1) / page * page : bytes;
    const auto doubled = request > SIZE_MAX / 2 ? SIZE_MAX : request * 2;
    const auto with_pages = request > SIZE_MAX - 2 * page ? SIZE_MAX : request + 2 * page;
    const auto bound = std::min(doubled, with_pages);
    if (bound > SIZE_MAX - (page - 1)) throw std::overflow_error("expert metadata extent is too large");
    return (bound + page - 1) / page * page;
}
std::unique_ptr<SegmentPlan> segment_make_plan(std::vector<tyloi::weights::Segments> definitions,
        const std::function<SegmentPlan::Reader(const std::string&)>& open) {
    if (definitions.empty()) throw std::invalid_argument("expert segment cache requires layers");
    auto plan = std::make_unique<SegmentPlan>();
    const auto page = std::size_t(getpagesize());
    std::size_t entries = 0;
    for (auto& definition : definitions) {
        if (definition.experts < 1 || definition.fields.empty())
            throw std::invalid_argument("expert segment layer is empty");
        std::vector<std::size_t> extents;
        extents.reserve(definition.fields.size());
        for (const auto& field : definition.fields) {
            auto found = plan->readers.find(field.tensor);
            if (found == plan->readers.end()) found = plan->readers.emplace(field.tensor, open(field.tensor)).first;
            if (found->second.extent > SIZE_MAX) throw std::overflow_error("expert tensor extent is too large");
            extents.push_back(static_cast<std::size_t>(found->second.extent));
        }
        const auto table_bytes = segment_metadata_reservation(std::size_t(definition.experts) * sizeof(std::uint64_t));
        if (definition.fields.size() > (SIZE_MAX - plan->table_reserve) / table_bytes)
            throw std::overflow_error("expert metadata extent is too large");
        plan->table_reserve += definition.fields.size() * table_bytes;
        SegmentPlan::Layer layer(std::move(definition), extents, page);
        if (layer.layout.backing() > SIZE_MAX - plan->backing)
            throw std::overflow_error("expert segment cache extent is too large");
        plan->backing += layer.layout.backing();
        plan->maximum_entry = std::max(plan->maximum_entry, layer.layout.max_entry());
        plan->total_entry_bytes += layer.layout.layer_bytes();
        plan->maximum_layer_bytes = std::max(plan->maximum_layer_bytes, layer.layout.layer_bytes());
        if (std::size_t(layer.source.experts) > SIZE_MAX - entries)
            throw std::overflow_error("expert count is too large");
        entries += std::size_t(layer.source.experts);
        for (int expert = 0; expert < layer.source.experts; ++expert) {
            const auto bytes = layer.layout.entry_size(expert);
            if (bytes > SIZE_MAX - plan->capacity) throw std::overflow_error("expert cache capacity is too large");
            plan->capacity += bytes;
        }
        plan->layers.push_back(std::move(layer));
    }
    plan->capacity = tyloi::metal::Pool::capacity_for(plan->capacity, plan->maximum_entry, entries);
    plan->metadata_reserve = plan->table_reserve;
    if (plan->capacity > SIZE_MAX - plan->metadata_reserve)
        throw std::overflow_error("expert cache capacity is too large");
    return plan;
}

namespace {
std::function<SegmentPlan::Reader(const std::string&)> source_reader(const tyloi::source::Source& source) {
    return [&source, serial = source.supports_parallel_tensor_reads() ? std::shared_ptr<std::mutex>{} : std::make_shared<std::mutex>()]
        (const std::string& name) {
            auto batch = source.tensor_batch_reader(name);
            auto reader = batch ? tyloi::source::Source::TensorReader{} : source.tensor_reader(name);
            const auto extent = tyloi::require_tensor(source, name).nbytes;
            return SegmentPlan::Reader{extent, [reader = std::move(reader), batch = std::move(batch), serial]
                (std::uint64_t offset, std::span<const std::span<std::byte>> targets) {
                std::unique_lock<std::mutex> lock;
                if (serial) lock = std::unique_lock<std::mutex>(*serial);
                if (batch) {
                    std::vector<tyloi::ReadSpan> spans;
                    spans.reserve(targets.size());
                    for (const auto target : targets) {
                        spans.push_back({offset, target.data(), target.size()});
                        offset += target.size();
                    }
                    batch(spans);
                } else {
                    for (const auto target : targets) {
                        reader(offset, target.data(), target.size());
                        offset += target.size();
                    }
                }
            }};
        };
}
} // namespace

struct MlxExpertSegmentCache::Impl {
    using Reader = Plan::Impl::Reader;
    using Layer = Plan::Impl::Layer;
    struct Entry {
        std::size_t offset = 0, bytes = 0;
        std::list<std::uint64_t>::iterator replaceable_recency;
        bool preferred = false;
    };
    std::unordered_map<std::string, Reader> readers;
    std::vector<Layer> layers;
    std::vector<std::size_t> prefill_next_rows;
    std::size_t budget, total_limit, metadata_reserve;
    std::atomic<std::size_t> metadata_bytes{0};
    std::size_t backing = 0, resident = 0, maximum_entry = 0, full_capacity = 0;
    std::unordered_set<const void*> metadata_owners;
    int workers;
    bool prefill_read_ahead;
    tyloi::HostParallelPool io;
    mutable std::mutex mutex;
    std::list<std::uint64_t> replaceable;
    bool prefill_priority_active = false;
    long double total_entry_bytes = 0, maximum_layer_bytes = 0;
    tyloi::weights::Lru<std::uint64_t, Entry> entries{std::nullopt};
    MlxSsdExpertCacheStats counters;
    std::optional<array> pending;
    MlxResourceTelemetry resources;
    std::unique_ptr<tyloi::metal::Pool> pool;
    struct Prefetch {
        std::size_t layer = 0;
        std::vector<std::pair<std::uint64_t, Entry>> entries;
        std::size_t calls = 0;
        std::future<double> read;
    };
    std::optional<Prefetch> prefetched;

    void wait_pending() {
        if (pending) { pending->eval(); pending.reset(); }
    }

    void record_metadata(const std::vector<std::shared_ptr<tyloi::TensorStorage>>& storage) {
        auto owners = metadata_owners;
        auto bytes = metadata_bytes.load(std::memory_order_relaxed);
        for (const auto& item : storage) {
            if (!item || !item->owner) continue;
            if (owners.insert(item->owner.get()).second) {
                if (item->bytes > metadata_reserve - bytes)
                    throw std::runtime_error("expert metadata exceeds its admitted reservation");
                bytes += item->bytes;
            }
        }
        metadata_owners = std::move(owners);
        metadata_bytes.store(bytes, std::memory_order_relaxed);
        if (pool) resources.set({0, 0, pool->capacity() + bytes, backing, 0});
    }

    Impl(std::unique_ptr<Plan::Impl> plan, std::size_t limit, int threads, bool read_ahead, bool reserve_capacity)
        : readers(std::move(plan->readers)), layers(std::move(plan->layers)),
          budget(limit >= plan->metadata_reserve ? limit - plan->metadata_reserve : 0),
          total_limit(limit), metadata_reserve(plan->metadata_reserve), backing(plan->backing), maximum_entry(plan->maximum_entry), full_capacity(plan->capacity),
          workers(threads), prefill_read_ahead(read_ahead),
          total_entry_bytes(plan->total_entry_bytes), maximum_layer_bytes(plan->maximum_layer_bytes) {
        if (!budget || workers < 1 || layers.empty())
            throw std::invalid_argument("expert segment cache requires a budget and layers");
        if (budget < maximum_entry) throw std::invalid_argument("expert cache budget cannot hold one expert");
        for (auto& layer : layers) {
            const std::vector<std::uint64_t> empty(layer.source.experts, 0);
            for (std::size_t i = 0; i < layer.source.fields.size(); ++i) {
                layer.tables.emplace_back(empty.begin(), Shape{layer.source.experts}, uint64);
                layer.tables.back().eval();
                tyloi::metal::AllocationScope::record(layer.tables.back());
                record_metadata(tyloi::metal::backings(layer.tables.back()));
            }
        }
        if (metadata_bytes > plan->table_reserve)
            throw std::runtime_error("expert address tables exceed their admitted metadata reservation");
        prefill_next_rows.assign(layers.size(), SIZE_MAX);
        pool = std::make_unique<tyloi::metal::Pool>(budget, maximum_entry, reserve_capacity ? plan->capacity : 0);
        resources.set({0, 0, metadata_bytes.load(std::memory_order_relaxed), backing, 0});
    }

    void end_prefill_priority() noexcept {
        prefill_priority_active = false;
        replaceable.clear();
    }
    void rebuild_replaceable() {
        replaceable.clear();
        try {
            for (auto key : entries.keys()) {
                auto& entry = entries.at(key);
                if (entry.preferred) continue;
                replaceable.push_back(key);
                entry.replaceable_recency = std::prev(replaceable.end());
            }
        } catch (const std::bad_alloc&) { end_prefill_priority(); }
    }
    void configure_prefill_priority(std::size_t index, std::size_t rows) {
        if (rows <= 6 || rows < std::size_t(layers[index].source.experts)) {
            if (prefill_priority_active) end_prefill_priority();
            return;
        }
        const auto usable = static_cast<long double>(budget) - 2 * maximum_layer_bytes;
        if (!prefill_priority_active && rows >= std::size_t(layers[index].source.experts) &&
            static_cast<long double>(budget) < total_entry_bytes && usable > 0 &&
            usable * layers[index].source.experts >= total_entry_bytes) {
            for (const auto& [key, entry] : entries.entries()) entries.at(key).preferred = false;
            prefill_priority_active = true;
            rebuild_replaceable();
        }
    }
    void inserted(std::uint64_t key) {
        if (!prefill_priority_active) return;
        try {
            replaceable.push_front(key);
            entries.at(key).replaceable_recency = replaceable.begin();
        } catch (const std::bad_alloc&) { end_prefill_priority(); }
    }
    void touch(std::uint64_t key) {
        auto& entry = entries.at(key);
        entries.get(key);
        if (prefill_priority_active && !entry.preferred)
            replaceable.splice(replaceable.begin(), replaceable, entry.replaceable_recency);
    }
    void retain_prefill_tail(std::size_t index, const std::vector<std::int32_t>& ordered) {
        if (!prefill_priority_active) return;
        for (const auto& [key, entry] : entries.entries()) if (std::size_t(key >> 32) == index) entries.at(key).preferred = false;
        rebuild_replaceable();
        if (!prefill_priority_active) return;
        const auto usable = std::max(0.L, static_cast<long double>(budget) - 2 * maximum_layer_bytes);
        const auto quota = std::min(ordered.size(), std::size_t(usable * layers[index].source.experts / total_entry_bytes));
        for (auto position = ordered.size() - quota; position < ordered.size(); ++position) {
            auto& entry = entries.at((std::uint64_t(index) << 32) | std::uint32_t(ordered[position]));
            entry.preferred = true;
            replaceable.erase(entry.replaceable_recency);
        }
    }
    void erase(std::uint64_t key) {
        const auto& entry = entries.at(key);
        for (auto& table : layers.at(std::size_t(key >> 32)).tables)
            table.data<std::uint64_t>()[std::uint32_t(key)] = 0;
        resident -= entry.bytes;
        pool->release(entry.offset, entry.bytes);
        if (prefill_priority_active && !entry.preferred) replaceable.erase(entry.replaceable_recency);
        entries.erase(key); ++counters.evictions;
    }
    bool evict_one(const std::unordered_set<std::uint64_t>& protected_keys) {
        if (prefill_priority_active) for (auto it = replaceable.rbegin(); it != replaceable.rend(); ++it) {
            if (protected_keys.contains(*it)) continue;
            erase(*it); return true;
        }
        if (const auto* key = entries.oldest(protected_keys)) {
            erase(*key); return true;
        }
        return false;
    }
    void evict(std::size_t incoming, const std::unordered_set<std::uint64_t>& protected_keys = {}) {
        while ((resident > budget || incoming > budget - resident) && evict_one(protected_keys)) {}
        if (resident > budget || incoming > budget - resident)
            throw std::runtime_error("expert cache budget cannot hold this batch");
        resources.set({0, 0, pool->capacity() + metadata_bytes.load(std::memory_order_relaxed), backing, 0});
    }

    void finish_prefetch(std::size_t layer_index = SIZE_MAX, std::span<const std::int32_t> selected = {}) {
        if (!prefetched) return;
        auto state = std::move(*prefetched);
        prefetched.reset();
        const auto start = Clock::now();
        try {
            counters.io_seconds += state.read.get();
            counters.prefill_wait_seconds += std::chrono::duration<double>(Clock::now() - start).count();
            const auto count = state.entries.size();
            for (auto& [key, entry] : state.entries) {
                if (state.layer == layer_index && !std::binary_search(selected.begin(), selected.end(), std::int32_t(key))) {
                    pool->release(entry.offset, entry.bytes); resident -= entry.bytes;
                    entry.offset = SIZE_MAX;
                    continue;
                }
                entries.put(key, entry, entry.bytes);
                inserted(key);
                entry.offset = SIZE_MAX;
            }
            counters.read_calls += state.calls;
            counters.loads += count;
            counters.prefill_expert_reads += count;
            for (const auto& item : state.entries) {
                const auto bytes = layers[state.layer].layout.payload_size(std::int32_t(item.first));
                counters.bytes_read += bytes; counters.prefill_bytes_read += bytes;
            }
            ++counters.prefill_read_batches;
        } catch (...) {
            for (const auto& [key, entry] : state.entries) if (entry.offset != SIZE_MAX) {
                pool->release(entry.offset, entry.bytes); resident -= entry.bytes;
            }
            ++counters.failures;
            throw;
        }
    }

    void prefetch_next(std::size_t current, std::size_t rows, std::size_t in_flight = 0) {
        if (!prefill_read_ahead || prefetched || current + 1 >= layers.size()) return;
        const auto index = current + 1;
        auto& layer = layers[index];
        if (std::min(rows, prefill_next_rows[current]) < std::size_t(layer.source.experts) ||
            rows < std::size_t(layers[current].source.experts)) return;
        std::size_t reserve = 0;
        for (int expert = 0; expert < layer.source.experts; ++expert) {
            const auto key = (std::uint64_t(index) << 32) | std::uint32_t(expert);
            if (!entries.find(key)) reserve += layer.layout.entry_size(expert);
        }
        if (!reserve || resident > budget || in_flight > budget - resident || reserve > budget - resident - in_flight) return;
        Prefetch state; state.layer = index;
        struct ReadJob {
            std::string tensor;
            std::uint64_t offset;
            std::vector<std::span<std::byte>> targets;
        };
        std::vector<ReadJob> jobs;
        try {
            state.entries.reserve(layer.source.experts);
            for (int expert = 0; expert < layer.source.experts; ++expert) {
                const auto key = (std::uint64_t(index) << 32) | std::uint32_t(expert);
                if (entries.find(key)) continue;
                const auto bytes = layer.layout.entry_size(expert);
                const auto offset = pool->allocate(bytes);
                if (!offset) {
                    for (const auto& item : state.entries) pool->release(item.second.offset, item.second.bytes);
                    resources.set({0, 0, pool->capacity() + metadata_bytes.load(std::memory_order_relaxed), backing, 0});
                    return;
                }
                state.entries.push_back({key, Entry{*offset, bytes, {}}});
            }
            for (std::size_t field = 0; field < layer.source.fields.size(); ++field) {
                const auto& source = layer.source.fields[field];
                const auto maximum = std::max<std::size_t>(1, (32u << 20) / source.expert_bytes);
                for (std::size_t first = 0; first < state.entries.size();) {
                    auto last = first + 1;
                    while (last < state.entries.size() && last - first < maximum &&
                        state.entries[last].first == state.entries[last - 1].first + 1 &&
                        source.begin(std::int32_t(state.entries[last].first)) == source.begin(std::int32_t(state.entries[last - 1].first)) +
                            source.size(std::int32_t(state.entries[last - 1].first))) ++last;
                    ReadJob job{source.tensor, source.begin(std::int32_t(state.entries[first].first)), {}};
                    for (auto item = first; item < last; ++item) {
                        const int e = std::int32_t(state.entries[item].first);
                        job.targets.emplace_back(pool->data(state.entries[item].second.offset + layer.layout.field_offset(field, e)), source.size(e));
                    }
                    jobs.push_back(std::move(job)); first = last;
                }
            }
            state.calls = jobs.size();
            state.read = std::async(std::launch::async, [this, jobs = std::move(jobs)] {
                const auto start = Clock::now();
                io.run(0, std::int64_t(jobs.size()), 1, workers, [&](auto first, auto last) {
                    for (auto item = first; item < last; ++item) {
                        const auto& job = jobs[std::size_t(item)];
                        std::as_const(readers).at(job.tensor).read(job.offset, job.targets);
                    }
                });
                return std::chrono::duration<double>(Clock::now() - start).count();
            });
            for (const auto& item : state.entries) resident += item.second.bytes;
            prefetched.emplace(std::move(state));
            ++counters.overlapped_read_submissions;
            resources.set({0, 0, pool->capacity() + metadata_bytes.load(std::memory_order_relaxed), backing, 0});
        } catch (...) {
            if (state.read.valid()) state.read.wait();
            for (const auto& item : state.entries) pool->release(item.second.offset, item.second.bytes);
            resources.set({0, 0, pool->capacity() + metadata_bytes.load(std::memory_order_relaxed), backing, 0});
            throw;
        }
    }

    void prepare(std::size_t index, const std::vector<std::int32_t>& selected,
        std::size_t first_fields, const std::function<void()>& stage) {
        auto& layer = layers.at(index);
        std::vector<std::pair<std::uint64_t, Entry>> pending;
        pending.reserve(selected.size());
        bool staged = false;
        struct Cleanup {
            Impl& cache;
            std::vector<std::pair<std::uint64_t, Entry>>& pending;
            bool& staged;
            ~Cleanup() {
                if (staged && std::any_of(pending.begin(), pending.end(),
                    [](const auto& item) { return item.second.offset != SIZE_MAX; })) synchronize();
                for (const auto& item : pending) if (item.second.offset != SIZE_MAX)
                    cache.pool->release(item.second.offset, item.second.bytes);
                cache.resources.set({0, 0, cache.pool->capacity() + cache.metadata_bytes.load(std::memory_order_relaxed), cache.backing, 0});
            }
        } cleanup{*this, pending, staged};
        std::unordered_set<std::uint64_t> protected_keys;
        std::size_t reserve = 0;
        for (const auto expert : selected) {
            const auto key = (std::uint64_t(index) << 32) | std::uint32_t(expert);
            protected_keys.insert(key);
            if (!entries.find(key)) reserve += layer.layout.entry_size(expert);
        }
        evict(reserve, protected_keys);
        for (const auto expert : selected) {
            const auto key = (std::uint64_t(index) << 32) | std::uint32_t(expert);
            ++counters.requests;
            if (entries.find(key)) {
                ++counters.hits; touch(key); continue;
            }
            const auto bytes = layer.layout.entry_size(expert);
            auto offset = pool->allocate(bytes);
            while (!offset && evict_one(protected_keys)) offset = pool->allocate(bytes);
            if (!offset) throw std::runtime_error("expert batch cannot fit the memory pool");
            pending.push_back({key, Entry{*offset, bytes, {}}});
        }
        struct ReadJob {
            std::size_t field;
            std::uint64_t offset;
            std::vector<std::span<std::byte>> targets;
        };
        std::vector<ReadJob> jobs;
        std::size_t first_jobs = 0;
        for (std::size_t field = 0; field < layer.source.fields.size(); ++field) {
            const auto& source = layer.source.fields[field];
            const auto maximum = std::max<std::size_t>(1, (32u << 20) / source.expert_bytes);
            for (std::size_t first = 0; first < pending.size();) {
                auto last = first + 1;
                while (last < pending.size() && last - first < maximum && pending[last].first == pending[last - 1].first + 1 &&
                    source.begin(std::int32_t(pending[last].first)) == source.begin(std::int32_t(pending[last - 1].first)) +
                        source.size(std::int32_t(pending[last - 1].first))) ++last;
                ReadJob job{field, source.begin(std::int32_t(pending[first].first)), {}};
                job.targets.reserve(last - first);
                for (auto item = first; item < last; ++item) {
                    const int e = std::int32_t(pending[item].first);
                    job.targets.emplace_back(pool->data(pending[item].second.offset + layer.layout.field_offset(field, e)), source.size(e));
                }
                jobs.push_back(std::move(job)); first = last;
            }
            if (field + 1 == first_fields) first_jobs = jobs.size();
        }
        for (auto& table : layer.tables)
            std::fill_n(table.data<std::uint64_t>(), layer.source.experts, std::uint64_t{0});
        const auto publish = [&](std::size_t first, std::size_t last) {
            const auto assign = [&](std::int32_t expert, std::size_t offset) {
                for (auto field = first; field < last; ++field)
                    layer.tables[field].data<std::uint64_t>()[expert] = pool->address(offset + layer.layout.field_offset(field, expert));
            };
            for (const auto expert : selected) {
                const auto found = entries.find((std::uint64_t(index) << 32) | std::uint32_t(expert));
                if (found) assign(expert, found->offset);
            }
            for (const auto& [key, entry] : pending) assign(std::int32_t(key), entry.offset);
        };
        const auto start = Clock::now();
        const auto read = [&](std::int64_t begin, std::int64_t end) {
            for (auto job = begin; job < end; ++job) {
                const auto& read = jobs[std::size_t(job)];
                const auto& field = layer.source.fields[read.field];
                std::as_const(readers).at(field.tensor).read(read.offset, read.targets);
            }
        };
        try {
            if (first_jobs && first_jobs < jobs.size()) {
                io.run_staged(std::int64_t(jobs.size()), std::int64_t(first_jobs), workers, read, [&] {
                    publish(0, first_fields);
                    staged = true;
                    stage();
                });
                ++counters.overlapped_read_batches;
            } else io.run(0, std::int64_t(jobs.size()), 1, workers, read);
        } catch (...) {
            if (staged) synchronize();
            throw;
        }
        counters.io_seconds += std::chrono::duration<double>(Clock::now() - start).count(); counters.read_calls += jobs.size();
        publish(staged ? first_fields : 0, layer.tables.size());
        for (auto& [key, entry] : pending) {
            entries.put(key, entry, entry.bytes);
            inserted(key);
            resident += entry.bytes; entry.offset = SIZE_MAX;
            counters.bytes_read += layer.layout.payload_size(std::int32_t(key)); ++counters.misses; ++counters.loads;
        }
    }
};

MlxExpertSegmentCache::Plan::Plan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MlxExpertSegmentCache::Plan::Plan(Plan&&) noexcept = default;
MlxExpertSegmentCache::Plan& MlxExpertSegmentCache::Plan::operator=(Plan&&) noexcept = default;
MlxExpertSegmentCache::Plan::~Plan() = default;
std::size_t MlxExpertSegmentCache::Plan::minimum() const noexcept { return impl_->maximum_entry + impl_->metadata_reserve; }
std::size_t MlxExpertSegmentCache::Plan::capacity() const noexcept { return impl_->capacity + impl_->metadata_reserve; }
void MlxExpertSegmentCache::Plan::reserve_metadata(std::size_t bytes) {
    if (bytes > SIZE_MAX - impl_->metadata_reserve || impl_->capacity > SIZE_MAX - impl_->metadata_reserve - bytes)
        throw std::overflow_error("expert metadata extent is too large");
    impl_->metadata_reserve += bytes;
}

MlxExpertSegmentCache::Plan MlxExpertSegmentCache::plan(
        const tyloi::source::Source& source, std::vector<tyloi::weights::Segments> layers) {
    return Plan(segment_make_plan(std::move(layers), source_reader(source)));
}
MlxExpertSegmentCache::MlxExpertSegmentCache(Plan plan, std::size_t budget, int workers, bool read_ahead)
    : impl_(std::make_shared<Impl>(std::move(plan.impl_), budget, workers, read_ahead, plan.reserve_capacity_)) {}
MlxExpertSegmentCache::MlxExpertSegmentCache(const tyloi::source::Source& source,
    std::vector<tyloi::weights::Segments> layers, std::size_t budget, int workers, bool read_ahead)
    : MlxExpertSegmentCache(plan(source, std::move(layers)), budget, workers, read_ahead) {}
void MlxExpertSegmentCache::record_metadata(const std::vector<std::shared_ptr<tyloi::TensorStorage>>& storage) {
    impl_->record_metadata(storage);
}
const std::vector<array>& MlxExpertSegmentCache::address_tables(std::size_t layer) const { return impl_->layers.at(layer).tables; }
const std::shared_ptr<tyloi::metal::ResidentAllocations>& MlxExpertSegmentCache::residency() const noexcept { return impl_->pool->residency(); }
void MlxExpertSegmentCache::preload() {
    std::lock_guard lock(impl_->mutex);
    impl_->wait_pending();
    impl_->finish_prefetch();
    if (impl_->budget < impl_->full_capacity)
        throw std::invalid_argument("expert cache budget cannot preload every expert");
    impl_->end_prefill_priority();
    for (std::size_t index = 0; index < impl_->layers.size(); ++index) {
        std::vector<std::int32_t> experts(impl_->layers[index].source.experts);
        std::iota(experts.begin(), experts.end(), 0);
        impl_->prepare(index, experts, 0, [] {});
    }
}
void MlxExpertSegmentCache::set_prefill_next_rows(std::size_t layer, std::size_t rows) {
    std::lock_guard lock(impl_->mutex);
    impl_->prefill_next_rows.at(layer) = rows;
}

array MlxExpertSegmentCache::apply(std::size_t index, const array& ids, std::size_t first_fields,
    const std::function<array()>& first_stage, const std::function<array(const array&, bool)>& second_stage,
    const array* parallel) {
    std::lock_guard lock(impl_->mutex);
    const auto& layer = impl_->layers.at(index);
    if (!first_fields || first_fields >= layer.source.fields.size())
        throw std::invalid_argument("invalid expert pipeline field boundary");
    if (ids.ndim() != 2 || ids.shape(0) < 1 || ids.shape(1) < 1 || (ids.dtype() != int32 && ids.dtype() != uint32))
        throw std::invalid_argument("expert cache requires rank-two integer routes");
    const auto begin = Clock::now();
    auto routes = contiguous(astype(ids, int32)); routes.eval();
    impl_->wait_pending();
    impl_->counters.route_sync_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
    std::vector<std::int32_t> selected(routes.data<std::int32_t>(), routes.data<std::int32_t>() + routes.size());
    std::sort(selected.begin(), selected.end()); selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    if (selected.front() < 0 || selected.back() >= layer.source.experts)
        throw std::invalid_argument("expert routes are outside the model");
    if (parallel && !parallel->is_available() && std::any_of(selected.begin(), selected.end(), [&](auto expert) {
            return !impl_->entries.find((std::uint64_t(index) << 32) | std::uint32_t(expert));
        })) async_eval({*parallel});
    impl_->finish_prefetch(index, selected);
    impl_->configure_prefill_priority(index, std::size_t(ids.shape(0)));
    std::vector<std::size_t> last_use(layer.source.experts, 0);
    for (std::size_t position = 0; position < routes.size(); ++position)
        last_use[routes.data<std::int32_t>()[position]] = position;
    const auto batch_size = impl_->pool->batch_capacity(layer.layout.max_entry());
    if (!batch_size) throw std::runtime_error("expert cache budget cannot hold one expert");
    const auto byte_capacity = batch_size * layer.layout.max_entry();
    std::size_t selected_bytes = 0;
    for (const auto expert : selected) selected_bytes += layer.layout.entry_size(expert);
    const bool partial = selected_bytes > byte_capacity;
    std::optional<array> output;
    Dtype dtype = float16;
    for (std::size_t start = 0; start < selected.size();) {
        std::size_t end = start, bytes = 0;
        while (end < selected.size() && layer.layout.entry_size(selected[end]) <= byte_capacity - bytes)
            bytes += layer.layout.entry_size(selected[end++]);
        const std::vector<std::int32_t> batch(selected.begin() + start, selected.begin() + end);
        std::size_t in_flight = 0;
        if (!partial && impl_->prefill_read_ahead && std::size_t(ids.shape(0)) >= std::size_t(layer.source.experts))
            for (const auto expert : batch)
                if (!impl_->entries.find((std::uint64_t(index) << 32) | std::uint32_t(expert)))
                    in_flight += layer.layout.entry_size(expert);
        const auto io_before = impl_->counters.io_seconds;
        const auto prepare_start = Clock::now();
        std::optional<array> intermediate;
        impl_->prepare(index, batch, first_fields, [&] {
            intermediate = first_stage();
            std::vector<array> submitted{*intermediate};
            if (parallel && !parallel->is_available()) submitted.push_back(*parallel);
            async_eval(submitted);
            if (!partial) impl_->prefetch_next(index, std::size_t(ids.shape(0)), in_flight);
        });
        impl_->counters.prepare_seconds += std::max(0.0, std::chrono::duration<double>(Clock::now() - prepare_start).count()
            - (impl_->counters.io_seconds - io_before));
        const auto graph_start = Clock::now();
        if (!intermediate) intermediate = first_stage();
        if (!partial) impl_->prefetch_next(index, std::size_t(ids.shape(0)));
        auto value = second_stage(*intermediate, partial);
        if (!output) dtype = value.dtype();
        if (partial) value = astype(value, float32);
        output = output ? *output + value : std::move(value);
        if (end < selected.size()) output->eval();
        else { impl_->pending = *output; async_eval({*output}); }
        auto recency = batch;
        std::sort(recency.begin(), recency.end(), [&](auto a, auto b) { return last_use[a] < last_use[b]; });
        for (const auto expert : recency) {
            const auto key = (std::uint64_t(index) << 32) | std::uint32_t(expert);
            impl_->touch(key);
        }
        impl_->retain_prefill_tail(index, recency);
        impl_->counters.routed_graph_seconds += std::chrono::duration<double>(Clock::now() - graph_start).count();
        start = end;
    }
    return output->dtype() == dtype ? *output : astype(*output, dtype);
}

std::size_t MlxExpertSegmentCache::limit_bytes() const noexcept { return impl_->total_limit; }
void MlxExpertSegmentCache::drain() {
    std::lock_guard lock(impl_->mutex);
    impl_->wait_pending();
    impl_->finish_prefetch();
}
std::size_t MlxExpertSegmentCache::backing_bytes() const noexcept { return impl_->backing; }
std::size_t MlxExpertSegmentCache::resident_bytes() const noexcept { return impl_->pool->capacity() + impl_->metadata_bytes; }
std::size_t MlxExpertSegmentCache::set_limit(std::size_t bytes) {
    if (!bytes) throw std::invalid_argument("expert cache budget must be nonzero");
    std::lock_guard lock(impl_->mutex);
    impl_->wait_pending();
    impl_->finish_prefetch();
    if (bytes < impl_->metadata_reserve) throw std::invalid_argument("expert cache budget cannot hold metadata");
    const auto pool_budget = bytes - impl_->metadata_reserve;
    const auto before = impl_->pool->capacity(), target = pool_budget / std::size_t(getpagesize()) * std::size_t(getpagesize());
    if (!target) throw std::invalid_argument("expert cache budget is smaller than a page");
    if (target < impl_->maximum_entry) throw std::invalid_argument("expert cache budget cannot hold one expert");
    impl_->end_prefill_priority();
    for (auto it = impl_->entries.entries().begin(); it != impl_->entries.entries().end();) {
        auto current = it++;
        const auto& entry = current->second.value;
        if (entry.offset > target || entry.bytes > target - entry.offset) impl_->erase(current->first);
    }
    impl_->pool->set_limit(pool_budget); impl_->budget = pool_budget; impl_->total_limit = bytes; impl_->evict(0);
    return before - impl_->pool->capacity();
}
MlxSsdExpertCacheStats MlxExpertSegmentCache::stats() const {
    std::lock_guard lock(impl_->mutex);
    auto result = impl_->counters;
    result.resident_experts = impl_->entries.entries().size(); result.resident_bytes = impl_->pool->capacity() + impl_->metadata_bytes;
    std::size_t payload = 0;
    for (const auto& [key, entry] : impl_->entries.entries()) payload += impl_->layers.at(std::size_t(key >> 32)).layout.payload_size(std::int32_t(key));
    result.cached_payload_bytes = payload; result.cache_metadata_bytes = impl_->metadata_bytes;
    result.pinned_cache_bytes = impl_->pool->pinned_bytes();
    return result;
}
void MlxExpertSegmentCache::clear() {
    std::lock_guard lock(impl_->mutex);
    impl_->wait_pending();
    impl_->finish_prefetch();
    impl_->end_prefill_priority();
    impl_->entries.clear(); impl_->resident = 0;
    for (auto& layer : impl_->layers) for (auto& table : layer.tables)
        std::fill_n(table.data<std::uint64_t>(), layer.source.experts, std::uint64_t{0});
    impl_->pool->clear(); impl_->resources.set({0, 0, impl_->metadata_bytes.load(std::memory_order_relaxed), impl_->backing, 0});
}
} // namespace tyloi::metal
