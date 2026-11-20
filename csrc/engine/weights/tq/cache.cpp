#include "engine/weights/tq/cache.h"

#include "engine/weights/tq/image.h"
#include "engine/weights/moe/residency/retention.h"
#include "engine/weights/moe/residency/routes.h"
#include "engine/backends/cuda/ops/projection/tq/tensor.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/allocation.h"
#include "engine/backends/cuda/native/memory.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "engine/backends/cuda/native/stream.h"
#include "engine/weights/encoding/tq/table.h"
#include "engine/backends/cuda/executor/tq/resources.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/executor/graph/registry.h"
#include "env.h"
#include "engine/memory/host.h"
#include "components/error.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <exception>
#include <unordered_set>


namespace {
constexpr std::size_t host_headroom = std::size_t(2) << 30;
constexpr std::size_t device_alignment = 64 << 10;
std::size_t device_bytes(std::size_t bytes) {
    if(bytes>SIZE_MAX-(device_alignment-1))throw std::overflow_error("TQ device image size overflows");
    // Reuse size classes instead of fragmenting the CUDA pool with variable-bit images.
    return (bytes+device_alignment-1)&~(device_alignment-1);
}
std::size_t available_host_bytes(std::size_t process_limit) {
    try {
        const auto memory=tyloi::sample_host_memory();
        return memory.headroom(process_limit);
    } catch(const std::runtime_error&) {return 0;}
}
}

struct TqCache::Upload {
    tyloi::cuda::Event event{cudaEventDisableTiming};
    cudaStream_t producer;
    tyloi::cuda::GraphRetirement retirement;
    explicit Upload(cudaStream_t stream) : producer(stream) {
        retirement.owner=this;
        retirement.dispose=[](void* value) noexcept {delete static_cast<Upload*>(value);};
    }
    static std::shared_ptr<Upload> create(cudaStream_t stream) {
        return {new Upload(stream),[](Upload* value) noexcept {
            tyloi::cuda::retire_graph_resource(&value->retirement);
        }};
    }
};

struct TqCache::ViewTable {
    int device;
    cudaStream_t stream;
    ::tyloi::DenseTensor views;
    std::vector<std::uintptr_t> images;
    std::vector<std::uint8_t> bytes;
};

struct TqCache::Routes {
    struct Bank {
        std::array<std::uint64_t, 3> sources;
        int device;
        std::size_t first, count;
        bool decoded = false;
    };
    tyloi::engine::RouteHistory history{
        {}, tyloi::engine::RouteHistory::HeatMode::count, tyloi::engine::expert_usage_retention};
    std::vector<Bank> banks;
    std::vector<tyloi::weights::ExpertRetention> experts;
    std::optional<std::size_t> budget;
    std::uint64_t windows = 0;
    bool prefill = false;
    std::unordered_set<int> pending;
};

struct TqCache::HostAccounting {
    std::atomic<std::size_t> physical{0};
};

struct TqCache::HostImage {
    tyloi::TqLayout layout;
    std::vector<std::uint8_t> bytes;
    std::size_t source_bytes;
    bool pin_attempted=false,registered=false;
    int pin_device=0;
    tyloi::cuda::GraphRetirement retirement;
    std::shared_ptr<HostAccounting> accounting;
    static void destroy(void* owner) noexcept {
        auto* image=static_cast<HostImage*>(owner);
        const auto capacity=image->bytes.capacity();
        auto accounting=image->accounting;
        delete image;
        tyloi::cuda::tensor_host_bytes.fetch_sub(capacity,std::memory_order_relaxed);
        accounting->physical.fetch_sub(capacity,std::memory_order_relaxed);
    }
    explicit HostImage(const std::vector<std::uint8_t>& wire,
                       std::shared_ptr<HostAccounting> owner)
        :layout(wire.data(),wire.size()),bytes(tq_image(wire,layout)),source_bytes(wire.size()),
         accounting(std::move(owner)) {
        retirement.owner=this;
        retirement.dispose=destroy;
        tyloi::cuda::charge_tensor_host_bytes(bytes.capacity());
        accounting->physical.fetch_add(bytes.capacity(),std::memory_order_relaxed);
    }
    HostImage(std::vector<std::uint8_t>&& image, std::size_t wire_bytes,
              std::shared_ptr<HostAccounting> owner)
        :layout(image.data(),wire_bytes),bytes(std::move(image)),source_bytes(wire_bytes),
         accounting(std::move(owner)) {
        retirement.owner=this;
        retirement.dispose=destroy;
        tyloi::cuda::charge_tensor_host_bytes(bytes.capacity());
        accounting->physical.fetch_add(bytes.capacity(),std::memory_order_relaxed);
    }
    HostImage(const HostImage&)=delete;
    HostImage& operator=(const HostImage&)=delete;
    static void release(HostImage* image) noexcept {
        if(image->registered)tyloi::cuda::retire_graph_resource(&image->retirement);
        else destroy(image);
    }
    static std::shared_ptr<HostImage> create(const std::vector<std::uint8_t>& wire,
                                             std::shared_ptr<HostAccounting> accounting) {
        return {new HostImage(wire,std::move(accounting)),release};
    }
    static std::shared_ptr<HostImage> download(const DeviceEntry& entry,
                                               std::shared_ptr<HostAccounting> accounting) {
        const auto& source=entry.weight.image();
        const MfqCudaGuard guard(source.get_device());
        const auto stream=mfq_get_current_cuda_stream(source.get_device());
        TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),entry.uploaded->event.get(),0));
        std::vector<std::uint8_t> bytes(entry.weight.layout().bytes);
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(bytes.data(),source.data_ptr(),bytes.size(),
            cudaMemcpyDeviceToHost,stream.stream()));
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        return {new HostImage(std::move(bytes),entry.source_bytes,std::move(accounting)),release};
    }
    // Pin only a reused host image. One-use SSD reads retain bounded
    // staging; repeated RAM transfers avoid another full CPU copy.
    void try_pin() {
        if(pin_attempted)return;
        pin_attempted=true;
        TYLOINFER_CUDA_CHECK(cudaGetDevice(&pin_device));
        const auto result=cudaHostRegister(bytes.data(),bytes.size(),cudaHostRegisterPortable);
        if(result==cudaSuccess){registered=true;return;}
        if(tyloi::cuda::pinned_allocation_rejected(result) || result==cudaErrorNotSupported ||
           result==cudaErrorOperatingSystem || result==cudaErrorNotPermitted ||
           result==cudaErrorHostMemoryAlreadyRegistered) {
            (void)cudaGetLastError();return;
        }
        TYLOINFER_CUDA_CHECK(result);
    }
    ~HostImage()noexcept {
        if(registered)try {
            const tyloi::cuda::DeviceGuard guard(pin_device);
            if(cudaHostUnregister(bytes.data())!=cudaSuccess)(void)cudaGetLastError();
        }catch(...) {}
    }
};

struct TqCache::Stage {
    tyloi::cuda::HostBuffer host;
    tyloi::cuda::Event uploaded;
    std::size_t cursor=0;
    bool pending=false;
    cudaStream_t stream=nullptr;
    explicit Stage(std::size_t bytes):host(bytes),uploaded(cudaEventDisableTiming) {
        tyloi::cuda::charge_tensor_host_bytes(bytes);
    }
    ~Stage(){
        if(pending)try{uploaded.synchronize();}catch(...) {}
        tyloi::cuda::tensor_host_bytes.fetch_sub(host.size());
    }
    void reuse(){if(pending){uploaded.synchronize();pending=false;}cursor=0;}
};

TqCache::HostTransfer::~HostTransfer() noexcept { try { if(ready)ready->event.synchronize(); } catch (...) {} }

std::size_t TqCache::reclaim_host_locked(std::size_t bytes) {
    std::size_t released = 0;
    while (released < bytes && !host_.empty()) {
        const auto key = *host_.oldest();
        const auto before = host_accounting_->physical.load(std::memory_order_relaxed);
        host_.erase(key);
        const auto after = host_accounting_->physical.load(std::memory_order_relaxed);
        if (before > after) {
            const auto delta = before - after;
            released = delta > SIZE_MAX - released ? SIZE_MAX : released + delta;
        }
    }
    return released;
}

std::size_t TqCache::reserve_host_headroom_locked(std::size_t bytes) {
    if (!bytes || host_.empty()) return 0;
    std::size_t released = 0;
    while (!host_.empty()) {
        const auto available = available_host_bytes(process_host_limit_);
        if (available >= bytes) break;
        const auto reclaimed = reclaim_host_locked(bytes - available);
        released = reclaimed > SIZE_MAX - released ? SIZE_MAX : released + reclaimed;
        // Freed expert images may remain in the host allocator's arenas.
        tyloi::trim_host_cache();
    }
    return released;
}

std::size_t TqCache::reserve_host_locked(std::size_t bytes) {
    if(!bytes)return 0;
    if(bytes>SIZE_MAX-host_headroom)throw std::overflow_error("TQ host reservation overflows");
    return reserve_host_headroom_locked(bytes + host_headroom);
}

bool TqCache::admit_host_locked(std::size_t bytes, std::size_t reserve) const {
    const auto limit = host_.limit();
    // LRU usage is only cached payload. Admission must include evicted images
    // whose leases or registration retirement still own their backing.
    const auto physical = host_accounting_->physical.load(std::memory_order_relaxed);
    if (limit && (physical > *limit || bytes > *limit - physical)) return false;
    if (bytes > SIZE_MAX - reserve) return false;
    return available_host_bytes(process_host_limit_) >= bytes + reserve;
}

std::size_t TqCache::trim_device(std::size_t bytes, bool protect, int only_device) {
    const auto before=device_.used();
    while(!device_.empty() && before-device_.used()<bytes) {
        const DeviceKey* oldest=nullptr;
        if(only_device<0) {
            oldest=device_.oldest(retained_);
            if(!oldest && !protect)oldest=device_.oldest();
        }else for(auto it=device_.keys().rbegin();it!=device_.keys().rend();++it) {
            if(it->device==only_device && (!protect || !retained_.contains(*it))){oldest=&*it;break;}
        }
        if(!oldest)break;
        const auto key = *oldest;
        const auto& entry=device_.at(key);
        const auto needed=entry.weight.layout().bytes;
        // Preserve an evicted GPU image in RAM before falling back to its file.
        if(!host_.find(key.image) && needed<=*host_.limit()) {
            reserve_host_locked(needed);
            host_.reserve(needed);
            if(admit_host_locked(needed,host_headroom)) {
                auto image=HostImage::download(entry,host_accounting_);
                if(admit_host_locked(0,host_headroom))
                    host_.put(key.image,std::move(image),needed);
            }
        }
        device_.evict(key);
    }
    return before-device_.used();
}

bool TqCache::reserve_device(std::size_t bytes, int device) {
    const auto limit = device_.limit();
    if (!limit) return true;
    const auto keep = bytes < *limit ? *limit - bytes : 0;
    if (device_.used() > keep) trim_device(device_.used() - keep, true);
    if (bytes <= *limit - device_.used()) return true;
    // A cold miss can use a leased image without entering the cache. Retire
    // retained backing too if that transient upload would exceed headroom.
    const auto available = tyloi::cuda::default_context(device)->available_memory();
    if (available < bytes) trim_device(bytes - available);
    return false;
}

void TqCache::report()const {
    if(profile_)std::fprintf(stderr,"tq_cache calls=%llu gpu_hits=%llu host_hits=%llu source_bytes=%llu h2d_bytes=%llu fetch_ms=%.3f upload_submit_ms=%.3f gpu_bytes=%zu host_bytes=%zu host_limit=%zu direct_h2d_bytes=%llu pin_attempts=%llu pin_failures=%llu\n",
        static_cast<unsigned long long>(calls_),static_cast<unsigned long long>(gpu_hits_),
        static_cast<unsigned long long>(host_hits_),static_cast<unsigned long long>(source_bytes_),
        static_cast<unsigned long long>(transfer_bytes_),fetch_ms_,upload_submit_ms_,device_.used(),host_.used(),*host_.limit(),
        static_cast<unsigned long long>(direct_transfer_bytes_),static_cast<unsigned long long>(pin_attempts_),
        static_cast<unsigned long long>(pin_failures_));
}

void TqCache::upload(const ::tyloi::DenseTensor& destination,const std::vector<std::uint8_t>& bytes) {
    const auto started=std::chrono::steady_clock::now();
    auto& staging=stages_[destination.get_device()];
    auto* stage=staging.stages[staging.index].get();
    if(!stage || bytes.size()>stage->host.size()-stage->cursor) {
        staging.index^=1;
        auto& selected=staging.stages[staging.index];
        if(selected)selected->reuse();
        // Bound normal staging, while allowing any valid projection larger
        // than the normal chunk. This is a host DMA arena, not a GPU limit.
        const auto capacity=std::max<std::size_t>(std::size_t(32)<<20,bytes.size());
        if(!selected || selected->host.size()<capacity)selected=std::make_unique<Stage>(capacity);
        stage=selected.get();
    }
    auto* source=static_cast<std::uint8_t*>(stage->host.data())+stage->cursor;
    std::memcpy(source,bytes.data(),bytes.size());
    const auto stream=mfq_current_cuda_stream();
    // A staging arena can serve different request streams. Chain its
    // copies so the final reuse fence covers every outstanding DMA.
    if(stage->pending && stage->stream!=stream)
        TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream,stage->uploaded.get(),0));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(destination.data_ptr(),source,bytes.size(),cudaMemcpyHostToDevice,stream));
    try {
        stage->uploaded.record(stream);
    } catch (...) {
        // The copy was already submitted. Keep its staging bytes intact until it retires.
        (void)cudaStreamSynchronize(stream);
        throw;
    }
    stage->pending=true;stage->stream=stream;
    stage->cursor+=bytes.size();transfer_bytes_+=bytes.size();
    upload_submit_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}

TqCache::TqCache(std::size_t budget,std::size_t host_budget,std::size_t process_limit,bool retain_all,
              std::shared_ptr<TqExec> tq)
    :tq_(std::move(tq)),
     host_accounting_(std::make_shared<HostAccounting>()),
     host_(host_budget?host_budget:available_host_bytes(process_limit)/5*4),
     device_(retain_all?std::nullopt:std::optional<std::size_t>(budget)),
     device_budget_(device_.limit()),
     process_host_limit_(process_limit),
     ordinary_host_limit_(*host_.limit()),
     profile_(tyloi::cuda::env_lookup("TYLOINFER_TQ_PROFILE_CACHE")!=nullptr) {
    if(!tq_)throw std::invalid_argument("TQ expert cache requires its execution pool");
}

TqCache::~TqCache(){
    host_transfers_.clear();
    report();
}

std::uint64_t TqCache::add(std::shared_ptr<const tyloi::TqTable> table, std::int64_t rows, bool resident) {
    if (!table || rows < 1 || table->width() < 1 || table->rows() % rows ||
        table->rows() / rows < 1 || table->rows() / rows > UINT32_MAX)
        throw std::invalid_argument("invalid TQ expert source geometry");
    std::lock_guard<std::mutex> guard(mutex_);
    if (sources_.size() == UINT32_MAX)
        throw std::overflow_error("too many TQ projection sources");
    const auto bytes = table->storage_bytes();
    const auto experts = table->rows() / rows;
    if (resident) {
        const tyloi::cuda::AllocationScope weight_memory;
        const tyloi::cuda::ResidentAllocationScope permanent;
        const auto device = mfq_current_cuda_device();
        const auto stream = mfq_current_cuda_stream();
        const auto options = ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA,device)).dtype(::tyloi::kUInt8);
        Source bank{table,rows,{}, {}};
        std::vector<std::size_t> offsets{0};
        offsets.reserve(experts+1);
        for (std::uint64_t expert=0;expert<experts;++expert) {
            const auto size=table->expert_image_bytes(expert,rows);
            if(size>std::size_t(INT64_MAX)-15 || offsets.back()>std::size_t(INT64_MAX)-((size+15)&~std::size_t(15)))
                throw std::overflow_error("TQ resident expert bank is too large");
            offsets.push_back(offsets.back()+((size+15)&~std::size_t(15)));
        }
        auto packed=::tyloi::cuda::empty({static_cast<std::int64_t>(offsets.back())},options);
        bank.resident.reserve(experts);
        std::vector<TrellisProjectionView> views;
        views.reserve(experts);
        for (std::uint64_t expert=0;expert<experts;++expert) {
            const auto wire=table->slice_rows(expert*rows,(expert+1)*rows);
            const tyloi::TqLayout layout(wire.data(),wire.size());
            const auto image=tq_image(wire,layout);
            if(image.size()>offsets[expert+1]-offsets[expert])
                throw std::logic_error("TQ resident image exceeds its interval");
            auto storage=packed.narrow(0,offsets[expert],image.size());
            upload(storage,image);
            bank.resident.push_back(tq_view(std::move(storage),layout,table->width()));
            views.push_back(tyloi::cuda::tq::view(bank.resident.back()));
        }
        auto descriptors=::tyloi::cuda::from_blob(views.data(),
            {static_cast<std::int64_t>(views.size()*sizeof(TrellisProjectionView))},
            ::tyloi::TensorOptions().dtype(::tyloi::kUInt8));
        bank.resident_views=descriptors.to(packed.device(),false,false);
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto resident_bytes=packed.nbytes()+bank.resident_views.nbytes();
        if(resident_bytes>SIZE_MAX-predictor_resident_bytes_)
            throw std::overflow_error("TQ resident expert byte count overflows");
        sources_.push_back(std::move(bank));
        predictor_resident_bytes_+=resident_bytes;
        return sources_.size();
    }
    const auto images = table->routed_image_storage_bound(experts);
    if (bytes > SIZE_MAX - payload_bytes_ || images > SIZE_MAX - resident_bound_ ||
        experts > (SIZE_MAX - resident_bound_ - images) / (device_alignment - 1))
        throw std::overflow_error("TQ source payload exceeds size limits");
    sources_.push_back({std::move(table), rows, {}, {}});
    payload_bytes_ += bytes;
    resident_bound_ += images + experts * (device_alignment - 1);
    return sources_.size();
}

::tyloi::DenseTensor TqCache::resident_views(std::uint64_t source, int device) {
    std::lock_guard lock(mutex_);
    const auto& views=sources_.at(source-1).resident_views;
    return views.defined() && views.get_device()==device ? views : ::tyloi::DenseTensor{};
}

std::size_t TqCache::budget() {
    std::lock_guard<std::mutex> guard(mutex_);
    return device_budget_.value_or(std::max(resident_bound_,device_.used()));
}

bool TqCache::has_cached_sources() {
    std::lock_guard lock(mutex_);
    return resident_bound_ != 0;
}

std::size_t TqCache::peak_expert_bytes(const std::array<std::uint64_t,3>& banks,
        std::span<const std::int64_t> experts) {
    std::lock_guard lock(mutex_);
    for (const auto bank : banks)
        if (!bank || bank > sources_.size()) throw std::out_of_range("unknown TQ source");
    std::size_t peak = 0;
    for (const auto expert : experts) {
        if (expert < 0) throw std::out_of_range("TQ routed expert ID is out of bounds");
        std::size_t sum = 0;
        for (const auto bank : banks) {
            const auto& source = sources_[bank - 1];
            const auto bytes = device_bytes(source.table->expert_image_bytes(expert, source.rows));
            if (bytes > SIZE_MAX - sum) throw std::overflow_error("TQ expert image size overflow");
            sum += bytes;
        }
        peak = std::max(peak, sum);
    }
    return peak;
}

std::size_t TqCache::lendable_bytes(int device) {
    const auto* option = tyloi::cuda::env_lookup("TYLOINFER_PREFILL_BORROW");
    if (option && option[0] == '0') return 0;
    std::lock_guard guard(mutex_);
    std::size_t largest = 0;
    // shortcut: price one contiguous image, revisit if a prefill caller supplies its full allocation sequence.
    for (const auto& key : device_.keys())
        if (key.device == device) largest = std::max(largest, device_.at(key).lendable_bytes());
    return largest;
}

std::shared_ptr<tyloi::TensorStorage> TqCache::take_workspace(int device, std::size_t bytes) {
    const auto* option = tyloi::cuda::env_lookup("TYLOINFER_PREFILL_BORROW");
    if (!bytes || (option && option[0] == '0')) return {};
    std::lock_guard guard(mutex_);
    const DeviceKey* candidate = nullptr;
    for (auto it = device_.keys().rbegin(); it != device_.keys().rend(); ++it) {
        if (it->device != device || device_.at(*it).lendable_bytes() < bytes) continue;
        if (!candidate) candidate = &*it;
        if (!retained_.contains(*it)) {
            candidate = &*it;
            break;
        }
    }
    if (!candidate) return {};
    auto storage = device_.at(*candidate).weight.storage();
    device_.evict(*candidate);
    // The loan still occupies those bytes until the next decode boundary.
    device_.limit(device_.used());
    assert(storage->bytes >= bytes && reinterpret_cast<std::uintptr_t>(storage->base) % 256 == 0);
    return storage;
}

void TqCache::resize(std::size_t bytes) {
    std::lock_guard<std::mutex> guard(mutex_);
    device_budget_ = bytes;
    if(device_.used()>bytes)trim_device(device_.used()-bytes);
    device_.limit(bytes);
}

std::uint64_t TqCache::streamed_bytes() {
    std::lock_guard<std::mutex> guard(mutex_);
    return transfer_bytes_;
}

void TqCache::adapt_routes(bool prefill) {
    if (!routes_ || !learning_ || !device_.limit()) return;
    auto& r = *routes_;
    const auto budget = *device_.limit();
    const bool rebuild = !r.budget || *r.budget != budget;
    if (prefill) {
        if (!r.prefill && !rebuild) return;
    } else if (!rebuild && ++r.windows < tyloi::engine::expert_usage_interval) {
        return;
    }
    r.history.flush();
    auto items = r.experts;
    for (std::size_t bank = 0; bank < r.banks.size(); ++bank) {
        const auto& group = r.banks[bank];
        for (std::size_t expert = 0; expert < group.count; ++expert)
            items[group.first + expert].heat = r.history.heat(bank, static_cast<int>(expert));
    }
    tyloi::weights::select_expert_retention(items, budget, rebuild);
    decltype(retained_) next;
    for (const auto& group : r.banks) {
        for (std::size_t expert = 0; expert < group.count; ++expert)
            if (items[group.first + expert].retained)
                for (const auto source : group.sources)
                    next.insert({(source << 32) | expert, group.device});
    }
    retained_.swap(next);
    r.experts.swap(items);
    r.pending.clear();
    for (const auto& group : r.banks) {
        for (std::size_t expert = 0; expert < group.count; ++expert) {
            if (!r.experts[group.first + expert].retained) continue;
            for (const auto source : group.sources) {
                if (!device_.find({(source << 32) | expert, group.device})) {
                    r.pending.insert(group.device);
                    break;
                }
            }
        }
    }
    if (r.windows) r.history.decay(tyloi::engine::expert_usage_decay(r.windows));
    r.windows = 0;
    r.prefill = false;
    r.budget = budget;
}

void TqCache::observe_routes(const std::array<std::uint64_t, 3>& banks,
        std::span<const std::int64_t> ids, int rows, int row_offset, int row_total,
        int device, bool independent) {
    std::lock_guard lock(mutex_);
    if (!device_.limit() || (!learning_ && (!routes_ || !routes_->history.pending()))) return;
    if (rows < 1 || ids.empty() || ids.size() % std::size_t(rows) ||
        row_offset < 0 || row_total <= 0 || row_offset > row_total || rows > row_total - row_offset)
        throw std::invalid_argument("invalid TQ route geometry");
    const auto& first = sources_.at(banks[0] - 1);
    if (std::any_of(banks.begin(), banks.end(), [&](auto id) {
            return !sources_.at(id-1).resident.empty();
        })) return;
    const auto experts = first.table->rows() / first.rows;
    if (experts > INT_MAX) return;
    std::vector<std::int32_t> published;
    published.reserve(ids.size());
    for (const auto id : ids) {
        if (id < 0 || std::uint64_t(id) >= experts) throw std::out_of_range("TQ route expert ID");
        published.push_back(static_cast<std::int32_t>(id));
    }
    if (!routes_) routes_ = std::make_unique<Routes>();
    auto& r = *routes_;
    auto found = std::find_if(r.banks.begin(), r.banks.end(), [&](const auto& group) {
        return group.device == device && group.sources == banks;
    });
    const auto bundle = static_cast<std::size_t>(found - r.banks.begin());
    if (found == r.banks.end()) {
        // Every protected entry belongs to one complete three-bank group.
        for (const auto& group : r.banks)
            if (group.device == device)
                for (const auto source : banks)
                    if (std::find(group.sources.begin(), group.sources.end(), source) != group.sources.end())
                        return;
        std::vector<const Source*> projections;
        projections.reserve(banks.size());
        for (auto it = banks.begin(); it != banks.end(); ++it) {
            const auto& source = sources_.at(*it - 1);
            if (source.table->rows() / source.rows != experts)
                throw std::invalid_argument("TQ expert group geometry differs");
            if (std::find(banks.begin(), it, *it) == it) projections.push_back(&source);
        }
        std::vector<tyloi::weights::ExpertRetention> items;
        items.reserve(experts);
        for (std::uint64_t expert = 0; expert < experts; ++expert) {
            std::size_t bytes = 0;
            for (const auto* source : projections) {
                const auto size = device_bytes(source->table->expert_image_bytes(expert, source->rows));
                if (size > SIZE_MAX - bytes) throw std::overflow_error("TQ expert group size overflows");
                bytes += size;
            }
            items.push_back({(banks[0] << 32) | expert, device, bytes, 0, false});
        }
        r.banks.reserve(r.banks.size() + 1);
        r.experts.reserve(r.experts.size() + items.size());
        r.history.add_bundle({static_cast<int>(experts), true});
        r.banks.push_back({banks, device, r.experts.size(), items.size()});
        r.experts.insert(r.experts.end(), items.begin(), items.end());
    }
    if (!r.history.pending() && rows > 8 && !independent) {
        if (!r.banks[bundle].decoded) {
            r.history.stage_prefill(bundle, published, rows);
            r.prefill = true;
        }
        return;
    }
    const bool journal = r.history.pending();
    r.history.observe(bundle, published, rows, independent && !journal,
                      row_offset, row_total, learning_);
    if (!journal) {
        r.banks[bundle].decoded = true;
        // Count one decode window when the first group has all its rows.
        if (bundle == 0 && rows == row_total - row_offset) adapt_routes(false);
    }
}

void TqCache::begin_routes(int rows) {
    std::lock_guard lock(mutex_);
    if (!routes_) routes_ = std::make_unique<Routes>();
    routes_->history.begin(rows);
}

void TqCache::commit_routes(std::span<const std::uint8_t> keep, bool single_sequence) {
    std::lock_guard lock(mutex_);
    if (!routes_) throw std::logic_error("TQ routes have no active window");
    auto& r = *routes_;
    for (const auto& route : r.history.selected(keep))
        if (learning_) r.banks[route.bundle].decoded = true;
    if (r.history.commit(keep, single_sequence, learning_).heat_routes) adapt_routes(false);
}

void TqCache::discard_routes() noexcept {
    std::lock_guard lock(mutex_);
    if (routes_) routes_->history.discard();
}

bool TqCache::set_learning(bool enabled) {
    std::lock_guard lock(mutex_);
    return std::exchange(learning_, enabled);
}

void TqCache::prepare_decode(int device, std::size_t workspace_reserve) {
    const MfqCudaGuard device_guard(device);
    {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto context = tyloi::cuda::default_context(device);
        context->reclaim_idle_memory();
        const auto available = context->available_memory();
        const auto headroom = available > workspace_reserve ? available - workspace_reserve : 0;
        // Temporary workspace eviction must not permanently lower the configured budget.
        device_.limit(std::min(device_budget_.value_or(SIZE_MAX), device_.used() + headroom));
        if(!routes_ || !routes_->history.pending())adapt_routes(true);
    }
    admit_routes(device,workspace_reserve);
}

void TqCache::admit_routes(int device, std::size_t workspace_reserve) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!routes_ || !learning_) return;
    if (routes_->prefill && !routes_->history.pending())adapt_routes(true);
    if (!routes_->pending.contains(device)) return;
    const tyloi::cuda::RoutedWeightAllocationScope resident;
    const MfqCudaGuard device_guard(device);
    auto& r=*routes_;
    struct Candidate {std::size_t bank,expert;float heat;};
    std::vector<Candidate> candidates;
    for (std::size_t bank=0;bank<r.banks.size();++bank) {
        const auto& group=r.banks[bank];if(group.device!=device)continue;
        for (std::size_t expert=0;expert<group.count;++expert) {
            const auto& item=r.experts[group.first+expert];
            if(item.retained)candidates.push_back({bank,expert,item.heat});
        }
    }
    std::stable_sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.heat>b.heat;});
    const auto context=tyloi::cuda::default_context(device);
    for (const auto& candidate:candidates) {
        const auto& group=r.banks[candidate.bank];
        std::vector<DeviceKey> missing;
        std::size_t bytes=0;
        for (const auto source:group.sources) {
            const DeviceKey key{(source<<32)|candidate.expert,device};
            if(device_.find(key) || std::find(missing.begin(),missing.end(),key)!=missing.end())continue;
            const auto& bank=sources_.at(source-1);
            const auto size=device_bytes(bank.table->expert_image_bytes(candidate.expert,bank.rows));
            if(size>SIZE_MAX-bytes)throw std::overflow_error("TQ route admission size overflows");
            bytes+=size;missing.push_back(key);
        }
        if(missing.empty())continue;
        const auto limit=device_.limit().value_or(SIZE_MAX);
        if(bytes>limit) {++route_admit_skips_;continue;}
        if(device_.used()>limit-bytes)trim_device(device_.used()-(limit-bytes),true);
        if(device_.used()>limit-bytes || bytes>SIZE_MAX-workspace_reserve) {
            ++route_admit_skips_;continue;
        }
        const auto required=bytes+workspace_reserve;
        auto available=context->available_memory();
        if(available<required) {
            context->reclaim_idle_memory();
            available=context->available_memory();
        }
        while(available<required) {
            const auto before=device_.used();
            trim_device(required-available,true,device);
            if(device_.used()==before)break;
            context->reclaim_idle_memory();
            available=context->available_memory();
        }
        if(available<required) {++route_admit_skips_;continue;}
        try {
            const auto expert=std::int64_t(candidate.expert);
            for(const auto& key:missing)get_locked(key.image>>32,&expert,1,device,false);
            for(const auto& key:missing)if(!device_.find(key))
                throw std::runtime_error("TQ route admission did not retain a complete expert");
        } catch(const MfqResourceExhausted&) {
            for(const auto& key:missing)device_.erase(key);
            ++route_admit_failures_;continue;
        } catch(...) {
            for(const auto& key:missing)device_.erase(key);
            ++route_admit_failures_;throw;
        }
        ++route_admit_groups_;route_admit_images_+=missing.size();route_admit_bytes_+=bytes;
    }
    r.pending.erase(device);
}

void TqCache::reclaim(int device, std::size_t available_bytes) {
    const MfqCudaGuard device_guard(device);
    std::lock_guard<std::mutex> guard(mutex_);
    const auto context = tyloi::cuda::default_context(device);
    context->reclaim_idle_memory();
    while (!device_.empty()) {
        const auto available = context->available_memory();
        if (available >= available_bytes) break;
        trim_device(available_bytes - available);
        device_.limit(device_.used());
        context->reclaim_idle_memory();
    }
}

std::size_t TqCache::reclaim(std::size_t bytes) {
    std::lock_guard<std::mutex> guard(mutex_);
    return reclaim_host_locked(bytes);
}

std::size_t TqCache::reclaim_duplicates(std::size_t bytes) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!bytes) return 0;
    std::unordered_set<std::uint64_t> resident;
    for (const auto& [key, _] : device_.entries()) resident.insert(key.image);
    std::vector<std::uint64_t> duplicates;
    for (auto key = host_.keys().rbegin(); key != host_.keys().rend(); ++key)
        if (resident.contains(*key)) duplicates.push_back(*key);
    const auto before = host_accounting_->physical.load(std::memory_order_relaxed);
    for (const auto key : duplicates) {
        host_.erase(key);
        const auto after = host_accounting_->physical.load(std::memory_order_relaxed);
        if (before > after && before - after >= bytes) break;
    }
    const auto after = host_accounting_->physical.load(std::memory_order_relaxed);
    return before > after ? before - after : 0;
}

std::size_t TqCache::share_host(std::size_t free_bytes) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!free_bytes) {
        host_.limit(ordinary_host_limit_);
        return 0;
    }
    const auto before = host_accounting_->physical.load(std::memory_order_relaxed);
    reserve_host_locked(free_bytes);
    const auto physical = host_accounting_->physical.load(std::memory_order_relaxed);
    host_.limit(tyloi::shared_host_cache_limit(ordinary_host_limit_, physical,
        available_host_bytes(process_host_limit_), free_bytes));
    const auto after = host_accounting_->physical.load(std::memory_order_relaxed);
    return before > after ? before - after : 0;
}

void TqCache::preload(int device, std::size_t workspace_reserve) {
    const tyloi::cuda::RoutedWeightAllocationScope resident;
    const MfqCudaGuard device_guard(device);
    std::lock_guard<std::mutex> guard(mutex_);
    const auto context = tyloi::cuda::default_context(device);
    const auto stream = mfq_current_cuda_stream();
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
    context->reclaim_idle_memory();
    // Request KV and pinned prefill buffers need RAM after startup residency.
    const auto host_reserve = std::max(workspace_reserve,host_headroom);
    // Fill GPU first, then RAM, distributing initial experts across all layers.
    for (const bool gpu : {true, false}) {
        bool exhausted = false;
        for (std::uint64_t expert = 0; !exhausted; ++expert) {
            bool found = false;
            for (std::size_t s = 0; s < sources_.size(); ++s) {
                const auto& source = sources_[s];
                if(!source.resident.empty())continue;
                if (expert >= source.table->rows() / source.rows) continue;
                found = true;
                const auto key = (std::uint64_t(s + 1) << 32) | expert;
                const DeviceKey device_key{key, device};
                if (device_.find(device_key)) {
                    host_.erase(key);
                    continue;
                }
                if (!gpu && host_.find(key)) continue;
                const auto image_bytes = source.table->expert_image_bytes(expert, source.rows);
                const auto bound = gpu ? device_bytes(image_bytes) : image_bytes;
                if (gpu) {
                    const auto available = context->available_memory();
                    const auto remaining = device_.limit().value_or(SIZE_MAX) - device_.used();
                    if (bound > remaining || available <= workspace_reserve ||
                        bound > available - workspace_reserve) {
                        exhausted = true;
                        break;
                    }
                } else {
                    if (!admit_host_locked(bound,host_reserve)) {
                        exhausted = true;
                        break;
                    }
                }
                std::shared_ptr<HostImage> image;
                if (const auto* cached = host_.find(key)) image = *cached;
                else {
                    const auto wire = source.table->slice_rows(expert * source.rows, (expert + 1) * source.rows);
                    source_bytes_ += wire.size();
                    ++source_reads_;
                    image = HostImage::create(wire,host_accounting_);
                }
                if (gpu) {
                    const auto capacity=device_bytes(image->bytes.size());
                    auto storage = ::tyloi::cuda::empty({static_cast<std::int64_t>(capacity)},
                        ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA, device)).dtype(::tyloi::kUInt8));
                    auto weight = tq_view(storage.narrow(0,0,image->bytes.size()), image->layout, source.table->width());
                    auto ready = Upload::create(stream);
                    upload(storage, image->bytes);
                    ready->event.record(stream);
                    device_.put(device_key, DeviceEntry{std::move(weight), std::move(ready), image->source_bytes}, capacity, true);
                    host_.erase(key);
                } else {
                    if (!admit_host_locked(0,host_reserve)) {
                        exhausted = true;
                        break;
                    }
                    host_.put(key, image, image->bytes.size(), true);
                }
            }
            if (!found) break;
        }
        // Later cache misses must preserve the same workspace headroom.
        if (gpu) device_.limit(device_.used());
    }
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::fprintf(stderr, "tq_cache_preload gpu_bytes=%zu host_bytes=%zu workspace_reserve_bytes=%zu\n",
        device_.used(), host_.used(), workspace_reserve);
}

std::size_t TqCache::reserve(std::size_t bytes) {
    std::lock_guard<std::mutex> guard(mutex_);return reserve_host_locked(bytes);
}

std::vector<std::pair<std::string,double>> TqCache::metrics() {
    std::lock_guard<std::mutex> guard(mutex_);
    std::size_t covered=0,staging=0;
    std::unordered_set<std::uint64_t> covered_images;
    covered_images.reserve(device_.entries().size()+host_.entries().size());
    for(const auto& [key,entry]:device_.entries())
        if(covered_images.insert(key.image).second)covered+=entry.value.source_bytes;
    for(const auto& [key,entry]:host_.entries())
        if(covered_images.insert(key).second)covered+=entry.value->source_bytes;
    for(const auto& [_,arenas]:stages_)for(const auto& stage:arenas.stages)
        if(stage)staging+=stage->host.size();
    const auto cold=payload_bytes_>covered?payload_bytes_-covered:0;
    const auto misses=calls_-gpu_hits_;
    const auto physical=host_accounting_->physical.load(std::memory_order_relaxed);
    return {{"resident_expert_cache_bytes",double(device_.used())},
        {"tq_predictor_resident_expert_bytes",double(predictor_resident_bytes_)},
        {"expert_cache_budget_bytes",double(device_.limit().value_or(std::max(resident_bound_,device_.used())))},
        {"ram_expert_enabled",host_.used()?1.0:0.0},
        {"ram_expert_payload_bytes",double(host_.used())},
        {"ram_expert_physical_bytes",double(physical)},
        {"ram_expert_cache_budget_bytes",double(*host_.limit())},
        {"ram_expert_pcie_read_bytes",double(transfer_bytes_)},
        {"ram_expert_direct_pcie_read_bytes",double(direct_transfer_bytes_)},
        {"ram_expert_cache_hits",double(host_hits_)},
        {"ram_expert_cache_misses",double(misses-host_hits_)},
        {"ram_expert_cache_evictions",double(host_.evictions())},
        {"ssd_expert_enabled",cold?1.0:0.0},
        {"ssd_expert_payload_bytes",double(cold)},
        {"ssd_expert_bytes_read",double(source_bytes_)},
        {"ssd_expert_read_count",double(source_reads_)},
        {"expert_source_read_batches",double(source_batches_)},
        {"expert_logical_transfer_bytes",double(transfer_bytes_)},
        {"expert_host_fetch_ms",fetch_ms_},{"expert_host_copy_ms",upload_submit_ms_},
        {"expert_decode_transfer_bytes",double(staging)},
        {"expert_cache_hits",double(gpu_hits_)},
        {"expert_cache_misses",double(misses)},
        {"expert_cache_evictions",double(device_.evictions())},
        {"tq_route_admit_groups",double(route_admit_groups_)},
        {"tq_route_admit_images",double(route_admit_images_)},
        {"tq_route_admit_bytes",double(route_admit_bytes_)},
        {"tq_route_admit_skips",double(route_admit_skips_)},
        {"tq_route_admit_failures",double(route_admit_failures_)},
        {"host_tensor_bytes",double(tyloi::cuda::tensor_host_bytes.load())}};
}

::tyloi::DenseTensor TqCache::views(std::uint64_t source,
        const std::int64_t* ids,const std::vector<::tyloi::encoding::TqTensor>& weights,int device) {
    const MfqCudaGuard device_guard(device);
    std::lock_guard<std::mutex> guard(mutex_);
    if (!source || source > sources_.size()) throw std::out_of_range("unknown TQ source");
    const auto& bank = sources_[source - 1];
    if (bank.resident_views.defined() && bank.resident_views.get_device()==device)
        return bank.resident_views;
    const auto experts = static_cast<std::int64_t>(bank.table->rows() / bank.rows);
    for (std::size_t i = 0; i < weights.size(); ++i)
        if (!ids || ids[i] < 0 || ids[i] >= experts)
            throw std::out_of_range("TQ view expert ID is out of bounds");
    const auto stream=mfq_current_cuda_stream();
    auto& tables=view_tables_[source];
    auto found=std::find_if(tables.begin(),tables.end(),[&](const ViewTable& table) {
        return table.device==device && table.stream==stream;
    });
    if(found==tables.end()) {
        const tyloi::cuda::ResidentAllocationScope resident;
        auto views=::tyloi::cuda::empty({experts*static_cast<std::int64_t>(sizeof(TrellisProjectionView))},
            ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA,device)).dtype(::tyloi::kUInt8));
        tables.push_back({device,stream,std::move(views),std::vector<std::uintptr_t>(experts),
            std::vector<std::uint8_t>(experts*sizeof(TrellisProjectionView))});
        found=std::prev(tables.end());
    }
    // Each stream has independent metadata, so a concurrent upload cannot
    // overwrite a view another request is reading. Tables own no expert
    // images; the caller's selected weights retain their normal lifetime.
    bool changed=false;
    for(std::size_t i=0;i<weights.size();++i) {
        const auto image=reinterpret_cast<std::uintptr_t>(weights[i].image().data_ptr());
        if(found->images[ids[i]]==image)continue;
        const auto view=tyloi::cuda::tq::view(weights[i]);
        std::memcpy(found->bytes.data()+ids[i]*sizeof(TrellisProjectionView),&view,sizeof(view));
        found->images[ids[i]]=image;changed=true;
    }
    // One small DMA for changed metadata, no per-expert transfers and no
    // host synchronization. Resident hits submit no metadata work at all.
    if(changed)try {upload(found->views,found->bytes);}
    catch(...) {std::fill(found->images.begin(),found->images.end(),0);throw;}
    return found->views;
}

std::vector<::tyloi::encoding::TqTensor> TqCache::get(std::uint64_t source,const std::int64_t* ids,
        std::size_t count,int device,UploadArena* arena) {
    const tyloi::cuda::RoutedWeightAllocationScope resident;
    const MfqCudaGuard device_guard(device);
    std::lock_guard<std::mutex> guard(mutex_);
    return get_locked(source,ids,count,device,true,arena);
}

std::vector<::tyloi::encoding::TqTensor> TqCache::get_locked(std::uint64_t source,const std::int64_t* ids,
        std::size_t count,int device,bool demand,UploadArena* arena) {
    if (!source || source > sources_.size()) throw std::out_of_range("unknown TQ source");
    const auto& bank = sources_[source - 1];
    const auto& table = *bank.table;
    const auto out = bank.rows;
    for (std::size_t i = 0; i < count; ++i)
        if (!ids || ids[i] < 0 || std::uint64_t(ids[i]) >= table.rows() / out)
            throw std::out_of_range("TQ routed expert ID is out of bounds");
    if (bank.resident_views.defined() && bank.resident_views.get_device()==device) {
        std::vector<::tyloi::encoding::TqTensor> result;
        result.reserve(count);
        for(std::size_t i=0;i<count;++i)result.push_back(bank.resident[ids[i]]);
        return result;
    }
    for(auto it=host_transfers_.begin();it!=host_transfers_.end();) {
        if(it->ready->event.ready()) {
            it->ready.reset();
            it=host_transfers_.erase(it);
        }else ++it;
    }
    std::vector<std::int64_t> missing;
    std::unordered_map<std::uint32_t, std::size_t> source_images;
    if (count > 1) {
        for (std::size_t i = 0; i < count; ++i) {
            const auto expert = static_cast<std::uint32_t>(ids[i]);
            const auto key = (source << 32) | expert;
            if (!device_.find({key, device}) && !host_.find(key) &&
                source_images.try_emplace(expert, missing.size()).second)
                missing.push_back(expert);
        }
    }
    std::vector<std::vector<std::uint8_t>> prefetched;
    if (missing.size() > 1) {
        const auto started = std::chrono::steady_clock::now();
        auto headroom = available_host_bytes(process_host_limit_);
        std::size_t largest = 0;
        bool fits = true;
        for (const auto expert : missing) {
            const auto bytes = table.expert_image_bytes(expert, out);
            if (bytes > headroom) { fits = false; break; }
            headroom -= bytes;
            largest = std::max(largest, bytes);
        }
        if (fits && largest <= headroom) {
            prefetched = table.slice_experts(missing, out);
            ++source_batches_;
        }
        fetch_ms_ += std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now() - started).count();
    }
    std::vector<std::shared_ptr<HostImage>> prepared_images(prefetched.size());
    const auto resources=tq_->get(device);
    std::vector<::tyloi::encoding::TqTensor> result;result.reserve(count);
    const auto stream=mfq_get_current_cuda_stream(device);
    std::shared_ptr<Upload> uploaded;
    auto transfer=host_transfers_.end();
    std::unordered_set<cudaEvent_t> waited;
    std::vector<DeviceKey> inserted;
    inserted.reserve(count);
    std::exception_ptr failure;
    try {
    for(std::size_t index=0;index<count;++index) {
    const auto expert=static_cast<std::uint32_t>(ids[index]);
    const auto key=(source<<32)|expert;
    const DeviceKey device_key{key,device};
    if(demand)++calls_;
    if(const auto* resident=device_.get(device_key)) {
        if(resident->uploaded->producer!=stream.stream()) {
            const auto event=resident->uploaded->event.get();
            if(waited.insert(event).second)TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),event,0));
            resources->retain(stream);
            mfq_cuda_record_stream(resident->weight,stream);
        }
        if(demand)++gpu_hits_;
        result.push_back(resident->weight);
        if(demand && (calls_&4095)==0)report();
        continue;
    }
    const auto fetch_started=std::chrono::steady_clock::now();
    std::shared_ptr<HostImage> image;
    if(const auto* cached=host_.get(key)) {
        image=*cached;
        if(demand)++host_hits_;
        if(resources->host_register() && !image->pin_attempted) {
            if(auto registration=tyloi::cuda::try_host_registration();registration.owns_lock()) {
                image->try_pin();++pin_attempts_;
                if(!image->registered)++pin_failures_;
            }
        }
    } else {
        const auto prepared = source_images.find(expert);
        const auto batch_index = prepared == source_images.end() ? prefetched.size() : prepared->second;
        if (batch_index < prefetched.size() && prepared_images[batch_index]) {
            image = prepared_images[batch_index];
            if(demand)++host_hits_;
        } else {
            // Earlier admissions can evict a hit that was outside the original batch.
            auto wire = batch_index < prefetched.size() ? std::move(prefetched[batch_index]) :
                table.slice_rows(std::uint64_t(expert)*out,std::uint64_t(expert+1)*out);
            reserve_host_locked(wire.size());
            source_bytes_ += wire.size();
            ++source_reads_;
            image = HostImage::create(wire, host_accounting_);
            host_.reserve(image->bytes.size());
            reserve_host_headroom_locked(host_headroom);
            if (admit_host_locked(0, host_headroom)) host_.put(key, image, image->bytes.size());
            if (batch_index < prefetched.size()) prepared_images[batch_index] = image;
        }
    }
    fetch_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fetch_started).count();
    const auto needed=image->bytes.size();
    const auto capacity=device_bytes(needed);
    const bool cacheable=!arena && reserve_device(capacity, device);
    // The local image owns the upload source; its RAM slot can hold a GPU victim.
    if(cacheable)host_.erase(key);
    ::tyloi::DenseTensor storage;
    if (arena) {
        if (arena->used > arena->image.nbytes() || capacity > arena->image.nbytes() - arena->used)
            throw std::logic_error("TQ expert images exceed the upload arena");
        storage = arena->image.narrow(0, arena->used, capacity);
        arena->used += capacity;
    } else {
        storage = ::tyloi::cuda::empty({static_cast<std::int64_t>(capacity)},
            ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA,device)).dtype(::tyloi::kUInt8));
    }
    auto weight=tq_view(storage.narrow(0,0,needed),image->layout,table.width());
    if(image->registered) {
        if(!uploaded)uploaded=Upload::create(stream.stream());
        if(transfer==host_transfers_.end()) {
            host_transfers_.emplace_back();transfer=std::prev(host_transfers_.end());transfer->ready=uploaded;
        }
        // A host-cache eviction must not unpin a source still consumed
        // by asynchronous DMA. One event retains the whole call's images.
        transfer->images.push_back(image);
        const auto started=std::chrono::steady_clock::now();
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(storage.data_ptr(),image->bytes.data(),image->bytes.size(),
            cudaMemcpyHostToDevice,stream.stream()));
        transfer_bytes_+=image->bytes.size();direct_transfer_bytes_+=image->bytes.size();
        upload_submit_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }else upload(storage,image->bytes);
    if(cacheable) {
        if(!uploaded)uploaded=Upload::create(stream.stream());
        device_.put(device_key,DeviceEntry{weight,uploaded,image->source_bytes},capacity);
        inserted.push_back(device_key);
    }
    result.push_back(std::move(weight));
    if(demand && (calls_&4095)==0)report();
    }
    }catch(...) {
        failure=std::current_exception();
    }
    if(uploaded)try {
        uploaded->event.record(stream.stream());
    } catch (...) {
        if(!failure)failure=std::current_exception();
        (void)cudaStreamSynchronize(stream.stream());
        if(transfer!=host_transfers_.end())host_transfers_.erase(transfer);
    }
    if(failure) {
        // A partially submitted batch cannot become a later cache hit.
        for(const auto& key:inserted)device_.erase(key);
        std::rethrow_exception(failure);
    }
    return result;
}
