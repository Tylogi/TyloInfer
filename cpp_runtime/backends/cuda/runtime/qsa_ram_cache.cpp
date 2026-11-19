#include "qsa_ram_cache.h"
#include "mfq_cuda_process_memory.h"
#include "pinned_allocation_retry.h"
#include "prefill_projection_plan.h"
#include "kernels/qsa_ram.h"
#include "mfq_cuda_context.h"
#include "mfq_cuda_resident_allocations.h"
#include "decode_window.h"
#include "moe_pipeline.h"
#include "tyloinfer/host_memory.h"
#include "tyloinfer/host_parallel.h"
#include "tyloinfer/runtime.h"
#include "tyloinfer/qsa_tq_cpu.h"
#include <algorithm>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <thread>
#include <tuple>
#include <unordered_map>

namespace mfq::cuda {
namespace tb=mfq_tensor_backend;
namespace tq=tyloinfer::qsa;
thread_local QsaRamDecodeBudgetScope* QsaRamDecodeBudgetScope::active_=nullptr;
QsaRamDecodeBudgetScope::QsaRamDecodeBudgetScope(int device,bool defer_shared_cache)
    :previous_(active_),device_(device),defer_shared_cache_(defer_shared_cache) {active_=this;}
QsaRamDecodeBudgetScope::~QsaRamDecodeBudgetScope(){active_=previous_;}
bool QsaRamDecodeBudgetScope::claim_optional_reclaim(int device) {
    for(auto* scope=active_;scope;scope=scope->previous_) {
        if(scope->device_!=device)continue;
        return !std::exchange(scope->claimed_,true);
    }
    return true;
}
bool QsaRamDecodeBudgetScope::defer_shared_cache(int device) {
    for(auto* scope=active_;scope;scope=scope->previous_)
        if(scope->device_==device)return scope->defer_shared_cache_;
    return false;
}

bool qsa_host_metadata_enabled(bool prefer_host) {
    const auto* flag=std::getenv("TYLOINFER_QSA_HOST_METADATA");
    if(flag?flag[0]!='1':!prefer_host)return false;
    int device=0,mapped=0;
    TYLOINFER_CUDA_CHECK(cudaGetDevice(&device));
    TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,device));
    return mapped!=0;
}

tb::Tensor qsa_descriptor_table(std::int64_t rows,std::int64_t stride,const tb::TensorOptions& options,bool mapped) {
    const auto bytes=options.dtype(tb::kUInt8);
    if(!mapped)return tb::zeros({rows,stride},bytes);
    auto host=tb::zeros({rows,stride},bytes.device(tb::kCPU).pinned_memory(true));
    void* address=nullptr;
    TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&address,host.data_ptr(),0));
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const std::array<std::int64_t,2> shape{rows,stride};
    auto view=make_contiguous_view(address,shape,bytes.scalar_type(),bytes.target_device());
    auto storage=std::make_shared<TensorStorage>();
    storage->base=address;storage->bytes=view.nbytes();storage->device=bytes.target_device();
    storage->owner=std::make_shared<tb::Tensor>(std::move(host));
    return tb::Tensor(std::move(storage),view);
#else
    return tb::from_blob(address,{rows,stride},[host=std::move(host)](void*){},bytes);
#endif
}
namespace {
using RamReclaimKey=std::pair<int,cudaStream_t>;
struct RamReclaimRegistration;
struct RamReclaimRegistry {
    std::mutex mutex;
    std::map<RamReclaimKey,std::vector<std::weak_ptr<RamReclaimRegistration>>> entries;
};
std::shared_ptr<RamReclaimRegistry> ram_reclaim_registry() {
    static auto registry=std::make_shared<RamReclaimRegistry>();return registry;
}
struct RamReclaimRegistration {
    RamReclaimKey key;
    QsaRamReclaimer reclaim;
    unsigned kind=0;
    std::weak_ptr<RamReclaimRegistry> registry;
    ~RamReclaimRegistration() {
        if(auto owner=registry.lock()) {
            std::lock_guard<std::mutex> lock(owner->mutex);
            const auto it=owner->entries.find(key);
            if(it==owner->entries.end())return;
            auto& values=it->second;
            values.erase(std::remove_if(values.begin(),values.end(),
                [](const auto& value){return value.expired();}),values.end());
            if(values.empty())owner->entries.erase(it);
        }
    }
};
void check(int code){TYLOINFER_CUDA_CHECK(cudaError_t(code));}
std::size_t setting(const char* name,std::size_t fallback=0) {
    const auto* value=std::getenv(name);if(!value || !*value)return fallback;
    std::size_t used=0;const auto result=std::stoull(value,&used);
    if(value[0]=='-' || used!=std::strlen(value))throw std::invalid_argument(std::string("Invalid ")+name);
    return std::size_t(result);
}
std::size_t tq4_bytes(int rows){return ((2*std::size_t(rows)+(rows+1)/2+(3*rows+7)/8+3)&~std::size_t(3))+std::size_t(rows)*128;}
tq::MicroblockImage copy(const tq::RamKvImage& image) {
    tq::MicroblockImage result;result.key.assign(image.key,image.key+image.key_bytes);
    result.value.assign(image.value,image.value+image.value_bytes);
    result.key_bits=image.key_bits;result.value_bits=image.value_bits;
    result.key_state_bits=image.key_state_bits;result.value_state_bits=image.value_state_bits;
    return result;
}
struct RamAccount {
    std::atomic<std::size_t> physical{0},logical{0},target{0},restore_target{0};
    std::atomic<std::size_t> restore_shared{0},shared_prefix_ram{0};
    std::mutex owner_mutex;
    QsaRamCache* owner=nullptr;
    bool can_spill=false;
    bool fixed_ram_only=false;
    std::size_t charged_ram() const {
        const auto bytes=logical.load();
        return bytes-std::min(bytes,shared_prefix_ram.load());
    }
};
struct PrefillTransfer {
    std::shared_ptr<RamAccount> account;
    std::size_t charge=0,packet_bytes=0;
    std::array<tb::Tensor,2> packets,records;
    std::array<cudaEvent_t,2> done{};
    ~PrefillTransfer() {
        for(auto event:done)if(event) {
            (void)cudaEventSynchronize(event);(void)cudaEventDestroy(event);
        }
        if(charge)account->physical.fetch_sub(charge);
    }
};
struct RamForegroundCodec {
    std::atomic<bool>& flag;bool previous;
    explicit RamForegroundCodec(std::atomic<bool>& value):flag(value),previous(value.exchange(true)){}
    ~RamForegroundCodec(){flag.store(previous);}
};
struct RamPinnedPool;
void trim_pinned_pool(const std::shared_ptr<RamPinnedPool>& pool);
struct RamClaims {
    std::vector<std::uint8_t>& marks;
    std::vector<std::size_t> ids;
    bool active=true;
    RamClaims(std::vector<std::uint8_t>& storage,std::size_t capacity,std::size_t count):marks(storage) {
        if(marks.size()<capacity)marks.resize(capacity);ids.reserve(count);
    }
    void add(std::size_t id) {
        if(!marks.at(id)){ids.push_back(id);marks[id]=1;}
    }
    void clear()noexcept {
        if(active){for(auto id:ids)marks[id]=0;active=false;}
    }
    ~RamClaims(){clear();}
};
struct RamRetirement {
    cudaEvent_t done=nullptr;
    std::vector<tq::RamKvChange> owners;
    tb::Tensor descriptors,ids,host_descriptors,host_ids;
    const tq::RamBlockView* mapped_descriptors=nullptr;
    const std::int32_t* mapped_ids=nullptr;
    std::shared_ptr<void> temporary;
};
struct SharedRamStageCache {
    tb::Tensor slots,locks,frame,payload;
    tq::RamStageCache layout;
    std::size_t bytes=0;std::uint64_t next_owner=0;
    bool spare_vram;
    SharedRamStageCache(std::size_t budget,std::size_t stride,const tb::TensorOptions& options,bool spare)
        :spare_vram(spare) {
        const ResidentAllocationScope resident;
        const auto unit=tq::ram_cache_ways*(stride+sizeof(tq::RamCacheSlot))+sizeof(std::uint32_t);
        const auto sets=std::min<std::size_t>(INT_MAX/tq::ram_cache_ways,budget>8?(budget-8)/unit:0);
        if(!sets)return;
        slots=tb::empty({int64_t(sets*tq::ram_cache_ways*sizeof(tq::RamCacheSlot))},options.dtype(tb::kUInt8));
        locks=tb::empty({int64_t(sets)},options.dtype(tb::kInt32));frame=tb::empty({8},options.dtype(tb::kUInt8));
        payload=tb::empty({int64_t(sets*tq::ram_cache_ways*stride)},options.dtype(tb::kUInt8));
        layout={slots.data_ptr<tq::RamCacheSlot>(),locks.data_ptr<std::uint32_t>(),frame.data_ptr<std::uint64_t>(),
            payload.data_ptr<std::uint8_t>(),payload.nbytes(),stride,int(sets)};
        bytes=slots.nbytes()+locks.nbytes()+frame.nbytes()+payload.nbytes();
        check(tq::tq_reset_ram_stage_cache(layout,mfq_current_cuda_stream()));
    }
    tq::RamStageCache acquire() {
        if(next_owner==UINT64_MAX)throw std::overflow_error("QSA shared cache identity exhausted");
        auto result=layout;result.owner=++next_owner;return result;
    }
};
struct RamWorker {
    int device;cudaStream_t stream;
    std::mutex mutex;std::condition_variable changed;
    std::deque<std::pair<std::uint64_t,std::packaged_task<void()>>> jobs;
    std::vector<std::weak_ptr<RamAccount>> accounts;
    std::vector<const RamAccount*> restore_participants;
    struct ResidentPrefixFile {
        std::weak_ptr<const mfq::FileRangeReader> reader;
        std::unordered_map<std::uint64_t,std::weak_ptr<const tq::RamKvImage>> images;
    };
    // Consulted on the serialized model thread, while restoring a checkpoint.
    std::unordered_map<const mfq::FileRangeReader*,ResidentPrefixFile> prefix_images;
    tq::RamKvImagePtr shared_prefix_image(const tq::RamKvSnapshotBlock& block) const {
        const auto* backing=block.backing.get();
        if(!backing || !backing->source || !backing->source->file_reader)return {};
        const auto& source=*backing->source;
        if(backing->offset>UINT64_MAX-source.file_offset)return {};
        const auto file=prefix_images.find(source.file_reader.get());
        if(file==prefix_images.end() || file->second.reader.lock()!=source.file_reader)return {};
        const auto entry=file->second.images.find(source.file_offset+backing->offset);
        if(entry==file->second.images.end())return {};
        auto image=entry->second.lock();
        if(!image || image->key_bytes!=backing->key_bytes || image->value_bytes!=backing->value_bytes ||
                image->key_bits!=backing->key_bits || image->value_bits!=backing->value_bits ||
                image->key_state_bits!=backing->key_state_bits || image->value_state_bits!=backing->value_state_bits)return {};
        return image;
    }
    void remember_prefix_images(const tq::RamKvSnapshot& source,const tq::RamKvSnapshot& resident) {
        std::erase_if(prefix_images,[](const auto& item){return item.second.reader.expired();});
        for(std::size_t id=0;id<source.blocks.size();++id) {
            const auto& block=source.blocks[id];
            const auto& image=resident.blocks[id].image;
            if(!image || !block.backing || !block.backing->source ||
                    resident.blocks[id].tier!=block.backing->tier)continue;
            const auto& range=*block.backing->source;
            if(!range.file_reader || block.backing->offset>UINT64_MAX-range.file_offset)continue;
            auto& file=prefix_images[range.file_reader.get()];
            if(file.reader.lock()!=range.file_reader) {
                file.images.clear();
                file.reader=range.file_reader;
            }
            file.images[range.file_offset+block.backing->offset]=image;
        }
    }
    void rebalance_restore();
    std::shared_ptr<RamAccount> pool_account=std::make_shared<RamAccount>();
    std::atomic<std::size_t> workspace_bytes{0};
    std::shared_ptr<RamPinnedPool> pinned_pool;
    std::shared_ptr<PrefillTransfer> prefill_transfer;
    std::shared_ptr<SharedRamStageCache> stage_cache;
    mfq::HostParallelPool cpu_codecs;
    // Cache calls are serialized on their owning model thread. Collect all
    // layers' finished publications instead of retaining a previous layer's
    // obsolete pinned images until that layer runs again.
    std::mutex retired_mutex;
    std::vector<RamRetirement> retired;
    std::vector<RamRetirement> publications;
    // Model-thread scratch shared by its layers. Clear only selected IDs,
    // keeping union construction independent of the total context length.
    std::vector<std::uint8_t> claims;
    std::uint64_t next_job=0;
    std::atomic<std::uint64_t> running_job{0},urgent_through{0};
    bool stop=false;std::thread worker;
    explicit RamWorker(int device,cudaStream_t stream):device(device),stream(stream),
        cpu_codecs([device](std::size_t){check(cudaSetDevice(device));}),worker([this]{
        for(;;) {
            std::pair<std::uint64_t,std::packaged_task<void()>> task;
            {std::unique_lock<std::mutex> lock(mutex);changed.wait(lock,[&]{return stop || !jobs.empty();});
                if(jobs.empty())return;task=std::move(jobs.front());jobs.pop_front();}
            running_job.store(task.first);task.second();running_job.store(0);
        }
    }){}
    ~RamWorker(){
        try{reserve_qsa_ram_budget(device,stream,0);}catch(...){}
        {std::lock_guard<std::mutex> lock(mutex);stop=true;changed.notify_all();}worker.join();
        for(auto& p:retired)if(p.done){(void)cudaEventSynchronize(p.done);(void)cudaEventDestroy(p.done);}
        for(auto& p:publications)if(p.done)(void)cudaEventDestroy(p.done);
    }
    std::size_t collect() {
        std::vector<RamRetirement> ready;
        {
            std::lock_guard<std::mutex> lock(retired_mutex);
            for(auto it=retired.begin();it!=retired.end();) {
                const auto status=cudaEventQuery(it->done);
                if(status==cudaSuccess) {
                    ready.push_back(std::move(*it));it=retired.erase(it);
                }else {if(status!=cudaErrorNotReady)check(status);++it;}
            }
        }
        // Releasing a slab can enter an allocator. Keep this outside the
        // retirement lock so model publication and codec work can proceed.
        const auto completed=ready.size();
        for(auto& p:ready) {
            if(p.host_ids.defined()) {
                // Its event has completed. Re-record it only after this whole
                // publication is acquired again, together with its buffers.
                p.owners.clear();publications.push_back(std::move(p));
            }else {check(cudaEventDestroy(p.done));p.done=nullptr;}
        }
        ready.clear();trim_pinned_pool(pinned_pool);
        return completed;
    }
    RamRetirement publication(std::size_t count,const tb::TensorOptions& options,bool mapped) {
        auto best=publications.end();
        for(auto it=publications.begin();it!=publications.end();++it)
            if(std::size_t(it->host_ids.numel())>=count &&
                    (best==publications.end() || it->host_ids.numel()<best->host_ids.numel()))best=it;
        if(best!=publications.end()) {
            auto result=std::move(*best);*best=std::move(publications.back());publications.pop_back();
            return result;
        }
        std::size_t capacity=1;while(capacity<count)capacity*=2;
        ResidentAllocationScope resident;
        RamRetirement result;
        result.host_ids=tb::empty({int64_t(capacity)},options.device(tb::kCPU).dtype(tb::kInt32).pinned_memory(true));
        result.host_descriptors=tb::empty({int64_t(capacity*sizeof(tq::RamBlockView))},options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
        if(mapped) {
            void* pointer=nullptr;check(cudaHostGetDevicePointer(&pointer,result.host_ids.data_ptr(),0));
            result.mapped_ids=static_cast<const std::int32_t*>(pointer);
            check(cudaHostGetDevicePointer(&pointer,result.host_descriptors.data_ptr(),0));
            result.mapped_descriptors=static_cast<const tq::RamBlockView*>(pointer);
        }else {
            result.ids=tb::empty({int64_t(capacity)},options.dtype(tb::kInt32));
            result.descriptors=tb::empty({int64_t(capacity*sizeof(tq::RamBlockView))},options.dtype(tb::kUInt8));
        }
        pool_account->physical.fetch_add(result.host_ids.nbytes()+result.host_descriptors.nbytes());
        return result;
    }
    std::size_t release_readers() {
        std::vector<std::vector<tq::RamKvChange>> ready;
        {
            std::lock_guard<std::mutex> lock(retired_mutex);
            for(auto& p:retired) {
                if(p.owners.empty())continue;
                const auto status=cudaEventQuery(p.done);
                if(status==cudaSuccess)ready.push_back(std::move(p.owners));
                else if(status!=cudaErrorNotReady)check(status);
            }
        }
        // Background work may return image slots, but must not cudaFreeHost
        // transfer descriptors or release model-stream GPU allocations. Such
        // frees can wait for a host-served graph that the model is submitting.
        // Keep the empty retirement records for the model thread's collect().
        return ready.size();
    }
    void retire(RamRetirement pending) {
        std::lock_guard<std::mutex> lock(retired_mutex);
        retired.push_back(std::move(pending));
    }
    std::future<void> submit(std::function<void()> work,std::uint64_t& sequence) {
        std::packaged_task<void()> task(std::move(work));auto result=task.get_future();
        {std::lock_guard<std::mutex> lock(mutex);
            if(next_job==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("QSA RAM worker sequence exhausted");
            sequence=++next_job;jobs.emplace_back(sequence,std::move(task));changed.notify_one();}return result;
    }
    void prioritize(std::uint64_t through) {
        auto previous=urgent_through.load();
        while(previous<through && !urgent_through.compare_exchange_weak(previous,through)){}
    }
    std::size_t reclaim_capacity(std::size_t per_cache) {
        const auto physical=pool_account->physical.load();std::size_t logical=0,count=0;
        {std::lock_guard<std::mutex> lock(mutex);
            for(const auto& weak:accounts)if(auto a=weak.lock()){
                const auto bytes=a->logical.load();logical+=a->charged_ram();
                count+=bytes || a->target.load() || a->restore_target.load();
            }}
        const auto overhead=physical>logical?physical-logical:0;
        const auto maximum=std::numeric_limits<std::size_t>::max();
        if(!count || per_cache>(maximum-overhead)/count)throw std::overflow_error("QSA RAM protected capacity overflow");
        const auto target=per_cache*count+overhead;
        const auto free=mfq::sample_host_memory().available_host_bytes();
        const auto shortage=tq::ram_kv_reclaim_bytes(free,physical,target);
        return reclaim_qsa_ram_capacity(device,stream,shortage);
    }
    std::size_t last_reservation=0,last_reservation_free=0;
    std::size_t budget(const RamAccount* request=nullptr,bool rebalance=true) {
        std::size_t physical=pool_account->physical.load(),logical=0,target=0,count=0;
        std::size_t fixed_pending=0,request_resident=0,request_pending=0,request_shared=0;
        bool fixed_ram_only=true;
        {std::lock_guard<std::mutex> lock(mutex);
            for(auto it=accounts.begin();it!=accounts.end();)if(auto a=it->lock()) {
                const auto bytes=a->logical.load(),wanted=std::max(a->target.load(),a->restore_target.load());
                const auto shared=std::min(bytes,a->shared_prefix_ram.load());
                logical+=bytes-shared;
                const auto resident=std::max(bytes,a->restore_shared.load());
                const auto pending=wanted>resident?wanted-resident:0;
                fixed_pending+=pending;
                if(a.get()==request){request_resident=resident;request_pending=pending;request_shared=shared;}
                // SSD-backed KV competes for spare RAM, not for expert RAM.
                // Its complete-context target still counts it as an active
                // layer, but may not reserve space by evicting experts.
                if(!a->can_spill)target+=wanted;
                // An unused MTP head or empty session owns no KV share. Its
                // small staging allocation remains in the physical overhead.
                if(bytes || wanted) {
                    ++count;
                    fixed_ram_only=fixed_ram_only && a->fixed_ram_only;
                }
                ++it;
            }else it=accounts.erase(it);}
        auto free=mfq::sample_host_memory().available_host_bytes();
        const auto workspace=workspace_bytes.load();
        const auto kv_free=fixed_ram_only?0:tq::ram_kv_reclaim_bytes(0,physical,logical);
        const auto required_free=workspace>std::numeric_limits<std::size_t>::max()-kv_free?
            std::numeric_limits<std::size_t>::max():workspace+kv_free;
        const auto shortage=required_free>free?required_free-free:0;
        // Called on the model thread, outside store/pool locks and graph
        // capture. Release disposable prefix RAM before quantizing live KV.
        if(rebalance && shortage && QsaRamDecodeBudgetScope::claim_optional_reclaim(device) &&
                reclaim_qsa_ram(device,stream,shortage)) {
            free=mfq::sample_host_memory().available_host_bytes();
            physical=pool_account->physical.load();
        }
        const auto overhead=physical>logical?physical-logical:0;
        const auto maximum=std::numeric_limits<std::size_t>::max();
        const auto* configured_cap=std::getenv("TYLOINFER_KV_RAM_LIMIT_BYTES");
        const auto cap=configured_cap && *configured_cap?setting("TYLOINFER_KV_RAM_LIMIT_BYTES"):maximum;
        target=std::min(target,cap);
        if(target>maximum-overhead)throw std::overflow_error("QSA shared RAM target overflow");
        const auto wanted=target?target+overhead:0;
        // Only RAM-only layers require expert reclamation. Existing spillable
        // images may exceed that target and must not inflate its reservation.
        const auto required=fixed_ram_only?std::min(fixed_pending,cap>physical?cap-physical:0):
            wanted?tq::ram_kv_reclaim_bytes(0,std::min(physical,wanted),wanted):0;
        constexpr std::size_t granule=4ULL<<20;
        const auto rounded=required>maximum-(granule-1)?maximum:
            (required+granule-1)/granule*granule;
        if(rebalance && (rounded!=last_reservation ||
                (free<rounded && last_reservation_free>free && last_reservation_free-free>=granule))) {
            reserve_qsa_ram_budget(device,stream,rounded);last_reservation=rounded;
            free=mfq::sample_host_memory().available_host_bytes();
            physical=pool_account->physical.load();last_reservation_free=free;
        }
        // Pending prefill outputs and descriptor growth must remain possible
        // throughout the chunk, after every background KV maintenance pass.
        const auto free_for_kv=free-std::min(free,workspace);
        auto total=fixed_ram_only?physical+std::min<std::size_t>(free_for_kv,maximum-physical):
            tq::ram_kv_budget(free_for_kv,physical);
        total=std::min(total,cap);
        if(fixed_ram_only && request && total>=physical) {
            // The pool already charges resident bytes once. Divide growth,
            // rather than splitting existing images again between requests.
            const auto free=total-physical;
            const auto growth=fixed_pending>free?
                std::size_t(static_cast<long double>(free)*request_pending/fixed_pending):
                request_pending+(free-fixed_pending)/std::max<std::size_t>(count,1);
            return request_resident+std::min(growth,maximum-request_resident);
        }
        const auto share=count && total>overhead?(total-overhead)/count:0;
        return share+std::min(request_shared,maximum-share);
    }
};
// One background encoder per serialized model execution stream. Different
// streams never share staging memory or contend through a single codec job.
std::shared_ptr<RamWorker> worker_for(int device,cudaStream_t stream) {
    static thread_local std::map<std::pair<int,cudaStream_t>,std::weak_ptr<RamWorker>> workers;
    auto& weak=workers[{device,stream}];auto worker=weak.lock();
    if(!worker){worker=std::make_shared<RamWorker>(device,stream);weak=worker;}return worker;
}
thread_local bool ram_pinned_foreground=false;
thread_local bool ram_pageable_allocation=false;
struct RamPinnedForeground {
    bool previous=ram_pinned_foreground;
    RamPinnedForeground(){ram_pinned_foreground=true;}
    ~RamPinnedForeground(){ram_pinned_foreground=previous;}
};
struct RamPageableScope {
    bool previous=ram_pageable_allocation;
    explicit RamPageableScope(bool allowed){ram_pageable_allocation=allowed;}
    ~RamPageableScope(){ram_pageable_allocation=previous;}
};
struct RamSlab {
    tb::Tensor host;
    std::uint8_t* mapped=nullptr;
    std::size_t stride=0;
    std::vector<unsigned> free;
    unsigned occupied=0;
};
struct RamPinnedPool:std::enable_shared_from_this<RamPinnedPool> {
    std::mutex mutex;
    // Only slabs with free slots are indexed. Full slabs remain owned by
    // their images, so allocation does not scan the entire context.
    std::map<std::size_t,std::vector<std::shared_ptr<RamSlab>>> available;
    std::map<std::size_t,std::size_t> cached_empty_slots;
    std::map<std::size_t,std::size_t> total_slabs;
    tb::TensorOptions options;std::shared_ptr<RamAccount> account;
    int device=0;cudaStream_t stream=nullptr;
    bool pageable=false;
    std::shared_ptr<RamPinnedPool> pageable_pool;
    std::atomic<std::size_t> own_bytes{0},pinned_ceiling{SIZE_MAX};
    bool has_pageable()const{return pageable_pool && pageable_pool->own_bytes.load()!=0;}
    std::atomic<std::size_t> pending_pinned{0};
    tb::Tensor allocate_storage(std::size_t bytes) {
        if(pageable) {
            try{return tb::empty({int64_t(bytes)},options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(false));}
            catch(const std::bad_alloc&){throw MfqResourceExhausted("QSA RAM allocation failed: requested="+std::to_string(bytes));}
        }
        const auto allocate=[&]{return tb::empty({int64_t(bytes)},options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));};
        // A request scope already performs one allocator-only recovery. Do not
        // stack a second retry over a failed central allocation.
        if(ram_pageable_allocation || (ram_pinned_foreground && pinned_allocation_scope_active()))return allocate();
        return allocate_with_pinned_recovery(bytes,ram_pinned_foreground,allocate,
            [&](std::size_t n){return reclaim_qsa_pinned(device,stream,n);},
            [&](std::size_t n){
                auto previous=pending_pinned.load();
                while(previous<n && !pending_pinned.compare_exchange_weak(previous,n)){}
            });
    }
    void resolve_pinned() {
        const auto bytes=pending_pinned.exchange(0);
        if(bytes && !reclaim_qsa_pinned(device,stream,bytes))
            throw MfqResourceExhausted("CUDA pinned allocation failed: requested="+std::to_string(bytes)+" optional_pinned_released=0");
    }
    struct Allocation {
        std::shared_ptr<RamPinnedPool> pool;std::shared_ptr<RamSlab> slab;unsigned slot=0;
        ~Allocation() {
            if(!slab)return;
            std::lock_guard<std::mutex> lock(pool->mutex);
            auto& list=pool->available[slab->stride];
            if(slab->free.empty())list.push_back(slab);
            slab->free.push_back(slot);--slab->occupied;
            if(!slab->occupied) {
                auto& cached=pool->cached_empty_slots[slab->stride];
                // Slot returns also run on codec threads. Leave physical
                // pinned-page release to trim() on the model thread.
                cached+=slab->free.size();
            }
            // Drop this allocation's last slab reference before making an
            // empty slab visible to the model thread's physical trimming.
            slab.reset();
        }
    };
    static std::size_t image_stride(std::size_t key,std::size_t value) {
        constexpr auto maximum=std::numeric_limits<std::size_t>::max();
        if(!key || !value || key>maximum-15)throw std::invalid_argument("Invalid QSA RAM image extent");
        const auto aligned=(key+15)&~std::size_t(15);
        if(value>maximum-aligned || aligned+value>maximum-15)
            throw std::invalid_argument("QSA RAM image extent overflow");
        return (aligned+value+15)&~std::size_t(15);
    }
    static std::size_t image_stride(const tq::MicroblockImage& image) {
        return image_stride(image.key.size(),image.value.size());
    }
    tq::RamKvImagePtr reserve_image(const tq::RamKvImage& image,bool grow,std::size_t pending=1) {
        const auto stride=image_stride(image.key_bytes,image.value_bytes);
        const auto key_extent=(image.key_bytes+15)&~std::size_t(15);
        std::shared_ptr<Allocation> owner;std::shared_ptr<tq::RamKvImage> result;
        const auto prepare=[&] {
            owner=std::make_shared<Allocation>();owner->pool=shared_from_this();
            result=std::make_shared<tq::RamKvImage>();result->owner=owner;
        };
        if(grow)prepare();
        std::shared_ptr<RamSlab> slab;unsigned slot=0;
        {
            std::unique_lock<std::mutex> lock(mutex);
            const auto found=available.find(stride);
            if(!grow && (found==available.end() || found->second.empty()))return {};
            if(!grow)prepare();
            auto& list=available[stride];
            cached_empty_slots.try_emplace(stride,0);
            if(!list.empty()) {
                slab=list.back();
                if(!slab->occupied)cached_empty_slots[stride]-=slab->free.size();
            }
            if(!slab) {
                if(!grow)return {};
                // Slot returns run in a shared_ptr destructor. Reserve its
                // free-slab index here so returning a full slab cannot allocate.
                slab=std::make_shared<RamSlab>();slab->stride=stride;
                const auto ordinary=std::max<std::size_t>(1,(1ULL<<20)/stride);
                const auto slots=std::max(ordinary,std::min<std::size_t>(pending,tq::RamPressureConfig{}.quantize_tokens/4));
                // A pinned release can join a stream whose host task needs this
                // pool. Never enter an allocator/reclaimer with its lock held.
                lock.unlock();
                slab->host=allocate_storage(slots*stride);
                if(!pageable) {
                    void* mapped=nullptr;check(cudaHostGetDevicePointer(&mapped,slab->host.data_ptr(),0));
                    slab->mapped=static_cast<std::uint8_t*>(mapped);
                }
                slab->free.reserve(slots);for(unsigned i=0;i<slots;++i)slab->free.push_back(unsigned(slots-1-i));
                lock.lock();
                auto& total=total_slabs[stride];
                if(list.capacity()<=total)list.reserve(std::max(total+1,list.capacity()*2));
                list.push_back(slab);++total;account->physical.fetch_add(slots*stride);own_bytes.fetch_add(slots*stride);
            }
            slot=slab->free.back();slab->free.pop_back();++slab->occupied;
            if(slab->free.empty())list.pop_back();
        }
        owner->slab=slab;owner->slot=slot;
        auto* key=slab->host.data_ptr<std::uint8_t>()+slot*stride;
        result->key=key;result->value=key+key_extent;result->key_bytes=image.key_bytes;result->value_bytes=image.value_bytes;
        result->key_bits=image.key_bits;result->value_bits=image.value_bits;
        result->key_state_bits=image.key_state_bits;result->value_state_bits=image.value_state_bits;
        return result;
    }
    tq::RamKvImagePtr allocate(const tq::MicroblockImage& image,std::size_t* remaining=nullptr,bool allow_pageable=false) {
        tq::RamKvImage layout;
        layout.key=image.key.data();layout.value=image.value.data();
        layout.key_bytes=image.key.size();layout.value_bytes=image.value.size();
        layout.key_bits=image.key_bits;layout.value_bits=image.value_bits;
        layout.key_state_bits=image.key_state_bits;layout.value_state_bits=image.value_state_bits;
        return allocate_view(layout,remaining,allow_pageable);
    }
    tq::RamKvImagePtr allocate_view(const tq::RamKvImage& image,std::size_t* remaining=nullptr,bool allow_pageable=false) {
        std::size_t pending=1;
        if(remaining) {
            // Count an image once before choosing its storage pool. Parallel
            // codecs share this lock even when pinned allocation falls back.
            std::lock_guard lock(mutex);
            if(!*remaining)throw std::logic_error("QSA RAM allocation batch exhausted");
            pending=(*remaining)--;
        }
        return allocate_counted(image,pending,allow_pageable);
    }
    tq::RamKvImagePtr allocate_counted(const tq::RamKvImage& image,std::size_t pending,bool allow_pageable) {
        auto result=reserve_counted(image,pending,allow_pageable);
        std::memcpy(const_cast<std::uint8_t*>(result->key),image.key,image.key_bytes);
        std::memcpy(const_cast<std::uint8_t*>(result->value),image.value,image.value_bytes);
        return result;
    }
    tq::RamKvImagePtr reserve_counted(const tq::RamKvImage& image,std::size_t pending,bool allow_pageable) {
        tq::RamKvImagePtr result;
        std::optional<PinnedReclaimPause> no_expert_reclaim;
        if(allow_pageable)no_expert_reclaim.emplace();
        RamPageableScope storage_scope(allow_pageable);
        const auto ceiling=pinned_ceiling.load(),live=pinned_memory_budget_stats().live_bytes;
        if(allow_pageable && ceiling!=SIZE_MAX && live>=ceiling) {
            result=reserve_image(image,false,pending);
            if(!result)return pageable_pool->reserve_counted(image,pending,false);
        }
        try{if(!result)result=reserve_image(image,true,pending);}
        catch(const MfqResourceExhausted&) {
            if(!allow_pageable)throw;
            pinned_ceiling.store(pinned_memory_budget_stats().live_bytes);
            return pageable_pool->reserve_counted(image,pending,false);
        }
        return result;
    }
    tq::RamKvImagePtr try_reserve(const tq::RamKvSnapshotBlock& source) {
        if(source.image) {
            const auto owner=std::static_pointer_cast<const Allocation>(source.image->owner);
            return owner->slab->mapped?source.image:tq::RamKvImagePtr{};
        }
        if(!source.backing)throw std::runtime_error("QSA SSD source has no retained backing");
        const auto& backing=*source.backing;tq::RamKvImage layout;
        layout.key_bytes=backing.key_bytes;layout.value_bytes=backing.value_bytes;
        layout.key_bits=backing.key_bits;layout.value_bits=backing.value_bits;
        layout.key_state_bits=backing.key_state_bits;layout.value_state_bits=backing.value_state_bits;
        // A graph may wait for this CPU task: consume only existing locked
        // slots, never call a CUDA allocator or trim pinned slabs here.
        return reserve_image(layout,false);
    }
    std::vector<tq::RamKvImagePtr> allocate_batch(const std::vector<tq::MicroblockImage>& images,bool allow_pageable=false) {
        std::map<std::size_t,std::size_t> remaining;
        for(const auto& image:images)++remaining[image_stride(image)];
        std::vector<tq::RamKvImagePtr> result;result.reserve(images.size());
        for(const auto& image:images)result.push_back(allocate(image,&remaining.at(image_stride(image)),allow_pageable));
        return result;
    }
    std::vector<tq::RamKvRelocation> compact(const std::vector<tq::RamKvImagePtr>& images,std::size_t goal) {
        auto result=pageable_pool?pageable_pool->compact(images,goal):std::vector<tq::RamKvRelocation>{};
        std::size_t evacuated=0;std::set<const RamSlab*> released;
        for(const auto& moved:result) {
            const auto owner=std::static_pointer_cast<const Allocation>(images[moved.index]->owner);
            if(released.insert(owner->slab.get()).second)evacuated+=owner->slab->host.nbytes();
        }
        if(evacuated>=goal)return result;
        struct Group {std::shared_ptr<RamSlab> slab;std::vector<std::size_t> indices;bool exclusive=true;};
        std::map<RamSlab*,Group> grouped;
        for(std::size_t i=0;i<images.size();++i) {
            const auto owner=std::static_pointer_cast<const Allocation>(images[i]->owner);
            if(owner->pool.get()!=this)continue;
            auto& group=grouped[owner->slab.get()];group.slab=owner->slab;group.indices.push_back(i);
            group.exclusive&=images[i].use_count()==2 && owner.use_count()==2;
        }
        std::vector<Group*> order;order.reserve(grouped.size());
        for(auto& [key,group]:grouped)if(group.exclusive)order.push_back(&group);
        std::sort(order.begin(),order.end(),[](const auto* a,const auto* b) {
            return a->indices.size()*b->slab->host.numel()<b->indices.size()*a->slab->host.numel();
        });
        struct Excluded {
            RamPinnedPool& pool;std::vector<std::shared_ptr<RamSlab>> slabs;
            ~Excluded(){std::lock_guard<std::mutex> lock(pool.mutex);
                for(auto& slab:slabs)pool.available.at(slab->stride).push_back(std::move(slab));}
        } excluded{*this,{}};
        {
            std::lock_guard<std::mutex> lock(mutex);std::size_t available_slabs=0;
            for(const auto& [stride,list]:available)available_slabs+=list.size();
            excluded.slabs.reserve(available_slabs);
            // Keep compaction within this store. Mixing different layers in
            // destination slabs would prevent later whole-slab evacuation.
            for(auto& [stride,list]:available)for(auto it=list.begin();it!=list.end();) {
                const auto found=grouped.find(it->get());
                const bool owned=found!=grouped.end() && found->second.exclusive &&
                    found->second.indices.size()==(*it)->occupied;
                if((*it)->occupied && !owned) {
                    excluded.slabs.push_back(std::move(*it));*it=std::move(list.back());list.pop_back();
                }else ++it;
            }
        }
        result.reserve(images.size());
        for(const auto* group:order) {
            const auto& slab=group->slab;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if(slab->occupied!=group->indices.size() || slab->free.empty())continue;
                auto& list=available.at(slab->stride);std::size_t slots=0;
                for(const auto& candidate:list)if(candidate!=slab)slots+=candidate->free.size();
                if(slots<slab->occupied)continue;
                const auto found=std::find(list.begin(),list.end(),slab);
                if(found==list.end())continue;
                // Source images remain alive until descriptor publication.
                // Keep every evacuated slab out of subsequent allocations.
                *found=std::move(list.back());list.pop_back();excluded.slabs.push_back(slab);
            }
            const auto first=result.size();
            for(const auto index:group->indices) {
                const auto& old=*images[index];auto replacement=reserve_image(old,false);
                if(!replacement){result.resize(first);break;}
                std::memcpy(const_cast<std::uint8_t*>(replacement->key),old.key,old.key_bytes);
                std::memcpy(const_cast<std::uint8_t*>(replacement->value),old.value,old.value_bytes);
                result.push_back({index,std::move(replacement)});
            }
            if(result.size()-first==group->indices.size())evacuated+=slab->host.nbytes();
            if(evacuated>=goal)break;
        }
        return result;
    }
    void trim(std::size_t retain=2048/4) {
        std::vector<std::shared_ptr<RamSlab>> unused;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for(auto& [stride,cached]:cached_empty_slots) {
                if(cached<=retain)continue;
                auto& list=available.at(stride);
                for(auto it=list.begin();it!=list.end() && cached>retain;) {
                    const auto& slab=*it;
                    if(slab->occupied){++it;continue;}
                    auto& total=total_slabs.at(stride);
                    unused.push_back(slab);
                    cached-=slab->free.size();--total;
                    account->physical.fetch_sub(std::size_t(slab->host.numel()));
                    own_bytes.fetch_sub(std::size_t(slab->host.numel()));
                    it=list.erase(it);
                }
            }
        }
        if(pageable_pool)pageable_pool->trim(retain);
    }
};
void trim_pinned_pool(const std::shared_ptr<RamPinnedPool>& pool) {
    if(!pool)return;
    cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture));
    if(capture!=cudaStreamCaptureStatusNone)return;
    const auto status=cudaStreamQuery(mfq_current_cuda_stream());
    if(status==cudaSuccess)pool->trim();
    else if(status!=cudaErrorNotReady)check(status);
}
struct SsdGraphArena {
    tb::Tensor payload,ids,images,backup,saved;
    std::uint8_t* mapped_payload=nullptr;
    std::int32_t* mapped_ids=nullptr;
    tq::RamBlockView* mapped_images=nullptr;
    std::shared_ptr<RamAccount> account;
    std::size_t bytes=0,stride=0;
    ~SsdGraphArena(){if(bytes)account->physical.fetch_sub(bytes);}
};
struct RamStageProbe {
    tb::Tensor records,host_records,copied,host_copied;
    std::shared_ptr<RamAccount> account;
    std::size_t host_bytes=0;
    int layer=-1,tile=-1,total=0;
    ~RamStageProbe(){if(host_bytes)account->physical.fetch_sub(host_bytes);}
};
struct RamStageProbeBatch {
    std::vector<std::shared_ptr<RamStageProbe>> stages;
    std::uint64_t replay=0;
    void finish() {
        ++replay;
        using Identity=std::tuple<std::uintptr_t,std::uintptr_t,std::uint64_t,unsigned,unsigned,unsigned,unsigned>;
        struct Layer {
            std::uint64_t calls=0,copied=0,unique=0,valid=0,immutable=0,mutable_bytes=0;
            std::uint64_t fp16=0,affine8=0,tq=0,identity_repeat_bytes=0,logical_repeat_blocks=0;
            std::set<int> logical;
            std::set<Identity> identities;
        };
        std::map<int,Layer> layers;
        for(const auto& stage:stages) {
            auto& layer=layers[stage->layer];++layer.calls;
            layer.copied+=*stage->host_copied.data_ptr<std::uint64_t>();
            const auto* records=stage->host_records.data_ptr<tq::RamSelectedRecord>();
            std::set<int> local;
            for(int at=0;at<stage->total;++at) {
                const auto& r=records[at];
                if(!r.selected.token_mask || !r.source.image.key || !r.source.image.value)continue;
                ++layer.valid;
                if(!local.insert(r.selected.id).second)continue;
                ++layer.unique;
                const auto bytes=std::uint64_t(r.source.key_bytes)+r.source.value_bytes;
                if(r.source.generation==tq::ram_uncached_generation)layer.mutable_bytes+=bytes;
                else {
                    layer.immutable+=bytes;
                    if(!layer.logical.insert(r.selected.id).second)++layer.logical_repeat_blocks;
                    const Identity identity{reinterpret_cast<std::uintptr_t>(r.source.image.key),reinterpret_cast<std::uintptr_t>(r.source.image.value),r.source.generation,
                        r.source.image.key_bits,r.source.image.value_bits,r.source.key_bytes,r.source.value_bytes};
                    if(!layer.identities.insert(identity).second)layer.identity_repeat_bytes+=bytes;
                }
                const auto tier=[&](unsigned bits,unsigned n) {
                    if(!(bits&255u))layer.fp16+=n;
                    else if((bits&255u)==254)layer.affine8+=n;
                    else layer.tq+=n;
                };
                tier(r.source.image.key_bits,r.source.key_bytes);
                tier(r.source.image.value_bits,r.source.value_bytes);
            }
        }
        for(const auto& entry:layers) {
            const auto& v=entry.second;
            std::fprintf(stderr,"native_qsa_stage_probe actual_match=1 replay=%llu layer=%d calls=%llu copied_bytes=%llu unique_blocks=%llu valid_selections=%llu immutable_encoded_bytes=%llu mutable_device_bytes=%llu fp16_bytes=%llu affine8_bytes=%llu tq_bytes=%llu logical_repeat_blocks=%llu same_descriptor_repeat_bytes=%llu descriptor_match_not_content_proof=1 diagnostic_only=1\n",
                static_cast<unsigned long long>(replay),entry.first,
                static_cast<unsigned long long>(v.calls),static_cast<unsigned long long>(v.copied),
                static_cast<unsigned long long>(v.unique),static_cast<unsigned long long>(v.valid),
                static_cast<unsigned long long>(v.immutable),static_cast<unsigned long long>(v.mutable_bytes),
                static_cast<unsigned long long>(v.fp16),static_cast<unsigned long long>(v.affine8),
                static_cast<unsigned long long>(v.tq),static_cast<unsigned long long>(v.logical_repeat_blocks),
                static_cast<unsigned long long>(v.identity_repeat_bytes));
        }
    }
};
struct StageArena {
    StageArena()=default;
    StageArena(const StageArena&)=delete;
    StageArena& operator=(const StageArena&)=delete;
    tb::Tensor payload,first,views,missing,host_missing,host_selected;
    std::int32_t* mapped_missing=nullptr;
    tq::SelectedBlock* mapped_selected=nullptr;
    // CPU tasks run in stream order. Preserve their small selection list
    // before releasing the GPU to overwrite the shared mapped export.
    std::vector<tq::SelectedBlock> served_selected;
    std::shared_ptr<SsdGraphArena> ssd;
    std::shared_ptr<RamAccount> pinned_account;
    std::size_t stride=0,host_bytes=0;
    int queries=0;
    ~StageArena(){if(host_bytes)pinned_account->physical.fetch_sub(host_bytes);}
};
std::shared_ptr<StageArena> arena_for(int device,cudaStream_t stream,int capacity,int count,int queries,std::size_t stride,const tb::TensorOptions& options,
        const std::shared_ptr<RamAccount>& pinned_account,bool prefill) {
    using Key=std::tuple<int,cudaStream_t,int,int,int,std::size_t,bool>;
    static thread_local std::map<Key,std::weak_ptr<StageArena>> arenas;
    auto& weak=arenas[{device,stream,capacity,count,queries,stride,prefill}];auto arena=weak.lock();
    if(!arena) {
        const ResidentAllocationScope resident(!prefill);
        arena=std::make_shared<StageArena>();arena->stride=stride;arena->pinned_account=pinned_account;arena->queries=queries;
        const auto slots=int64_t(queries)*count;
        arena->payload=tb::empty({slots*int64_t(stride)},options.dtype(tb::kUInt8));
        arena->first=tb::empty({capacity},options.dtype(tb::kInt32));
        arena->missing=tb::empty({1+slots},options.dtype(tb::kInt32));
        arena->host_missing=tb::empty({1+slots},options.device(tb::kCPU).dtype(tb::kInt32).pinned_memory(true));
        arena->host_selected=tb::empty({slots*int64_t(sizeof(tq::SelectedBlock))},options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
        const auto mapped=[](const tb::Tensor& tensor) {
            void* pointer=nullptr;const auto status=cudaHostGetDevicePointer(&pointer,tensor.data_ptr(),0);
            if(status==cudaSuccess)return pointer;
            if(status!=cudaErrorInvalidValue && status!=cudaErrorNotSupported && status!=cudaErrorMemoryAllocation)check(status);
            (void)cudaGetLastError();return static_cast<void*>(nullptr);
        };
        arena->mapped_missing=static_cast<std::int32_t*>(mapped(arena->host_missing));
        arena->mapped_selected=static_cast<tq::SelectedBlock*>(mapped(arena->host_selected));
        arena->views=tb::empty({slots*int64_t(sizeof(tq::TqBlockView))},options.dtype(tb::kUInt8));weak=arena;
        arena->served_selected.resize(std::size_t(slots));
        arena->host_bytes=arena->host_missing.nbytes()+arena->host_selected.nbytes()+arena->served_selected.capacity()*sizeof(tq::SelectedBlock);
        pinned_account->physical.fetch_add(arena->host_bytes);
        if(std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH")) {
            static thread_local unsigned reported=0;
            const auto mask=1u<<(queries-1);
            if(!(reported&mask)) {
                reported|=mask;
                std::fprintf(stderr,"native_qsa_stage_arena actual_match=1 queries=%d vram_payload_bytes=%zu\n",
                    queries,arena->payload.nbytes());
            }
        }
    }
    return arena;
}
}
static std::shared_ptr<void> register_ram_reclaimer(int device,cudaStream_t stream,QsaRamReclaimer reclaim,unsigned kind) {
    if(!reclaim)return {};
    const auto registry=ram_reclaim_registry();
    auto registration=std::make_shared<RamReclaimRegistration>();
    registration->key={device,stream};registration->reclaim=std::move(reclaim);registration->registry=registry;registration->kind=kind;
    {std::lock_guard<std::mutex> lock(registry->mutex);registry->entries[registration->key].push_back(registration);}
    return registration;
}
std::shared_ptr<void> register_qsa_ram_reclaimer(int device,cudaStream_t stream,QsaRamReclaimer reclaim) {
    return register_ram_reclaimer(device,stream,std::move(reclaim),0);
}
std::shared_ptr<void> register_qsa_ram_capacity_reclaimer(int device,cudaStream_t stream,QsaRamReclaimer reclaim) {
    return register_ram_reclaimer(device,stream,std::move(reclaim),1);
}
static std::size_t reclaim_ram(int device,cudaStream_t stream,std::size_t bytes,unsigned kind) {
    if(!bytes && kind!=2)return 0;
    const auto registry=ram_reclaim_registry();
    std::vector<std::shared_ptr<RamReclaimRegistration>> callbacks;
    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        const auto it=registry->entries.find({device,stream});
        if(it==registry->entries.end())return 0;
        // Reserve before locking weak entries: an allocation failure must
        // not destroy their last owner while the registry mutex is held.
        callbacks.reserve(it->second.size());
        for(const auto& weak:it->second)if(auto owner=weak.lock();owner && owner->kind==kind)callbacks.push_back(std::move(owner));
    }
    std::size_t released=0;
    for(const auto& callback:callbacks) {
        const auto n=callback->reclaim(kind==2?bytes:bytes-released);
        released+=std::min(n,kind==2?std::numeric_limits<std::size_t>::max()-released:bytes-released);
        if(kind!=2 && released==bytes)break;
    }
    return released;
}
std::size_t reclaim_qsa_ram(int device,cudaStream_t stream,std::size_t bytes) {
    return reclaim_ram(device,stream,bytes,0);
}
std::size_t reclaim_qsa_ram_capacity(int device,cudaStream_t stream,std::size_t bytes) {
    return reclaim_ram(device,stream,bytes,1);
}
std::shared_ptr<void> register_qsa_ram_budget_reserver(int device,cudaStream_t stream,QsaRamReclaimer reserve) {
    return register_ram_reclaimer(device,stream,std::move(reserve),2);
}
std::size_t reserve_qsa_ram_budget(int device,cudaStream_t stream,std::size_t free_bytes) {
    return reclaim_ram(device,stream,free_bytes,2);
}
std::shared_ptr<void> register_qsa_pinned_reclaimer(int device,cudaStream_t stream,QsaRamReclaimer reclaim) {
    return register_ram_reclaimer(device,stream,std::move(reclaim),3);
}
std::size_t reclaim_qsa_pinned(int device,cudaStream_t stream,std::size_t bytes) {
    return reclaim_ram(device,stream,bytes,3);
}
static std::shared_ptr<void> hold_worker_workspace(std::shared_ptr<RamWorker> owner,std::size_t bytes) {
    if(!bytes)return {};
    struct Reservation {
        std::shared_ptr<RamWorker> worker;std::size_t bytes;
        Reservation(std::shared_ptr<RamWorker> owner,std::size_t size):worker(std::move(owner)),bytes(size) {
            auto previous=worker->workspace_bytes.load();
            do {
                if(bytes>std::numeric_limits<std::size_t>::max()-previous)
                    throw std::overflow_error("QSA host workspace overflow");
            }while(!worker->workspace_bytes.compare_exchange_weak(previous,previous+bytes));
        }
        ~Reservation(){worker->workspace_bytes.fetch_sub(bytes);}
    };
    return std::make_shared<Reservation>(std::move(owner),bytes);
}
void prepare_qsa_ram_workspace(std::size_t bytes,
        const std::function<QsaRamWorkspaceProgress()>& reclaim) {
    auto previous=reclaim();bool retried=false;
    while(mfq::sample_host_memory().available_host_bytes()<bytes) {
        const auto current=reclaim();
        if(mfq::sample_host_memory().available_host_bytes()>=bytes)return;
        const bool progress=current.logical<previous.logical || current.physical<previous.physical;
        if(!progress && retried)
            throw std::runtime_error("Insufficient host memory for prefill workspace: required="+
                std::to_string(bytes)+" available="+
                std::to_string(mfq::sample_host_memory().available_host_bytes())+
                " logical_kv="+std::to_string(current.logical)+" pinned_kv="+std::to_string(current.physical));
        retried=!progress;previous=current;
    }
}
struct QsaRamCache::Impl {
    int capacity,rows,count,bits,device=0;std::size_t fp16_bytes,affine_bytes,cache_stride,spill_stride;
    bool ssd_enabled=false,mapped_capability=false,adaptive_precision=true,share_resident_prefix=false;
    const bool host_metadata;
    bool reported_ssd_fetch=false;
    bool profile_staging=false,direct_mapped_staging=false;
    char stage_probe_key=0;
    std::size_t sampled_budget=0;
    bool budget_sampled=false;
    std::atomic<bool> foreground_codec{false};
    // CPU transfer/flag builders must not change subsequent GPU allocations.
    const tb::TensorOptions options;
    Encode encode;
    std::shared_ptr<RamWorker> worker;
    std::shared_ptr<RamAccount> account=std::make_shared<RamAccount>();
    std::shared_ptr<RamPinnedPool> pool;
    std::unique_ptr<tq::RamKvStore> store;
    std::future<void> maintenance;
    std::uint64_t maintenance_sequence=0;
    std::atomic<std::size_t> required_capacity{0};
    tb::Tensor sources,live,cache_slots,cache_locks,cache_frame,cache_payload,cache_window_frames;
    tq::RamStageCache gpu_cache;
    bool shared_gpu_cache=false;
    bool use_spare_vram=false;
    bool prefill_staging=false;
    tb::Tensor cache_parameters,host_cache_parameters;
    std::uint64_t publication_revision=0,prepared_revision=0;
    std::size_t prefetched_blocks=0;
    std::shared_ptr<StageArena> arena;
    std::uint64_t graph_epoch=0;
    std::vector<tq::RamBlockView> host_sources;
    struct PublishedSource {std::weak_ptr<const tq::RamKvImage> image;std::shared_ptr<const tq::RamKvBacking> backing;};
    std::vector<PublishedSource> published_sources;
    std::mutex queue_mutex;std::vector<tq::RamKvChange> queued;
    void queue(const std::vector<tq::RamKvChange>& changes) {
        std::lock_guard<std::mutex> lock(queue_mutex);const auto needed=queued.size()+changes.size();
        if(needed>queued.capacity())queued.reserve(std::max(needed,queued.capacity()+queued.capacity()/2));
        queued.insert(queued.end(),changes.begin(),changes.end());
    }
    using Pending=RamRetirement;
    Impl(int c,int r,int n,int b,const tb::TensorOptions& o,const tb::Tensor& logical,bool ssd,const std::string& directory,
            Encode e,bool adaptive,bool prefer_host_metadata,bool spare_vram)
        :capacity(c),rows(r),count(n),bits(b),fp16_bytes(std::size_t(rows)*512),affine_bytes(((3*rows+3)&~3)+std::size_t(rows)*256),
         adaptive_precision(adaptive),host_metadata(qsa_host_metadata_enabled(prefer_host_metadata)),options(o),encode(std::move(e)),live(logical) {
        if(c<=0 || r<=0 || n<=0 || (b!=16 && b!=8 && b!=4))throw std::invalid_argument("Invalid QSA RAM cache geometry");
        ssd_enabled=ssd;account->can_spill=ssd;account->fixed_ram_only=!ssd && !adaptive;
        profile_staging=setting("TYLOINFER_PROFILE_QSA_STAGING")!=0;
        direct_mapped_staging=setting("TYLOINFER_QSA_RAM_DIRECT_ATTENTION")!=0;
        check(cudaGetDevice(&device));
        int mapped=0;check(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,device));mapped_capability=mapped!=0;
        worker=worker_for(device,mfq_current_cuda_stream());
        if(!worker->pinned_pool) {
            worker->pinned_pool=std::make_shared<RamPinnedPool>();worker->pinned_pool->options=options;
            worker->pinned_pool->account=worker->pool_account;
            worker->pinned_pool->device=device;worker->pinned_pool->stream=worker->stream;
            auto fallback=std::make_shared<RamPinnedPool>();fallback->pageable=true;
            fallback->options=options;fallback->account=worker->pool_account;
            fallback->device=device;fallback->stream=worker->stream;
            worker->pinned_pool->pageable_pool=std::move(fallback);
        }
        pool=worker->pinned_pool;
        {std::lock_guard<std::mutex> lock(worker->mutex);worker->accounts.push_back(account);}
        tq::RamKvStoreConfig config;config.pressure.canonical_tier=b==16?tq::RamKvTier::Fp16:b==8?tq::RamKvTier::Affine8:tq::RamKvTier::Tq4;
        config.pressure.ssd_enabled=ssd;config.pressure.adaptive_precision=adaptive_precision;
        if(ssd) {
            mfq::cache::QsaKvStoreConfig disk;disk.directory=directory;disk.buffer_bytes=disk.budget_bytes=1ULL<<20;
            config.session_ssd=std::make_shared<mfq::cache::QsaKvStore>(disk);
        }
        store=std::make_unique<tq::RamKvStore>(config,[this](const auto& image){return pool->allocate(image,nullptr,mapped_capability);},
            [this](const auto& images,auto tier){return convert(images,tier);},
            [this](const auto& changes){queue(changes);},
            [this](const auto& image){
                auto result=pool->allocate_view(image,nullptr,mapped_capability);
                static thread_local bool reported=false;
                if(!reported && std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
                    reported=true;std::fprintf(stderr,
                        "native_qsa_ram_admit_view actual_match=1 temporary_kv_vectors=0\n");
                }
                return result;
            });
        if(live.defined())sources=qsa_descriptor_table(capacity,sizeof(tq::RamBlockView),options,host_metadata);
        const auto tq4=tq4_bytes(rows);
        cache_stride=((b==16?2*fp16_bytes:b==8?2*affine_bytes:2*tq4)+31)&~std::size_t(15);
        spill_stride=(2*tq4+31)&~std::size_t(15);
        const auto* private_cache=std::getenv("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES");
        const auto cache_bytes=setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES");
        const auto sets=std::min<std::size_t>(INT_MAX/tq::ram_cache_ways,cache_bytes/cache_stride/tq::ram_cache_ways);
        if(sets) {
            const auto payload_bytes=sets*tq::ram_cache_ways*cache_stride;
            // RAM-only caches also acquire compressed tiers under pressure.
            // Reserve their metadata capacity without enlarging the payload.
            const auto max_sets=std::min<std::size_t>(INT_MAX/tq::ram_cache_ways,
                payload_bytes/std::min(cache_stride,spill_stride)/tq::ram_cache_ways);
            cache_slots=tb::empty({int64_t(max_sets*tq::ram_cache_ways*sizeof(tq::RamCacheSlot))},options.dtype(tb::kUInt8));
            cache_locks=tb::empty({int64_t(max_sets)},options.dtype(tb::kInt32));cache_frame=tb::empty({8},options.dtype(tb::kUInt8));
            cache_payload=tb::empty({int64_t(payload_bytes)},options.dtype(tb::kUInt8));
            gpu_cache={cache_slots.data_ptr<tq::RamCacheSlot>(),cache_locks.data_ptr<std::uint32_t>(),cache_frame.data_ptr<std::uint64_t>(),
                cache_payload.data_ptr<std::uint8_t>(),std::size_t(cache_payload.numel()),cache_stride,int(sets)};
            check(tq::tq_reset_ram_stage_cache(gpu_cache,mfq_current_cuda_stream()));
        }
        if(!private_cache || !*private_cache) {
            shared_gpu_cache=true;
            const auto* shared_limit=std::getenv("TYLOINFER_QSA_RAM_SHARED_VRAM_CACHE_BYTES");
            use_spare_vram=spare_vram && (!shared_limit || !*shared_limit);
            cache_window_frames=tb::zeros({2},options.dtype(tb::kInt64));
        }
        const auto parameter_options=options;
        cache_parameters=tb::empty({int64_t(sizeof(tq::RamStageCache))},parameter_options.dtype(tb::kUInt8));
        host_cache_parameters=tb::empty({int64_t(sizeof(tq::RamStageCache))},parameter_options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
        upload_cache_parameters();
        worker->pool_account->physical.fetch_add(host_cache_parameters.nbytes());
    }
    std::shared_ptr<RamStageProbe> staging_probe(int queries,int selected_count) {
        auto* window=DecodeWindow::recording();
        // This diagnostic describes the cache-free RAM copy path only. Never
        // turn an SSD graph task into a host boundary or change its leases.
        if(!profile_staging || !window || !window->layer_profiling() || ssd_enabled || gpu_cache.sets)return {};
        static const char batch_key=0;
        auto batch=window->workspace<RamStageProbeBatch>(&batch_key,[]{return std::make_shared<RamStageProbeBatch>();});
        auto result=window->workspace<RamStageProbe>(&stage_probe_key,[&] {
            auto probe=std::make_shared<RamStageProbe>();
            probe->account=worker->pool_account;probe->total=queries*selected_count;
            probe->layer=window->profile_layer();probe->tile=window->profile_tile();
            const auto bytes=int64_t(probe->total)*int64_t(sizeof(tq::RamSelectedRecord));
            probe->records=tb::empty({bytes},options.dtype(tb::kUInt8));
            probe->host_records=tb::empty({bytes},options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
            probe->copied=tb::empty({8},options.dtype(tb::kUInt8));
            probe->host_copied=tb::empty({8},options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
            probe->host_bytes=probe->host_records.nbytes()+probe->host_copied.nbytes();
            probe->account->physical.fetch_add(probe->host_bytes);batch->stages.push_back(probe);return probe;
        });
        if(result->total!=queries*selected_count || result->layer!=window->profile_layer() || result->tile!=window->profile_tile())
            throw std::logic_error("QSA staging diagnostic geometry changed within a graph");
        window->enroll_once({batch.get(),[]{},[]{},[batch]{batch->finish();},[]{},[]{},[]{}});
        return result;
    }
    void upload_cache_parameters() {
        std::memcpy(host_cache_parameters.data_ptr(),&gpu_cache,sizeof(gpu_cache));
        check(cudaMemcpyAsync(cache_parameters.data_ptr(),host_cache_parameters.data_ptr(),sizeof(gpu_cache),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
    }
    std::size_t staging_stride() const {
        if(prefill_staging && account->fixed_ram_only && bits==8 && !gpu_cache.sets &&
                !pool->has_pageable() && !store->stats().tokens[0])
            return RamPinnedPool::image_stride(affine_bytes,affine_bytes);
        return 2*fp16_bytes;
    }
    void configure_cache(bool has_ssd,bool allocate_shared_cache=true,bool graph_ready=false) {
        if(shared_gpu_cache) {
            bool changed=false;
            // A fresh prefill does not need a decode cache. Initialize the
            // optional shared arena when a decode path first recalls KV.
            if(!gpu_cache.owner && !prefill_staging &&
                (worker->stage_cache || (allocate_shared_cache && (!use_spare_vram || graph_ready)))) {
                if(!worker->stage_cache) {
                    const auto context=default_context(device);
                    const auto available=context->available_memory();
                    const auto wanted=use_spare_vram?available:
                        setting("TYLOINFER_QSA_RAM_SHARED_VRAM_CACHE_BYTES",qsa_shared_vram_cache_default);
                    auto bytes=std::min(wanted,available);
                    while(!worker->stage_cache) {
                        try {
                            auto cache=std::make_shared<SharedRamStageCache>(bytes,spill_stride,options,use_spare_vram);
                            if(cache->bytes && QsaRamDecodeBudgetScope::defer_shared_cache(device) && context->memory_stats().limit) {
                                check(cudaStreamSynchronize(mfq_current_cuda_stream()));
                                context->check_memory_limit();
                            }
                            worker->stage_cache=std::move(cache);
                        } catch(const CudaMemoryLimitExceeded& failure) {
                            const auto set_bytes=tq::ram_cache_ways*(spill_stride+sizeof(tq::RamCacheSlot))+sizeof(std::uint32_t);
                            bytes-=std::min(bytes,std::max(failure.shortfall(),set_bytes));
                            context->reclaim_idle_memory();
                        } catch(const MfqResourceExhausted&) {
                            if(!bytes)throw;
                            (void)cudaGetLastError();
                            bytes=use_spare_vram?bytes/2:0;
                            if(use_spare_vram)context->reclaim_idle_memory();
                        }
                    }
                    if(std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
                        std::fprintf(stderr,"native_qsa_shared_vram_cache requested=%zu available=%zu allocated=%zu spare_vram=%d\n",
                            wanted,available,worker->stage_cache->bytes,int(use_spare_vram));
                }
                gpu_cache=worker->stage_cache->acquire();
                gpu_cache.window_frames=cache_window_frames.data_ptr<std::uint64_t>();
                changed=true;
            }
            // A bounded L1 should prioritize sources requiring a CPU packet
            // or SSD read over directly mapped RAM. Their hydrated pass owns
            // admission; ordinary mapped copies can use transient GPU space.
            const bool expensive=pool->has_pageable() || store->stats().tokens[3]!=0;
            if(changed || gpu_cache.expensive_only!=expensive){gpu_cache.expensive_only=expensive;upload_cache_parameters();}
            return;
        }
        if(!gpu_cache.sets)return;
        // All tiers share the payload byte budget. Use the smallest present
        // tier as a base slot; larger images own aligned groups of slots.
        // Ordinary RAM graphs read this layout through the device descriptor.
        const auto usage=store->stats();
        auto stride=cache_stride;
        if(has_ssd || usage.tokens[2] || usage.tokens[3])stride=std::min(stride,spill_stride);
        else if(usage.tokens[1])stride=std::min(stride,(2*affine_bytes+31)&~std::size_t(15));
        if(gpu_cache.slot_bytes==stride)return;
        gpu_cache.slot_bytes=stride;
        gpu_cache.sets=int(std::min<std::size_t>(cache_locks.numel(),gpu_cache.payload_bytes/stride/tq::ram_cache_ways));
        check(tq::tq_reset_ram_stage_cache(gpu_cache,mfq_current_cuda_stream()));
        upload_cache_parameters();++publication_revision;
        if(std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
            "native_qsa_ram_cache_layout actual_match=1 ssd_priority=%d slot_bytes=%zu slots=%d payload_bytes=%zu\n",
            int(has_ssd),stride,gpu_cache.sets*tq::ram_cache_ways,gpu_cache.payload_bytes);
    }
    tq::RamStageCache cache_envelope()const {
        if(shared_gpu_cache)return gpu_cache;
        auto result=gpu_cache;result.slot_bytes=std::min(cache_stride,spill_stride);
        result.sets=int(cache_locks.numel());return result;
    }
    ~Impl() {
        try {wait();}catch(...){}
        (void)cudaStreamSynchronize(mfq_current_cuda_stream());
        try {worker->collect();}catch(...){}
        if(host_cache_parameters.defined())worker->pool_account->physical.fetch_sub(host_cache_parameters.nbytes());
    }
    std::vector<tq::RamKvImagePtr> convert(const std::vector<tq::RamKvImagePtr>& input,tq::RamKvTier tier) {
        check(cudaSetDevice(device));std::vector<tq::RamKvImagePtr> result(input.size());
        if(tier==tq::RamKvTier::Affine8) {
            std::size_t remaining=input.size();
            const auto running=worker->running_job.load();
            const bool urgent=foreground_codec.load() || (running && running<=worker->urgent_through.load());
            const auto cpus=std::max(1u,std::thread::hardware_concurrency());
            const int threads=int(std::min<std::size_t>(input.size(),urgent?cpus:std::max(1u,cpus/2)));
            if(std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
                static thread_local unsigned reported=0;const unsigned kind=urgent?1u:2u;
                if(!(reported&kind)){reported|=kind;
                    std::fprintf(stderr,"native_qsa_ram_affine actual_match=1 threads=%d urgent=%d shared_pinned_pool=1\n",threads,int(urgent));}
            }
            worker->cpu_codecs.run(0,std::int64_t(input.size()),1,threads,[&](auto first,auto last) {
                check(cudaSetDevice(device));
                for(auto i=first;i<last;++i) {
                    const auto& source=*input[i];
                    tq::MicroblockImage image;
                    if(source.key_bits==0 && source.value_bits==0 && source.key_state_bits==16 && source.value_state_bits==16) {
                        image.key=tq::encode_cpu_qsa_affine8(source.key,source.key_bytes,rows);
                        image.value=tq::encode_cpu_qsa_affine8(source.value,source.value_bytes,rows);
                    } else image=tq::CpuTqBlockFitter(copy(source),rows,1).image({8,8});
                    result[i]=pool->allocate(image,&remaining,mapped_capability);
                }
            });
        }else {
            std::vector<tq::MicroblockImage> source;source.reserve(input.size());for(const auto& image:input)source.push_back(copy(*image));
            const auto images=encode(source,std::vector<tq::Precision>(input.size(),{4,4}));
            if(images.size()!=input.size())throw std::runtime_error("QSA RAM GPU codec changed block count");
            result=pool->allocate_batch(images,mapped_capability);
        }
        return result;
    }
    std::size_t target(std::size_t bytes)const {
        const auto cap=setting("TYLOINFER_QSA_RAM_BUDGET_BYTES");
        return cap?std::min(bytes,cap):bytes;
    }
    std::size_t budget(bool rebalance=true)const {
        const auto available=worker->budget(account.get(),rebalance),cap=setting("TYLOINFER_QSA_RAM_BUDGET_BYTES");
        return cap?std::min(cap,available):available;
    }
    std::size_t maintenance_budget(std::size_t limit)const {
        if(!ssd_enabled)return limit;
        // Leave one encoding batch for appends and hot recalls. Bound the
        // margin for small caches so pressure does not retrigger per token.
        const auto blocks=std::size_t(tq::RamPressureConfig{}.quantize_tokens/4);
        const auto margin=std::min(limit/32,2*fp16_bytes*blocks);
        return limit-margin;
    }
    std::size_t restore_append_blocks(std::size_t incoming)const {
        return incoming && incoming<std::size_t(capacity)?std::min(std::size_t(capacity)-incoming,
            std::size_t(tq::RamPressureConfig{}.quantize_tokens/4)):0;
    }
    struct GraphRead {
        std::size_t id;std::uint64_t generation;tq::RamKvImage image;
        tq::RamKvImagePtr resident;
        bool existing_ram=false;
    };
    struct GraphTask {
        Impl* owner;
        std::shared_ptr<StageArena> arena;
        std::shared_ptr<RamAccount> account;
        tb::Tensor flags;
        std::uint32_t* mapped=nullptr;
        int queries,count;
        bool profile=false;
        std::uint64_t calls=0,device_hits=0,wait_ns=0,service_ns=0,read_ns=0,copied_bytes=0,missing_images=0;
        std::uint64_t history_wait_ns=0,admission_ns=0;
        std::vector<GraphRead> reads;
        std::vector<tq::RamKvSnapshotBlock> incoming;
        std::vector<tq::RamKvImagePtr> locked;
        std::vector<tq::RamKvReadInto> load_requests;
        std::future<void> history_job;
        std::uint64_t history_sequence=0;
        std::shared_ptr<std::vector<std::uint8_t>> fallback=std::make_shared<std::vector<std::uint8_t>>();
        std::size_t fallback_charge=0;
        GraphTask(Impl& value,int q,int n,bool trace=false):owner(&value),arena(value.arena),account(value.worker->pool_account),queries(q),count(n),profile(trace) {
            flags=tb::zeros({16},value.options.device(tb::kCPU).dtype(tb::kInt32).pinned_memory(true));
            void* pointer=nullptr;check(cudaHostGetDevicePointer(&pointer,flags.data_ptr(),0));mapped=static_cast<std::uint32_t*>(pointer);
            account->physical.fetch_add(flags.nbytes());
        }
        ~GraphTask(){clear();account->physical.fetch_sub(flags.nbytes()+fallback_charge);
            if(profile && calls)std::fprintf(stderr,
                "native_qsa_host_service calls=%llu device_hits=%llu wait_ns=%llu service_ns=%llu read_ns=%llu copied_bytes=%llu missing_images=%llu history_wait_ns=%llu admission_ns=%llu\n",
                static_cast<unsigned long long>(calls),static_cast<unsigned long long>(device_hits),static_cast<unsigned long long>(wait_ns),
                static_cast<unsigned long long>(service_ns),static_cast<unsigned long long>(read_ns),
                static_cast<unsigned long long>(copied_bytes),static_cast<unsigned long long>(missing_images),
                static_cast<unsigned long long>(history_wait_ns),static_cast<unsigned long long>(admission_ns));
        }
        std::uint32_t* host(){return flags.data_ptr<std::uint32_t>();}
        void join_history(){if(history_job.valid())history_job.get();}
        void clear()noexcept {
            try{join_history();}catch(...){}
            reads.clear();load_requests.clear();incoming.clear();locked.clear();
        }
        void reset(){clear();std::memset(flags.data_ptr(),0,flags.nbytes());std::atomic_thread_fence(std::memory_order_seq_cst);}
        void reset_for_replay() {
            // The captured kernels load the live cache layout. Size the
            // optional cache after the graph has claimed its required VRAM.
            if(owner->shared_gpu_cache && !owner->gpu_cache.owner)owner->configure_cache(true,true,true);
            reset();
        }
        void cancel()noexcept {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *reinterpret_cast<volatile std::uint32_t*>(host()+tq::RamGraphReady)=2;
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        void serve() {
            using Clock=std::chrono::steady_clock;
            auto point=profile?Clock::now():Clock::time_point{};
            wait_route_publication(host()+tq::RamGraphRequest,mfq_current_cuda_stream());
            if(profile){++calls;const auto now=Clock::now();wait_ns+=std::uint64_t(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now-point).count());point=now;}
            // A device-completed selection can already have advanced the
            // shared staging arena. Its private control word is the only
            // host data this task may read in that case.
            if(host()[tq::RamGraphRequest]==tq::RamGraphDeviceReady){if(profile)++device_hits;return;}
            auto& s=*owner;auto& a=*arena;auto& disk=*a.ssd;
            const auto* missing=a.host_missing.data_ptr<std::int32_t>();
            if(missing[0]<0 || missing[0]>queries*count)throw std::runtime_error("Invalid graph QSA SSD miss count");
            const auto stored=s.store->stats().blocks;
            std::memcpy(a.served_selected.data(),a.host_selected.data_ptr(),std::size_t(queries)*count*sizeof(tq::SelectedBlock));
            auto* records=disk.images.data_ptr<tq::RamBlockView>();
            auto* fetched=disk.ids.data_ptr<std::int32_t>();
            reads.reserve(missing[0]);
            if(profile)missing_images+=std::uint64_t(missing[0]);
            load_requests.reserve(missing[0]);
            incoming.reserve(missing[0]);locked.reserve(missing[0]);std::size_t required=0;
            for(int at=0;at<missing[0];++at) {
                const auto id=std::size_t(missing[at+1]);
                if(id>=stored)throw std::runtime_error("Graph QSA SSD selection has no archived source");
                const auto& published=s.published_sources.at(id);
                tq::RamKvSnapshotBlock source;source.image=published.image.lock();source.backing=published.backing;
                source.generation=s.host_sources.at(id).generation;
                if(!source.image && !source.backing)throw std::runtime_error("Published QSA source lost its immutable owner");
                source.tier=source.image?(source.image->key_bits==0 && source.image->value_bits==0?tq::RamKvTier::Fp16:
                    source.image->key_bits==tq::affine8_tag && source.image->value_bits==tq::affine8_tag?tq::RamKvTier::Affine8:tq::RamKvTier::Tq4):source.backing->tier;
                const auto kb=source.image?source.image->key_bytes:source.backing->key_bytes;
                const auto vb=source.image?source.image->value_bytes:source.backing->value_bytes;
                const auto extent=(kb+15)&~std::size_t(15);
                if(extent>disk.stride || vb>disk.stride-extent || required>SIZE_MAX-extent-vb)
                    throw std::runtime_error("Graph QSA SSD image exceeds its prepared slot");
                auto resident=s.pool->try_reserve(source);
                if(!resident && !source.image)required+=extent+vb;
                locked.push_back(std::move(resident));incoming.push_back(std::move(source));
            }
            // The mapped GPU stage is shared between layers. Its bytes may
            // be overwritten before deferred RAM admission runs. One compact
            // task-owned buffer preserves fallback images through that join.
            if(fallback->size()<required) {
                fallback->resize(required);
                const auto extra=fallback->capacity()-fallback_charge;
                account->physical.fetch_add(extra);fallback_charge+=extra;
            }
            std::size_t fallback_at=0;
            for(int at=0;at<missing[0];++at) {
                const auto id=std::size_t(missing[at+1]);
                if(id>=stored)throw std::runtime_error("Graph QSA SSD selection has no archived source");
                const auto& source=incoming[at];
                const auto& resident=locked[at];tq::RamKvImage image;
                const auto index=reads.size(),offset=index*disk.stride;
                if(resident) {
                    const auto extent=std::size_t(resident->value-resident->key);
                    if(extent>disk.stride || resident->value_bytes>disk.stride-extent)
                        throw std::runtime_error("Graph QSA SSD image exceeds its prepared slot");
                    if(!source.image)load_requests.push_back({&source,const_cast<std::uint8_t*>(resident->key),
                        extent+resident->value_bytes});
                    image=*resident;records[index]=s.record({id,source.generation,resident,{},source.image?source.tier:source.backing->tier});
                }else if(source.image) {
                    image=*source.image;
                    const auto key_extent=std::size_t(image.value-image.key);
                    records[index]={{disk.mapped_payload+offset,disk.mapped_payload+offset+key_extent,
                        tq::tq_precision_flags(image.key_bits,image.key_state_bits),tq::tq_precision_flags(image.value_bits,image.value_state_bits)},
                        std::uint32_t(image.key_bytes),std::uint32_t(image.value_bytes),source.generation};
                }else {
                    const auto kb=source.image?source.image->key_bytes:source.backing->key_bytes;
                    const auto vb=source.image?source.image->value_bytes:source.backing->value_bytes;
                    const auto key_extent=(kb+15)&~std::size_t(15);
                    if(key_extent>disk.stride || vb>disk.stride-key_extent)throw std::runtime_error("Graph QSA SSD image exceeds its prepared slot");
                    auto* retained=fallback->data()+fallback_at;fallback_at+=key_extent+vb;
                    load_requests.push_back({&source,retained,key_extent+vb});
                    image.owner=fallback;image.key=retained;image.value=retained+key_extent;image.key_bytes=kb;image.value_bytes=vb;
                    image.key_bits=source.image?source.image->key_bits:source.backing->key_bits;
                    image.value_bits=source.image?source.image->value_bits:source.backing->value_bits;
                    image.key_state_bits=source.image?source.image->key_state_bits:source.backing->key_state_bits;
                    image.value_state_bits=source.image?source.image->value_state_bits:source.backing->value_state_bits;
                    records[index]={{disk.mapped_payload+offset,disk.mapped_payload+offset+key_extent,
                        tq::tq_precision_flags(image.key_bits,image.key_state_bits),tq::tq_precision_flags(image.value_bits,image.value_state_bits)},
                        std::uint32_t(image.key_bytes),std::uint32_t(image.value_bytes),source.generation};
                }
                fetched[index]=std::int32_t(id);
                reads.push_back({id,source.generation,std::move(image),std::move(resident),bool(source.image)});
            }
            const auto read_begin=profile?Clock::now():Clock::time_point{};
            s.store->read_cpu_into_batch(load_requests,false);
            if(profile)read_ns+=std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-read_begin).count());
            for(std::size_t at=0;at<reads.size();++at)if(!reads[at].resident) {
                const auto& image=reads[at].image;auto* payload=disk.payload.data_ptr<std::uint8_t>()+at*disk.stride;
                const auto extent=std::size_t(image.value-image.key);
                std::memcpy(payload,image.key,image.key_bytes);std::memcpy(payload+extent,image.value,image.value_bytes);
                if(profile)copied_bytes+=image.bytes();
            }
            host()[tq::RamGraphImages]=std::uint32_t(reads.size());
            publish_mapped_flag(host()+tq::RamGraphReady);
            if(profile)service_ns+=std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-point).count());
            if(!reads.empty() && std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
                const auto direct=std::count_if(reads.begin(),reads.end(),[](const auto& read){return bool(read.resident);});
                const unsigned kinds=(direct?1u:0u)|(std::size_t(direct)<reads.size()?2u:0u);
                static thread_local unsigned reported[9]{};
                if(kinds&~reported[queries]) {
                    reported[queries]|=kinds;
                    std::fprintf(stderr,"native_qsa_ram_read_into actual_match=1 queries=%d direct_slots=%zu fallback_views=%zu no_pinned_growth=1\n",
                        queries,std::size_t(direct),reads.size()-std::size_t(direct));
                }
            }
            // Publication queues retain replaced RAM owners until their GPU
            // readers finish. Admission runs only after the whole graph has
            // joined, so access-history updates need not stall this copy.
            RamClaims ids(s.worker->claims,s.capacity,std::size_t(queries)*count);
            const auto* selected=a.served_selected.data();
            for(int at=0;at<queries*count;++at)if(selected[at].id>=0 && std::size_t(selected[at].id)<stored && selected[at].token_mask)
                ids.add(std::size_t(selected[at].id));
            ids.clear();
            const auto epoch=s.graph_epoch;
            // The ready flag releases the GPU before LRU work. Keep that work
            // off the host data-service thread and join it before admission.
            // Pending descriptor publications retain replaced RAM images.
            history_job=s.worker->submit([this,selected=std::move(ids.ids),epoch] {
                owner->store->touch(selected,epoch);
            },history_sequence);
        }
        void finish() {
            using Clock=std::chrono::steady_clock;
            const auto started=profile?Clock::now():Clock::time_point{};
            join_history();
            const auto history_done=profile?Clock::now():Clock::time_point{};
            if(profile)history_wait_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(history_done-started).count();
            if(host()[tq::RamGraphError])throw std::runtime_error("Graph QSA SSD source changed during its read");
            // DecodeWindow joins every GPU reader before calling finish.
            // Published descriptors and selected sources retain every image;
            // the completed history task only updates LRU/access metadata.
            if(reads.empty())return;
            if(std::all_of(reads.begin(),reads.end(),[](const auto& read){return read.existing_ram;})) {
                clear();return;
            }
            auto& s=*owner;
            if(s.prefill_staging) {
                const auto limit=s.budget();
                for(const auto& read:reads) {
                    if(read.existing_ram)continue;
                    if(read.resident)s.store->admit_read(read.id,read.generation,read.resident,limit);
                    else s.store->admit_read(read.id,read.generation,read.image,limit);
                }
            } else if(std::any_of(reads.begin(),reads.end(),[](const auto& read) {
                return !read.existing_ram && read.resident;
            })) {
                // Reuse an existing RAM slot without growing the pinned pool
                // or reclaiming capacity while the next token waits.
                const auto limit=s.budget(false);
                const auto used=s.store->stats().ram_bytes;
                auto free=limit-std::min(limit,used);
                for(const auto& read:reads)if(!read.existing_ram && read.resident && read.image.bytes()<=free) {
                    s.store->admit_read(read.id,read.generation,read.resident,limit,false);
                    free-=read.image.bytes();
                }
            }
            if(!reads.empty() && !s.reported_ssd_fetch && std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
                s.reported_ssd_fetch=true;const auto stats=s.store->stats();
                std::fprintf(stderr,"qsa_ram_ssd_fetch actual_match=1 misses=%zu read_bytes=%llu promotions=%llu graph_task=1 packed_hits_skip_ssd=1\n",
                    reads.size(),static_cast<unsigned long long>(stats.ssd_read_bytes),static_cast<unsigned long long>(stats.promoted_blocks));
            }
            s.update_account();clear();
            if(profile)admission_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-history_done).count();
        }
    };
    void update_account() {
        const auto stats=store->stats();account->logical.store(stats.ram_bytes);
        account->shared_prefix_ram.store(stats.shared_prefix_ram_bytes);
        account->target.store(target(stats.resident_target_bytes));
    }
    std::size_t compacted_physical=0,compacted_logical=0;
    void resolve_capacity() {
        pool->resolve_pinned();
        auto needed=required_capacity.load();if(!needed)return;
        const auto before=budget();auto after=before;std::size_t released=0;
        required_capacity.store(0);
        if(needed>after) {
            // Finish already-submitted finite work before treating leases as
            // unavailable capacity. No graph can wait for this foreground path.
            const auto retire=[&] {
                drain();check(cudaStreamSynchronize(mfq_current_cuda_stream()));drain();
            };
            retire();
            pool->trim(0);needed=store->stats().ram_bytes;after=budget();
            if(needed>after && ssd_enabled && store->evict_backed(maintenance_budget(after)).ram_bytes<needed) {
                update_account();retire();pool->trim(0);needed=store->stats().ram_bytes;after=budget();
            }
            if(needed<=after){sampled_budget=after;budget_sampled=true;return;}
            RamForegroundCodec foreground(foreground_codec);
            // Quantizing images cannot release pages stranded in partial
            // pinned slabs. Pack exact images into existing holes first.
            const auto physical=worker->pool_account->physical.load();
            const auto free=mfq::sample_host_memory().available_host_bytes();
            const auto shortage=std::max(needed-after,tq::ram_kv_reclaim_bytes(free,physical,physical));
            std::size_t logical=0;
            {std::lock_guard<std::mutex> lock(worker->mutex);
                for(const auto& weak:worker->accounts)if(const auto account=weak.lock())logical+=account->charged_ram();}
            const auto quantum=std::max<std::size_t>(4ULL<<20,physical/64);
            const auto changed=[](auto a,auto b){return a>b?a-b:b-a;};
            if(shortage && (!compacted_physical || changed(physical,compacted_physical)>=quantum ||
                    changed(logical,compacted_logical)>=quantum)) {
                // Relocation validates this store's versions. The slab pool
                // protects slot claims and can decline a contested destination,
                // so pending codecs in other layers need not finish first.
                retire();pool->trim(0);needed=store->stats().ram_bytes;after=budget();
                if(needed<=after){sampled_budget=after;budget_sampled=true;return;}
                const auto compact_before=worker->pool_account->physical.load();
                const auto moved=store->relocate([&](const auto& images){return pool->compact(images,shortage);});
                if(moved) {
                    retire();pool->trim(0);needed=store->stats().ram_bytes;after=budget();
                    if(std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
                        "native_qsa_ram_compact actual_match=1 images=%zu physical_before=%zu physical_after=%zu\n",
                        moved,compact_before,worker->pool_account->physical.load());
                }
                compacted_physical=worker->pool_account->physical.load();compacted_logical=logical;
            }
            if(needed<=after) {sampled_budget=after;budget_sampled=true;return;}
            store->maintain(maintenance_budget(after),false);update_account();retire();
            needed=store->stats().ram_bytes;after=budget();
            // Explicit per-layer caps cannot borrow optional expert RAM.
            const auto cap=setting("TYLOINFER_QSA_RAM_BUDGET_BYTES");
            if(needed>after && !ssd_enabled && (!cap || after<cap)) {
                released=worker->reclaim_capacity(target(needed));after=budget();
            }
            if(needed>after) {
                // Spill cold KV before displacing RAM experts. RAM-only
                // configurations instead reclaim capacity or report exhaustion.
                auto result=store->maintain(maintenance_budget(after),ssd_enabled);
                if(ssd_enabled && !result.fits()) {
                    // A partial high-precision batch cannot be requantized.
                    // Preserve its encoding and move cold blocks to SSD.
                    result=store->spill_preserving_precision(maintenance_budget(after));
                }
                update_account();
                if(!result.fits() && !result.needs_readers)throw std::runtime_error(ssd_enabled?
                    "QSA RAM cannot hold the protected high-precision KV tiers; reduce the context":
                    "QSA RAM KV exhausted; enable session SSD KV or reduce the context");
            }
        }
        sampled_budget=after;budget_sampled=true;
        if(released && std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
            "native_qsa_ram_capacity_reclaim actual_match=1 required=%zu before=%zu after=%zu released=%zu foreground=1\n",
            needed,before,after,released);
    }
    void wait(){
        if(maintenance.valid()) {
            if(maintenance.wait_for(std::chrono::seconds(0))!=std::future_status::ready)
                worker->prioritize(maintenance_sequence);
            maintenance.get();
        }
        resolve_capacity();
    }
    tq::RamBlockView record(const tq::RamKvChange& change) {
        auto result=host_sources.at(change.id);result.generation=change.generation;
        const auto& image=change.image;
        if(!image){
            result.image.key=result.image.value=nullptr;
            if(const auto& source=change.backing) {
                result.image.key_bits=tq::tq_precision_flags(source->key_bits,source->key_state_bits);
                result.image.value_bits=tq::tq_precision_flags(source->value_bits,source->value_state_bits);
                result.key_bytes=std::uint32_t(source->key_bytes);result.value_bytes=std::uint32_t(source->value_bytes);
            }
            return result;
        }
        const auto allocation=std::static_pointer_cast<const RamPinnedPool::Allocation>(image->owner);
        const auto offset=std::uintptr_t(image->key)-std::uintptr_t(allocation->slab->host.data_ptr());
        const auto mapped=allocation->slab->mapped;
        result.image={mapped?mapped+offset:nullptr,mapped?mapped+offset+(image->value-image->key):nullptr,
            tq::tq_precision_flags(image->key_bits,image->key_state_bits),tq::tq_precision_flags(image->value_bits,image->value_state_bits)};
        result.key_bytes=std::uint32_t(image->key_bytes);result.value_bytes=std::uint32_t(image->value_bytes);return result;
    }
    void drain(tq::RamBlockView* host_table=nullptr,tq::TqBlockView* host_live=nullptr) {
        if(maintenance.valid() && maintenance.wait_for(std::chrono::seconds(0))==std::future_status::ready)maintenance.get();
        worker->collect();
        Pending p;
        {std::lock_guard<std::mutex> lock(queue_mutex);p.owners.swap(queued);}
        if(p.owners.empty())return;
        const auto last=std::max_element(p.owners.begin(),p.owners.end(),[](const auto& a,const auto& b){return a.id<b.id;})->id;
        if(last>=std::size_t(capacity))throw std::out_of_range("QSA RAM publication exceeds capacity");
        if(host_sources.size()<=last)host_sources.resize(last+1);
        if(published_sources.size()<=last)published_sources.resize(last+1);
        // Several precision transitions can replace an ID. Retain only its
        // final descriptor, without allocating a tree node for every block.
        RamClaims latest(worker->claims,capacity,p.owners.size());
        for(const auto& change:p.owners){
            if(change.id<prefetched_blocks && host_sources[change.id].generation!=change.generation)
                prefetched_blocks=change.id;
            auto rec=record(change);host_sources[change.id]=rec;
            published_sources[change.id]={change.image,change.backing};latest.add(change.id);}
        latest.clear();const auto n=latest.ids.size();
        if(host_table) {
            // Admission runs after the last reader. Publish straight into
            // mapped metadata when a locked transfer packet cannot be kept.
            for(auto id:latest.ids){host_table[id]=host_sources[id];host_live[id]=host_sources[id].image;}
            ++publication_revision;
            update_account();
            return;
        }
        auto publication=worker->publication(n,options,mapped_capability);publication.owners=std::move(p.owners);p=std::move(publication);
        auto* ids=p.host_ids.data_ptr<std::int32_t>();auto* descriptors=p.host_descriptors.data_ptr<tq::RamBlockView>();
        std::size_t at=0;for(auto id:latest.ids){ids[at]=std::int32_t(id);descriptors[at++]=host_sources[id];}
        const auto* ids_source=p.mapped_ids;const auto* descriptors_source=p.mapped_descriptors;
        if(!ids_source) {
            check(cudaMemcpyAsync(p.ids.data_ptr(),p.host_ids.data_ptr(),n*sizeof(std::int32_t),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            check(cudaMemcpyAsync(p.descriptors.data_ptr(),p.host_descriptors.data_ptr(),n*sizeof(tq::RamBlockView),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            ids_source=p.ids.data_ptr<std::int32_t>();descriptors_source=p.descriptors.data_ptr<tq::RamBlockView>();
        }
        // Consume these small immutable records directly on devices that can
        // map pinned host memory. The retirement owns both the records and
        // replaced images until the stream has passed their publication.
        check(tq::tq_publish_ram_sources(sources.data_ptr<tq::RamBlockView>(),capacity,ids_source,descriptors_source,int(n),mfq_current_cuda_stream(),live.data_ptr<tq::TqBlockView>()));
        ++publication_revision;
        const bool reused=p.done!=nullptr;
        if(!p.done)check(cudaEventCreateWithFlags(&p.done,cudaEventDisableTiming));
        check(cudaEventRecord(p.done,mfq_current_cuda_stream()));worker->retire(std::move(p));
        if(std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
            static thread_local unsigned reported=0;const unsigned mask=(mapped_capability?1u:2u)|(reused?4u:8u);
            if((reported&mask)!=mask){reported|=mask;std::fprintf(stderr,
                "native_qsa_ram_publish actual_match=1 mapped_descriptors=%d h2d_copies=%d event_reused=%d\n",
                int(mapped_capability),mapped_capability?0:2,int(reused));}
        }
        update_account();
    }
};
QsaRamCache::QsaRamCache(int c,int r,int n,int bits,const tb::TensorOptions& options,const Tensor& live,bool ssd,const std::string& directory,
        Encode encode,bool adaptive_precision,bool prefer_host_metadata,bool use_spare_vram)
    :impl_(std::make_unique<Impl>(c,r,n,bits,options,live,ssd,directory,std::move(encode),adaptive_precision,prefer_host_metadata,
        use_spare_vram)){
    std::lock_guard lock(impl_->account->owner_mutex);impl_->account->owner=this;
}
void RamWorker::rebalance_restore() {
    std::vector<std::shared_ptr<RamAccount>> active;
    std::vector<const RamAccount*> identities;
    {
        std::lock_guard lock(mutex);
        for(const auto& weak:accounts)if(auto account=weak.lock())
            if(account->logical.load() || account->target.load() || account->restore_target.load()) {
                identities.push_back(account.get());active.push_back(std::move(account));
            }
    }
    if(identities==restore_participants)return;
    restore_participants=std::move(identities);
    // A new request changes every live layer's share. Start their optional
    // compression while its prefix is being read, rather than on its first
    // generated tokens. Admission and descriptor publication remain serialized.
    for(const auto& account:active)if(account->logical.load()) {
        std::lock_guard lock(account->owner_mutex);
        if(account->owner)account->owner->observe({},0);
    }
}
QsaRamCache::~QsaRamCache(){
    const auto account=impl_->account;
    std::lock_guard lock(account->owner_mutex);account->owner=nullptr;
    impl_.reset();account->logical.store(0);account->target.store(0);account->restore_target.store(0);
}
std::shared_ptr<void> QsaRamCache::hold_workspace(std::size_t bytes) {
    return hold_worker_workspace(impl_->worker,bytes);
}
std::size_t QsaRamCache::reclaim_pinned_for_admission(std::size_t bytes) {
    if(!bytes || !impl_->ssd_enabled || !impl_->host_metadata || !impl_->sources.defined())return 0;
    RamPinnedForeground pinned_foreground;
    auto& s=*impl_;
    cudaPointerAttributes sources{},live{};
    check(cudaPointerGetAttributes(&sources,s.sources.data_ptr()));
    check(cudaPointerGetAttributes(&live,s.live.data_ptr()));
    if(!sources.hostPointer || !live.hostPointer)return 0;
    const auto before=s.pool->own_bytes.load();
    const auto released=[&] {return before-std::min(before,s.pool->own_bytes.load());};
    if(s.maintenance.valid()) {
        s.worker->prioritize(s.maintenance_sequence);
        s.maintenance.get();
    }
    check(cudaStreamSynchronize(mfq_current_cuda_stream()));
    const auto publish=[&] {
        s.drain(static_cast<tq::RamBlockView*>(sources.hostPointer),static_cast<tq::TqBlockView*>(live.hostPointer));
        s.pool->trim(0);
    };
    publish();
    if(released()<bytes) {
        const auto resident=s.store->stats().ram_bytes;
        s.store->evict_backed(resident-std::min(resident,bytes-released()));
        publish();
    }
    return released();
}
QsaRamWorkspaceProgress QsaRamCache::reclaim_shared_workspace() {
    RamPinnedForeground pinned_foreground;
    const auto worker=impl_->worker;
    std::vector<std::shared_ptr<RamAccount>> accounts;
    {
        std::lock_guard lock(worker->mutex);
        for(const auto& weak:worker->accounts)if(auto account=weak.lock())accounts.push_back(std::move(account));
    }
    // The predictor can share this pool without appearing in the target's
    // layer list. Refresh and maintain every live participant of the budget.
    std::size_t accounted_before=0,actual_before=0;
    for(const auto& account:accounts) {
        std::lock_guard lock(account->owner_mutex);
        if(account->owner) {
            accounted_before+=account->logical.load();account->owner->impl_->update_account();
            actual_before+=account->logical.load();
        }
    }
    for(const auto& account:accounts) {
        std::lock_guard lock(account->owner_mutex);
        if(account->owner)account->owner->wait();
    }
    check(cudaStreamSynchronize(mfq_current_cuda_stream()));
    QsaRamWorkspaceProgress progress;
    for(const auto& account:accounts) {
        std::lock_guard lock(account->owner_mutex);
        if(!account->owner)continue;
        account->owner->drain();account->owner->impl_->update_account();
        progress.logical+=account->logical.load();
    }
    progress.physical=worker->pool_account->physical.load();
    if(std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
        "native_qsa_shared_workspace actual_match=1 participants=%zu workspace_bytes=%zu accounted_before=%zu actual_before=%zu logical_bytes=%zu physical_bytes=%zu free_bytes=%llu\n",
        accounts.size(),worker->workspace_bytes.load(),accounted_before,actual_before,progress.logical,progress.physical,
        static_cast<unsigned long long>(mfq::sample_host_memory().available_host_bytes()));
    return progress;
}
void QsaRamCache::prepare(int queries,bool exact_width,bool allocate_shared_cache) {
    if(queries<1 || queries>8)throw std::invalid_argument("Invalid QSA staging query count");
    RamPinnedForeground pinned_foreground;
    auto& s=*impl_;
    const auto stride=s.staging_stride();
    if(!s.arena || s.arena->stride!=stride || s.arena->queries<queries || (exact_width && s.arena->queries!=queries))
        s.arena=arena_for(s.device,mfq_current_cuda_stream(),s.capacity,s.count,queries,stride,
            s.options,s.worker->pool_account,s.prefill_staging);
    s.drain();s.configure_cache(s.store->stats().tokens[3]!=0 || s.pool->has_pageable(),allocate_shared_cache);
    s.prepared_revision=s.publication_revision;
}
void QsaRamCache::activate_shared_cache() {
    auto& s=*impl_;
    if(s.shared_gpu_cache && !s.gpu_cache.owner && !s.direct_mapped_staging && !s.profile_staging)
        s.configure_cache(s.store->stats().tokens[3]!=0 || s.pool->has_pageable(),true,true);
}
std::size_t QsaRamCache::release_shared_device_cache(bool spare_only) {
    const auto& worker=impl_->worker;
    if(!worker->stage_cache || (spare_only && !worker->stage_cache->spare_vram))return 0;
    const MfqCudaGuard device(impl_->device);
    check(cudaDeviceSynchronize());
    std::vector<std::shared_ptr<RamAccount>> accounts;
    {
        std::lock_guard lock(worker->mutex);
        for(const auto& weak:worker->accounts)if(auto account=weak.lock())accounts.push_back(std::move(account));
    }
    // Cache entries are transient replicas. Clear each device descriptor
    // before returning their shared backing to the allocator.
    for(const auto& account:accounts) {
        std::lock_guard lock(account->owner_mutex);
        if(!account->owner)continue;
        auto& state=*account->owner->impl_;
        if(!state.shared_gpu_cache || !state.gpu_cache.owner)continue;
        state.gpu_cache={};
        state.upload_cache_parameters();
    }
    check(cudaStreamSynchronize(mfq_current_cuda_stream()));
    const auto bytes=worker->stage_cache->bytes;
    worker->stage_cache.reset();
    if(std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
        std::fprintf(stderr,"native_qsa_shared_vram_cache_released bytes=%zu\n",bytes);
    return bytes;
}
void QsaRamCache::begin_prefill_stage() {
    if(!impl_->prefill_staging)impl_->arena.reset();
    impl_->prefill_staging=true;
}
void QsaRamCache::prepare_prefill_stage(int queries,std::uint64_t epoch) {
    begin_prefill_stage();
    prepare(queries,true);
    const auto& s=*impl_;
    if(s.mapped_capability && (s.pool->has_pageable() || s.store->stats().tokens[3]))prepare_graph_task(epoch);
}
std::size_t QsaRamCache::fixed_prefill_ram_reservation(std::size_t target_blocks) const {
    const auto& s=*impl_;
    if(s.ssd_enabled || s.adaptive_precision)return 0;
    const auto stats=s.store->stats();
    const auto part=s.bits==16?s.fp16_bytes:s.bits==8?s.affine_bytes:tq4_bytes(s.rows);
    // Resident fixed KV is already charged in the shared pool and the host
    // memory sample. Admit only the new images still needed by this request.
    return (target_blocks>stats.blocks?target_blocks-stats.blocks:0)*RamPinnedPool::image_stride(part,part);
}
void QsaRamCache::release_prefill_stage() {
    impl_->prefill_staging=false;
    impl_->arena.reset();
    impl_->prefetched_blocks=0;
}
void QsaRamCache::trim_stage_queries(int queries) {
    auto& arena=impl_->arena;
    if(arena && arena->queries>queries)arena.reset();
}
QsaRamCache::Tensor QsaRamCache::prepared_views()const {
    if(!impl_->arena)throw std::logic_error("QSA RAM staging arena was not prepared");
    if(auto* window=DecodeWindow::recording())
        window->workspace<StageArena>(impl_->arena.get(),[&]{return impl_->arena;});
    return impl_->arena->views;
}
std::optional<QsaRamPrefillStage> QsaRamCache::stage_prefill(std::size_t blocks,std::size_t reserve_blocks,
        const QsaRamPrefillStage* previous) {
    RamPinnedForeground pinned_foreground;
    auto& s=*impl_;if(!blocks)return {};
    const auto first_block=previous?previous->blocks:0;
    if(first_block>blocks)throw std::logic_error("QSA prefill staging exceeds live history");
    const auto stream=mfq_current_cuda_stream();cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("Layer KV staging requires uncaptured prefill");
    constexpr auto maximum_io_records=(1ULL<<20)/sizeof(tq::RamKvReadInto);
    const auto io_bytes=tq::RamKvStore::sequential_read_workspace(std::min(blocks-first_block,std::size_t(maximum_io_records)));
    const auto io_reservation=hold_worker_workspace(s.worker,io_bytes);
    if(mfq::sample_host_memory().available_host_bytes()<io_bytes)
        prepare_qsa_ram_workspace(io_bytes,[this]{return reclaim_shared_workspace();});
    // Re-evaluate physical pressure before taking leases that prevent KV
    // reclamation. Waiting only for an earlier codec plan uses a stale budget.
    wait();s.drain();
    if(first_block>s.prefetched_blocks)return {};
    const auto snapshot=s.store->snapshot(blocks-first_block,first_block);
    const auto previous_bytes=previous?previous->used_bytes:0;
    std::size_t bytes=previous_bytes,largest=0;
    const auto image_bytes=[](const tq::RamKvSnapshotBlock& source) {
        const auto kb=source.image?source.image->key_bytes:source.backing->key_bytes;
        const auto vb=source.image?source.image->value_bytes:source.backing->value_bytes;
        if(kb>SIZE_MAX-15)throw std::overflow_error("Prefill KV image overflows");
        const auto extent=(kb+15)&~std::size_t(15);
        if(extent>SIZE_MAX-15 || vb>SIZE_MAX-15-extent)throw std::overflow_error("Prefill KV image overflows");
        return (extent+vb+15)&~std::size_t(15);
    };
    for(const auto& source:snapshot->blocks) {
        if(!source.image && !source.backing)throw std::runtime_error("Prefill KV source is absent");
        const auto count=image_bytes(source);
        if(count>SIZE_MAX-bytes)throw std::overflow_error("Prefill KV layer overflows");
        bytes+=count;largest=std::max(largest,count);
    }
    // Only complete blocks and the possible live tail are addressed by this
    // prefill tile. The maximum-context descriptor table can be much larger.
    const auto live_rows=std::min(blocks+1,std::size_t(s.capacity));
    QsaRamPrefillStage result;
    if(previous) {
        if(bytes>previous->payload.nbytes() || live_rows>std::size_t(previous->views.size(0)))return {};
        result=*previous;
        if(live_rows>blocks)result.views.narrow(0,blocks,1).copy_(s.live.narrow(0,blocks,1));
    } else {
        auto payload_bytes=bytes;
        auto table_rows=live_rows;
        const auto available=available_prefill_workspace(*default_context(s.device));
        const auto table_bytes=table_rows*sizeof(tq::TqBlockView);
        if(table_bytes>available || bytes>available-table_bytes)return {};
        if(reserve_blocks>blocks) {
            const auto canonical=s.bits==16?s.fp16_bytes:s.bits==8?s.affine_bytes:tq4_bytes(s.rows);
            const auto extent=std::max(largest,RamPinnedPool::image_stride(canonical,canonical));
            const auto added=std::min(std::min(reserve_blocks,std::size_t(s.capacity))-blocks,
                (available-table_bytes-bytes)/(extent+sizeof(tq::TqBlockView)));
            payload_bytes+=added*extent;
            table_rows=std::min(live_rows+added,std::size_t(s.capacity));
        }
        result.payload=tb::empty({int64_t(payload_bytes)},s.options.dtype(tb::kUInt8));
        result.views=tb::empty({int64_t(table_rows),s.live.size(1)},s.live.options());
        if(live_rows>blocks)result.views.narrow(0,blocks,1).copy_(s.live.narrow(0,blocks,1));
    }
    result.blocks=blocks;result.used_bytes=bytes;
    if(first_block==blocks)return result;
    constexpr auto maximum_records=(1ULL<<20)/sizeof(tq::RamKvReadInto);
    const auto packet_bytes=std::max(largest,std::size_t(8ULL<<20));
    auto& retained=s.worker->prefill_transfer;
    if(!retained || retained->packet_bytes<packet_bytes) {
        if(retained)for(auto event:retained->done)if(event)check(cudaEventSynchronize(event));
        auto next=std::make_shared<PrefillTransfer>();next->account=s.worker->pool_account;
        next->packet_bytes=packet_bytes;
        const auto cpu=s.options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true);
        for(int slot=0;slot<2;++slot) {
            next->packets[slot]=tb::empty({int64_t(packet_bytes)},cpu);
            next->records[slot]=tb::empty({int64_t(maximum_records*sizeof(tq::TqBlockView))},cpu);
        }
        for(int slot=0;slot<2;++slot)next->charge+=next->packets[slot].nbytes()+next->records[slot].nbytes();
        next->account->physical.fetch_add(next->charge);retained=std::move(next);
    }
    // All layers on this model stream share the ring. Each slot's event
    // protects both its encoded packet and its bounded descriptor packet.
    const auto host=retained;
    result.payload.record_stream(reinterpret_cast<std::uintptr_t>(stream));
    result.views.record_stream(reinterpret_cast<std::uintptr_t>(stream));
    constexpr int ring=2;
    std::size_t first=first_block,offset=previous_bytes,turn=0;bool submitted=false;
    try {
        while(first<blocks) {
            const auto slot=turn++%ring;
            if(host->done[slot])check(cudaEventSynchronize(host->done[slot]));
            auto* records=host->records[slot].data_ptr<tq::TqBlockView>();
            std::vector<tq::RamKvReadInto> reads;
            std::size_t used=0,end=first,packed=0;
            // Bound the CPU request descriptors independently of KV image
            // sizes; this packet limit never limits the GPU attention tile.
            while(end<blocks && reads.size()<maximum_records) {
                const auto& source=snapshot->blocks[end-first_block];const auto count=image_bytes(source);
                if(count>packet_bytes-used)break;
                const auto kb=source.image?source.image->key_bytes:source.backing->key_bytes;
                const auto vb=source.image?source.image->value_bytes:source.backing->value_bytes;
                const auto key_bits=source.image?source.image->key_bits:source.backing->key_bits;
                const auto value_bits=source.image?source.image->value_bits:source.backing->value_bits;
                const auto key_states=source.image?source.image->key_state_bits:source.backing->key_state_bits;
                const auto value_states=source.image?source.image->value_state_bits:source.backing->value_state_bits;
                auto* device=result.payload.data_ptr<std::uint8_t>()+offset+used;
                records[end-first]={device,device+((kb+15)&~std::size_t(15)),
                    tq::tq_precision_flags(key_bits,key_states),tq::tq_precision_flags(value_bits,value_states)};
                reads.push_back({&source,host->packets[slot].data_ptr<std::uint8_t>()+packed,kb+vb});
                packed+=kb+vb;
                used+=count;++end;
            }
            s.store->read_packed_cpu_into_batch(reads);
            // Expand backwards in the existing packet so unread source bytes
            // survive overlap while device descriptors retain their alignment.
            std::size_t expanded=used;
            for(auto i=reads.size();i-->0;) {
                const auto& source=*reads[i].source;
                const auto kb=source.image?source.image->key_bytes:source.backing->key_bytes;
                const auto vb=source.image?source.image->value_bytes:source.backing->value_bytes;
                const auto extent=(kb+15)&~std::size_t(15);
                expanded-=image_bytes(source);
                auto* destination=host->packets[slot].data_ptr<std::uint8_t>()+expanded;
                if(destination!=reads[i].destination || extent!=kb) {
                    std::memmove(destination+extent,reads[i].destination+kb,vb);
                    std::memmove(destination,reads[i].destination,kb);
                }
            }
            check(cudaMemcpyAsync(result.payload.data_ptr<std::uint8_t>()+offset,
                host->packets[slot].data_ptr(),used,cudaMemcpyHostToDevice,stream));submitted=true;
            check(cudaMemcpyAsync(result.views.data_ptr<tq::TqBlockView>()+first,records,
                (end-first)*sizeof(*records),cudaMemcpyHostToDevice,stream));
            if(!host->done[slot])check(cudaEventCreateWithFlags(&host->done[slot],cudaEventDisableTiming));
            check(cudaEventRecord(host->done[slot],stream));offset+=used;first=end;
        }
    }catch(...){
        if(submitted)(void)cudaStreamSynchronize(stream);
        throw;
    }
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_QSA_TQ");trace && trace[0]=='1')std::fprintf(stderr,
        "native_qsa_prefill_layer_stage actual_match=1 blocks=%zu encoded_bytes=%zu original_precision=1 packet_bytes=%zu payload_cuda=%d descriptors_cuda=%d pinned_ring_bytes=%zu shared_host_ring=1 reused_blocks=%zu\n",
        blocks,bytes-previous_bytes,packet_bytes,int(result.payload.is_cuda()),int(result.views.is_cuda()),host->charge,first_block);
    s.prefetched_blocks=blocks;
    return result;
}
bool QsaRamCache::collect_graph_selection(const Tensor& selected,int queries,int count,std::uint32_t* graph_control) {
    auto& s=*impl_;
    if(!s.arena || queries<1 || queries>s.arena->queries || count!=s.count)throw std::logic_error("QSA graph selection was not prepared");
    auto& a=*s.arena;const auto stream=mfq_current_cuda_stream();
    if(!a.mapped_missing || !a.mapped_selected)return false;
    check(tq::tq_collect_ram_missing(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,
        selected.data_ptr<tq::SelectedBlock>(),queries,count,a.first.data_ptr<std::uint32_t>(),
        a.missing.data_ptr<std::int32_t>()+1,a.missing.data_ptr<int>(),
        static_cast<const tq::RamStageCache*>(s.cache_parameters.data_ptr()),stream));
    check(tq::tq_export_ram_selection(a.missing.data_ptr<std::int32_t>(),selected.data_ptr<tq::SelectedBlock>(),
        a.mapped_missing,a.mapped_selected,queries,count,stream,graph_control));
    return true;
}
bool QsaRamCache::graph_task_supported()const {
    const auto& s=*impl_;
    return s.mapped_capability && (!s.arena || (s.arena->mapped_missing && s.arena->mapped_selected)) && s.store->graph_reads_supported();
}
void QsaRamCache::prepare_graph_task(std::uint64_t epoch,int queries) {
    RamPinnedForeground pinned_foreground;
    auto& s=*impl_;
    const bool defer_cache=QsaRamDecodeBudgetScope::defer_shared_cache(s.device);
    if(!s.arena || (queries && queries!=s.arena->queries))prepare(std::max(1,queries),queries!=0,!defer_cache);
    s.drain();
    // Published descriptors retain their previous immutable images through
    // the last GPU reader. Appending KV and applying the history policy price
    // any capacity change. A recall alone can keep using the published images
    // while optional encoding continues.
    if(s.required_capacity.load())wait();
    if(!graph_task_supported())throw std::logic_error("QSA SSD graph task requires mapped control memory");
    // The task graph keeps one compact geometry across RAM/SSD changes.
    // Larger images own slot groups; only images exceeding a set bypass it.
    s.configure_cache(true,!defer_cache);s.graph_epoch=epoch;
    if(!s.arena->ssd) {
        const ResidentAllocationScope resident(!s.prefill_staging);
        // Payloads are opaque and may retain alignment padding from a prefix.
        // Match the existing bounded transient slot instead of assuming an
        // exact codec byte count for every externally retained backing.
        auto disk=std::make_shared<SsdGraphArena>();disk->account=s.worker->pool_account;disk->stride=s.arena->stride;
        const auto n=int64_t(s.arena->queries)*s.count;const auto host=s.options.device(tb::kCPU).pinned_memory(true);
        disk->payload=tb::empty({n*int64_t(disk->stride)},host.dtype(tb::kUInt8));
        disk->ids=tb::empty({n},host.dtype(tb::kInt32));
        disk->images=tb::empty({n*int64_t(sizeof(tq::RamBlockView))},host.dtype(tb::kUInt8));
        disk->backup=tb::empty({n*int64_t(sizeof(tq::RamBlockView))},s.options.dtype(tb::kUInt8));
        disk->saved=tb::empty({1},s.options.dtype(tb::kInt32));
        void* pointer=nullptr;check(cudaHostGetDevicePointer(&pointer,disk->payload.data_ptr(),0));disk->mapped_payload=static_cast<std::uint8_t*>(pointer);
        check(cudaHostGetDevicePointer(&pointer,disk->ids.data_ptr(),0));disk->mapped_ids=static_cast<std::int32_t*>(pointer);
        check(cudaHostGetDevicePointer(&pointer,disk->images.data_ptr(),0));disk->mapped_images=static_cast<tq::RamBlockView*>(pointer);
        disk->bytes=disk->payload.nbytes()+disk->ids.nbytes()+disk->images.nbytes();disk->account->physical.fetch_add(disk->bytes);
        s.arena->ssd=std::move(disk);
    }
}
QsaRamCache::Tensor QsaRamCache::stage_graph_task(const Tensor& selected,int queries,int count,const Tensor& live,const Tensor& position,int appended) {
    auto& s=*impl_;auto* window=DecodeWindow::recording();
    if(!window || !s.arena || !s.arena->ssd || queries<1 || queries>s.arena->queries || count!=s.count)
        throw std::logic_error("QSA SSD graph task was not prepared");
    auto state=window->workspace<Impl::GraphTask>(&s,[&]{return std::make_shared<Impl::GraphTask>(s,queries,count,
        window->layer_profiling() || std::getenv("TYLOINFER_TRACE_NATIVE_CUDA_GRAPH"));});
    if(state->queries!=queries || state->count!=count)throw std::logic_error("QSA SSD graph task query geometry changed");
    const auto stream=mfq_current_cuda_stream();cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(stream,&capture));
    const bool warm=capture==cudaStreamCaptureStatusNone;
    const bool deferred_cache=s.shared_gpu_cache &&
        (s.use_spare_vram || (!s.gpu_cache.owner && QsaRamDecodeBudgetScope::defer_shared_cache(s.device)));
    if(warm)state->reset();
    window->enroll({state.get(),[state,deferred_cache]{if(deferred_cache)state->reset_for_replay();else state->reset();},[state]{state->serve();},[state]{state->finish();},
        [state]{state->cancel();},[state]{state->clear();},[state]{state->clear();}});
    auto& a=*s.arena;auto& disk=*a.ssd;
    if(appended>0)check(tq::tq_publish_ram_append(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,live.data_ptr<tq::TqBlockView>(),
        position.defined()?position.data_ptr<int64_t>():nullptr,int64_t(s.graph_epoch),appended,unsigned(s.fp16_bytes),unsigned(s.affine_bytes),stream));
    if(!collect_graph_selection(selected,queries,count,state->mapped))throw std::logic_error("QSA graph task lost mapped selection memory");
    const auto stage=[&](tq::RamStageStep step) {
        const auto* device_cache=deferred_cache?static_cast<const tq::RamStageCache*>(s.cache_parameters.data_ptr()):nullptr;
        check(tq::tq_stage_ram_selected_step(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,selected.data_ptr<tq::SelectedBlock>(),queries,count,
            a.first.data_ptr<std::uint32_t>(),a.views.data_ptr<tq::TqBlockView>(),a.payload.data_ptr(),a.payload.numel(),a.stride,s.gpu_cache,step,nullptr,nullptr,stream,device_cache));
    };
    stage(tq::RamStageStep::Available);
    if(warm)state->serve();
    wait_mapped_flag(state->mapped+tq::RamGraphReady,stream);
    check(tq::tq_publish_ram_graph(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,disk.mapped_ids,disk.mapped_images,
        state->mapped,disk.backup.data_ptr<tq::RamBlockView>(),disk.saved.data_ptr<std::uint32_t>(),queries*count,stream));
    check(tq::tq_cancel_ram_graph_selection(selected.data_ptr<tq::SelectedBlock>(),queries*count,disk.saved.data_ptr<std::uint32_t>(),stream));
    stage(tq::RamStageStep::Hydrated);
    check(tq::tq_restore_ram_graph(s.sources.data_ptr<tq::RamBlockView>(),disk.mapped_ids,disk.backup.data_ptr<tq::RamBlockView>(),
        disk.saved.data_ptr<std::uint32_t>(),queries*count,stream));
    if(std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH")) {
        static thread_local unsigned reported=0;const unsigned mask=1u<<(queries-1);
        if(!(reported&mask)){reported|=mask;std::fprintf(stderr,"native_qsa_ssd_graph_task actual_match=1 queries=%d single_submission=1 bounded_cpu_reads=1 deferred_lease=1\n",queries);}
    }
    return a.views;
}
void QsaRamCache::append(std::vector<tq::MicroblockImage> images,std::uint64_t epoch,bool fresh) {
    RamPinnedForeground pinned_foreground;
    auto& s=*impl_;
    if(s.bits==4 && (!fresh || !s.adaptive_precision)) {
        RamForegroundCodec foreground(s.foreground_codec);
        images=s.encode(images,std::vector<tq::Precision>(images.size(),{4,4}));
    }
    const auto tq4=tq4_bytes(s.rows);
    const auto tier=s.bits==16?tq::RamKvTier::Fp16:s.bits==8 || (fresh && s.adaptive_precision)?tq::RamKvTier::Affine8:tq::RamKvTier::Tq4;
    for(const auto& image:s.pool->allocate_batch(images,s.mapped_capability))s.store->append_at_tier(image,{2*s.fp16_bytes,2*s.affine_bytes,2*tq4},tier,epoch);
    s.update_account();s.drain();
    const auto usage=s.store->stats();
    if(!s.budget_sampled || usage.ram_bytes>s.sampled_budget || (s.bits==4 && usage.tokens[1]>=2048))
        observe({},epoch);
}
void QsaRamCache::drain(){
    RamPinnedForeground pinned_foreground;impl_->drain();}
bool QsaRamCache::discard_idle_descriptors() {
    auto& s=*impl_;
    if(s.host_metadata || s.store->stats().blocks)return false;
    if(s.sources.defined())s.drain();
    s.sources={};s.live={};
    return true;
}
void QsaRamCache::bind_descriptors(const Tensor& live) {
    auto& s=*impl_;
    if(s.sources.defined() || !live.defined())throw std::logic_error("Invalid QSA descriptor rebind");
    s.sources=qsa_descriptor_table(s.capacity,sizeof(tq::RamBlockView),s.options,s.host_metadata);
    s.live=live;
}
void QsaRamCache::release_idle() {
    RamPinnedForeground pinned_foreground;
    auto& s=*impl_;const auto& worker=s.worker;
    {
        std::lock_guard<std::mutex> lock(worker->mutex);
        for(const auto& weak:worker->accounts)if(auto account=weak.lock())
            if(account->logical.load() || account->restore_target.load())return;
    }
    bool pending=false;
    {std::lock_guard<std::mutex> lock(worker->retired_mutex);pending=!worker->retired.empty();}
    const auto before=worker->pool_account->physical.load();
    MfqCudaGuard device(s.device);
    // The last reset publishes null descriptors asynchronously. Join only its
    // finite stream work, then release the retired images across all layers.
    if(pending)check(cudaStreamSynchronize(worker->stream));
    worker->collect();
    worker->prefill_transfer.reset();
    worker->prefix_images.clear();
    auto metadata=s.store->release_idle_metadata()+s.host_sources.capacity()*sizeof(tq::RamBlockView)+
        s.published_sources.capacity()*sizeof(Impl::PublishedSource);
    std::vector<tq::RamBlockView>().swap(s.host_sources);
    std::vector<Impl::PublishedSource>().swap(s.published_sources);
    {
        std::lock_guard<std::mutex> lock(s.queue_mutex);
        if(s.queued.empty()) {metadata+=s.queued.capacity()*sizeof(tq::RamKvChange);std::vector<tq::RamKvChange>().swap(s.queued);}
    }
    for(auto& publication:worker->publications) {
        metadata+=publication.owners.capacity()*sizeof(tq::RamKvChange);
        std::vector<tq::RamKvChange>().swap(publication.owners);
    }
    worker->budget();
    const auto after=worker->pool_account->physical.load();
    if(before>after && std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
        "native_qsa_ram_idle_release actual_match=1 before_bytes=%zu after_bytes=%zu released_bytes=%zu\n",before,after,before-after);
    if(metadata && std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
        "native_qsa_ram_idle_metadata actual_match=1 released_bytes=%zu\n",metadata);
}
QsaRamCache::Tensor QsaRamCache::stage(const Tensor& selected,int queries,int count,const Tensor& live,
        const Tensor& position,std::int64_t host_position,int appended,bool selection_ready) {
    RamPinnedForeground pinned_foreground;
    auto& s=*impl_;if(queries<1 || queries>8 || count!=s.count)throw std::invalid_argument("QSA RAM staging requires its prepared query window");
    if(!s.arena || queries>s.arena->queries || (s.prefill_staging && s.arena->stride!=s.staging_stride()))
        prepare(queries);
    if(auto* window=DecodeWindow::recording())
        window->workspace<StageArena>(s.arena.get(),[&]{return s.arena;});
    std::vector<tq::RamKvChange> leased,temporary;
    struct CopyGuard {
        cudaStream_t stream;bool pending=false;
        // SSD errors can unwind after the first copy was submitted. Its RAM
        // leases must outlive that finite copy even when no attention runs.
        ~CopyGuard(){if(pending)(void)cudaStreamSynchronize(stream);}
    } copy_guard{mfq_current_cuda_stream()};
    bool split_copy=false,hydrate_missing=false;
    const auto publish_append=[&] {
        if(appended>0)check(tq::tq_publish_ram_append(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,live.data_ptr<tq::TqBlockView>(),
            position.defined()?position.data_ptr<int64_t>():nullptr,host_position,appended,unsigned(s.fp16_bytes),unsigned(s.affine_bytes),mfq_current_cuda_stream()));
    };
    const auto stage_step=[&](tq::RamStageStep step) {
        auto& a=*s.arena;
        check(tq::tq_stage_ram_selected_step(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,selected.data_ptr<tq::SelectedBlock>(),queries,count,
            a.first.data_ptr<std::uint32_t>(),a.views.data_ptr<tq::TqBlockView>(),a.payload.data_ptr(),a.payload.numel(),a.stride,s.gpu_cache,step,nullptr,nullptr,mfq_current_cuda_stream()));
    };
    const bool has_ssd=s.store->stats().tokens[3]!=0 || s.pool->has_pageable();
    if(!DecodeWindow::recording())s.configure_cache(has_ssd);
    const bool deferred_cache=s.shared_gpu_cache && !s.profile_staging && DecodeWindow::recording() &&
        (s.use_spare_vram || (!s.gpu_cache.owner && QsaRamDecodeBudgetScope::defer_shared_cache(s.device)));
    if(s.direct_mapped_staging && !s.profile_staging && DecodeWindow::recording() &&
            s.mapped_capability && !s.ssd_enabled && !has_ssd && !s.gpu_cache.sets) {
        // A captured window joins attention before adapting the store. Pending
        // replacements retain previous image owners until publication retires
        // on this same stream, after the preceding attention readers.
        publish_append();auto& a=*s.arena;
        check(tq::tq_resolve_ram_selected(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,
            selected.data_ptr<tq::SelectedBlock>(),queries,count,a.views.data_ptr<tq::TqBlockView>(),a.stride,mfq_current_cuda_stream()));
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static thread_local unsigned reported=0;const unsigned mask=1u<<(queries-1);
            if(!(reported&mask)){reported|=mask;std::fprintf(stderr,
                "native_qsa_ram_direct_attention actual_match=1 queries=%d bits=%d payload_staging=0 selected_views=1 captured_window=1\n",queries,s.bits);}
        }
        return a.views;
    }
    const auto probe=s.staging_probe(queries,count);
    if(has_ssd) {
        cudaStreamCaptureStatus status;check(cudaStreamIsCapturing(mfq_current_cuda_stream(),&status));
        if(status!=cudaStreamCaptureStatusNone)throw std::logic_error("SSD KV fetch requires an eager graph boundary");
        // Finish this layer's outstanding replacements before collecting its
        // actual SSD misses. The new selected-hit protection keeps packed
        // hits readable even while incoming misses exceed cache capacity.
        s.wait();s.drain();
        // Background spill can first make SSD staging necessary during this
        // call. Prepare the mapped workspace on the path that will use it.
        if(s.mapped_capability && (!DecodeWindow::recording() || !s.arena->ssd)) {
            prepare_graph_task(std::uint64_t(host_position));
            selection_ready=false;
        }
        auto& a=*s.arena;
        // The preceding graph segment has already joined these pinned copies.
        // Background precision/placement changes invalidate that snapshot;
        // repeat only the small collector when a publication raced it.
        s.configure_cache(has_ssd);
        const bool reuse_selection=selection_ready && s.prepared_revision==s.publication_revision;
        if(!reuse_selection) {
        check(tq::tq_collect_ram_missing(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,
            selected.data_ptr<tq::SelectedBlock>(),queries,count,a.first.data_ptr<std::uint32_t>(),
            a.missing.data_ptr<std::int32_t>()+1,a.missing.data_ptr<int>(),s.gpu_cache,mfq_current_cuda_stream()));
        check(cudaMemcpyAsync(a.host_missing.data_ptr(),a.missing.data_ptr(),std::size_t(1+queries*count)*sizeof(int),cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        check(cudaMemcpyAsync(a.host_selected.data_ptr(),selected.data_ptr(),std::size_t(queries)*count*sizeof(tq::SelectedBlock),cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        }
        if(selection_ready && std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
            static thread_local unsigned reported=0;const unsigned mask=reuse_selection?1u:2u;
            if(!(reported&mask)){reported|=mask;std::fprintf(stderr,"native_qsa_graph_selection actual_match=1 reused=%d revision_check=1\n",int(reuse_selection));}
        }
        const auto* blocks=a.host_selected.data_ptr<tq::SelectedBlock>();
        // The copy above also retires the descriptor upload. Its publication
        // owner is no longer a reader lease and must not prevent admission.
        s.drain();
        const auto stored=s.store->stats().blocks;
        const auto* missing=a.host_missing.data_ptr<std::int32_t>();
        if(missing[0]<0 || missing[0]>queries*count)throw std::runtime_error("Invalid QSA RAM miss count");
        for(int at=0;at<missing[0];++at)if(std::size_t(missing[at+1])<stored){hydrate_missing=true;break;}
        {
            // This layer's producer was joined above. Its store retains the
            // immutable sources while the selected leases are assembled;
            // admission cannot start until all those leases are acquired.
            publish_append();copy_guard.pending=true;stage_step(tq::RamStageStep::Available);
            split_copy=true;
            check_stream_progress(mfq_current_cuda_stream());
        }
        if(s.mapped_capability) {
            // Eager verification and prefill share the graph's bounded host
            // transfer packet for ordinary RAM and SSD-backed images.
            auto* archived=a.host_missing.data_ptr<std::int32_t>();int retained=0;
            for(int at=0;at<archived[0];++at)
                if(std::size_t(archived[at+1])<stored)archived[++retained]=archived[at+1];
            archived[0]=retained;
            auto task=std::make_shared<Impl::GraphTask>(s,queries,count);
            task->host()[tq::RamGraphRequest]=1;task->serve();
            auto& disk=*a.ssd;const auto stream=mfq_current_cuda_stream();
            check(tq::tq_publish_ram_graph(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,disk.mapped_ids,disk.mapped_images,
                task->mapped,disk.backup.data_ptr<tq::RamBlockView>(),disk.saved.data_ptr<std::uint32_t>(),queries*count,stream));
            check(tq::tq_cancel_ram_graph_selection(selected.data_ptr<tq::SelectedBlock>(),queries*count,disk.saved.data_ptr<std::uint32_t>(),stream));
            stage_step(tq::RamStageStep::Hydrated);
            check(tq::tq_restore_ram_graph(s.sources.data_ptr<tq::RamBlockView>(),disk.mapped_ids,disk.backup.data_ptr<tq::RamBlockView>(),
                disk.saved.data_ptr<std::uint32_t>(),queries*count,stream));
            check(cudaStreamSynchronize(stream));copy_guard.pending=false;task->finish();s.drain();
            return a.views;
        }
        RamClaims ids(s.worker->claims,s.capacity,std::size_t(queries)*count);
        for(int i=0;i<queries*count;++i)if(blocks[i].id>=0 && std::size_t(blocks[i].id)<stored && blocks[i].token_mask)ids.add(std::size_t(blocks[i].id));
        ids.clear();
        // Pin all current RAM selections before admitting any SSD miss, so
        // admission cannot evict another block used by this same window.
        leased=s.store->lease(ids.ids,std::uint64_t(host_position));
        const auto admission_budget=s.budget();
        RamForegroundCodec foreground(s.foreground_codec);
        for(int at=0;at<missing[0];++at) {
            const auto id=std::size_t(missing[at+1]);if(id>=stored || s.store->resident(id))continue;
            auto image=s.store->read(id,admission_budget);
            tq::RamKvChange change{id,s.store->generation(id),image,{},tq::RamKvTier::Tq4};leased.push_back(change);
            if(!s.store->resident(id))temporary.push_back(change);
        }
        if(missing[0] && !s.reported_ssd_fetch && std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
            s.reported_ssd_fetch=true;const auto stats=s.store->stats();
            std::fprintf(stderr,"qsa_ram_ssd_fetch actual_match=1 misses=%d read_bytes=%llu promotions=%llu packed_hits_skip_ssd=1\n",
                missing[0],static_cast<unsigned long long>(stats.ssd_read_bytes),static_cast<unsigned long long>(stats.promoted_blocks));
        }
        if(!temporary.empty()) {
            s.queue(temporary);
        }
        s.drain();
        if(split_copy && std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
            static thread_local unsigned reported=0;const unsigned mask=1u<<(queries-1);
            if(!(reported&mask)){reported|=mask;std::fprintf(stderr,
                "native_qsa_ram_ssd_overlap actual_match=1 queries=%d misses=%d cache_bytes=%zu resident_copy_before_read=1 copy_before_lease=1 pinned_selection=1\n",queries,missing[0],s.gpu_cache.payload_bytes);}
        }
    }
    // New GPU appends are not immutable RAM records yet. They bypass the
    // persistent LRU until their archive has completed and been published.
    if(!split_copy)publish_append();
    auto& a=*s.arena;
    if(split_copy) {if(hydrate_missing)stage_step(tq::RamStageStep::Hydrated);}
    else if(deferred_cache)check(tq::tq_stage_ram_selected_step(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,
        selected.data_ptr<tq::SelectedBlock>(),queries,count,a.first.data_ptr<std::uint32_t>(),a.views.data_ptr<tq::TqBlockView>(),
        a.payload.data_ptr(),a.payload.numel(),a.stride,s.gpu_cache,tq::RamStageStep::Available,nullptr,nullptr,
        mfq_current_cuda_stream(),static_cast<const tq::RamStageCache*>(s.cache_parameters.data_ptr())));
    else if(s.gpu_cache.sets)check(tq::tq_stage_ram_selected_cached(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,selected.data_ptr<tq::SelectedBlock>(),queries,count,
        a.first.data_ptr<std::uint32_t>(),a.views.data_ptr<tq::TqBlockView>(),a.payload.data_ptr(),a.payload.numel(),a.stride,s.cache_envelope(),nullptr,nullptr,mfq_current_cuda_stream(),
        static_cast<const tq::RamStageCache*>(s.cache_parameters.data_ptr())));
    else check(tq::tq_stage_ram_selected(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,selected.data_ptr<tq::SelectedBlock>(),queries,count,
        a.first.data_ptr<std::uint32_t>(),a.views.data_ptr<tq::TqBlockView>(),a.payload.data_ptr(),a.payload.numel(),a.stride,
        probe?probe->copied.data_ptr<std::uint64_t>():nullptr,mfq_current_cuda_stream(),a.stride<2*s.fp16_bytes));
    if(probe) {
        const auto stream=mfq_current_cuda_stream();
        check(tq::tq_export_ram_selected(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,
            selected.data_ptr<tq::SelectedBlock>(),queries,count,probe->records.data_ptr<tq::RamSelectedRecord>(),stream));
        check(cudaMemcpyAsync(probe->host_records.data_ptr(),probe->records.data_ptr(),probe->host_records.nbytes(),cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(probe->host_copied.data_ptr(),probe->copied.data_ptr(),8,cudaMemcpyDeviceToHost,stream));
    }
    if(!temporary.empty()) {
        for(auto& change:temporary){change.previous=std::move(change.image);change.tier=tq::RamKvTier::Ssd;}
        s.queue(temporary);
        s.drain();
    }
    if(!leased.empty()) {
        Impl::Pending pending;pending.owners=std::move(leased);
        check(cudaEventCreateWithFlags(&pending.done,cudaEventDisableTiming));check(cudaEventRecord(pending.done,mfq_current_cuda_stream()));
        s.worker->retire(std::move(pending));
    }
    copy_guard.pending=false;
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static thread_local unsigned reported=0;const unsigned mask=1u<<(queries-1);
        if(!(reported&mask)){reported|=mask;std::fprintf(stderr,"native_qsa_ram_stage actual_match=1 queries=%d bits=%d union_copy=1 cache_bytes=%zu shared_stream_arena=1\n",queries,s.bits,s.gpu_cache.payload_bytes);}
    }
    return a.views;
}
void QsaRamCache::observe(const std::vector<double>& hits,std::uint64_t epoch) {
    RamPinnedForeground pinned_foreground;
    auto& s=*impl_;s.drain();std::vector<std::size_t> touched;
    const auto blocks=s.store->stats().blocks;for(std::size_t id=0;id<std::min(blocks,hits.size());++id)if(hits[id]>0)touched.push_back(id);
    s.store->touch(touched,epoch);
    if(s.maintenance.valid()) {
        if(s.maintenance.wait_for(std::chrono::seconds(0))!=std::future_status::ready)return;
        s.maintenance.get();
    }
    s.resolve_capacity();
    const auto physical_budget=s.budget();s.sampled_budget=physical_budget;s.budget_sampled=true;
    const auto usage=s.store->stats();
    const bool pressure=usage.ram_bytes>physical_budget;
    const bool fit_affine_batch=!pressure && s.bits==4 && usage.tokens[1]>=2048;
    auto target=s.maintenance_budget(physical_budget);
    if(fit_affine_batch) {
        // Keep a partial batch of fresh Affine8 images off the serial decode
        // path. Fit only complete2048-token batches on the background worker.
        const auto saving=512*(2*s.affine_bytes-2*tq4_bytes(s.rows));
        target=std::min(target,usage.ram_bytes-saving);
    }
    if(usage.ram_bytes<=target)return;
    s.maintenance=s.worker->submit([&s,target,physical_budget,pressure,fit_affine_batch]{
        check(cudaSetDevice(s.device));s.worker->release_readers();
        try {
        const auto maintain=[&] {
            if(s.ssd_enabled && !fit_affine_batch) {
                auto result=s.store->evict_backed(target);
                if(result.fits() || !pressure)return result;
            }
            return s.store->maintain(target,false);
        };
        auto result=maintain();
        // A completed GPU publication must stop pinning its old images even
        // when the model has moved on to other layers. Retry only after
        // releasing completed leases; long-lived readers remain untouched.
        while(result.needs_readers && s.worker->release_readers())
            result=maintain();
        // Fresh blocks can arrive during optional format conversion. Only
        // the actual RAM limit permits SSD eviction or a capacity error.
        if(!pressure && s.store->stats().ram_bytes>physical_budget)result=s.store->maintain(physical_budget,false);
        const auto stats=s.store->stats();s.update_account();
        if(std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
            std::fprintf(stderr,"qsa_ram_pressure actual_match=1 budget_bytes=%zu ram_bytes=%zu pooled_pinned_bytes=%zu fp16_tokens=%llu affine8_tokens=%llu tq4_tokens=%llu ssd_tokens=%llu quantized=%llu spilled=%llu maintenance_budget=%zu\n",
                physical_budget,stats.ram_bytes,s.pool->account->physical.load(),static_cast<unsigned long long>(stats.tokens[0]),static_cast<unsigned long long>(stats.tokens[1]),
                static_cast<unsigned long long>(stats.tokens[2]),static_cast<unsigned long long>(stats.tokens[3]),
                static_cast<unsigned long long>(stats.quantized_blocks),static_cast<unsigned long long>(stats.spilled_blocks),target);
        }
        // Expert cache reclamation may synchronize/free pinned allocations;
        // hand the capacity request back to the model thread, outside this
        // background codec job and every store/pool/graph wait lock.
        if(stats.ram_bytes>physical_budget)
            s.required_capacity.store(stats.ram_bytes);
        }catch(const DeferredPinnedAllocation& error) {
            auto previous=s.pool->pending_pinned.load();
            while(previous<error.bytes && !s.pool->pending_pinned.compare_exchange_weak(previous,error.bytes)){}
            // Partial codec publications remain valid. Hand quota recovery to
            // the model thread; never cudaFreeHost behind a waiting graph.
            s.update_account();s.required_capacity.store(s.store->stats().ram_bytes);
        }
    },s.maintenance_sequence);
}
void QsaRamCache::wait(){
    RamPinnedForeground pinned_foreground;
    auto& s=*impl_;s.wait();s.drain();
    bool retried_readers=false;
    while(s.store->stats().ram_bytes>s.budget()) {
        const auto logical=s.store->stats().ram_bytes;
        const auto physical=s.worker->pool_account->physical.load();
        s.required_capacity.store(logical);s.resolve_capacity();
        // Recompute pressure after publication releases its original images.
        // Logical eviction alone can leave the next pinned transfer without
        // physical pages; the resulting overhead changes the shared budget.
        check(cudaStreamSynchronize(mfq_current_cuda_stream()));s.drain();
        const bool progress=s.store->stats().ram_bytes<logical ||
            s.worker->pool_account->physical.load()<physical;
        if(!progress && retried_readers)break;
        // A completed publication can release a lease without releasing any
        // pages. Retry once, while external long-lived readers stay bounded.
        retried_readers=!progress;
    }
}
void QsaRamCache::truncate(std::size_t blocks){
    RamPinnedForeground pinned_foreground;
    auto& s=*impl_;
    if(blocks)s.wait();
    // Truncation invalidates in-flight encoding through the store's version
    // checks. An empty session has no capacity requirement to resolve.
    s.store->truncate(blocks);s.update_account();
    if(!blocks) {
        if(s.maintenance.valid())s.maintenance.get();
        s.required_capacity.store(0);
        s.budget_sampled=false;
    }
    s.drain();
    if(blocks)s.worker->budget();
}
bool QsaRamCache::graph_safe()const{return (!impl_->ssd_enabled || !impl_->maintenance.valid()) && impl_->store->stats().tokens[3]==0 && !impl_->pool->has_pageable();}
bool QsaRamCache::encoding_urgent()const{
    if(impl_->foreground_codec.load())return true;
    const auto& worker=*impl_->worker;const auto job=worker.running_job.load();
    return job && job<=worker.urgent_through.load();
}
std::shared_ptr<tq::RamKvSnapshot> QsaRamCache::snapshot(std::size_t blocks) {
    wait();return impl_->store->snapshot(blocks);
}
std::shared_ptr<void> QsaRamCache::reserve_restore(const tq::RamKvSnapshot& state,bool borrow_prefix) {
    auto& s=*impl_;std::array<long double,3> precision{};
    if(state.blocks.size()>std::size_t(s.capacity))
        throw std::invalid_argument("QSA restore metadata exceeds cache capacity");
    // All layers reserve before any payload is hydrated. Include descriptor
    // pages and one policy batch of appends in the physical RAM sample. Exact
    // prefix capacity would force every layer to grow at its first new block.
    const auto incoming=state.blocks.size();
    const auto append=s.restore_append_blocks(incoming);
    const auto metadata_blocks=incoming+append;
    s.store->reserve_metadata(metadata_blocks);
    if(metadata_blocks>s.host_sources.capacity()) {
        const auto live=s.host_sources.size();
        s.host_sources.resize(metadata_blocks);s.host_sources.resize(live);
    }
    if(metadata_blocks>s.published_sources.capacity()) {
        const auto live=s.published_sources.size();
        s.published_sources.resize(metadata_blocks);s.published_sources.resize(live);
    }
    std::size_t shared=0;
    for(const auto& block:state.blocks) {
        for(unsigned tier=0;tier<3;++tier)
            precision[tier]+=block.image && unsigned(block.tier)==tier?block.image->bytes():block.bytes[tier];
        if(borrow_prefix && s.share_resident_prefix && s.account->fixed_ram_only)
            if(const auto image=s.worker->shared_prefix_image(block))shared+=image->bytes();
    }
    const auto canonical=s.bits==16?tq::RamKvTier::Fp16:s.bits==8?tq::RamKvTier::Affine8:tq::RamKvTier::Tq4;
    const auto target=s.target(s.adaptive_precision?tq::ram_precision_target(state.blocks.size(),precision,canonical):
        std::size_t(precision[unsigned(canonical)]));
    struct Reservation {
        std::shared_ptr<RamAccount> account;std::shared_ptr<RamWorker> worker;
        std::size_t previous,previous_shared;
        Reservation(std::shared_ptr<RamAccount> a,std::shared_ptr<RamWorker> w,std::size_t bytes,std::size_t shared)
            :account(std::move(a)),worker(std::move(w)),previous(account->restore_target.exchange(bytes)),
             previous_shared(account->restore_shared.exchange(shared)){}
        ~Reservation(){
            account->restore_shared.store(previous_shared);
            account->restore_target.store(previous);
            try{worker->budget();}catch(...){}
        }
    };
    return std::make_shared<Reservation>(s.account,s.worker,target,shared);
}
void QsaRamCache::enable_prefix_sharing() {impl_->share_resident_prefix=true;}
bool QsaRamCache::restore_can_place_in_ram(std::size_t bytes) const {
    const auto& s=*impl_;
    if(!s.account->fixed_ram_only || !s.account->restore_shared.load())return false;
    const auto& worker=*s.worker;
    auto free=mfq::sample_host_memory().available_host_bytes();
    free-=std::min(free,worker.workspace_bytes.load());
    std::lock_guard lock(s.worker->mutex);
    for(const auto& weak:worker.accounts)if(const auto account=weak.lock()) {
        const auto target=std::max(account->target.load(),account->restore_target.load());
        const auto resident=std::max(account->logical.load(),account->restore_shared.load());
        if(target>resident)free-=std::min(free,target-resident);
    }
    return bytes<=free;
}
void QsaRamCache::restore(const tq::RamKvSnapshot& state,bool borrow_prefix) {
    RamPinnedForeground pinned_foreground;
    restore(state,borrow_prefix,{});
}
void QsaRamCache::restore(const tq::RamKvSnapshot& state,bool borrow_prefix,
                          const std::function<void(std::size_t)>& block_read,
                          std::span<const QsaRamCache* const> peers) {
    RamPinnedForeground pinned_foreground;
    auto& s=*impl_;
    const auto stream=mfq_current_cuda_stream();
    // Prefix restoration can fit a block before entering generation. Give
    // its required codec buffers the same pinned-memory recovery as decode.
    const PinnedAllocationScope pinned(s.device,stream,[&](std::size_t bytes) {
        return reclaim_qsa_pinned(s.device,stream,bytes);
    });
    s.wait();check(cudaStreamSynchronize(mfq_current_cuda_stream()));s.drain();
    // Publish descriptors without a late pinned allocation after hydration
    // has consumed the available locked-page capacity.
    if(!state.blocks.empty()) {
        auto publication=s.worker->publication(state.blocks.size(),s.options,s.mapped_capability);
        s.worker->publications.push_back(std::move(publication));
    }
    if(s.gpu_cache.sets && !s.shared_gpu_cache)check(tq::tq_reset_ram_stage_cache(s.gpu_cache,mfq_current_cuda_stream()));
    RamForegroundCodec foreground(s.foreground_codec);
    const auto written=s.store->stats().ssd_written_bytes;
    const bool share=borrow_prefix && s.share_resident_prefix && s.account->fixed_ram_only;
    const bool share_peers=borrow_prefix && s.share_resident_prefix && s.ssd_enabled && !peers.empty();
    const auto reservation=reserve_restore(state,borrow_prefix);
    std::vector<tq::RamKvImagePtr> shared_images;
    std::size_t shared_bytes=0,peer_count=0;
    if(share && s.account->restore_shared.load()) {
        shared_images.reserve(state.blocks.size());
        for(const auto& block:state.blocks) {
            auto image=s.worker->shared_prefix_image(block);
            if(image)shared_bytes+=image->bytes();
            shared_images.push_back(std::move(image));
        }
        s.account->restore_shared.store(shared_bytes);
    } else if(share_peers) {
        shared_images.resize(state.blocks.size());
        for(const auto* peer:peers) {
            if(!peer || peer==this || peer->impl_->worker!=s.worker)continue;
            const auto& store=*peer->impl_->store;
            const auto common=store.snapshot(std::min(state.blocks.size(),store.stats().blocks));
            const auto before=shared_bytes;
            for(std::size_t id=0;id<common->blocks.size();++id) {
                if(shared_images[id] || state.blocks[id].tier==tq::RamKvTier::Ssd)continue;
                const auto& src=common->blocks[id];
                const auto& backing=state.blocks[id].backing;
                if(!src.image || !src.backing || !src.backing->source || !backing || !backing->source)continue;
                const auto& source=*src.backing->source;const auto& target=*backing->source;
                if(!source.file_reader || source.file_reader!=target.file_reader ||
                        src.backing->offset>UINT64_MAX-source.file_offset || backing->offset>UINT64_MAX-target.file_offset ||
                        source.file_offset+src.backing->offset!=target.file_offset+backing->offset)continue;
                const auto& image=*src.image;
                if(image.key_bytes!=backing->key_bytes || image.value_bytes!=backing->value_bytes ||
                        image.key_bits!=backing->key_bits || image.value_bits!=backing->value_bits ||
                        image.key_state_bits!=backing->key_state_bits || image.value_state_bits!=backing->value_state_bits)continue;
                // A separate lease lets either store evict its own entry. The
                // backing allocation is retired only after all stores release it.
                auto alias=std::make_shared<tq::RamKvImage>(image);
                alias->shared_prefix_allocation=true;
                shared_bytes+=alias->bytes();
                shared_images[id]=std::move(alias);
            }
            if(shared_bytes>before)++peer_count;
        }
    }
    s.worker->rebalance_restore();
    const auto physical_budget=s.budget();
    // Leave a bounded append transaction when restored blocks can be demoted
    // or kept on SSD. Use the same policy geometry as the restore planner so
    // new microblocks need not trigger foreground conversion immediately.
    const auto append=s.restore_append_blocks(state.blocks.size());
    const auto fresh_bytes=2*(s.bits==16?s.fp16_bytes:s.affine_bytes);
    const auto growth=append && fresh_bytes>SIZE_MAX/append?SIZE_MAX:append*fresh_bytes;
    const auto append_bytes=s.ssd_enabled || s.adaptive_precision?growth:0;
    const auto budget=physical_budget;
    const auto restore=[&]{s.store->restore_reserved(state,budget,borrow_prefix,block_read,
        [&](const auto& image,std::size_t remaining){return s.pool->reserve_counted(image,remaining,s.mapped_capability);},
        append_bytes,shared_images.empty()?nullptr:&shared_images);
        s.update_account();s.drain();};
    try{restore();}
    catch(...){
        if(std::getenv("TYLOINFER_TRACE_QSA_TQ"))
            std::fprintf(stderr,"qsa_ram_restore_rejected budget_bytes=%zu target_bytes=%zu shared_bytes=%zu free_host_bytes=%llu blocks=%zu\n",
                budget,s.account->restore_target.load(),shared_bytes,
                static_cast<unsigned long long>(mfq::sample_host_memory().available_host_bytes()),state.blocks.size());
        s.update_account();throw;
    }
    s.update_account();s.drain();
    if(share)s.worker->remember_prefix_images(state,*s.store->snapshot(state.blocks.size()));
    if(shared_bytes && std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
        const auto resident=share?shared_bytes:s.account->shared_prefix_ram.load();
        std::fprintf(stderr,"native_qsa_shared_prefix_images bytes=%zu eligible_bytes=%zu blocks=%zu peers=%zu\n",
            resident,shared_bytes,state.blocks.size(),peer_count);
    }
    if(std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
        const auto stats=s.store->stats();
        std::fprintf(stderr,"qsa_ram_restore actual_match=1 borrow_prefix=%d ram_bytes=%zu budget_bytes=%zu target_bytes=%zu fp16_tokens=%llu affine8_tokens=%llu tq4_tokens=%llu ssd_tokens=%llu session_ssd_written_bytes=%llu\n",
            int(borrow_prefix),stats.ram_bytes,budget,s.account->restore_target.load(),
            static_cast<unsigned long long>(stats.tokens[0]),static_cast<unsigned long long>(stats.tokens[1]),
            static_cast<unsigned long long>(stats.tokens[2]),static_cast<unsigned long long>(stats.tokens[3]),
            static_cast<unsigned long long>(stats.ssd_written_bytes-written));
    }
}
tq::MicroblockImage QsaRamCache::reconstruction_image(std::size_t id){
    RamPinnedForeground pinned_foreground;
    auto& s=*impl_;RamForegroundCodec foreground(s.foreground_codec);return copy(*s.store->read(id,s.budget(),false));
}
QsaRamMemory QsaRamCache::memory()const {
    const auto& s=*impl_;const auto stats=s.store->stats();
    return {s.worker->pool_account.get(),stats.ram_bytes,s.worker->pool_account->physical.load(),
        stats.resident_target_bytes,stats.session_disk_bytes,0,stats.tokens,s.ssd_enabled,s.adaptive_precision,
        s.pool->pageable_pool?s.pool->pageable_pool->own_bytes.load():0,
        s.shared_gpu_cache && s.worker->stage_cache?s.worker->stage_cache->bytes:0};
}
std::vector<QsaRamCache::Tensor> QsaRamCache::resident_tensors()const {
    const auto& s=*impl_;return {s.host_metadata?Tensor{}:s.sources,s.cache_slots,s.cache_locks,s.cache_frame,s.cache_payload,
        s.cache_parameters,s.cache_window_frames};
}
tq::AdaptiveCacheStats QsaRamCache::stats()const {
    const auto a=impl_->store->stats();tq::AdaptiveCacheStats result;result.blocks=a.blocks;result.resident_bytes=a.ram_bytes;
    result.blocks_per_tier[5]=std::size_t(a.tokens[0]+a.tokens[1])/4;result.blocks_per_tier[0]=std::size_t(a.tokens[2])/4;
    result.demotions=a.quantized_blocks;result.promotions=a.promoted_blocks;result.canonical_read_bytes=a.ssd_read_bytes;
    return result;
}
} // namespace mfq::cuda
