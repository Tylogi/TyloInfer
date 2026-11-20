#include "kernels/qsa_ram.h"
#include "qsa_tq_cache.h"
#include "mfq_cuda_profiler.h"
#include "qsa_cache_layout.h"
#include "qsa_ram_cache.h"
#include "qsa_indexer_cpu.h"
#include "pinned_allocation_retry.h"
#include "prefill_projection_plan.h"
#include "prefill_budget.h"
#include "qsa_tq_session.h"
#include "kernels/qsa_tq_fa.h"
#include "mfq_cuda_context.h"
#include "decode_window.h"
#include "moe_pipeline.h"
#include "kernels/qsa_tq.h"
#include "kernels/qsa_tq_encode.h"
#include "mfq_cuda_resident_allocations.h"
#include "tyloinfer/kernels/cuda/qwen4_exp.h"
#include "tyloinfer/file_read_window.h"
#include "tyloinfer/host_memory.h"
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <map>
#include <numeric>
#include <tuple>
#include <unordered_map>

namespace mfq::cuda {
namespace tb=mfq_tensor_backend;
namespace tq=tyloinfer::qsa;
namespace {
void check(int code) {TYLOINFER_CUDA_CHECK(cudaError_t(code));}
std::size_t option(const char* name,std::size_t fallback) {
    const auto* s=std::getenv(name);if(!s || !*s)return fallback;
    std::size_t used=0;const auto value=std::stoull(s,&used);
    if(used!=std::strlen(s) || s[0]=='-')throw std::invalid_argument(std::string("invalid ")+name);
    return std::size_t(value);
}
struct TqCodecBuffer {
    void* pointer=nullptr;std::size_t bytes=0;
    TqCodecBuffer()=default;TqCodecBuffer(const TqCodecBuffer&)=delete;TqCodecBuffer& operator=(const TqCodecBuffer&)=delete;
    ~TqCodecBuffer(){if(pointer)(void)cudaFree(pointer);}
    void reserve(std::size_t n) {if(n<=bytes)return;if(pointer)check(cudaFree(pointer));pointer=nullptr;bytes=0;
        check(cudaMalloc(&pointer,n));bytes=n;}
    template<class T> T* as(std::size_t offset=0)const{return reinterpret_cast<T*>(static_cast<std::uint8_t*>(pointer)+offset);}
};
struct TqCodecHostBuffer {
    void* pointer=nullptr;std::size_t bytes=0;
    TqCodecHostBuffer()=default;TqCodecHostBuffer(const TqCodecHostBuffer&)=delete;
    TqCodecHostBuffer& operator=(const TqCodecHostBuffer&)=delete;
    ~TqCodecHostBuffer(){if(pointer)(void)release_pinned_host(pointer,bytes);}
    void reserve(std::size_t n) {
        if(n<=bytes)return;
        void* replacement=nullptr;
        const auto status=allocate_pinned_host(&replacement,n,cudaHostAllocDefault,"qsa_codec");
        if(pinned_allocation_rejected(status)) {
            (void)cudaGetLastError();
            const auto message="CUDA pinned allocation failed: requested="+std::to_string(n)+" role=qsa_codec";
            throw DeferredPinnedAllocation(n,message.c_str());
        }
        check(status);
        if(pointer)check(release_pinned_host(pointer,bytes));pointer=replacement;bytes=n;
    }
    template<class T>T* as(std::size_t offset=0)const{return reinterpret_cast<T*>(static_cast<std::uint8_t*>(pointer)+offset);}
};
struct TqEncoderStream {
    cudaStream_t value=nullptr;
    ~TqEncoderStream(){if(value)(void)cudaStreamDestroy(value);}
};
std::size_t codec_output_stride(int rows) {
    return (3*std::size_t(rows)+7)/8+259*std::size_t(rows)+16;
}
std::size_t plan_codec_batch(std::size_t blocks,int rows,tq::TqEncodePlan& plan) {
    const auto budget=option("TYLOINFER_QSA_TQ_WORKSPACE_BYTES",32ULL<<20);
    // Fit one pressure-policy transaction, reducing the batch only when
    // the configured scratch budget cannot hold its actual geometry.
    auto batch=std::min(blocks,std::size_t(tq::RamPressureConfig{}.quantize_tokens/4));
    for(;;) {
        const auto error=tq::tq_make_encode_plan(&plan,int(2*batch),rows,tq::TqEncodeProfile::Fast,budget);
        if(error!=int(cudaErrorMemoryAllocation) || batch==1){check(error);return batch;}
        batch=(batch+1)/2;
    }
}
struct TqCodecLayout {
    std::size_t source,views,decoded,encoded,block_sizes,workspace;
    TqCodecLayout(const tq::TqEncodePlan& plan,std::size_t input_bytes)
        :source(input_bytes),views(std::size_t(plan.blocks/2)*sizeof(tq::TqBlockView)),
         decoded(std::size_t(plan.rows)*256*sizeof(float)),encoded(std::size_t(plan.blocks)*codec_output_stride(plan.rows_per_block)),
         block_sizes(std::size_t(plan.blocks)*sizeof(std::uint32_t)),workspace(plan.workspace_bytes){}
};
struct TqCodecArena {
    std::mutex mutex;std::once_flag initialized;
    TqEncoderStream owner;cudaStream_t stream=nullptr;
    TqCodecBuffer source,source_views,decoded,encoded,sizes,flags,status,targets,workspace;
    TqCodecHostBuffer input,output;
    std::size_t reported_bytes=0;
    std::size_t device_bytes()const {
        return source.bytes+source_views.bytes+decoded.bytes+encoded.bytes+sizes.bytes+flags.bytes+status.bytes+targets.bytes+workspace.bytes;
    }
    std::size_t growth_bytes(const TqCodecLayout& layout)const {
        const auto growth=[](std::size_t required,const TqCodecBuffer& buffer) {
            return required-std::min(required,buffer.bytes);
        };
        return growth(layout.source,source)+growth(layout.views,source_views)+growth(layout.decoded,decoded)+
            growth(layout.encoded,encoded)+growth(layout.block_sizes,sizes)+growth(layout.block_sizes,flags)+
            growth(sizeof(int),status)+growth(layout.block_sizes,targets)+growth(layout.workspace,workspace);
    }
};
std::shared_ptr<TqCodecArena> shared_codec_arena(int device,cudaStream_t stream) {
    // Foreground imports and the background worker can run on different host
    // threads. Both must find the same arena for this model execution stream.
    static std::mutex mutex;
    static std::map<std::pair<int,cudaStream_t>,std::weak_ptr<TqCodecArena>> arenas;
    std::lock_guard<std::mutex> lock(mutex);auto& slot=arenas[{device,stream}];auto arena=slot.lock();
    if(!arena){arena=std::make_shared<TqCodecArena>();slot=arena;}return arena;
}
tb::Tensor upload(const void* data,std::size_t bytes,const tb::TensorOptions& options) {
    auto host=tb::empty({int64_t(bytes)},options.device(tb::kCPU).dtype(tb::kUInt8));
    if(bytes)std::memcpy(host.data_ptr(),data,bytes);return host.to(options.dtype(tb::kUInt8));
}
tb::Tensor signs(int width,bool value,const tb::TensorOptions& options) {
    std::uint32_t state=0x6d2b79f5u^(std::uint32_t(width)*7919u+unsigned(value));
    auto host=tb::empty({width},options.device(tb::kCPU).dtype(tb::kFloat32));
    for(int d=0;d<width;++d){state^=state<<13;state^=state>>17;state^=state<<5;host.data_ptr<float>()[d]=(state&1)?1.f:-1.f;}
    return host.to(options.dtype(tb::kFloat32));
}
}
std::string qsa_cache_layout_json() {
    const auto views=sizeof(tq::TqBlockView)+sizeof(tq::RamBlockView)+sizeof(float);
    const auto host=tq::RamKvStore::metadata_bytes_per_block()+sizeof(QsaIndexArchive)+2*sizeof(void*)+
        sizeof(std::weak_ptr<const tq::RamKvImage>)+sizeof(std::shared_ptr<const tq::RamKvBacking>);
    const auto* private_cache=std::getenv("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES");
    const auto shared=private_cache && *private_cache?0:option("TYLOINFER_QSA_RAM_SHARED_VRAM_CACHE_BYTES",qsa_shared_vram_cache_default);
    const auto dense_shared=private_cache && *private_cache?0:option("TYLOINFER_QSA_RAM_SHARED_VRAM_CACHE_BYTES",0);
    return "{\"version\":1,\"tokens_per_block\":4,\"quantize_tokens\":2048,\"gpu_view_bytes\":"+
        std::to_string(views)+",\"host_block_metadata_bytes\":"+std::to_string(host)+",\"shared_vram_cache_bytes\":"+
        std::to_string(shared)+",\"dense_shared_vram_cache_bytes\":"+std::to_string(dense_shared)+",\"private_vram_cache_bytes_per_layer\":"+
        std::to_string(option("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES",0))+",\"graph_query_capacity\":8}";
}
std::string qsa_cache_mode() {
    if(const auto* value=std::getenv("TYLOINFER_QSA_CACHE_MODE");value && *value) {
        const std::string mode(value);
        if(mode=="auto" || mode=="fp16" || mode=="affine8" || mode=="tq4" || mode=="legacy")return mode;
        throw std::invalid_argument("QSA cache mode must be auto, fp16, affine8, tq4 or legacy");
    }
    if(const auto* value=std::getenv("TYLOINFER_QSA_RAM_KV_BITS");value && *value) {
        switch(option("TYLOINFER_QSA_RAM_KV_BITS",0)) {
            case 0:return "legacy";case 16:return "fp16";case 8:return "affine8";case 4:return "tq4";
            default:throw std::invalid_argument("QSA RAM KV bits must be 0, 16, 8 or 4");
        }
    }
    const auto* value=std::getenv("TYLOINFER_QSA_ADAPTIVE_TQ");
    return value && std::string(value)=="1"?"experimental":"legacy";
}
bool qsa_tq_cache_enabled() {return qsa_cache_mode()!="legacy";}
static bool qsa_cpu_indexer_enabled() {
    const auto* flag=std::getenv("TYLOINFER_QSA_CPU_INDEXER");
    return flag && flag[0]=='1' && qsa_tq_cache_enabled();
}

struct QsaTqCache::Impl {
    // Adjacent layers share one pinned transfer image on their worker thread.
    // Weak pooling ties its lifetime to the live model caches. A lease also
    // serializes reuse if a cache later migrates to another worker thread.
    struct ArchiveArena {
        tb::Tensor host;
        mfq::cuda::StreamHandle stream;
        mfq::cuda::Event ready,done;
        std::mutex mutex;
        explicit ArchiveArena(int device):stream(mfq::cuda::stream_from_pool(false,device)){}
    };
    struct ArchiveLease {
        std::shared_ptr<ArchiveArena> arena;
        std::unique_lock<std::mutex> lock;
        bool submitted=false;
        explicit ArchiveLease(std::shared_ptr<ArchiveArena> a):arena(std::move(a)),lock(arena->mutex){}
        ~ArchiveLease(){if(submitted)(void)cudaStreamSynchronize(arena->stream.stream());}
        void wait(){if(submitted){arena->done.synchronize();submitted=false;}}
    };
    std::shared_ptr<ArchiveArena> archive_arena;
    std::unique_ptr<ArchiveLease> lease_archive() {
        if(!archive_arena) {
            static thread_local std::unordered_map<int,std::weak_ptr<ArchiveArena>> arenas;
            auto& slot=arenas[device];archive_arena=slot.lock();
            if(!archive_arena){archive_arena=std::make_shared<ArchiveArena>(device);slot=archive_arena;}
        }
        return std::make_unique<ArchiveLease>(archive_arena);
    }
    struct ImageBatch {tb::Tensor payload;std::vector<std::uint8_t> host;std::vector<tq::TqBlockView> records;};
    struct Retired {cudaEvent_t done=nullptr;std::vector<std::shared_ptr<ImageBatch>> images;};
    struct Upload {cudaEvent_t done=nullptr;std::shared_ptr<ImageBatch> image;};
    int64_t maximum=0,tokens=0;int heads,kv_heads,width,padded,index_heads,index_width,budget,rows,capacity,device=0;
    std::size_t byte_budget=0,canonical_bytes=0;unsigned interval=128;
    const tb::TensorOptions options;
    // The next use (or wait/reset) consumes this snapshot before appending
    // new blocks, preserving its observation length and token timeline.
    struct HistoryTransfer {
        tb::Tensor host;
        cudaEvent_t done=nullptr;
        bool pending=false;
        std::uint64_t tokens=0;
        std::size_t budget=0;
        int pools=0;
        ~HistoryTransfer() {
            if(pending)(void)cudaEventSynchronize(done);
            if(done)(void)cudaEventDestroy(done);
        }
    } history_transfer;
    tb::Tensor ks,vs,relative,views,index_codes,index_scales,index_zero,hits,mass;
    struct IndexPointers {
        std::int16_t* codes=nullptr;
        std::uint16_t* scales=nullptr;
        std::int16_t* zero=nullptr;
    } device_index;
    bool host_indexer;
    bool migrated_indexer=false;
    bool host_metadata=false;
    std::uint64_t host_index_calls=0;
    double host_index_ms=0,host_index_transfer_ms=0;
    tb::Tensor tail_k,tail_v,tail_index,pending_k,pending_v;
    tb::Tensor tail_storage_k,tail_storage_v,tail_storage_index,pending_modes_k,pending_modes_v;
    tb::Tensor rotated_tail_k,rotated_tail_v;
    tb::Tensor append_k,append_v,append_index,archive;
    int append_capacity=0;bool append_small=true,pending_graph=false,ssd_cache=false,graph_complete=false;
    int64_t append_input_position=-1;
    int graph_queries=1,graph_phase=0,graph_blocks=0;
    bool batch_prepared=false;
    tb::Tensor seed_k,seed_v,seed_index,seed_rotated_k,seed_rotated_v;
    tb::Tensor graph_destination,graph_archive,graph_archive_host;
    std::uint8_t* graph_archive_mapped=nullptr;
    int64_t graph_archive_first=-1;
    int graph_archive_count=0;
    std::shared_ptr<ImageBatch> graph_image;
    tb::Tensor canonical_slab;
    int64_t canonical_cycle=-1;
    struct AttentionWorkspace {
        int queries=0,pools=0,parts=0;
        tb::Tensor ranked,blocks,rotated,partial,stats,logmass,norm,indexer;
        float* block_mass_data() const {return logmass.defined()?logmass.data_ptr<float>():nullptr;}
        float* log_normalizer_data() const {return norm.defined()?norm.data_ptr<float>():nullptr;}
    } decode_attention,prefill_attention,pending_attention;
    std::shared_ptr<const AttentionWorkspace> shared_pending_attention;
    cudaStream_t pending_attention_stream=nullptr;
    std::array<AttentionWorkspace,7> batch_attention;
    struct BatchAppendWorkspace {
        tb::Tensor key,value,index,fp16_key,fp16_value,rotated_key,rotated_value,packed;
    };
    std::array<BatchAppendWorkspace,7> batch_append;
    AttentionWorkspace& graph_workspace(int queries) {
        return queries==1?pending_attention:batch_attention.at(std::size_t(queries-2));
    }
    int graph_destination_slots() const {
        // A RAM-backed replay publishes completed blocks into immutable host
        // storage before the next replay. Eight queries can complete two.
        return ram?2:int(interval/4);
    }
    tq::PendingTail committed_tail() {
        return {tail_storage_k.data_ptr<float>(),tail_storage_v.data_ptr<float>(),tail_storage_index.data_ptr<float>(),
            rotated_tail_k.data_ptr<float>(),rotated_tail_v.data_ptr<float>()};
    }
    tq::PendingTail seeded_tail() {
        return {seed_k.data_ptr<float>(),seed_v.data_ptr<float>(),seed_index.data_ptr<float>(),
            seed_rotated_k.data_ptr<float>(),seed_rotated_v.data_ptr<float>()};
    }
    void cancel_batch_warmup() {
        if(!batch_prepared)return;
        check(tq::tq_copy_pending_tail(seeded_tail(),committed_tail(),kv_heads,width,padded,index_width,mfq_current_cuda_stream()));
        batch_prepared=false;
    }
    std::unique_ptr<tq::TqPrefillGraph> prefill_graph;
    std::optional<QsaRamPrefillStage> prefill_layer_stage;
    std::size_t prefill_layer_end=0;
    std::unique_ptr<tq::AdaptiveCache> cache;
    std::unique_ptr<QsaRamCache> ram;
    int ram_bits=0;bool adaptive_precision=false;
    std::shared_ptr<mfq::cache::QsaKvStore> raw_index_store;
    std::vector<std::shared_ptr<QsaIndexArchive>> index_archive;
    std::vector<std::shared_ptr<ImageBatch>> owners;
    std::vector<Retired> retired;
    std::vector<Upload> uploads;
    std::mutex queue_mutex;std::vector<tq::BlockReplacement> queued;
    std::size_t resident_append_begin=0,resident_append_end=0;
    struct ResidentAppendScope {
        Impl& owner;
        ResidentAppendScope(Impl& value,std::size_t first,std::size_t count):owner(value) {
            std::lock_guard<std::mutex> lock(owner.queue_mutex);
            owner.resident_append_begin=first;owner.resident_append_end=first+count;
        }
        ~ResidentAppendScope() {
            std::lock_guard<std::mutex> lock(owner.queue_mutex);
            owner.resident_append_begin=owner.resident_append_end=0;
        }
    };
    std::string directory;
    cudaStream_t codec_stream_key=nullptr;std::once_flag bind_encoder;
    std::shared_ptr<TqCodecArena> encoder;
    const bool use_spare_vram;

    Impl(int64_t max,int h,int kh,int w,int ih,int iw,int b,const tb::TensorOptions& opts,bool ssd,bool prefer_host_indexer,
            bool prefer_host_metadata,bool spare_vram)
        :maximum(max),heads(h),kv_heads(kh),width(w),padded(w<=256?256:w<=512?512:1024),index_heads(ih),
         index_width(iw),budget(b),rows(0),capacity(0),options(opts),host_indexer(prefer_host_indexer || qsa_cpu_indexer_enabled()),
         ssd_cache(ssd),use_spare_vram(spare_vram) {
        if(max<=0 || max>INT_MAX-3 || h<=0 || kh<=0 || h%kh || w<=0 || w>1024 || ih<=0 || iw<=0 || b<=0 || b%4)
            throw std::invalid_argument("invalid adaptive QSA geometry");
        const auto row_count=4LL*kh*(padded/256);
        // Native compact images have 32-bit lengths and integer row strides.
        if(row_count>(INT_MAX-16)/259)throw std::invalid_argument("adaptive QSA image stride exceeds 32 bits");
        rows=int(row_count);capacity=int((max+3)/4);
        const auto mode=qsa_cache_mode();adaptive_precision=mode=="auto";
        ram_bits=mode=="fp16"?16:mode=="auto" || mode=="affine8"?8:mode=="tq4"?4:0;
        if(!ram_bits && mode!="experimental")throw std::invalid_argument("Legacy QSA cannot construct a RAM precision cache");
        host_metadata=qsa_host_metadata_enabled(prefer_host_metadata || (ram_bits && !ssd_cache));
        std::fprintf(stderr,"native_qsa_cache_mode mode=%s canonical_bits=%d adaptive=%d ssd=%d metadata=%s actual_match=1\n",
            mode.c_str(),ram_bits,int(adaptive_precision),int(ssd_cache),host_metadata?"host":"device");
        check(cudaGetDevice(&device));codec_stream_key=mfq_current_cuda_stream();
        const auto update=option("TYLOINFER_QSA_TQ_UPDATE_TOKENS",128);
        if(update!=128 && update!=256)throw std::invalid_argument("adaptive QSA update interval must be 128 or 256");
        interval=unsigned(update);
        byte_budget=option("TYLOINFER_QSA_TQ_BUDGET_BYTES",0);
        const auto* path=std::getenv("TYLOINFER_QSA_TQ_DIRECTORY");
        if(ssd_cache && (!path || !*path))throw std::invalid_argument("adaptive QSA requires TYLOINFER_QSA_TQ_DIRECTORY for its original KV cache");
        static std::atomic<std::uint64_t> next{0};
        if(ssd_cache)directory=(std::filesystem::path(path)/("sequence-"+std::to_string(++next))).string();
        ks=signs(padded,false,options);vs=signs(padded,true,options);
        std::array<__half,16> rel;for(int i=0;i<16;++i)rel[i]=__float2half_rn(std::exp2(-2.f+4.f*i/15.f));
        relative=upload(rel.data(),sizeof(rel),options);
        // An exclusive SSD cache may never own a sequence when all requests
        // are batched. Materialize its descriptor tables only when needed.
        if(!ssd_cache || !ram_bits) {
            views=qsa_descriptor_table(capacity,sizeof(tq::TqBlockView),options,host_metadata);
            hits=tb::zeros({capacity},options.dtype(tb::kFloat32));
            if(!ram_bits)mass=tb::zeros_like(hits);
        }
        canonical_bytes=ram_bits==16?std::size_t(rows)*512:((3*rows+3)&~3)+256*rows;
        tail_storage_k=tb::empty({1,4,kv_heads,width},options.dtype(tb::kFloat32));tail_storage_v=tb::empty_like(tail_storage_k);
        tail_storage_index=tb::empty({1,4,index_width},options.dtype(tb::kFloat32));
        rotated_tail_k=tb::empty({4,kv_heads,padded},options.dtype(tb::kFloat32));rotated_tail_v=tb::empty_like(rotated_tail_k);
        pending_modes_k=tb::empty({rows,256},options.dtype(tb::kFloat32));pending_modes_v=tb::empty_like(pending_modes_k);
        pending_k=tb::empty({rows,256},options.dtype(tb::kFloat16));pending_v=tb::empty_like(pending_k);
        initialize_cache();
    }
    struct IndexStorage {
        tb::Tensor codes,scales,zeros;
        IndexPointers pointers;
    };
    IndexStorage new_indexer_storage(bool host) {
        RequestStateAllocationScope resident(!host,RequestStateAllocationScope::Kind::Indexer);
        const auto storage=host?options.device(tb::kCPU).pinned_memory(true):options;
        IndexStorage index{tb::empty({capacity,index_width},storage.dtype(tb::kInt16)),
            tb::empty({capacity},storage.dtype(tb::kFloat16)),tb::empty({capacity},storage.dtype(tb::kInt16)),{}};
        const auto alias=[&](const tb::Tensor& tensor) {
            auto* pointer=tensor.data_ptr();
            if(host)check(cudaHostGetDevicePointer(&pointer,tensor.data_ptr(),0));
            return pointer;
        };
        index.pointers={static_cast<std::int16_t*>(alias(index.codes)),static_cast<std::uint16_t*>(alias(index.scales)),
            static_cast<std::int16_t*>(alias(index.zeros))};
        return index;
    }
    void bind_indexer_storage(IndexStorage index,bool host) {
        device_index=index.pointers;
        index_codes=std::move(index.codes);
        index_scales=std::move(index.scales);
        index_zero=std::move(index.zeros);
        host_indexer=host;
    }
    void allocate_indexer() {
        if(index_codes.defined())return;
        try {bind_indexer_storage(new_indexer_storage(host_indexer),host_indexer);}
        catch(const MfqResourceExhausted&) {
            const auto* placement=std::getenv("TYLOINFER_QSA_CPU_INDEXER");
            if(host_indexer || (placement && *placement))throw;
            const auto bytes=std::size_t(capacity)*(std::size_t(index_width)+2)*sizeof(std::int16_t);
            if(!ssd_cache && (!ram || !ram->restore_can_place_in_ram(bytes)))throw;
            int mapped=0;
            check(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,device));
            if(!mapped)throw;
            (void)cudaGetLastError();
            if(!ssd_cache)default_context(device)->reclaim_idle_memory();
            bind_indexer_storage(new_indexer_storage(true),true);
            if(std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))std::fprintf(stderr,
                "native_qsa_indexer_placement tier=host reason=device_allocation bytes=%zu\n",
                index_codes.nbytes()+index_scales.nbytes()+index_zero.nbytes());
        }
        if(host_indexer && std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
            "native_qsa_cpu_indexer allocation_bytes=%zu capacity=%d width=%d heads=%d\n",
            index_codes.nbytes()+index_scales.nbytes()+index_zero.nbytes(),capacity,index_width,index_heads);
    }
    ~Impl() {
        (void)cudaSetDevice(device);
        if(history_transfer.pending){(void)cudaEventSynchronize(history_transfer.done);history_transfer.pending=false;}
        ram.reset();cache.reset();(void)cudaStreamSynchronize(mfq_current_cuda_stream());
        for(auto& old:retired)if(old.done)(void)cudaEventDestroy(old.done);
        for(auto& upload:uploads)if(upload.done)(void)cudaEventDestroy(upload.done);
    }
    void initialize_cache() {
        if(ram_bits) {
            ram=std::make_unique<QsaRamCache>(capacity,rows,(budget+6)/4,ram_bits,options,views,ssd_cache,directory+"/ram-kv",
                [this](const auto& images,const auto& precision){return reencode(images,precision);},adaptive_precision,host_metadata,use_spare_vram);
            mfq::cache::QsaKvStoreConfig index;index.budget_bytes=index.buffer_bytes=4ULL<<20;index.directory=directory+"/index-tail";
            if(ssd_cache)raw_index_store=std::make_shared<mfq::cache::QsaKvStore>(index);
            return;
        }
        tq::AdaptiveCacheConfig config;config.directory=directory+"/kv";config.rows_per_kind=rows;
        config.ssd_cache=ssd_cache;
        config.tq_state_bits=14;config.policy.update_tokens=interval;
        cache=std::make_unique<tq::AdaptiveCache>(config,tq::AdaptiveCache::ReencodeBatch(
            [this](const auto& images,const auto& precision){return reencode(images,precision);}),
            [this](const auto& batch){
                std::lock_guard<std::mutex> lock(queue_mutex);
                for(const auto& change:batch) {
                    if(change.tier==5 && change.block>=resident_append_begin && change.block<resident_append_end)continue;
                    queued.push_back(change);
                }
            });
        mfq::cache::QsaKvStoreConfig index;index.budget_bytes=index.buffer_bytes=4ULL<<20;index.directory=directory+"/index-tail";
        if(ssd_cache)raw_index_store=std::make_shared<mfq::cache::QsaKvStore>(index);
    }
    void ensure_descriptors() {
        if(views.defined())return;
        // These tables remain live throughout decode, after prefill returns its workspace.
        const ResidentAllocationScope resident;
        auto table=qsa_descriptor_table(capacity,sizeof(tq::TqBlockView),options,host_metadata);
        auto new_hits=tb::zeros({capacity},options.dtype(tb::kFloat32));
        ram->bind_descriptors(table);
        views=std::move(table);hits=std::move(new_hits);
    }
    float* mass_data() const {return mass.defined()?mass.data_ptr<float>():nullptr;}
    std::size_t index_archive_bytes()const {return std::size_t(index_width)*8+(ram?std::size_t(index_width+2)*2:0);}
    void save_index(std::vector<std::uint8_t> bytes) {
        auto image=std::make_shared<QsaIndexArchive>();
        if(ssd_cache){image->store=raw_index_store;image->block=raw_index_store->write(std::move(bytes));}
        else {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            auto resident=std::make_unique<const std::vector<std::uint8_t>>(std::move(bytes));
            const auto capacity=resident->capacity();charge_tensor_host_bytes(capacity);
            image->resident=std::shared_ptr<const std::vector<std::uint8_t>>(resident.release(),
                [capacity](const std::vector<std::uint8_t>* value){delete value;tensor_host_bytes.fetch_sub(capacity);});
#else
            image->resident=std::make_shared<const std::vector<std::uint8_t>>(std::move(bytes));
#endif
        }
        index_archive.push_back(std::move(image));
    }
    std::vector<tq::MicroblockImage> reencode(const std::vector<tq::MicroblockImage>& input,const std::vector<tq::Precision>& precision) {
        if(input.size()!=precision.size() || input.size()>std::size_t(INT_MAX/2))
            throw std::invalid_argument("QSA encoder precision count mismatch");
        if(input.empty())return {};
        check(cudaSetDevice(device));
        tq::TqEncodePlan checked_plan;
        const auto batch=plan_codec_batch(input.size(),rows,checked_plan);
        if(batch<input.size()) {
            std::vector<tq::MicroblockImage> result;result.reserve(input.size());
            for(std::size_t begin=0;begin<input.size();begin+=batch) {
                const auto end=std::min(input.size(),begin+batch);
                auto part=reencode({input.begin()+begin,input.begin()+end},{precision.begin()+begin,precision.begin()+end});
                for(auto& image:part)result.push_back(std::move(image));
            }
            return result;
        }
        std::call_once(bind_encoder,[this]{encoder=ram_bits?shared_codec_arena(device,codec_stream_key):std::make_shared<TqCodecArena>();});
        auto& arena=*encoder;std::lock_guard<std::mutex> encoder_guard(arena.mutex);
        auto& encoder_stream=arena.stream;auto& encoder_owner=arena.owner;
        auto& source=arena.source;auto& source_views=arena.source_views;auto& decoded=arena.decoded;
        auto& encoded=arena.encoded;auto& sizes=arena.sizes;auto& flags=arena.flags;
        auto& status=arena.status;auto& targets=arena.targets;auto& workspace=arena.workspace;
        auto& encoder_input=arena.input;auto& encoder_output=arena.output;
        check(cudaSetDevice(device));
        // Short contexts use only canonical Q8. Create an encoder stream when
        // the background worker first needs secondary quantization.
        std::call_once(arena.initialized,[&]{
            int least=0,greatest=0;check(cudaDeviceGetStreamPriorityRange(&least,&greatest));
            check(cudaStreamCreateWithPriority(&encoder_stream,cudaStreamNonBlocking,least));
            encoder_owner.value=encoder_stream;
        });
        const int n=int(input.size()),count=2*n;
        const auto stride=codec_output_stride(rows);
        std::size_t bytes=0;std::vector<std::size_t> offsets;
        for(int kind=0;kind<2;++kind)for(const auto& image:input) {
            const auto& src=kind?image.value:image.key;offsets.push_back(bytes);bytes+=src.size();
        }
        const TqCodecLayout layout(checked_plan,bytes);
        source.reserve(layout.source);source_views.reserve(layout.views);
        decoded.reserve(layout.decoded);encoded.reserve(layout.encoded);
        sizes.reserve(layout.block_sizes);flags.reserve(layout.block_sizes);status.reserve(sizeof(int));targets.reserve(layout.block_sizes);
        const auto records_at=(bytes+15)&~std::size_t(15),budgets_at=records_at+n*sizeof(tq::TqBlockView);
        encoder_input.reserve(budgets_at+count*4);
        auto* records=encoder_input.as<tq::TqBlockView>(records_at);auto* budgets=encoder_input.as<int>(budgets_at);
        for(int kind=0;kind<2;++kind)for(int i=0;i<n;++i) {
            const auto& image=kind?input[i].value:input[i].key;
            std::memcpy(encoder_input.as<std::uint8_t>(offsets[kind*n+i]),image.data(),image.size());
        }
        for(int i=0;i<n;++i) {
            records[i]={source.as<std::uint8_t>(offsets[i]),source.as<std::uint8_t>(offsets[n+i]),
                tq::tq_precision_flags(input[i].key_bits,input[i].key_state_bits),tq::tq_precision_flags(input[i].value_bits,input[i].value_state_bits)};
            budgets[i]=rows*precision[i].key_bits;budgets[n+i]=rows*precision[i].value_bits;
        }
        check(cudaMemcpyAsync(source.pointer,encoder_input.pointer,bytes,cudaMemcpyHostToDevice,encoder_stream));
        check(cudaMemcpyAsync(source_views.pointer,records,n*sizeof(records[0]),cudaMemcpyHostToDevice,encoder_stream));
        check(cudaMemcpyAsync(targets.pointer,budgets,count*4,cudaMemcpyHostToDevice,encoder_stream));
        check(tq::tq_decode_modes(source_views.as<tq::TqBlockView>(),decoded.as<float>(),decoded.as<float>(std::size_t(n)*rows*256*4),
            relative.data_ptr<std::uint16_t>(),n,kv_heads,padded,encoder_stream));
        auto plan=checked_plan;
        if(ram) {
            const auto* explicit_limit=std::getenv("TYLOINFER_QSA_BACKGROUND_ROWS");
            // The low-priority stream and actual workspace/kernel resources
            // bound asynchronous fitting. Artificially limiting it to a few
            // persistent CTAs leaves codecs queued until speculative rollback.
            const bool urgent=ram->encoding_urgent();
            const auto limit=option("TYLOINFER_QSA_BACKGROUND_ROWS",0);
            if(limit>INT_MAX)throw std::invalid_argument("QSA encoder row limit exceeds 32 bits");
            if(limit)check(tq::tq_limit_encode_waves(&plan,int(limit)));
            if(std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
                static thread_local unsigned reported=0;const auto category=urgent?1u:2u;
                if(!(reported&category)){reported|=category;
                    std::fprintf(stderr,"native_qsa_ram_codec actual_match=1 urgent=%d active_row_limit=%zu explicit_limit=%d\n",
                        int(urgent),limit,int(explicit_limit && *explicit_limit));}
            }
        }
        workspace.reserve(plan.workspace_bytes);
        check(tq::tq_encode_modes(plan,decoded.as<float>(),targets.as<int>(),0,1.1f,encoded.as<std::uint8_t>(),stride,
            sizes.as<std::uint32_t>(),flags.as<std::uint32_t>(),status.as<int>(),workspace.pointer,workspace.bytes,encoder_stream));
        const auto packed_bytes=std::size_t(count)*stride,lengths_at=(packed_bytes+15)&~std::size_t(15);
        const auto tags_at=lengths_at+count*4,error_at=tags_at+count*4;
        encoder_output.reserve(error_at+4);
        if(std::getenv("TYLOINFER_TRACE_QSA_TQ") && arena.reported_bytes!=arena.device_bytes()) {
            arena.reported_bytes=arena.device_bytes();
            std::fprintf(stderr,"native_qsa_codec_workspace actual_match=1 shared_stream_arena=%d device_bytes=%zu host_bytes=%zu\n",
                int(ram_bits!=0),arena.reported_bytes,encoder_input.bytes+encoder_output.bytes);
        }
        auto* packed=encoder_output.as<std::uint8_t>();auto* lengths=encoder_output.as<std::uint32_t>(lengths_at);
        auto* tags=encoder_output.as<std::uint32_t>(tags_at);auto* error=encoder_output.as<int>(error_at);
        // Pageable D2H may wait inside the driver's context lock. A model
        // graph can meanwhile wait for a host-served expert DMA that needs
        // that same lock. Pinned endpoints let every copy submit before the
        // encoder waits, preserving progress on both streams.
        check(cudaMemcpyAsync(packed,encoded.pointer,packed_bytes,cudaMemcpyDeviceToHost,encoder_stream));
        check(cudaMemcpyAsync(lengths,sizes.pointer,count*4,cudaMemcpyDeviceToHost,encoder_stream));
        check(cudaMemcpyAsync(tags,flags.pointer,count*4,cudaMemcpyDeviceToHost,encoder_stream));
        check(cudaMemcpyAsync(error,status.pointer,4,cudaMemcpyDeviceToHost,encoder_stream));
        check(cudaStreamSynchronize(encoder_stream));check(*error);
        std::vector<tq::MicroblockImage> result(n);
        for(int i=0;i<n;++i) {
            if(lengths[i]>stride || lengths[n+i]>stride)throw std::runtime_error("QSA encoder returned an invalid packed length");
            auto& dst=result[i];dst.key.assign(packed+std::size_t(i)*stride,packed+std::size_t(i)*stride+lengths[i]);
            dst.value.assign(packed+std::size_t(n+i)*stride,packed+std::size_t(n+i)*stride+lengths[n+i]);
            dst.key_bits=std::uint8_t(tags[i]);dst.value_bits=std::uint8_t(tags[n+i]);dst.key_state_bits=dst.value_state_bits=14;
        }
        return result;
    }
    void retire(std::vector<std::shared_ptr<ImageBatch>> previous) {
        if(previous.empty())return;Retired old;old.images=std::move(previous);
        check(cudaEventCreateWithFlags(&old.done,cudaEventDisableTiming));check(cudaEventRecord(old.done,mfq_current_cuda_stream()));retired.push_back(std::move(old));
    }
    void ensure_owner_slots(std::size_t count) {
        if(count>std::size_t(capacity))throw std::out_of_range("QSA append owner range exceeds capacity");
        if(owners.size()<count)owners.resize(count);
    }
    void retain_images(const std::shared_ptr<ImageBatch>& group,const std::vector<std::size_t>& ids) {
        if(!ids.empty())ensure_owner_slots(*std::max_element(ids.begin(),ids.end())+1);
        std::vector<std::shared_ptr<ImageBatch>> previous;
        previous.reserve(ids.size());
        for(const auto id:ids) {
            if(owners[id])previous.push_back(std::move(owners[id]));owners[id]=group;
        }
        retire(std::move(previous));
    }
    void publish_images(const std::shared_ptr<ImageBatch>& group,const std::vector<std::size_t>& ids) {
        bool contiguous=!ids.empty();
        for(std::size_t i=1;i<ids.size();++i)contiguous&=ids[i]==ids[0]+i;
        if(contiguous)check(cudaMemcpyAsync(views.data_ptr<std::uint8_t>()+ids[0]*sizeof(tq::TqBlockView),
            group->records.data(),ids.size()*sizeof(tq::TqBlockView),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
        for(std::size_t i=0;i<ids.size();++i) {
            const auto id=ids[i];const auto& view=group->records[i];
            if(!contiguous)check(cudaMemcpyAsync(views.data_ptr<std::uint8_t>()+id*sizeof(view),&view,sizeof(view),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
        }
        Upload submitted;submitted.image=group;check(cudaEventCreateWithFlags(&submitted.done,cudaEventDisableTiming));
        check(cudaEventRecord(submitted.done,mfq_current_cuda_stream()));uploads.push_back(std::move(submitted));
        retain_images(group,ids);
    }
    AttentionWorkspace& prepare_workspace(int queries,int pools,int parts,bool pending,bool gpu_scores,
        bool share_pending=false) {
        const bool small=queries<=8;
        const bool decode=small && append_small;
        // Captured GPU-only calls retain these addresses across eager prefill
        // and multi-row verification, which may resize the other arenas.
        auto& current=pending?graph_workspace(queries):decode?decode_attention:prefill_attention;
        const auto stream=share_pending?mfq_current_cuda_stream():nullptr;
        if(current.queries>=queries && current.pools>=pools && (pending?current.parts==parts:current.parts>=parts) &&
                (!gpu_scores || current.ranked.defined()) &&
                (!share_pending || (shared_pending_attention && pending_attention_stream==stream)))return current;
        using SharedKey=std::tuple<int,cudaStream_t,int,int,int,int,int,int,int,int,bool,bool>;
        static thread_local std::map<SharedKey,std::weak_ptr<const AttentionWorkspace>> shared;
        std::weak_ptr<const AttentionWorkspace>* slot=nullptr;
        if(share_pending) {
            // Decode serializes the QSA invocations on this stream.
            // The tensors are transient, but the wrapper stays per layer so
            // each graph task retains its own mapped controls and CPU query.
            slot=&shared[{device,stream,queries,capacity,heads,padded,parts,budget,index_heads,index_width,gpu_scores,bool(ram)}];
            if(auto workspace=slot->lock()) {
                current=*workspace;
                shared_pending_attention=std::move(workspace);
                pending_attention_stream=stream;
                return current;
            }
        }
        ResidentAllocationScope resident(pending || decode);
        // Publish a new shared workspace only after all allocations succeed.
        AttentionWorkspace fresh;
        auto& a=share_pending?fresh:current;
        const int count=(budget+6)/4;
        int p=small?capacity:std::max(1,a.pools);
        while(p<pools)p=std::min(capacity,p>capacity/2?capacity:2*p);
        const auto score_budget=option("TYLOINFER_QSA_PREFILL_SCORE_BYTES",64ULL<<20);
        const auto score_queries=std::max<std::size_t>(16,score_budget/(sizeof(float)*std::size_t(p)));
        const int q=small?std::max(queries,a.queries):std::max(queries,int(std::min<std::size_t>(a.queries,score_queries)));
        a.queries=0;
        const auto ensure=[&](tb::Tensor& buffer,int64_t size,const tb::TensorOptions& storage) {
            if(!pending) {
                if(buffer.defined() && buffer.numel()>=size)return;
                // The previous tile has already queued its consumers on the
                // same stream. Release its pages before growing this tensor.
                buffer={};
            }
            buffer=tb::empty({size},storage);
        };
        const auto f=options.dtype(tb::kFloat32);
        if(gpu_scores) {
            const auto indexer_bytes=tq::indexer_affine16_score_scratch(q,index_heads,index_width);
            if(indexer_bytes)ensure(a.indexer,int64_t(indexer_bytes),options.dtype(tb::kUInt8));
            else a.indexer={};
            ensure(a.ranked,int64_t(q)*p,f);
        }else {a.indexer={};a.ranked={};}
        ensure(a.blocks,int64_t(q)*count*int64_t(sizeof(tq::SelectedBlock)),options.dtype(tb::kUInt8));
        ensure(a.rotated,int64_t(q)*heads*padded,f);
        auto partial_elements=int64_t(q)*heads*parts*padded;
        if(!small && padded==256) {
            const auto bytes=tq::tq_prefill_attention_scratch(q,heads,4*p);
            partial_elements=std::max(partial_elements,int64_t((bytes+3)/4));
        }
        ensure(a.partial,partial_elements,f);
        ensure(a.stats,int64_t(q)*heads*parts*2,f);
        if(!ram) {
            ensure(a.logmass,int64_t(q)*heads*count,f);
            ensure(a.norm,int64_t(q)*heads,f);
        }
        a.queries=q;a.pools=p;a.parts=parts;
        if(slot) {
            shared_pending_attention=std::make_shared<AttentionWorkspace>(std::move(fresh));
            current=*shared_pending_attention;
            pending_attention_stream=stream;
            *slot=shared_pending_attention;
            if(std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH")) {
                std::size_t bytes=0;
                for(const auto* tensor:{&current.ranked,&current.blocks,&current.rotated,&current.partial,
                        &current.stats,&current.logmass,&current.norm,&current.indexer})
                    if(tensor->defined())bytes+=tensor->nbytes();
                std::fprintf(stderr,"native_qsa_shared_attention_workspace gpu_scores=%d workspace_bytes=%zu\n",
                    int(gpu_scores),bytes);
            }
        }
        return current;
    }
    void indexer_scores(const tb::Tensor& query,const tb::Tensor& positions,tb::Tensor output,
                         int queries,int pools,AttentionWorkspace& scratch,bool batched_prefill=false,
                         const IndexPointers* staged=nullptr) {
        const auto& index=staged?*staged:device_index;
        // Small forward calls preserve decode numerics, including eager warmup.
        // Batched prefill retains its caller-owned integer query workspace.
        if(batched_prefill) {
            check(tq::indexer_affine16_scores(query.data_ptr<float>(),index.codes,
                index.scales,index.zero,positions.data_ptr<int64_t>(),
                output.data_ptr<float>(),queries,index_heads,index_width,pools,mfq_current_cuda_stream(),
                scratch.indexer.defined()?scratch.indexer.data_ptr():nullptr,
                scratch.indexer.defined()?scratch.indexer.nbytes():0));
            return;
        }
        check(tq::indexer_affine16_scores(query.data_ptr<float>(),index.codes,
            index.scales,index.zero,positions.data_ptr<int64_t>(),
            output.data_ptr<float>(),queries,index_heads,index_width,pools,mfq_current_cuda_stream()));
    }
    tb::Tensor select_host_indexer(const tb::Tensor& query,const tb::Tensor& positions,int queries,int pools) {
        using Clock=std::chrono::steady_clock;
        const bool trace=std::getenv("TYLOINFER_TRACE_QSA_TQ")!=nullptr;
        const auto begin=trace?Clock::now():Clock::time_point{};
        const auto host=options.device(tb::kCPU);
        auto q=query.to(host.dtype(tb::kFloat32));
        auto pos=positions.to(host.dtype(tb::kInt64));
        auto blocks=tb::empty({1,queries,(budget+6)/4,2},host.dtype(tb::kInt32));
        const auto ready=trace?Clock::now():Clock::time_point{};
        mfq::cpu::qsa_indexer_topk(q.data_ptr<float>(),pos.data_ptr<std::int64_t>(),queries,index_heads,index_width,
            index_codes.data_ptr<std::int16_t>(),index_scales.data_ptr<std::uint16_t>(),index_zero.data_ptr<std::int16_t>(),
            pools,budget,mfq_get_num_threads(),blocks.data_ptr<std::int32_t>());
        const auto ranked=trace?Clock::now():Clock::time_point{};
        auto result=blocks.to(options.dtype(tb::kInt32));
        if(trace) {
            ++host_index_calls;
            host_index_ms+=std::chrono::duration<double,std::milli>(ranked-ready).count();
            host_index_transfer_ms+=std::chrono::duration<double,std::milli>((ready-begin)+(Clock::now()-ranked)).count();
        }
        return result;
    }
    tb::Tensor select_graph_indexer(const tb::Tensor& query,const tb::Tensor& positions,int queries,AttentionWorkspace& scratch,
                                   bool mark_scores=false) {
        auto* window=DecodeWindow::recording();
        if(!host_indexer) {
            auto ranked=scratch.ranked.narrow(0,0,int64_t(queries)*capacity).reshape({1,queries,capacity});
            indexer_scores(query,positions,ranked,queries,capacity,scratch);
            if(mark_scores && window)window->mark_current("qsa_cache_scores_end");
            return mfq_qwen4_exp::live_topk_selected_blocks(ranked,positions,4,std::min(budget/4,capacity),budget);
        }
        if(!window)throw std::logic_error("CPU QSA graph requires a decode window");
        struct Buffers {
            tb::Tensor query,positions,selected,flags;
            std::uint32_t* mapped_flags=nullptr;
            float* mapped_query=nullptr;
            std::int64_t* mapped_positions=nullptr;
            bool ready=false;
            bool profile=false;
            std::uint64_t score_ns=0,score_calls=0;
            ~Buffers() {
                if(profile && score_calls)std::fprintf(stderr,"native_qsa_cpu_graph_indexer calls=%llu score_ns=%llu\n",
                    static_cast<unsigned long long>(score_calls),static_cast<unsigned long long>(score_ns));
            }
            void reset() {
                auto* control=flags.data_ptr<std::uint32_t>();
                control[0]=control[1]=0;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                ready=false;
            }
            void publish() noexcept {
                ready=true;
                publish_mapped_flag(flags.data_ptr<std::uint32_t>()+1);
            }
            void cancel() noexcept {
                if(ready)return;
                std::memset(selected.data_ptr(),-1,selected.nbytes());
                publish();
            }
        };
        const auto host=options.device(tb::kCPU).pinned_memory(true);
        const auto buffers=window->workspace<Buffers>(&scratch,[&] {
            auto owned=std::make_shared<Buffers>(Buffers{
                tb::empty({query.numel()},host.dtype(tb::kFloat32)),
                tb::empty({queries},host.dtype(tb::kInt64)),
                tb::empty({1,queries,(budget+6)/4,2},host.dtype(tb::kInt32)),{}});
            owned->profile=std::getenv("TYLOINFER_TRACE_NATIVE_CUDA_GRAPH");
            const auto* task=std::getenv("TYLOINFER_QSA_CPU_INDEXER_GRAPH_TASK");
            if(window->allows_qsa_indexer_graph_task() && (!task || task[0]!='0')) {
                try {
                    owned->flags=tb::empty({2},host.dtype(tb::kInt32));
                    const auto alias=[](const tb::Tensor& tensor)->void* {
                        void* address=nullptr;
                        const auto status=cudaHostGetDevicePointer(&address,tensor.data_ptr(),0);
                        if(status==cudaSuccess)return address;
                        if(status!=cudaErrorInvalidValue && status!=cudaErrorNotSupported && status!=cudaErrorMemoryAllocation)
                            check(status);
                        (void)cudaGetLastError();
                        return nullptr;
                    };
                    owned->mapped_flags=static_cast<std::uint32_t*>(alias(owned->flags));
                    if(owned->mapped_flags) {
                        owned->mapped_query=static_cast<float*>(alias(owned->query));
                        owned->mapped_positions=static_cast<std::int64_t*>(alias(owned->positions));
                    }
                } catch(const MfqResourceExhausted&) {
                    (void)cudaGetLastError();
                }
                if(!owned->mapped_flags)owned->flags={};
            }
            return owned;
        });
        const auto stream=mfq_current_cuda_stream();
        auto selected=scratch.blocks.narrow(0,0,buffers->selected.nbytes());
        const bool mapped_request=buffers->mapped_query && buffers->mapped_positions;
        if(!mapped_request) {
            check(cudaMemcpyAsync(buffers->query.data_ptr(),query.data_ptr(),query.nbytes(),cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(buffers->positions.data_ptr(),positions.data_ptr(),std::size_t(queries)*sizeof(std::int64_t),cudaMemcpyDeviceToHost,stream));
        }
        const auto score=[this,buffers,queries] {
            const bool trace=std::getenv("TYLOINFER_TRACE_QSA_TQ")!=nullptr;
            const auto begin=trace || buffers->profile?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
            mfq::cpu::qsa_indexer_topk(buffers->query.data_ptr<float>(),buffers->positions.data_ptr<std::int64_t>(),
                queries,index_heads,index_width,index_codes.data_ptr<std::int16_t>(),index_scales.data_ptr<std::uint16_t>(),
                index_zero.data_ptr<std::int16_t>(),capacity,budget,mfq_get_num_threads(),buffers->selected.data_ptr<std::int32_t>());
            if(trace || buffers->profile) {
                const auto elapsed=std::chrono::steady_clock::now()-begin;
                if(buffers->profile) {
                    ++buffers->score_calls;
                    buffers->score_ns+=std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
                }
                if(trace) {
                    ++host_index_calls;
                    host_index_ms+=std::chrono::duration<double,std::milli>(elapsed).count();
                }
            }
        };
        cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(stream,&capture));
        if(buffers->mapped_flags) {
            const bool warm=capture==cudaStreamCaptureStatusNone;
            if(warm)buffers->reset();
            const auto serve=[score,buffers,stream] {
                wait_route_publication(buffers->flags.data_ptr<std::uint32_t>(),stream);
                score();
                buffers->publish();
            };
            window->enroll({buffers.get(),[buffers]{buffers->reset();},serve,{},[buffers]{buffers->cancel();}});
            if(mapped_request) {
                check(tq::tq_publish_indexer_request(query.data_ptr<float>(),positions.data_ptr<std::int64_t>(),
                    query.numel(),queries,buffers->mapped_query,buffers->mapped_positions,buffers->mapped_flags,stream));
            } else {
                signal_mapped_flag(buffers->mapped_flags,stream);
                check(cudaGetLastError());
            }
            if(warm)serve();
            wait_mapped_flag(buffers->mapped_flags+1,stream);
            check(cudaGetLastError());
            if(std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH")) {
                static std::atomic<bool> reported{false};
                if(!reported.exchange(true))std::fprintf(stderr,"native_qsa_cpu_indexer_graph_task actual_match=1 mapped_request=%d\n",int(mapped_request));
            }
        } else {
            window->host_barrier(score);
            if(capture==cudaStreamCaptureStatusNone){check(cudaStreamSynchronize(stream));score();}
        }
        check(cudaMemcpyAsync(selected.data_ptr(),buffers->selected.data_ptr(),buffers->selected.nbytes(),cudaMemcpyHostToDevice,stream));
        if(mark_scores)window->mark_current("qsa_cache_scores_end");
        return selected;
    }
    void drain() {
        for(auto it=uploads.begin();it!=uploads.end();) {
            const auto result=cudaEventQuery(it->done);
            if(result==cudaSuccess) {
                check(cudaEventDestroy(it->done));std::vector<std::uint8_t>().swap(it->image->host);
                std::vector<tq::TqBlockView>().swap(it->image->records);it=uploads.erase(it);
            }else {if(result!=cudaErrorNotReady)check(result);++it;}
        }
        for(auto it=retired.begin();it!=retired.end();) {
            const auto result=cudaEventQuery(it->done);
            if(result==cudaSuccess){check(cudaEventDestroy(it->done));it=retired.erase(it);}
            else {if(result!=cudaErrorNotReady)check(result);++it;}
        }
        if(ram){ram->drain();return;}
        std::vector<tq::BlockReplacement> batch;
        {std::lock_guard<std::mutex> lock(queue_mutex);batch.swap(queued);}
        if(batch.empty())return;
        auto group=std::make_shared<ImageBatch>();std::vector<std::size_t> key_offsets,value_offsets;
        for(const auto& entry:batch) {
            key_offsets.push_back(group->host.size());group->host.insert(group->host.end(),entry.image->key.begin(),entry.image->key.end());
            value_offsets.push_back(group->host.size());group->host.insert(group->host.end(),entry.image->value.begin(),entry.image->value.end());
        }
        {
            const ResidentAllocationScope resident;
            group->payload=upload(group->host.data(),group->host.size(),options);
        }
        group->records.resize(batch.size());
        std::vector<std::size_t> ids;
        for(std::size_t i=0;i<batch.size();++i) {
            const auto& entry=batch[i];const auto id=entry.block;
            ids.push_back(id);
            auto& view=group->records[i];view={group->payload.data_ptr<std::uint8_t>()+key_offsets[i],group->payload.data_ptr<std::uint8_t>()+value_offsets[i],
                tq::tq_precision_flags(entry.image->key_bits,entry.image->key_state_bits),tq::tq_precision_flags(entry.image->value_bits,entry.image->value_state_bits)};
        }
        // The upload owner retains every host descriptor until its event.
        publish_images(group,ids);
    }
    void publish_pending(const tb::Tensor& k,const tb::Tensor& v,int64_t id) {
        pending_k=k;pending_v=v;
        check(tq::tq_publish_contiguous_views({k.data_ptr<std::uint8_t>(),v.data_ptr<std::uint8_t>(),0,0},
            views.data_ptr<tq::TqBlockView>()+id,int(k.nbytes()),1,mfq_current_cuda_stream()));
    }
    bool graph_host_boundary=false,graph_ssd_task=false;
    tb::Tensor stage_pending(const tb::Tensor& selected,int queries,int count,const tb::Tensor& positions) {
        if(!ram)return views;
        if(graph_ssd_task)return ram->stage_graph_task(selected,queries,count,views,positions,queries);
        // Top-k emits each complete block once, followed by the causal tail.
        if(!graph_host_boundary)
            return ram->stage(selected,queries,count,views,positions,tokens,queries,false,queries==1);
        auto* window=DecodeWindow::recording();
        if(!window)throw std::logic_error("SSD QSA graph requires its decode window host boundary");
        cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture));
        const bool selection_in_graph=ram->collect_graph_selection(selected,queries,count);
        // Only the selected RAM/SSD images need a host boundary. Append,
        // index scoring, selection and attention keep their captured GPU work.
        // Read the live epoch when replaying, rather than retaining the epoch
        // of the first capture in LRU updates.
        window->host_barrier([this,selected,queries,count,positions,selection_in_graph] {
            ram->stage(selected,queries,count,views,positions,tokens,queries,selection_in_graph);
        });
        if(capture==cudaStreamCaptureStatusNone) {
            check(cudaStreamSynchronize(mfq_current_cuda_stream()));
            ram->stage(selected,queries,count,views,positions,tokens,queries,selection_in_graph);
        }
        return ram->prepared_views();
    }
    tb::Tensor forward_pending_batch(const tb::Tensor& query,const tb::Tensor& key,const tb::Tensor& value,const tb::Tensor& iq,
        const tb::Tensor& raw,const tb::Tensor& positions,const tb::Tensor& gate,tb::Tensor output,
        const QsaTqCache::Pool& pool,const tb::Tensor& absolute_positions) {
        const int n=int(key.size(1));
        if(!pending_graph || !batch_prepared || n!=graph_queries || !pool)
            throw std::logic_error("QSA batch graph was not prepared");
        const auto convert=[](const tb::Tensor& x) {
            auto y=(x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32)?x:x.to(tb::kFloat32);
            return y.stride(y.dim()-1)==1?y:y.contiguous();
        };
        auto k=convert(key),v=convert(value),r=convert(raw);
        auto& batch=batch_append.at(std::size_t(n-2));
        const int staged=int(batch.key.size(1)),blocks=staged/4;
        auto absolute=absolute_positions.defined()?absolute_positions:positions;
        if(absolute.dim()==1)absolute=absolute.reshape({1,n});
        tb::Tensor starts;
        tq::GraphPoolStarts pool_starts{};
        if(absolute.is_cuda() && absolute.device()==positions.device() && absolute.is_contiguous() &&
                absolute.dim()>0 && absolute.size(-1)==n && absolute.numel()>=n && absolute.numel()/n<=INT_MAX &&
                (absolute.scalar_type()==tb::kInt64 || absolute.scalar_type()==tb::kInt32)) {
            auto shape=absolute.sizes().vec();shape.back()=blocks;
            starts=tb::empty(shape,positions.options());
            pool_starts={absolute.data_ptr(),starts.data_ptr<int64_t>(),int(absolute.numel()/n),absolute.scalar_type()==tb::kInt32};
        }
        check(tq::tq_stage_graph_batch(k.data_ptr(),v.data_ptr(),r.data_ptr(),k.scalar_type()==tb::kFloat16,v.scalar_type()==tb::kFloat16,r.scalar_type()==tb::kFloat16,
            seeded_tail(),positions.data_ptr<int64_t>(),batch.key.data_ptr<float>(),batch.value.data_ptr<float>(),batch.index.data_ptr<float>(),
            n,staged,kv_heads,width,index_width,k.stride(1),k.stride(2),v.stride(1),v.stride(2),r.stride(1),
            mfq_current_cuda_stream(),starts.defined()?&pool_starts:nullptr));
        const tq::Affine8Pending pending{batch.fp16_key.data_ptr<std::uint16_t>(),batch.fp16_value.data_ptr<std::uint16_t>(),
            batch.rotated_key.data_ptr<float>(),batch.rotated_value.data_ptr<float>()};
        const auto* canonical_k=reinterpret_cast<const std::uint8_t*>(pending.keys);
        const auto* canonical_v=reinterpret_cast<const std::uint8_t*>(pending.values);
        if(ram_bits!=16) {
            auto* packed=batch.packed.data_ptr<std::uint8_t>();
            check(tq::tq_prepare_affine8(batch.key.data_ptr(),batch.value.data_ptr(),false,ks.data_ptr<float>(),vs.data_ptr<float>(),
                packed,packed+blocks*canonical_bytes,staged,kv_heads,width,padded,mfq_current_cuda_stream(),&pending));
            canonical_k=packed;canonical_v=packed+blocks*canonical_bytes;
        } else {
            check(tq::tq_prepare_modes(batch.key.data_ptr(),batch.value.data_ptr(),false,ks.data_ptr<float>(),vs.data_ptr<float>(),
                nullptr,nullptr,pending.keys,pending.values,staged,kv_heads,width,padded,mfq_current_cuda_stream(),
                pending.rotated_keys,pending.rotated_values));
        }
        if(!starts.defined()) {
            auto first=absolute.narrow(-1,0,1)-positions.reshape({-1}).narrow(0,0,1).remainder(4);
            starts=tb::arange(blocks,positions.options())*4+first;
            auto last=absolute.narrow(-1,n-1,1);
            starts=tb::where(starts<=last,starts,last);
        }
        auto pooled=pool(batch.index,starts).to(tb::kFloat32).contiguous();
        check(tq::indexer_affine16_append_batch(pooled.data_ptr<float>(),device_index.codes,device_index.scales,
            device_index.zero,positions.data_ptr<int64_t>(),blocks,index_width,capacity,mfq_current_cuda_stream()));
        tq::PendingBatchPublication publication{
            positions.data_ptr<int64_t>(),canonical_k,canonical_v,batch.fp16_key.data_ptr<std::uint16_t>(),batch.fp16_value.data_ptr<std::uint16_t>(),
            batch.key.data_ptr<float>(),batch.value.data_ptr<float>(),batch.index.data_ptr<float>(),batch.rotated_key.data_ptr<float>(),batch.rotated_value.data_ptr<float>(),
            device_index.codes,device_index.scales,device_index.zero,graph_destination.data_ptr<tq::TqBlockView>(),
            views.data_ptr<tq::TqBlockView>(),graph_archive_mapped?graph_archive_mapped:graph_archive.data_ptr<std::uint8_t>(),committed_tail(),
            n,kv_heads,width,padded,index_width,int(canonical_bytes),graph_destination_slots(),bool(ram)};
        check(tq::tq_publish_graph_batch(publication,mfq_current_cuda_stream()));
        if(!graph_archive_mapped)check(cudaMemcpyAsync(graph_archive_host.data_ptr(),graph_archive.data_ptr(),graph_archive.numel(),cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        if(!output.defined())return {};
        auto q=convert(query),g=convert(gate);
        auto& a=graph_workspace(n);
        const int count=(budget+6)/4;
        auto selected=select_graph_indexer(iq.to(tb::kFloat32).contiguous(),positions,n,a);
        cudaStreamCaptureStatus status;check(cudaStreamIsCapturing(mfq_current_cuda_stream(),&status));const bool observe=status!=cudaStreamCaptureStatusNone;
        const auto operands=stage_pending(selected,n,count,positions);
        check(tq::tq_sparse_attention(q.data_ptr(),ks.data_ptr<float>(),vs.data_ptr<float>(),relative.data_ptr<std::uint16_t>(),operands.data_ptr<tq::TqBlockView>(),
            selected.data_ptr<tq::SelectedBlock>(),positions.data_ptr<int64_t>(),output.data_ptr(),a.rotated.data_ptr<float>(),a.partial.data_ptr<float>(),a.stats.data_ptr<float>(),
            a.block_mass_data(),a.log_normalizer_data(),observe?hits.data_ptr<float>():nullptr,observe?mass_data():nullptr,n,heads,kv_heads,width,padded,count,a.parts,
            q.stride(2),q.stride(1),mfq_current_cuda_stream(),bool(ram),g.data_ptr(),g.scalar_type()==tb::kFloat16,output.scalar_type()==tb::kFloat16,
            g.stride(1),g.stride(2),q.scalar_type()==tb::kFloat16,true,false));
        return output;
    }
    tb::Tensor forward_pending_graph(const tb::Tensor& query,const tb::Tensor& key,const tb::Tensor& value,const tb::Tensor& iq,
        const tb::Tensor& raw,const tb::Tensor& positions,const tb::Tensor& gate,tb::Tensor output,
        const QsaTqCache::Pool& pool,const tb::Tensor& absolute_positions,const QsaTqCache::CompletePool& complete_pool) {
        if(key.size(1)>1)return forward_pending_batch(query,key,value,iq,raw,positions,gate,std::move(output),pool,absolute_positions);
        if(!pending_graph || key.size(1)!=1)throw std::logic_error("QSA device graph was not prepared");
        const auto mark=[](const char* phase){if(auto* window=DecodeWindow::recording())window->mark_current(phase);};
        mark("qsa_cache_begin");
        const auto convert=[](const tb::Tensor& x) {
            auto y=(x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32)?x:x.to(tb::kFloat32);
            return y.stride(y.dim()-1)==1?y:y.contiguous();
        };
        auto k=convert(key),v=convert(value),r=convert(raw);
        const bool attend=output.defined();
        auto q=attend?convert(query):Tensor{};
        auto& a=pending_attention;
        const tq::PendingQuery prepared_query=attend?
            tq::PendingQuery{q.data_ptr(),a.rotated.data_ptr<float>(),heads,q.stride(1),q.scalar_type()==tb::kFloat16}:
            tq::PendingQuery{};
        check(tq::tq_append_pending(k.data_ptr(),v.data_ptr(),r.data_ptr(),k.scalar_type()==tb::kFloat16,v.scalar_type()==tb::kFloat16,r.scalar_type()==tb::kFloat16,
            positions.data_ptr<int64_t>(),tail_storage_k.data_ptr<float>(),tail_storage_v.data_ptr<float>(),tail_storage_index.data_ptr<float>(),
            rotated_tail_k.data_ptr<float>(),rotated_tail_v.data_ptr<float>(),
            ks.data_ptr<float>(),vs.data_ptr<float>(),pending_k.data_ptr<std::uint16_t>(),pending_v.data_ptr<std::uint16_t>(),views.data_ptr<tq::TqBlockView>(),
            kv_heads,width,padded,index_width,k.stride(2),v.stride(2),mfq_current_cuda_stream(),
            pending_modes_k.data_ptr<float>(),pending_modes_v.data_ptr<float>(),attend?&prepared_query:nullptr));
        {
            if(!pool)throw std::logic_error("complete QSA graph requires indexer pooling");
            check(tq::tq_finalize_pending(pending_modes_k.data_ptr<float>(),pending_modes_v.data_ptr<float>(),tail_storage_index.data_ptr<float>(),
                graph_destination.data_ptr<tq::TqBlockView>(),views.data_ptr<tq::TqBlockView>(),positions.data_ptr<int64_t>(),
                graph_archive_mapped?graph_archive_mapped:graph_archive.data_ptr<std::uint8_t>(),rows,index_width,
                graph_destination_slots(),mfq_current_cuda_stream(),true));
            auto starts=absolute_positions.defined()?absolute_positions:positions;
            if(starts.dim()==1)starts=starts.reshape({1,1});
            auto pooled=complete_pool?complete_pool(tail_storage_index,starts,positions):tb::Tensor{};
            if(pooled.defined() && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH")) {
                static thread_local bool reported=false;
                if(!reported) {
                    reported=true;
                    std::fprintf(stderr,"native_qsa_indexer_pool_complete actual_match=1 position_offset_in_kernel=1\n");
                }
            }
            if(!pooled.defined()) {
                // All rotary axes share the cache position within this microblock.
                starts=starts-positions.remainder(4);
                pooled=pool(tail_storage_index,starts);
            }
            pooled=pooled.to(tb::kFloat32).contiguous();
            check(tq::indexer_affine16_append(pooled.data_ptr<float>(),device_index.codes,device_index.scales,
                device_index.zero,positions.data_ptr<int64_t>(),index_width,mfq_current_cuda_stream(),true));
            if(ram)check(tq::tq_archive_indexer(device_index.codes,device_index.scales,device_index.zero,
                graph_archive_mapped?graph_archive_mapped:graph_archive.data_ptr<std::uint8_t>(),2*canonical_bytes+8*index_width,
                1,index_width,0,positions.data_ptr<int64_t>(),mfq_current_cuda_stream(),true));
            if(!graph_archive_mapped)check(cudaMemcpyAsync(graph_archive_host.data_ptr(),graph_archive.data_ptr(),graph_archive.numel(),cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        }
        if(!attend)return {};
        auto g=convert(gate);
        const int columns=budget+3,count=(columns+3)/4,parts=a.parts;
        mark("qsa_cache_append_end");
        auto blocks=select_graph_indexer(iq.to(tb::kFloat32).contiguous(),positions,1,a,true);
        mark("qsa_cache_select_end");
        // Graph preparation runs this body repeatedly without committing a
        // token. Only the recorded replay may update its hit/attention history.
        cudaStreamCaptureStatus capture_status;
        check(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture_status));
        const bool observe=capture_status!=cudaStreamCaptureStatusNone;
        const auto operands=stage_pending(blocks,1,count,positions);
        if(ram)mark("qsa_cache_ram_staging_end");
        check(tq::tq_sparse_attention(q.data_ptr(),ks.data_ptr<float>(),vs.data_ptr<float>(),relative.data_ptr<std::uint16_t>(),operands.data_ptr<tq::TqBlockView>(),
            blocks.data_ptr<tq::SelectedBlock>(),positions.data_ptr<int64_t>(),output.data_ptr(),a.rotated.data_ptr<float>(),a.partial.data_ptr<float>(),a.stats.data_ptr<float>(),
            a.block_mass_data(),a.log_normalizer_data(),observe?hits.data_ptr<float>():nullptr,observe?mass_data():nullptr,1,heads,kv_heads,width,padded,count,parts,
            q.stride(2),q.stride(1),mfq_current_cuda_stream(),bool(ram),g.data_ptr(),g.scalar_type()==tb::kFloat16,output.scalar_type()==tb::kFloat16,
            g.stride(1),g.stride(2),q.scalar_type()==tb::kFloat16,true,true));
        mark("qsa_cache_attention_end");
        return output;
    }
    void append_graph_image() {
        finish_history();
        const auto id=std::size_t(tokens/4);
        // DecodeWindow has completed the entire model, including this pinned
        // archive copy. No layer boundary or additional GPU wait is required.
        if(!ram){ensure_owner_slots(id+1);owners[id]=std::move(graph_image);}
        const auto* bytes=graph_archive_host.data_ptr<std::uint8_t>();
        tq::MicroblockImage image;image.key.assign(bytes,bytes+canonical_bytes);
        image.value.assign(bytes+canonical_bytes,bytes+2*canonical_bytes);
        if(ram_bits==16)image.key_bits=image.value_bits=0;
        if(ram){ram->append({std::move(image)},std::uint64_t(id)*4,true);graph_image.reset();}
        else {ResidentAppendScope publication(*this,id,1);cache->append(std::move(image),std::uint64_t(id)*4);}
        const auto* raw=bytes+2*canonical_bytes;
        save_index(std::vector<std::uint8_t>(raw,raw+index_archive_bytes()));
    }
    void append_graph_batch() {
        finish_history();
        const auto first=std::size_t(tokens/4),record_bytes=2*canonical_bytes+index_archive_bytes();
        const auto* archived=graph_archive_host.data_ptr<std::uint8_t>();
        std::vector<tq::MicroblockImage> images;images.reserve(graph_blocks);
        for(int i=0;i<graph_blocks;++i) {
            const auto* source=archived+i*record_bytes;tq::MicroblockImage image;
            image.key.assign(source,source+canonical_bytes);image.value.assign(source+canonical_bytes,source+2*canonical_bytes);
            if(ram_bits==16)image.key_bits=image.value_bits=0;
            images.push_back(std::move(image));
            const auto* raw=source+2*canonical_bytes;save_index(std::vector<std::uint8_t>(raw,raw+index_archive_bytes()));
            if(!ram){ensure_owner_slots(first+i+1);owners[first+i]=graph_image;}
        }
        if(ram)ram->append(std::move(images),std::uint64_t(first)*4,true);
        else {ResidentAppendScope publication(*this,first,images.size());cache->append_batch(std::move(images),std::uint64_t(first)*4);}
        graph_image.reset();
    }
    void finish_history(bool wait=true) {
        auto& h=history_transfer;
        if(!h.pending)return;
        const auto ready=wait?cudaEventSynchronize(h.done):cudaEventQuery(h.done);
        if(!wait && ready==cudaErrorNotReady)return;
        check(ready);
        const auto* source=h.host.data_ptr<float>();
        std::vector<double> hits(source,source+h.pools);
        if(ram)ram->observe(hits,h.tokens);
        else cache->update(h.tokens,h.budget,std::move(hits),std::vector<double>(source+h.pools,source+2*h.pools));
        h.pending=false;
    }
    void update_history(int64_t start) {
        if((byte_budget || ram) && tokens/interval!=start/interval) {
            finish_history();
            auto& h=history_transfer;const int pools=int(tokens/4);
            const auto elements=int64_t(pools)*(mass.defined()?2:1);
            if(!h.host.defined() || h.host.numel()<elements)
                h.host=tb::empty({elements},options.device(tb::kCPU).dtype(tb::kFloat32).pinned_memory(true));
            if(!h.done)check(cudaEventCreateWithFlags(&h.done,cudaEventDisableTiming));
            auto* destination=h.host.data_ptr<float>();const auto bytes=std::size_t(pools)*sizeof(float);
            const auto stream=mfq_current_cuda_stream();
            check(cudaMemcpyAsync(destination,hits.data_ptr(),bytes,cudaMemcpyDeviceToHost,stream));
            if(mass.defined())check(cudaMemcpyAsync(destination+pools,mass.data_ptr(),bytes,cudaMemcpyDeviceToHost,stream));
            check(cudaEventRecord(h.done,stream));
            h.tokens=tokens;h.budget=byte_budget;h.pools=pools;h.pending=true;
            hits.zero_();if(mass.defined())mass.zero_();
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> printed{false};
                if(!printed.exchange(true))std::fprintf(stderr,"native_qsa_history_transfer actual_match=1 pinned_async=1 deferred_update=1\n");
            }
        }
    }
    tb::Tensor forward(const tb::Tensor& query,const tb::Tensor& keys,const tb::Tensor& values,const tb::Tensor& iq,
        const tb::Tensor& raw,const tb::Tensor& positions,const QsaTqCache::Pool& pool,bool history_only,
        const tb::Tensor& gate,tb::Tensor output,bool fp16_math,CudaProfiler* profiler) {
        ensure_descriptors();
        cancel_batch_warmup();finish_history();
        const auto t=keys.size(1),start=tokens;
        // Eager prefill can borrow temporary GPU pages. End that lease before returning.
        struct PrefillScratchLifetime {
            AttentionWorkspace& workspace;
            bool release;
            ~PrefillScratchLifetime(){if(release)workspace={};}
        } prefill_scratch{prefill_attention,t>8};
        pending_graph=false;
        append_small=t<=8;
        if(keys.size(0)!=1 || t<=0 || t>maximum-tokens || keys.size(2)!=kv_heads || keys.size(3)!=width ||
           values.sizes()!=keys.sizes() || raw.size(0)!=1 || raw.size(1)!=t || raw.size(2)!=index_width)
            throw std::invalid_argument("adaptive QSA input geometry mismatch");
        allocate_indexer();
        const auto supported=[](const tb::Tensor& x){return x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32;};
        auto input_k=supported(keys)?keys:keys.to(tb::kFloat32);if(input_k.stride(3)!=1)input_k=input_k.contiguous();
        auto input_v=supported(values)?values:values.to(tb::kFloat32);if(input_v.stride(3)!=1)input_v=input_v.contiguous();
        auto input_r=supported(raw)?raw:raw.to(tb::kFloat32);if(input_r.stride(2)!=1)input_r=input_r.contiguous();
        const int prior=int(tokens%4),staged=int(t)+prior;
        append_input_position=-1;
        if(append_capacity<staged) {
            ResidentAllocationScope resident;
            auto next_k=tb::empty({1,staged,kv_heads,width},options.dtype(tb::kFloat32));
            auto next_v=tb::empty_like(next_k);
            auto next_index=tb::empty({1,staged,index_width},options.dtype(tb::kFloat32));
            append_k=std::move(next_k);append_v=std::move(next_v);append_index=std::move(next_index);
            append_capacity=staged;
        }
        auto k=append_k.narrow(1,0,staged),v=append_v.narrow(1,0,staged),r=append_index.narrow(1,0,staged);
        check(tq::tq_stage_append(input_k.data_ptr(),input_v.data_ptr(),input_r.data_ptr(),input_k.scalar_type()==tb::kFloat16,
            input_v.scalar_type()==tb::kFloat16,input_r.scalar_type()==tb::kFloat16,prior?tail_k.data_ptr<float>():nullptr,
            prior?tail_v.data_ptr<float>():nullptr,prior?tail_index.data_ptr<float>():nullptr,k.data_ptr<float>(),v.data_ptr<float>(),r.data_ptr<float>(),
            int(t),prior,kv_heads,width,index_width,input_k.stride(1),input_k.stride(2),input_v.stride(1),input_v.stride(2),input_r.stride(1),mfq_current_cuda_stream()));
        append_input_position=start;
        const auto base=tokens/4,complete=k.size(1)/4,remainder=k.size(1)%4;
        tb::Tensor cpu_archive;
        std::unique_ptr<ArchiveLease> archive_lease;
        if(complete) {
            const auto span=complete*4;
            auto group=std::make_shared<ImageBatch>();
            {
                // Published images remain live until the cache replaces them
                // and the last reader finishes, possibly after prefill returns.
                const ResidentAllocationScope resident(!ram || !PrefillAllocationScope::active());
                group->payload=tb::empty({int64_t(2*complete*canonical_bytes)},options.dtype(tb::kUInt8));
            }
            auto* canonical_k=group->payload.data_ptr<std::uint8_t>();auto* canonical_v=canonical_k+complete*canonical_bytes;
            if(ram_bits==16) {
                auto modes=tb::empty({2*complete*rows*256},options.dtype(tb::kFloat32));
                check(tq::tq_prepare_modes(k.data_ptr(),v.data_ptr(),false,ks.data_ptr<float>(),vs.data_ptr<float>(),
                    modes.data_ptr<float>(),modes.data_ptr<float>()+complete*rows*256,
                    reinterpret_cast<std::uint16_t*>(canonical_k),reinterpret_cast<std::uint16_t*>(canonical_v),
                    int(span),kv_heads,width,padded,mfq_current_cuda_stream()));
            }else check(tq::tq_prepare_affine8(k.data_ptr(),v.data_ptr(),false,ks.data_ptr<float>(),vs.data_ptr<float>(),
                canonical_k,canonical_v,int(span),kv_heads,width,padded,mfq_current_cuda_stream()));
            const auto archive_bytes=2*complete*canonical_bytes+complete*index_archive_bytes();
            if(!archive.defined() || archive.numel()<archive_bytes) {
                const ResidentAllocationScope resident(!ram || !PrefillAllocationScope::active());
                archive=tb::empty({int64_t(archive_bytes)},options.dtype(tb::kUInt8));
            }
            check(tq::tq_pack_archive(canonical_k,canonical_v,r.data_ptr<float>(),archive.data_ptr<std::uint8_t>(),int(complete),int(canonical_bytes),index_width,mfq_current_cuda_stream()));
            auto origin=positions.narrow(-1,0,1)-start;
            if(origin.dim()==1)origin=origin.reshape({1,1});
            auto block_positions=tb::arange(complete,positions.options().dtype(tb::kInt64))*4+base*4+origin;
            auto pooled=pool(r.narrow(1,0,span),block_positions).to(tb::kFloat32).contiguous();
            check(tq::indexer_affine16_encode(pooled.data_ptr<float>(),device_index.codes+base*index_width,
                device_index.scales+base,device_index.zero+base,int(complete),index_width,mfq_current_cuda_stream()));
            if(ram)check(tq::tq_archive_indexer(device_index.codes,device_index.scales,device_index.zero,
                archive.data_ptr<std::uint8_t>(),2*complete*canonical_bytes+8*complete*index_width,int(complete),index_width,base,nullptr,mfq_current_cuda_stream()));
            {
                archive_lease=lease_archive();auto& arena=*archive_lease->arena;
                if(!arena.host.defined() || arena.host.numel()<archive_bytes)
                    arena.host=tb::empty({int64_t(archive_bytes)},options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
                cpu_archive=arena.host.narrow(0,0,archive_bytes);
                arena.ready.record(mfq_current_cuda_stream());
                check(cudaStreamWaitEvent(arena.stream.stream(),arena.ready.get(),0));
                archive_lease->submitted=true;
                check(cudaMemcpyAsync(cpu_archive.data_ptr(),archive.data_ptr(),archive_bytes,cudaMemcpyDeviceToHost,arena.stream.stream()));
                arena.done.record(arena.stream.stream());mfq_cuda_record_stream(archive,arena.stream);
                if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                    static std::atomic<unsigned> printed{0};const unsigned category=t<=8?1u:2u;
                    if(!(printed.fetch_or(category)&category))std::fprintf(stderr,"native_qsa_archive_overlap actual_match=1 queries=%lld bytes=%lld shared_pinned_arena=1 side_stream=1 device_descriptors=1\n",static_cast<long long>(t),static_cast<long long>(archive_bytes));
                }
            }
            // Fixed-size canonical planes need no host descriptor upload.
            // Publishing on the model stream lets the pinned archive copy run
            // while selection and attention consume these same GPU images.
            const unsigned tag=ram_bits==16?0:254;
            check(tq::tq_publish_contiguous_views({canonical_k,canonical_v,tag,tag},views.data_ptr<tq::TqBlockView>()+base,
                int(canonical_bytes),int(complete),mfq_current_cuda_stream()));
            std::vector<std::size_t> ids;ids.reserve(complete);
            for(int64_t i=0;i<complete;++i)ids.push_back(std::size_t(base+i));
            retain_images(group,ids);
        }
        bool archive_committed=false;
        const auto commit_archive=[&] {
            if(archive_committed)return;
            archive_committed=true;
            if(archive_lease)archive_lease->wait();
            // Bound transient images when SSD storage does not retain them.
            // GPU attention and its compute chunk are independent of this batch.
            const auto batch_blocks=ssd_cache
                ?std::max<std::size_t>(1,(8ULL<<20)/(2*canonical_bytes))
                :std::max<std::size_t>(1,std::size_t(complete));
            std::vector<tq::MicroblockImage> images;
            images.reserve(std::min<std::size_t>(std::size_t(complete),batch_blocks));
            for(int64_t i=0;i<complete;++i) {
                tq::MicroblockImage image;
                const auto* a=cpu_archive.data_ptr<std::uint8_t>()+i*canonical_bytes;const auto* b=cpu_archive.data_ptr<std::uint8_t>()+(complete+i)*canonical_bytes;
                image.key.assign(a,a+canonical_bytes);image.value.assign(b,b+canonical_bytes);
                if(ram_bits==16)image.key_bits=image.value_bits=0;
                images.push_back(std::move(image));
                const auto* index=cpu_archive.data_ptr<std::uint8_t>()+2*complete*canonical_bytes+i*4*index_width*2;
                // Allocate the complete image once. Appending pooled codes to
                // a raw-index vector previously grew every microblock's capacity.
                std::vector<std::uint8_t> bytes(index_archive_bytes());
                std::memcpy(bytes.data(),index,8*index_width);
                if(ram) {
                    const auto* codes=cpu_archive.data_ptr<std::uint8_t>()+2*complete*canonical_bytes+8*complete*index_width+i*(index_width+2)*2;
                    std::memcpy(bytes.data()+8*index_width,codes,(index_width+2)*2);
                }
                save_index(std::move(bytes));
                if(images.size()==batch_blocks || i+1==complete) {
                    const auto first=std::size_t(base+i+1)-images.size();
                    if(ram) {
                        const auto n=images.size();ram->append(std::move(images),std::uint64_t(first)*4,t<=8);
                        std::vector<std::shared_ptr<ImageBatch>> previous;
                        for(std::size_t id=first;id<first+n;++id)if(owners[id])previous.push_back(std::move(owners[id]));
                        retire(std::move(previous));
                    }else {
                        ResidentAppendScope publication(*this,first,images.size());cache->append_batch(std::move(images),std::uint64_t(first)*4);
                    }
                    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                        static std::atomic<bool> printed{false};
                        if(!printed.exchange(true))std::fprintf(stderr,"native_qsa_cpu_archive_batch actual_match=1 blocks=%lld ssd=%d prepublished_range=1\n",
                            static_cast<long long>(complete),int(ssd_cache));
                    }
                    images.clear();
                    if(i+1<complete)images.reserve(std::min<std::size_t>(std::size_t(complete-i-1),batch_blocks));
                }
            }
            if(archive_lease)archive_lease->lock.unlock();
        };
        tokens+=t;drain();
        check(tq::tq_commit_tail(k.data_ptr<float>(),v.data_ptr<float>(),r.data_ptr<float>(),tail_storage_k.data_ptr<float>(),
            tail_storage_v.data_ptr<float>(),tail_storage_index.data_ptr<float>(),staged,int(remainder),kv_heads,width,index_width,mfq_current_cuda_stream()));
        tail_k=remainder?tail_storage_k.narrow(1,0,remainder):tb::Tensor{};
        tail_v=remainder?tail_storage_v.narrow(1,0,remainder):tb::Tensor{};
        tail_index=remainder?tail_storage_index.narrow(1,0,remainder):tb::Tensor{};
        if(remainder) {
            check(tq::tq_prepare_modes(tail_k.data_ptr(),tail_v.data_ptr(),false,ks.data_ptr<float>(),vs.data_ptr<float>(),pending_modes_k.data_ptr<float>(),pending_modes_v.data_ptr<float>(),
                pending_k.data_ptr<std::uint16_t>(),pending_v.data_ptr<std::uint16_t>(),int(remainder),kv_heads,width,padded,mfq_current_cuda_stream(),
                rotated_tail_k.data_ptr<float>(),rotated_tail_v.data_ptr<float>()));
            publish_pending(pending_k,pending_v,tokens/4);
        }
        if(history_only){commit_archive();drain();return {};}
        // Bound indexer scores independently of the projection/FFN chunk.
        // A long target context must not multiply every layer's score arena
        // by the complete 8K compute tile. Query tiles preserve causal positions.
        const auto score_budget=option("TYLOINFER_QSA_PREFILL_SCORE_BYTES",64ULL<<20);
        auto score_pools=std::max(1,prefill_attention.pools);
        while(score_pools<tokens/4)score_pools=std::min(capacity,score_pools>capacity/2?capacity:2*score_pools);
        const auto row_bytes=std::size_t(score_pools)*sizeof(float);
        auto query_limit=std::max<int64_t>(16,int64_t(score_budget/row_bytes));
        // Until the first forward is complete, every canonical KV image is
        // already on the GPU. Do not publish RAM replacements between score
        // tiles: their query offsets are not evidence of pre-existing history.
        const bool fresh_prefill=ram && t>8 && start==0;
        std::optional<QsaRamPrefillStage> layer_stage;
        bool layer_stage_attempted=false;
        tb::Tensor decoded_operands,requested_blocks;
        bool decode_staging_attempted=false;
        const auto prefill_queries=t;
        tb::Tensor staged_codes,staged_scales,staged_zero;
        IndexPointers staged_index;
        bool host_scores=host_indexer && append_small;
        if(host_indexer && !host_scores && tokens/4) {
            const auto live=tokens/4;
            try {
                staged_codes=index_codes.narrow(0,0,live).to(options.dtype(tb::kInt16));
                staged_scales=index_scales.narrow(0,0,live).to(options.dtype(tb::kFloat16));
                staged_zero=index_zero.narrow(0,0,live).to(options.dtype(tb::kInt16));
                staged_index={staged_codes.data_ptr<std::int16_t>(),staged_scales.data_ptr<std::uint16_t>(),staged_zero.data_ptr<std::int16_t>()};
            } catch(const MfqResourceExhausted&) {
                staged_codes={};staged_scales={};staged_zero={};
                (void)cudaGetLastError();
                // Keep the affine indexer on the host when the current
                // workspace cannot stage an entire long-context layer.
                host_scores=true;query_limit=8;
            }
        }
        const int pools=int(tokens/4),columns=budget+3,count=(columns+3)/4;
        const auto partitions=[&](int query_rows,bool half) {
            const auto active=host_indexer && append_small?int(std::min<int64_t>(count,(tokens+3)/4)):count;
            return tq::tq_attention_partitions(query_rows,heads,kv_heads,padded,active,half,fp16_math,bool(ram && query_rows<=8));
        };
        if(t>8) {
            // Size the attention tile before launching scores or touching the
            // recall history. Free pages in a CUDA pool may be fragmented.
            const auto initial_limit=std::min(t,query_limit);
            for(;;) {
                const auto tiles=(t+query_limit-1)/query_limit;
                const auto rows=(t+tiles-1)/tiles;
                try {
                    prepare_workspace(int(rows),pools,partitions(int(rows),query.scalar_type()==tb::kFloat16),false,!host_scores);
                    break;
                } catch(const MfqResourceExhausted&) {
                    if(rows<=1)throw;
                    prefill_attention={};
                    default_context(device)->reclaim_idle_memory();
                    query_limit=std::max<int64_t>(1,rows/2);
                }
            }
            if(query_limit<initial_limit && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
                std::fprintf(stderr,"native_qsa_prefill_workspace_reduce queries=%lld previous=%lld limit=%lld\n",
                    (long long)t,(long long)initial_limit,(long long)query_limit);
        }
        const auto attend=[&](const tb::Tensor& query,const tb::Tensor& iq,
                const tb::Tensor& gate,tb::Tensor output,int64_t start) {
            const auto t=query.size(2);
            auto q=supported(query)?query:query.to(tb::kFloat32);if(q.stride(3)!=1)q=q.contiguous();
            auto absolute=tb::arange(start,start+t,options.dtype(tb::kInt64));
            tb::Tensor blocks;
            const int parts=partitions(int(t),q.scalar_type()==tb::kFloat16);
            auto& scratch=prepare_workspace(int(t),pools,parts,false,!host_scores);
            if(pools) {
                const auto select=[&] {
                    if(host_scores)return select_host_indexer(iq.to(tb::kFloat32).contiguous(),absolute,int(t),pools);
                    auto ranked=scratch.ranked.narrow(0,0,t*pools).reshape({1,t,pools});
                    indexer_scores(iq.to(tb::kFloat32).contiguous(),absolute,ranked,int(t),pools,scratch,
                        !append_small,host_indexer?&staged_index:nullptr);
                    return mfq_qwen4_exp::live_topk_selected_blocks(ranked,absolute,4,std::min(budget/4,pools),budget);
                };
                blocks=profiler?profiler->measure("prefill.qsa.indexer",select):select();
            } else {
                auto ids=tb::arange(columns,absolute.options()).reshape({1,1,columns}).expand({1,t,columns});
                auto selected=tb::where(ids<=absolute.reshape({1,t,1}),ids,tb::full_like(ids,-1)).to(tb::kInt32).contiguous();
                blocks=scratch.blocks.narrow(0,0,t*count*int64_t(sizeof(tq::SelectedBlock)));
                check(tq::tq_group_selected(selected.data_ptr<int32_t>(),blocks.data_ptr<tq::SelectedBlock>(),int(t),columns,mfq_current_cuda_stream()));
            }
            if(!output.defined())output=tb::empty({t,heads,width},options.dtype(tb::kFloat32));
            if(!supported(output) || !output.is_contiguous() || output.numel()!=t*heads*width)
                throw std::invalid_argument("adaptive QSA output must be contiguous FP16 or FP32");
            auto g=gate.defined() && !supported(gate)?gate.to(tb::kFloat32):gate;
            if(g.defined() && (g.dim()!=4 || g.size(0)!=1 || g.size(1)!=t || g.size(2)!=heads || g.size(3)!=width))
                throw std::invalid_argument("adaptive QSA gate geometry mismatch");
            if(g.defined() && g.stride(3)!=1)g=g.contiguous();
            if(ram && t>8 && !fresh_prefill) {
                // Preserve original mixed precision and copy one layer once for
                // all query tiles. If available workspace cannot hold it,
                // stream each selected union through the bounded RAM arena.
                commit_archive();drain();
                if(!layer_stage_attempted) {
                    layer_stage_attempted=true;
                    // A short append cannot reuse a whole history layer when
                    // even its total number of selections is smaller.
                    const auto complete_blocks=std::size_t(tokens/4);
                    if(prefill_layer_stage || tokens/4<=prefill_queries*int64_t(count)) {
                        try {
                            if(prefill_layer_stage) {
                                layer_stage=ram->stage_prefill(complete_blocks,prefill_layer_end,&*prefill_layer_stage);
                                if(!layer_stage)prefill_layer_stage.reset();
                            }
                            if(!layer_stage)layer_stage=ram->stage_prefill(complete_blocks,prefill_layer_end);
                            if(layer_stage && prefill_layer_end>complete_blocks)prefill_layer_stage=layer_stage;
                        } catch(const MfqResourceExhausted&) {
                            prefill_layer_stage.reset();
                            (void)cudaGetLastError();
                        }
                    }
                }
                if(!layer_stage) {
                    const auto grouped=blocks.reshape({t,-1});
                    // The selected blocks retain their storage. Drop the wide
                    // scoring workspace before reserving the recall tiles.
                    prefill_attention={};
                    struct ReleaseRecallWorkspace {
                        QsaRamCache& cache;
                        ~ReleaseRecallWorkspace(){cache.release_prefill_stage();}
                    } release_recall{*ram};
                    const auto split_for=[&](int n) {
                        return tq::tq_attention_partitions(n,heads,kv_heads,padded,count,
                            q.scalar_type()==tb::kFloat16,fp16_math,true);
                    };
                    const auto wanted=int(std::min<int64_t>(qsa_prefill_recall_tile_rows,t));
                    int rows=wanted;
                    for(;;) {
                        try {
                            // Admit I/O staging and attention scratch together before
                            // a recall can publish any temporary source descriptors.
                            ram->prepare_prefill_stage(rows,std::uint64_t(tokens-remainder));
                            prepare_workspace(rows,pools,std::max(split_for(rows),split_for(int(t%rows))),false,false);
                            break;
                        } catch(const MfqResourceExhausted&) {
                            if(rows==1)throw;
                            ram->release_prefill_stage();
                            prefill_attention={};
                            (void)cudaGetLastError();
                            default_context(device)->reclaim_idle_memory();
                            rows/=2;
                        }
                    }
                    if(rows<wanted && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
                        std::fprintf(stderr,"native_qsa_prefill_stage_budget requested=%d actual=%d\n",wanted,rows);
                    const auto recall=[&] {
                        for(int64_t begin=0;begin<t;begin+=rows) {
                            const int n=int(std::min<int64_t>(rows,t-begin));
                            auto selected=grouped.narrow(0,begin,n);
                            auto operands=ram->stage(selected,n,count,views,{},tokens-remainder,int(remainder));
                            const int split=split_for(n);
                            auto& tile=prepare_workspace(n,pools,split,false,false);
                            auto tile_query=q.narrow(2,begin,n),tile_positions=absolute.narrow(0,begin,n);
                            auto tile_output=output.reshape({t,heads,width}).narrow(0,begin,n);
                            auto tile_gate=g.defined()?g.narrow(1,begin,n):tb::Tensor{};
                            check(tq::tq_sparse_attention_staged(nullptr,nullptr,tile_query.data_ptr(),ks.data_ptr<float>(),vs.data_ptr<float>(),
                                relative.data_ptr<std::uint16_t>(),operands.data_ptr<tq::TqBlockView>(),selected.data_ptr<tq::SelectedBlock>(),
                                tile_positions.data_ptr<int64_t>(),tile_output.data_ptr(),tile.rotated.data_ptr<float>(),tile.partial.data_ptr<float>(),
                                tile.stats.data_ptr<float>(),tile.block_mass_data(),tile.log_normalizer_data(),hits.data_ptr<float>(),mass_data(),
                                n,heads,kv_heads,width,padded,count,split,tile_query.stride(2),tile_query.stride(1),mfq_current_cuda_stream(),true,
                                tile_gate.defined()?tile_gate.data_ptr():nullptr,tile_gate.defined() && tile_gate.scalar_type()==tb::kFloat16,
                                tile_output.scalar_type()==tb::kFloat16,tile_gate.defined()?tile_gate.stride(1):0,tile_gate.defined()?tile_gate.stride(2):0,
                                tile_query.scalar_type()==tb::kFloat16,fp16_math));
                        }
                        return output;
                    };
                    return profiler?profiler->measure("prefill.qsa.recall_attention",recall):recall();
                }
            }
            // Decode the layer once and reuse it across every score tile.
            // Long histories must not lose tensor-core attention solely
            // because they exceed a fixed small temporary allocation.
            const auto block_count=(tokens+3)/4;
            const auto staged_elements=2*block_count*kv_heads*4LL*padded;
            const auto* staging_option=std::getenv("TYLOINFER_QSA_PREFILL_STAGING");
            const bool use_staging=prefill_queries>8 && heads/kv_heads>=4 &&
                (fp16_math || q.scalar_type()==tb::kFloat16) &&
                (!staging_option || staging_option[0]!='0') && block_count<=INT_MAX/4 &&
                block_count<prefill_queries*count/2 && tq::tq_prefill_staging_available(heads,kv_heads,padded);
            if(use_staging && !decode_staging_attempted) {
                decode_staging_attempted=true;
                const auto available=available_prefill_workspace(*default_context(mfq_current_cuda_device()));
                const auto limit=option("TYLOINFER_QSA_PREFILL_STAGING_BYTES",SIZE_MAX);
                const auto bytes=std::size_t(staged_elements)*sizeof(std::uint16_t);
                const auto metadata=std::size_t(block_count)*(sizeof(tq::SelectedBlock)+5*sizeof(std::uint32_t));
                if(bytes<=limit && bytes<=available && metadata<=available-bytes) {
                    try {
                        auto operands=tb::empty({staged_elements},options.dtype(tb::kFloat16));
                        auto requested=tb::empty({block_count},options.dtype(tb::kInt32));
                        auto ids=tb::arange(block_count*4,options.dtype(tb::kInt32));
                        auto selected=tb::empty({int64_t(block_count*sizeof(tq::SelectedBlock))},options.dtype(tb::kUInt8));
                        check(tq::tq_group_selected(ids.data_ptr<int32_t>(),selected.data_ptr<tq::SelectedBlock>(),
                            1,int(block_count*4),mfq_current_cuda_stream()));
                        const auto& source_views=layer_stage?layer_stage->views:views;
                        auto* keys=operands.data_ptr<std::uint16_t>();
                        check(tq::tq_stage_attention_blocks(source_views.data_ptr<tq::TqBlockView>(),relative.data_ptr<std::uint16_t>(),
                            keys,keys+staged_elements/2,int(block_count),kv_heads,padded,mfq_current_cuda_stream(),
                            selected.data_ptr<tq::SelectedBlock>(),1,int(block_count),requested.data_ptr<std::uint32_t>()));
                        decoded_operands=std::move(operands);
                        requested_blocks=std::move(requested);
                    } catch(const MfqResourceExhausted&) {
                        (void)cudaGetLastError();
                    }
                    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1')
                        std::fprintf(stderr,"native_qsa_prefill_staging actual_match=%d queries=%lld blocks=%lld bytes=%zu shared_score_tiles=1\n",
                            int(decoded_operands.defined()),(long long)prefill_queries,(long long)block_count,bytes);
                }
            }
            bool fa_used=false;
            const auto* fa_setting=std::getenv("TYLOINFER_QSA_PREFILL_FA");
            if((!fa_setting || fa_setting[0]!='0') && decoded_operands.defined() && padded==256) {
                tq::TqPrefillAttention a;
                a.keys=static_cast<const std::uint16_t*>(decoded_operands.data_ptr());a.values=a.keys+staged_elements/2;
                a.query=q.data_ptr();a.key_signs=ks.data_ptr<float>();a.value_signs=vs.data_ptr<float>();
                a.selected=blocks.data_ptr<tq::SelectedBlock>();a.positions=absolute.data_ptr<int64_t>();a.output=output.data_ptr();
                a.rotated=scratch.rotated.data_ptr<float>();a.temporary=scratch.partial.data_ptr<float>();a.temporary_bytes=scratch.partial.nbytes();
                a.logmass=scratch.block_mass_data();a.norm=scratch.log_normalizer_data();a.hits=hits.data_ptr<float>();a.mass=mass_data();
                a.queries=int(t);a.heads=heads;a.kv_heads=kv_heads;a.width=width;a.kv_rows=int(block_count*4);a.count=count;
                a.query_token_stride=q.stride(2);a.query_head_stride=q.stride(1);a.query_half=q.scalar_type()==tb::kFloat16;
                a.gate=g.defined()?g.data_ptr():nullptr;a.gate_half=g.defined() && g.scalar_type()==tb::kFloat16;a.output_half=output.scalar_type()==tb::kFloat16;
                a.gate_token_stride=g.defined()?g.stride(1):0;a.gate_head_stride=g.defined()?g.stride(2):0;a.stream=mfq_current_cuda_stream();
                if(!prefill_graph)prefill_graph=std::make_unique<tq::TqPrefillGraph>();
                const auto status=prefill_graph->run(a);
                if(status!=int(cudaErrorNotSupported)){check(status);fa_used=true;}
            }
            const bool streamed_ram=ram && t<=8 && !layer_stage;
            const auto operands=streamed_ram?ram->stage(blocks,int(t),count,views,{},
                archive_committed?tokens-remainder:start,archive_committed?int(remainder):int(t)):
                layer_stage?layer_stage->views:views;
            if(!fa_used)check(tq::tq_sparse_attention_staged(decoded_operands.defined()?static_cast<const std::uint16_t*>(decoded_operands.data_ptr()):nullptr,
                decoded_operands.defined()?static_cast<const std::uint16_t*>(decoded_operands.data_ptr())+staged_elements/2:nullptr,q.data_ptr(),ks.data_ptr<float>(),vs.data_ptr<float>(),relative.data_ptr<std::uint16_t>(),operands.data_ptr<tq::TqBlockView>(),
                blocks.data_ptr<tq::SelectedBlock>(),absolute.data_ptr<int64_t>(),output.data_ptr(),scratch.rotated.data_ptr<float>(),scratch.partial.data_ptr<float>(),scratch.stats.data_ptr<float>(),
                scratch.block_mass_data(),scratch.log_normalizer_data(),hits.data_ptr<float>(),mass_data(),int(t),heads,kv_heads,width,padded,count,parts,
                q.stride(2),q.stride(1),mfq_current_cuda_stream(),streamed_ram,g.defined()?g.data_ptr():nullptr,g.defined() && g.scalar_type()==tb::kFloat16,
                output.scalar_type()==tb::kFloat16,g.defined()?g.stride(1):0,g.defined()?g.stride(2):0,q.scalar_type()==tb::kFloat16,fp16_math));
            if(!fresh_prefill){commit_archive();drain();}
            return output;
        };
        if(t>8 && t>query_limit) {
            if(!output.defined())output=tb::empty({t,heads,width},options.dtype(tb::kFloat32));
            if(!supported(output) || !output.is_contiguous() || output.numel()!=t*heads*width)
                throw std::invalid_argument("adaptive QSA output must be contiguous FP16 or FP32");
            const auto tiles=(t+query_limit-1)/query_limit;
            int64_t begin=0;
            for(int64_t tile=0;tile<tiles;++tile) {
                const auto n=(t-begin+tiles-tile-1)/(tiles-tile);
                attend(query.narrow(2,begin,n),iq.narrow(1,begin,n),
                    gate.defined()?gate.narrow(1,begin,n):tb::Tensor{},
                    output.reshape({t,heads,width}).narrow(0,begin,n),start+begin);
                begin+=n;
            }
            if(std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
                "native_qsa_prefill_score_tiles actual_match=1 queries=%lld limit=%lld score_budget=%zu pools=%d\n",
                (long long)t,(long long)query_limit,score_budget,capacity);
        }else output=attend(query,iq,gate,std::move(output),start);
        commit_archive();drain();
        update_history(start);
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_QSA_TQ");trace && trace[0]=='1' && (start==0 || tokens/interval!=start/interval)) {
            const auto s=ram?ram->stats():cache->stats();std::fprintf(stderr,"qsa_adaptive_tq tokens=%lld resident=%zu demotions=%llu promotions=%llu ssd_read=%llu state_bits=14 actual_match=1 workspace_reuse=1 direct_canonical=1 fused_gate=%d gqa_group=%d\n",
                static_cast<long long>(tokens),s.resident_bytes,static_cast<unsigned long long>(s.demotions),static_cast<unsigned long long>(s.promotions),static_cast<unsigned long long>(s.canonical_read_bytes),int(gate.defined()),heads/kv_heads>4?8:4);
        }
        return output;
    }
    void truncate(int64_t keep) {
        cancel_batch_warmup();
        if(keep<0 || keep>tokens)throw std::invalid_argument("invalid adaptive QSA truncation");
        if(keep==tokens)return;
        const auto complete=keep/4,remainder=keep%4;
        const bool local_tail=keep && ram && !ssd_cache && pending_graph && complete==tokens/4 &&
            (!remainder || tail_k.defined());
        const bool archived_tail=remainder && complete<tokens/4 && complete>=graph_archive_first &&
            complete<graph_archive_first+graph_archive_count;
        if(!keep && host_index_calls) {
            std::fprintf(stderr,"native_qsa_cpu_indexer calls=%llu score_ms=%.3f transfer_ms=%.3f\n",
                static_cast<unsigned long long>(host_index_calls),host_index_ms,host_index_transfer_ms);
            host_index_calls=0;host_index_ms=host_index_transfer_ms=0;
        }
        if(!local_tail && (keep || !ram)) {
            finish_history();
            if(ram)ram->wait();else cache->wait();
            drain();
        } else if(!keep && history_transfer.pending) {
            // Retire the transfer without scheduling work on discarded KV.
            check(cudaEventSynchronize(history_transfer.done));
            history_transfer.pending=false;
        }
        // Trimming within the pending block touches only stream-ordered
        // buffers and leaves the published archive and its readers intact.
        // The completed graph already published this archive. Rebuilding its
        // tail only reads the graph's own image and writes stream-ordered KV.
        // Waiting here would serialize rollback with the preceding layer.
        if(keep && !local_tail && !archived_tail)check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        if(remainder && keep==append_input_position && append_k.defined() && append_v.defined() && append_index.defined()) {
            // The input stage retains the original partial block even if the
            // append failed before writing its completed archive entry.
            check(tq::tq_commit_tail(append_k.data_ptr<float>(),append_v.data_ptr<float>(),append_index.data_ptr<float>(),
                tail_storage_k.data_ptr<float>(),tail_storage_v.data_ptr<float>(),tail_storage_index.data_ptr<float>(),
                int(remainder),int(remainder),kv_heads,width,index_width,mfq_current_cuda_stream()));
            tail_k=tail_storage_k.narrow(1,0,remainder);tail_v=tail_storage_v.narrow(1,0,remainder);
            tail_index=tail_storage_index.narrow(1,0,remainder);
        }else if(remainder && keep/4==tokens/4 && tail_k.defined()) {
            tail_k=tail_k.narrow(1,0,remainder).contiguous();tail_v=tail_v.narrow(1,0,remainder).contiguous();tail_index=tail_index.narrow(1,0,remainder).contiguous();
        }else if(archived_tail) {
            const auto stream=mfq_current_cuda_stream();
            const auto* view=graph_destination.data_ptr<tq::TqBlockView>()+complete%graph_destination_slots();
            const auto* archive_data=graph_archive_mapped?graph_archive_mapped:graph_archive.data_ptr<std::uint8_t>();
            const auto stride=2*canonical_bytes+index_archive_bytes();
            const auto* raw=archive_data+(complete-graph_archive_first)*stride+2*canonical_bytes;
            const tq::RestoreTailIndex index{reinterpret_cast<const std::uint16_t*>(raw),
                tail_storage_index.data_ptr<float>(),int(remainder)*index_width};
            check(tq::tq_decode_modes(view,pending_modes_k.data_ptr<float>(),pending_modes_v.data_ptr<float>(),
                relative.data_ptr<std::uint16_t>(),1,kv_heads,padded,stream));
            check(tq::tq_restore_modes(pending_modes_k.data_ptr<float>(),pending_modes_v.data_ptr<float>(),
                ks.data_ptr<float>(),vs.data_ptr<float>(),tail_storage_k.data_ptr<float>(),tail_storage_v.data_ptr<float>(),
                1,kv_heads,width,padded,stream,&index));
            tail_k=tail_storage_k.narrow(1,0,remainder);tail_v=tail_storage_v.narrow(1,0,remainder);
            tail_index=tail_storage_index.narrow(1,0,remainder);
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported{false};
                if(!reported.exchange(true))std::fprintf(stderr,"native_qsa_ram_rollback actual_match=1 graph_archive_reuse=1\n");
            }
        }else if(remainder) {
            const auto image=ram?ram->reconstruction_image(complete):cache->reconstruction_image(complete);
            std::vector<std::uint8_t> data=image.key;data.insert(data.end(),image.value.begin(),image.value.end());
            auto bytes=upload(data.data(),data.size(),options);
            tq::TqBlockView record{bytes.data_ptr<std::uint8_t>(),bytes.data_ptr<std::uint8_t>()+image.key.size(),
                tq::tq_precision_flags(image.key_bits,image.key_state_bits),tq::tq_precision_flags(image.value_bits,image.value_state_bits)};
            auto view=upload(&record,sizeof(record),options);
            auto km=tb::empty({rows,256},options.dtype(tb::kFloat32)),vm=tb::empty_like(km);
            auto k=tb::empty({1,4,kv_heads,width},options.dtype(tb::kFloat32)),v=tb::empty_like(k);
            check(tq::tq_decode_modes(view.data_ptr<tq::TqBlockView>(),km.data_ptr<float>(),vm.data_ptr<float>(),relative.data_ptr<std::uint16_t>(),1,kv_heads,padded,mfq_current_cuda_stream()));
            check(tq::tq_restore_modes(km.data_ptr<float>(),vm.data_ptr<float>(),ks.data_ptr<float>(),vs.data_ptr<float>(),k.data_ptr<float>(),v.data_ptr<float>(),1,kv_heads,width,padded,mfq_current_cuda_stream()));
            tail_k=k.narrow(1,0,remainder).contiguous();tail_v=v.narrow(1,0,remainder).contiguous();
            auto index=index_archive.at(complete)->read();
            auto host=tb::empty({1,4,index_width},options.device(tb::kCPU).dtype(tb::kFloat16));
            if(index->size()!=index_archive_bytes())throw std::runtime_error("QSA raw index archive length changed");
            std::memcpy(host.data_ptr(),index->data(),8*index_width);tail_index=host.to(options.dtype(tb::kFloat32)).narrow(1,0,remainder).contiguous();
        }else {tail_k={};tail_v={};tail_index={};}
        if(!local_tail) {
            graph_archive_count=0;
            if(ram)ram->truncate(complete);else cache->truncate(complete);
            index_archive.resize(complete);
            {std::lock_guard<std::mutex> lock(queue_mutex);queued.clear();resident_append_begin=resident_append_end=0;}
            for(std::size_t i=std::size_t(complete);i<owners.size();++i)owners[i].reset();
        }
        tokens=keep;append_input_position=-1;hits.zero_();if(mass.defined())mass.zero_();
        if(!keep){canonical_slab={};graph_image.reset();}
        if(remainder) {
            check(tq::tq_prepare_modes(tail_k.data_ptr(),tail_v.data_ptr(),false,ks.data_ptr<float>(),vs.data_ptr<float>(),pending_modes_k.data_ptr<float>(),pending_modes_v.data_ptr<float>(),
                pending_k.data_ptr<std::uint16_t>(),pending_v.data_ptr<std::uint16_t>(),int(remainder),kv_heads,width,padded,mfq_current_cuda_stream(),
                rotated_tail_k.data_ptr<float>(),rotated_tail_v.data_ptr<float>()));publish_pending(pending_k,pending_v,complete);
        }
        if(const auto* trace=local_tail?std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"):nullptr;trace && trace[0]=='1') {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true))std::fprintf(stderr,"native_qsa_ram_rollback actual_match=1 pending_block_only=1\n");
        }
    }
};
QsaTqCache::QsaTqCache(int64_t max,int heads,int kv_heads,int width,int index_heads,int index_width,int budget,
        const tb::TensorOptions& options,bool ssd,bool prefer_host_indexer,bool prefer_host_metadata,bool use_spare_vram)
    :impl_(std::make_unique<Impl>(max,heads,kv_heads,width,index_heads,index_width,budget,options,ssd,prefer_host_indexer,prefer_host_metadata,
        use_spare_vram)){}
QsaTqCache::~QsaTqCache()=default;
int64_t QsaTqCache::position()const{return impl_->tokens;}
void QsaTqCache::reset(){impl_->truncate(0);}
void QsaTqCache::truncate(int64_t keep){impl_->truncate(keep);}
void QsaTqCache::wait(){impl_->finish_history();if(impl_->ram)impl_->ram->wait();else impl_->cache->wait();impl_->drain();}
void QsaTqCache::collect(){impl_->drain();}
void QsaTqCache::release_idle(bool release_indexer){
    if(impl_->tokens)return;
    if(impl_->ram)impl_->ram->release_idle();
    impl_->drain();
    if(release_indexer) {
        impl_->index_codes={};
        impl_->index_scales={};
        impl_->index_zero={};
        impl_->device_index={};
        if(impl_->migrated_indexer) {
            impl_->host_indexer=false;
            impl_->migrated_indexer=false;
        }
    }
    // Graphs kept between exclusive requests retain the device tensors above
    // and their own immutable image leases.
    const auto bytes=impl_->index_archive.capacity()*sizeof(std::shared_ptr<QsaIndexArchive>)+
        impl_->owners.capacity()*sizeof(std::shared_ptr<Impl::ImageBatch>);
    std::vector<std::shared_ptr<QsaIndexArchive>>().swap(impl_->index_archive);
    std::vector<std::shared_ptr<Impl::ImageBatch>>().swap(impl_->owners);
    if(bytes && std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
        "native_qsa_tq_idle_metadata actual_match=1 released_bytes=%zu\n",bytes);
}
void QsaTqCache::ensure_descriptors(){impl_->ensure_descriptors();}
std::size_t QsaTqCache::discard_idle_descriptors() {
    auto& s=*impl_;
    if(!s.ram || s.tokens || !s.views.defined() || !s.ram->discard_idle_descriptors())return 0;
    const auto bytes=s.views.nbytes()+s.hits.nbytes()+std::size_t(s.capacity)*sizeof(tq::RamBlockView);
    s.views={};s.hits={};
    return bytes;
}
std::size_t QsaTqCache::indexer_storage_bytes() const {
    return std::size_t(impl_->capacity)*(std::size_t(impl_->index_width)+2)*sizeof(std::int16_t);
}
std::size_t QsaTqCache::prefill_scratch_row_bytes() const {
    const auto& a=*impl_;
    const auto selected=std::size_t((a.budget+6)/4);
    // Rotated Q, minimum partial output, normalization and selection live
    // in the resident arena. Its pages cannot serve the activation pool.
    const auto history=a.ram?0:selected+1;
    return sizeof(float)*std::size_t(a.heads)*(2*std::size_t(a.padded)+history+2)+
        selected*sizeof(tq::SelectedBlock);
}
std::size_t QsaTqCache::indexer_growth_bytes() const {
    return impl_->host_indexer || impl_->index_codes.defined()?0:indexer_storage_bytes();
}
void QsaTqCache::reserve_indexer() {impl_->allocate_indexer();}
bool QsaTqCache::place_unallocated_indexer_in_ram(bool reserved_restore) {
    const auto bytes=indexer_growth_bytes();
    if(!impl_->ram || !bytes || (reserved_restore && !restore_can_place_in_ram(bytes)))
        return false;
    impl_->host_indexer=true;
    return true;
}
std::size_t QsaTqCache::migrate_indexer_to_ram() {
    return migrate_indexer(true,0);
}
std::size_t QsaTqCache::migrate_indexer_to_device(std::size_t decode_workspace) {
    return migrate_indexer(false,decode_workspace);
}
std::size_t QsaTqCache::migrate_indexer(bool host,std::size_t decode_workspace) {
    auto& s=*impl_;
    if(!s.ram || s.host_indexer==host || !s.index_codes.defined())return 0;
    const auto bytes=indexer_storage_bytes();
    const auto context=host?nullptr:default_context(s.device);
    if(host) {
        if(bytes>mfq::sample_host_memory().available_host_bytes())return 0;
        int mapped=0;
        check(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,s.device));
        if(!mapped)return 0;
    } else {
        auto free=context->available_memory();
        if((free<decode_workspace || bytes>free-decode_workspace) && s.ram->release_shared_device_cache(true)) {
            context->reclaim_idle_memory();
            free=context->available_memory();
        }
        if(free<decode_workspace || bytes>free-decode_workspace)return 0;
    }
    Impl::IndexStorage index;
    try {
        index=s.new_indexer_storage(host);
        if(const auto blocks=s.tokens/4) {
            index.codes.narrow(0,0,blocks).copy_(s.index_codes.narrow(0,0,blocks));
            index.scales.narrow(0,0,blocks).copy_(s.index_scales.narrow(0,0,blocks));
            index.zeros.narrow(0,0,blocks).copy_(s.index_zero.narrow(0,0,blocks));
        }
        check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        if(context) {
            context->check_memory_limit();
            if(context->available_memory()<decode_workspace)return 0;
        }
    } catch(const MfqResourceExhausted&) {
        (void)cudaGetLastError();
        return 0;
    }
    s.bind_indexer_storage(std::move(index),host);
    s.migrated_indexer=host;
    return bytes;
}
bool QsaTqCache::indexer_migrated_to_ram() const {return impl_->migrated_indexer;}
std::size_t QsaTqCache::minimum_recall_workspace_bytes() const {
    const auto& s=*impl_;
    if(!s.ram)return 0;
    return std::size_t((s.budget+6)/4)*std::size_t(s.rows)*2*256*sizeof(std::uint16_t);
}
std::size_t QsaTqCache::append_attention_workspace_bytes(int64_t requested) const {
    const auto& s=*impl_;
    if(requested<=0 || !s.tokens || !s.ram)return 0;
    const auto blocks=std::min<int64_t>(s.capacity,(s.tokens+requested)/4);
    const bool small=requested<=8;
    int pools=small?s.capacity:1;
    while(pools<blocks)pools=pools>s.capacity/2?s.capacity:2*pools;
    const auto score_budget=option("TYLOINFER_QSA_PREFILL_SCORE_BYTES",64ULL<<20);
    const auto limit=std::max<std::size_t>(16,score_budget/(sizeof(float)*std::size_t(pools)));
    const int queries=int(std::min<std::size_t>(requested,limit));
    const int count=(s.budget+6)/4;
    const bool gpu_scores=!s.host_indexer || !small;
    const auto scores=gpu_scores?std::size_t(queries)*pools*sizeof(float)+
        tq::indexer_affine16_score_scratch(queries,s.index_heads,s.index_width):0;
    const auto index_copy=s.host_indexer && !small?
        std::size_t(blocks)*(s.index_width+2)*sizeof(std::int16_t):0;
    const auto partial=[&](int n,bool staged) {
        const auto parts=std::max(
            tq::tq_attention_partitions(n,s.heads,s.kv_heads,s.padded,count,false,true,staged),
            tq::tq_attention_partitions(n,s.heads,s.kv_heads,s.padded,count,true,true,staged));
        auto bytes=std::size_t(n)*s.heads*s.padded*sizeof(float)*std::max(0,parts-1);
        if(n>8 && s.padded==256)
            bytes=std::max(bytes,tq::tq_prefill_attention_scratch(n,s.heads,4*pools));
        return bytes;
    };
    const int recall_queries=int(std::min<int64_t>(requested,qsa_prefill_recall_tile_rows));
    // Packed activations and base per-row scratch are charged by the caller.
    // The staged Indexer stays live across scoring and recall.
    const auto recall=minimum_recall_workspace_bytes()*recall_queries+partial(recall_queries,true);
    return index_copy+std::max(scores+partial(queries,small),recall);
}

std::size_t QsaTqCache::append_packed_history_bytes(int64_t requested) const {
    const auto& s=*impl_;
    if(!s.ram || !s.tokens || requested<=8)return 0;
    const auto blocks=std::size_t(std::min(s.maximum,s.tokens+requested)/4);
    const auto selected=std::size_t((s.budget+6)/4);
    // Whole-layer staging amortizes its transfer only when recall could
    // otherwise touch at least that many blocks.
    if(!selected || blocks/std::size_t(requested)+(blocks%std::size_t(requested)!=0)>selected)return 0;
    const auto history=s.ram->memory();
    if(history.tokens[3])return 0;
    const auto additions=blocks-std::min(blocks,std::size_t(s.tokens/4));
    namespace budget=mfq::engine;
    const auto payload=budget::prefill_budget_add(history.payload,
        budget::prefill_budget_multiply(additions,2*s.canonical_bytes));
    // K and V each align to 16 bytes in the device staging allocation.
    const auto descriptors=budget::prefill_budget_multiply(blocks+1,sizeof(tq::TqBlockView)+30);
    return budget::prefill_budget_add(payload,descriptors);
}

std::size_t QsaTqCache::fixed_prefill_ram_reservation(int64_t target_tokens) const {
    const auto& s=*impl_;
    if(!s.ram || s.ssd_cache || s.adaptive_precision || target_tokens<=0)return 0;
    const auto target=std::size_t(std::min(target_tokens,s.maximum)/4);
    const auto existing=std::size_t(s.tokens/4);
    const auto growth=target>existing?target-existing:0;
    const auto index_bytes=s.index_archive_bytes()+sizeof(QsaIndexArchive)+sizeof(std::shared_ptr<QsaIndexArchive>);
    return mfq::engine::prefill_budget_add(s.ram->fixed_prefill_ram_reservation(target),growth*index_bytes);
}

std::size_t QsaTqCache::codec_workspace_growth_bytes(std::size_t prefix_tokens) const {
    const auto& s=*impl_;
    const auto blocks=std::min(std::size_t(s.capacity),prefix_tokens/4);
    if(!blocks || !s.ram || !s.ssd_cache || !s.adaptive_precision)return 0;
    tq::TqEncodePlan plan;
    const auto batch=plan_codec_batch(blocks,s.rows,plan);
    const auto wave_limit=option("TYLOINFER_QSA_BACKGROUND_ROWS",0);
    if(wave_limit>INT_MAX)throw std::invalid_argument("QSA encoder row limit exceeds 32 bits");
    if(wave_limit)check(tq::tq_limit_encode_waves(&plan,int(wave_limit)));
    const auto canonical_image=std::size_t(s.rows)*256*sizeof(std::uint16_t);
    const TqCodecLayout layout(plan,2*batch*canonical_image);
    const auto arena=shared_codec_arena(s.device,s.codec_stream_key);
    const std::lock_guard lock(arena->mutex);
    return arena->growth_bytes(layout);
}
void QsaTqCache::set_budget(std::size_t bytes){impl_->byte_budget=bytes;}
int64_t QsaTqCache::prefill_rows(int64_t requested)const {
    if(requested<=8)return requested;
    const auto& a=*impl_;
    const auto bytes=available_prefill_workspace(*default_context(a.ks.get_device()));
    const auto allowance=bytes/4;
    if(a.host_indexer && std::size_t(a.tokens/4)*(a.index_width+2)*sizeof(std::int16_t)>allowance)
        return 8;
    const auto count=(a.budget+6)/4;
    auto rows=requested;
    // Scores and attention scratch are already query-tiled internally.
    // Charge that bounded tile once, then price only append work per input
    // row; charging every row for the complete context splits weight GEMMs
    // and repeatedly reconstructs the same historical KV layer.
    const auto score_budget=option("TYLOINFER_QSA_PREFILL_SCORE_BYTES",64ULL<<20);
    const auto append_row_bytes=std::max<int64_t>(1,int64_t(a.kv_heads)*a.padded*16+int64_t(a.index_width)*4);
    for(int attempt=0;attempt<3;++attempt) {
        auto score_pools=std::max(1,a.prefill_attention.pools);
        const auto active=(a.tokens+rows+3)/4;
        while(score_pools<active)score_pools=std::min(a.capacity,score_pools>a.capacity/2?a.capacity:2*score_pools);
        const auto queries=std::min(rows,std::max<int64_t>(16,int64_t(score_budget/(4ULL*score_pools))));
        const auto parts=tq::tq_attention_partitions(int(queries),a.heads,a.kv_heads,a.padded,count,false,true);
        const auto partial=std::max(int64_t(a.heads)*parts*a.padded*4,int64_t(a.heads)*a.padded*4+7*score_pools+8);
        const auto history=a.ram?0:int64_t(a.heads)*(count+1)*sizeof(float);
        const auto query_bytes=partial+int64_t(a.heads)*a.padded*4+int64_t(a.heads)*parts*8+
            history+8LL*score_pools+int64_t(count)*sizeof(tq::SelectedBlock);
        const auto fixed=std::size_t(queries)*query_bytes+
            tq::indexer_affine16_score_scratch(int(queries),a.index_heads,a.index_width);
        const auto next=fixed<allowance?std::min(requested,std::max<int64_t>(1,int64_t((allowance-fixed)/append_row_bytes))):
            std::min(rows,std::max<int64_t>(1,int64_t(allowance/std::max<int64_t>(1,query_bytes+append_row_bytes))));
        if(next>=rows)break;
        rows=next>=32?next/32*32:next;
    }
    return rows;
}
void QsaTqCache::begin_prefill_workspace(int64_t end_position) {
    if(impl_->ram) {
        impl_->ram->begin_prefill_stage();
        impl_->prefill_layer_end=std::size_t(end_position/4);
    }
}
void QsaTqCache::release_prefill_workspace() {
    impl_->prefill_layer_stage.reset();impl_->prefill_layer_end=0;
    impl_->prefill_attention=Impl::AttentionWorkspace{};
    // Long appends also use the small-query arena when streaming selected KV.
    impl_->decode_attention=Impl::AttentionWorkspace{};
    if(impl_->ram)impl_->ram->release_prefill_stage();
    // The archive transfer has its own stream lifetime. Decoding retains
    // separate append buffers, so each layer can return its prompt-sized one.
    impl_->archive={};
    if(!impl_->append_small) {
        impl_->append_k={};impl_->append_v={};impl_->append_index={};
        impl_->append_capacity=0;impl_->append_small=true;
    }
}
void QsaTqCache::trim_ram_stage_queries(int queries) {
    if(impl_->ram)impl_->ram->trim_stage_queries(queries);
}

int QsaTqCache::device_graph_kind(int64_t tokens)const{
    if(tokens<1 || tokens>8 || tokens>impl_->maximum-impl_->tokens)return 0;
    // Single-token completion is guarded by the live GPU position, so one
    // recorded graph covers every phase of the four-token microblock.
    const int kind=tokens==1?1:3;
    return kind+(impl_->ram && !impl_->ram->graph_safe()?4:0);
}
bool QsaTqCache::pending_graph_enabled()const{return impl_->pending_graph;}
bool QsaTqCache::graph_task_supported()const {
    // All stored encodings have immutable readers. Capability does not depend
    // on finishing history, fitting KV or reclaiming its current RAM budget.
    // Actual graph preparation settles and publishes pending versions.
    return impl_->ram && impl_->ram->graph_task_supported();
}
void QsaTqCache::prepare_decode(bool allow_pending){prepare_decode_tokens(1,allow_pending);}
void QsaTqCache::prepare_decode_tokens(int64_t queries,bool allow_pending,int host_mode,bool share_attention_workspace){
    impl_->ensure_descriptors();
    impl_->cancel_batch_warmup();
    impl_->finish_history();
    const bool prefill=impl_->prefill_attention.queries>0 || !impl_->append_small;
    impl_->prefill_attention=Impl::AttentionWorkspace{};
    if(prefill){impl_->append_k={};impl_->append_v={};impl_->append_index={};impl_->append_capacity=0;impl_->append_small=true;}
    impl_->drain();impl_->pending_graph=allow_pending && device_graph_kind(queries);
    impl_->graph_ssd_task=impl_->pending_graph && impl_->ram && host_mode==8;
    impl_->graph_host_boundary=!impl_->graph_ssd_task && impl_->pending_graph && impl_->ram && (host_mode || !impl_->ram->graph_safe());
    impl_->graph_queries=int(queries);impl_->graph_phase=int(impl_->tokens%4);
    impl_->graph_blocks=impl_->pending_graph?int((impl_->graph_phase+queries)/4):0;
    if(impl_->ram) {
        if(impl_->graph_ssd_task)impl_->ram->prepare_graph_task(std::uint64_t(impl_->tokens),int(queries));
        else impl_->ram->prepare(int(queries),true,!QsaRamDecodeBudgetScope::defer_shared_cache(impl_->device));
    }
    impl_->graph_complete=impl_->graph_blocks!=0;
    // A warmup can overwrite the publication buffer before it commits tokens.
    if(impl_->graph_complete)impl_->graph_archive_count=0;
    if(impl_->pending_graph) {
        ResidentAllocationScope resident;
        impl_->allocate_indexer();
        // This arena belongs to one fixed cache geometry and CUDA device.
        // Keep its plan across token replays, avoiding a device query per layer.
        auto& scratch=impl_->graph_workspace(int(queries));
        const int parts=scratch.parts>0?scratch.parts:
            tq::tq_attention_partitions(int(queries),impl_->heads,impl_->kv_heads,impl_->padded,(impl_->budget+6)/4,false,true);
        impl_->prepare_workspace(int(queries),impl_->capacity,parts,true,!impl_->host_indexer,
            share_attention_workspace && queries==1);
        if(queries>1) {
            auto& batch=impl_->batch_append.at(std::size_t(queries-2));
            if(!batch.key.defined()) {
                const int staged=int((queries+6)/4*4),blocks=staged/4;
                const auto f=impl_->options.dtype(tb::kFloat32),h=impl_->options.dtype(tb::kFloat16);
                batch.key=tb::empty({1,staged,impl_->kv_heads,impl_->width},f);batch.value=tb::empty_like(batch.key);
                batch.index=tb::empty({1,staged,impl_->index_width},f);
                batch.fp16_key=tb::empty({blocks,impl_->rows,256},h);batch.fp16_value=tb::empty_like(batch.fp16_key);
                batch.rotated_key=tb::empty({staged,impl_->kv_heads,impl_->padded},f);batch.rotated_value=tb::empty_like(batch.rotated_key);
                if(impl_->ram_bits!=16)batch.packed=tb::empty({int64_t(2*blocks*impl_->canonical_bytes)},impl_->options.dtype(tb::kUInt8));
            }
            if(!impl_->seed_k.defined()) {
                impl_->seed_k=tb::empty_like(impl_->tail_storage_k);impl_->seed_v=tb::empty_like(impl_->tail_storage_v);
                impl_->seed_index=tb::empty_like(impl_->tail_storage_index);
                impl_->seed_rotated_k=tb::empty_like(impl_->rotated_tail_k);impl_->seed_rotated_v=tb::empty_like(impl_->rotated_tail_v);
            }
        }
        {
            const int slots=impl_->graph_destination_slots();
            if(!impl_->graph_destination.defined()) {
                const auto storage=impl_->options.dtype(tb::kUInt8);
                auto destination=tb::empty({int64_t(sizeof(tq::TqBlockView))*slots},storage);
                // M<=8 can complete two four-token blocks. Keep addresses
                // stable when single-row and verification graphs alternate.
                const int64_t bytes=2*(2*impl_->canonical_bytes+impl_->index_archive_bytes());
                auto host=tb::empty({bytes},storage.device(tb::kCPU).pinned_memory(true));
                // Coalesced encoder stores avoid a tiny copy-engine transfer
                // between this layer and its attention kernels. Completion of
                // the model window makes the mapped archive readable by CPU.
                void* mapped=nullptr;
                if(cudaHostGetDevicePointer(&mapped,host.data_ptr(),0)!=cudaSuccess) {
                    (void)cudaGetLastError();mapped=nullptr;
                }
                auto device=mapped?tb::Tensor{}:tb::empty({bytes},storage);
                impl_->graph_archive=std::move(device);
                impl_->graph_archive_host=std::move(host);
                impl_->graph_archive_mapped=static_cast<std::uint8_t*>(mapped);
                impl_->graph_destination=std::move(destination);
                if(const auto* trace=std::getenv("TYLOINFER_TRACE_QSA_TQ");trace && trace[0]=='1')
                    std::fprintf(stderr,"native_qsa_tq_archive mapped=%d bytes=%lld\n",int(mapped!=nullptr),static_cast<long long>(bytes));
            }
            const int slot=int(impl_->tokens/4)%slots;
            const auto cycle=impl_->tokens/(4*slots);
            // GPU-only blocks retain their original payload across replays.
            // RAM-backed graphs can overwrite the same two transient slots.
            if(!impl_->canonical_slab.defined() || (!impl_->ram &&
                    (impl_->canonical_cycle!=cycle || slot+impl_->graph_blocks>slots))) {
                impl_->canonical_slab=tb::empty({int64_t(2*impl_->canonical_bytes*slots)},impl_->options.dtype(tb::kUInt8));
                impl_->canonical_cycle=cycle;
                const auto initialize=impl_->ram_bits==16?tq::tq_initialize_fp16_destinations:tq::tq_initialize_destinations;
                check(initialize(impl_->canonical_slab.data_ptr<std::uint8_t>(),impl_->graph_destination.data_ptr<tq::TqBlockView>(),
                    int(impl_->canonical_bytes),slots,mfq_current_cuda_stream()));
            }
            if(!impl_->ram && (impl_->graph_complete || queries>1)) {
                impl_->graph_image=std::make_shared<Impl::ImageBatch>();
                impl_->graph_image->payload=queries>1?impl_->canonical_slab:
                    impl_->canonical_slab.narrow(0,int64_t(2*impl_->canonical_bytes*slot),int64_t(2*impl_->canonical_bytes));
            }
        }
        const auto prior=impl_->tokens%4;
        if(prior) {
            if(impl_->tail_k.data_ptr()!=impl_->tail_storage_k.data_ptr())impl_->tail_storage_k.narrow(1,0,prior).copy_(impl_->tail_k);
            if(impl_->tail_v.data_ptr()!=impl_->tail_storage_v.data_ptr())impl_->tail_storage_v.narrow(1,0,prior).copy_(impl_->tail_v);
            if(impl_->tail_index.data_ptr()!=impl_->tail_storage_index.data_ptr())impl_->tail_storage_index.narrow(1,0,prior).copy_(impl_->tail_index);
        }
        if(queries>1) {
            check(tq::tq_copy_pending_tail(impl_->committed_tail(),impl_->seeded_tail(),impl_->kv_heads,impl_->width,impl_->padded,impl_->index_width,mfq_current_cuda_stream()));
            impl_->batch_prepared=true;
        }
    }
}
void QsaTqCache::activate_graph_cache() {
    if(impl_->ram && impl_->pending_graph && !impl_->graph_host_boundary && !impl_->graph_ssd_task)
        impl_->ram->activate_shared_cache();
}
std::size_t QsaTqCache::release_shared_device_cache(bool spare_only) {
    return impl_->ram?impl_->ram->release_shared_device_cache(spare_only):0;
}
void QsaTqCache::advance_graph(int64_t tokens){
    if(!impl_->pending_graph)return;
    if(tokens!=impl_->graph_queries || (tokens==1?impl_->graph_complete!=(impl_->tokens%4==3):impl_->graph_phase!=impl_->tokens%4))
        throw std::logic_error("QSA graph replay changed block phase");
    const auto start=impl_->tokens;
    if(impl_->graph_complete) {
        if(tokens==1)impl_->append_graph_image();else impl_->append_graph_batch();
        // Fixed Affine8 and FP16 store the graph image without requantization.
        if(impl_->ram && !impl_->ssd_cache && !impl_->adaptive_precision && (impl_->ram_bits==8 || impl_->ram_bits==16)) {
            impl_->graph_archive_first=start/4;
            impl_->graph_archive_count=impl_->graph_blocks;
        }
    }
    impl_->batch_prepared=false;
    impl_->tokens+=tokens;const auto remainder=impl_->tokens%4;
    impl_->tail_k=remainder?impl_->tail_storage_k.narrow(1,0,remainder):Tensor{};
    impl_->tail_v=remainder?impl_->tail_storage_v.narrow(1,0,remainder):Tensor{};
    impl_->tail_index=remainder?impl_->tail_storage_index.narrow(1,0,remainder):Tensor{};
    impl_->update_history(start);
}
QsaTqCache::Tensor QsaTqCache::forward_pending_graph(const Tensor& q,const Tensor& k,const Tensor& v,const Tensor& iq,const Tensor& raw,
    const Tensor& positions,const Tensor& gate,Tensor output,const Pool& pool,const Tensor& absolute,const CompletePool& complete_pool){
    return impl_->forward_pending_graph(q,k,v,iq,raw,positions,gate,std::move(output),pool,absolute,complete_pool);}
QsaTqCache::Tensor QsaTqCache::forward(const Tensor& q,const Tensor& k,const Tensor& v,const Tensor& iq,const Tensor& raw,
    const Tensor& positions,const Pool& pool,bool history_only,const Tensor& gate,Tensor output,bool fp16_math,CudaProfiler* profiler) {
    return impl_->forward(q,k,v,iq,raw,positions,pool,history_only,gate,std::move(output),fp16_math,profiler);
}
std::shared_ptr<void> QsaTqCache::hold_host_workspace(std::size_t bytes) {
    return impl_->ram?impl_->ram->hold_workspace(bytes):std::shared_ptr<void>{};
}
QsaRamWorkspaceProgress QsaTqCache::reclaim_shared_host_workspace() {
    return impl_->ram?impl_->ram->reclaim_shared_workspace():QsaRamWorkspaceProgress{};
}
std::size_t QsaTqCache::reclaim_pinned_for_admission(std::size_t bytes) {
    return impl_->ram?impl_->ram->reclaim_pinned_for_admission(bytes):0;
}
std::optional<QsaRamMemory> QsaTqCache::ram_memory()const {
    if(!impl_->ram)return {};auto result=impl_->ram->memory();
    if(impl_->raw_index_store)result.index_disk=impl_->raw_index_store->stats().disk_bytes;
    return result;
}
std::vector<QsaTqCache::Tensor> QsaTqCache::resident_tensors()const {
    const auto& s=*impl_;
    std::vector<Tensor> result{s.ks,s.vs,s.relative,s.host_metadata?Tensor{}:s.views,s.index_codes,s.index_scales,s.index_zero,s.hits,s.mass,
        s.tail_storage_k,s.tail_storage_v,s.tail_storage_index,s.rotated_tail_k,s.rotated_tail_v,s.canonical_slab};
    if(s.ram) {
        auto ram=s.ram->resident_tensors();result.insert(result.end(),ram.begin(),ram.end());
    }else for(const auto& owner:s.owners)if(owner)result.push_back(owner->payload);
    return result;
}
tq::AdaptiveCacheStats QsaTqCache::stats()const{impl_->finish_history(false);return impl_->ram?impl_->ram->stats():impl_->cache->stats();}
std::uint64_t QsaTqCache::ssd_read_bytes()const {
    return (impl_->ram?impl_->ram->stats():impl_->cache->stats()).canonical_read_bytes;
}
bool QsaTqCache::supports_session()const{return bool(impl_->ram);}
std::shared_ptr<QsaTqSession> QsaTqCache::capture_session(std::int64_t count) {
    auto& s=*impl_;
    if(!s.ram || count<=0 || count>s.tokens)throw std::invalid_argument("Invalid RAM QSA session boundary");
    wait();const auto complete=std::size_t(count/4);const auto remainder=count%4;
    auto state=std::make_shared<QsaTqSession>();state->tokens=count;state->heads=s.heads;state->kv_heads=s.kv_heads;
    state->width=s.width;state->padded=s.padded;state->index_width=s.index_width;state->ram_bits=s.adaptive_precision?0:s.ram_bits;
    state->kv=s.ram->snapshot(complete);state->index.assign(s.index_archive.begin(),s.index_archive.begin()+complete);
    const auto host=[](const Tensor& input){return input.to(tb::kCPU).contiguous();};
    if(!remainder)return state;
    if(count/4==s.tokens/4 && s.tail_k.defined()) {
        state->tail_keys=host(s.tail_k.narrow(1,0,remainder));state->tail_values=host(s.tail_v.narrow(1,0,remainder));
        state->tail_index=host(s.tail_index.narrow(1,0,remainder));return state;
    }
    const auto image=s.ram->reconstruction_image(complete);
    std::vector<std::uint8_t> data=image.key;data.insert(data.end(),image.value.begin(),image.value.end());
    auto bytes=upload(data.data(),data.size(),s.options);
    tq::TqBlockView record{bytes.data_ptr<std::uint8_t>(),bytes.data_ptr<std::uint8_t>()+image.key.size(),
        tq::tq_precision_flags(image.key_bits,image.key_state_bits),tq::tq_precision_flags(image.value_bits,image.value_state_bits)};
    auto view=upload(&record,sizeof(record),s.options);
    auto km=tb::empty({s.rows,256},s.options.dtype(tb::kFloat32)),vm=tb::empty_like(km);
    auto k=tb::empty({1,4,s.kv_heads,s.width},s.options.dtype(tb::kFloat32)),v=tb::empty_like(k);
    check(tq::tq_decode_modes(view.data_ptr<tq::TqBlockView>(),km.data_ptr<float>(),vm.data_ptr<float>(),s.relative.data_ptr<std::uint16_t>(),1,s.kv_heads,s.padded,mfq_current_cuda_stream()));
    check(tq::tq_restore_modes(km.data_ptr<float>(),vm.data_ptr<float>(),s.ks.data_ptr<float>(),s.vs.data_ptr<float>(),k.data_ptr<float>(),v.data_ptr<float>(),1,s.kv_heads,s.width,s.padded,mfq_current_cuda_stream()));
    state->tail_keys=host(k.narrow(1,0,remainder));state->tail_values=host(v.narrow(1,0,remainder));
    const auto index=s.index_archive.at(complete)->read();
    if(index->size()!=s.index_archive_bytes())throw std::runtime_error("QSA snapshot raw index length changed");
    auto raw=tb::empty({1,4,s.index_width},s.options.device(tb::kCPU).dtype(tb::kFloat16));
    std::memcpy(raw.data_ptr(),index->data(),8*s.index_width);
    state->tail_index=raw.to(tb::kFloat32).narrow(1,0,remainder).contiguous();return state;
}
std::shared_ptr<void> QsaTqCache::reserve_session_restore(const QsaTqSession& state) {
    auto& s=*impl_;
    if(!s.ram || state.tokens<=0 || state.tokens>s.maximum || state.heads!=s.heads || state.kv_heads!=s.kv_heads ||
            state.width!=s.width || state.padded!=s.padded || state.index_width!=s.index_width || state.ram_bits!=(s.adaptive_precision?0:s.ram_bits) ||
            !state.kv || state.kv->blocks.size()!=std::size_t(state.tokens/4) || state.index.size()!=state.kv->blocks.size())
        throw std::invalid_argument("Incompatible RAM QSA session geometry");
    // Allocate the required locked Indexer storage while the request is
    // still being admitted, before any layer starts hydrating optional KV.
    s.ensure_descriptors();
    if(s.host_indexer && state.borrow_prefix)s.allocate_indexer();
    return s.ram->reserve_restore(*state.kv,state.borrow_prefix);
}
void QsaTqCache::enable_prefix_sharing() {
    if(impl_->ram)impl_->ram->enable_prefix_sharing();
}
bool QsaTqCache::restore_can_place_in_ram(std::size_t bytes) const {
    return impl_->ram && impl_->ram->restore_can_place_in_ram(bytes);
}
void QsaTqCache::restore_session(const QsaTqSession& state,bool borrow_prefix,std::span<const QsaTqCache* const> peers) {
    auto& s=*impl_;const auto count=state.tokens,remainder=count%4;
    if(!s.ram || count<=0 || count>s.maximum || state.heads!=s.heads || state.kv_heads!=s.kv_heads ||
            state.width!=s.width || state.padded!=s.padded || state.index_width!=s.index_width || state.ram_bits!=(s.adaptive_precision?0:s.ram_bits) ||
            !state.kv || state.kv->blocks.size()!=std::size_t(count/4) || state.index.size()!=state.kv->blocks.size())
        throw std::invalid_argument("Incompatible RAM QSA session geometry");
    if(remainder)for(const auto* tail:{&state.tail_keys,&state.tail_values,&state.tail_index})
        if(!tail->defined() || tail->size(0)!=1 || tail->size(1)!=remainder || tail->scalar_type()!=tb::kFloat32)
            throw std::invalid_argument("Incompatible RAM QSA session tail");
    if(remainder && (state.tail_keys.dim()!=4 || state.tail_keys.size(2)!=s.kv_heads || state.tail_keys.size(3)!=s.width ||
            state.tail_values.sizes()!=state.tail_keys.sizes() || state.tail_index.dim()!=3 || state.tail_index.size(2)!=s.index_width))
        throw std::invalid_argument("Incompatible RAM QSA session tail width");
    s.truncate(0);
    try {
        s.ensure_descriptors();
        s.allocate_indexer();
        const auto blocks=state.index.size();
        const Impl* peer=nullptr;
        std::vector<const QsaRamCache*> ram_peers;
        if(borrow_prefix)for(const auto* candidate:peers)if(candidate && candidate!=this) {
            const auto* source=candidate->impl_.get();
            if(source->device!=s.device || source->index_width!=s.index_width || !source->index_codes.defined())continue;
            if(!peer)peer=source;
            if(source->ram)ram_peers.push_back(source->ram.get());
        }
        struct IndexRange {std::size_t first,last;};
        std::vector<IndexRange> copies;
        std::size_t copied=0;
        if(peer) {
            const auto peer_blocks=std::min({blocks,peer->index_archive.size(),std::size_t(peer->tokens/4)});
            for(std::size_t id=0;id<peer_blocks;++id) {
                const auto& requested=state.index[id];
                const auto& live=peer->index_archive[id];
                // The retained file reader identifies the validated version.
                // Equal token counts alone do not prove equal index codes.
                if(!requested || !live || !requested->source || !live->source ||
                    !requested->source->file_reader || requested->source->file_reader!=live->source->file_reader ||
                    requested->source->file_offset!=live->source->file_offset ||
                    requested->offset!=live->offset || requested->bytes!=live->bytes)continue;
                if(copies.empty() || copies.back().last!=id)copies.push_back({id,id});
                copies.back().last=id+1;
                ++copied;
            }
        }
        tyloinfer::ScopedFileReadWindow reads(borrow_prefix);
        for(const auto& [first,last]:copies) {
            const auto length=last-first;
            s.index_codes.narrow(0,first,length).copy_(peer->index_codes.narrow(0,first,length));
            s.index_scales.narrow(0,first,length).copy_(peer->index_scales.narrow(0,first,length));
            s.index_zero.narrow(0,first,length).copy_(peer->index_zero.narrow(0,first,length));
        }
        if(copied) {
            check(cudaStreamSynchronize(mfq_current_cuda_stream()));
            if(std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
                "native_qsa_prefix_index_copy blocks=%zu total=%zu ranges=%zu source_host=%d target_host=%d\n",
                copied,blocks,copies.size(),int(peer->host_indexer),int(s.host_indexer));
        }
        if(borrow_prefix)s.index_archive=state.index;
        if(copied==blocks) {
            s.ram->restore(*state.kv,borrow_prefix,{},ram_peers);
        } else {
            const auto stride=std::size_t(s.index_width+2)*2;
            const auto step=std::max<std::size_t>(1,std::min(blocks-copied,(2U<<20)/stride));
            auto staging=tb::empty({int64_t(step*stride)},s.options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
            std::size_t begin=0,filled=0;
            std::vector<mfq::FileReadRange> index_reads;
            if(borrow_prefix)index_reads.reserve(step);
            const auto flush=[&] {
                if(!filled)return;
                if(!index_reads.empty()) {
                    mfq::FileRangeReader::read_many(index_reads);
                    index_reads.clear();
                }
                const auto* input=staging.data_ptr<std::uint8_t>();const auto n=filled;
                if(s.host_indexer) {
                    for(std::size_t row=0;row<n;++row) {
                        const auto* record=input+row*stride;const auto id=begin+row;
                        std::memcpy(s.index_codes.data_ptr<std::int16_t>()+id*s.index_width,record,s.index_width*2);
                        std::memcpy(s.index_scales.data_ptr<std::uint16_t>()+id,record+s.index_width*2,2);
                        std::memcpy(s.index_zero.data_ptr<std::int16_t>()+id,record+s.index_width*2+2,2);
                    }
                    begin+=filled;filled=0;return;
                }
                check(cudaMemcpy2DAsync(s.index_codes.data_ptr<std::int16_t>()+begin*s.index_width,s.index_width*2,
                    input,stride,s.index_width*2,n,cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
                check(cudaMemcpy2DAsync(s.index_scales.data_ptr<std::uint16_t>()+begin,2,
                    input+2*s.index_width,stride,2,n,cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
                check(cudaMemcpy2DAsync(s.index_zero.data_ptr<std::int16_t>()+begin,2,
                    input+2*s.index_width+2,stride,2,n,cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
                check(cudaStreamSynchronize(mfq_current_cuda_stream()));
                begin+=filled;filled=0;
            };
            auto reused=copies.begin();
            s.ram->restore(*state.kv,borrow_prefix,[&](std::size_t id) {
                while(reused!=copies.end() && id>=reused->last)++reused;
                if(reused!=copies.end() && id>=reused->first) {flush();return;}
                if(!filled)begin=id;
                // KV and its index archive are adjacent on disk. Reuse that window,
                // but do not prefetch unneeded KV when the block remains SSD-only.
                tyloinfer::ScopedFileReadWindow exact_index_reads(false);
                if(!state.index[id])throw std::runtime_error("Missing QSA snapshot index archive");
                auto* destination=staging.data_ptr<std::uint8_t>()+filled*stride;
                if(borrow_prefix) {
                    state.index[id]->read_slice_into(8*s.index_width,destination,stride,s.index_archive_bytes(),&index_reads);
                }else {
                    const auto bytes=state.index[id]->read();
                    if(bytes->size()!=s.index_archive_bytes())throw std::runtime_error("Incompatible QSA snapshot index archive length");
                    std::memcpy(destination,bytes->data()+8*s.index_width,stride);
                    s.save_index(std::vector<std::uint8_t>(*bytes));
                }
                if(++filled==step)flush();
            },ram_peers);
            flush();
        }
        s.tokens=count;s.pending_graph=false;s.graph_image.reset();s.hits.zero_();
        if(s.mass.defined())s.mass.zero_();
        if(remainder) {
            s.tail_k=state.tail_keys.to(s.options.dtype(tb::kFloat32)).contiguous();
            s.tail_v=state.tail_values.to(s.options.dtype(tb::kFloat32)).contiguous();
            s.tail_index=state.tail_index.to(s.options.dtype(tb::kFloat32)).contiguous();
            check(tq::tq_prepare_modes(s.tail_k.data_ptr(),s.tail_v.data_ptr(),false,s.ks.data_ptr<float>(),s.vs.data_ptr<float>(),
                s.pending_modes_k.data_ptr<float>(),s.pending_modes_v.data_ptr<float>(),s.pending_k.data_ptr<std::uint16_t>(),s.pending_v.data_ptr<std::uint16_t>(),
                int(remainder),s.kv_heads,s.width,s.padded,mfq_current_cuda_stream(),s.rotated_tail_k.data_ptr<float>(),s.rotated_tail_v.data_ptr<float>()));
            s.publish_pending(s.pending_k,s.pending_v,count/4);
        }
    }catch(...) {
        // A failed companion read can follow already published KV batches.
        // Keep an empty, retryable cache even though tokens was still zero.
        try{s.ram->truncate(0);}catch(...){}
        s.index_archive.clear();s.tokens=0;s.pending_graph=false;s.graph_image.reset();
        s.tail_k={};s.tail_v={};s.tail_index={};
        throw;
    }
}
} // namespace mfq::cuda
