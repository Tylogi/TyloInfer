#include "runtime/routed_host_cache.h"
#include "runtime/routed_event_retirement.h"
#include "runtime/routed_prefill_routes.h"
#include "runtime/routed_prefill_workspace_plan.h"
#include "storage/routed_host_image.h"
#include "runtime/routed_admission_executor.h"
#include "runtime/routed_resident_views.h"
#include "runtime/routed_moe_schedule.h"
#include "runtime/routed_completion_order.h"
#include "tyloinfer/host_copy.h"
#include "runtime/routed_prefill_pipeline.h"
#include "runtime/prefill_projection_plan.h"
#include "gguf_prefill_fusion.h"
#include "kernels/gguf.h"
#include "runtime/routed_workspace_plan.h"
#include "runtime/routed_prefill_loan.h"
#include "runtime/routed_graph_pools.h"
#include "runtime/routed_expert_journal.h"
#include "runtime/routed_expert_groups.h"
#include "tq.h"
#include "tq_detail.h"
#include "tyloinfer/tq_rows.h"
#include "tyloinfer/runtime.h"
#include "mfq_weight_memory.h"
#include "storage/moe_expert_cache.h"
#include "storage/moe_pinned_expert_arena.h"
#include "storage/device_image_arena.h"
#include "storage/pinned_expert_reuse.h"
#include "storage/routed_prefill_stager.h"
#include "runtime/routed_image_layout.h"
#include "quant_linear.h"
#include "moe.h"
#include "mfq_cuda_context.h"
#include "mfq_cuda_resident_allocations.h"
#include "mfq_cuda_activation_ops.h"
#include "runtime/moe_pipeline.h"
#include "runtime/decode_window.h"
#include "runtime/independent_branch.h"
#include "runtime/routed_transfer.h"
#include "runtime/routed_stream_fence.h"
#include "runtime/expert_route_usage.h"
#include "runtime/routed_adaptive_slots.h"
#include "runtime/routed_allocation_capacity.h"
#include "runtime/execution_options.h"
#include "tyloinfer/host_memory.h"
#include "tyloinfer/cpu_wait.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <list>
#include <mutex>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <climits>
#include <exception>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace tb=mfq_tensor_backend;
using tq_detail::logical_shape;
using tq_detail::row_parts;
using tq_detail::join_rows;
using tq_detail::tq_image;
using tq_detail::status;

// Fence the actual eager reads before returning their image leases. Recording
// consumers only when storage is finally freed can place a wait behind a later
// graph that is itself waiting for an upload on the allocation stream.
class TqExpertReadFence {
    MfqCudaStream consumer_;
    std::vector<MfqCudaStream> producers_;
public:
    TqExpertReadFence(std::initializer_list<const std::vector<TqWeight>*> banks,int device)
        :consumer_(mfq_get_current_cuda_stream(device)) {
        for(const auto* bank:banks)for(const auto& weight:*bank) {
            if(!weight.allocation_stream || weight.allocation_stream->stream()==consumer_.stream())continue;
            const auto& producer=*weight.allocation_stream;
            if(std::none_of(producers_.begin(),producers_.end(),[&](const auto& s){return s.stream()==producer.stream();}))
                producers_.push_back(producer);
        }
    }
    ~TqExpertReadFence()noexcept {
        if(producers_.empty())return;
        try {
            const MfqCudaGuard guard(consumer_.device());
            mfq::cuda::Event consumed(cudaEventDisableTiming);consumed.record(consumer_.stream());
            for(const auto& producer:producers_)
                TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(producer.stream(),consumed.get(),0));
        }catch(...) {(void)cudaStreamSynchronize(consumer_.stream());}
    }
};

// Pinned plans are reused only after their last consumer completes. Keeping
// them with model resources avoids a host allocation/free barrier per batch.
struct TqPrefillPlan {
    tb::Tensor host,device;
    mfq::cuda::Event done;
    ~TqPrefillPlan()noexcept {try{done.synchronize();}catch(...) {}}
};
struct TqPrefillPlanLease {
    std::shared_ptr<TqPrefillPlan> plan;
    cudaStream_t stream;
    ~TqPrefillPlanLease()noexcept {
        // Also fence uploads on exceptional exits before allowing reuse.
        if(cudaEventRecord(plan->done.get(),stream)!=cudaSuccess)(void)cudaStreamSynchronize(stream);
    }
};

struct TqPrefillWeightStage {
    tb::Tensor image;
    std::size_t cursor=0;
    tb::Tensor allocate(std::size_t bytes) {
        if(bytes>SIZE_MAX-15)throw std::overflow_error("TQ prefill image size overflow");
        bytes=(bytes+15)&~std::size_t(15);
        if(cursor>image.nbytes() || bytes>image.nbytes()-cursor)
            throw MfqResourceExhausted("TQ prefill weight stage is exhausted");
        auto result=image.narrow(0,int64_t(cursor),int64_t(bytes));cursor+=bytes;return result;
    }
};

struct TqPrefillWeightPipeline {
    struct Stage {
        TqPrefillWeightStage workspace;
        std::array<std::vector<TqWeight>,3> banks;
    };
    MfqCudaStream compute,upload;
    std::array<Stage,mfq::cuda::RoutedPrefillPipeline::depth> stages;
    mfq::cuda::RoutedPrefillPipeline schedule;
    TqPrefillWeightPipeline(MfqCudaStream parent,MfqCudaStream transfer)
        :compute(std::move(parent)),upload(std::move(transfer)),schedule(compute.stream(),upload.stream()) {}
};

// cuBLAS configuration is mutable. Give each execution stream a stable handle
// so simultaneous prefill requests never redirect each other's GEMM work.
// Weights and expert caches retain the pool, including streams/workspaces used
// by captured graphs; its lifetime follows the loaded TQ model resources.
class TqGemmResources {
    struct Entry {
        MfqCudaStream stream;
        std::unique_ptr<mfq::cuda::BlasHandle> handle;
    };
    int device_;
    bool indexed_mma_=false;
    bool prefill_mma_=false;
    bool host_register_=false;
    std::mutex mutex_;
    std::unordered_map<cudaStream_t,Entry> handles_;
    std::vector<std::shared_ptr<TqPrefillPlan>> prefill_plans_;
public:
    explicit TqGemmResources(int device):device_(device) {
        int major=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
        // The m16n8k16 FP16 instruction used by wide row tiles requires SM80.
        indexed_mma_=major>=8;
        // FP16 m16n16k16 WMMA is available from SM70. cp.async is optional.
        prefill_mma_=major>=7;
        int supported=0;
        const auto registered=cudaDeviceGetAttribute(&supported,cudaDevAttrHostRegisterSupported,device);
        if(registered==cudaSuccess)host_register_=supported!=0;
        else if(registered==cudaErrorInvalidValue)(void)cudaGetLastError();
        else TYLOINFER_CUDA_CHECK(registered);
    }
    bool indexed_mma()const noexcept{return indexed_mma_;}
    bool prefill_mma()const noexcept{return prefill_mma_;}
    bool host_register()const noexcept{return host_register_;}
    std::shared_ptr<TqPrefillPlan> prefill_plan(std::int64_t bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        for(auto it=prefill_plans_.begin();it!=prefill_plans_.end();) {
            if(it->use_count()!=1 || !(*it)->done.ready()){++it;continue;}
            if((*it)->host.numel()>=bytes) {
                mfq_cuda_record_stream((*it)->device,mfq_get_current_cuda_stream(device_));
                return *it;
            }
            // Drop obsolete, completed capacities rather than accumulating
            // every prompt size ever used by the model.
            it=prefill_plans_.erase(it);
        }
        auto plan=std::make_shared<TqPrefillPlan>();
        const auto capacity=(bytes+4095)&~std::int64_t(4095);
        plan->host=tb::empty({capacity},tb::TensorOptions().dtype(tb::kUInt8).pinned_memory(true));
        {
            const mfq::cuda::ResidentAllocationScope resident;
            plan->device=tb::empty({capacity},tb::TensorOptions().dtype(tb::kUInt8).device(tb::Device(tb::kCUDA,device_)));
        }
        prefill_plans_.push_back(plan);
        return plan;
    }
    void retain(const MfqCudaStream& stream) {
        std::lock_guard<std::mutex> guard(mutex_);
        handles_.try_emplace(stream.stream(),Entry{stream,{}});
    }
    cublasHandle_t get() {
        const auto stream=mfq_get_current_cuda_stream(device_);
        std::lock_guard<std::mutex> guard(mutex_);
        auto found=handles_.find(stream.stream());
        if(found==handles_.end())found=handles_.emplace(stream.stream(),Entry{stream,{}}).first;
        if(!found->second.handle) {
            auto handle=std::make_unique<mfq::cuda::BlasHandle>(device_);
            handle->set_stream(stream.stream());
            found->second.handle=std::move(handle);
        }
        return found->second.handle->get();
    }
};

namespace {
std::shared_ptr<TqGemmResources> tq_gemm_resources(int device) {
    static std::mutex mutex;
    static std::unordered_map<int,std::weak_ptr<TqGemmResources>> pools;
    std::lock_guard<std::mutex> guard(mutex);
    if(auto existing=pools[device].lock())return existing;
    auto result=std::make_shared<TqGemmResources>(device);pools[device]=result;
    return result;
}
int dtype(tb::ScalarType type) {
    if(type==tb::kFloat16)return 0;
    if(type==tb::kBFloat16)return 1;
    if(type==tb::kFloat32)return 2;
    throw std::invalid_argument("TQ activation dtype must be FP16, BF16 or FP32");
}

int gemm_min_tokens(tb::ScalarType type) {
    // Token reuse handles short decode windows. Larger groups amortize the
    // transient weight decode and GEMM launch; no model/device restriction.
    return type==tb::kFloat16?32:128;
}

bool short_q8(tb::ScalarType type,int64_t tokens) {
    return tokens>=1 && tokens<=4 &&
        (type==tb::kFloat32 || type==tb::kFloat16);
}

void project_tokens(const TqWeight& weight,const void* input,void* output,int tokens,
        int type,void* workspace=nullptr,std::uint64_t workspace_bytes=0) {
    if(workspace && (type==0 || type==2) && tokens<=4) {
        status(mfq_tq_q8_tokens(weight.view,input,output,tokens,1,type,workspace,workspace_bytes,mfq_current_cuda_stream()));
    }else if(workspace && tokens>=(type==0?32:128)) {
        if(type==0)status(mfq_tq_gemm(weight.view,input,output,tokens,workspace,workspace_bytes,
            weight.gemm->get(),mfq_current_cuda_stream()));
        else status(mfq_tq_float_gemm(weight.view,input,output,tokens,1,type,workspace,workspace_bytes,
            weight.gemm->get(),mfq_current_cuda_stream()));
    }else status(mfq_tq_gemv_tokens(weight.view,input,output,tokens,type,mfq_current_cuda_stream()));
}

tb::Tensor projection_workspace(const TqWeight& weight,int tokens,tb::ScalarType type,int groups=1) {
    // A single-token projection fitting one output tile cannot amortize a
    // separate Q8 preparation launch. The direct TQ kernel also cooperates
    // across warps on these narrow contractions, instead of serializing K
    // within one warp. Parallel projection groups still share Q8 preparation.
    if(tokens==1 && weight.out<=TRELLIS_GROUP_ROWS)return {};
    if(short_q8(type,tokens) && groups<=65535) {
        const auto columns=(std::uint64_t(weight.neuron_len)+511)/512*512;
        const auto bytes=std::uint64_t(groups)*tokens*columns*3/2;
        return tb::empty({static_cast<std::int64_t>(bytes)},weight.image.options().dtype(tb::kUInt8));
    }
    if(tokens<gemm_min_tokens(type))return {};
    // Bound temporary decode storage independently of the matrix size and
    // prompt length; experts retain only their compressed images in caches.
    auto bytes=std::max<std::uint64_t>(std::uint64_t(8)<<20,
        type==tb::kFloat16?
            std::uint64_t(weight.neuron_len)*2+15+(weight.view.bias?std::uint64_t(tokens)*4:0):
            std::uint64_t(weight.neuron_len)*8+36);
    if(type==tb::kFloat16 && tokens>=256) {
        // A larger row tile amortizes decode/GEMM submissions and avoids
        // narrow GEMMs on long prefill. This remains temporary storage, bounded
        // by useful matrix rows, 64 MiB, and 1/16 of currently free VRAM.
        const auto row_bytes=std::uint64_t(weight.neuron_len)*2+(weight.view.bias?std::uint64_t(tokens)*4:0);
        const auto wanted=std::min<std::uint64_t>(std::uint64_t(64)<<20,
            std::min<std::uint64_t>(weight.out/groups,65535)*row_bytes+15);
        if(wanted>bytes) {
            std::size_t available=0,total=0;TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&available,&total));
            bytes=std::max<std::uint64_t>(bytes,std::min<std::uint64_t>(wanted,available/16));
        }
    }
    if(bytes>(std::uint64_t(8)<<20) && std::getenv("TYLOINFER_TQ_PROFILE_CACHE"))
        std::fprintf(stderr,"tq_projection_workspace tokens=%d K=%lld N=%lld groups=%d bytes=%llu\n",
            tokens,static_cast<long long>(weight.neuron_len),static_cast<long long>(weight.out),groups,
            static_cast<unsigned long long>(bytes));
    return tb::empty({static_cast<std::int64_t>(bytes)},weight.image.options().dtype(tb::kUInt8));
}
}

namespace {
TqWeight tq_view(tb::Tensor image,const mfq::TqLayout& m,std::int64_t columns) {
    if(columns<1 || columns>m.width || m.rows>INT_MAX || m.width>INT_MAX)
        throw std::invalid_argument("invalid TQ CUDA logical shape");
    const auto words_offset=(m.packed+3)&~std::size_t(3),relative_offset=words_offset+m.packed_bytes;
    TqWeight result;result.image=std::move(image);
    if(result.image.is_cuda())result.gemm=tq_gemm_resources(result.image.get_device());
    auto* data=result.image.data_ptr<std::uint8_t>();
    int gslog=0;for(auto gs=m.gs;gs>1;gs>>=1)++gslog;
    result.view={reinterpret_cast<const std::uint32_t*>(data+words_offset),
        reinterpret_cast<const std::uint64_t*>(data+64),data+m.scales,data+relative_offset,nullptr,
        static_cast<int>(m.rows),static_cast<int>(m.width),static_cast<int>(columns),gslog,int(m.columns)};
    result.out=m.rows;result.neuron_len=columns;
    return result;
}
}
TqWeight to_device_tq(const std::vector<std::uint8_t>& blob,std::int64_t columns,int device,bool cuda) {
    const mfq::TqLayout m(blob.data(),blob.size());
    const auto image=tq_image(blob,m);
    const MfqCudaGuard guard(device);
    auto storage=tb::from_blob(const_cast<std::uint8_t*>(image.data()),{static_cast<std::int64_t>(image.size())},
        tb::TensorOptions().dtype(tb::kUInt8)).clone();
    if(cuda)storage=storage.to(tb::Device(tb::kCUDA,device),false,false).contiguous();
    return tq_view(std::move(storage),m,columns);
}

std::array<std::int64_t,2> tq_projection_shape(const mfq::ModelSource& source,const std::string& name) {
    const auto shape=logical_shape(source,name);
    if(shape.size()!=2)throw std::invalid_argument("ordinary TQ projection must be a matrix: "+name);
    const auto* record=source.find_tensor(name);
    if(!record)throw std::invalid_argument("missing TQ projection: "+name);
    (void)mfq::tq_tensor_shape(record->nbytes,source.tensor_reader(name),shape);
    return {shape[0],shape[1]};
}

TqWeight load_tq_weight(const mfq::ModelSource& source,const std::string& name,int device,bool cuda) {
    const auto shape=tq_projection_shape(source,name);
    auto bytes=source.read(name);
    std::vector<std::uint8_t> blob(bytes.size());
    std::memcpy(blob.data(),bytes.data(),bytes.size());
    auto result=to_device_tq(blob,shape[1],device,cuda);
    if(result.out!=shape[0])throw std::invalid_argument("TQ projection output shape differs: "+name);
    return result;
}

TqWeight TqWeight::with_image(tb::Tensor storage)const {
    if(!storage.defined() || !storage.is_contiguous() || storage.scalar_type()!=tb::kUInt8 ||
            storage.numel()!=image.numel())
        throw std::invalid_argument("TQ replacement image must preserve contiguous encoded bytes");
    TqWeight result=*this;
    result.image=std::move(storage);
    result.gemm=result.image.is_cuda()?tq_gemm_resources(result.image.get_device()):nullptr;
    const auto old=reinterpret_cast<std::uintptr_t>(image.data_ptr());
    const auto target=reinterpret_cast<std::uintptr_t>(result.image.data_ptr());
    auto address=[&](const void* p) {
        if(!p)return std::uintptr_t(0);
        const auto offset=reinterpret_cast<std::uintptr_t>(p)-old;
        if(offset>=image.nbytes())throw std::invalid_argument("TQ view points outside its image");
        return target+offset;
    };
    result.view.words=reinterpret_cast<const std::uint32_t*>(address(view.words));
    result.view.descriptors=reinterpret_cast<const std::uint64_t*>(address(view.descriptors));
    result.view.scales=reinterpret_cast<const void*>(address(view.scales));
    result.view.relative=reinterpret_cast<const void*>(address(view.relative));
    result.view.bias=reinterpret_cast<const float*>(address(view.bias));
    return result;
}
namespace {
TqWeight promote_tq(const TqWeight& weight,const tb::Device& device) {
    return weight.with_image(weight.image.to(device,false,false).contiguous());
}

tb::Tensor cached_tq_prefill_exponents(const TqWeight& w) {
    return prefill_weight_exponents(w.image,w.out,w.neuron_len,
        {1,reinterpret_cast<std::uintptr_t>(w.view.words),reinterpret_cast<std::uintptr_t>(w.view.descriptors),
         reinterpret_cast<std::uintptr_t>(w.view.scales),reinterpret_cast<std::uintptr_t>(w.view.relative),
         std::uintptr_t(w.view.stored_columns),std::uintptr_t(w.view.gslog),std::uintptr_t(w.view.columns)},
        [&](const tb::Tensor& exponents) {
            status(mfq_tq_prefill_row_exponents(w.view,0,int(w.out),exponents.data_ptr<int32_t>(),
                mfq_get_current_cuda_stream(w.image.get_device()).stream()));
        });
}

tb::Tensor try_tq_prefill_projection(const TqWeight& weight,const tb::Tensor& input,int64_t groups=1,
        const tb::Tensor& residual={}) {
    const auto type=input.scalar_type();
    const auto tokens=input.numel()/(groups*weight.neuron_len);
    const auto* setting=mfq::cuda::prefill_tensorcore_setting();
    // Match GGUF's automatic reduced-accumulation prefill mode. Biased
    // projections retain their bias-before-output-rounding implementation.
    if(tokens<=32 || ((!setting || setting[0]=='a') && tokens<512) ||
       (type!=tb::kFloat16 && type!=tb::kFloat32) || weight.view.bias ||
       (setting && setting[0]!='1' && setting[0]!='a') ||
       !mfq::cuda::gguf::grouped_f16_available(input.get_device()))return {};
    if(residual.defined() && (groups!=1 || type!=tb::kFloat16 ||
            !weight.can_forward_prefill_residual(tokens,residual)))return {};
    if(groups==1 && type==tb::kFloat32) {
        const auto product=prefill_product(prefill_projection_weight(weight),input);
        if(product.defined()) {
            auto shape=input.sizes().vec();shape.back()=weight.out;
            return prefill_materialize(product).reshape(shape);
        }
    }
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    const auto row_count=weight.out/groups;
    const bool scaled=type==tb::kFloat32;
    const auto cached_exponents=scaled?cached_tq_prefill_exponents(weight):tb::Tensor{};
    if(scaled && !cached_exponents.defined())return {};
    const auto bytes_per_row=std::uint64_t(weight.neuron_len)*2+(scaled?std::uint64_t(tokens)*2:0);
    const auto tile=std::min<int64_t>(65535,mfq::cuda::prefill_weight_rows(row_count,bytes_per_row,32,
        input.get_device(),true));
    auto shape=input.sizes().vec();shape.pop_back();if(groups==1)shape=input.sizes().vec();shape.back()=weight.out;
    auto output=tb::empty({tokens,weight.out},input.options());
    tb::Tensor residual_workspace;
    if(residual.defined())mfq_cuda_record_stream(residual,stream);
    const auto decoder=tile<row_count?mfq::cuda::stream_from_pool(false,input.get_device()):stream;
    const int buffers=tile<row_count?2:1;
    std::array<tb::Tensor,2> decoded,halves,exponents;
    std::array<mfq::cuda::Event,2> ready,consumed;
    for(int i=0;i<buffers;++i) {
        decoded[i]=tb::empty({tile,weight.neuron_len},input.options().dtype(tb::kFloat16));
        if(scaled)halves[i]=decoded[i];
        for(const auto& tensor:{decoded[i],halves[i],exponents[i]})if(tensor.defined()) {
            mfq_cuda_record_stream(tensor,stream);mfq_cuda_record_stream(tensor,decoder);
        }
    }
    auto ids=scaled?tb::Tensor{}:tb::arange(weight.out,input.options().dtype(tb::kInt64));
    auto product=scaled?tb::empty({tokens,tile},input.options().dtype(tb::kFloat16)):tb::Tensor{};
    if(ids.defined())mfq_cuda_record_stream(ids,decoder);
    if(cached_exponents.defined())mfq_cuda_record_stream(cached_exponents,decoder);
    mfq_cuda_record_stream(weight.image,decoder);
    mfq::cuda::Event source_ready;source_ready.record(stream.stream());
    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(decoder.stream(),source_ready.get(),0));
    for(int64_t group=0;group<groups;++group) {
        auto source=groups==1?input.reshape({tokens,weight.neuron_len}):
            input.reshape({tokens,groups,weight.neuron_len}).select(1,group).contiguous();
        tb::Tensor packed,input_exponents;
        if(scaled) {
            if(!GgufPrefillInputScope::lookup(source,packed,input_exponents)) {
                packed=tb::empty(source.sizes(),source.options().dtype(tb::kFloat16));
                input_exponents=tb::empty({tokens},source.options().dtype(tb::kInt32));
                mfq::cuda::gguf::pack_scaled_f16_unit(source.data_ptr<float>(),packed.data_ptr(),
                    input_exponents.data_ptr<int32_t>(),tokens,weight.neuron_len,stream.stream());
            }
        } else packed=source;
        const auto prepare=[&](int64_t index) {
            const auto first=index*tile,count=std::min(tile,row_count-first);
            const int slot=int(index%buffers);
            if(scaled) {
                exponents[slot]=cached_exponents.narrow(0,group*row_count+first,count);
                status(mfq_tq_prefill_scaled_rows(weight.view,int(group*row_count+first),int(count),
                    exponents[slot].data_ptr<int32_t>(),halves[slot].data_ptr(),decoder.stream()));
            } else status(mfq_tq_embedding(weight.view,ids.data_ptr<int64_t>()+group*row_count+first,int(count),
                decoded[slot].data_ptr(),0,decoder.stream()));
            ready[slot].record(decoder.stream());
        };
        const auto tiles=(row_count+tile-1)/tile;
        for(int i=0;i<buffers && i<tiles;++i)prepare(i);
        for(int64_t index=0;index<tiles;++index) {
            const auto first=index*tile,count=std::min(tile,row_count-first);
            const int slot=int(index%buffers);
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),ready[slot].get(),0));
            auto* destination=static_cast<uint8_t*>(output.data_ptr())+(group*row_count+first)*output.element_size();
            if(residual.defined()) {
                prefill_residual_gemm(packed,decoded[slot],residual,output,first,count,weight.gemm->get(),residual_workspace);
            } else mfq::cuda::gguf::prefill_gemm_f16_accum(packed.data_ptr(),scaled?halves[slot].data_ptr():decoded[slot].data_ptr(),
                scaled?product.data_ptr():nullptr,static_cast<uint8_t*>(output.data_ptr())+
                (group*row_count+first)*output.element_size(),!scaled,
                scaled?input_exponents.data_ptr<int32_t>():nullptr,scaled?exponents[slot].data_ptr<int32_t>():nullptr,
                tokens,count,weight.neuron_len,weight.out,weight.gemm->get(),stream.stream());
            consumed[slot].record(stream.stream());
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(decoder.stream(),consumed[slot].get(),0));
            if(index+buffers<tiles)prepare(index+buffers);
        }
        for(const auto& tensor:{packed,input_exponents})if(tensor.defined())mfq_cuda_record_stream(tensor,stream);
    }
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> seen[2]{};
        if(!seen[scaled].exchange(true))std::fprintf(stderr,
            "native_tq_prefill_shared_gemm actual_match=1 M=%lld N=%lld K=%lld groups=%lld tile_rows=%lld buffers=%d scaled=%d f16_accum=1\n",
            static_cast<long long>(tokens),static_cast<long long>(weight.out),static_cast<long long>(weight.neuron_len),
            static_cast<long long>(groups),static_cast<long long>(tile),buffers,int(scaled));
    }
    return output.reshape(shape);
}
}

PrefillProjectionWeight prefill_projection_weight(const TqWeight& w) {
    if(!w.image.is_cuda() || w.view.bias || w.out<1 || w.neuron_len<1)return {};
    return {w.image,w.out,w.neuron_len,std::uint64_t(w.neuron_len)*2,[w] {
        const MfqCudaGuard guard(w.image.get_device());
        const auto stream=mfq_get_current_cuda_stream(w.image.get_device());
        auto exponents=cached_tq_prefill_exponents(w);if(!exponents.defined())return PrefillWeightImage{};
        auto image=tb::empty({w.out,w.neuron_len},w.image.options().dtype(tb::kFloat16));
        for(const auto& tensor:{w.image,image,exponents})mfq_cuda_record_stream(tensor,stream);
        for(int64_t begin=0;begin<w.out;begin+=65535) {
            const auto count=std::min<int64_t>(65535,w.out-begin);
            status(mfq_tq_prefill_scaled_rows(w.view,int(begin),int(count),exponents.data_ptr<int32_t>()+begin,
                static_cast<std::uint8_t*>(image.data_ptr())+begin*w.neuron_len*2,stream.stream()));
        }
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true))std::fprintf(stderr,
                "native_tq_prefill_weight_decoder actual_match=1 shared_product=1 cached_exponents=1 direct_scaled_f16=1 rows=%lld columns=%lld\n",
                static_cast<long long>(w.out),static_cast<long long>(w.neuron_len));
        }
        return PrefillWeightImage{image,exponents};
    }};
}

bool TqWeight::can_forward_prefill_residual(int64_t tokens,const tb::Tensor& residual)const {
    return !view.bias && can_prefill_residual(image,out,neuron_len,tokens,residual);
}
tb::Tensor TqWeight::forward_prefill_residual(tb::Tensor input,const tb::Tensor& residual)const {
    if(!input.is_cuda() || input.device()!=image.device() || input.dim()<1 || input.size(-1)!=neuron_len ||
            input.scalar_type()!=tb::kFloat16)return {};
    const MfqCudaGuard guard(image.get_device());
    return try_tq_prefill_projection(*this,input.contiguous(),1,residual);
}

tb::Tensor TqWeight::forward(tb::Tensor input)const {
    if(!image.is_cuda() && input.is_cuda())return promote_tq(*this,input.device()).forward(std::move(input));
    if(input.size(-1)!=neuron_len || !input.is_cuda() || input.get_device()!=image.get_device())
        throw std::invalid_argument("TQ projection input shape or device differs");
    const MfqCudaGuard guard(image.get_device());
    const auto source=input.contiguous();auto shape=input.sizes().vec();shape.back()=out;
    if(auto result=try_tq_prefill_projection(*this,source);result.defined())return result;
    auto result=tb::empty(shape,input.options());
    const auto tokens=source.numel()/neuron_len;
    const auto element=source.element_size();
    const auto workspace=projection_workspace(*this,static_cast<int>(std::min<std::int64_t>(tokens,65535)),source.scalar_type());
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        project_tokens(*this,static_cast<const std::uint8_t*>(source.data_ptr())+start*neuron_len*element,
            static_cast<std::uint8_t*>(result.data_ptr())+start*out*element,count,dtype(input.scalar_type()),
            workspace.defined()?workspace.data_ptr():nullptr,workspace.defined()?workspace.nbytes():0);
    }
    return result;
}
tb::Tensor TqWeight::forward_groupwise(tb::Tensor input,std::int64_t groups)const {
    if(!image.is_cuda())return promote_tq(*this,input.device()).forward_groupwise(std::move(input),groups);
    if(!input.is_cuda() || input.get_device()!=image.get_device() || groups<1 || groups>INT_MAX ||
       out%groups || input.dim()<2 || input.size(-1)!=neuron_len || input.size(-2)!=groups ||
       neuron_len>INT_MAX/groups)
        throw std::invalid_argument("TQ grouped input must match the weight's group rows and width");
    const MfqCudaGuard guard(image.get_device());
    const auto source=input.contiguous();
    if(auto result=try_tq_prefill_projection(*this,source,groups);result.defined())return result;
    const auto tokens=source.numel()/(groups*neuron_len);
    const auto element=source.element_size();
    auto shape=input.sizes().vec();shape.pop_back();shape.back()=out;
    auto result=tb::empty(shape,input.options());
    const auto workspace=projection_workspace(*this,static_cast<int>(std::min<std::int64_t>(tokens,65535)),source.scalar_type(),static_cast<int>(groups));
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        const auto* x=static_cast<const uint8_t*>(source.data_ptr())+start*groups*neuron_len*element;
        auto* y=static_cast<uint8_t*>(result.data_ptr())+start*out*element;
        if(workspace.defined() && short_q8(source.scalar_type(),count)) {
            status(mfq_tq_q8_tokens(view,x,y,count,static_cast<int>(groups),dtype(source.scalar_type()),workspace.data_ptr(),
                workspace.nbytes(),mfq_current_cuda_stream()));
        }else if(workspace.defined() && count>=gemm_min_tokens(source.scalar_type())) {
            if(source.scalar_type()==tb::kFloat16)status(mfq_tq_grouped_gemm(view,x,y,count,static_cast<int>(groups),workspace.data_ptr(),
                workspace.nbytes(),gemm->get(),mfq_current_cuda_stream()));
            else status(mfq_tq_float_gemm(view,x,y,count,static_cast<int>(groups),dtype(source.scalar_type()),workspace.data_ptr(),
                workspace.nbytes(),gemm->get(),mfq_current_cuda_stream()));
        }
        else status(mfq_tq_grouped_tokens(view,x,y,count,static_cast<int>(groups),dtype(source.scalar_type()),mfq_current_cuda_stream()));
    }
    return result;
}

tb::Tensor TqWeight::try_residual(tb::Tensor input,const tb::Tensor& normalized,
        std::int64_t streams,bool injection)const {
    if(!image.is_cuda() || !input.is_cuda() || image.device()!=input.device() || input.dim()<1 ||
            input.size(-1)!=neuron_len || streams<1 || streams>INT_MAX)return {};
    const auto tokens=input.numel()/neuron_len;
    if(tokens<1 || tokens>4)return {};
    // Residual BF16 inputs already promote to FP32 in the separate path.
    // FP16 retains its intermediate rounding through that existing path.
    if(input.scalar_type()==tb::kBFloat16)input=input.to(tb::kFloat32);
    if(input.scalar_type()!=tb::kFloat32)return {};
    auto shape=input.sizes().vec();shape.back()=out;
    if(normalized.defined()) {
        if(out%streams || normalized.device()!=input.device() || !normalized.is_contiguous() ||
                normalized.scalar_type()!=tb::kFloat32 || normalized.sizes().vec()!=shape)return {};
        shape.back()/=streams;
    }
    const MfqCudaGuard guard(input.device());
    const auto workspace=projection_workspace(*this,int(tokens),input.scalar_type());
    if(!workspace.defined())return {};
    const auto source=input.contiguous();
    auto result=tb::empty(shape,input.options());
    status(mfq_tq_q8_residual_tokens(view,source.data_ptr<float>(),
        normalized.defined()?normalized.data_ptr<float>():nullptr,result.data_ptr<float>(),
        int(tokens),int(streams),injection,workspace.data_ptr(),workspace.nbytes(),mfq_current_cuda_stream()));
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<unsigned> seen{0};
        const auto bit=1u<<(unsigned(tokens)+4u*(normalized.defined()?2u:injection?1u:0u));
        if(!(seen.fetch_or(bit,std::memory_order_relaxed)&bit))
            std::fprintf(stderr,"native_tq_residual_epilogue actual_match=1 mode=%s rows=%lld K=%lld N=%lld streams=%lld\n",
                normalized.defined()?"mix":injection?"injection":"activation",static_cast<long long>(tokens),
                static_cast<long long>(neuron_len),static_cast<long long>(out),static_cast<long long>(streams));
    }
    return result;
}

std::vector<tb::Tensor> TqWeight::try_residual_down(tb::Tensor input,const tb::Tensor& injection,
        std::int64_t streams)const {
    if(!image.is_cuda() || !input.is_cuda() || image.device()!=input.device() || input.dim()<1 ||
            input.size(-1)!=neuron_len || neuron_len<1 || streams<1 || streams>INT_MAX ||
            !injection.defined() || !injection.is_cuda() || injection.device()!=input.device() ||
            injection.dim()!=2 || injection.size(0)!=streams || injection.size(1)!=neuron_len ||
            injection.scalar_type()!=tb::kBFloat16 || !injection.is_contiguous())return {};
    const auto tokens=input.numel()/neuron_len;
    if(tokens<1 || tokens>4)return {};
    if(input.scalar_type()==tb::kBFloat16)input=input.to(tb::kFloat32);
    if(input.scalar_type()!=tb::kFloat32)return {};
    const MfqCudaGuard guard(input.device());
    const auto workspace=projection_workspace(*this,int(tokens),input.scalar_type());
    if(!workspace.defined())return {};
    const auto source=input.contiguous();
    auto shape=input.sizes().vec();shape.back()=out;
    auto low=tb::empty(shape,input.options());shape.back()=streams;
    auto gates=tb::empty(shape,input.options());
    status(mfq_tq_q8_residual_down(view,source.data_ptr<float>(),static_cast<const uint16_t*>(injection.data_ptr()),
        low.data_ptr<float>(),gates.data_ptr<float>(),int(tokens),int(streams),
        workspace.data_ptr(),workspace.nbytes(),mfq_current_cuda_stream()));
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<unsigned> seen{0};const auto bit=1u<<unsigned(tokens);
        if(!(seen.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
            "native_tq_residual_down_injection actual_match=1 rows=%lld K=%lld N=%lld streams=%lld launches=1\n",
            static_cast<long long>(tokens),static_cast<long long>(neuron_len),
            static_cast<long long>(out),static_cast<long long>(streams));
    }
    return {low,gates};
}

tb::Tensor TqWeight::embedding(tb::Tensor ids)const {
    if(!image.is_cuda())return promote_tq(*this,ids.is_cuda()?ids.device():tb::Device(tb::kCUDA,mfq_current_cuda_device())).embedding(std::move(ids));
    const MfqCudaGuard guard(image.get_device());
    auto source=ids.to(image.device()).to(tb::kInt64).contiguous();
    auto shape=ids.sizes().vec();shape.push_back(neuron_len);
    auto result=tb::empty(shape,image.options().dtype(tb::kFloat16));
    for(std::int64_t start=0;start<source.numel();start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,source.numel()-start));
        status(mfq_tq_embedding(view,source.data_ptr<std::int64_t>()+start,count,
            static_cast<std::uint8_t*>(result.data_ptr())+start*neuron_len*2,0,mfq_current_cuda_stream()));
    }
    return result;
}

TqProjectionGroup::TqProjectionGroup(std::vector<TqWeight> weights):weights_(std::move(weights)) {
    if(weights_.empty())throw std::invalid_argument("empty TQ projection group");
    input_=weights_[0].neuron_len;
    if(input_<1 || input_>INT_MAX)throw std::invalid_argument("invalid TQ group input width");
    const auto device=weights_[0].image.device();
    std::vector<TrellisProjectionJob> jobs;jobs.reserve(weights_.size());
    std::int64_t blocks=0;
    for(const auto& weight:weights_) {
        if(weight.out<1 || weight.out!=weight.view.rows || weight.view.input_columns!=input_ ||
           !weight.image.is_cuda() || weight.image.device()!=device || weight.neuron_len!=input_)
            throw std::invalid_argument("TQ group input widths and CUDA devices must match");
        blocks+=(weight.out+TRELLIS_GROUP_ROWS-1)/TRELLIS_GROUP_ROWS;
        if(blocks>INT_MAX || output_>INT_MAX-weight.out)
            throw std::overflow_error("TQ projection group exceeds CUDA index range");
        jobs.push_back({weight.view,output_,static_cast<int>(blocks)});
        output_+=weight.out;
    }
    const MfqCudaGuard guard(weights_[0].image.get_device());
    blocks_=static_cast<int>(blocks);
    const auto bytes=static_cast<std::int64_t>(jobs.size()*sizeof(TrellisProjectionJob));
    plan_=tb::from_blob(jobs.data(),{bytes},tb::TensorOptions().dtype(tb::kUInt8))
        .clone().to(device,false,false).contiguous();
    host_plan_=std::move(jobs);
}

void TqProjectionGroup::join_prefix(std::size_t count) {
    if(count<1 || count>weights_.size())throw std::invalid_argument("invalid TQ joined prefix");
    joined_prefix_=count;joined_rows_=0;
    for(std::size_t i=0;i<count;++i)joined_rows_+=weights_[i].out;
}

std::vector<tb::Tensor> TqProjectionGroup::forward(tb::Tensor input)const {
    if(!defined() || !input.is_cuda() || input.device()!=plan_.device() || input.size(-1)!=input_)
        throw std::invalid_argument("TQ group input shape or device differs");
    const MfqCudaGuard guard(input.get_device());
    const auto source=input.contiguous();
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    mfq_cuda_record_stream(source,stream);mfq_cuda_record_stream(plan_,stream);
    for(const auto& weight:weights_)mfq_cuda_record_stream(weight.image,stream);
    const auto tokens=source.numel()/input_;
    const auto join=[&](std::vector<tb::Tensor> values) {
        if(joined_prefix_>1) {
            auto combined=tb::cat(std::vector<tb::Tensor>(values.begin(),values.begin()+joined_prefix_),-1);
            values.erase(values.begin(),values.begin()+joined_prefix_);
            values.insert(values.begin(),std::move(combined));
        }
        return values;
    };
    if(tokens>=gemm_min_tokens(source.scalar_type())) {
        GgufPrefillInputScope prepared(source);
        std::vector<tb::Tensor> result;result.reserve(weights_.size());
        for(const auto& weight:weights_)result.push_back(weight.forward(source));
        return join(std::move(result));
    }
    if(tokens>1 || short_q8(source.scalar_type(),tokens)) {
        // Projection-major storage gives every output a contiguous view,
        // while one static row plan fuses all parallel token-reuse kernels.
        auto output=tb::empty({tokens*output_},source.options());
        const auto* jobs=reinterpret_cast<const TrellisProjectionJob*>(plan_.data_ptr());
        const bool q8=short_q8(source.scalar_type(),tokens) && weights_.size()<65535;
        if(q8) {
            const auto columns=(std::uint64_t(input_)+511)/512*512;
            const bool column_scaled=std::any_of(host_plan_.begin(),host_plan_.end(),
                [](const auto& job){return job.weight.columns!=0;});
            const auto slots=column_scaled?weights_.size()+1:1;
            const auto bytes=slots*tokens*columns*3/2;
            auto workspace=tb::empty({static_cast<std::int64_t>(bytes)},source.options().dtype(tb::kUInt8));
            status(mfq_tq_q8_parallel_tokens(host_plan_.data(),jobs,static_cast<int>(weights_.size()),blocks_,source.data_ptr(),
                output.data_ptr(),static_cast<int>(tokens),static_cast<int>(input_),dtype(source.scalar_type()),workspace.data_ptr(),
                workspace.nbytes(),joined_prefix_>1?static_cast<int>(joined_prefix_):0,
                static_cast<int>(joined_rows_),mfq_current_cuda_stream()));
        }else status(mfq_tq_parallel_tokens(jobs,static_cast<int>(weights_.size()),blocks_,source.data_ptr(),output.data_ptr(),
            static_cast<int>(tokens),static_cast<int>(input_),dtype(source.scalar_type()),mfq_current_cuda_stream()));
        std::vector<tb::Tensor> result;result.reserve(weights_.size());
        std::int64_t offset=0;
        for(std::size_t i=0;i<weights_.size();) {
            const auto count=q8 && i==0?joined_prefix_:1;
            std::int64_t rows=0;
            for(std::size_t j=0;j<count;++j)rows+=weights_[i+j].out;
            auto shape=input.sizes().vec();shape.back()=rows;
            result.push_back(output.narrow(0,offset*tokens,rows*tokens).reshape(shape));offset+=rows;i+=count;
        }
        return q8?result:join(std::move(result));
    }
    auto shape=input.sizes().vec();shape.back()=output_;
    auto output=tb::empty(shape,input.options());
    const auto element=source.element_size();
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        status(mfq_tq_parallel(reinterpret_cast<const TrellisProjectionJob*>(plan_.data_ptr()),
            static_cast<int>(weights_.size()),blocks_,
            static_cast<const std::uint8_t*>(source.data_ptr())+start*input_*element,
            static_cast<std::uint8_t*>(output.data_ptr())+start*output_*element,
            count,static_cast<int>(input_),static_cast<int>(output_),dtype(input.scalar_type()),mfq_current_cuda_stream()));
    }
    std::vector<tb::Tensor> result;result.reserve(weights_.size());
    std::int64_t offset=0;
    for(std::size_t i=0;i<weights_.size();) {
        const auto count=i==0?joined_prefix_:1;
        std::int64_t rows=0;
        for(std::size_t j=0;j<count;++j)rows+=weights_[i+j].out;
        result.push_back(output.narrow(-1,offset,rows));offset+=rows;i+=count;
    }
    return result;
}

tb::Tensor TqProjectionGroup::swiglu(tb::Tensor input)const {
    if(weights_.size()!=2 || weights_[0].out!=weights_[1].out || !input.is_cuda() ||
       input.device()!=plan_.device() || input.size(-1)!=input_)
        throw std::invalid_argument("TQ SwiGLU group requires two matching projections");
    const MfqCudaGuard guard(input.get_device());
    const auto source=input.contiguous();
    if(source.numel()/input_>=gemm_min_tokens(source.scalar_type())) {
        const auto gate=weights_[0].forward(source),up=weights_[1].forward(source);
        return silu_mul_cuda(gate,up);
    }
    const auto rows=weights_[0].out;
    auto shape=input.sizes().vec();shape.back()=rows;
    auto output=tb::empty(shape,input.options());
    const auto tokens=source.numel()/input_;
    const auto element=source.element_size();
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        status(mfq_tq_swiglu_tokens(weights_[0].view,weights_[1].view,
            static_cast<const std::uint8_t*>(source.data_ptr())+start*input_*element,
            static_cast<std::uint8_t*>(output.data_ptr())+start*rows*element,
            count,dtype(input.scalar_type()),mfq_current_cuda_stream()));
    }
    return output;
}

namespace {
std::size_t available_host_bytes(std::size_t process_limit) {
    try {
        return mfq::host_memory_headroom(mfq::sample_host_memory(),process_limit);
    } catch(const std::runtime_error&) {return 0;}
}
}

// Ordinary TQ projection images share a byte-budget LRU. The storage and
// lifetime policy is independent of the row precision and does not use MFE.
struct TqHostImage:mfq::cuda::RoutedHostImage<> {
    mfq::TqLayout layout;
    std::size_t source_bytes;
    TqHostImage(mfq::TqLayout value,std::size_t source_count,std::shared_ptr<MoePinnedExpertArena::Block> pinned)
        :layout(value),source_bytes(source_count) {
        allocate(((layout.packed+3)&~std::size_t(3))+layout.packed_bytes+32,std::move(pinned));
    }
    explicit TqHostImage(std::vector<std::uint8_t> wire,std::shared_ptr<MoePinnedExpertArena::Block> block)
        :layout(wire.data(),wire.size()),source_bytes(wire.size()) {
        const auto words=(layout.packed+3)&~std::size_t(3),relative=words+layout.packed_bytes;
        allocate(relative+32,std::move(block));
        std::memcpy(data(),wire.data(),layout.packed);
        std::memset(data()+layout.packed,0,words-layout.packed);
        std::memcpy(data()+words,wire.data()+layout.packed,layout.packed_bytes);
        constexpr std::uint16_t table[]{0x3400,0x34d0,0x35ca,0x36f7,0x3830,0x390a,0x3a10,0x3b4b,
            0x3c63,0x3d47,0x3e59,0x3fa4,0x4098,0x4187,0x42a6,0x4400};
        std::memcpy(data()+relative,table,sizeof(table));
    }
    TqHostImage(const TqHostImage&)=delete;
    TqHostImage& operator=(const TqHostImage&)=delete;
    static std::shared_ptr<TqHostImage> create(std::vector<std::uint8_t> wire,
            std::shared_ptr<MoePinnedExpertArena::Block> block={}) {
        return std::make_shared<TqHostImage>(std::move(wire),std::move(block));
    }
};

class TqExpertCache:private mfq::cuda::RoutedHostCache<TqHostImage> {
    struct CpuTimer {
        double* total;
        std::chrono::steady_clock::time_point start;
        explicit CpuTimer(double* value):total(value),start(value?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{}) {}
        ~CpuTimer(){if(total)*total+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}
    };
    using HostImage=TqHostImage;
    struct HostTransfer {
        std::optional<MfqCudaStream> producer;
        std::shared_ptr<mfq::cuda::Event> ready;
        std::shared_ptr<std::atomic<bool>> submitted,recorded;
        std::shared_ptr<mfq::cuda::RoutedCompletionPosition> position;
        std::vector<std::shared_ptr<HostImage>> images;
        std::vector<tb::Tensor> destinations;
        std::vector<std::shared_ptr<void>> device_backings;
        bool completed=false;
        bool recorded_submission()const noexcept {
            return submitted && recorded && submitted->load(std::memory_order_acquire) &&
                recorded->load(std::memory_order_acquire);
        }
        ~HostTransfer()noexcept {
            if(completed)return;
            try {
                if(ready && recorded_submission())ready->synchronize();
                else if(producer && submitted && submitted->load(std::memory_order_acquire)) {
                    const MfqCudaGuard device(producer->device());
                    (void)cudaStreamSynchronize(producer->stream());
                }
            }catch(...) {}
        }
    };
    struct DeviceKey {
        std::uint64_t image;
        int device;
        bool operator==(const DeviceKey&)const=default;
    };
    struct DeviceKeyHash {
        std::size_t operator()(const DeviceKey& key)const noexcept {
            return std::hash<std::uint64_t>{}(key.image)^(std::hash<int>{}(key.device)<<1);
        }
    };
    struct DeviceEntry {
        TqWeight weight;
        std::list<DeviceKey>::iterator recent;
        std::size_t source_bytes=0;
        mfq::TqLayout layout;
        DeviceImageArena* arena=nullptr;
        std::list<DeviceKey>::iterator arena_position;
        std::optional<std::list<DeviceKey>::iterator> cold;
        std::optional<std::uint64_t> group;
    };
    struct ViewTable {
        int device;
        cudaStream_t stream;
        tb::Tensor views;
        std::vector<std::uintptr_t> images;
        std::vector<std::uint8_t> bytes;
    };
    struct RouteReadback {
        int device;
        MfqCudaStream stream;
        std::mutex mutex;
        std::unique_ptr<mfq::cuda::HostBuffer> host;
        void* mapped=nullptr;
        bool can_map=false;
        RouteReadback(int device,MfqCudaStream stream):device(device),stream(std::move(stream)) {
            int supported=0;
            TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&supported,cudaDevAttrCanMapHostMemory,device));
            can_map=supported!=0;
        }
    };
    struct Stage {
        mfq::cuda::HostBuffer host;
        std::uint8_t* mapped=nullptr;
        mfq::cuda::Event uploaded;
        std::size_t cursor=0;
        bool pending=false;
        explicit Stage(std::size_t bytes):host(bytes,true),uploaded(cudaEventDisableTiming) {
            void* alias=nullptr;
            TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&alias,host.data(),0));
            mapped=static_cast<std::uint8_t*>(alias);
            mfq::cuda::charge_tensor_host_bytes(bytes);
        }
        ~Stage(){
            if(pending)try{uploaded.synchronize();}catch(...) {}
            mfq::cuda::tensor_host_bytes.fetch_sub(host.size());
        }
        void reuse(){if(pending){uploaded.synchronize();pending=false;}cursor=0;}
    };
    struct DeviceStages {
        std::optional<MfqCudaStream> owner;
        std::array<std::unique_ptr<Stage>,2> stages;
        unsigned index=0;
    };
    std::unordered_map<DeviceKey,DeviceEntry,DeviceKeyHash> device_;
    struct ExpertSlot {
        tb::Tensor image;
        DeviceImageArena* arena;
        mfq::cuda::RoutedExpertImageLayout layout;
        unsigned residents=0;
    };
    std::unordered_map<DeviceKey,ExpertSlot,DeviceKeyHash> expert_slots_;
    std::uint64_t expert_slot_reuses_=0;
    std::unordered_map<int,std::size_t> pending_expert_bytes_;
    std::unordered_map<std::uint64_t,std::vector<ViewTable>> view_tables_;
    std::vector<std::shared_ptr<RouteReadback>> route_readbacks_;
    std::unordered_map<std::uint64_t,std::size_t> resident_copies_;
    std::list<DeviceKey> device_recent_;
    std::size_t device_used_=0,host_limit_=0,ordinary_host_limit_=0,device_limit_=0;
    std::vector<std::shared_ptr<MoePinnedExpertArena>> host_arenas_;
    std::size_t host_reserved_=0,host_pageable_=0,pinned_ceiling_=SIZE_MAX;
    std::unique_ptr<RoutedPrefillStager> prefill_host_stager_;
    std::unordered_map<std::uint64_t,int> staged_host_jobs_;
    std::unordered_map<int,std::size_t> device_usage_,decode_limits_;
    using PrefillLoan=mfq::cuda::RoutedPrefillLoanState<mfq::cuda::Context,tb::Tensor,MfqCudaStream>;
    struct WorkspaceReservation {
        int device;
        std::size_t limit,required;
        MfqCudaStream stream;
        std::thread::id thread;
        std::shared_ptr<PrefillLoan> loan;
    };
    std::unordered_map<std::uint64_t,WorkspaceReservation> workspace_limits_;
    std::uint64_t next_workspace_token_=0;
    struct DecodeReserve {std::size_t growth=0;std::optional<std::size_t> transient;};
    std::unordered_map<int,DecodeReserve> decode_reserves_;
    std::size_t device_used_locked(int device)const {
        const auto found=device_usage_.find(device),pending=pending_expert_bytes_.find(device);
        return (found==device_usage_.end()?0:found->second)+(pending==pending_expert_bytes_.end()?0:pending->second);
    }
    std::size_t device_limit_locked(int device=mfq_current_cuda_device())const {
        auto limit=retain_all_?std::max(resident_bytes_bound_,device_used_locked(device)):device_limit_;
        if(auto found=decode_limits_.find(device);found!=decode_limits_.end())limit=std::min(limit,found->second);
        for(const auto& [token,reservation]:workspace_limits_)
            if(reservation.device==device)limit=std::min(limit,reservation.limit);
        return limit;
    }
    struct RoutePolicy {
        mfq::cuda::ExpertRouteUsage usage;
        mfq::cuda::RoutedExpertBanks banks;
        std::list<DeviceKey> cold;
        std::uint64_t windows=0,last_decay=0;
        bool seeded=false,ready=false;
        std::size_t assignment_budget=SIZE_MAX;
        std::vector<mfq::cuda::AdaptiveImageSwap> replacements;
        std::uint64_t observations=0;
        std::unordered_set<std::uint64_t> decoded_banks;
        std::uint64_t replacement_pairs=0,replacement_approved=0,replacement_budget_rejected=0;
        std::uint64_t replacement_leased=0,replacement_resized=0,replacement_missing_slots=0;
        std::uint64_t replacement_heat_rejected=0,replacement_multi_victim=0;
        void register_banks(const std::array<std::uint64_t,3>& sources) {
            banks.add(sources);
        }
        template<class Count> void select(std::size_t budget,Count&& count) {
            if(!ready || assignment_budget!=budget) {
                usage.select(budget,mfq::cuda::expert_replacement_margin,&banks);
                replacements.clear();assignment_budget=budget;ready=true;return;
            }
            const auto groups=banks.groups();
            std::vector<mfq::cuda::RoutedImageCandidate> candidates,victims;
            for(const auto& group:groups) {
                const auto experts=std::size_t(count(group[0]));
                for(std::size_t expert=0;expert<experts;++expert) {
                    const auto key=(group[0]<<32)|expert;const auto score=usage.score(key);
                    const bool retained=usage.retained(key);
                    if(!retained && score<2.f)continue;
                    const auto bytes=usage.expert_bytes(key,banks);
                    if(!bytes)continue;
                    if(retained)victims.push_back({score,key,bytes});
                    else candidates.push_back({score,key,bytes});
                }
            }
            mfq::cuda::AdaptiveImageSwapStats stats;
            replacements=mfq::cuda::adaptive_image_swaps(usage,candidates,victims,budget,banks,96,&stats);
            replacement_pairs+=stats.proposed;replacement_budget_rejected+=stats.capacity_rejected;
            replacement_heat_rejected+=stats.heat_rejected;replacement_multi_victim+=stats.multi_victim;
            replacement_approved+=replacements.size();
        }
    };
    struct PendingRoute {int device,rows;std::array<std::uint64_t,3> sources;std::vector<RoutedExpertTouch> touches;};
    std::unordered_map<int,RoutePolicy> routes_;
    RoutedExpertJournal route_journal_;
    std::vector<RoutedExpertTouch> route_touches_;
    std::vector<PendingRoute> speculative_routes_;
    int speculative_rows_=0;
    bool reuse_learning_=true;
    void refresh_expert_priority_locked(int device,std::uint64_t expert) {
        auto& policy=routes_[device];
        policy.banks.visit_expert(expert,[&](auto image) {
            const DeviceKey key{image,device};const auto found=device_.find(key);
            if(found==device_.end())return;
            auto& entry=found->second;
            if(policy.usage.retained(image)) {
                if(entry.cold){policy.cold.erase(*entry.cold);entry.cold.reset();}
            }else if(!entry.cold)entry.cold=policy.cold.insert(policy.cold.end(),key);
        });
    }
    void refresh_device_priority_locked(int device) {
        auto& policy=routes_[device];
        for(const auto& key:device_recent_)if(key.device==device) {
            auto& entry=device_.at(key);
            if(policy.usage.retained(key.image)) {
                if(entry.cold){policy.cold.erase(*entry.cold);entry.cold.reset();}
            }else if(entry.cold)policy.cold.splice(policy.cold.end(),policy.cold,*entry.cold);
            else entry.cold=policy.cold.insert(policy.cold.end(),key);
        }
    }
    bool finish_routes_locked(int device,bool frequent=false) {
        auto& policy=routes_[device];
        if(!policy.seeded || !reuse_learning_)return false;
        if(mfq::cuda::routed_admission_due(++policy.windows,frequent)) {
            CpuTimer timer(profile_?&route_policy_ms_:nullptr);
            const auto limit=device_limit_locked(device);
            const bool rebuild=!policy.ready || policy.assignment_budget!=limit;
            policy.select(limit,[&](auto source){return source_info_.at(source).experts;});
            policy.ready=true;policy.usage.decay(policy.windows-policy.last_decay);policy.last_decay=policy.windows;
            // Apply admission decisions to replacement too. Rebuild only the
            // cold ordering at an adaptation boundary, never scan on a miss.
            if(rebuild)refresh_device_priority_locked(device);
            else for(const auto& replacement:policy.replacements) {
                for(const auto victim:replacement.victims)refresh_expert_priority_locked(device,victim);
                refresh_expert_priority_locked(device,replacement.incoming);
            }
            return true;
        }
        return false;
    }
    bool retain_all_=false;
    std::size_t payload_bytes_=0,predictor_resident_bytes_=0;
    std::size_t resident_bytes_bound_=0;
    std::size_t process_host_limit_=0;
    std::uint64_t host_evictions_=0,device_evictions_=0,source_reads_=0,device_reuses_=0;
    std::unordered_map<int,std::shared_ptr<TqGemmResources>> gemm_;
    mfq::cuda::RoutedGraphPools graph_pools_;
    std::uint64_t sources_=0;
    std::mutex mutex_;
    std::unordered_map<int,std::unordered_map<cudaStream_t,DeviceStages>> stages_;
    std::unordered_map<int,MfqCudaStream> transfer_streams_,admission_streams_;
    mfq::cuda::RoutedCompletionEvents completion_events_;
    std::unordered_map<int,std::vector<std::shared_ptr<DeviceImageArena>>> device_arenas_;
    std::unordered_map<DeviceImageArena*,std::list<DeviceKey>> arena_entries_;
    struct SourceInfo {
        std::shared_ptr<TqRowTable> table;
        std::int64_t rows,experts;
        std::shared_ptr<TqResidentBank> resident;
    };
    std::unordered_map<std::uint64_t,SourceInfo> source_info_;
    void register_banks_locked(RoutePolicy& policy,const std::array<std::uint64_t,3>& sources) {
        for(const auto source:sources)(void)source_info_.at(source);
        if(source_info_.at(sources[0]).resident)return;
        policy.register_banks(sources);
    }
    using ResidentViews=mfq::cuda::RoutedResidentViews<TrellisProjectionView>;
    using LayerAdmission=mfq::cuda::RoutedLayerAdmission<ResidentViews,mfq::cuda::Event,tb::Tensor>;
    std::map<std::array<std::uint64_t,4>,std::shared_ptr<ResidentViews>> resident_views_;
    struct ResidentRole {std::weak_ptr<ResidentViews> views;int role;};
    std::map<std::pair<int,std::uint64_t>,std::vector<ResidentRole>> resident_roles_;
    std::unordered_map<int,unsigned> replay_readers_;
    mfq::cuda::RoutedEventRetirement deferred_events_;
    bool replay_active_locked()const {
        return std::any_of(replay_readers_.begin(),replay_readers_.end(),[](const auto& item){return item.second!=0;});
    }
    std::unordered_set<int> resident_updates_;
    std::unique_ptr<RoutedAdmission> admission_executor_=std::make_unique<RoutedAdmission>();
    std::shared_ptr<LayerAdmission> layer_admission_;
    std::unordered_map<int,tb::Tensor> admission_ready_words_;
    void update_resident_view_locked(DeviceKey key,TrellisProjectionView value) {
        const auto found=resident_roles_.find({key.device,key.image>>32});
        if(found==resident_roles_.end())return;
        for(const auto& binding:found->second)if(auto views=binding.views.lock())
            views->update(binding.role,std::uint32_t(key.image),value,value.words!=nullptr);
        resident_updates_.insert(key.device);
    }
    void join_admission(RoutedAdmission::Transition& admission) {
        std::shared_ptr<LayerAdmission> epoch;
        {std::lock_guard lock(mutex_);epoch=layer_admission_;}
        try {
            admission.wait();
            if(epoch && epoch->submitted_done)epoch->submitted_done->synchronize();
        }catch(...) {
            std::lock_guard lock(mutex_);if(layer_admission_==epoch)layer_admission_.reset();throw;
        }
        std::lock_guard lock(mutex_);
        if(layer_admission_==epoch)layer_admission_.reset();
    }
    void wait_admission(){admission_executor_->transition([&](auto& admission){join_admission(admission);});}
    void abort_layer_admission(const std::shared_ptr<LayerAdmission>& epoch,std::exception_ptr error) {
        if(!epoch)return;
        std::lock_guard lock(mutex_);
        epoch->abort(std::move(error),[](auto& view){view.invalidate();},mfq::cuda::publish_mapped_flag);
    }
    void queue_admission(int device,mfq::cuda::RoutedImageUse use=mfq::cuda::RoutedImageUse::Admission) {
        admission_executor_->transition([&](auto& admission) {
            join_admission(admission);
            const auto upload=admission_stream(device);
            const MfqCudaStreamGuard upload_guard(upload);
            struct Batch {
                std::uint64_t source;SourceInfo info;std::vector<std::int64_t> experts;
                std::pair<std::uint64_t,unsigned> order;
            };
            std::vector<Batch> batches;
            {
                std::lock_guard lock(mutex_);
                auto policy=routes_.find(device);
                if(policy==routes_.end() || !policy->second.ready || replay_readers_[device])return;
                for(const auto& replacement:policy->second.replacements) {
                    const auto incoming=replacement.incoming;const DeviceKey new_key{incoming,device};
                    const auto* banks=policy->second.banks.find(incoming>>32);
                    if(!banks)continue;
                    std::array<std::size_t,3> sizes{};
                    for(unsigned role=0;role<3;++role) {
                        const auto& info=source_info_.at((*banks)[role]);
                        sizes[role]=info.table->expert_image_bytes(std::uint32_t(incoming),info.rows);
                    }
                    const mfq::cuda::RoutedExpertImageLayout layout(*banks,sizes);
                    tb::Tensor image;DeviceImageArena* arena=nullptr;
                    for(const auto victim:replacement.victims) {
                        const auto old=expert_slots_.find(DeviceKey{victim,device});
                        if(old==expert_slots_.end()){++policy->second.replacement_missing_slots;continue;}
                        const auto& slot=old->second;
                        if(slot.image.weak_storage_owner().use_count()!=slot.residents+1) {
                            ++policy->second.replacement_leased;continue;
                        }
                        if(slot.image.nbytes()>=layout.bytes &&
                                (!image.defined() || slot.image.nbytes()<image.nbytes())) {
                            image=slot.image;arena=slot.arena;
                        }
                    }
                    // Leased storage stays immutable until its readers release
                    // it. An arena cannot recycle that interval prematurely.
                    for(const auto victim:replacement.victims)erase_expert_locked(DeviceKey{victim,device});
                    if(!image.defined()){++policy->second.replacement_resized;continue;}
                    if(expert_slots_.count(new_key))continue;
                    erase_expert_locked(new_key);
                    const auto bytes=image.nbytes();
                    expert_slots_.emplace(new_key,ExpertSlot{std::move(image),arena,layout});
                    ++expert_slot_reuses_;
                    pending_expert_bytes_[device]+=bytes;
                }
                policy->second.replacements.clear();
                // Convert the prompt's independent projection allocations at
                // the safe decode boundary. A grouped slot always owns all
                // three banks; partial prompt images cannot strand its space.
                for(const auto key:policy->second.usage.retained_keys()) {
                    const auto* banks=policy->second.banks.find(key>>32);
                    if(!banks || (key>>32)!=(*banks)[0] || expert_slots_.count(DeviceKey{key,device}))continue;
                    erase_expert_locked(DeviceKey{key,device});
                }
                std::unordered_map<std::uint64_t,std::size_t> groups;
                for(const auto key:policy->second.usage.retained_keys()) {
                    if(device_.count(DeviceKey{key,device}))continue;
                    const auto source=key>>32;
                    const auto [group,inserted]=groups.try_emplace(source,batches.size());
                    if(inserted) {
                        const auto* banks=policy->second.banks.find(source);
                        const auto role=banks?unsigned(std::find(banks->begin(),banks->end(),source)-banks->begin()):0u;
                        batches.push_back({source,source_info_.at(source),{},
                            {banks?(*banks)[0]:source,role}});
                    }
                    batches[group->second].experts.push_back(std::uint32_t(key));
                }
                // Release each earlier layer after its complete banks, even
                // when usage entries were first observed in another order.
                std::sort(batches.begin(),batches.end(),[](const auto& a,const auto& b){return a.order<b.order;});
            }
            if(batches.empty())return;
            auto epoch=std::make_shared<LayerAdmission>();epoch->device=device;
            {
                std::lock_guard lock(mutex_);
                for(const auto& [_,view]:resident_views_)if(view->device==device) {
                    if(std::all_of(view->sources.begin(),view->sources.end(),[&](auto source) {
                        return bool(source_info_.at(source).resident);
                    }))continue;
                    if(view->rebuild)view->republish();
                    epoch->views.push_back(view);
                }
                if(epoch->views.empty())epoch.reset();
                else {
                    mfq::cuda::arm_routed_layer_admission(*epoch,admission_ready_words_[device],upload.stream());
                    layer_admission_=epoch;
                }
            }
            auto run=[this,device,upload,epoch,use,batches=std::move(batches)] {
                const MfqCudaStreamGuard stream(upload);
                try {
                    std::unordered_set<std::uint64_t> pending;
                    for(const auto& batch:batches)pending.insert(batch.source);
                    std::unordered_set<const ResidentViews*> released;
                    if(epoch)epoch->release_unchanged([&](const auto& view) {
                        return std::any_of(view.sources.begin(),view.sources.end(),[&](auto source){return pending.count(source)!=0;});
                    },mfq::cuda::publish_mapped_flag);
                    if(epoch)for(const auto& view:epoch->views)
                        if(*reinterpret_cast<const volatile std::uint32_t*>(view->admission_host()))released.insert(view.get());
                    for(const auto& batch:batches) {
                        (void)get_rows(batch.source,batch.experts.data(),batch.experts.size(),
                            *batch.info.table,batch.info.rows,device,nullptr,use);
                        pending.erase(batch.source);
                        if(epoch) {
                            std::vector<std::uint32_t*> ready;
                            {
                                std::lock_guard lock(mutex_);epoch->require_active();
                                for(const auto& view:epoch->views)if(!released.count(view.get()) &&
                                    std::none_of(view->sources.begin(),view->sources.end(),[&](auto source){return pending.count(source)!=0;})) {
                                    released.insert(view.get());ready.push_back(view->admission_host());
                                }
                            }
                            mfq::cuda::publish_routed_admission_layers(*epoch,ready,upload.stream());
                        }
                    }
                    if(epoch)epoch->submitted_done->record(upload.stream());
                    else TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(upload.stream()));
                    std::lock_guard lock(mutex_);resident_updates_.erase(device);
                }catch(...) {
                    const auto error=std::current_exception();abort_layer_admission(epoch,error);
                    (void)cudaStreamSynchronize(upload.stream());std::rethrow_exception(error);
                }
            };
            try{admission.submit(std::move(run));}
            catch(...){abort_layer_admission(epoch,std::current_exception());throw;}
        });
    }
    std::list<HostTransfer> host_transfers_;
    mfq::cuda::RoutedCompletionClock transfer_clock_;
    std::uint64_t calls_=0,gpu_hits_=0,host_hits_=0,source_bytes_=0,source_batches_=0,transfer_bytes_=0;
    std::uint64_t direct_transfer_bytes_=0,pin_attempts_=0,pin_failures_=0,native_batches_=0,writeback_bytes_=0,unpinned_bytes_=0;
    std::uint64_t decode_calls_=0,decode_hits_=0,decode_transfer_=0,decode_source_=0,decode_writeback_=0;
    std::uint64_t admission_transfer_=0,restore_transfer_=0;
    std::uint64_t decode_metadata_bytes_=0,decode_metadata_updates_=0,decode_resident_bypasses_=0;
    double decode_metadata_submit_ms_=0,decode_resident_touch_ms_=0,decode_admission_wait_ms_=0;
    std::uint64_t upload_groups_=0,retirement_queries_=0;
    std::uint64_t staged_packets_=0;
    std::uint64_t host_compacted_bytes_=0,host_trimmed_bytes_=0;
    std::uint64_t host_evacuated_bytes_=0,host_evacuated_images_=0;
    bool host_evacuation_blocked_=false;
    std::size_t pin_fragment_probe_bytes_=SIZE_MAX;
    std::uint64_t pin_fragment_retries_=0,pin_fragment_reuses_=0,pin_interval_compactions_=0;
    double fetch_ms_=0,upload_submit_ms_=0;
    double get_rows_ms_=0,allocation_ms_=0,registration_ms_=0,source_prepare_ms_=0,route_policy_ms_=0;
    double source_slice_ms_=0,host_admission_ms_=0,host_pack_ms_=0;
    double pin_reuse_ms_=0,unpin_copy_ms_=0,image_pack_ms_=0,pageable_repin_ms_=0;
    double unpin_allocate_ms_=0,unpin_memcpy_ms_=0,unpin_release_ms_=0;
    double stage_copy_ms_=0,stage_wait_ms_=0;
    double decode_get_rows_ms_=0,decode_fetch_ms_=0,decode_copy_ms_=0,decode_stage_wait_ms_=0;
    double decode_allocation_ms_=0,decode_submit_ms_=0,decode_repin_ms_=0;
    std::uint64_t arena_recycles_=0,retained_evictions_=0,decode_arena_recycles_=0,decode_retained_evictions_=0;
    const bool profile_=std::getenv("TYLOINFER_TQ_PROFILE_CACHE")!=nullptr;
    void mark_host_duplicate_locked(std::uint64_t key) {
        if(!host_.count(key))return;
        host_index_duplicate(key,resident_copies_.count(key)!=0);host_index_touch(key);
    }
    void erase_host_locked(std::uint64_t key) {
        const auto found=host_.find(key);if(found==host_.end())return;
        if(layer_admission_ || replay_active_locked())deferred_events_.retain(found->second.image->ready);
        if(!found->second.image->block)host_pageable_-=found->second.image->fallback.nbytes();
        host_index_erase(key);++host_evictions_;
    }
    void insert_host_locked(std::uint64_t key,const std::shared_ptr<HostImage>& image) {
        host_index_insert(key,image,resident_copies_.count(key)!=0);
        if(!image->block)host_pageable_+=image->fallback.nbytes();
    }
    void demote_device_locked(const DeviceKey& key,const DeviceEntry& entry) {
        const mfq::cuda::RoutedHostEvacuationBlock block_evacuation(host_evacuation_blocked_);
        if(host_.count(key.image))return;
        const auto bytes=((entry.layout.packed+3)&~std::size_t(3))+entry.layout.packed_bytes+32;
        if(bytes>host_limit_)return;
        admit_host_space_locked(bytes);
        auto block=allocate_pinned(bytes);
        // Pageable D2H may synchronize a waiting compute graph. Immutable
        // source recovery remains available when no pinned interval is free.
        if(!block)return;
        auto image=std::make_shared<HostImage>(entry.layout,entry.source_bytes,std::move(block));
        auto stream=transfer_streams_.at(key.device);
        // Replacement writes must follow a victim's D2H on the same lane.
        // Admission has its own lane so later layers cannot queue ahead of
        // demand uploads for the layer currently being decoded.
        if(const auto found=admission_streams_.find(key.device);found!=admission_streams_.end() &&
                found->second.stream()==mfq_get_current_cuda_stream(key.device).stream())stream=found->second;
        const MfqCudaGuard device(key.device);
        image->ready=completion_events_.at(key.device,stream.stream());
        image->producer=stream.stream();
        host_transfers_.emplace_back();auto& transfer=host_transfers_.back();
        transfer.producer=stream;transfer.ready=image->ready;
        transfer.submitted=std::make_shared<std::atomic<bool>>(false);
        transfer.recorded=std::make_shared<std::atomic<bool>>(false);
        transfer.position=std::make_shared<mfq::cuda::RoutedCompletionPosition>();
        transfer.images.push_back(image);
        if(!entry.arena)throw std::logic_error("TQ resident image has no backing arena");
        transfer.device_backings.push_back(entry.arena->retain_backing());
        try {
            transfer.submitted->store(true,std::memory_order_release);
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(image->data(),entry.weight.image.data_ptr(),bytes,
                cudaMemcpyDeviceToHost,stream.stream()));
            image->ready->record(stream.stream());
            *transfer.position=mfq::cuda::record_routed_completion(transfer_clock_,stream.device(),stream.stream());
            transfer.recorded->store(true,std::memory_order_release);
        }catch(...) {(void)cudaStreamSynchronize(stream.stream());host_transfers_.pop_back();throw;}
        writeback_bytes_+=bytes;
        insert_host_locked(key.image,image);
    }
    void erase_device_locked(DeviceKey key,bool preserve=true) {
        const auto found=device_.find(key);
        if(routes_.at(key.device).usage.retained(key.image))++retained_evictions_;
        if(preserve)demote_device_locked(key,found->second);
        const auto bytes=found->second.weight.image.nbytes();
        routes_.at(key.device).usage.resident(key.image,false);
        update_resident_view_locked(key,{});
        if(found->second.cold)routes_.at(key.device).cold.erase(*found->second.cold);
        arena_entries_.at(found->second.arena).erase(found->second.arena_position);
        device_used_-=bytes;device_usage_[key.device]-=bytes;device_recent_.erase(found->second.recent);
        if(found->second.group) {
            const DeviceKey group{*found->second.group,key.device};auto& slot=expert_slots_.at(group);
            pending_expert_bytes_[key.device]+=bytes;
            if(!--slot.residents) {
                pending_expert_bytes_[key.device]-=slot.image.nbytes();expert_slots_.erase(group);
            }
        }
        const auto copies=resident_copies_.find(key.image);
        if(!--copies->second) {
            resident_copies_.erase(copies);
            host_index_duplicate(key.image,false);
        }
        device_.erase(found);++device_evictions_;
    }
    void erase_expert_locked(DeviceKey key) {
        // Replacement uses the same complete-expert unit as admission. Raw
        // projection images keep their actual encoded sizes and arena leases.
        routes_.at(key.device).banks.visit_expert(key.image,[&](auto image) {
            const DeviceKey victim{image,key.device};if(device_.count(victim))erase_device_locked(victim);
        });
    }
    bool evict_oldest_device_locked(int device) {
        if(auto policy=routes_.find(device);policy!=routes_.end() && policy->second.ready && !policy->second.cold.empty()) {
            erase_expert_locked(policy->second.cold.back());return true;
        }
        const auto oldest=std::find_if(device_recent_.rbegin(),device_recent_.rend(),
            [device](const auto& key){return key.device==device;});
        if(oldest==device_recent_.rend())return false;
        erase_expert_locked(*oldest);return true;
    }
    void admit_host_space_locked(std::size_t bytes) {
        // Prefer a revocable CPU duplicate over the only RAM copy of an
        // expert. Reserve its interval before packing the incoming image.
        while(bytes<=host_limit_ && host_used_>host_limit_-bytes && !host_recent_.empty())
            erase_host_locked(host_index_victim());
    }
    std::size_t device_reserved_locked(int device)const {
        std::size_t bytes=0;
        if(auto found=device_arenas_.find(device);found!=device_arenas_.end())
            for(const auto& arena:found->second)bytes+=arena->bytes();
        return bytes;
    }
    void release_empty_device_arenas_locked(int device) {
        auto& arenas=device_arenas_[device];
        std::erase_if(arenas,[&](const auto& arena) {
            if(arena->used_bytes())return false;
            arena_entries_.erase(arena.get());return true;
        });
    }
    std::size_t evict_device_arena_locked(int device,std::shared_ptr<DeviceImageArena>* recycled=nullptr,std::size_t minimum=0) {
        auto& arenas=device_arenas_[device];
        auto candidates=arenas;
        std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b) {
            return a->bytes()-a->used_bytes()>b->bytes()-b->used_bytes();
        });
        for(const auto& arena:candidates) {
            if(arena->bytes()<minimum)continue;
            std::size_t covered=0;bool busy=false;
            std::unordered_set<std::uint64_t> groups;
            const auto owners=arena_entries_.find(arena.get());
            if(owners!=arena_entries_.end())for(const auto& key:owners->second) {
                const auto& entry=device_.at(key);
                if(!entry.group)covered+=entry.weight.image.nbytes();
                else if(groups.insert(*entry.group).second)
                    covered+=expert_slots_.at(DeviceKey{*entry.group,device}).image.nbytes();
                const auto owned=entry.group?expert_slots_.at(DeviceKey{*entry.group,device}).residents+1:1;
                if(entry.weight.image.weak_storage_owner().use_count()!=owned){busy=true;break;}
            }
            if(busy || covered!=arena->used_bytes())continue;
            // The arena owns an index of its live entries. Reclaiming one
            // backing region never scans unrelated layers or other devices.
            if(owners!=arena_entries_.end())while(!owners->second.empty())
                erase_expert_locked(owners->second.front());
            const auto released=arena->bytes();
            if(recycled)*recycled=arena;
            else {arena_entries_.erase(arena.get());std::erase(arenas,arena);}
            ++arena_recycles_;return released;
        }
        return 0;
    }
    tb::Tensor allocate_device_image(std::size_t bytes,int device,std::size_t limit,bool& cacheable,DeviceImageArena*& owner,
            DeviceImageArena* transfer=nullptr,bool fallback=true) {
        owner=nullptr;
        if(!cacheable) {
            if(transfer) {
                auto image=transfer->allocate(bytes);
                if(!image.defined())throw MfqResourceExhausted("TQ transfer workspace is exhausted");
                return image;
            }
            return tb::empty({static_cast<int64_t>(bytes)},
                tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
        }
        auto& arenas=device_arenas_[device];
        for(const auto& arena:arenas)if(auto image=arena->allocate(bytes);image.defined()) {
            owner=arena.get();++device_reuses_;return image;
        }
        const auto aligned=(bytes+15)&~std::size_t(15);
        if(transfer) {
            // A running single-submission graph waits for this CPU task. It
            // cannot tolerate driver allocations/frees on the upload path.
            std::shared_ptr<DeviceImageArena> recycled;
            if(evict_device_arena_locked(device,&recycled,aligned)) {
                owner=recycled.get();++device_reuses_;return recycled->allocate(bytes);
            }
            cacheable=false;auto image=transfer->allocate(bytes);
            if(!image.defined())throw MfqResourceExhausted("TQ transfer workspace is exhausted");
            return image;
        }
        auto reserved=device_reserved_locked(device);
        while(aligned<=limit && reserved>limit-aligned) {
            if(!evict_device_arena_locked(device))break;
            reserved=device_reserved_locked(device);
        }
        if(aligned<=limit && reserved<=limit-aligned) {
            const auto capacity=std::max(aligned,std::min<std::size_t>(std::size_t(64)<<20,limit-reserved));
            auto arena=std::make_shared<DeviceImageArena>(capacity,device);
            auto image=arena->allocate(bytes);owner=arena.get();arenas.push_back(std::move(arena));return image;
        }
        if(!fallback){cacheable=false;return {};}
        // Leased rows can temporarily outlive a tiny cache or workspace loan.
        // They remain ordinary transient images and cannot enlarge the cache.
        cacheable=false;
        return tb::empty({static_cast<int64_t>(bytes)},
            tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
    }
    std::shared_ptr<MoePinnedExpertArena::Block> reuse_free_pinned(std::size_t bytes) {
        for(const auto& arena:host_arenas_)if(auto block=arena->allocate(bytes))return block;
        return {};
    }
    std::shared_ptr<MoePinnedExpertArena::Block> allocate_pinned(std::size_t bytes) {
        const auto allocate=[&] {
            return allocate_routed_pinned(host_arenas_,pinned_duplicates_,bytes,host_limit_,
            !replay_readers_[mfq_current_cuda_device()],host_reserved_,pinned_ceiling_,pin_failures_,
            [&](std::uint64_t key) {
                const auto& image=host_.at(key).image;return image->block && image.use_count()==1;
            },[&](std::uint64_t key){erase_host_locked(key);},
            [&]{return available_host_bytes(process_host_limit_);},[&](std::size_t){++pin_attempts_;},profile_?&registration_ms_:nullptr);
        };
        auto block=allocate();
        if(block || replay_readers_[mfq_current_cuda_device()] || layer_admission_ ||
                mfq::cuda::DecodeWindow::recording())return block;
        std::size_t holes=0,largest_arena=0;
        for(const auto& arena:host_arenas_) {
            holes+=arena->bytes()-arena->used_bytes();
            largest_arena=std::max(largest_arena,arena->bytes());
        }
        pin_fragment_probe_bytes_=bytes>SIZE_MAX-pin_fragment_probe_bytes_?SIZE_MAX:pin_fragment_probe_bytes_+bytes;
        if(holes<bytes || pin_fragment_probe_bytes_<largest_arena)return {};
        // Match the GGUF failed-demand threshold. Only a failed pinned
        // allocation scans host images; live DMA owners prevent their movement.
        pin_fragment_probe_bytes_=0;++pin_fragment_retries_;
        retire_host_transfers_locked();
        std::unordered_map<const void*,std::vector<std::uint64_t>> keys;
        for(const auto& [key,entry]:host_)if(entry.image->block)
            keys[entry.image->block->allocation_id()].push_back(key);
        const bool compacted=compact_host_interval(host_arenas_,keys,bytes,host_compacted_bytes_);
        if(compacted)++pin_interval_compactions_;
        else reclaim_host_locked(bytes,true);
        block=allocate();
        if(block){++pin_fragment_reuses_;pin_fragment_probe_bytes_=SIZE_MAX;}
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1')
            std::fprintf(stderr,"native_tq_pinned_fragment_reuse actual_match=1 requested_bytes=%zu free_locked_bytes=%zu reused=%d probes=%llu interval_compacted=%d\n",
                bytes,holes,int(bool(block)),static_cast<unsigned long long>(pin_fragment_retries_),int(compacted));
        return block;
    }
    void release_empty_arenas() {
        auto registration=mfq::cuda::try_host_registration();
        if(!registration.owns_lock())return;
        for(auto it=host_arenas_.begin();it!=host_arenas_.end();)
            if(!(*it)->used_bytes()){host_reserved_-=(*it)->bytes();it=host_arenas_.erase(it);}else ++it;
    }
    void retire_host_transfers_locked() {
        const mfq::cuda::Event* checked=nullptr;bool complete=false;
        mfq::cuda::RoutedCompletionProbe probe;
        const bool defer=layer_admission_ || replay_active_locked();
        for(auto it=host_transfers_.begin();it!=host_transfers_.end();) {
            // CUDA considers a never-recorded event complete. Keep queued
            // descriptors and their sources until their actual submission.
            if(!it->completed && !it->recorded_submission()){++it;continue;}
            if(!it->completed && it->ready.get()!=checked) {
                checked=it->ready.get();
                complete=probe.ready(it->position?*it->position:mfq::cuda::RoutedCompletionPosition{},[&] {
                    ++retirement_queries_;return checked->ready();
                });
            }
            if(!it->completed && !complete){++it;continue;}
            if(defer) {
                deferred_events_.retain(it->ready);
                for(const auto& image:it->images)deferred_events_.retain(image->ready);
            }
            it->completed=true;it=host_transfers_.erase(it);
        }
    }
    void settle_host_transfers_locked() {
        auto registration=mfq::cuda::try_host_registration();
        if(!registration.owns_lock())return;
        for(const auto& transfer:host_transfers_)
            if(!transfer.completed && transfer.recorded_submission())transfer.ready->synchronize();
        retire_host_transfers_locked();
    }
    std::size_t host_allocated_locked()const {
        return host_reserved_+host_pageable_;
    }
    // Publish ordinary admissions and RAM-pressure migrations through the same
    // residency indices and mapped address table. Migration enters at the cold
    // end so it cannot displace a repeatedly used expert.
    void commit_device_image_locked(DeviceKey key,const TqWeight& weight,const HostImage& image,
            DeviceImageArena* arena,std::optional<std::uint64_t> group={},bool migration=false) {
        if(!arena)throw std::logic_error("TQ resident image lost its arena owner");
        auto& policy=routes_.at(key.device);
        auto* group_slot=group?&expert_slots_.at(DeviceKey{*group,key.device}):nullptr;
        auto* pending=group?&pending_expert_bytes_.at(key.device):nullptr;
        policy.usage.image(key.image,weight.image.nbytes());
        if(resident_roles_.count({key.device,key.image>>32}))resident_updates_.insert(key.device);
        auto& device_bytes=device_usage_.try_emplace(key.device,0).first->second;
        auto& arena_keys=arena_entries_[arena];
        const auto [copies,created]=resident_copies_.try_emplace(key.image,0);
        std::optional<std::list<DeviceKey>::iterator> cold;
        std::list<DeviceKey>::iterator recent;
        bool arena_linked=false,recent_linked=false,indexed=false;
        try {
            arena_keys.push_front(key);arena_linked=true;
            recent=device_recent_.insert(migration?device_recent_.end():device_recent_.begin(),key);recent_linked=true;
            if(migration || !policy.ready || !policy.usage.retained(key.image))
                cold=policy.cold.insert(migration?policy.cold.end():policy.cold.begin(),key);
            const auto [_,inserted]=device_.emplace(key,DeviceEntry{weight,recent,image.source_bytes,
                image.layout,arena,arena_keys.begin(),cold,group});
            if(!inserted)throw std::logic_error("duplicate TQ resident image admission");
            ++copies->second;
            if(group_slot){++group_slot->residents;*pending-=weight.image.nbytes();}
            device_used_+=weight.image.nbytes();device_bytes+=weight.image.nbytes();indexed=true;
            policy.usage.resident(key.image,true);
            update_resident_view_locked(key,weight.view);
            mark_host_duplicate_locked(key.image);
        }catch(...) {
            if(indexed)erase_device_locked(key,false);
            else {
                if(cold)policy.cold.erase(*cold);
                if(recent_linked)device_recent_.erase(recent);
                if(arena_linked)arena_keys.pop_front();
                if(created)resident_copies_.erase(copies);
            }
            throw;
        }
    }
    tb::Tensor allocate_idle_device_image_locked(std::size_t bytes,int device,std::size_t reserve,
            const std::shared_ptr<mfq::cuda::Context>& context,DeviceImageArena*& owner) {
        owner=nullptr;
        const auto aligned=mfq::cuda::routed_aligned_bytes(bytes),limit=device_limit_locked(device);
        if(aligned>mfq::cuda::routed_host_evacuation_capacity(
                limit,device_used_locked(device),context->available_memory(),reserve))return {};
        auto& arenas=device_arenas_[device];
        for(const auto& arena:arenas)if(auto image=arena->allocate(aligned);image.defined()) {
            owner=arena.get();++device_reuses_;return image;
        }
        // Exact new storage only: never recycle an occupied arena, evict a
        // resident, or inflate the request to the ordinary growth quantum.
        const auto reserved=device_reserved_locked(device);
        if(aligned>limit-std::min(limit,reserved))return {};
        const mfq::cuda::RoutedWeightAllocationScope allocation;
        auto arena=std::make_shared<DeviceImageArena>(aligned,device);
        auto image=arena->allocate(aligned);owner=arena.get();arenas.push_back(std::move(arena));return image;
    }
    void evacuate_host_allocation_locked(const void* allocation) {
        const auto device=mfq_current_cuda_device();
        if(host_evacuation_blocked_ || layer_admission_ || replay_readers_[device] ||
                mfq::cuda::DecodeWindow::recording() || !decode_reserves_.count(device))return;
        const auto context=mfq::cuda::default_context(device);
        if(context->workspace_loan_active())return;
        for(const auto& [_,reservation]:workspace_limits_)if(reservation.device==device)return;
        cudaStreamCaptureStatus capture;
        TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture));
        if(capture!=cudaStreamCaptureStatusNone)return;
        const auto found_stream=transfer_streams_.find(device);
        if(found_stream==transfer_streams_.end())return;
        const auto stream=found_stream->second;
        TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(),&capture));
        if(capture!=cudaStreamCaptureStatusNone)return;
        const auto& saved=decode_reserves_.at(device);
        const auto reserve=mfq::cuda::routed_saturating_add(saved.growth,
            saved.transient.value_or(mfq::cuda::runtime_options::workspace_reserve_bytes()));
        if(context->available_memory()<=reserve)return;
        auto candidates=host_evacuation_candidates(allocation,[&](auto key) {
            const auto found=source_info_.find(key>>32);
            return found!=source_info_.end() && !found->second.resident;
        },[](const auto& image){return !image.ready || image.ready_on_host || image.ready->ready();});
        struct Move {
            DeviceKey key;
            std::shared_ptr<HostImage> image;
            TqWeight weight;
            DeviceImageArena* arena;
            std::optional<std::uint64_t> group;
            std::size_t reserved=0;
        };
        std::vector<Move> moves;moves.reserve(candidates.size());
        std::vector<mfq::cuda::DmaCopy> copies;copies.reserve(candidates.size());
        std::vector<CUdeviceptr> aliases;aliases.reserve(candidates.size());
        std::vector<DeviceKey> created_slots,committed;
        created_slots.reserve(candidates.size());committed.reserve(candidates.size());
        const MfqCudaStreamGuard upload(stream);
        const auto cleanup=[&] {
            for(const auto key:committed)if(device_.count(key))erase_device_locked(key,false);
            for(const auto key:created_slots)if(const auto found=expert_slots_.find(key);
                    found!=expert_slots_.end() && !found->second.residents) {
                pending_expert_bytes_[device]-=found->second.image.nbytes();expert_slots_.erase(found);
            }
            for(const auto& move:moves)pending_expert_bytes_[device]-=move.reserved;
            moves.clear();release_empty_device_arenas_locked(device);
        };
        try {
            auto& policy=routes_[device];
            for(const auto key:candidates) {
                const auto& image=host_.at(key).image;
                const auto source=key>>32,expert=std::uint64_t(std::uint32_t(key));
                const auto needed=mfq::cuda::routed_aligned_bytes(image->size());
                DeviceImageArena* arena=nullptr;
                std::optional<std::uint64_t> group;
                tb::Tensor storage;
                if(const auto* banks=policy.banks.find(source)) {
                    const DeviceKey group_key{((*banks)[0]<<32)|expert,device};
                    auto slot=expert_slots_.find(group_key);
                    if(slot==expert_slots_.end()) {
                        bool partial=false;std::array<std::size_t,3> sizes{};
                        for(unsigned role=0;role<3;++role) {
                            const auto& info=source_info_.at((*banks)[role]);
                            sizes[role]=info.table->expert_image_bytes(expert,info.rows);
                            partial|=device_.count(DeviceKey{((*banks)[role]<<32)|expert,device})!=0;
                        }
                        if(!partial) {
                            const mfq::cuda::RoutedExpertImageLayout layout(*banks,sizes);
                            auto slot_image=allocate_idle_device_image_locked(layout.bytes,device,reserve,context,arena);
                            if(!slot_image.defined())continue;
                            pending_expert_bytes_.try_emplace(device,0);
                            slot=expert_slots_.emplace(group_key,ExpertSlot{std::move(slot_image),arena,layout}).first;
                            created_slots.push_back(group_key);pending_expert_bytes_[device]+=layout.bytes;
                        }
                    }
                    if(slot!=expert_slots_.end()) {
                        const auto role=std::find(banks->begin(),banks->end(),source)-banks->begin();
                        if(needed!=slot->second.layout.sizes[role])throw std::logic_error("TQ migration slot byte count differs");
                        storage=slot->second.image.narrow(0,slot->second.layout.offsets[role],needed);
                        arena=slot->second.arena;group=group_key.image;
                    }
                }
                if(!storage.defined())storage=allocate_idle_device_image_locked(needed,device,reserve,context,arena);
                if(!storage.defined())continue;
                auto weight=tq_view(std::move(storage),image->layout,source_info_.at(source).table->width());
                weight.allocation_stream=stream;
                pending_expert_bytes_.try_emplace(device,0);
                moves.push_back({DeviceKey{key,device},image,std::move(weight),arena,group,group?0:needed});
                pending_expert_bytes_[device]+=moves.back().reserved;
                copies.push_back({moves.back().weight.image.data_ptr(),image->data(),image->size()});
                aliases.push_back(reinterpret_cast<CUdeviceptr>(image->mapped_data(device)));
            }
            if(moves.empty()){cleanup();return;}
            std::size_t index=0;
            if(mfq::cuda::submit_routed_transfers(copies,stream.stream(),mfq::cuda::routed_dma_mode(),
                    [&](const void*,std::size_t){return aliases[index++];}))++native_batches_;
            for(const auto& move:moves){transfer_bytes_+=move.image->size();direct_transfer_bytes_+=move.image->size();}
            // No mapped residency address is published, and no RAM owner is
            // removed, until this independent upload lane has completed.
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            if(context->available_memory()<reserve){cleanup();context->reclaim_idle_memory();return;}
            for(auto& move:moves) {
                commit_device_image_locked(move.key,move.weight,*move.image,move.arena,move.group,true);
                committed.push_back(move.key);
                pending_expert_bytes_[device]-=move.reserved;move.reserved=0;
            }
        }catch(const MfqResourceExhausted&) {
            (void)cudaStreamSynchronize(stream.stream());cleanup();context->reclaim_idle_memory();return;
        }catch(const mfq::cuda::Error& error) {
            (void)cudaStreamSynchronize(stream.stream());cleanup();
            if(std::string(error.what()).find(cudaGetErrorString(cudaErrorMemoryAllocation))==std::string::npos)throw;
            (void)cudaGetLastError();context->reclaim_idle_memory();return;
        }catch(...) {(void)cudaStreamSynchronize(stream.stream());cleanup();throw;}
        for(const auto& move:moves){host_evacuated_bytes_+=move.weight.image.nbytes();++host_evacuated_images_;}
    }
    std::size_t reclaim_host_locked(std::size_t bytes,bool duplicates_only=false) {
        retire_host_transfers_locked();
        const auto before=host_allocated_locked();release_empty_arenas();
        const auto victims=host_allocation_victims(host_arenas_);
        std::unordered_set<const void*> retained;
        for(bool duplicates:{true,false}) {
            if(!duplicates && duplicates_only)break;
            for(const auto& victim:victims) {
                if(before-host_allocated_locked()>=bytes)return before-host_allocated_locked();
                if(!duplicates)evacuate_host_allocation_locked(victim.allocation);
                for(const auto key:victim.keys)if(const auto found=host_.find(key);found!=host_.end()) {
                    const auto& image=found->second.image;
                    const auto allocation=image->block?image->block->allocation_id():static_cast<const void*>(image.get());
                    if(allocation==victim.allocation && (!duplicates || found->second.duplicate))erase_host_locked(key);
                }
                if(duplicates)host_trimmed_bytes_+=trim_host_image_ranges(victim.allocation,victim.keys);
                release_empty_arenas();
                if(duplicates && before-host_allocated_locked()<bytes && !retained.count(victim.allocation)) {
                    host_compacted_bytes_+=relocate_host_images(host_arenas_,victim.allocation,victim.keys,retained,
                        [](std::uint64_t,const void*,const void*){});
                    release_empty_arenas();
                }
            }
        }
        return before-host_allocated_locked();
    }
    std::size_t reserve_host_locked(std::size_t bytes) {
        if(!bytes)return 0;
        auto available=available_host_bytes(process_host_limit_);
        available=host_headroom_after_allocator_trim(bytes,available,
            []{mfq_release_host_allocator_cache();},[&]{return available_host_bytes(process_host_limit_);});
        return available>=bytes || host_.empty()?0:reclaim_host_locked(bytes-available);
    }
    void report()const {
        if(profile_)std::fprintf(stderr,"tq_cache calls=%llu gpu_hits=%llu host_hits=%llu source_bytes=%llu h2d_bytes=%llu fetch_ms=%.3f upload_submit_ms=%.3f gpu_bytes=%zu host_bytes=%zu host_limit=%zu direct_h2d_bytes=%llu pin_attempts=%llu pin_failures=%llu\n",
            static_cast<unsigned long long>(calls_),static_cast<unsigned long long>(gpu_hits_),
            static_cast<unsigned long long>(host_hits_),static_cast<unsigned long long>(source_bytes_),
            static_cast<unsigned long long>(transfer_bytes_),fetch_ms_,upload_submit_ms_,device_used_,host_used_,host_limit_,
            static_cast<unsigned long long>(direct_transfer_bytes_),static_cast<unsigned long long>(pin_attempts_),
            static_cast<unsigned long long>(pin_failures_));
    }
    template<class Bytes,class Queue,class Flush>
    void stage_upload(const tb::Tensor& destination,const Bytes& bytes,Queue&& queue,Flush&& flush) {
        const bool trace=std::getenv("TYLOINFER_TQ_TRACE_ROWS")!=nullptr;
        const auto step=[&](const char* phase) {
            if(trace){std::fprintf(stderr,"tq_dma step=%s bytes=%zu\n",phase,bytes.size());std::fflush(stderr);}
        };
        step("begin");
        const auto started=std::chrono::steady_clock::now();
        const auto stream=mfq_get_current_cuda_stream(destination.get_device());
        auto& staging=stages_[destination.get_device()][stream.stream()];
        staging.owner=stream;
        auto* stage=staging.stages[staging.index].get();
        if(!stage || bytes.size()>stage->host.size()-stage->cursor) {
            // Submit every queued reference before an interval can be reused.
            // A bank larger than the ring is streamed in bounded packets.
            flush();
            staging.index^=1;
            auto& selected=staging.stages[staging.index];
            if(selected) {
                step("reuse_wait");
                {CpuTimer timer(profile_?&stage_wait_ms_:nullptr);selected->reuse();}
                step("reuse_done");
            }
            // Bound normal staging, while allowing any valid projection larger
            // than the normal chunk. This is a host DMA arena, not a GPU limit.
            const auto capacity=std::max<std::size_t>(std::size_t(32)<<20,bytes.size());
            if(!selected || selected->host.size()<capacity)selected=std::make_unique<Stage>(capacity);
            stage=selected.get();
        }
        auto* source=static_cast<std::uint8_t*>(stage->host.data())+stage->cursor;
        step("cpu_copy");
        {CpuTimer timer(profile_?&stage_copy_ms_:nullptr);mfq::copy_host_range({source,bytes.data(),bytes.size()});}
        step("cpu_done");
        // Keep DMA and projection-metadata staging independent. Reusing a
        // completion event across streams can wait on a later graph replay
        // which is itself waiting for this upload.
        step("gpu_copy");
        queue(mfq::cuda::DmaCopy{destination.data_ptr(),source,bytes.size()},
            reinterpret_cast<CUdeviceptr>(stage->mapped+stage->cursor),stage);
        step("gpu_submitted");
        stage->cursor+=bytes.size();transfer_bytes_+=bytes.size();
        upload_submit_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }
    template<class Bytes> void upload(const tb::Tensor& destination,const Bytes& bytes) {
        const auto stream=mfq_get_current_cuda_stream(destination.get_device());
        stage_upload(destination,bytes,[&](const mfq::cuda::DmaCopy& copy,CUdeviceptr,Stage* stage) {
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(copy.destination,copy.source,copy.bytes,cudaMemcpyHostToDevice,stream.stream()));
            stage->uploaded.record(stream.stream());stage->pending=true;
        },[]{});
    }
public:
    struct UploadBatch {
        TqExpertCache& owner;
        MfqCudaStream stream,consumer;
        std::shared_ptr<mfq::cuda::Event> ready;
        std::shared_ptr<std::atomic<bool>> submitted=std::make_shared<std::atomic<bool>>(false);
        std::shared_ptr<std::atomic<bool>> recorded=std::make_shared<std::atomic<bool>>(false);
        std::shared_ptr<mfq::cuda::RoutedCompletionPosition> position=std::make_shared<mfq::cuda::RoutedCompletionPosition>();
        std::vector<DeviceKey> admitted;
        bool finished=false;
        UploadBatch(TqExpertCache& cache,MfqCudaStream upload,MfqCudaStream compute)
            :owner(cache),stream(std::move(upload)),consumer(std::move(compute)) {}
        UploadBatch(const UploadBatch&)=delete;
        UploadBatch& operator=(const UploadBatch&)=delete;
        ~UploadBatch(){if(!finished)owner.abort_uploads(*this);}
    };
    std::unique_ptr<UploadBatch> begin_uploads(int device) {
        return std::make_unique<UploadBatch>(*this,transfer_stream(device),mfq_get_current_cuda_stream(device));
    }
    void finish_uploads(UploadBatch& batch) {
        std::lock_guard lock(mutex_);
        if(batch.finished)return;
        const MfqCudaGuard device(batch.stream.device());
        try {
            if(batch.ready) {
                ++upload_groups_;
                batch.ready->record(batch.stream.stream());
                *batch.position=mfq::cuda::record_routed_completion(transfer_clock_,batch.stream.device(),batch.stream.stream());
                batch.recorded->store(true,std::memory_order_release);
                if(batch.consumer.stream()!=batch.stream.stream())
                    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(batch.consumer.stream(),batch.ready->get(),0));
            }
            batch.submitted->store(true,std::memory_order_release);batch.finished=true;
        }catch(...) {
            (void)cudaStreamSynchronize(batch.stream.stream());throw;
        }
    }
    void upload_decode_views(const tb::Tensor& target,std::size_t offset,const void* source,std::size_t bytes) {
        // The graph stage owns this pinned packet until its DMA and replay
        // finish. Upload only its selected slots, without a staging-ring copy.
        const auto started=profile_?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(target.data_ptr<std::uint8_t>()+offset,source,bytes,
            cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
        std::lock_guard lock(mutex_);
        decode_metadata_bytes_+=bytes;++decode_metadata_updates_;transfer_bytes_+=bytes;
        if(profile_)decode_metadata_submit_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }
    void touch_decode_residents(const std::array<std::uint64_t,3>& sources,
            std::span<const std::int64_t> experts,int device) {
        if(experts.empty())return;
        std::lock_guard lock(mutex_);
        if(std::all_of(sources.begin(),sources.end(),[&](auto source) {
            return bool(source_info_.at(source).resident);
        }))return;
        const auto started=profile_?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        if(!replay_readers_[device])throw std::logic_error("resident decode touch requires a frozen cache epoch");
        for(const auto source:sources)for(const auto expert:experts) {
            if(const auto& permanent=source_info_.at(source).resident) {
                if(permanent->packed.get_device()!=device || expert<0 ||
                        std::uint64_t(expert)>=permanent->experts.size())
                    throw std::invalid_argument("TQ permanent resident view mismatch");
                continue;
            }
            const auto found=device_.find({(source<<32)|std::uint64_t(expert),device});
            if(found==device_.end())throw std::logic_error("published resident expert lost its backing image");
            device_recent_.splice(device_recent_.begin(),device_recent_,found->second.recent);
            if(found->second.cold) {
                auto& cold=routes_.at(device).cold;cold.splice(cold.begin(),cold,*found->second.cold);
            }
        }
        const auto hits=experts.size()*sources.size();
        calls_+=hits;gpu_hits_+=hits;decode_calls_+=hits;decode_hits_+=hits;
        decode_resident_bypasses_+=hits;
        if(profile_) {
            const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            decode_resident_touch_ms_+=elapsed;decode_get_rows_ms_+=elapsed;get_rows_ms_+=elapsed;
        }
    }
    void abort_uploads(UploadBatch& batch) noexcept {
        try {
            const MfqCudaGuard device(batch.stream.device());
            std::lock_guard lock(mutex_);
            // Submitted banks and their sources remain alive until this
            // retirement fence, including failure while preparing a later bank.
            if(batch.ready) {
                batch.ready->record(batch.stream.stream());
                *batch.position=mfq::cuda::record_routed_completion(transfer_clock_,batch.stream.device(),batch.stream.stream());
                batch.recorded->store(true,std::memory_order_release);
            }
            batch.submitted->store(true,std::memory_order_release);
            for(const auto& key:batch.admitted)if(device_.count(key))erase_device_locked(key,false);
            batch.finished=true;
        }catch(...) {
            try {
                const MfqCudaGuard device(batch.stream.device());
                if(cudaStreamSynchronize(batch.stream.stream())==cudaSuccess) {
                    std::lock_guard lock(mutex_);
                    // A failed event record has no queryable fence. A completed
                    // stream is nevertheless an authoritative lifetime boundary.
                    for(auto& transfer:host_transfers_)
                        if(transfer.submitted==batch.submitted)transfer.completed=true;
                    for(const auto& key:batch.admitted)if(device_.count(key))erase_device_locked(key,false);
                    batch.finished=true;
                }
            }catch(...) {}
        }
    }
    TqExpertCache(std::size_t budget,std::size_t host_budget,std::size_t process_limit,bool retain_all)
        :host_limit_(host_budget?host_budget:available_host_bytes(process_limit)),ordinary_host_limit_(host_limit_),device_limit_(budget),
         retain_all_(retain_all),process_host_limit_(process_limit) {}
    bool set_reuse_learning(bool enabled) {
        std::lock_guard lock(mutex_);return std::exchange(reuse_learning_,enabled);
    }
    void record_prefill(const std::array<std::uint64_t,3>& sources,
            const std::vector<std::int64_t>& experts,const std::vector<std::vector<std::int64_t>>& slots,
            std::size_t rows,int device) {
        std::lock_guard lock(mutex_);
        if(!reuse_learning_ || source_info_.at(sources[0]).resident)return;
        auto& policy=routes_[device];
        register_banks_locked(policy,sources);
        // Once decode has trained this bank, prompt preparation must preserve
        // its working set, matching the existing fixed-slot cache lifecycle.
        if(policy.decoded_banks.count(sources[0]))return;
        for(auto source:sources)policy.usage.reset_prefill(source);
        for(std::size_t i=0;i<experts.size();++i)for(auto source:sources)
            policy.usage.prefill((source<<32)|std::uint64_t(experts[i]),slots[i].size(),rows);
        policy.seeded=true;policy.ready=false;
    }
    void observe_routes(const std::array<std::uint64_t,3>& sources,const std::int64_t* experts,
            int rows,int count,int device) {
        std::lock_guard lock(mutex_);
        if(!reuse_learning_ || source_info_.at(sources[0]).resident)return;
        register_banks_locked(routes_[device],sources);
        route_journal_.collect({experts,std::size_t(rows)*count},std::size_t(source_info_.at(sources[0]).experts),
            route_touches_,[&](auto expert) {
                for(auto source:sources)if(!device_.count(DeviceKey{(source<<32)|std::uint64_t(expert),device}))return true;
                return false;
            });
        if(speculative_rows_) {speculative_routes_.push_back({device,rows,sources,route_touches_});return;}
        auto& policy=routes_[device];
        if(!route_touches_.empty())policy.decoded_banks.insert(sources[0]);
        for(const auto& touch:route_touches_)for(auto source:sources)for(std::uint64_t i=0;i<touch.uses;++i) {
            policy.usage.observe((source<<32)|std::uint64_t(touch.expert));++policy.observations;policy.seeded=true;
        }
    }
    void finish_routes(int device) {
        bool refresh=false;
        std::vector<std::shared_ptr<mfq::cuda::Event>> retired;
        {
            std::lock_guard lock(mutex_);
            if(replay_readers_[device])--replay_readers_[device];
            if(!replay_active_locked())retired=deferred_events_.take();
            if(!speculative_rows_)refresh=finish_routes_locked(device);
        }
        retired.clear();
        if(refresh)queue_admission(device);
    }
    void begin_replay(int device) {
        admission_executor_->transition([&](auto& admission) {
            // Freeze metadata after submission. Each layer joins its own DMA
            // through the same admission flag protocol used by GGUF.
            admission.wait();
            const auto upload=transfer_stream(device);
            std::lock_guard lock(mutex_);
            if(layer_admission_ && layer_admission_->device==device)layer_admission_->require_active();
            if(resident_updates_.count(device)) {
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(upload.stream()));
                resident_updates_.erase(device);
            }
            for(const auto& [_,view]:resident_views_)if(view->device==device && view->rebuild)view->republish();
            ++replay_readers_[device];
        });
    }
    void release_deferred_events() {
        std::vector<std::shared_ptr<mfq::cuda::Event>> retired;
        {
            std::lock_guard lock(mutex_);
            if(!replay_active_locked())retired=deferred_events_.take();
        }
        retired.clear();
    }
    void wait_ready(){wait_admission();release_deferred_events();}
    std::shared_ptr<void> prepare_prefill_staging(const std::array<std::uint64_t,3>& sources,
            std::span<const std::int64_t> experts,std::span<const std::size_t> ends,int device) {
        if(!RoutedPrefillStager::enabled() || experts.empty())return {};
        std::lock_guard lock(mutex_);
        if(!staged_host_jobs_.empty())throw std::logic_error("overlapping routed host staging");
        RoutedPrefillStagePlan plan;
        std::size_t begin=0;
        for(const auto end:ends) {
            for(const auto source:sources)for(auto i=begin;i<end;++i) {
                const auto key=(source<<32)|std::uint64_t(experts[i]);
                const auto found=host_.find(key);
                if(found==host_.end() || found->second.image->pinned() || device_.count({key,device}))continue;
                const auto& image=found->second.image;
                plan.add(key,{image->data(),image->size(),image});
            }
            begin=end;
        }
        return plan.start(prefill_host_stager_,staged_host_jobs_,device,mutex_);
    }
    void wait_layer_admission(const std::shared_ptr<ResidentViews>& view) {
        const auto started=profile_?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        if(!view){wait_admission();return;}
        std::shared_ptr<LayerAdmission> epoch;
        {
            std::lock_guard lock(mutex_);epoch=layer_admission_;
            if(!epoch || epoch->device!=view->device)return;
        }
        const auto* ready=reinterpret_cast<const volatile std::uint32_t*>(view->admission_host());
        mfq::cuda::wait_routed_layer_admission([&]{return *ready!=0;},mfq::cpu::spin_pause,[&] {
            auto error=std::make_exception_ptr(std::runtime_error("routed layer admission timed out"));
            abort_layer_admission(epoch,error);std::rethrow_exception(error);
        });
        std::lock_guard lock(mutex_);epoch->require_active();
        if(profile_)decode_admission_wait_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }
    void cancel_replay(int device) noexcept {
        std::lock_guard lock(mutex_);
        if(auto found=replay_readers_.find(device);found!=replay_readers_.end() && found->second)--found->second;
    }
    std::shared_ptr<ResidentViews> resident_views(std::array<std::uint64_t,3> sources,std::int64_t experts,int device) {
        return admission_executor_->transition([&](auto& admission) {
            join_admission(admission);
            std::lock_guard lock(mutex_);
            register_banks_locked(routes_[device],sources);
            const std::array<std::uint64_t,4> key{sources[0],sources[1],sources[2],std::uint64_t(device)};
            if(auto found=resident_views_.find(key);found!=resident_views_.end())return found->second;
            auto views=std::make_shared<ResidentViews>(sources,experts,device);
            for(int role=0;role<3;++role) {
                if(const auto& permanent=source_info_.at(sources[role]).resident) {
                    if(permanent->packed.get_device()!=device || permanent->experts.size()!=std::size_t(experts))
                        throw std::invalid_argument("TQ permanent resident view mismatch");
                    const auto stream=mfq_get_current_cuda_stream(device).stream();
                    permanent->packed.record_stream(reinterpret_cast<std::uintptr_t>(stream));
                    permanent->views.record_stream(reinterpret_cast<std::uintptr_t>(stream));
                    for(std::int64_t expert=0;expert<experts;++expert)
                        views->update(role,expert,permanent->experts[expert].view,true);
                    continue;
                }
                resident_roles_[{device,sources[role]}].push_back({views,role});
                for(std::int64_t expert=0;expert<experts;++expert) {
                    const auto found=device_.find(DeviceKey{(sources[role]<<32)|std::uint64_t(expert),device});
                    if(found!=device_.end())views->update(role,expert,found->second.weight.view,true);
                }
            }
            resident_views_.emplace(key,views);return views;
        });
    }
    void begin_routes(int rows) {
        if(rows<1)throw std::invalid_argument("TQ speculative route window must be positive");
        std::lock_guard lock(mutex_);
        if(speculative_rows_)throw std::logic_error("TQ speculative route window already active");
        speculative_rows_=rows;speculative_routes_.clear();
    }
    void commit_routes(int rows) {
        std::unique_lock lock(mutex_);
        if(rows<0 || rows>speculative_rows_)throw std::invalid_argument("TQ accepted route rows exceed window");
        std::unordered_set<int> devices;
        if(reuse_learning_)for(const auto& route:speculative_routes_)
            if(mfq::cuda::routed_observed_rows(rows,route.rows)) {
                auto& policy=routes_[route.device];
                if(!route.touches.empty())policy.decoded_banks.insert(route.sources[0]);
                for(const auto& touch:route.touches)for(auto source:route.sources)for(std::uint64_t i=0;i<touch.uses;++i) {
                    policy.usage.observe((source<<32)|std::uint64_t(touch.expert));++policy.observations;
                }
                policy.seeded=true;devices.insert(route.device);
            }
        speculative_routes_.clear();speculative_rows_=0;
        for(auto device:devices)finish_routes_locked(device,true);
        lock.unlock();
        for(auto device:devices)queue_admission(device);
    }
    void discard_routes() noexcept {
        std::lock_guard lock(mutex_);speculative_routes_.clear();speculative_rows_=0;
    }
    MfqCudaStream transfer_stream(int device) {
        std::lock_guard<std::mutex> guard(mutex_);
        auto found=transfer_streams_.find(device);
        if(found==transfer_streams_.end())found=transfer_streams_.emplace(device,mfq_get_stream_from_pool(true,device)).first;
        completion_events_.prepare(device,found->second.stream());
        return found->second;
    }
    MfqCudaStream admission_stream(int device) {
        std::lock_guard lock(mutex_);
        auto found=admission_streams_.find(device);
        if(found==admission_streams_.end())found=admission_streams_.emplace(device,mfq_get_stream_from_pool(false,device)).first;
        completion_events_.prepare(device,found->second.stream());
        return found->second;
    }
    void prepare_transfer_staging(int device) {
        const auto upload=transfer_stream(device);
        std::lock_guard<std::mutex> guard(mutex_);
        auto& staging=stages_[device][upload.stream()];staging.owner=upload;
        for(auto& stage:staging.stages)if(!stage)stage=std::make_unique<Stage>(std::size_t(32)<<20);
    }
    ~TqExpertCache(){
        try{wait_admission();}catch(...){}
        {
            std::lock_guard lock(mutex_);
            std::unordered_set<PrefillLoan*> visited;
            for(const auto& [_,reservation]:workspace_limits_)if(reservation.loan && visited.insert(reservation.loan.get()).second) {
                try {
                    const MfqCudaGuard device(reservation.device);
                    const MfqCudaStreamGuard stream(reservation.stream);
                    fence_workspace_locked(reservation.device);reservation.loan->finish();
                }catch(const std::exception& error) {
                    // An outstanding tensor still owns its pointer through
                    // Context. The shared loan retains its arena on failure.
                    std::fprintf(stderr,"TQ prefill backing retained during cache shutdown: %s\n",error.what());
                }
            }
        }
        admission_executor_.reset();
        host_transfers_.clear();
        device_.clear();graph_pools_.clear();
        report();
    }
    void prepare_graph_pool() {
        wait_admission();
        const auto stream=mfq_get_current_cuda_stream();
        std::lock_guard<std::mutex> guard(mutex_);
        graph_pools_.prepare(stream);
    }
    std::uint64_t register_source(std::shared_ptr<TqRowTable> table,std::int64_t rows,std::int64_t experts,
            std::shared_ptr<TqResidentBank> resident={}) {
        const auto bytes=table->storage_bytes(),resident_bound=table->routed_image_storage_bound(experts);
        std::lock_guard<std::mutex> guard(mutex_);
        if(resident ? resident->packed.nbytes()>SIZE_MAX-predictor_resident_bytes_ :
                bytes>SIZE_MAX-payload_bytes_ || resident_bound>SIZE_MAX-resident_bytes_bound_)
            throw std::overflow_error("TQ source payload exceeds size limits");
        if(++sources_>UINT32_MAX)throw std::overflow_error("too many TQ projection sources");
        if(resident)predictor_resident_bytes_+=resident->packed.nbytes();
        else {payload_bytes_+=bytes;resident_bytes_bound_+=resident_bound;}
        source_info_.emplace(sources_,SourceInfo{std::move(table),rows,experts,std::move(resident)});
        return sources_;
    }
    std::size_t source_bytes() {
        std::lock_guard lock(mutex_);return payload_bytes_;
    }
    std::size_t budget_bytes() {
        std::lock_guard<std::mutex> guard(mutex_);
        return payload_bytes_?device_limit_locked():0;
    }
    std::size_t lendable_bytes() {
        std::lock_guard<std::mutex> guard(mutex_);
        return std::max(device_used_locked(mfq_current_cuda_device()),device_reserved_locked(mfq_current_cuda_device()));
    }
private:
    void fence_workspace_locked(int device) {
        const auto current=mfq_get_current_cuda_stream(device);
        const auto producer=[device](const auto& streams) {
            const auto found=streams.find(device);
            return found==streams.end()?cudaStream_t{}:found->second.stream();
        };
        mfq::cuda::synchronize_routed_streams(current,
            {producer(transfer_streams_),producer(admission_streams_)});
        settle_host_transfers_locked();
    }
    void reserve_workspace_locked(std::size_t bytes,const std::shared_ptr<mfq::cuda::Context>& context,int device) {
        // The loan allocator cannot combine independent arena intervals. Keep
        // ordinary headroom for a single allocation as large as the complete
        // request; small borrowed holes are optional reuse, never this credit.
        auto available=context->available_memory();
        if(available>=bytes || context->workspace_largest_available(mfq_get_current_cuda_stream(device).stream())>=bytes)return;
        if(replay_readers_[device])
            throw MfqResourceExhausted("expert weights are leased by an active decode window");
        fence_workspace_locked(device);
        while(available<bytes && (!device_recent_.empty() || device_reserved_locked(device))) {
            const auto needed=bytes-available;
            std::size_t released=0;
            while(released<needed) {
                if(const auto chunk=evict_device_arena_locked(device)) {released+=chunk;continue;}
                const auto before=device_used_locked(device);
                if(!evict_oldest_device_locked(device))break;
                released+=before-device_used_locked(device);
            }
            settle_host_transfers_locked();
            release_empty_device_arenas_locked(device);
            context->reclaim_idle_memory();
            available=context->available_memory();
            if(!released)break;
        }
    }
public:
    void reserve_workspace(std::size_t bytes) {
        admission_executor_->transition([&](auto& admission) {
            join_admission(admission);
            const auto device=mfq_current_cuda_device();
            const auto context=mfq::cuda::default_context(device);
            context->reclaim_idle_memory();
            std::lock_guard<std::mutex> guard(mutex_);
            reserve_workspace_locked(bytes,context,device);
        });
    }
    std::pair<std::uint64_t,std::size_t> begin_workspace(std::size_t bytes) {
        std::pair<std::uint64_t,std::size_t> result{};
        admission_executor_->transition([&](auto& admission) {
            join_admission(admission);
            const auto device=mfq_current_cuda_device();
            const auto context=mfq::cuda::default_context(device);
            const auto stream=mfq_get_current_cuda_stream(device);
            context->reclaim_idle_memory();
            std::lock_guard<std::mutex> guard(mutex_);
            if(next_workspace_token_==UINT64_MAX)throw std::overflow_error("TQ workspace reservation token overflow");
            std::shared_ptr<PrefillLoan> loan;
            for(const auto& [_,reservation]:workspace_limits_)if(reservation.device==device && reservation.loan) {
                loan=reservation.loan;
                if(!loan->matches(context,stream))
                    throw MfqResourceExhausted("TQ prefill workspace is owned by another stream or thread");
                break;
            }
            const auto before=context->available_memory();
            auto available=before;
            bool started=false;
            const auto borrow=[&](bool complete) {
                if(loan || replay_readers_[device] || context->workspace_loan_active())return;
                // Only borrow already free intervals. In particular, never
                // evict a whole fixed-size arena just to manufacture a loan.
                fence_workspace_locked(device);
                available=context->available_memory();
                std::shared_ptr<DeviceImageArena> candidate;
                if(const auto found=device_arenas_.find(device);found!=device_arenas_.end())
                    for(const auto& arena:found->second)
                        if(!candidate || arena->largest_available()>candidate->largest_available())candidate=arena;
                const auto largest=candidate?candidate->largest_available():0;
                const auto request=complete?mfq::cuda::routed_workspace_full_borrow_request(bytes,largest):
                    mfq::cuda::routed_workspace_borrow_request(bytes,before,available,largest);
                if(request)if(auto storage=candidate->allocate(request);storage.defined()) {
                    if(complete && mfq::cuda::routed_workspace_region_bytes(storage.data_ptr(),storage.nbytes())<bytes)return;
                    try {loan=PrefillLoan::begin(context,stream,std::move(storage));started=true;}
                    catch(const mfq::cuda::Error& error) {
                        // A suspended or fully occupied GGUF loan reports zero
                        // available bytes but still owns Context's single slot.
                        // Its existing owner remains untouched in this case.
                        if(std::string_view(error.what())!="CUDA workspace loan is already active")throw;
                    }
                }
            };
            borrow(true);
            // A single free interval can cover the complete additional working
            // set. Otherwise keep full ordinary headroom; aggregate loan holes
            // are deliberately never subtracted from that reserve.
            const bool contiguous=context->workspace_largest_available(stream.stream())>=bytes;
            if(!contiguous) {
                reserve_workspace_locked(bytes,context,device);available=context->available_memory();
                if(loan && available<bytes)
                    throw MfqResourceExhausted("insufficient ordinary CUDA workspace for a nested prefill loan");
                borrow(false);
            }
            const auto capacity=mfq::cuda::routed_saturating_add(
                mfq::cuda::routed_saturating_add(device_used_locked(device),available),contiguous?bytes:0);
            const auto token=++next_workspace_token_;
            try {
                workspace_limits_.emplace(token,WorkspaceReservation{device,capacity>bytes?capacity-bytes:0,bytes,
                    stream,std::this_thread::get_id(),loan});
            }catch(...) {if(started)loan->finish();throw;}
            result={token,loan?loan->bytes():0};
        });
        return result;
    }
    void suspend_workspace(std::uint64_t token) {
        admission_executor_->transition([&](auto& admission) {
            join_admission(admission);
            std::lock_guard<std::mutex> guard(mutex_);
            const auto found=workspace_limits_.find(token);if(found==workspace_limits_.end())return;
            if(found->second.thread!=std::this_thread::get_id())
                throw std::logic_error("TQ workspace reservation belongs to another thread");
            const MfqCudaGuard device(found->second.device);
            const MfqCudaStreamGuard stream(found->second.stream);
            if(const auto loan=found->second.loan) {
                loan->suspend();
                const auto context=mfq::cuda::default_context(found->second.device);
                // The caller now clones persistent outputs out of the loan,
                // as in GGUF. Cover every currently live loan byte so those
                // copies do not depend on another borrowed interval.
                auto copies=context->workspace_live_bytes();
                for(const auto& [other,reservation]:workspace_limits_)
                    if(other!=token && reservation.loan==loan)copies=std::max(copies,reservation.required);
                reserve_workspace_locked(copies,context,found->second.device);
                const auto available=context->available_memory();
                if(available<copies)throw MfqResourceExhausted("insufficient ordinary CUDA workspace for prefill output copies");
                const auto capacity=mfq::cuda::routed_saturating_add(device_used_locked(found->second.device),available);
                const auto limit=capacity>copies?capacity-copies:0;
                for(auto& [_,reservation]:workspace_limits_)if(reservation.loan==loan)
                    reservation.limit=std::min(reservation.limit,limit);
            }
        });
    }
    void end_workspace(std::uint64_t token) {
        admission_executor_->transition([&](auto& admission) {
            join_admission(admission);
            std::lock_guard<std::mutex> guard(mutex_);
            const auto found=workspace_limits_.find(token);if(found==workspace_limits_.end())return;
            const auto& reservation=found->second;
            if(reservation.thread!=std::this_thread::get_id())
                throw std::logic_error("TQ workspace reservation belongs to another thread");
            if(const auto loan=reservation.loan) {
                const bool shared=std::any_of(workspace_limits_.begin(),workspace_limits_.end(),[&](const auto& item) {
                    return item.first!=token && item.second.loan==loan;
                });
                if(!shared) {
                    const MfqCudaGuard device(reservation.device);
                    const MfqCudaStreamGuard stream(reservation.stream);
                    fence_workspace_locked(reservation.device);
                    loan->finish(); // Preserve the token and owner if a tensor is still live.
                }
            }
            workspace_limits_.erase(found);
        });
    }
    void abandon_workspace(std::uint64_t token) noexcept {
        std::shared_ptr<PrefillLoan> abandoned;
        try {
            std::lock_guard<std::mutex> guard(mutex_);
            const auto found=workspace_limits_.find(token);if(found==workspace_limits_.end())return;
            abandoned=std::move(found->second.loan);
            workspace_limits_.erase(found);
        }catch(...) {
            std::fprintf(stderr,"TQ prefill reservation abandonment failed\n");
        }
        // Drop the owner outside the cache mutex. Other reservations retain
        // their shared loan; the last owner retires Context's loan, whose
        // backing remains alive until its final live tensor is released.
    }
    void finalize(std::size_t growth) {
        const auto workspace=mfq::cuda::runtime_options::workspace_reserve_bytes();
        const auto reserve=growth>SIZE_MAX-workspace?SIZE_MAX:growth+workspace;
        reserve_workspace(reserve);
        const auto device=mfq_current_cuda_device();
        const auto available=mfq::cuda::default_context(device)->available_memory();
        std::lock_guard lock(mutex_);
        decode_reserves_[device]={growth,workspace};
        decode_limits_[device]=std::max(device_used_locked(device),device_reserved_locked(device))+
            (available>reserve?available-reserve:0);
    }
    void constrain_capture(std::size_t bytes) {
        const auto device=mfq_current_cuda_device();
        const auto available=mfq::cuda::default_context(device)->available_memory();
        std::lock_guard lock(mutex_);
        const auto capacity=std::max(device_used_locked(device),device_reserved_locked(device))+
            (available>bytes?available-bytes:0);
        decode_limits_[device]=std::min(device_limit_locked(device),capacity);
    }
    void restore_after_capture() {
        std::size_t growth;std::optional<std::size_t> transient;
        {std::lock_guard lock(mutex_);const auto saved=decode_reserves_[mfq_current_cuda_device()];
         growth=saved.growth;transient=saved.transient;}
        restore_decode(growth,transient);
    }
    void restore_decode(std::size_t growth,std::optional<std::size_t> transient) {
        wait_admission();
        const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
        context->reclaim_idle_memory();
        {std::lock_guard lock(mutex_);retire_host_transfers_locked();}
        context->reclaim_idle_memory();
        const auto configured=std::getenv("TYLOINFER_CUDA_WORKSPACE_RESERVE_GIB")?
            std::optional(mfq::cuda::runtime_options::workspace_reserve_bytes()):std::nullopt;
        const auto reserve=mfq::cuda::routed_decode_workspace_reserve(growth,transient,configured);
        reserve_workspace(reserve);
        const auto available=context->available_memory();
        const auto upload=transfer_stream(mfq_current_cuda_device());
        std::unique_lock<std::mutex> guard(mutex_);
        if(replay_readers_[mfq_current_cuda_device()])
            throw MfqResourceExhausted("expert workspace cannot restore during an active decode window");
        decode_reserves_[mfq_current_cuda_device()]={growth,transient};
        // The requested cache ceiling remains unchanged. Admission follows
        // current physical headroom after KV and decode workspaces are priced.
        const auto device=mfq_current_cuda_device();
        std::size_t reusable=0;
        for(const auto& [key,entry]:device_)if(key.device==device) {
            reusable=mfq::cuda::routed_reusable_pool_bytes(entry.weight.image.data_ptr());break;
        }
        decode_limits_[device]=mfq::cuda::routed_cache_capacity(
            std::max(device_used_locked(device),device_reserved_locked(device)),reusable,available,reserve);
        const auto limit=device_limit_locked();
        while(device_used_locked(device)>limit)if(!evict_oldest_device_locked(device))break;
        while(device_reserved_locked(device)>limit)if(!evict_device_arena_locked(device))break;
        release_empty_device_arenas_locked(device);
        auto& policy=routes_[device];
        if(policy.seeded) {
            CpuTimer timer(profile_?&route_policy_ms_:nullptr);
            policy.select(limit,[&](auto source){return source_info_.at(source).experts;});
            refresh_device_priority_locked(device);
        }
        auto reserved=device_reserved_locked(device);
        const MfqCudaStreamGuard stream(upload);
        const mfq::cuda::RoutedWeightAllocationScope allocation;
        while(reserved<limit) {
            const auto bytes=std::min<std::size_t>(std::size_t(64)<<20,limit-reserved);
            try {device_arenas_[device].push_back(std::make_shared<DeviceImageArena>(bytes,device));}
            catch(const MfqResourceExhausted&){break;}
            reserved+=bytes;
        }
        guard.unlock();queue_admission(device,mfq::cuda::RoutedImageUse::Restore);wait_admission();
    }
    void resize(std::size_t bytes) {
        admission_executor_->transition([&](auto& admission) {
            join_admission(admission);
            std::lock_guard<std::mutex> guard(mutex_);
            for(const auto& [_,readers]:replay_readers_)if(readers)
                throw MfqResourceExhausted("expert cache cannot resize during an active decode window");
            retain_all_=false;
            device_limit_=bytes;
            for(const auto& [device,_]:device_arenas_)
                while(device_used_locked(device)>bytes && evict_oldest_device_locked(device)) {}
            settle_host_transfers_locked();
            for(const auto& [device,_]:device_arenas_) {
                while(device_reserved_locked(device)>bytes)
                    if(!evict_device_arena_locked(device))break;
                release_empty_device_arenas_locked(device);
            }
        });
    }
    std::size_t reclaim_host(std::size_t bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        return reclaim_host_locked(bytes);
    }
    void resize_host(std::size_t bytes) {
        std::lock_guard lock(mutex_);
        const bool ordinary=host_limit_==ordinary_host_limit_;
        ordinary_host_limit_=bytes;host_limit_=ordinary?bytes:std::min(host_limit_,bytes);
        const auto allocated=host_allocated_locked();
        if(allocated>bytes)(void)reclaim_host_locked(allocated-bytes);
    }
    std::size_t reclaim_duplicates(std::size_t bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        return reclaim_host_locked(bytes,true);
    }
    std::size_t reserve_host(std::size_t bytes) {
        std::lock_guard<std::mutex> guard(mutex_);return reserve_host_locked(bytes);
    }
    std::size_t share_host(std::size_t bytes) {
        const auto released=bytes?reserve_host(bytes):0;
        std::lock_guard guard(mutex_);
        host_limit_=mfq::cuda::routed_shared_host_limit(ordinary_host_limit_,host_used_,
            available_host_bytes(process_host_limit_),bytes);
        return released;
    }
    std::uint64_t streamed_bytes() {
        std::lock_guard<std::mutex> guard(mutex_); return transfer_bytes_;
    }
    std::array<double,2> transfer_timings() {
        std::lock_guard lock(mutex_);return {fetch_ms_,upload_submit_ms_};
    }
    std::vector<std::pair<std::string,double>> memory_metrics() {
        std::lock_guard<std::mutex> guard(mutex_);
        std::size_t covered=0,staging=0;
        std::unordered_set<std::uint64_t> covered_images;
        covered_images.reserve(device_.size()+host_.size());
        for(const auto& [key,entry]:device_)
            if(covered_images.insert(key.image).second)covered+=entry.source_bytes;
        for(const auto& [key,entry]:host_)
            if(covered_images.insert(key).second)covered+=entry.image->source_bytes;
        for(const auto& [_,streams]:stages_)for(const auto& [stream,arenas]:streams)for(const auto& stage:arenas.stages)
            if(stage)staging+=stage->host.size();
        const auto cold=payload_bytes_>covered?payload_bytes_-covered:0;
        const auto misses=calls_-gpu_hits_;
        std::vector<std::pair<std::string,double>> result{{"resident_expert_cache_bytes",double(device_used_)},
            {"expert_complete_slot_reuses",double(expert_slot_reuses_)},
            {"expert_cache_allocation_reuses",double(device_reuses_)},
            {"expert_cache_reserved_bytes",double([&]{std::size_t sum=0;for(const auto& [device,_]:device_arenas_)sum+=device_reserved_locked(device);return sum;}())},
            {"expert_cache_budget_bytes",double([&] {
                if(routes_.empty())return device_limit_locked();
                std::size_t sum=0;for(const auto& [device,_]:routes_)sum+=device_limit_locked(device);return sum;
            }())},
            {"expert_cache_configured_budget_bytes",double(retain_all_?std::max(resident_bytes_bound_,device_used_):device_limit_)},
            {"tq_predictor_resident_expert_bytes",double(predictor_resident_bytes_)},
            {"ram_expert_enabled",host_used_?1.0:0.0},
            {"ram_expert_payload_bytes",double(host_used_)},
            {"ram_expert_duplicate_bytes",double([&]{std::size_t sum=0;for(const auto key:host_duplicates_)sum+=host_.at(key).image->size();return sum;}())},
            {"ram_expert_cache_budget_bytes",double(host_limit_)},
            {"ram_expert_cache_allocated_bytes",double(host_allocated_locked())},
            {"ram_expert_pageable_payload_bytes",double(host_pageable_)},
            {"ram_expert_compacted_bytes",double(host_compacted_bytes_)},
            {"ram_expert_trimmed_pinned_bytes",double(host_trimmed_bytes_)},
            {"ram_expert_evacuated_to_vram_bytes",double(host_evacuated_bytes_)},
            {"ram_expert_evacuated_to_vram_images",double(host_evacuated_images_)},
            {"ram_expert_pinned_fragment_retries",double(pin_fragment_retries_)},
            {"ram_expert_pinned_interval_compactions",double(pin_interval_compactions_)},
            {"ram_expert_pinned_fragment_reuses",double(pin_fragment_reuses_)},
            {"ram_expert_pinned_reserved_bytes",double(host_reserved_)},
            {"ram_expert_pcie_read_bytes",double(transfer_bytes_)},
            {"ram_expert_direct_pcie_read_bytes",double(direct_transfer_bytes_)},
            {"ram_expert_cache_hits",double(host_hits_)},
            {"expert_native_dma_batches",double(native_batches_)},
            {"expert_upload_groups",double(upload_groups_)},{"expert_transfer_retire_queries",double(retirement_queries_)},
            {"ram_expert_writeback_bytes",double(writeback_bytes_)},
            {"ram_expert_unpinned_copy_bytes",double(unpinned_bytes_)},
            {"expert_admission_pending_routes",double([&]{std::uint64_t n=0;for(const auto& route:speculative_routes_)
                for(const auto& touch:route.touches)n+=3*touch.uses;return n;}())},
            {"expert_admission_observations",double([&]{std::uint64_t n=0;for(const auto& [_,policy]:routes_)n+=policy.observations;return n;}())},
            {"expert_admission_retained_images",double([&]{std::size_t n=0;for(const auto& [_,policy]:routes_)n+=policy.usage.retained_images();return n;}())},
            {"expert_admission_resident_images",double([&]{std::size_t n=0;for(const auto& [_,policy]:routes_)n+=policy.usage.retained_resident_images();return n;}())},
            {"expert_decode_calls",double(decode_calls_)},{"expert_decode_hits",double(decode_hits_)},
            {"expert_decode_pcie_read_bytes",double(decode_transfer_)},
            {"expert_admission_pcie_read_bytes",double(admission_transfer_)},
            {"expert_restore_pcie_read_bytes",double(restore_transfer_)},
            {"expert_background_admission_pcie_read_bytes",double(admission_transfer_-restore_transfer_)},
            {"expert_decode_metadata_bytes",double(decode_metadata_bytes_)},
            {"expert_decode_metadata_updates",double(decode_metadata_updates_)},
            {"expert_decode_resident_bypasses",double(decode_resident_bypasses_)},
            {"expert_decode_metadata_submit_ms",decode_metadata_submit_ms_},
            {"expert_decode_resident_touch_ms",decode_resident_touch_ms_},
            {"expert_decode_admission_wait_ms",decode_admission_wait_ms_},
            {"expert_admission_independent_lanes",double([&] {
                std::size_t count=0;
                for(const auto& [device,stream]:admission_streams_)
                    if(const auto found=transfer_streams_.find(device);found!=transfer_streams_.end() &&
                            found->second.stream()!=stream.stream())++count;
                return count;
            }())},
            {"expert_decode_source_read_bytes",double(decode_source_)},
            {"expert_decode_writeback_bytes",double(decode_writeback_)},
            {"expert_decode_host_get_rows_ms",decode_get_rows_ms_},
            {"expert_decode_host_fetch_ms",decode_fetch_ms_},
            {"expert_decode_cpu_stage_copy_ms",decode_copy_ms_},
            {"expert_decode_cpu_stage_wait_ms",decode_stage_wait_ms_},
            {"expert_decode_host_allocation_ms",decode_allocation_ms_},
            {"expert_decode_host_upload_submit_ms",decode_submit_ms_},
            {"expert_decode_host_repin_ms",decode_repin_ms_},
            {"expert_decode_arena_recycles",double(decode_arena_recycles_)},
            {"expert_decode_retained_evictions",double(decode_retained_evictions_)},
            {"expert_admission_retained_route_mass",[&]{double n=0;for(const auto& [_,p]:routes_)n+=p.usage.retained_route_mass().first;return n;}()},
            {"expert_admission_total_route_mass",[&]{double n=0;for(const auto& [_,p]:routes_)n+=p.usage.retained_route_mass().second;return n;}()},
            {"ram_expert_cache_misses",double(misses-host_hits_)},
            {"ram_expert_cache_evictions",double(host_evictions_)},
            {"ssd_expert_enabled",cold?1.0:0.0},
            {"ssd_expert_payload_bytes",double(cold)},
            {"ssd_expert_bytes_read",double(source_bytes_)},
            {"ssd_expert_read_count",double(source_reads_)},
            {"expert_source_read_batches",double(source_batches_)},
            {"expert_logical_transfer_bytes",double(transfer_bytes_)},
            {"expert_host_fetch_ms",fetch_ms_},{"expert_host_copy_ms",upload_submit_ms_},
            {"expert_batched_upload_packets",double(staged_packets_)},
            {"expert_batched_cpu_copy_ms",stage_copy_ms_},{"expert_batched_stage_wait_ms",stage_wait_ms_},
            {"expert_cache_get_rows_ms",get_rows_ms_},{"expert_allocation_ms",allocation_ms_},
            {"expert_route_policy_ms",route_policy_ms_},
            {"expert_host_register_ms",registration_ms_},{"expert_source_prepare_ms",source_prepare_ms_},
            {"expert_source_slice_ms",source_slice_ms_},{"expert_host_admission_ms",host_admission_ms_},
            {"expert_host_pack_ms",host_pack_ms_},
            {"expert_host_pin_reuse_ms",pin_reuse_ms_},{"expert_host_unpin_copy_ms",unpin_copy_ms_},
            {"expert_host_image_pack_ms",image_pack_ms_},{"expert_host_pageable_repin_ms",pageable_repin_ms_},
            {"expert_host_unpin_allocate_ms",unpin_allocate_ms_},{"expert_host_unpin_memcpy_ms",unpin_memcpy_ms_},
            {"expert_host_unpin_release_ms",unpin_release_ms_},
            {"ram_expert_pageable_capacity_bytes",double([&]{std::size_t sum=0;for(const auto& [_,entry]:host_)sum+=entry.image->pageable_capacity();return sum;}())},
            {"expert_decode_transfer_bytes",double(staging)},
            {"expert_cache_hits",double(gpu_hits_)},
            {"expert_cache_misses",double(misses)},
            {"expert_cache_evictions",double(device_evictions_)},
            {"host_tensor_bytes",double(mfq::cuda::tensor_host_bytes.load())}};
        if(profile_) {
            const auto policy_sum=[&](auto value) {
                double sum=0;for(const auto& [_,policy]:routes_)sum+=value(policy);return sum;
            };
            result.emplace_back("expert_admission_selected_bytes",policy_sum([](const auto& p){return p.usage.retained_bytes();}));
            result.emplace_back("expert_admission_measured_selected_bytes",policy_sum([](const auto& p){return p.usage.measured_retained_bytes();}));
            result.emplace_back("expert_replacement_pairs",policy_sum([](const auto& p){return p.replacement_pairs;}));
            result.emplace_back("expert_replacement_approved",policy_sum([](const auto& p){return p.replacement_approved;}));
            result.emplace_back("expert_replacement_budget_rejected",policy_sum([](const auto& p){return p.replacement_budget_rejected;}));
            result.emplace_back("expert_replacement_leased",policy_sum([](const auto& p){return p.replacement_leased;}));
            result.emplace_back("expert_replacement_resized",policy_sum([](const auto& p){return p.replacement_resized;}));
            result.emplace_back("expert_replacement_missing_slots",policy_sum([](const auto& p){return p.replacement_missing_slots;}));
            result.emplace_back("expert_replacement_heat_rejected",policy_sum([](const auto& p){return p.replacement_heat_rejected;}));
            result.emplace_back("expert_replacement_multi_victim",policy_sum([](const auto& p){return p.replacement_multi_victim;}));
            result.emplace_back("expert_admission_unretained_mass_below_one",policy_sum([](const auto& p){return p.usage.unretained_route_mass_below(1.f);}));
            result.emplace_back("expert_admission_unretained_mass_below_two",policy_sum([](const auto& p){return p.usage.unretained_route_mass_below(2.f);}));
            double ranked_mass=0;std::size_t ranked_images=0;
            double stable_mass=0;
            for(const auto& [device,policy]:routes_) {
                auto ranked=policy.usage;ranked.select(device_limit_locked(device),0,&policy.banks);
                ranked_mass+=ranked.retained_route_mass().first;ranked_images+=ranked.retained_images();
                ranked=policy.usage;ranked.select(device_limit_locked(device),mfq::cuda::expert_replacement_margin,&policy.banks);
                stable_mass+=ranked.retained_route_mass().first;
            }
            result.emplace_back("expert_admission_global_rank_route_mass",ranked_mass);
            result.emplace_back("expert_admission_global_rank_images",double(ranked_images));
            result.emplace_back("expert_admission_stable_global_rank_route_mass",stable_mass);
        }
        return result;
    }
    std::vector<std::int64_t> read_routes(const tb::Tensor& ids) {
        const auto source=ids.contiguous();
        std::vector<std::int64_t> result(source.numel());
        if(!source.numel())return result;
        if(!source.is_cuda() || (source.scalar_type()!=tb::kInt32 && source.scalar_type()!=tb::kInt64)) {
            const auto host=source.cpu().to(tb::kInt64);
            std::copy_n(host.data_ptr<std::int64_t>(),result.size(),result.data());return result;
        }
        const MfqCudaGuard device_guard(source.get_device());
        const auto stream=mfq_get_current_cuda_stream(source.get_device());
        std::shared_ptr<RouteReadback> readback;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            for(const auto& candidate:route_readbacks_)
                if(candidate->device==source.get_device() && candidate->stream.stream()==stream.stream()) {
                    readback=candidate;break;
                }
            if(!readback) {
                readback=std::make_shared<RouteReadback>(source.get_device(),stream);
                route_readbacks_.push_back(readback);
            }
        }
        // Keep a stable DMA destination per execution stream. Pageable tensor
        // allocation and GPU integer conversion are unnecessary for this ID list.
        std::lock_guard<std::mutex> guard(readback->mutex);
        const auto required=source.nbytes()+64;
        if(!readback->host || readback->host->size()<required) {
            const auto bytes=readback->host && readback->host->size()<=SIZE_MAX/2?
                std::max(required,readback->host->size()*2):required;
            readback->host=std::make_unique<mfq::cuda::HostBuffer>(bytes,readback->can_map);
            readback->mapped=nullptr;
            if(readback->can_map)TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&readback->mapped,readback->host->data(),0));
        }
        auto* values=static_cast<std::uint8_t*>(readback->host->data())+64;
        if(readback->mapped && source.scalar_type()==tb::kInt32 && source.numel()<=INT_MAX) {
            auto* ready=static_cast<std::uint32_t*>(readback->host->data());
            *static_cast<volatile std::uint32_t*>(ready)=0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            // Reuse the format-independent route publication already used by
            // DecodeWindow. The GPU publishes IDs plus a system-visible flag;
            // no copy-engine transfer or stream-wide synchronization is needed.
            mfq::cuda::moe_publish_routes(nullptr,source.data_ptr<std::int32_t>(),nullptr,
                reinterpret_cast<std::int32_t*>(readback->mapped)+16,
                static_cast<std::uint32_t*>(readback->mapped),0,static_cast<int>(source.numel()),stream.stream());
            TYLOINFER_CUDA_CHECK(cudaGetLastError());
            mfq::cuda::wait_route_publication(ready,stream.stream());
        }else {
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(values,source.data_ptr(),source.nbytes(),cudaMemcpyDeviceToHost,stream.stream()));
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        }
        if(source.scalar_type()==tb::kInt32) {
            std::copy_n(reinterpret_cast<const std::int32_t*>(values),result.size(),result.data());
        }else std::copy_n(reinterpret_cast<const std::int64_t*>(values),result.size(),result.data());
        return result;
    }
    tb::Tensor projection_table(std::uint64_t source,std::int64_t experts,
            const std::int64_t* ids,const std::vector<TqWeight>& weights,int device,cudaStream_t consumer=nullptr) {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto stream=consumer?consumer:mfq_current_cuda_stream();
        if(const auto& permanent=source_info_.at(source).resident) {
            if(permanent->packed.get_device()!=device || permanent->experts.size()!=std::size_t(experts))
                throw std::invalid_argument("TQ permanent projection table mismatch");
            permanent->packed.record_stream(reinterpret_cast<std::uintptr_t>(stream));
            permanent->views.record_stream(reinterpret_cast<std::uintptr_t>(stream));
            return permanent->views;
        }
        auto& tables=view_tables_[source];
        auto found=std::find_if(tables.begin(),tables.end(),[&](const ViewTable& table) {
            return table.device==device && table.stream==stream;
        });
        if(found==tables.end()) {
            auto views=tb::empty({experts*static_cast<std::int64_t>(sizeof(TrellisProjectionView))},
                tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
            tables.push_back({device,stream,std::move(views),std::vector<std::uintptr_t>(experts),
                std::vector<std::uint8_t>(experts*sizeof(TrellisProjectionView))});
            found=std::prev(tables.end());
        }
        // Each stream has independent metadata, so a concurrent upload cannot
        // overwrite a view another request is reading. Tables own no expert
        // images; the caller's selected weights retain their normal lifetime.
        bool changed=false;
        for(std::size_t i=0;i<weights.size();++i) {
            const auto image=reinterpret_cast<std::uintptr_t>(weights[i].image.data_ptr());
            if(found->images[ids[i]]==image)continue;
            std::memcpy(found->bytes.data()+ids[i]*sizeof(TrellisProjectionView),&weights[i].view,sizeof(TrellisProjectionView));
            found->images[ids[i]]=image;changed=true;
        }
        // One small DMA for changed metadata, no per-expert transfers and no
        // host synchronization. Resident hits submit no metadata work at all.
        if(changed)try {
            upload(found->views,found->bytes);
            decode_metadata_bytes_+=found->bytes.size();++decode_metadata_updates_;
        }
        catch(...) {std::fill(found->images.begin(),found->images.end(),0);throw;}
        return found->views;
    }
    std::vector<TqWeight> get_rows(std::uint64_t source,const std::int64_t* ids,std::size_t count,
            const TqRowTable& table,std::int64_t out,int device,DeviceImageArena* workspace=nullptr,
            mfq::cuda::RoutedImageUse use=mfq::cuda::RoutedImageUse::Eager,
            UploadBatch* batch=nullptr,TqPrefillWeightStage* prefill=nullptr) {
        const bool decode_replay=use==mfq::cuda::RoutedImageUse::Decode;
        const MfqCudaGuard device_guard(device);
        const auto consumer=mfq_get_current_cuda_stream(device);
        const auto stream=use==mfq::cuda::RoutedImageUse::Admission || use==mfq::cuda::RoutedImageUse::Restore?
            admission_stream(device):transfer_stream(device);
        const MfqCudaStreamGuard upload_guard(stream);
        std::lock_guard<std::mutex> guard(mutex_);
        const mfq::cuda::RoutedHostEvacuationBlock block_evacuation(host_evacuation_blocked_);
        if(const auto& permanent=source_info_.at(source).resident) {
            if(permanent->packed.get_device()!=device)
                throw std::invalid_argument("TQ permanent resident expert device mismatch");
            std::vector<TqWeight> result;result.reserve(count);
            for(std::size_t index=0;index<count;++index) {
                if(ids[index]<0 || std::uint64_t(ids[index])>=permanent->experts.size())
                    throw std::out_of_range("TQ routed expert ID is out of bounds");
                result.push_back(permanent->experts[ids[index]]);
            }
            permanent->packed.record_stream(reinterpret_cast<std::uintptr_t>(consumer.stream()));
            return result;
        }
        const std::array<std::uint64_t,5> before{calls_,gpu_hits_,transfer_bytes_,source_bytes_,writeback_bytes_};
        const auto decode_started=profile_ && decode_replay?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        const std::array<double,6> cpu_before{fetch_ms_,stage_copy_ms_,stage_wait_ms_,allocation_ms_,upload_submit_ms_,pageable_repin_ms_};
        const auto arena_before=arena_recycles_,retained_before=retained_evictions_;
        CpuTimer total_timer(profile_?&get_rows_ms_:nullptr);
        const auto resources=tq_gemm_resources(device);
        gemm_.try_emplace(device,resources);
        std::vector<TqWeight> result;result.reserve(count);
        std::vector<mfq::cuda::DmaCopy> copies;copies.reserve(count);
        std::vector<CUdeviceptr> aliases;aliases.reserve(count);
        std::array<Stage*,2> queued_stages{};
        std::size_t queued_stage_count=0;
        std::vector<DeviceKey> admitted;admitted.reserve(count);
        std::shared_ptr<std::atomic<bool>> submitted=batch?batch->submitted:nullptr;
        std::shared_ptr<std::atomic<bool>> recorded=batch?batch->recorded:nullptr;
        auto position=batch?batch->position:std::shared_ptr<mfq::cuda::RoutedCompletionPosition>{};
        const auto submit_copies=[&] {
            if(copies.empty())return;
            // Prepare the following bank while this bank's DMA is in flight.
            std::size_t index=0;
            try {
                // Mark possible DMA before submitting: a partial failure must
                // retain its CPU sources until the stream has been joined.
                if(submitted)submitted->store(true,std::memory_order_release);
                if(mfq::cuda::submit_routed_transfers(copies,stream.stream(),mfq::cuda::routed_dma_mode(),
                        [&](const void*,std::size_t){return aliases[index++];}))++native_batches_;
                for(std::size_t i=0;i<queued_stage_count;++i) {
                    auto* stage=queued_stages[i];
                    stage->uploaded.record(stream.stream());stage->pending=true;
                }
                if(queued_stage_count)++staged_packets_;
            }catch(...) {
                copies.clear();aliases.clear();queued_stage_count=0;
                (void)cudaStreamSynchronize(stream.stream());throw;
            }
            copies.clear();aliases.clear();queued_stage_count=0;
        };
        const char* trace_source=std::getenv("TYLOINFER_TQ_TRACE_ROWS");
        const bool trace=trace_source && (std::strcmp(trace_source,"all")==0 ||
            (decode_replay && std::strcmp(trace_source,"decode")==0) || std::strtoull(trace_source,nullptr,10)==source);
        const auto trace_step=[&](const char* step,std::uint32_t expert=0) {
            if(trace){std::fprintf(stderr,"tq_rows source=%llu expert=%u step=%s\n",
                static_cast<unsigned long long>(source),expert,step);std::fflush(stderr);}
        };
        trace_step("entered");
        std::shared_ptr<mfq::cuda::Event> uploaded=batch?batch->ready:nullptr;
        retire_host_transfers_locked();trace_step("retired");
        std::vector<std::int64_t> missing;
        std::unordered_map<std::uint32_t,std::size_t> source_images;
        for(std::size_t index=0;index<count;++index) {
            if(ids[index]<0 || std::uint64_t(ids[index])>=table.rows()/std::uint64_t(out))
                throw std::out_of_range("TQ routed expert ID is out of bounds");
            const auto expert=static_cast<std::uint32_t>(ids[index]);
            const auto key=(source<<32)|expert;
            if(!device_.count({key,device}) && !host_.count(key) && !staged_host_jobs_.count(key)) {
                const auto [_,inserted]=source_images.try_emplace(expert,missing.size());
                if(inserted)missing.push_back(expert);
            }
        }
        std::size_t source_bound=0,largest_source=0;
        for(const auto expert:missing) {
            const auto bytes=table.expert_image_bytes(std::uint64_t(expert),out);
            source_bound=mfq::cuda::routed_saturating_add(source_bound,bytes);largest_source=std::max(largest_source,bytes);
        }
        const auto source_peak=mfq::cuda::routed_saturating_add(source_bound,largest_source);
        const bool batch_source=missing.size()>1 && source_peak<=available_host_bytes(process_host_limit_);
        auto prefetched=batch_source?table.slice_experts(missing,std::uint64_t(out)):
            std::vector<std::vector<std::uint8_t>>{};
        std::vector<std::shared_ptr<HostImage>> prepared_images(missing.size());
        if(!prefetched.empty())++source_batches_;
        auto transfer=host_transfers_.end();
        try {
        for(std::size_t index=0;index<count;++index) {
        const auto expert=static_cast<std::uint32_t>(ids[index]);trace_step("lookup",expert);
        const auto key=(source<<32)|expert;
        const DeviceKey device_key{key,device};
        const auto staged=staged_host_jobs_.find(key);
        ++calls_;
        const auto resident=device_.find(device_key);
        if(resident!=device_.end()) {
            if(staged!=staged_host_jobs_.end()) {
                prefill_host_stager_->skip(staged->second);prefill_host_stager_->pass(staged->second+1);
            }
            device_recent_.splice(device_recent_.begin(),device_recent_,resident->second.recent);
            if(resident->second.cold) {
                auto& cold=routes_.at(device).cold;cold.splice(cold.begin(),cold,*resident->second.cold);
            }
            ++gpu_hits_;result.push_back(resident->second.weight);
            if((calls_&4095)==0)report();
            continue;
        }
        trace_step("fetch",expert);
        const auto fetch_started=std::chrono::steady_clock::now();
        std::shared_ptr<HostImage> image;
        const auto cached=host_.find(key);
        if(staged!=staged_host_jobs_.end()) {
            // The prepared copy owns these exact bytes. Repinning would free
            // a pageable source while its background memcpy is still active.
            image=std::static_pointer_cast<HostImage>(prefill_host_stager_->jobs[staged->second].owner);
            ++host_hits_;if(cached!=host_.end())mark_host_duplicate_locked(key);
        }else if(cached!=host_.end()) {
            host_recent_.splice(host_recent_.begin(),host_recent_,cached->second.recent);
            image=cached->second.image;++host_hits_;mark_host_duplicate_locked(key);
            if(resources->host_register() && !image->pinned()) {
                CpuTimer timer(profile_?&pageable_repin_ms_:nullptr);
                image->pin(reuse_free_pinned(image->size()));
                if(image->pinned())mark_host_duplicate_locked(key);
            }
        } else if(const auto prepared=source_images.find(expert);
                prepared!=source_images.end() && prepared_images[prepared->second]) {
            image=prepared_images[prepared->second];++host_hits_;
        } else {
            CpuTimer source_timer(profile_?&source_prepare_ms_:nullptr);
            auto wire=[&] {
                CpuTimer timer(profile_?&source_slice_ms_:nullptr);
                // Earlier admissions can evict a hit that was absent from the initial miss batch.
                if(const auto prepared=source_images.find(expert);
                        !prefetched.empty() && prepared!=source_images.end())
                    return std::move(prefetched.at(prepared->second));
                return table.slice_rows(std::uint64_t(expert)*out,std::uint64_t(expert+1)*out);
            }();
            // Only source misses allocate new host images. Reclaim optional
            // images against current RAM/commit headroom before packing one;
            // GPU hits and ordinary RAM streaming do not poll process memory.
            const bool admit=[&] {
                CpuTimer timer(profile_?&host_admission_ms_:nullptr);
                reserve_host_locked(wire.size());
                return available_host_bytes(process_host_limit_)>=wire.size();
            }();
            source_bytes_+=wire.size();++source_reads_;
            {
                CpuTimer timer(profile_?&host_pack_ms_:nullptr);
                const mfq::TqLayout layout(wire.data(),wire.size());
                const auto bytes=((layout.packed+3)&~std::size_t(3))+layout.packed_bytes+32;
                if(admit)admit_host_space_locked(bytes);
                auto block=admit && resources->host_register()?allocate_pinned(bytes):nullptr;
                CpuTimer image_timer(profile_?&image_pack_ms_:nullptr);
                image=HostImage::create(std::move(wire),std::move(block));
            }
            if(const auto prepared=source_images.find(expert);prepared!=source_images.end())
                prepared_images[prepared->second]=image;
            if(admit && image->size()<=host_limit_) {
                insert_host_locked(key,image);
            }
        }
        if(image->ready && image->producer!=stream.stream())
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),image->ready->get(),0));
        fetch_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fetch_started).count();
        trace_step("allocate",expert);
        const auto needed=(image->size()+15)&~std::size_t(15);
        auto& policy=routes_[device];policy.usage.image(key,needed);
        // Captured consumers use an immutable resident epoch. Cache admission
        // runs after the complete window, on the common admission executor.
        // Prompt misses belong to the bounded, fenced prefill stages. As in
        // GGUF, only reuse existing residents here; admission follows prefill.
        bool cacheable=!prefill && !replay_readers_[device] && (!policy.ready || policy.usage.retained(key));
        DeviceImageArena* owning_arena=nullptr;
        std::optional<std::uint64_t> expert_group;
        auto weight=[&] {
        CpuTimer allocation_timer(profile_?&allocation_ms_:nullptr);
        const auto limit=device_limit_locked();
        cacheable=cacheable && needed<=limit;
        tb::Tensor storage;
        if(cacheable)if(const auto* banks=policy.banks.find(source)) {
            const DeviceKey group{((*banks)[0]<<32)|expert,device};
            auto slot=expert_slots_.find(group);
            if(slot==expert_slots_.end() && policy.ready) {
                std::array<std::size_t,3> sizes{};
                bool partial=false;
                for(unsigned role=0;role<3;++role) {
                    const auto& info=source_info_.at((*banks)[role]);
                    sizes[role]=info.table->expert_image_bytes(expert,info.rows);
                    partial|=device_.count(DeviceKey{((*banks)[role]<<32)|expert,device})!=0;
                }
                if(!partial) {
                    const mfq::cuda::RoutedExpertImageLayout layout(*banks,sizes);
                    if(layout.bytes>limit)cacheable=false;
                    while(cacheable && layout.bytes>limit-std::min(limit,device_used_locked(device)))
                        if(!evict_oldest_device_locked(device))break;
                    if(cacheable) {
                        const mfq::cuda::RoutedWeightAllocationScope allocation;
                        auto image=allocate_device_image(layout.bytes,device,limit,cacheable,owning_arena,nullptr,false);
                        if(image.defined()) {
                            slot=expert_slots_.emplace(group,ExpertSlot{std::move(image),owning_arena,layout}).first;
                            pending_expert_bytes_[device]+=layout.bytes;
                        }
                    }
                }
            }
            if(slot!=expert_slots_.end()) {
                const auto role=std::find(banks->begin(),banks->end(),source)-banks->begin();
                if(needed!=slot->second.layout.sizes[role])throw std::logic_error("TQ expert slot byte count differs");
                storage=slot->second.image.narrow(0,slot->second.layout.offsets[role],needed);
                owning_arena=slot->second.arena;expert_group=group.image;
            }
        }
        // Return the old interval to the arena, which splits the next image
        // to its actual size. Reusing an oversized tensor wholesale steadily
        // inflates cache entries and requires a linear search for each miss.
        while(!storage.defined() && cacheable && needed>limit-std::min(limit,device_used_locked(device)))
            if(!evict_oldest_device_locked(device))break;
        trace_step("evicted",expert);
        const mfq::cuda::RoutedWeightAllocationScope allocation;
        if(!storage.defined())storage=!cacheable && prefill?prefill->allocate(needed):
            allocate_device_image(needed,device,limit,cacheable,owning_arena,workspace);
        trace_step("allocated",expert);
        auto result=tq_view(std::move(storage),image->layout,table.width());
        result.allocation_stream=stream;return result;
        }();
        trace_step("upload",expert);
        if(staged!=staged_host_jobs_.end()) {
            submit_copies();
            const std::uint8_t* prepared;
            {CpuTimer timer(profile_?&stage_wait_ms_:nullptr);prepared=prefill_host_stager_->wait(staged->second);}
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(weight.image.data_ptr(),prepared,image->size(),
                cudaMemcpyHostToDevice,stream.stream()));
            prefill_host_stager_->issued_one(staged->second,stream.stream());
            transfer_bytes_+=image->size();
        }else if(image->pinned()) {
            trace_step("direct_event",expert);
            if(!uploaded) {
                uploaded=completion_events_.at(device,stream.stream());
                if(batch)batch->ready=uploaded;
            }
            if(transfer==host_transfers_.end()) {
                if(!submitted)submitted=std::make_shared<std::atomic<bool>>(false);
                if(!recorded)recorded=std::make_shared<std::atomic<bool>>(false);
                if(!position)position=std::make_shared<mfq::cuda::RoutedCompletionPosition>();
                host_transfers_.emplace_back();transfer=std::prev(host_transfers_.end());
                transfer->producer=stream;transfer->ready=uploaded;
                transfer->submitted=submitted;transfer->recorded=recorded;
                transfer->position=position;
            }
            // A host-cache eviction must not unpin a source still consumed
            // by asynchronous DMA. One event retains the whole call's images.
            transfer->images.push_back(image);transfer->destinations.push_back(weight.image);
            const auto started=std::chrono::steady_clock::now();
            trace_step("direct_copy",expert);
            copies.push_back({weight.image.data_ptr(),image->data(),image->size()});
            aliases.push_back(reinterpret_cast<CUdeviceptr>(image->mapped_data(device)));
            transfer_bytes_+=image->size();direct_transfer_bytes_+=image->size();
            upload_submit_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        }else {
            trace_step("staged_copy",expert);
            stage_upload(weight.image,*image,
                [&](const mfq::cuda::DmaCopy& copy,CUdeviceptr alias,Stage* stage) {
                    copies.push_back(copy);aliases.push_back(alias);
                    const auto end=queued_stages.begin()+queued_stage_count;
                    if(std::find(queued_stages.begin(),end,stage)==end) {
                        if(queued_stage_count==queued_stages.size())
                            throw std::logic_error("TQ DMA packet exceeded its staging ring");
                        queued_stages[queued_stage_count++]=stage;
                    }
                },submit_copies);
        }
        trace_step("admit",expert);
        if(cacheable && weight.image.numel()<=device_limit_locked()) {
            commit_device_image_locked(device_key,weight,*image,owning_arena,expert_group);
            admitted.push_back(device_key);if(batch)batch->admitted.push_back(device_key);
        }
        result.push_back(std::move(weight));
        if((calls_&4095)==0)report();
        }
        submit_copies();
        }catch(...) {
            const auto failure=std::current_exception();
            if(!batch)try {
                submit_copies();
                if(uploaded) {
                    uploaded->record(stream.stream());
                    if(position)*position=mfq::cuda::record_routed_completion(transfer_clock_,stream.device(),stream.stream());
                    if(recorded)recorded->store(true,std::memory_order_release);
                }
            }catch(...) {
                if(cudaStreamSynchronize(stream.stream())==cudaSuccess && transfer!=host_transfers_.end())
                    transfer->completed=true;
            }
            // A partially submitted batch cannot become a later GPU hit.
            // Its immutable RAM source remains available for a clean retry.
            for(const auto& key:admitted)if(device_.count(key))erase_device_locked(key,false);
            for(auto slot=expert_slots_.begin();slot!=expert_slots_.end();) {
                if(slot->first.device!=device || slot->second.residents){++slot;continue;}
                pending_expert_bytes_[device]-=slot->second.image.nbytes();slot=expert_slots_.erase(slot);
            }
            std::rethrow_exception(failure);
        }
        // Keep allocation, upload and eventual free on one independent stream
        // for both prefill and decode. A compute-stream free queued behind a
        // waiting decode graph cannot supply the next expert upload.
        trace_step("record_ready");
        try {
            if(batch) {
                if(consumer.stream()!=batch->consumer.stream())throw std::logic_error("TQ upload batch changed its consumer stream");
            }else if(consumer.stream()!=stream.stream()) {
                if(!uploaded)uploaded=completion_events_.at(device,stream.stream());
                uploaded->record(stream.stream());
                TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(consumer.stream(),uploaded->get(),0));
            }else if(uploaded)uploaded->record(stream.stream());
            if(!batch && uploaded && recorded) {
                if(position)*position=mfq::cuda::record_routed_completion(transfer_clock_,stream.device(),stream.stream());
                recorded->store(true,std::memory_order_release);
            }
        }catch(...) {
            if(!batch && cudaStreamSynchronize(stream.stream())==cudaSuccess && transfer!=host_transfers_.end())
                transfer->completed=true;
            throw;
        }
        if(use==mfq::cuda::RoutedImageUse::Admission || use==mfq::cuda::RoutedImageUse::Restore)
            admission_transfer_+=transfer_bytes_-before[2];
        if(use==mfq::cuda::RoutedImageUse::Restore)restore_transfer_+=transfer_bytes_-before[2];
        if(decode_replay) {
            decode_calls_+=calls_-before[0];decode_hits_+=gpu_hits_-before[1];
            decode_transfer_+=transfer_bytes_-before[2];decode_source_+=source_bytes_-before[3];
            decode_writeback_+=writeback_bytes_-before[4];
            decode_arena_recycles_+=arena_recycles_-arena_before;
            decode_retained_evictions_+=retained_evictions_-retained_before;
            if(profile_) {
                decode_get_rows_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-decode_started).count();
                decode_fetch_ms_+=fetch_ms_-cpu_before[0];decode_copy_ms_+=stage_copy_ms_-cpu_before[1];
                decode_stage_wait_ms_+=stage_wait_ms_-cpu_before[2];decode_allocation_ms_+=allocation_ms_-cpu_before[3];
                decode_submit_ms_+=upload_submit_ms_-cpu_before[4];decode_repin_ms_+=pageable_repin_ms_-cpu_before[5];
            }
        }
        trace_step("return");
        return result;
    }
};

std::size_t tq_expert_source_bytes(const std::shared_ptr<TqExpertCache>& cache) {
    return cache?cache->source_bytes():0;
}
std::size_t tq_expert_cache_budget_bytes(const std::shared_ptr<TqExpertCache>& cache) {
    return cache?cache->budget_bytes():0;
}
void finalize_tq_expert_cache(const std::shared_ptr<TqExpertCache>& cache,std::size_t growth) {
    if(cache)cache->finalize(growth);
}
void resize_tq_expert_cache(const std::shared_ptr<TqExpertCache>& cache,std::size_t bytes) {
    if(cache)cache->resize(bytes);
}
std::size_t tq_prefill_lendable_bytes(const std::shared_ptr<TqExpertCache>& cache) {
    return cache?cache->lendable_bytes():0;
}
void reserve_tq_expert_workspace(const std::shared_ptr<TqExpertCache>& cache,std::size_t bytes) {
    if(cache)cache->reserve_workspace(bytes);
}
void constrain_tq_capture_workspace(const std::shared_ptr<TqExpertCache>& cache,std::size_t bytes) {
    if(cache)cache->constrain_capture(bytes);
}
void restore_tq_decode_workspace(const std::shared_ptr<TqExpertCache>& cache,std::size_t growth,
        std::optional<std::size_t> transient) {
    if(cache)cache->restore_decode(growth,transient);
}
void restore_tq_decode_workspace_after_capture(const std::shared_ptr<TqExpertCache>& cache) {
    if(cache)cache->restore_after_capture();
}
bool set_tq_reuse_learning(const std::shared_ptr<TqExpertCache>& cache,bool enabled) {
    return cache?cache->set_reuse_learning(enabled):false;
}
void begin_tq_speculative_routes(const std::shared_ptr<TqExpertCache>& cache,int rows) {
    if(cache)cache->begin_routes(rows);
}
void commit_tq_speculative_routes(const std::shared_ptr<TqExpertCache>& cache,int rows) {
    if(cache)cache->commit_routes(rows);
}
void discard_tq_speculative_routes(const std::shared_ptr<TqExpertCache>& cache) noexcept {
    if(cache)cache->discard_routes();
}
TqPrefillWorkspaceLoan::~TqPrefillWorkspaceLoan() {
    try {finish();return;}
    catch(const std::exception& error) {std::fprintf(stderr,"TQ prefill restore failed: %s\n",error.what());}
    catch(...) {std::fprintf(stderr,"TQ prefill restore failed\n");}
    if(token_)cache_->abandon_workspace(token_);
    token_=0;
}
std::size_t TqPrefillWorkspaceLoan::begin(std::size_t bytes) {
    finish();
    if(!cache_)return 0;
    const auto reservation=cache_->begin_workspace(bytes);
    token_=reservation.first;return reservation.second;
}
void TqPrefillWorkspaceLoan::suspend() {if(token_)cache_->suspend_workspace(token_);}
void TqPrefillWorkspaceLoan::finish() {
    if(token_)cache_->end_workspace(token_);
    token_=0;
}
std::size_t reclaim_tq_expert_host(const std::shared_ptr<TqExpertCache>& cache,std::size_t bytes) {
    return cache?cache->reclaim_host(bytes):0;
}
void resize_tq_expert_host(const std::shared_ptr<TqExpertCache>& cache,std::size_t bytes) {
    if(cache)cache->resize_host(bytes);
}
std::size_t reclaim_tq_expert_duplicates(const std::shared_ptr<TqExpertCache>& cache,std::size_t bytes) {
    return cache?cache->reclaim_duplicates(bytes):0;
}
std::size_t reserve_tq_expert_host(const std::shared_ptr<TqExpertCache>& cache,std::size_t bytes) {
    return cache?cache->reserve_host(bytes):0;
}
std::size_t share_tq_expert_host(const std::shared_ptr<TqExpertCache>& cache,std::size_t bytes) {
    return cache?cache->share_host(bytes):0;
}
std::vector<std::pair<std::string,double>> tq_expert_memory_metrics(const std::shared_ptr<TqExpertCache>& cache) {
    return cache?cache->memory_metrics():std::vector<std::pair<std::string,double>>{};
}
std::uint64_t tq_expert_streamed_bytes(const std::shared_ptr<TqExpertCache>& cache) {
    return cache?cache->streamed_bytes():0;
}
std::array<double,2> tq_expert_transfer_timings(const std::shared_ptr<TqExpertCache>& cache) {
    return cache?cache->transfer_timings():std::array<double,2>{};
}

namespace {
std::shared_ptr<TqResidentBank> load_tq_resident_bank(const TqRowTable& table,
        std::int64_t experts,std::int64_t out,std::int64_t input,int device) {
    const MfqCudaGuard guard(device);
    const mfq::cuda::WeightAllocationScope weights;
    const mfq::cuda::ResidentAllocationScope permanent;
    auto bank=std::make_shared<TqResidentBank>();
    bank->gemm=tq_gemm_resources(device);
    std::vector<std::size_t> offsets;offsets.reserve(std::size_t(experts)+1);offsets.push_back(0);
    for(std::int64_t expert=0;expert<experts;++expert) {
        const auto bytes=table.expert_image_bytes(std::uint64_t(expert),std::uint64_t(out));
        if(bytes>std::size_t(INT64_MAX)-15 || offsets.back()>std::size_t(INT64_MAX)-((bytes+15)&~std::size_t(15)))
            throw std::overflow_error("TQ permanent expert bank exceeds tensor size limits");
        offsets.push_back(offsets.back()+((bytes+15)&~std::size_t(15)));
    }
    bank->packed=tb::empty({std::int64_t(offsets.back())},
        tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
    bank->experts.reserve(std::size_t(experts));
    std::vector<TrellisProjectionView> views;views.reserve(std::size_t(experts));
    const auto stream=mfq_current_cuda_stream();
    for(std::int64_t expert=0;expert<experts;++expert) {
        // Read one expert at a time. Only its immutable encoded image enters
        // the resident allocation; no decoded weights or RAM cache copy remain.
        const auto wire=table.slice_rows(std::uint64_t(expert)*out,std::uint64_t(expert+1)*out);
        const mfq::TqLayout layout(wire.data(),wire.size());
        const auto image=tq_image(wire,layout);
        if(image.size()>offsets[expert+1]-offsets[expert])
            throw std::overflow_error("TQ permanent expert image exceeds its interval");
        auto storage=bank->packed.narrow(0,std::int64_t(offsets[expert]),std::int64_t(image.size()));
        // Synchronize each bounded upload before its temporary CPU owner leaves.
        constexpr std::size_t chunk_bytes=4ULL<<20;
        for(std::size_t offset=0;offset<image.size();) {
            const auto bytes=std::min(chunk_bytes,image.size()-offset);
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(storage.data_ptr<std::uint8_t>()+offset,
                image.data()+offset,bytes,cudaMemcpyHostToDevice,stream));
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));offset+=bytes;
        }
        bank->experts.push_back(tq_view(std::move(storage),layout,input));
        views.push_back(bank->experts.back().view);
    }
    if(views.size()>std::size_t(INT64_MAX)/sizeof(TrellisProjectionView))
        throw std::overflow_error("TQ permanent expert descriptor table exceeds tensor size limits");
    bank->views=tb::from_blob(views.data(),{std::int64_t(views.size()*sizeof(TrellisProjectionView))},
        tb::TensorOptions().dtype(tb::kUInt8)).clone().to(bank->packed.device(),false,false).contiguous();
    return bank;
}
}

TqRoutedWeight::TqRoutedWeight(CudaExecutionContext& execution,const mfq::ModelSource& source,
        const std::string& name,std::int64_t count,std::int64_t outputs,std::int64_t columns)
    :table(std::make_shared<TqRowTable>(source,name)),experts(count),out(outputs),input(columns) {
    if(experts<1 || experts>UINT32_MAX || out<1 || input<1 ||
       table->rows()!=std::uint64_t(experts)*out || table->width()!=input)
        throw std::invalid_argument("TQ routed projection shape differs: "+name);
    table->cache_row_metadata();
    if(name.starts_with("predictor.")) {
        resident=load_tq_resident_bank(*table,experts,out,input,mfq_current_cuda_device());
        std::fprintf(stderr,"native_tq_predictor_resident actual_match=1 bank=%s bytes=%zu device=%d\n",
            name.c_str(),resident->packed.nbytes(),resident->packed.get_device());
    }
    if(!execution.tq_expert_cache) {
        const auto budget = execution.config.moe_device_cache_bytes.value_or(
            moe_expert_cache_budget_bytes(execution.moe_expert_cache));
        const bool explicit_budget = execution.config.moe_device_cache_bytes.has_value() ||
            bool(execution.moe_expert_cache);
        execution.tq_expert_cache=std::make_shared<TqExpertCache>(
            budget,execution.config.moe_host_cache_bytes,
            execution.config.moe_host_physical_bytes,!explicit_budget);
    }
    cache=execution.tq_expert_cache;
    source_id=cache->register_source(table,out,experts,resident);
}
std::shared_ptr<TqRoutedWeight> load_tq_routed_weight(CudaExecutionContext& execution,
        const mfq::ModelSource& source,const std::string& name) {
    const auto shape=logical_shape(source,name);
    if(shape.size()!=3)throw std::invalid_argument("TQ expert bank must be [experts,out,in]: "+name);
    return std::make_shared<TqRoutedWeight>(execution,source,name,shape[0],shape[1],shape[2]);
}
tb::Tensor TqRoutedWeight::forward(tb::Tensor input_value,tb::Tensor ids)const {
    if(ids.dim()!=2 || ids.size(1)<1 || ids.size(1)>65535 || input_value.size(-1)!=input ||
       !input_value.is_cuda() || (input_value.dim()!=2 && input_value.dim()!=3))
        throw std::invalid_argument("invalid TQ routed activation geometry");
    const auto tokens=ids.size(0),routes=ids.size(1);
    if(input_value.size(0)!=tokens || (input_value.dim()==3 && input_value.size(1)!=routes))
        throw std::invalid_argument("TQ routed token/route count differs");
    const auto source=input_value.contiguous();
    auto result=tb::empty({tokens,routes,out},source.options());
    if(!tokens)return result;
    const MfqCudaGuard guard(source.get_device());
    if(!resident)cache->wait_ready();
    auto host=ids.contiguous().cpu().to(tb::kInt64);
    const auto* selected=host.data_ptr<std::int64_t>();
    std::vector<TrellisProjectionView> views;views.reserve(host.numel());
    for(std::int64_t i=0;i<host.numel();++i) {
        if(selected[i]<0 || selected[i]>=experts)throw std::out_of_range("TQ routed expert ID is out of bounds");
    }
    const auto owners=cache->get_rows(source_id,selected,host.numel(),*table,out,source.get_device());
    const TqExpertReadFence reads({&owners},source.get_device());
    for(const auto& owner:owners)views.push_back(owner.view);
    const auto bytes=static_cast<std::int64_t>(views.size()*sizeof(TrellisProjectionView));
    const auto metadata=tb::from_blob(views.data(),{bytes},tb::TensorOptions().dtype(tb::kUInt8))
        .clone().to(source.device(),false,false).contiguous();
    const auto* data=reinterpret_cast<const TrellisProjectionView*>(metadata.data_ptr());
    const auto element=source.element_size();
    for(std::int64_t token=0;token<tokens;++token) {
        const auto input_offset=token*(source.dim()==3?routes:1)*input*element;
        const auto output_offset=token*routes*out*element;
        status(mfq_tq_gemv_batch(data+token*routes,static_cast<const std::uint8_t*>(source.data_ptr())+input_offset,
            static_cast<std::uint8_t*>(result.data_ptr())+output_offset,static_cast<int>(routes),static_cast<int>(out),
            source.dim()==3?static_cast<int>(input):0,static_cast<int>(out),dtype(source.scalar_type()),
            0,0,mfq_current_cuda_stream()));
    }
    return result;
}

bool prepare_tq_graph_pool(const std::shared_ptr<TqExpertCache>& cache) {
    if(!cache)return false;
    cache->prepare_graph_pool();return true;
}
void wait_tq_expert_admission(const std::shared_ptr<TqExpertCache>& cache) {
    if(cache)cache->wait_ready();
}
bool tq_graph_supports_tokens(std::int64_t tokens)noexcept {
#if CUDART_VERSION >= 11060
    return tokens>0;
#else
    return tokens==1;
#endif
}

namespace {
using TqIndexedTiles=std::array<std::vector<TrellisIndexedProjectionJob>,4>;
using TqWideTiles=std::array<std::vector<TrellisIndexedWideProjectionJob>,4>;
struct TqIndexedJobs {
    TqIndexedTiles gu,down;
    TqWideTiles wide;
    std::vector<TrellisIndexedWidePairJob> wide_gu;
};
TqIndexedJobs tq_indexed_jobs(const std::vector<TqWeight>& gate,
        const std::vector<TqWeight>& up,const std::vector<TqWeight>& down,
        const std::vector<std::vector<std::int64_t>>& slots,std::size_t begin,
        std::int64_t routes,std::int64_t input,std::int64_t hidden,std::int64_t output,bool wide=false,
        bool make_gu=true,bool make_down=true) {
    TqIndexedTiles gu_tiles,down_tiles;
    TqWideTiles wide_tiles;
    std::vector<TrellisIndexedWidePairJob> wide_gu;
    for(std::size_t i=0;i<gate.size();++i) {
        const auto& rows=slots[begin+i];
        const auto complete=wide?rows.size()/8*8:0;
        for(std::size_t start=0;make_gu && start<complete;start+=8) {
            TrellisIndexedWidePairJob job{};job.weight=gate[i].view;job.second=up[i].view;job.count=8;
            for(int t=0;t<8;++t) {
                const auto slot=rows[start+t];job.input_offsets[t]=(slot/routes)*input;job.output_offsets[t]=slot*hidden;
            }
            wide_gu.push_back(job);
        }
        // Eight columns fill the MMA instruction. Keep incomplete groups in
        // the short SIMT kernel instead of repeating padded MMA work.
        auto chunks=(rows.size()-complete+3)/4;
        for(std::size_t start=complete;(make_gu || (make_down && !wide)) && start<rows.size();--chunks) {
            const auto tile=(rows.size()-start+chunks-1)/chunks;
            TrellisIndexedProjectionJob gu{},dn{};
            if(make_gu){gu.weight=gate[i].view;gu.second=up[i].view;}
            if(make_down)dn.weight=down[i].view;
            gu.count=dn.count=int(tile);
            for(int t=0;t<gu.count;++t) {
                const auto slot=rows[start+t];
                gu.input_offsets[t]=(slot/routes)*input;gu.output_offsets[t]=slot*hidden;
                dn.input_offsets[t]=slot*hidden;dn.output_offsets[t]=slot*output;
            }
            if(make_gu)gu_tiles[tile-1].push_back(gu);
            if(make_down && !wide)down_tiles[tile-1].push_back(dn);
            start+=tile;
        }
        if(wide && make_down) {
            auto down_chunks=(rows.size()+7)/8;
            for(std::size_t start=0;start<rows.size();--down_chunks) {
                const auto tile=(rows.size()-start+down_chunks-1)/down_chunks;
                TrellisIndexedWideProjectionJob job{};job.weight=down[i].view;job.count=int(tile);
                for(int t=0;t<job.count;++t) {
                    const auto slot=rows[start+t];job.input_offsets[t]=slot*hidden;job.output_offsets[t]=slot*output;
                }
                if(tile>4)wide_tiles[tile-5].push_back(job);
                else {
                    TrellisIndexedProjectionJob short_job{};short_job.weight=job.weight;short_job.count=job.count;
                    std::copy_n(job.input_offsets,job.count,short_job.input_offsets);
                    std::copy_n(job.output_offsets,job.count,short_job.output_offsets);
                    down_tiles[tile-1].push_back(short_job);
                }
                start+=tile;
            }
        }
    }
    return {std::move(gu_tiles),std::move(down_tiles),std::move(wide_tiles),std::move(wide_gu)};
}
struct TqGraphLease {
    std::array<std::vector<TqWeight>,3> weights;
    std::unordered_map<int,MfqCudaStream> transfers;
    std::unordered_map<int,std::shared_ptr<DeviceImageArena>> staging;
    void clear(){for(auto& w:weights)w.clear();}
};
const char tq_graph_lease_identity=0;
struct TqMoeGraphStage {
    std::array<std::shared_ptr<TqRoutedWeight>,3> banks;
    std::shared_ptr<TqGraphLease> lease;
    MfqCudaStream stream,upload_stream;
    std::unique_ptr<mfq::cuda::HostBuffer> host_ids;
    std::array<tb::Tensor,3> views;
    std::shared_ptr<mfq::cuda::RoutedResidentViews<TrellisProjectionView>> resident_views;
    tb::Tensor resident_plan,stream_views;
    std::vector<std::size_t> expert_offsets;
    tb::Tensor input,ids,weights,hidden,pairs,result,plan,route_snapshot,cancelled;
    tb::ScalarType route_dtype;
    void* mapped_routes=nullptr;
    std::vector<int64_t> widened_routes;
    std::vector<int64_t> route_experts,resident_experts,observation_routes;
    int routes,type,tokens;
    std::size_t slots=0,route_bytes=0,plan_bytes=0;
    bool wide=false,direct=false;
    static constexpr std::size_t publication_stride=128;
    std::size_t control_bytes=64;
    static constexpr int upload_word=publication_stride/sizeof(uint32_t),abort_word=2*upload_word,
        down_word=3*upload_word;
    struct Plane {std::size_t offset=0;unsigned capacity=0,count=0;int tile=0,kind=0;};
    std::array<Plane,13> planes{};
    struct KernelChunk {int plane;std::size_t first,segment;unsigned capacity;cudaGraphNode_t node;};
    std::vector<KernelChunk> kernels;
    TqMoeGraphStage(const TqMoeBanks& b,const tb::Tensor& source,int count,tb::ScalarType route_type,
                   std::shared_ptr<TqGraphLease> l)
        :banks{b.gate,b.up,b.down},lease(std::move(l)),stream(mfq_get_current_cuda_stream(source.get_device())),upload_stream(stream),
         route_dtype(route_type),
         routes(count),type(dtype(source.scalar_type())),tokens(int(source.size(0))) {
        slots=std::size_t(tokens)*routes;
        if(slots>INT_MAX)throw std::overflow_error("TQ graph route count exceeds plan limits");
        route_bytes=slots*sizeof(int64_t);
        if(tokens>1) {
            wide=type==0 && tq_gemm_resources(source.get_device())->indexed_mma();
            for(int i=0;i<int(planes.size());++i) {
                auto& plane=planes[i];
                // GU1..4, GU8, Down1..4, Down5..8, in execution order.
                plane.kind=i<4?0:i==4?1:i<9?2:3;
                plane.tile=i<4?i+1:i==4?8:i-4;
                plane.offset=plan_bytes;
                // Every job consumes tile distinct routing slots. This bound
                // reserves less metadata than giving every tile all slots.
                plane.capacity=(plane.kind==1 || plane.kind==3) && !wide?0:unsigned(slots/plane.tile);
                const auto bytes=plane.kind==1?sizeof(TrellisIndexedWidePairJob):
                    plane.kind==3?sizeof(TrellisIndexedWideProjectionJob):sizeof(TrellisIndexedProjectionJob);
                if(plane.capacity>(std::size_t(INT64_MAX)-route_bytes-plan_bytes)/bytes)
                    throw std::overflow_error("TQ graph plan size exceeds tensor limits");
                plan_bytes+=plane.capacity*bytes;
            }
            plan=tb::empty({static_cast<int64_t>(plan_bytes)},source.options().dtype(tb::kUInt8));
            kernels.reserve(planes.size()*((slots+65534)/65535));
        }
        int can_map=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&can_map,cudaDevAttrCanMapHostMemory,source.get_device()));
        const auto* mapped_option=std::getenv("TYLOINFER_TQ_GRAPH_MAPPED_ROUTES");
        const bool integral=route_dtype==tb::kInt32 || route_dtype==tb::kInt64;
        const auto words=slots*(route_dtype==tb::kInt64?2:1);
        const bool map=can_map && integral && words<=INT_MAX && (!mapped_option || mapped_option[0]!='0');
        // The first cache line is the publication flag. Int64 IDs are copied
        // as two int32 words, preserving their full range without narrowing.
        const bool single_submission=map &&
            mfq::cuda::default_context(source.get_device())->supports_async_allocations();
        if(single_submission && tokens==1)plan_bytes=3*slots*sizeof(TrellisProjectionView);
        // Route publication is written by the GPU; upload/abort publication
        // is written by the CPU. Their words must not share a cache line.
        control_bytes=single_submission?4*publication_stride:64;
        host_ids=std::make_unique<mfq::cuda::HostBuffer>(control_bytes+route_bytes+plan_bytes,map);
        if(map)TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&mapped_routes,host_ids->data(),0));
        // Fixed grids consume zero-filled unused jobs, as in the GGUF grouped
        // graph path. Route changes do not require a host graph boundary.
        direct=single_submission && mapped_routes;
        if(direct) {
            if(tokens==1) {
                stream_views=tb::empty({static_cast<std::int64_t>(plan_bytes)},source.options().dtype(tb::kUInt8));
                expert_offsets.resize(std::size_t(banks[0]->experts));
            }
            auto found=lease->transfers.find(source.get_device());
            if(found==lease->transfers.end())found=lease->transfers.emplace(source.get_device(),
                banks[0]->cache->transfer_stream(source.get_device())).first;
            upload_stream=found->second;
            const auto bound=[](const TqRoutedWeight& bank) {
                return bank.table->row_image_bytes_bound(bank.out);
            };
            const auto one=bound(*banks[0])+bound(*banks[1])+bound(*banks[2]);
            const auto images=std::min(std::size_t(banks[0]->experts),slots);
            if(one>SIZE_MAX/images)throw std::overflow_error("TQ transfer geometry exceeds size limits");
            const auto bytes=one*images;
            auto& staging=lease->staging[source.get_device()];
            if(!staging || staging->bytes()<bytes) {
                const MfqCudaStreamGuard transfer(upload_stream);
                const mfq::cuda::RoutedWeightAllocationScope allocation;
                staging=std::make_shared<DeviceImageArena>(bytes,source.get_device());
            }
            banks[0]->cache->prepare_transfer_staging(source.get_device());
            cancelled=tb::empty({1},source.options().dtype(tb::kInt32));
            const auto resident_bytes=tokens==1?3*slots*sizeof(TrellisProjectionView):2*slots*sizeof(TrellisIndexedProjectionJob);
            resident_plan=tb::empty({static_cast<int64_t>(resident_bytes)},source.options().dtype(tb::kUInt8));
        }
        if(map)resident_views=banks[0]->cache->resident_views(
            {banks[0]->source_id,banks[1]->source_id,banks[2]->source_id},banks[0]->experts,source.get_device());
        if(mapped_routes) {
            route_snapshot=tb::empty({tokens,count},source.options().dtype(route_dtype));
            if(route_dtype==tb::kInt32)widened_routes.resize(slots);
        }
        input=tb::empty_like(source);
        ids=tb::empty({tokens,count},source.options().dtype(tb::kInt64));
        weights=tb::empty({tokens,count},source.options().dtype(tb::kFloat32));
        hidden=tb::empty({int64_t(slots),b.up->out},source.options());
        pairs=tb::empty({tokens,count,b.down->out},source.options());
        result=tb::empty({tokens,b.down->out},source.options());
        if(tokens==1)for(int role=0;role<3;++role)views[role]=banks[role]->cache->projection_table(
            banks[role]->source_id,banks[role]->experts,nullptr,{},source.get_device());
        mfq::cuda::charge_tensor_host_bytes(host_ids->size());
    }
    void reset_routes() {
        if(!mapped_routes)return;
        auto* flags=static_cast<volatile uint32_t*>(host_ids->data());
        flags[0]=0;if(direct){flags[upload_word]=0;flags[down_word]=0;flags[abort_word]=0;}
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    void cancel() {
        if(!direct)return;
        auto* flags=static_cast<volatile uint32_t*>(host_ids->data());
        mfq::cuda::publish_mapped_flag(const_cast<uint32_t*>(flags)+abort_word);
        mfq::cuda::publish_mapped_flag(const_cast<uint32_t*>(flags)+upload_word);
        mfq::cuda::publish_mapped_flag(const_cast<uint32_t*>(flags)+down_word);
    }
    void* route_data()const {return static_cast<uint8_t*>(host_ids->data())+control_bytes;}
    void publish_routes(const tb::Tensor& current,bool views,const int64_t* device_ids) {
        if(!mapped_routes) {
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(route_data(),device_ids,route_bytes,cudaMemcpyDeviceToHost,stream.stream()));
            return;
        }
        reset_routes();
        const auto* source=&current;
        if(!views || !current.is_contiguous()){route_snapshot.copy_(current);source=&route_snapshot;}
        mfq::cuda::moe_publish_routes(nullptr,reinterpret_cast<const int32_t*>(source->data_ptr()),nullptr,
            reinterpret_cast<int32_t*>(static_cast<uint8_t*>(mapped_routes)+control_bytes),
            static_cast<uint32_t*>(mapped_routes),0,int(slots*(route_dtype==tb::kInt64?2:1)),stream.stream());
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
    }
    ~TqMoeGraphStage() {
        // The D2H destination belongs to this stage, rather than the tensor
        // allocator. A window destroys its graphs before releasing the stage.
        int previous=0;const bool restore=cudaGetDevice(&previous)==cudaSuccess;
        (void)cudaSetDevice(stream.device());(void)cudaStreamSynchronize(stream.stream());
        if(direct)(void)cudaStreamSynchronize(upload_stream.stream());
        mfq::cuda::tensor_host_bytes.fetch_sub(host_ids->size());
        if(restore)(void)cudaSetDevice(previous);
    }
    void load(mfq::cuda::DecodeWindow* window=nullptr) {
        const bool trace=std::getenv("TYLOINFER_TQ_TRACE_HANDOFF")!=nullptr;
        const auto trace_step=[&](const char* step,int role=-1) {
            if(trace){std::fprintf(stderr,"tq_handoff source=%llu step=%s role=%d\n",
                static_cast<unsigned long long>(banks[0]->source_id),step,role);std::fflush(stderr);}
        };
        trace_step("begin");
        if(!std::all_of(banks.begin(),banks.end(),[](const auto& bank){return bool(bank->resident);}))
            banks[0]->cache->wait_layer_admission(resident_views);
        const MfqCudaStreamGuard guard(upload_stream);
        std::optional<mfq::cuda::NonblockingAllocationScope> allocation;
        if(direct && window)allocation.emplace();
        const auto* selected=static_cast<const int64_t*>(route_data());
        if(mapped_routes && route_dtype==tb::kInt32) {
            std::copy_n(static_cast<const int32_t*>(route_data()),slots,widened_routes.data());
            selected=widened_routes.data();
        }
        // The next layer may reuse the mapped route packet after Down is
        // published. Learning uses the same owned snapshot as the GGUF path.
        if(direct){observation_routes.assign(selected,selected+slots);selected=observation_routes.data();}
        const auto observe=[&] {
            if(window)banks[0]->cache->observe_routes(
                {banks[0]->source_id,banks[1]->source_id,banks[2]->source_id},selected,tokens,routes,stream.device());
        };
        auto& unique=route_experts;
        std::vector<std::vector<int64_t>> grouped;
        mfq::cuda::group_routed_experts({selected,slots},banks[0]->experts,unique,tokens==1?nullptr:&grouped);
        const bool resident_partition=direct && window;
        const bool compact_views=resident_partition && tokens==1;
        bool streamed_views=false;
        if(resident_partition) {
            resident_experts.clear();std::size_t cold=0;
            for(std::size_t index=0;index<unique.size();++index) {
                const auto expert=unique[index];
                bool resident=true;
                for(int role=0;role<3;++role)resident&=resident_views->pointers(role)[expert].words!=nullptr;
                if(compact_views)expert_offsets[std::size_t(expert)]=resident?SIZE_MAX:cold;
                if(resident)resident_experts.push_back(expert);
                else {
                    unique[cold]=expert;
                    if(tokens>1 && cold!=index)grouped[cold]=std::move(grouped[index]);
                    ++cold;
                }
            }
            unique.resize(cold);if(tokens>1)grouped.resize(cold);
            streamed_views=!unique.empty();
            banks[0]->cache->touch_decode_residents(
                {banks[0]->source_id,banks[1]->source_id,banks[2]->source_id},resident_experts,stream.device());
        }
        // The preceding GPU segment has consumed the prior layer's images.
        // Keep one live layer across all TQ sources, independently of cache
        // ownership, so graph capture cannot pin every layer's experts.
        lease->clear();trace_step("released");
        if(compact_views && unique.empty()) {
            // Release the GPU before bookkeeping, matching direct GGUF DMA.
            mfq::cuda::publish_mapped_flag(static_cast<uint32_t*>(host_ids->data())+upload_word);
            mfq::cuda::publish_mapped_flag(static_cast<uint32_t*>(host_ids->data())+down_word);
            observe();return;
        }
        const bool shared_cache=banks[0]->cache==banks[1]->cache && banks[0]->cache==banks[2]->cache;
        auto batch=window && shared_cache?banks[0]->cache->begin_uploads(stream.device()):nullptr;
        for(int role=0;role<3;++role) {
            trace_step("weights",role);
            const auto& bank=*banks[role];
            lease->weights[role]=bank.cache->get_rows(bank.source_id,unique.data(),unique.size(),*bank.table,bank.out,stream.device(),
                direct && window?lease->staging.at(stream.device()).get():nullptr,
                window?mfq::cuda::RoutedImageUse::Decode:mfq::cuda::RoutedImageUse::Eager,batch.get());
            trace_step("views",role);
            if(tokens==1) {
                if(compact_views) {
                    if(streamed_views) {
                        auto* packet=reinterpret_cast<TrellisProjectionView*>(
                            static_cast<std::uint8_t*>(route_data())+route_bytes);
                        for(std::size_t slot=0;slot<slots;++slot) {
                            const auto offset=expert_offsets[std::size_t(selected[slot])];
                            packet[std::size_t(role)*slots+slot]=offset==SIZE_MAX?TrellisProjectionView{}:
                                lease->weights[role][offset].view;
                        }
                        if(role==1 || role==2) {
                            const auto first=role==1?0:2*slots,count=role==1?2*slots:slots;
                            bank.cache->upload_decode_views(stream_views,first*sizeof(TrellisProjectionView),
                                packet+first,count*sizeof(TrellisProjectionView));
                        }
                    }
                }else {
                    const auto table=bank.cache->projection_table(bank.source_id,bank.experts,unique.data(),
                        lease->weights[role],stream.device(),stream.stream());
                    if(table.data_ptr()!=views[role].data_ptr())throw std::logic_error("TQ graph view table address changed");
                }
            }
            if(role==1 && direct) {
                if(tokens>1)upload_indexed_plan(grouped,0,5,window);
                // Gate/Up can consume their images while Down is prepared and
                // transferred. The transaction still owns all three banks.
                mfq::cuda::enqueue_routed_completion(static_cast<uint32_t*>(host_ids->data())+upload_word,
                    static_cast<uint32_t*>(mapped_routes)+upload_word,upload_stream.stream());
                mfq::cuda::check_stream_progress(upload_stream.stream());
            }
        }
        if(batch)banks[0]->cache->finish_uploads(*batch);
        if(tokens==1) {
            if(direct) {
                // Publish readiness on the DMA stream after its copies. The
                // compute graph waits on this flag; the CPU can immediately
                // serve the next stage without a driver synchronization.
                mfq::cuda::enqueue_routed_completion(static_cast<uint32_t*>(host_ids->data())+down_word,
                    static_cast<uint32_t*>(mapped_routes)+down_word,upload_stream.stream());
                mfq::cuda::check_stream_progress(upload_stream.stream());
                trace_step("dma_ready_queued");
            }
            observe();return;
        }
        if(direct) {
            upload_indexed_plan(grouped,5,int(planes.size()),window);
            mfq::cuda::enqueue_routed_completion(static_cast<uint32_t*>(host_ids->data())+down_word,
                static_cast<uint32_t*>(mapped_routes)+down_word,upload_stream.stream());
            mfq::cuda::check_stream_progress(upload_stream.stream());
        }else upload_indexed_plan(grouped,0,int(planes.size()),window);
        observe();
    }
    void upload_indexed_plan(const std::vector<std::vector<std::int64_t>>& grouped,int first,int last,
            mfq::cuda::DecodeWindow* window) {
        auto [gu,dn,mma,gu8]=tq_indexed_jobs(lease->weights[0],lease->weights[1],lease->weights[2],
            grouped,0,routes,banks[0]->input,banks[1]->out,banks[2]->out,wide,first<5,last>5);
        auto* packet=static_cast<std::uint8_t*>(route_data())+route_bytes;
        const auto offset=planes[first].offset,end=last==int(planes.size())?plan_bytes:planes[last].offset;
        std::memset(packet+offset,0,end-offset);
        for(int i=first;i<last;++i) {
            auto& plane=planes[i];
            const auto fill=[&](const auto& jobs) {
                if(jobs.size()>plane.capacity)throw std::overflow_error("TQ graph job count exceeds prepared plan");
                plane.count=unsigned(jobs.size());
                if(!jobs.empty())std::memcpy(packet+plane.offset,jobs.data(),jobs.size()*sizeof(jobs[0]));
            };
            if(plane.kind==0)fill(gu[plane.tile-1]);
            else if(plane.kind==1)fill(gu8);
            else if(plane.kind==2)fill(dn[plane.tile-1]);
            else fill(mma[plane.tile-5]);
        }
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(plan.data_ptr<std::uint8_t>()+offset,packet+offset,end-offset,
            cudaMemcpyHostToDevice,direct?upload_stream.stream():stream.stream()));
        if(window && !direct)for(const auto& kernel:kernels)if(kernel.plane>=first && kernel.plane<last) {
            const auto count=kernel.first>=planes[kernel.plane].count?0:
                unsigned(std::min<std::size_t>(kernel.capacity,planes[kernel.plane].count-kernel.first));
            window->set_kernel_rows(kernel.segment,kernel.node,count);
        }
        if(window && std::getenv("TYLOINFER_TQ_PROFILE_CACHE")) {
            unsigned mask=0,jobs=0;
            for(int i=0;i<int(planes.size());++i)if(planes[i].count){mask|=1u<<i;jobs+=planes[i].count;}
            static std::atomic<unsigned> seen{0};
            if(mask & ~seen.fetch_or(mask,std::memory_order_relaxed))
                std::fprintf(stderr,"tq_indexed_dispatch mode=graph dtype=%d tokens=%d jobs=%u gu8_jobs=%u plane_mask=%u plan_bytes=%zu\n",
                    type,tokens,jobs,planes[4].count,mask,plan_bytes);
        }
    }
    void indexed(float limit,mfq::cuda::DecodeWindow& window,bool recording,const void* input_values,bool down) {
#if CUDART_VERSION >= 11060
        if(recording && !down)kernels.clear();
        for(int i=down?5:0;i<(down?int(planes.size()):5);++i) {
            const auto& plane=planes[i];
            for(std::size_t first=0;first<plane.capacity;first+=65535) {
                const auto capacity=unsigned(std::min<std::size_t>(65535,plane.capacity-first));
                const auto active=first>=plane.count?0:
                    unsigned(std::min<std::size_t>(capacity,plane.count-first));
                const auto count=recording || direct?capacity:std::max(1u,active);
                const auto* address=plan.data_ptr<std::uint8_t>()+plane.offset;
                const auto* jobs=reinterpret_cast<const TrellisIndexedProjectionJob*>(address)+first;
                if(direct) {
                    const auto* abort=reinterpret_cast<const std::uint32_t*>(cancelled.data_ptr());
                    if(plane.kind==0 || plane.kind==2)status(mfq_tq_indexed_guarded(jobs,
                        down?hidden.data_ptr():input_values,down?pairs.data_ptr():hidden.data_ptr(),int(count),
                        int(down?banks[2]->out:banks[1]->out),type,plane.tile,limit,!down,abort,stream.stream()));
                    else {
                        const auto stride=down?sizeof(TrellisIndexedWideProjectionJob):sizeof(TrellisIndexedWidePairJob);
                        status(mfq_tq_indexed_wide_guarded(address+first*stride,
                            down?hidden.data_ptr():input_values,down?pairs.data_ptr():hidden.data_ptr(),int(count),
                            int(down?banks[2]->out:banks[1]->out),type,plane.tile,limit,!down,abort,stream.stream()));
                    }
                }else if(plane.kind==0)status(mfq_tq_indexed_swiglu(jobs,input_values,hidden.data_ptr(),int(count),
                    int(banks[1]->out),type,plane.tile,limit,stream.stream()));
                else if(plane.kind==1)status(mfq_tq_indexed_wide_swiglu(
                    reinterpret_cast<const TrellisIndexedWidePairJob*>(address)+first,
                    input_values,hidden.data_ptr(),int(count),int(banks[1]->out),type,plane.tile,limit,stream.stream()));
                else if(plane.kind==2)status(mfq_tq_indexed(jobs,hidden.data_ptr(),pairs.data_ptr(),int(count),
                    int(banks[2]->out),type,plane.tile,stream.stream()));
                else status(mfq_tq_indexed_wide(reinterpret_cast<const TrellisIndexedWideProjectionJob*>(address)+first,
                    hidden.data_ptr(),pairs.data_ptr(),int(count),int(banks[2]->out),type,plane.tile,stream.stream()));
                if(recording && !direct) {
                    cudaStreamCaptureStatus capture;const cudaGraphNode_t* dependencies=nullptr;std::size_t n=0;
#if CUDART_VERSION >= 13000
                    TYLOINFER_CUDA_CHECK(cudaStreamGetCaptureInfo(stream.stream(),&capture,nullptr,nullptr,&dependencies,nullptr,&n));
#else
                    TYLOINFER_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(stream.stream(),&capture,nullptr,nullptr,&dependencies,&n));
#endif
                    if(capture!=cudaStreamCaptureStatusActive || n!=1)throw std::logic_error("TQ kernel capture has no unique node");
                    kernels.push_back({i,first,window.segments()-1,capacity,dependencies[0]});
                }
            }
            if(i==4)window.mark_current("gate_up_end");
        }
#else
        (void)limit;(void)window;(void)recording;(void)input_values;(void)down;
        throw std::runtime_error("multi-token TQ graph plan updates require CUDA11.6 or newer");
#endif
    }
};
tb::Tensor tq_graph_forward(const TqMoeBanks& banks,const tb::Tensor& source,const tb::Tensor& ids,
        const tb::Tensor& weights,float limit,mfq::cuda::DecodeWindow& window,const std::function<void()>& independent_work,
        CudaIndependentBranchExecutor* shared_executor) {
    window.mark_current("expert_begin");
    const auto cache=banks.gate->cache;
    const auto device=source.get_device();
    auto entered=window.workspace<bool>(cache.get(),[]{return std::make_shared<bool>(false);});
    auto lease=window.workspace<TqGraphLease>(&tq_graph_lease_identity,[]{return std::make_shared<TqGraphLease>();});
    const bool immutable=banks.gate->resident && banks.up->resident && banks.down->resident;
    if(!immutable)window.enroll_once({cache.get(),[cache,device,entered]{cache->begin_replay(device);*entered=true;},[]{},
        [cache,device,entered,lease]{if(*entered){lease->clear();*entered=false;cache->finish_routes(device);}},
        [cache,device,entered]{if(*entered){*entered=false;cache->cancel_replay(device);}},
        [cache,device,entered]{
            if(*entered){*entered=false;cache->cancel_replay(device);}
            cache->release_deferred_events();
        },{}});
    auto stage=window.workspace<TqMoeGraphStage>(&banks,[&] {
        return std::make_shared<TqMoeGraphStage>(banks,source,int(ids.size(1)),ids.scalar_type(),lease);
    });
    if(stage->routes!=ids.size(1) || stage->input.sizes()!=source.sizes() || stage->input.scalar_type()!=source.scalar_type() ||
       stage->route_dtype!=ids.scalar_type())
        throw std::invalid_argument("TQ graph stage geometry changed");
    const bool views=mfq::cuda::runtime_options::window_input_views() && source.is_contiguous() &&
        ids.is_contiguous() && weights.is_contiguous() && weights.scalar_type()==tb::kFloat32;
    if(std::getenv("TYLOINFER_TQ_PROFILE_CACHE")) {
        const unsigned mode=(stage->mapped_routes?(stage->route_dtype==tb::kInt32?1u:2u):4u)|(views?8u:0u);
        static std::atomic<unsigned> seen{0};const unsigned bit=1u<<mode;
        if(!(seen.fetch_or(bit,std::memory_order_relaxed)&bit))
            std::fprintf(stderr,"tq_graph_handoff mapped_routes=%d route_bits=%d input_views=%d single_submission=%d slot_views=%d\n",
                int(stage->mapped_routes!=nullptr),stage->route_dtype==tb::kInt32?32:64,int(views),int(stage->direct),
                int(stage->direct && stage->tokens==1));
    }
    if(!views){stage->input.copy_(source);stage->weights.copy_(weights);}
    const int64_t* device_ids=nullptr;
    if(views && ids.scalar_type()==tb::kInt64)device_ids=ids.data_ptr<int64_t>();
    else if(stage->direct || !stage->mapped_routes || !views) {
        stage->ids.copy_(ids);device_ids=stage->ids.data_ptr<int64_t>();
    }
    stage->publish_routes(ids,views,device_ids);
    window.mark_current("cpu_wait_begin");
    cudaStreamCaptureStatus capture;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stage->stream.stream(),&capture));
    mfq::cuda::prepare_routed_graph(&window,capture,stage->mapped_routes!=nullptr,stage->direct,
        {stage.get(),[stage]{stage->reset_routes();},{},[lease]{lease->clear();},
            [stage]{stage->cancel();},[lease]{lease->clear();},[lease]{lease->clear();}},
        [stage] {
            if(stage->mapped_routes)mfq::cuda::wait_route_publication(
                static_cast<const uint32_t*>(stage->host_ids->data()),stage->stream.stream());
            else TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stage->stream.stream()));
        },[stage]{stage->load();},[stage,&window]{stage->load(&window);});
    // A segmented graph must join any side stream before ending its segment.
    // Fork after the host barrier; a single-submission graph also overlaps
    // the shared branch with its remaining routed DMA wait.
    const auto compute=[&]() -> tb::Tensor {
        const auto* input_values=views?source.data_ptr():stage->input.data_ptr();
        const bool overlap=stage->direct && capture==cudaStreamCaptureStatusActive;
        auto* selected=overlap && stage->tokens==1?reinterpret_cast<TrellisProjectionView*>(stage->resident_plan.data_ptr()):nullptr;
        const auto* abort=stage->direct?reinterpret_cast<const uint32_t*>(stage->cancelled.data_ptr()):nullptr;
        return mfq::cuda::execute_routed_moe([]{},[]{},[&] {
            if(!overlap)return;
            mfq::cuda::wait_mapped_flag(stage->resident_views->admission_device(),stage->stream.stream());
            if(stage->tokens>1) {
                auto* gu=reinterpret_cast<TrellisIndexedProjectionJob*>(stage->resident_plan.data_ptr());
                auto* dn=gu+stage->slots;
                const auto tile=std::min(stage->tokens,4);
                status(mfq_tq_prepare_resident_indexed(stage->resident_views->device_pointers(0),
                    stage->resident_views->device_pointers(1),stage->resident_views->device_pointers(2),device_ids,
                    gu,dn,int(stage->slots),stage->routes,int(banks.gate->experts),int(banks.gate->input),
                    int(banks.up->out),int(banks.down->out),tile,stage->stream.stream()));
                window.mark_current("hot_begin");
                status(mfq_tq_indexed_swiglu(gu,input_values,stage->hidden.data_ptr(),int(stage->slots),
                    int(banks.up->out),stage->type,tile,limit,stage->stream.stream()));
                window.mark_current("resident_gate_up_end");
                status(mfq_tq_indexed(dn,stage->hidden.data_ptr(),stage->pairs.data_ptr(),int(stage->slots),
                    int(banks.down->out),stage->type,tile,stage->stream.stream()));
                window.mark_current("resident_down_end");return;
            }
            status(mfq_tq_snapshot_resident(stage->resident_views->device_pointers(0),
                stage->resident_views->device_pointers(1),stage->resident_views->device_pointers(2),device_ids,
                selected,stage->routes,int(banks.gate->experts),stage->stream.stream()));
            window.mark_current("hot_begin");
            status(mfq_tq_partition_swiglu(selected,selected+stage->routes,selected,device_ids,input_values,
                stage->hidden.data_ptr(),stage->routes,int(banks.gate->experts),int(banks.up->out),stage->type,limit,
                true,nullptr,stage->stream.stream()));
            window.mark_current("resident_gate_up_end");
            status(mfq_tq_partition_down(selected+2*stage->routes,selected,device_ids,stage->hidden.data_ptr(),
                stage->pairs.data_ptr(),stage->routes,int(banks.gate->experts),int(banks.down->out),int(banks.up->out),
                stage->type,true,nullptr,stage->stream.stream()));
            window.mark_current("resident_down_end");
        },[&] {
            if(stage->direct) {
                auto* flags=static_cast<uint32_t*>(stage->mapped_routes);
                mfq::cuda::wait_mapped_plan(flags+TqMoeGraphStage::upload_word,nullptr,nullptr,0,
                    flags+TqMoeGraphStage::abort_word,reinterpret_cast<uint32_t*>(stage->cancelled.data_ptr()),stage->stream.stream());
            }
            window.mark_current(overlap?"stream_gate_up_join":"hot_begin");
        },[&] {
            if(stage->tokens>1) {
                stage->indexed(limit,window,capture==cudaStreamCaptureStatusActive,input_values,false);return;
            }
            const auto* gate=reinterpret_cast<const TrellisProjectionView*>(
                overlap?stage->stream_views.data_ptr():stage->views[0].data_ptr());
            const auto* up=overlap?gate+stage->slots:reinterpret_cast<const TrellisProjectionView*>(stage->views[1].data_ptr());
            if(overlap)status(mfq_tq_partition_slot_swiglu(gate,up,selected,device_ids,input_values,stage->hidden.data_ptr(),
                stage->routes,int(banks.gate->experts),int(banks.up->out),stage->type,limit,abort,stage->stream.stream()));
            else if(stage->direct)status(mfq_tq_routed_guarded_swiglu(gate,up,device_ids,input_values,stage->hidden.data_ptr(),
                stage->routes,int(banks.up->out),stage->type,limit,abort,stage->stream.stream()));
            else status(mfq_tq_routed_swiglu(gate,up,device_ids,input_values,stage->hidden.data_ptr(),
                stage->routes,int(banks.up->out),stage->type,limit,stage->stream.stream()));
            window.mark_current("gate_up_end");
        },[&] {
            if(!stage->direct)return;
            auto* flags=static_cast<uint32_t*>(stage->mapped_routes);
            mfq::cuda::wait_mapped_plan(flags+TqMoeGraphStage::down_word,nullptr,nullptr,0,
                flags+TqMoeGraphStage::abort_word,reinterpret_cast<uint32_t*>(stage->cancelled.data_ptr()),stage->stream.stream());
            window.mark_current("stream_down_join");
        },[&] {
            if(stage->tokens>1) {
                stage->indexed(limit,window,capture==cudaStreamCaptureStatusActive,input_values,true);return;
            }
            const auto* down=overlap?reinterpret_cast<const TrellisProjectionView*>(stage->stream_views.data_ptr())+2*stage->slots:
                reinterpret_cast<const TrellisProjectionView*>(stage->views[2].data_ptr());
            if(overlap)status(mfq_tq_partition_slot_down(down,selected,device_ids,stage->hidden.data_ptr(),stage->pairs.data_ptr(),
                stage->routes,int(banks.gate->experts),int(banks.down->out),int(banks.up->out),stage->type,abort,stage->stream.stream()));
            else if(stage->direct)status(mfq_tq_routed_guarded_down(down,device_ids,stage->hidden.data_ptr(),stage->pairs.data_ptr(),
                stage->routes,int(banks.down->out),int(banks.up->out),stage->type,abort,stage->stream.stream()));
            else status(mfq_tq_routed_down(down,device_ids,stage->hidden.data_ptr(),stage->pairs.data_ptr(),
                stage->routes,int(banks.down->out),int(banks.up->out),stage->type,stage->stream.stream()));
        },[&] {
            window.mark_current("down_end");
            status(mfq_tq_weighted_reduce(stage->pairs.data_ptr(),views?weights.data_ptr<float>():stage->weights.data_ptr<float>(),stage->result.data_ptr(),
                stage->tokens,stage->routes,int(banks.down->out),stage->type,stage->stream.stream()));
            window.mark_current("hot_end");return stage->result;
        });
    };
    if(auto output=try_independent_branch(shared_executor,independent_work,compute,stage->tokens,
            stage->upload_stream.stream()))return *output;
    if(independent_work)independent_work();
    return compute();
}
}

tb::Tensor TqMoeBanks::forward(tb::Tensor input_value,tb::Tensor ids,
        tb::Tensor weights,float limit,CudaProfiler* profiler,std::int64_t expert_begin,std::int64_t expert_end,
        const std::function<void()>& independent_work,CudaIndependentBranchExecutor* shared_executor)const {
    const auto phase=[&](const char* name,auto&& operation) {
        return profiler?profiler->measure(name,operation):operation();
    };
    if(!gate || !up || !down || gate->experts!=up->experts || up->experts!=down->experts ||
       gate->input!=up->input || gate->out!=up->out || down->input!=up->out ||
       ids.dim()!=2 || ids.size(1)<1 || ids.size(1)>INT_MAX || input_value.dim()!=2 ||
       !input_value.is_cuda() || weights.device()!=input_value.device() ||
       input_value.size(0)!=ids.size(0) || input_value.size(1)!=up->input ||
       weights.sizes().vec()!=ids.sizes().vec() || !std::isfinite(limit) || limit<0)
        throw std::invalid_argument("invalid TQ three-bank MoE geometry");
    if(expert_end==-1)expert_end=gate->experts;
    if(expert_begin<0 || expert_end<expert_begin || expert_end>gate->experts)
        throw std::invalid_argument("invalid TQ expert ownership range");
    const bool partitioned=expert_begin!=0 || expert_end!=gate->experts;
    const auto tokens=ids.size(0),routes=ids.size(1),hidden=up->out,width=down->out;
    if(tokens>INT_MAX || hidden>INT_MAX || width>INT_MAX || up->input>INT_MAX)
        throw std::overflow_error("TQ MoE activation dimensions exceed CUDA limits");
    const auto source=input_value.contiguous();
    const MfqCudaGuard guard(source.get_device());
    if(tokens<=32 && shared_executor && independent_work && !mfq::cuda::DecodeWindow::recording()) {
        if(auto output=try_independent_branch(shared_executor,independent_work,[&] {
            return forward(source,ids,weights,limit,profiler,expert_begin,expert_end);
        },tokens,gate->cache->transfer_stream(source.get_device()).stream()))return *output;
    }
    if(auto* window=mfq::cuda::DecodeWindow::recording();!partitioned && window && tq_graph_supports_tokens(tokens) && ids.device()==source.device())
        return tq_graph_forward(*this,source,ids,weights,limit,*window,independent_work,shared_executor);
    gate->cache->wait_ready();
    if(independent_work)independent_work();
    auto result=tb::empty({tokens,width},source.options());
    if(!tokens)return result;
    const auto type=dtype(source.scalar_type());
    const auto coefficients=weights.to(tb::kFloat32).contiguous();
    // Exactly one route readback for all three banks, independent of the
    // number of selected tokens and their expert reuse pattern.
    // Router output is normally int32. Widen this tiny ID list on CPU after
    // the readback, avoiding a GPU allocation/conversion before every wait.
    const auto host=phase("tq.moe.routes",[&] {return gate->cache->read_routes(ids);});
    std::vector<std::int64_t> unique;
    std::vector<std::vector<std::int64_t>> slots;
    mfq::cuda::group_routed_experts(host,up->experts,unique,&slots,expert_begin,expert_end);
    if(unique.empty())return tb::zeros({tokens,width},source.options());
    if(tokens>32 && gate->cache==up->cache && gate->cache==down->cache)
        gate->cache->record_prefill({gate->source_id,up->source_id,down->source_id},unique,slots,tokens,source.get_device());
    // Choose GEMM by each expert's actual token count. A long prompt can
    // still have many sparsely selected experts that belong in indexed jobs.
    std::size_t long_groups=0;
    for(std::size_t i=0;i<unique.size();++i)if(slots[i].size()>=std::size_t(gemm_min_tokens(source.scalar_type()))) {
        std::swap(unique[i],unique[long_groups]);std::swap(slots[i],slots[long_groups]);++long_groups;
    }
    const auto resources=tq_gemm_resources(source.get_device());
    std::size_t prefill_groups=0;
    if(type==0 && (width==up->input || (tokens>32 && resources->prefill_mma())) && hidden<=INT_MAX/2) {
        // Batch up to two 128-token reuse tiles to amortize expert dispatch.
        // Larger groups retain GEMM, which decodes each weight tile once.
        if(tokens>32)prefill_groups=long_groups=unique.size();
        else for(std::size_t i=0;i<long_groups;++i)if(slots[i].size()<=256) {
            std::swap(unique[i],unique[prefill_groups]);std::swap(slots[i],slots[prefill_groups]);++prefill_groups;
        }
    }
    auto pairs=partitioned?tb::zeros({tokens,routes,width},source.options()):tb::empty({tokens,routes,width},source.options());
    auto flat_pairs=pairs.reshape({tokens*routes,width});
    const auto upload_plan=[&](const auto& values) {
        if(values.empty())return tb::Tensor{};
        const auto bytes=static_cast<std::int64_t>(values.size()*sizeof(values[0]));
        return tb::from_blob(const_cast<void*>(static_cast<const void*>(values.data())),{bytes},
            tb::TensorOptions().dtype(tb::kUInt8)).clone().to(source.device(),false,false).contiguous();
    };
    if(prefill_groups) {
        // Similar token counts share a compute tile. A frequently selected
        // expert must not force padding every sparse expert to its capacity.
        std::vector<std::pair<int64_t,std::vector<int64_t>>> sorted;
        sorted.reserve(prefill_groups);
        for(std::size_t i=0;i<prefill_groups;++i)sorted.emplace_back(unique[i],std::move(slots[i]));
        std::stable_sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){return a.second.size()>b.second.size();});
        std::size_t selected_slots=0;
        for(std::size_t i=0;i<prefill_groups;++i) {
            unique[i]=sorted[i].first;slots[i]=std::move(sorted[i].second);selected_slots+=slots[i].size();
        }
        const auto stream=mfq_get_current_cuda_stream(source.get_device());
        resources->retain(stream);
        std::size_t gemm_groups=0;
        // PP follows the GGUF decoded-GEMM schedule. Short verification keeps
        // its direct tile kernel; long prompts decode each expert only once.
        while(gemm_groups<prefill_groups && (tokens>32 || !resources->prefill_mma() || slots[gemm_groups].size()>128))++gemm_groups;
        const auto job_bytes=(prefill_groups*sizeof(TrellisPrefillMoeJob)+15)&~std::size_t(15);
        const mfq::cuda::RoutedPrefillRoutes route_plan(prefill_groups,[&](auto i)->const auto&{return slots[i];});
        const auto bounds_offset=job_bytes+selected_slots*sizeof(int64_t);
        const auto static_bytes=(selected_slots+prefill_groups+1)*sizeof(int64_t);
        TqPrefillPlanLease lease{resources->prefill_plan(job_bytes+static_bytes),stream.stream()};
        auto* host_bytes=static_cast<std::uint8_t*>(lease.plan->host.data_ptr());
        auto* device_bytes=static_cast<std::uint8_t*>(lease.plan->device.data_ptr());
        auto* host_slots=reinterpret_cast<int64_t*>(host_bytes+job_bytes);
        const auto& slot_offsets=route_plan.bounds;
        std::memcpy(host_slots,route_plan.slots.data(),selected_slots*sizeof(int64_t));
        std::memcpy(host_bytes+bounds_offset,slot_offsets.data(),slot_offsets.size()*sizeof(int64_t));
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(device_bytes+job_bytes,host_slots,static_bytes,
            cudaMemcpyHostToDevice,stream.stream()));
        const auto free_bytes=[&] {
            return phase("tq.moe.capacity",[&] {
                std::size_t available=0,total=0;TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&available,&total));return available;
            });
        };
        const auto available=free_bytes();
        std::size_t image_per_expert=0;
        for(std::size_t i=0;i<prefill_groups;++i) {
            std::size_t bytes=0;
            for(const auto* bank:{gate.get(),up.get(),down.get()}) {
                const auto size=bank->table->expert_image_bytes(unique[i],bank->out);
                bytes=mfq::cuda::routed_saturating_add(bytes,(size+15)&~std::size_t(15));
            }
            image_per_expert=std::max(image_per_expert,bytes);
        }
        // Match the bounded compressed PP staging used by the GGUF adapter.
        // Actual encoded sizes include row metadata and heterogeneous bits.
        const auto image_budget=std::max<std::size_t>(image_per_expert,
            std::min<std::size_t>(std::size_t(64)<<20,available/4));
        const auto image_capacity=std::min(mfq::cuda::routed_prefill_group_capacity(prefill_groups,image_per_expert,
            2.L*(2.L*up->input*hidden+static_cast<long double>(hidden)*width)),
            std::max<std::size_t>(1,image_budget/image_per_expert));
        const auto group_ends=mfq::cuda::routed_prefill_group_ends(prefill_groups,image_capacity,
            [&](auto i){return slots[i].size();});
        // Price the compute batches that these image groups actually allow.
        // Compact kernels need only valid routes; legacy verification keeps
        // its padded buffers. Neither workspace scales with the whole layer.
        const auto workspace=mfq::cuda::routed_prefill_workspace_plan(group_ends,gemm_groups,
            [&](auto i){return slots[i].size();},
            mfq::cuda::RoutedPrefillWorkspaceShape{std::uint64_t(up->input),std::uint64_t(hidden),std::uint64_t(width)},
            std::min<std::uint64_t>(std::uint64_t(gemm_groups?64:16)<<20,available/16),
            std::min<std::uint64_t>(std::uint64_t(256)<<20,available/4),resources->prefill_mma());
        const auto image_batch=workspace.peak_image_batch;
        if(workspace.activation_bytes>std::uint64_t(INT64_MAX) || workspace.decoded_bytes>std::uint64_t(INT64_MAX))
            throw std::overflow_error("TQ prefill workspace exceeds tensor byte limits");
        auto activations=tb::empty({static_cast<int64_t>(workspace.activation_bytes)},source.options().dtype(tb::kUInt8));
        tb::Tensor decoded;
        cublasHandle_t blas=nullptr;
        if(gemm_groups) {
            decoded=tb::empty({static_cast<int64_t>(workspace.decoded_bytes)},source.options().dtype(tb::kUInt8));
            if(std::any_of(workspace.tiles.begin(),workspace.tiles.end(),
                    [](const auto& tile){return tile.gemm && !tile.compact;}))blas=resources->get();
        }
        TqPrefillWeightPipeline pipeline(stream,gate->cache->transfer_stream(source.get_device()));
        {
            const MfqCudaStreamGuard transfer(pipeline.upload);
            const mfq::cuda::RoutedWeightAllocationScope temporary(false);
            for(auto& stage:pipeline.stages)stage.workspace.image=tb::empty(
                {int64_t(image_batch*image_per_expert)},source.options().dtype(tb::kUInt8));
        }
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<bool> seen{false};
            if(!seen.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                "native_tq_prefill_weight_pipeline actual_match=1 arenas=2 gate_up_early=1 decoded_gemm=%d compact_gemm=%zu compute_tiles=%zu activation_bytes=%llu decoded_bytes=%llu bytes_per_arena=%zu image_batch=%zu\n",
                int(gemm_groups==prefill_groups),std::count_if(workspace.tiles.begin(),workspace.tiles.end(),
                    [](const auto& tile){return tile.compact;}),workspace.tiles.size(),
                static_cast<unsigned long long>(workspace.activation_bytes),
                static_cast<unsigned long long>(workspace.decoded_bytes),image_batch*image_per_expert,image_batch);
        }
        auto staging=gate->cache==up->cache && gate->cache==down->cache?
            gate->cache->prepare_prefill_staging({gate->source_id,up->source_id,down->source_id},
                {unique.data(),prefill_groups},group_ends,source.get_device()):std::shared_ptr<void>{};
        if(std::getenv("TYLOINFER_TQ_PROFILE_CACHE"))std::fprintf(stderr,
            "tq_prefill_moe jobs=%zu gemm_jobs=%zu scratch=%lld decoded=%lld image_batch=%zu\n",
            prefill_groups,gemm_groups,static_cast<long long>(activations.numel()),
            static_cast<long long>(decoded.defined()?decoded.numel():0),image_batch);
        const auto acquire=[&](std::size_t group) {
            const auto begin=group?group_ends[group-1]:0;
            const auto count=group_ends[group]-begin;
            auto& stage=pipeline.stages[group%pipeline.stages.size()];
            const MfqCudaStreamGuard transfer(pipeline.upload);
            // Cache intervals remain leased until their reader dependency has
            // been published on the stream that may overwrite them.
            for(auto& bank:stage.banks)bank.clear();
            stage.workspace.cursor=0;
            stage.banks=phase("tq.moe.weights",[&] {
                std::array<std::vector<TqWeight>,3> banks;
                banks[0]=
                    gate->cache->get_rows(gate->source_id,unique.data()+begin,count,*gate->table,gate->out,source.get_device(),
                        nullptr,mfq::cuda::RoutedImageUse::Eager,nullptr,&stage.workspace);
                banks[1]=
                    up->cache->get_rows(up->source_id,unique.data()+begin,count,*up->table,up->out,source.get_device(),
                        nullptr,mfq::cuda::RoutedImageUse::Eager,nullptr,&stage.workspace);
                pipeline.schedule.gate_up_ready[group%pipeline.stages.size()].record(pipeline.upload.stream());
                banks[2]=
                    down->cache->get_rows(down->source_id,unique.data()+begin,count,*down->table,down->out,source.get_device(),
                        nullptr,mfq::cuda::RoutedImageUse::Eager,nullptr,&stage.workspace);
                return banks;
            });
            for(std::size_t i=0;i<count;++i) {
                const auto index=begin+i;
                const TrellisPrefillMoeJob job{stage.banks[0][i].view,stage.banks[1][i].view,stage.banks[2][i].view,
                    static_cast<int64_t>(slot_offsets[index]),static_cast<int>(slots[index].size())};
                std::memcpy(host_bytes+index*sizeof(job),&job,sizeof(job));
            }
        };
        pipeline.schedule.run(group_ends.size(),acquire,[&](std::size_t group) {
            const auto begin=group?group_ends[group-1]:0;
            const auto count=group_ends[group]-begin;
            auto& stage=pipeline.stages[group%pipeline.stages.size()];
            const auto offset=begin*sizeof(TrellisPrefillMoeJob);
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(device_bytes+offset,host_bytes+offset,count*sizeof(TrellisPrefillMoeJob),
                cudaMemcpyHostToDevice,stream.stream()));
            pipeline.schedule.join(group,true);
            const auto first_compute=group?workspace.group_tile_ends[group-1]:0;
            for(auto compute=first_compute;compute<workspace.group_tile_ends[group];++compute) {
                const auto& compute_tile=workspace.tiles[compute];
                const auto tile=compute_tile.begin,tile_count=compute_tile.end-tile;
                const auto capacity=compute_tile.capacity;
                const bool gemm=compute_tile.gemm;
                const auto* plan=reinterpret_cast<const TrellisPrefillMoeJob*>(device_bytes+tile*sizeof(TrellisPrefillMoeJob));
                const auto* device_slots=reinterpret_cast<const int64_t*>(device_bytes+job_bytes);
                const auto* device_bounds=reinterpret_cast<const int64_t*>(device_bytes+bounds_offset)+tile;
                const auto down_ready=pipeline.schedule.ready[group%pipeline.stages.size()].get();
                if(gemm)phase("tq.moe.prefill_gemm",[&] {
                    // The common compact product uses SM70 FP16 WMMA. cuBLAS
                    // retains the established path on earlier instruction sets.
                    if(compute_tile.compact)status(mfq_tq_prefill_moe_compact(plan,device_slots,device_bounds,
                        source.data_ptr(),pairs.data_ptr(),static_cast<int>(tile_count),static_cast<int>(capacity),
                        slot_offsets[tile],static_cast<int64_t>(compute_tile.rows),
                        static_cast<int>(up->input),static_cast<int>(hidden),static_cast<int>(width),static_cast<int>(routes),limit,
                        decoded.data_ptr(),decoded.nbytes(),activations.data_ptr(),activations.nbytes(),down_ready,stream.stream()));
                    else status(mfq_tq_prefill_moe_gemm_staged(plan,device_slots,source.data_ptr(),pairs.data_ptr(),
                        static_cast<int>(tile_count),static_cast<int>(capacity),static_cast<int>(up->input),static_cast<int>(hidden),
                        static_cast<int>(routes),limit,decoded.data_ptr(),decoded.nbytes(),activations.data_ptr(),activations.nbytes(),
                        blas,down_ready,stream.stream()));
                    return 0;
                });
                else phase("tq.moe.prefill",[&] {status(mfq_tq_prefill_moe_staged(plan,device_slots,
                    source.data_ptr(),pairs.data_ptr(),static_cast<int>(tile_count),static_cast<int>(capacity),
                    static_cast<int>(up->input),static_cast<int>(hidden),static_cast<int>(routes),limit,
                    activations.data_ptr(),activations.nbytes(),down_ready,stream.stream()));return 0;});
            }
        });
    }
    if(prefill_groups<long_groups) {
        // Long prefill gathers one expert's rows and uses the ordinary
        // multi-token/GEMM kernels. Only one compressed expert and bounded
        // decoded tiles need to be live, regardless of the prompt length.
        for(std::size_t index=prefill_groups;index<long_groups;++index) {
            const auto expert=unique[index];
            const auto g=gate->cache->get_rows(gate->source_id,&expert,1,*gate->table,gate->out,source.get_device());
            const auto u=up->cache->get_rows(up->source_id,&expert,1,*up->table,up->out,source.get_device());
            const auto d=down->cache->get_rows(down->source_id,&expert,1,*down->table,down->out,source.get_device());
            const TqExpertReadFence reads({&g,&u,&d},source.get_device());
            std::vector<std::int64_t> token_ids;token_ids.reserve(slots[index].size());
            for(const auto slot:slots[index])token_ids.push_back(slot/routes);
            const auto token_index=tb::from_blob(token_ids.data(),{static_cast<std::int64_t>(token_ids.size())},
                tb::TensorOptions().dtype(tb::kInt64)).clone().to(source.device(),false,false);
            const auto slot_index=tb::from_blob(slots[index].data(),{static_cast<std::int64_t>(slots[index].size())},
                tb::TensorOptions().dtype(tb::kInt64)).clone().to(source.device(),false,false);
            const auto gathered=source.index_select(0,token_index);
            tb::Tensor activated;
            if(limit>0) {
                const auto gv=g[0].forward(gathered),uv=u[0].forward(gathered);
                if(type==0)activated=clipped_silu_mul_cuda(gv,uv,limit);
                else activated=(tb::silu(tb::clamp_max(gv.to(tb::kFloat32),limit))*
                    tb::clamp(uv.to(tb::kFloat32),-limit,limit)).to(source.scalar_type()).contiguous();
            }else activated=TqProjectionGroup({g[0],u[0]}).swiglu(gathered);
            flat_pairs.index_copy_(0,slot_index,d[0].forward(std::move(activated)));
        }
    }
    if(long_groups<unique.size()) {
        auto activated=tb::empty({tokens,routes,hidden},source.options());
        // The live batch is bounded by current free VRAM, not a device or
        // model whitelist. FP16-sized images conservatively bound every
        // supported 1..8-bit compressed row plus metadata and scale data.
        const auto image_bound=[](const TqRoutedWeight& w) {
            return std::uint64_t(w.out)*(((std::uint64_t(w.input)+255)&~std::uint64_t(255))*2+16)+512;
        };
        const auto per_expert=image_bound(*gate)+image_bound(*up)+image_bound(*down);
        for(std::size_t begin=long_groups;begin<unique.size();) {
            const auto free=phase("tq.moe.capacity",[&] {
                std::size_t available=0,total=0;TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&available,&total));return available;
            });
            const auto count=std::min<std::size_t>(std::min<std::size_t>(65535,unique.size()-begin),
                std::max<std::uint64_t>(1,(free/2)/per_expert));
            const auto banks=phase("tq.moe.weights",[&] {
                return std::array<std::vector<TqWeight>,3>{
                    gate->cache->get_rows(gate->source_id,unique.data()+begin,count,*gate->table,gate->out,source.get_device()),
                    up->cache->get_rows(up->source_id,unique.data()+begin,count,*up->table,up->out,source.get_device()),
                    down->cache->get_rows(down->source_id,unique.data()+begin,count,*down->table,down->out,source.get_device())};
            });
            const auto& g=banks[0];const auto& u=banks[1];const auto& d=banks[2];
            const TqExpertReadFence reads({&g,&u,&d},source.get_device());
            if(tokens==1 && unique.size()==static_cast<std::size_t>(routes)) {
                // Preserve the tuned single-vector geometry. Each selected
                // expert gets one slot, with no multi-token dummy work.
                const auto tables=phase("tq.moe.views",[&] {
                    return std::array<tb::Tensor,3>{
                        gate->cache->projection_table(gate->source_id,gate->experts,unique.data()+begin,g,source.get_device()),
                        up->cache->projection_table(up->source_id,up->experts,unique.data()+begin,u,source.get_device()),
                        down->cache->projection_table(down->source_id,down->experts,unique.data()+begin,d,source.get_device())};
                });
                const auto& gp=tables[0];const auto& upp=tables[1];const auto& dp=tables[2];
                const auto route_ids=ids.to(source.device()).to(tb::kInt64).contiguous();
                const auto element=source.element_size();
                auto* h=static_cast<std::uint8_t*>(activated.data_ptr())+begin*hidden*element;
                auto* p=static_cast<std::uint8_t*>(pairs.data_ptr())+begin*width*element;
                phase("tq.moe.gu",[&] {status(mfq_tq_routed_swiglu(reinterpret_cast<const TrellisProjectionView*>(gp.data_ptr()),
                    reinterpret_cast<const TrellisProjectionView*>(upp.data_ptr()),route_ids.data_ptr<int64_t>()+begin,
                    source.data_ptr(),h,static_cast<int>(count),static_cast<int>(hidden),type,limit,mfq_current_cuda_stream()));return 0;});
                phase("tq.moe.down",[&] {status(mfq_tq_routed_down(reinterpret_cast<const TrellisProjectionView*>(dp.data_ptr()),
                    route_ids.data_ptr<int64_t>()+begin,h,p,static_cast<int>(count),static_cast<int>(width),
                    static_cast<int>(hidden),type,mfq_current_cuda_stream()));return 0;});
            }else {
                auto [gu_tiles,down_tiles,wide_tiles,wide_gu]=tq_indexed_jobs(g,u,d,slots,begin,routes,up->input,hidden,width,
                    type==0 && g.front().gemm->indexed_mma());
                std::vector<TrellisIndexedProjectionJob> gu_jobs,down_jobs;
                std::vector<TrellisIndexedWideProjectionJob> wide_jobs;
                for(int tile=0;tile<4;++tile) {
                    gu_jobs.insert(gu_jobs.end(),gu_tiles[tile].begin(),gu_tiles[tile].end());
                    down_jobs.insert(down_jobs.end(),down_tiles[tile].begin(),down_tiles[tile].end());
                    wide_jobs.insert(wide_jobs.end(),wide_tiles[tile].begin(),wide_tiles[tile].end());
                }
                if(gu_jobs.size()>INT_MAX)throw std::overflow_error("too many TQ indexed jobs");
                const auto gp=upload_plan(gu_jobs),dp=upload_plan(down_jobs),wp=upload_plan(wide_jobs),gwp=upload_plan(wide_gu);
                const auto* gate_plan=gp.defined()?reinterpret_cast<const TrellisIndexedProjectionJob*>(gp.data_ptr()):nullptr;
                const auto* down_plan=dp.defined()?reinterpret_cast<const TrellisIndexedProjectionJob*>(dp.data_ptr()):nullptr;
                const auto* wide_plan=wp.defined()?reinterpret_cast<const TrellisIndexedWideProjectionJob*>(wp.data_ptr()):nullptr;
                std::size_t first=0;
                for(int tile=0;tile<4;++tile) {
                    const auto jobs=static_cast<int>(gu_tiles[tile].size());
                    if(!jobs)continue;
                    status(mfq_tq_indexed_swiglu(gate_plan+first,source.data_ptr(),activated.data_ptr(),jobs,
                        static_cast<int>(hidden),type,tile+1,limit,mfq_current_cuda_stream()));
                    first+=jobs;
                }
                if(!wide_gu.empty())status(mfq_tq_indexed_wide_swiglu(
                    reinterpret_cast<const TrellisIndexedWidePairJob*>(gwp.data_ptr()),source.data_ptr(),activated.data_ptr(),
                    static_cast<int>(wide_gu.size()),static_cast<int>(hidden),type,8,limit,mfq_current_cuda_stream()));
                first=0;
                for(int tile=0;tile<4;++tile) {
                    const auto jobs=static_cast<int>(down_tiles[tile].size());if(!jobs)continue;
                    status(mfq_tq_indexed(down_plan+first,activated.data_ptr(),pairs.data_ptr(),jobs,
                        static_cast<int>(width),type,tile+1,mfq_current_cuda_stream()));first+=jobs;
                }
                first=0;
                for(int tile=0;tile<4;++tile) {
                    const auto jobs=static_cast<int>(wide_tiles[tile].size());if(!jobs)continue;
                    status(mfq_tq_indexed_wide(wide_plan+first,activated.data_ptr(),pairs.data_ptr(),jobs,
                        static_cast<int>(width),type,tile+5,mfq_current_cuda_stream()));first+=jobs;
                }
            }
            begin+=count;
        }
    }
    phase("tq.moe.reduce",[&] {status(mfq_tq_weighted_reduce(pairs.data_ptr(),coefficients.data_ptr<float>(),result.data_ptr(),
        static_cast<int>(tokens),static_cast<int>(routes),static_cast<int>(width),type,mfq_current_cuda_stream()));return 0;});
    return result;
}

tb::Tensor TqMoeBanks::forward_parallel(CudaExecutionContext& execution,tb::Tensor input,
        tb::Tensor ids,tb::Tensor weights,float limit,const std::function<void()>& independent_work,CudaIndependentBranchExecutor* shared_executor)const {
    const auto& parallel=moe_parallel_config(execution);
    if(!parallel.enabled())return forward(std::move(input),std::move(ids),std::move(weights),limit,&execution.profiler,0,-1,independent_work,shared_executor);
    if(independent_work)independent_work();
    if(!gate)throw std::invalid_argument("TQ expert parallelism requires a Gate bank");
    const auto slices=plan_moe_expert_parallel_slices(parallel,gate->experts,"TQ experts");
    std::vector<tb::Tensor> partials(slices.size());
    for(std::size_t launch=0;launch<slices.size();++launch) {
        const auto index=model_parallel_launch_index(execution.config,launch,slices.size());
        const auto& slice=slices[index];
        const MfqCudaGuard guard(slice.device);
        auto local_input=tensor_to_cuda_device(execution.model_parallel_collectives,input,slice.device);
        auto local_ids=tensor_to_cuda_device(execution.model_parallel_collectives,ids,slice.device);
        auto local_weights=tensor_to_cuda_device(execution.model_parallel_collectives,weights,slice.device);
        partials[index]=forward(std::move(local_input),std::move(local_ids),std::move(local_weights),
            limit,&execution.profiler,slice.begin,slice.end);
    }
    return reduce_model_parallel_outputs(execution,std::move(partials));
}
