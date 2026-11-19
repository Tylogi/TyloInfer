#include "prefill_projection.h"
#include "runtime/prefill_projection_plan.h"
#include "kernels/gguf.h"
#include "kernels/gguf_mmq.h"
#include "mfq_cuda_context.h"
#include "mfq_cuda_resident_allocations.h"
#include "tyloinfer/runtime.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <limits>

namespace tb=mfq_tensor_backend;
using mfq::cuda::prefill_weight_rows;

PrefillProjectionWeight prefill_decoded_projection(const tb::Tensor& storage,
        std::int64_t rows,std::int64_t columns,std::function<tb::Tensor()> decode) {
    if(!storage.is_cuda() || !decode || rows<1 || columns<1 ||
            std::uint64_t(columns)>std::numeric_limits<std::uint64_t>::max()/6 ||
            rows>std::numeric_limits<std::int64_t>::max()/columns)return {};
    // Both the decoder's FP32 matrix and the normalized FP16 image are live
    // during packing. Price that peak before invoking the format decoder.
    auto reconstruct=[storage,rows,columns,decode=std::move(decode)] {
            const MfqCudaGuard guard(storage.get_device());
            auto decoded=decode();
            if(!decoded.is_cuda() || decoded.device()!=storage.device() || !decoded.is_contiguous() ||
                    decoded.scalar_type()!=tb::kFloat32 || decoded.dim()!=2 ||
                    decoded.size(0)!=rows || decoded.size(1)!=columns)
                throw std::runtime_error("prefill weight decoder must produce its contiguous FP32 matrix");
            return decoded;
        };
    return {storage,rows,columns,std::uint64_t(columns)*6,
        [storage,rows,columns,reconstruct] {
            const MfqCudaGuard guard(storage.get_device());
            auto decoded=reconstruct();
            auto image=tb::empty({rows,columns},storage.options().dtype(tb::kFloat16));
            auto exponents=tb::empty({rows},storage.options().dtype(tb::kInt32));
            const auto stream=mfq_get_current_cuda_stream(storage.get_device());
            for(const auto& tensor:{storage,decoded,image,exponents})mfq_cuda_record_stream(tensor,stream);
            mfq::cuda::gguf::pack_scaled_f16_unit(decoded.data_ptr<float>(),image.data_ptr(),
                exponents.data_ptr<std::int32_t>(),rows,columns,stream.stream());
            return PrefillWeightImage{image,exponents};
        },[storage,reconstruct] {
            const MfqCudaGuard guard(storage.get_device());
            return reconstruct().to(tb::kFloat16).contiguous();
        }};
}

bool can_prefill_residual(const tb::Tensor& storage,std::int64_t rows,std::int64_t columns,
        std::int64_t tokens,const tb::Tensor& residual) {
    const auto* setting=mfq::cuda::prefill_tensorcore_setting();
    if(tokens<=32 || (setting && setting[0]!='1' && setting[0]!='a') ||
            ((!setting || setting[0]=='a') && tokens<512) || !storage.is_cuda() || rows<1 || columns<1 ||
            rows>std::numeric_limits<int>::max() || columns>std::numeric_limits<int>::max() ||
            tokens>std::numeric_limits<int>::max() ||
            !residual.is_cuda() || residual.device()!=storage.device() || !residual.is_contiguous() ||
            residual.scalar_type()!=tb::kFloat16 || residual.dim()<1 || residual.size(-1)!=rows ||
            residual.numel()/rows!=tokens)return false;
    const MfqCudaGuard guard(storage.get_device());
    return mfq::cuda::gguf::grouped_f16_available(storage.get_device());
}

bool can_prefill_residual(const PrefillProjectionWeight& weight,std::int64_t tokens,
        const tb::Tensor& residual) {
    if(!weight || !weight.decode_f16 || !weight.decode_row_bytes ||
        !can_prefill_residual(weight.storage,weight.rows,weight.columns,tokens,residual) ||
        weight.decode_row_bytes>std::numeric_limits<std::uint64_t>::max()-std::uint64_t(tokens)*2)return false;
    const MfqCudaGuard guard(weight.storage.get_device());
    return prefill_weight_rows(weight.rows,weight.decode_row_bytes+std::uint64_t(tokens)*2,32,
        weight.storage.get_device())>=weight.rows;
}

void prefill_residual_gemm(const tb::Tensor& input,const tb::Tensor& weight,
        const tb::Tensor& residual,const tb::Tensor& output,std::int64_t first,
        std::int64_t count,void* blas,tb::Tensor& workspace) {
    TYLOINFER_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() && input.dim()==2 &&
        input.scalar_type()==tb::kFloat16 && input.size(0)>0 && input.size(1)>0 &&
        weight.device()==input.device() && weight.is_contiguous() && weight.scalar_type()==tb::kFloat16 &&
        weight.dim()==2 && weight.size(1)==input.size(1) && count>0 && count<=weight.size(0) &&
        output.device()==input.device() && output.is_contiguous() && output.dim()==2 &&
        output.scalar_type()==tb::kFloat16 && output.size(0)==input.size(0) &&
        first>=0 && first<=output.size(1) && count<=output.size(1)-first &&
        residual.device()==input.device() && residual.is_contiguous() && residual.scalar_type()==tb::kFloat16 &&
        residual.numel()==output.numel() && residual.dim()>0 && residual.size(-1)==output.size(1) &&
        input.size(0)<=std::numeric_limits<int>::max() && input.size(1)<=std::numeric_limits<int>::max() &&
        output.size(1)<=std::numeric_limits<int>::max() && blas,
        "residual prefill requires compatible contiguous FP16 operands and an in-bounds output tile");
    const MfqCudaGuard guard(input.get_device());
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    for(const auto& tensor:{input,weight,residual,output})mfq_cuda_record_stream(tensor,stream);
    const auto* addend=static_cast<const std::uint8_t*>(residual.data_ptr())+first*2;
    auto* destination=static_cast<std::uint8_t*>(output.data_ptr())+first*2;
    auto plan=mfq::cuda::gguf::prepare_prefill_residual_gemm(input.data_ptr(),weight.data_ptr(),addend,
        destination,input.size(0),count,input.size(1),output.size(1),stream.stream());
    if(workspace.defined() && (workspace.device()!=input.device() ||
            workspace.scalar_type()!=tb::kUInt8 || !workspace.is_contiguous()))workspace={};
    if(plan.workspace_bytes && (!workspace.defined() || workspace.nbytes()<plan.workspace_bytes)) {
        try {
            workspace=tb::empty({std::int64_t(plan.workspace_bytes)},input.options().dtype(tb::kUInt8));
        } catch(const MfqResourceExhausted&) {
            (void)cudaGetLastError();
            plan={};
        }
    }
    if(workspace.defined())mfq_cuda_record_stream(workspace,stream);
    mfq::cuda::gguf::prefill_gemm_f16_residual(input.data_ptr(),weight.data_ptr(),addend,destination,
        input.size(0),count,input.size(1),output.size(1),plan,
        plan && workspace.defined()?workspace.data_ptr():nullptr,
        plan && workspace.defined()?workspace.nbytes():0,blas,stream.stream());
}

tb::Tensor prefill_residual_projection(const PrefillProjectionWeight& weight,
        const tb::Tensor& input,const tb::Tensor& residual) {
    if(!weight || !input.is_cuda() || !input.is_contiguous() || input.dim()<1 || input.size(-1)!=weight.columns ||
        input.scalar_type()!=tb::kFloat16 || input.device()!=weight.storage.device() || weight.columns<1 ||
        !can_prefill_residual(weight,input.numel()/weight.columns,residual))return {};
    const MfqCudaGuard guard(input.get_device());
    auto decoded=weight.decode_f16();
    TYLOINFER_RUNTIME_CHECK(decoded.is_cuda() && decoded.device()==input.device() && decoded.is_contiguous() &&
        decoded.scalar_type()==tb::kFloat16 && decoded.dim()==2 && decoded.size(0)==weight.rows &&
        decoded.size(1)==weight.columns,"residual prefill decoder must produce its contiguous FP16 matrix");
    auto output=tb::empty({input.numel()/weight.columns,weight.rows},input.options());
    tb::Tensor workspace;
    auto& blas=mfq::cuda::default_context(input.get_device())->blas(mfq_current_cuda_stream());
    prefill_residual_gemm(input.reshape({-1,weight.columns}),decoded,residual,output,0,weight.rows,blas.get(),workspace);
    auto shape=input.sizes().vec();shape.back()=weight.rows;
    return output.reshape(shape);
}

tb::Tensor prefill_weight_exponents(const tb::Tensor& storage,std::int64_t rows,std::int64_t columns,
        const std::vector<std::uintptr_t>& layout,const std::function<void(const tb::Tensor&)>& prepare) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    struct Entry {
        std::weak_ptr<void> owner;const void* data=nullptr;int device=-1;
        std::int64_t rows=0,columns=0;std::vector<std::uintptr_t> layout;
        tb::Tensor exponents;mfq::cuda::Event ready;cudaStream_t producer=nullptr;
    };
    static std::mutex mutex;static std::vector<std::unique_ptr<Entry>> cache;
    const MfqCudaGuard guard(storage.get_device());
    const auto owner=storage.weak_storage_owner();const auto stream=mfq_get_current_cuda_stream(storage.get_device());
    std::lock_guard lock(mutex);
    for(auto i=cache.begin();i!=cache.end();) {
        if((*i)->owner.expired()){i=cache.erase(i);continue;}
        auto& entry=**i;++i;
        if(!entry.owner.owner_before(owner) && !owner.owner_before(entry.owner) &&
                entry.data==storage.data_ptr() && entry.device==storage.get_device() &&
                entry.rows==rows && entry.columns==columns && entry.layout==layout) {
            if(stream.stream()!=entry.producer && !entry.ready.ready()) {
                cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
                TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(),&capture));
                if(capture!=cudaStreamCaptureStatusNone)return {};
                TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),entry.ready.get(),0));
            }
            mfq_cuda_record_stream(entry.exponents,stream);return entry.exponents;
        }
    }
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(),&capture));
    if(capture!=cudaStreamCaptureStatusNone)return {};
    auto entry=std::make_unique<Entry>();entry->owner=owner;entry->data=storage.data_ptr();
    entry->device=storage.get_device();entry->rows=rows;entry->columns=columns;entry->layout=layout;
    {
        // Cached row metadata outlives the request's borrowed workspace.
        const mfq::cuda::ResidentAllocationScope resident;
        entry->exponents=tb::empty({rows},storage.options().dtype(tb::kInt32));
    }
    mfq_cuda_record_stream(storage,stream);mfq_cuda_record_stream(entry->exponents,stream);
    prepare(entry->exponents);entry->ready.record(stream.stream());entry->producer=stream.stream();
    auto result=entry->exponents;cache.push_back(std::move(entry));return result;
#else
    return {};
#endif
}

struct PrefillInputScope::Impl {
    static thread_local Impl* active;Impl* previous=nullptr;
    tb::Tensor source,half,exponents;
    std::optional<mfq::cuda::Event> ready;
    MmqPrefillInput mmq;
    cudaStream_t mmq_stream=nullptr;
    int mmq_layout=-1;
    bool same_input(const tb::Tensor& input) const {
        return input.defined() && input.dim()>=1 && input.is_contiguous() &&
            input.scalar_type()==source.scalar_type() && input.device()==source.device() &&
            input.data_ptr()==source.data_ptr() && input.numel()==source.numel() &&
            input.size(-1)==source.size(-1);
    }
};
thread_local PrefillInputScope::Impl* PrefillInputScope::Impl::active=nullptr;
PrefillInputScope::PrefillInputScope()=default;
std::unique_ptr<PrefillInputScope> PrefillInputScope::prepare_rms(
        const tb::Tensor& input,const tb::Tensor& weight,double eps,double offset,tb::Tensor& normalized) {
    normalized={};const auto* setting=mfq::cuda::prefill_tensorcore_setting();
    if(!input.is_cuda() || !input.is_contiguous() || input.dim()<1 || input.size(-1)<1 ||
            input.numel()/input.size(-1)<512 ||
            (input.scalar_type()!=tb::kFloat32 && input.scalar_type()!=tb::kFloat16 && input.scalar_type()!=tb::kBFloat16) ||
            weight.device()!=input.device() || weight.scalar_type()!=tb::kFloat32 || !weight.is_contiguous() ||
            weight.numel()!=input.size(-1) || !std::isfinite(eps) || eps<=0 || !std::isfinite(offset) ||
            (setting && setting[0]!='1' && setting[0]!='a'))return {};
    const MfqCudaGuard guard(input.get_device());const auto rows=input.numel()/input.size(-1),width=input.size(-1);
    if(!mfq::cuda::gguf::grouped_f16_available(input.get_device()) ||
            !mfq::cuda::gguf::rms_prefill_available(rows,width,input.get_device()))return {};
    auto scope=std::unique_ptr<PrefillInputScope>(new PrefillInputScope);
    auto plan=std::make_unique<Impl>();
    normalized=tb::empty(input.sizes(),input.options().dtype(tb::kFloat32));plan->source=normalized;
    plan->half=tb::empty(input.sizes(),input.options().dtype(tb::kFloat16));
    plan->exponents=tb::empty({rows},input.options().dtype(tb::kInt32));
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    for(const auto& tensor:{input,weight,normalized,plan->half,plan->exponents})mfq_cuda_record_stream(tensor,stream);
    const auto type=input.scalar_type()==tb::kFloat32?GGML_TYPE_F32:input.scalar_type()==tb::kFloat16?GGML_TYPE_F16:GGML_TYPE_BF16;
    mfq::cuda::gguf::pack_rms_prefill(input.data_ptr(),type,weight.data_ptr<float>(),rows,width,float(eps),float(offset),
        normalized.data_ptr<float>(),plan->half.data_ptr(),plan->exponents.data_ptr<int32_t>(),stream.stream());
    plan->ready.emplace().record(stream.stream());plan->previous=Impl::active;Impl::active=plan.get();scope->impl_=std::move(plan);
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true))std::fprintf(stderr,"native_prefill_rms_input actual_match=1 M=%lld K=%lld normalized_fp32=1 reuse_scaled_fp16=1\n",
            static_cast<long long>(rows),static_cast<long long>(width));
    }
    return scope;
}
PrefillInputScope::PrefillInputScope(const tb::Tensor& input) {
    const auto* setting=mfq::cuda::prefill_tensorcore_setting();
    const bool mmq=setting && setting[0]=='0';
    if(!input.is_cuda() || !input.is_contiguous() || input.dim()<1 ||
            input.size(-1)<1 || input.numel()/input.size(-1)<512 ||
            (input.scalar_type()!=tb::kFloat32 && !(mmq && input.scalar_type()==tb::kFloat16)) ||
            (!mmq && setting && setting[0]!='1' && setting[0]!='a'))return;
    const MfqCudaGuard guard(input.get_device());
    if(!mmq && !mfq::cuda::gguf::grouped_f16_available(input.get_device()))return;
    if(Impl::active && Impl::active->same_input(input))return;
    auto plan=std::make_unique<Impl>();plan->source=input;
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    if(mmq)plan->mmq_stream=stream.stream();
    else {
        plan->half=tb::empty(input.sizes(),input.options().dtype(tb::kFloat16));
        plan->exponents=tb::empty({input.numel()/input.size(-1)},input.options().dtype(tb::kInt32));
        for(const auto& tensor:{input,plan->half,plan->exponents})mfq_cuda_record_stream(tensor,stream);
        mfq::cuda::gguf::pack_scaled_f16_unit(input.data_ptr<float>(),plan->half.data_ptr(),plan->exponents.data_ptr<int32_t>(),
            input.numel()/input.size(-1),input.size(-1),stream.stream());
        plan->ready.emplace().record(stream.stream());
    }
    plan->previous=Impl::active;Impl::active=plan.get();impl_=std::move(plan);
}
PrefillInputScope::~PrefillInputScope(){if(impl_)Impl::active=impl_->previous;}
bool PrefillInputScope::lookup(const tb::Tensor& input,tb::Tensor& half,tb::Tensor& exponents) {
    const auto* plan=Impl::active;
    if(!plan || !plan->ready || !plan->same_input(input))return false;
    const MfqCudaGuard guard(input.get_device());
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),plan->ready->get(),0));
    half=plan->half;exponents=plan->exponents;
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "native_prefill_shared_input actual_match=1 M=%lld K=%lld reuse_scaled_fp16=1\n",
            static_cast<long long>(input.numel()/input.size(-1)),static_cast<long long>(input.size(-1)));
    }
    mfq_cuda_record_stream(half,stream);mfq_cuda_record_stream(exponents,stream);return true;
}
MmqPrefillInput PrefillInputScope::prepare_mmq(const tb::Tensor& input,int type) {
    namespace mmq=mfq::cuda::gguf::mmq;
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    const auto width=input.size(-1),rows=input.numel()/width;
    auto* plan=Impl::active;
    // Reuse only on the producing stream. Another layout replaces the one
    // cached image before allocating, keeping group scratch independent of size.
    const bool reuse=plan && !plan->ready && plan->mmq_stream==stream.stream() && plan->same_input(input);
    const auto layout=reuse?mmq::activation_layout(type):-1;
    if(reuse) {
        if(plan->mmq_layout==layout && plan->mmq.activation.defined()) {
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported{};
                if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                    "native_prefill_mmq_shared_input actual_match=1 M=%lld K=%lld layout=%d\n",
                    static_cast<long long>(rows),static_cast<long long>(width),layout);
            }
            return plan->mmq;
        }
        plan->mmq={};
    }
    const bool fp16=input.scalar_type()==tb::kFloat16 && reinterpret_cast<std::uintptr_t>(input.data_ptr())%8==0;
    auto source=fp16?input.contiguous():input.to(tb::kFloat32).contiguous();
    mfq_cuda_record_stream(source,stream);
    MmqPrefillInput result;
    tb::Tensor scaled;
    if(input.scalar_type()==tb::kFloat32) {
        scaled=tb::empty_like(source);
        result.exponents=tb::empty({rows},input.options().dtype(tb::kInt32));
        mfq_cuda_record_stream(scaled,stream);
        mfq_cuda_record_stream(result.exponents,stream);
        mfq::cuda::gguf::pack_scaled_f32(source.data_ptr<float>(),scaled.data_ptr<float>(),
            result.exponents.data_ptr<std::int32_t>(),rows,width,stream.stream());
    }
    result.activation=tb::empty({std::int64_t(mmq::activation_bytes(rows,width))},input.options().dtype(tb::kUInt8));
    mfq_cuda_record_stream(result.activation,stream);
    if(fp16)mmq::quantize_f16(source.data_ptr(),result.activation.data_ptr(),type,width,rows,stream.stream());
    else mmq::quantize((scaled.defined()?scaled:source).data_ptr<float>(),nullptr,result.activation.data_ptr(),type,width,rows,stream.stream());
    if(reuse){plan->mmq=result;plan->mmq_layout=layout;}
    return result;
}
PrefillProduct prefill_product(const PrefillProjectionWeight& w,const tb::Tensor& input) {
    const auto* setting=mfq::cuda::prefill_tensorcore_setting();
    if((setting && setting[0]!='1' && setting[0]!='a') || !w ||
            !w.storage.is_cuda() || !input.is_cuda() || input.device()!=w.storage.device() ||
            input.scalar_type()!=tb::kFloat32 || !input.is_contiguous() || input.dim()<1 ||
            input.size(-1)!=w.columns || w.columns<1 || w.rows<1 || input.numel()/w.columns<512)return {};
    const MfqCudaGuard guard(w.storage.get_device());
    if(!mfq::cuda::gguf::grouped_f16_available(w.storage.get_device()))return {};
    const auto tokens=input.numel()/w.columns;
    // The shared GEMM and row packer use CUDA/cuBLAS 32-bit launch dimensions.
    if(w.rows>std::numeric_limits<int>::max() || w.columns>std::numeric_limits<int>::max() ||
            tokens>std::numeric_limits<int>::max())return {};
    if(w.decode_row_bytes>std::numeric_limits<std::uint64_t>::max()-std::uint64_t(tokens)*2)return {};
    if(prefill_weight_rows(w.rows,w.decode_row_bytes+uint64_t(tokens)*2,32,w.storage.get_device())<w.rows)return {};
    auto decoded=w.decode();if(!decoded.defined())return {};
    auto columns=decoded.column_exponents;
    tb::Tensor packed,rows;
    const auto stream=mfq_get_current_cuda_stream(w.storage.get_device());
    if(!PrefillInputScope::lookup(input,packed,rows)) {
        packed=tb::empty({tokens,w.columns},input.options().dtype(tb::kFloat16));
        rows=tb::empty({tokens},input.options().dtype(tb::kInt32));
        mfq_cuda_record_stream(input,stream);mfq_cuda_record_stream(packed,stream);mfq_cuda_record_stream(rows,stream);
        mfq::cuda::gguf::pack_scaled_f16_unit(input.data_ptr<float>(),packed.data_ptr(),rows.data_ptr<int32_t>(),
            tokens,w.columns,stream.stream());
    }
    auto image=decoded.values;
    auto product=tb::empty({tokens,w.rows},input.options().dtype(tb::kFloat16));
    for(const auto& tensor:{w.storage,packed,rows,columns,image,product})mfq_cuda_record_stream(tensor,stream);

    auto& blas=mfq::cuda::default_context(w.storage.get_device())->blas(stream.stream());
    mfq::cuda::gguf::prefill_gemm_f16_accum(packed.data_ptr(),image.data_ptr(),nullptr,product.data_ptr(),true,
        nullptr,nullptr,tokens,w.rows,w.columns,w.rows,blas.get(),stream.stream());
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true))std::fprintf(stderr,
            "native_prefill_scaled_product actual_match=1 M=%lld N=%lld K=%lld restore_at_consumer=1\n",
            static_cast<long long>(tokens),static_cast<long long>(w.rows),static_cast<long long>(w.columns));
    }
    return {product,rows,columns};
}

static PrefillProduct rms_gated_product_impl(const PrefillProjectionWeight& w,const tb::Tensor& attended,
        const tb::Tensor& gate,const tb::Tensor& norm,double eps,bool silu,const PrefillProduct* scaled_gate) {
    const auto* setting=mfq::cuda::prefill_tensorcore_setting();
    if((setting && setting[0]!='1' && setting[0]!='a') || !w || !w.storage.is_cuda() ||
            !attended.is_cuda() || attended.device()!=w.storage.device() || !attended.is_contiguous() ||
            attended.scalar_type()!=tb::kFloat32 || attended.dim()!=4 ||
            norm.device()!=attended.device() || norm.scalar_type()!=tb::kFloat32 || !norm.is_contiguous() ||
            gate.device()!=attended.device() || gate.scalar_type()!=(scaled_gate?tb::kFloat16:tb::kFloat32) || gate.dim()!=3)return {};
    if(!std::isfinite(eps) || eps<=0)return {};
    const MfqCudaGuard guard(w.storage.get_device());
    const auto batch=attended.size(0),heads=attended.size(1),tokens=attended.size(2),width=attended.size(3);
    const auto m=batch*tokens;
    if(w.rows<1 || w.rows>std::numeric_limits<int>::max() || w.columns>std::numeric_limits<int>::max() ||
            m>std::numeric_limits<int>::max())return {};
    if(m<512 || heads*width!=w.columns || norm.numel()!=width || gate.size(0)!=batch || gate.size(1)!=tokens ||
            gate.size(2)!=heads*width || gate.stride(2)!=1 || gate.stride(1)<heads*width ||
            gate.stride(0)<tokens*gate.stride(1) || !mfq::cuda::gguf::grouped_f16_available(w.storage.get_device()) ||
            !mfq::cuda::gguf::rms_gated_prefill_available(w.columns,w.storage.get_device()))return {};
    if(scaled_gate) {
        for(const auto* e:{&scaled_gate->row_exponents,&scaled_gate->column_exponents})
            if(!e->defined() || !e->is_contiguous() || e->device()!=attended.device() || e->scalar_type()!=tb::kInt32)return {};
        if(scaled_gate->row_exponents.numel()!=m || scaled_gate->column_exponents.numel()!=w.columns)return {};
    }
    if(w.decode_row_bytes>std::numeric_limits<std::uint64_t>::max()-std::uint64_t(m)*2)return {};
    const auto row_bytes=w.decode_row_bytes+uint64_t(m)*2;
    if(prefill_weight_rows(w.rows,row_bytes,32,w.storage.get_device(),false)<w.rows)return {};
    auto decoded=w.decode();if(!decoded.defined())return {};
    auto exponents=decoded.column_exponents;
    auto image=decoded.values;
    auto packed=tb::empty({m,w.columns},attended.options().dtype(tb::kFloat16));
    auto rows=tb::empty({m},attended.options().dtype(tb::kInt32));
    auto product=tb::empty({m,w.rows},attended.options().dtype(tb::kFloat16));
    const auto stream=mfq_get_current_cuda_stream(w.storage.get_device());
    for(const auto& tensor:{w.storage,attended,gate,norm,exponents,image,packed,rows,product})mfq_cuda_record_stream(tensor,stream);
    if(scaled_gate) {
        mfq_cuda_record_stream(scaled_gate->row_exponents,stream);mfq_cuda_record_stream(scaled_gate->column_exponents,stream);
        mfq::cuda::gguf::pack_rms_scaled_gate_prefill(attended.data_ptr<float>(),gate.data_ptr(),norm.data_ptr<float>(),
            scaled_gate->row_exponents.data_ptr<int32_t>(),scaled_gate->column_exponents.data_ptr<int32_t>(),
            batch,heads,tokens,width,gate.stride(0),gate.stride(1),float(eps),silu,packed.data_ptr(),rows.data_ptr<int32_t>(),stream.stream());
    } else mfq::cuda::gguf::pack_rms_gated_prefill(attended.data_ptr<float>(),gate.data_ptr<float>(),norm.data_ptr<float>(),
            batch,heads,tokens,width,gate.stride(0),gate.stride(1),float(eps),silu,packed.data_ptr(),rows.data_ptr<int32_t>(),stream.stream());

    const auto context=mfq::cuda::default_context(w.storage.get_device());auto& blas=context->blas(stream.stream());
    mfq::cuda::gguf::prefill_gemm_f16_accum(packed.data_ptr(),image.data_ptr(),nullptr,product.data_ptr(),true,
        nullptr,nullptr,m,w.rows,w.columns,w.rows,blas.get(),stream.stream());
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> printed{false};if(!printed.exchange(true))std::fprintf(stderr,
            "native_prefill_rms_gate_pack actual_match=1 M=%lld N=%lld K=%lld fused_rms=1 fused_layout=1 fused_gate=%s fused_scale=1\n",
            static_cast<long long>(m),static_cast<long long>(w.rows),static_cast<long long>(w.columns),silu?"silu":"sigmoid");
    }
    return {product,rows,exponents};
}

PrefillProduct prefill_rms_gated_product(const PrefillProjectionWeight& w,const tb::Tensor& attended,
        const tb::Tensor& gate,const tb::Tensor& norm,double eps,bool silu) {
    return rms_gated_product_impl(w,attended,gate,norm,eps,silu,nullptr);
}
PrefillProduct prefill_rms_gated_product(const PrefillProjectionWeight& w,const tb::Tensor& attended,
        const PrefillProduct& gate,const tb::Tensor& norm,double eps,bool silu) {
    if(!gate.defined() || gate.values.dim()!=2 || !gate.values.is_contiguous() || attended.dim()!=4 ||
        gate.values.size(0)!=attended.size(0)*attended.size(2) ||
        gate.values.size(1)!=attended.size(1)*attended.size(3))return {};
    auto result=rms_gated_product_impl(w,attended,
        gate.values.reshape({attended.size(0),attended.size(2),gate.values.size(1)}),norm,eps,silu,&gate);
    if(result.defined())if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true))std::fprintf(stderr,"native_prefill_scaled_gate actual_match=1 restore_at_consumer=1\n");
    }
    return result;
}
tb::Tensor prefill_materialize(const PrefillProduct& product) {
    TYLOINFER_RUNTIME_CHECK(product.defined() && product.values.is_cuda() && product.values.is_contiguous() &&
        product.values.dim()==2 && product.values.scalar_type()==tb::kFloat16,
        "scaled prefill materialization requires a contiguous FP16 product");
    const auto rows=product.values.size(0),columns=product.values.size(1);
    for(const auto* e:{&product.row_exponents,&product.column_exponents})
        TYLOINFER_RUNTIME_CHECK(e->defined() && e->is_contiguous() && e->device()==product.values.device() && e->scalar_type()==tb::kInt32,
            "scaled prefill materialization exponents disagree");
    TYLOINFER_RUNTIME_CHECK(product.row_exponents.numel()==rows && product.column_exponents.numel()==columns,
        "scaled prefill materialization exponent sizes disagree");
    const MfqCudaGuard guard(product.values.get_device());
    auto output=tb::empty({rows,columns},product.values.options().dtype(tb::kFloat32));
    const auto stream=mfq_get_current_cuda_stream(output.get_device());
    for(const auto& t:{product.values,product.row_exponents,product.column_exponents,output})mfq_cuda_record_stream(t,stream);
    mfq::cuda::gguf::restore_scaled_f16(product.values.data_ptr(),output.data_ptr(),false,
        product.row_exponents.data_ptr<int32_t>(),product.column_exponents.data_ptr<int32_t>(),rows,columns,columns,stream.stream());
    return output;
}

std::vector<tb::Tensor> prefill_acc_rms(const PrefillProduct& product,const tb::Tensor& residual,
        const tb::Tensor& norm,double eps,double offset) {
    TYLOINFER_RUNTIME_CHECK(product.defined() && residual.is_cuda() && residual.is_contiguous() &&
        residual.scalar_type()==tb::kFloat16 && residual.dim()==2 && product.values.is_contiguous() &&
        product.values.device()==residual.device() && product.values.sizes()==residual.sizes() &&
        product.values.scalar_type()==tb::kFloat16 && norm.is_contiguous() && norm.device()==residual.device() &&
        norm.scalar_type()==tb::kFloat32 && norm.numel()==residual.size(1),"scaled prefill residual/RMS operands disagree");
    for(const auto* exponents:{&product.row_exponents,&product.column_exponents})
        TYLOINFER_RUNTIME_CHECK(exponents->defined() && exponents->is_cuda() && exponents->is_contiguous() &&
            exponents->device()==residual.device() && exponents->scalar_type()==tb::kInt32,
            "scaled prefill residual/RMS exponents disagree");
    TYLOINFER_RUNTIME_CHECK(product.row_exponents.numel()==residual.size(0) &&
        product.column_exponents.numel()==residual.size(1) && std::isfinite(eps) && eps>0,
        "scaled prefill residual/RMS exponent geometry or epsilon disagrees");
    const MfqCudaGuard guard(residual.get_device());
    auto sum=tb::empty_like(residual),normalized=tb::empty_like(residual);
    const auto stream=mfq_get_current_cuda_stream(residual.get_device());
    for(const auto& tensor:{residual,norm,product.values,product.row_exponents,product.column_exponents,sum,normalized})
        mfq_cuda_record_stream(tensor,stream);
    mfq::cuda::gguf::acc_rms_scaled_f16(product.values.data_ptr(),residual.data_ptr(),norm.data_ptr<float>(),
        product.row_exponents.data_ptr<int32_t>(),product.column_exponents.data_ptr<int32_t>(),
        residual.size(0),residual.size(1),float(eps),float(offset),sum.data_ptr(),normalized.data_ptr(),stream.stream());
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true))std::fprintf(stderr,"native_prefill_acc_rms actual_match=1 fused_restore=1 fused_residual=1 fused_rms=1\n");
    }
    return {sum,normalized};
}
