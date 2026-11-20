#include "../../kernels/projection/gguf/window.h"
#include "projection.h"
#include "prefill_fusion.h"
#include "rotation.h"
#include "row_correction.h"
#include "components/error.h"
#include "engine/backends/cuda/ops/projection/prefill.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "../../kernels/projection/gguf/linear.h"
#include "../../kernels/projection/gguf/residual/m1_residual.h"
#include "../../kernels/projection/gguf/compact/compact.h"
#include "../../kernels/projection/gguf/mmq/mmq.h"
#include "../../kernels/projection/gguf/quad/storage.h"
#include "../../kernels/projection/gguf/quad/matvec.h"
#include "../../kernels/projection/gguf/quad/small_m.h"
#include "../../kernels/projection/gguf/quad/decode.h"
#include "dense_schedule.h"
#include "engine/weights/encoding/gguf/decode.h"
#include "engine/weights/moe/prepare/projection.h"

#include <stdexcept>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdlib>
#include <atomic>
#include <cstdio>

#include <mutex>


namespace tyloi::cuda::gguf {

bool original_input(const GgmlQuantizedTensor& w) {
    return bool(w.correction) || bool(w.rotation) ||
        (w.type >= GGML_TYPE_TQ2_T && w.type <= GGML_TYPE_TQK4);
}

static bool quad_ready(const GgmlQuantizedTensor& w) {
    return w.quad_storage_rows && w.quad_row_begin % 4 == 0 && w.out % 4 == 0 &&
        w.quad_row_begin + w.out <= w.quad_storage_rows / 4 * 4;
}

static DenseTensor dequantize_stored(const GgmlQuantizedTensor&);
static DenseTensor embedding_base(const GgmlQuantizedTensor&, DenseTensor);
static DenseTensor project_m1(const GgmlQuantizedTensor&, const DenseTensor&, ScalarType);
static DenseTensor project_base(const GgmlQuantizedTensor&, DenseTensor, bool);
static DenseTensor project_base(const GgmlQuantizedTensor&, DenseTensor, bool, const DenseTensor&);
static DenseTensor project_prepared(const ProjectionInput&, const GgmlQuantizedTensor&);

namespace {
// Only row exponents are retained. Weak ownership prevents either model
// retention or reuse after an allocator gives another weight the same address.
struct PrefillRowScale {
    std::weak_ptr<void> owner;const void* data;ggml_type type;int device;
    std::int64_t rows,columns,quad_rows,quad_begin;
    ::tyloi::DenseTensor exponents;tyloi::cuda::Event ready;
};
::tyloi::DenseTensor prefill_cached_exponents(const GgmlQuantizedTensor& w) {
    static std::mutex mutex;static std::vector<std::unique_ptr<PrefillRowScale>> cache;
    const auto owner=w.packed.weak_storage_owner();const auto stream=mfq_get_current_cuda_stream(w.packed.get_device());
    std::lock_guard lock(mutex);
    for(auto i=cache.begin();i!=cache.end();) {
        if((*i)->owner.expired()){i=cache.erase(i);continue;}
        auto& entry=**i;++i;
        if(!entry.owner.owner_before(owner) && !owner.owner_before(entry.owner) &&
                entry.data==w.packed.data_ptr() && entry.type==w.type && entry.rows==w.out &&
                entry.columns==w.neuron_len && entry.quad_rows==w.quad_storage_rows && entry.quad_begin==w.quad_row_begin) {
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),entry.ready.get(),0));
            mfq_cuda_record_stream(entry.exponents,stream);return entry.exponents;
        }
    }
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(),&capture));
    if(capture!=cudaStreamCaptureStatusNone)return {};
    auto entry=std::make_unique<PrefillRowScale>();entry->owner=owner;entry->data=w.packed.data_ptr();entry->type=w.type;
    entry->device=w.packed.get_device();entry->rows=w.out;entry->columns=w.neuron_len;
    entry->quad_rows=w.quad_storage_rows;entry->quad_begin=w.quad_row_begin;
    {
        const ResidentAllocationScope resident;
        entry->exponents=::tyloi::cuda::empty({w.out},w.packed.options().dtype(::tyloi::kInt32));
    }
    mfq_cuda_record_stream(entry->exponents,stream);
    if(w.quad_storage_rows) {
        const auto rows=tyloi::cuda::prefill_weight_rows(w.out,w.row_bytes,128,w.packed.get_device(),false);
        auto image=::tyloi::cuda::empty({std::int64_t(rows*w.row_bytes)},w.packed.options());
        mfq_cuda_record_stream(image,stream);mfq_cuda_record_stream(w.packed,stream);
        for(std::int64_t begin=0;begin<w.out;begin+=rows) {
            const auto count=std::min(rows,w.out-begin);
            tyloi::cuda::gguf::unpack_quad_storage(w.type,w.packed.data_ptr(),w.quad_storage_rows,w.neuron_len,
                w.quad_row_begin+begin,count,nullptr,count,image.data_ptr(),w.row_bytes,stream.stream());
            tyloi::cuda::gguf::prefill_row_exponents(w.type,image.data_ptr(),w.row_bytes,count,w.neuron_len,
                entry->exponents.data_ptr<std::int32_t>()+begin,stream.stream());
        }
    } else {
        mfq_cuda_record_stream(w.packed,stream);
        tyloi::cuda::gguf::prefill_row_exponents(w.type,w.packed.data_ptr(),w.row_bytes,w.out,w.neuron_len,
            entry->exponents.data_ptr<std::int32_t>(),stream.stream());
    }
    entry->ready.record(stream.stream());auto result=entry->exponents;cache.push_back(std::move(entry));return result;
}
}

struct GgufPrefillInputScope::Impl {
    static thread_local Impl* active;Impl* previous=nullptr;
    ::tyloi::DenseTensor source,half,exponents;tyloi::cuda::Event ready;
};
thread_local GgufPrefillInputScope::Impl* GgufPrefillInputScope::Impl::active=nullptr;
GgufPrefillInputScope::GgufPrefillInputScope()=default;
std::unique_ptr<GgufPrefillInputScope> GgufPrefillInputScope::prepare_rms(
        const ::tyloi::DenseTensor& input,const ::tyloi::DenseTensor& weight,double eps,double offset,::tyloi::DenseTensor& normalized) {
    normalized={};const auto* setting=tyloi::cuda::env_lookup("TYLOINFER_GGUF_PREFILL_TENSORCORE");
    if(!input.is_cuda() || !input.is_contiguous() || input.dim()<1 || input.size(-1)<1 ||
            input.numel()/input.size(-1)<512 ||
            (input.scalar_type()!=::tyloi::kFloat32 && input.scalar_type()!=::tyloi::kFloat16 && input.scalar_type()!=::tyloi::kBFloat16) ||
            weight.device()!=input.device() || weight.scalar_type()!=::tyloi::kFloat32 || !weight.is_contiguous() ||
            weight.numel()!=input.size(-1) || !std::isfinite(eps) || eps<=0 || !std::isfinite(offset) ||
            (setting && setting[0]!='1' && setting[0]!='a'))return {};
    const MfqCudaGuard guard(input.get_device());const auto rows=input.numel()/input.size(-1),width=input.size(-1);
    if(!tyloi::cuda::gguf::grouped_f16_available(input.get_device()) ||
            !tyloi::cuda::gguf::rms_prefill_available(rows,width,input.get_device()))return {};
    auto scope=std::unique_ptr<GgufPrefillInputScope>(new GgufPrefillInputScope);
    auto plan=std::make_unique<Impl>();
    normalized=::tyloi::cuda::empty(input.sizes(),input.options().dtype(::tyloi::kFloat32));plan->source=normalized;
    plan->half=::tyloi::cuda::empty(input.sizes(),input.options().dtype(::tyloi::kFloat16));
    plan->exponents=::tyloi::cuda::empty({rows},input.options().dtype(::tyloi::kInt32));
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    for(const auto& tensor:{input,weight,normalized,plan->half,plan->exponents})mfq_cuda_record_stream(tensor,stream);
    const auto type=input.scalar_type()==::tyloi::kFloat32?GGML_TYPE_F32:input.scalar_type()==::tyloi::kFloat16?GGML_TYPE_F16:GGML_TYPE_BF16;
    tyloi::cuda::gguf::pack_rms_prefill(input.data_ptr(),type,weight.data_ptr<float>(),rows,width,float(eps),float(offset),
        normalized.data_ptr<float>(),plan->half.data_ptr(),plan->exponents.data_ptr<int32_t>(),stream.stream());
    plan->ready.record(stream.stream());plan->previous=Impl::active;Impl::active=plan.get();scope->impl_=std::move(plan);
    if(const auto* trace=tyloi::cuda::env_lookup("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true))std::fprintf(stderr,"native_gguf_prefill_rms_input actual_match=1 M=%lld K=%lld normalized_fp32=1 reuse_scaled_fp16=1\n",
            static_cast<long long>(rows),static_cast<long long>(width));
    }
    return scope;
}
GgufPrefillInputScope::GgufPrefillInputScope(const ::tyloi::DenseTensor& input) {
    const auto* setting=tyloi::cuda::env_lookup("TYLOINFER_GGUF_PREFILL_TENSORCORE");
    if(!input.is_cuda() || input.scalar_type()!=::tyloi::kFloat32 || !input.is_contiguous() || input.dim()<1 ||
            input.size(-1)<1 || input.numel()/input.size(-1)<512 ||
            (setting && setting[0]!='1' && setting[0]!='a'))return;
    const MfqCudaGuard guard(input.get_device());
    if(!tyloi::cuda::gguf::grouped_f16_available(input.get_device()))return;
    if(Impl::active && input.data_ptr()==Impl::active->source.data_ptr() &&
            input.device()==Impl::active->source.device() && input.sizes()==Impl::active->source.sizes())return;
    auto plan=std::make_unique<Impl>();plan->source=input;
    plan->half=::tyloi::cuda::empty(input.sizes(),input.options().dtype(::tyloi::kFloat16));
    plan->exponents=::tyloi::cuda::empty({input.numel()/input.size(-1)},input.options().dtype(::tyloi::kInt32));
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    for(const auto& tensor:{input,plan->half,plan->exponents})mfq_cuda_record_stream(tensor,stream);
    tyloi::cuda::gguf::pack_scaled_f16_unit(input.data_ptr<float>(),plan->half.data_ptr(),plan->exponents.data_ptr<int32_t>(),
        input.numel()/input.size(-1),input.size(-1),stream.stream());
    plan->ready.record(stream.stream());plan->previous=Impl::active;Impl::active=plan.get();impl_=std::move(plan);
}
GgufPrefillInputScope::~GgufPrefillInputScope(){if(impl_)Impl::active=impl_->previous;}
bool GgufPrefillInputScope::lookup(const ::tyloi::DenseTensor& input,::tyloi::DenseTensor& half,::tyloi::DenseTensor& exponents) {
    const auto* plan=Impl::active;
    if(!plan || !input.defined() || input.dim()<1 || !input.is_contiguous() ||
            input.scalar_type()!=::tyloi::kFloat32 || input.device()!=plan->source.device() || input.data_ptr()!=plan->source.data_ptr() || input.numel()!=plan->source.numel() ||
            input.size(-1)!=plan->source.size(-1))return false;
    const MfqCudaGuard guard(input.get_device());
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),plan->ready.get(),0));
    half=plan->half;exponents=plan->exponents;
    if(const auto* trace=tyloi::cuda::env_lookup("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "native_gguf_prefill_shared_input actual_match=1 M=%lld K=%lld reuse_scaled_fp16=1\n",
            static_cast<long long>(input.numel()/input.size(-1)),static_cast<long long>(input.size(-1)));
    }
    mfq_cuda_record_stream(half,stream);mfq_cuda_record_stream(exponents,stream);return true;
}
GgufPrefillProduct gguf_prefill_product(const GgmlQuantizedTensor& w, const DenseTensor& input,
        const DenseTensor& input_exponents) {
    const auto* setting = env_lookup("TYLOINFER_GGUF_PREFILL_TENSORCORE");
    const bool prepared = input_exponents.defined();
    if ((setting && setting[0] != '1' && setting[0] != 'a') || original_input(w) ||
        !w.packed.is_cuda() || !input.is_cuda() || input.device() != w.packed.device() ||
        input.scalar_type() != (prepared ? kFloat16 : kFloat32) || !input.is_contiguous() ||
        input.dim() < 1 || w.neuron_len <= 0 || input.size(-1) != w.neuron_len ||
        input.numel() / w.neuron_len < 512)
        return {};
    const auto tokens = input.numel() / w.neuron_len;
    if (prepared && (input_exponents.device() != input.device() || input_exponents.scalar_type() != kInt32 ||
        !input_exponents.is_contiguous() || input_exponents.numel() != tokens))
        return {};
    const MfqCudaGuard guard(w.packed.get_device());
    if (!grouped_f16_available(w.packed.get_device()) ||
        prefill_weight_rows(w.out, uint64_t(w.neuron_len) * 2 + uint64_t(tokens) * 2, 32, w.packed.get_device()) < w.out)
        return {};
    auto columns = prefill_cached_exponents(w);
    if (!columns.defined()) return {};
    DenseTensor packed, rows;
    const auto stream = mfq_get_current_cuda_stream(w.packed.get_device());
    if (prepared) {
        packed = input;
        rows = input_exponents;
    } else if (!GgufPrefillInputScope::lookup(input, packed, rows)) {
        packed = cuda::empty({tokens, w.neuron_len}, input.options().dtype(kFloat16));
        rows = cuda::empty({tokens}, input.options().dtype(kInt32));
        mfq_cuda_record_stream(input, stream);
        pack_scaled_f16_unit(input.data_ptr<float>(), packed.data_ptr(), rows.data_ptr<int32_t>(),
            tokens, w.neuron_len, stream.stream());
    }
    auto image=::tyloi::cuda::empty({w.out,w.neuron_len},input.options().dtype(::tyloi::kFloat16));
    auto product=::tyloi::cuda::empty({tokens,w.out},input.options().dtype(::tyloi::kFloat16));
    for(const auto& tensor:{w.packed,packed,rows,columns,image,product})mfq_cuda_record_stream(tensor,stream);
    if(w.quad_storage_rows)tyloi::cuda::gguf::decode_quad_rows_scaled(w.type,w.packed.data_ptr(),w.quad_storage_rows,
        w.neuron_len,w.quad_row_begin,w.out,columns.data_ptr<int32_t>(),image.data_ptr(),stream.stream());
    else tyloi::cuda::gguf::dequantize_rows_scaled(w.type,w.packed.data_ptr(),w.row_bytes,w.out,w.neuron_len,
        columns.data_ptr<int32_t>(),image.data_ptr(),stream.stream());
    auto& blas=tyloi::cuda::default_context(w.packed.get_device())->blas(stream.stream());
    tyloi::cuda::gguf::prefill_gemm_f16_accum(packed.data_ptr(),image.data_ptr(),nullptr,product.data_ptr(),true,
        nullptr,nullptr,tokens,w.out,w.neuron_len,w.out,blas.get(),stream.stream());
    if(const auto* trace=tyloi::cuda::env_lookup("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true))std::fprintf(stderr,
            "native_gguf_prefill_scaled_product actual_match=1 M=%lld N=%lld K=%lld restore_at_consumer=1\n",
            static_cast<long long>(tokens),static_cast<long long>(w.out),static_cast<long long>(w.neuron_len));
    }
    return {product,rows,columns};
}

std::array<DenseTensor, 2> pack_rms_gated_input(const DenseTensor& attended, const DenseTensor& gate,
        const DenseTensor& norm, double eps, bool silu, const GgufPrefillProduct& scaled_gate) {
    const auto* setting = env_lookup("TYLOINFER_GGUF_PREFILL_TENSORCORE");
    if ((setting && setting[0] != '1' && setting[0] != 'a') || !attended.is_cuda() ||
        !attended.is_contiguous() || attended.scalar_type() != kFloat32 || attended.dim() != 4 ||
        !norm.defined() || norm.device() != attended.device() || norm.scalar_type() != kFloat32 ||
        !norm.is_contiguous() || !std::isfinite(eps) || eps <= 0)
        return {};
    const auto batch = attended.size(0), heads = attended.size(1);
    const auto tokens = attended.size(2), width = attended.size(3);
    if (heads <= 0 || width <= 0) return {};
    const auto rows = batch * tokens, columns = heads * width;
    if (rows < 512 || norm.numel() != width) return {};
    const auto device = attended.get_device();
    const MfqCudaGuard guard(device);
    if (!grouped_f16_available(device) || !rms_gated_prefill_available(columns, device)) return {};

    auto values = gate;
    if (scaled_gate.defined()) {
        if (scaled_gate.values.dim() != 2 || !scaled_gate.values.is_contiguous() ||
            scaled_gate.values.size(0) != rows || scaled_gate.values.size(1) != columns)
            return {};
        for (const auto* e : {&scaled_gate.row_exponents, &scaled_gate.column_exponents})
            if (!e->defined() || !e->is_contiguous() ||
                e->device() != attended.device() || e->scalar_type() != kInt32)
                return {};
        if (scaled_gate.row_exponents.numel() != rows || scaled_gate.column_exponents.numel() != columns)
            return {};
        values = scaled_gate.values.reshape({batch, tokens, columns});
    }
    if (!values.defined() || values.device() != attended.device() ||
        values.scalar_type() != (scaled_gate.defined() ? kFloat16 : kFloat32) || values.dim() != 3 ||
        values.size(0) != batch || values.size(1) != tokens || values.size(2) != columns ||
        values.stride(2) != 1 || values.stride(1) < columns || values.stride(0) < tokens * values.stride(1))
        return {};

    auto packed = cuda::empty({rows, columns}, attended.options().dtype(kFloat16));
    auto exponents = cuda::empty({rows}, attended.options().dtype(kInt32));
    const auto stream = mfq_get_current_cuda_stream(device);
    for (const auto& tensor : {attended, values, norm, packed, exponents})
        mfq_cuda_record_stream(tensor, stream);
    if (scaled_gate.defined()) {
        mfq_cuda_record_stream(scaled_gate.row_exponents, stream);
        mfq_cuda_record_stream(scaled_gate.column_exponents, stream);
        pack_rms_scaled_gate_prefill(attended.data_ptr<float>(), values.data_ptr(), norm.data_ptr<float>(),
            scaled_gate.row_exponents.data_ptr<int32_t>(), scaled_gate.column_exponents.data_ptr<int32_t>(),
            batch, heads, tokens, width, values.stride(0), values.stride(1), float(eps), silu,
            packed.data_ptr(), exponents.data_ptr<int32_t>(), stream.stream());
    } else {
        pack_rms_gated_prefill(attended.data_ptr<float>(), values.data_ptr<float>(), norm.data_ptr<float>(),
            batch, heads, tokens, width, values.stride(0), values.stride(1), float(eps), silu,
            packed.data_ptr(), exponents.data_ptr<int32_t>(), stream.stream());
    }
    return {std::move(packed), std::move(exponents)};
}

GgufPrefillProduct gguf_rms_gated_product(const GgmlQuantizedTensor& weight, const DenseTensor& attended,
        const DenseTensor& gate, const DenseTensor& norm, double eps, bool silu) {
    auto [packed, exponents] = pack_rms_gated_input(attended, gate, norm, eps, silu);
    return packed.defined() ? gguf_prefill_product(weight, packed, exponents) : GgufPrefillProduct{};
}

GgufPrefillProduct gguf_rms_gated_product(const GgmlQuantizedTensor& weight, const DenseTensor& attended,
        const GgufPrefillProduct& gate, const DenseTensor& norm, double eps, bool silu) {
    auto [packed, exponents] = pack_rms_gated_input(attended, {}, norm, eps, silu, gate);
    return packed.defined() ? gguf_prefill_product(weight, packed, exponents) : GgufPrefillProduct{};
}

::tyloi::DenseTensor gguf_prefill_materialize(const GgufPrefillProduct& product) {
    if (!(product.defined() && product.values.is_cuda() && product.values.is_contiguous() &&
        product.values.dim()==2 && product.values.scalar_type()==::tyloi::kFloat16))
        throw std::runtime_error("scaled prefill materialization requires a contiguous FP16 product");
    const auto rows=product.values.size(0),columns=product.values.size(1);
    for(const auto* e:{&product.row_exponents,&product.column_exponents})
        {
            if (!(e->defined() && e->is_contiguous() && e->device()==product.values.device() && e->scalar_type()==::tyloi::kInt32))
                throw std::runtime_error("scaled prefill materialization exponents disagree");
        }
    if (!(product.row_exponents.numel()==rows && product.column_exponents.numel()==columns))
        throw std::runtime_error("scaled prefill materialization exponent sizes disagree");
    const MfqCudaGuard guard(product.values.get_device());
    auto output=::tyloi::cuda::empty({rows,columns},product.values.options().dtype(::tyloi::kFloat32));
    const auto stream=mfq_get_current_cuda_stream(output.get_device());
    for(const auto& t:{product.values,product.row_exponents,product.column_exponents,output})mfq_cuda_record_stream(t,stream);
    tyloi::cuda::gguf::restore_scaled_f16(product.values.data_ptr(),output.data_ptr(),false,
        product.row_exponents.data_ptr<int32_t>(),product.column_exponents.data_ptr<int32_t>(),rows,columns,columns,stream.stream());
    return output;
}

std::vector<::tyloi::DenseTensor> gguf_prefill_acc_rms(const GgufPrefillProduct& product,const ::tyloi::DenseTensor& residual,
        const ::tyloi::DenseTensor& norm,double eps,double offset) {
    if (!(product.defined() && residual.is_cuda() && residual.is_contiguous() &&
        residual.scalar_type()==::tyloi::kFloat16 && residual.dim()==2 && product.values.is_contiguous() &&
        product.values.device()==residual.device() && product.values.sizes()==residual.sizes() &&
        product.values.scalar_type()==::tyloi::kFloat16 && norm.is_contiguous() && norm.device()==residual.device() &&
        norm.scalar_type()==::tyloi::kFloat32 && norm.numel()==residual.size(1)))
        throw std::runtime_error("scaled prefill residual/RMS operands disagree");
    for(const auto* exponents:{&product.row_exponents,&product.column_exponents})
        {
            if (!(exponents->defined() && exponents->is_cuda() && exponents->is_contiguous() &&
            exponents->device()==residual.device() && exponents->scalar_type()==::tyloi::kInt32))
                throw std::runtime_error("scaled prefill residual/RMS exponents disagree");
        }
    if (!(product.row_exponents.numel()==residual.size(0) &&
        product.column_exponents.numel()==residual.size(1) && std::isfinite(eps) && eps>0))
        throw std::runtime_error("scaled prefill residual/RMS exponent geometry or epsilon disagrees");
    const MfqCudaGuard guard(residual.get_device());
    auto sum=::tyloi::cuda::empty_like(residual),normalized=::tyloi::cuda::empty_like(residual);
    const auto stream=mfq_get_current_cuda_stream(residual.get_device());
    for(const auto& tensor:{residual,norm,product.values,product.row_exponents,product.column_exponents,sum,normalized})
        mfq_cuda_record_stream(tensor,stream);
    tyloi::cuda::gguf::acc_rms_scaled_f16(product.values.data_ptr(),residual.data_ptr(),norm.data_ptr<float>(),
        product.row_exponents.data_ptr<int32_t>(),product.column_exponents.data_ptr<int32_t>(),
        residual.size(0),residual.size(1),float(eps),float(offset),sum.data_ptr(),normalized.data_ptr(),stream.stream());
    if(const auto* trace=tyloi::cuda::env_lookup("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true))std::fprintf(stderr,"native_gguf_prefill_acc_rms actual_match=1 fused_restore=1 fused_residual=1 fused_rms=1\n");
    }
    return {sum,normalized};
}


::tyloi::DenseTensor canonical_packed(const GgmlQuantizedTensor& w) {
    if(!w.quad_storage_rows)return w.packed;
    const MfqCudaGuard guard(w.packed.get_device());
    const auto stream=mfq_get_current_cuda_stream(w.packed.get_device());
    auto result=::tyloi::cuda::empty({std::int64_t(std::uint64_t(w.out)*w.row_bytes)},w.packed.options());
    mfq_cuda_record_stream(w.packed,stream);mfq_cuda_record_stream(result,stream);
    tyloi::cuda::gguf::unpack_quad_storage(w.type,w.packed.data_ptr(),w.quad_storage_rows,w.neuron_len,
        w.quad_row_begin,w.out,nullptr,w.out,result.data_ptr(),w.row_bytes,stream.stream());
    return result;
}

namespace {
tyloi::cuda::gguf::QuadProjection quad_projection(const GgmlQuantizedTensor& w,void* output) {
    return {{w.packed.data_ptr(),w.row_bytes,w.out,output,w.type},
        w.quad_storage_rows,w.quad_row_begin,bool(w.quad_storage_rows)};
}
}

::tyloi::DenseTensor gated_residual(const GgmlQuantizedTensor& w,
        const ::tyloi::DenseTensor& gate,const ::tyloi::DenseTensor& up,const ::tyloi::DenseTensor& residual) {
    if (!(!original_input(w) &&
        tyloi::cuda::gguf::supports_m1_residual(w.type) && gate.is_cuda() &&
        gate.device()==w.packed.device() && up.device()==gate.device() &&
        residual.device()==gate.device() && gate.scalar_type()==::tyloi::kFloat16 &&
        up.scalar_type()==::tyloi::kFloat16 && residual.scalar_type()==::tyloi::kFloat16 &&
        gate.is_contiguous() && up.is_contiguous() && residual.is_contiguous() &&
        gate.sizes()==up.sizes() && up.numel()==w.neuron_len && residual.numel()==w.out))
        throw std::runtime_error("fused GGUF SwiGLU/residual requires one contiguous FP16 row");
    const MfqCudaGuard guard(gate.get_device());
    const auto stream=mfq_get_current_cuda_stream(gate.get_device());
    auto activation=::tyloi::cuda::empty({int64_t(tyloi::cuda::gguf::q8_row_bytes(w.neuron_len))},
        gate.options().dtype(::tyloi::kUInt8));
    auto output=::tyloi::cuda::empty_like(residual);
    for(const auto& tensor:{gate,up,residual,w.packed,activation,output})mfq_cuda_record_stream(tensor,stream);
    const bool aligned=tyloi::cuda::gguf::prefers_m1_aligned(w.type);
    const auto quantize=aligned?tyloi::cuda::gguf::quantize_gated_input_m1_aligned:
        tyloi::cuda::gguf::quantize_gated_input;
    quantize(up.data_ptr(),gate.data_ptr(),true,1,w.neuron_len,true,activation.data_ptr(),stream.stream());
    if(quad_ready(w))tyloi::cuda::gguf::matvec_quad_m1(quad_projection(w,output.data_ptr()),
        w.neuron_len,activation.data_ptr(),aligned,true,residual.data_ptr(),stream.stream());
    else {
        const auto image=canonical_packed(w);mfq_cuda_record_stream(image,stream);
        tyloi::cuda::gguf::matvec_m1_residual(w.type,image.data_ptr(),w.row_bytes,w.out,w.neuron_len,
            activation.data_ptr(),aligned,residual.data_ptr(),output.data_ptr(),stream.stream());
    }
    return output;
}

bool same_input_basis(const GgmlQuantizedTensor& a, const GgmlQuantizedTensor& b) {
    return (!a.rotation && !b.rotation) || (a.rotation && b.rotation && (a.rotation->block==b.rotation->block && a.rotation->signs_cpu==b.rotation->signs_cpu));
}


::tyloi::DenseTensor dequantize_stored(const GgmlQuantizedTensor& w) {
    auto output = ::tyloi::cuda::empty({w.out, w.neuron_len}, w.packed.options().dtype(::tyloi::kFloat32));
    if (w.packed.is_cuda()) {
        MfqCudaGuard guard(w.packed.get_device());
        const auto stream=mfq_get_current_cuda_stream(w.packed.get_device());
        mfq_cuda_record_stream(w.packed,stream);mfq_cuda_record_stream(output,stream);
        if(w.quad_storage_rows)tyloi::cuda::gguf::decode_quad_rows(w.type,w.packed.data_ptr(),w.quad_storage_rows,
            w.neuron_len,w.quad_row_begin,w.out,nullptr,w.out,output.data_ptr(),false,stream.stream());
        else tyloi::cuda::gguf::dequantize_rows(w.type,w.packed.data_ptr(),w.row_bytes,w.out,w.neuron_len,
            output.data_ptr(),false,stream.stream());
    } else {
        for (std::int64_t row = 0; row < w.out; ++row)
            tyloi::gguf::decode_row(w.type, w.packed.data_ptr<std::uint8_t>() + row * w.row_bytes,
                output.data_ptr<float>() + row * w.neuron_len, w.neuron_len);
    }
    return output;
}

::tyloi::DenseTensor dequantize(const GgmlQuantizedTensor& w) {
    auto output = dequantize_stored(w);
    if (w.rotation) apply_rotation_inplace(*w.rotation,output, true);
    if (w.correction) {
        auto delta=dequantize_correction(*w.correction,w.packed.is_cuda()?w.packed.get_device():-1);
        if(output.is_cuda())output=output+delta;
        else for(std::int64_t i=0;i<output.numel();++i)output.data_ptr<float>()[i]+=delta.data_ptr<float>()[i];
    }
    return output;
}

::tyloi::DenseTensor embedding(const GgmlQuantizedTensor& w, ::tyloi::DenseTensor ids) {
    auto output=embedding_base(w,ids);
    return w.correction?add_correction_embedding(*w.correction,ids.to(output.device()),std::move(output)):output;
}

::tyloi::DenseTensor embedding_base(const GgmlQuantizedTensor& w, ::tyloi::DenseTensor ids) {
    auto shape = ids.sizes().vec();
    shape.push_back(w.neuron_len);
    ids = ids.to(w.packed.device()).to(::tyloi::kInt64).contiguous().reshape({-1});
    if (!w.packed.is_cuda()) {
        auto output = ::tyloi::cuda::zeros({ids.numel(), w.neuron_len}, w.packed.options().dtype(::tyloi::kFloat32));
        for (std::int64_t row = 0; row < ids.numel(); ++row) {
            const auto id = ids.data_ptr<std::int64_t>()[row];
            if (id < 0 || id >= w.out) continue;
            tyloi::gguf::decode_row(w.type, w.packed.data_ptr<std::uint8_t>() + id * w.row_bytes,
                output.data_ptr<float>() + row * w.neuron_len, w.neuron_len);
        }
        return output.reshape(shape);
    }
    MfqCudaGuard guard(w.packed.get_device());
    auto output = ::tyloi::cuda::empty({ids.numel(), w.neuron_len}, w.packed.options().dtype(::tyloi::kFloat32));
    const auto stream=mfq_get_current_cuda_stream(w.packed.get_device());
    mfq_cuda_record_stream(w.packed,stream); mfq_cuda_record_stream(ids,stream); mfq_cuda_record_stream(output,stream);
    if(w.quad_storage_rows) {
        tyloi::cuda::gguf::decode_quad_rows(w.type,w.packed.data_ptr(),w.quad_storage_rows,w.neuron_len,
            w.quad_row_begin,w.out,ids.data_ptr<std::int64_t>(),ids.numel(),output.data_ptr<float>(),false,stream.stream());
    } else tyloi::cuda::gguf::embedding_rows(w.type, w.packed.data_ptr(), w.row_bytes, w.out, w.neuron_len,
        ids.data_ptr<std::int64_t>(), ids.numel(), output.data_ptr<float>(), mfq_current_cuda_stream());
    if(w.rotation)apply_rotation_inplace(*w.rotation,output,true);
    return output.reshape(shape);
}

bool m1_aligned(const std::vector<GgmlQuantizedTensor>& weights,int64_t tokens) {
    return tokens==1 && !weights.empty() &&
        std::any_of(weights.begin(),weights.end(),[](const auto& w){return tyloi::cuda::gguf::prefers_m1_aligned(w.type);}) &&
        std::all_of(weights.begin(),weights.end(),[](const auto& w){return !original_input(w) && tyloi::cuda::gguf::supports_m1_aligned(w.type);});
}

ProjectionInput prepare_input(const std::vector<GgmlQuantizedTensor>& weights,::tyloi::DenseTensor input) {
    if(weights.empty())return {};
    const auto columns=weights.front().neuron_len;
    if (!(input.dim()>=1 && columns>0 && input.size(-1)==columns && input.is_cuda()))
        throw std::runtime_error("GGUF projection group input width/device mismatch");
    const auto tokens=input.numel()/columns;
    if (!(tokens<=32))
        throw std::runtime_error("GGUF shared quantization requires a small token window");
    for(const auto& weight:weights)
        {
            if (!(weight.neuron_len==columns && weight.packed.is_cuda() &&
            weight.packed.device()==input.device()))
                throw std::runtime_error("GGUF projection group contains incompatible input dimensions/devices");
        }
    const auto dtype=input.scalar_type()==::tyloi::kBFloat16?::tyloi::kFloat16:input.scalar_type();
    const MfqCudaGuard guard(input.get_device());
    const auto stream_handle=mfq_get_current_cuda_stream(input.get_device());
    const auto stream=stream_handle.stream();
    auto source=input.reshape({tokens,columns});
    auto original=source;
    if(source.scalar_type()!=::tyloi::kFloat16)source=source.to(::tyloi::kFloat32);
    source=source.contiguous();
    const auto supported=[&](const auto& weight){return !original_input(weight) && position_major_q8(weight,tokens);};
    const auto floating_codebook=[](const auto& weight){return original_input(weight);};
    const bool keep_original=std::any_of(weights.begin(),weights.end(),[](const auto& weight){return original_input(weight);});
    const bool interleaved=std::any_of(weights.begin(),weights.end(),supported);
    const bool native=std::any_of(weights.begin(),weights.end(),[&](const auto& weight){return !supported(weight) && !floating_codebook(weight);});
    const bool aligned_m1=m1_aligned(weights,tokens);
    auto activation=native?::tyloi::cuda::empty({std::int64_t(tokens*tyloi::cuda::gguf::q8_row_bytes(columns))},
        input.options().dtype(::tyloi::kUInt8)): ::tyloi::DenseTensor{};
    auto position_major=interleaved?::tyloi::cuda::empty({std::int64_t(tyloi::cuda::gguf::interleaved_activation_bytes(tokens,columns))},
        input.options().dtype(::tyloi::kUInt8)): ::tyloi::DenseTensor{};
    for(const auto& tensor:{source,activation,position_major})if(tensor.defined())mfq_cuda_record_stream(tensor,stream_handle);
    if(aligned_m1)tyloi::cuda::gguf::quantize_input_m1_aligned(source.data_ptr(),source.scalar_type()==::tyloi::kFloat16,columns,activation.data_ptr(),stream);
    else if(interleaved)
        tyloi::cuda::gguf::quantize_input_interleaved(source.data_ptr(),source.scalar_type()==::tyloi::kFloat16,tokens,columns,
            position_major.data_ptr(),activation.defined()?activation.data_ptr():nullptr,stream);
    else if(native && source.scalar_type()==::tyloi::kFloat16)
        tyloi::cuda::gguf::quantize_input_f16(source.data_ptr(),tokens,columns,activation.data_ptr(),stream);
    else if(native)tyloi::cuda::gguf::quantize_input(source.data_ptr<float>(),tokens,columns,activation.data_ptr(),stream);
    return {std::move(activation),std::move(position_major),keep_original?std::move(original): ::tyloi::DenseTensor{},input.sizes().vec(),tokens,dtype,aligned_m1};
}

ProjectionInput prepare_rms_input(const std::vector<GgmlQuantizedTensor>& weights, const DenseTensor& input,
        const DenseTensor& norm, double eps, double offset, DenseTensor& normalized) {
    normalized={};
    if(weights.empty() || !input.defined() || !input.is_cuda() || !input.is_contiguous() || input.dim()<1 ||
        (input.scalar_type()!=kFloat16 && input.scalar_type()!=kFloat32) ||
        !norm.defined() || !norm.is_cuda() || norm.device()!=input.device() || !norm.is_contiguous() ||
        norm.scalar_type()!=kFloat32 || norm.dim()<1 || !std::isfinite(eps) || eps<=0 ||
        !std::isfinite(float(eps)) || float(eps)<=0 || !std::isfinite(float(offset)))return {};
    const auto width=input.size(-1);
    if(width<1 || width>INT_MAX || width%32 || norm.numel()!=width || norm.size(-1)!=width)return {};
    const auto rows=input.numel()/width;
    if(rows<1 || rows>32)return {};
    for(const auto& weight:weights)
        if(original_input(weight) || weight.neuron_len!=width || !weight.packed.is_cuda() ||
            weight.packed.device()!=input.device())return {};
    const bool aligned=m1_aligned(weights,rows);
    const bool interleaved=std::any_of(weights.begin(),weights.end(),[&](const auto& w){return position_major_q8(w,rows);});
    const bool native=std::any_of(weights.begin(),weights.end(),[&](const auto& w){return !position_major_q8(w,rows);});
    const MfqCudaGuard guard(input.get_device());
    const auto handle=mfq_get_current_cuda_stream(input.get_device());
    auto value=::tyloi::cuda::empty(input.sizes(),input.options().dtype(kFloat32));
    auto packet=native?::tyloi::cuda::empty({std::int64_t(rows*q8_row_bytes(width))},input.options().dtype(kUInt8)):DenseTensor{};
    auto position=interleaved?::tyloi::cuda::empty({std::int64_t(interleaved_activation_bytes(rows,width))},input.options().dtype(kUInt8)):DenseTensor{};
    for(const auto& tensor:{input,norm,value,packet,position})if(tensor.defined())mfq_cuda_record_stream(tensor,handle);
    if(interleaved)quantize_rms_input_interleaved(input.data_ptr(),input.scalar_type()==kFloat16,
        norm.data_ptr<float>(),rows,width,float(eps),float(offset),value.data_ptr<float>(),position.data_ptr(),
        packet.defined()?packet.data_ptr():nullptr,handle.stream());
    else {
        const auto quantize=aligned?quantize_rms_input_m1_aligned:quantize_rms_input;
        quantize(input.data_ptr(),input.scalar_type()==kFloat16,norm.data_ptr<float>(),rows,width,
            float(eps),float(offset),value.data_ptr<float>(),packet.data_ptr(),handle.stream());
    }
    normalized=std::move(value);
    return {std::move(packet),std::move(position),{},input.sizes().vec(),rows,kFloat32,aligned};
}

::tyloi::DenseTensor project_prepared(const ProjectionInput& prepared, const GgmlQuantizedTensor& weight) {
    if (prepared.shape.empty())
        throw std::runtime_error("GGUF projection input was not prepared");
    if (!(!prepared.aligned_m1 || prepared.tokens==1))
        throw std::runtime_error("Aligned GGUF projection requires one token");
    const bool use_interleaved=position_major_q8(weight,prepared.tokens);
    auto output_shape=prepared.shape;output_shape.back()=weight.out;
    if(!prepared.tokens)return ::tyloi::cuda::empty(output_shape,weight.packed.options().dtype(prepared.dtype));
    if(original_input(weight)) {
        if (!prepared.original.defined())
            throw std::runtime_error("GGUF floating codebook projection requires the original activation");
        return project(weight,prepared.original).to(prepared.dtype).reshape(output_shape);
    }
    auto result=(prepared.aligned_m1?project_m1(weight,prepared.native,prepared.dtype==::tyloi::kFloat16?::tyloi::kFloat16: ::tyloi::kFloat32):project_q8(weight,use_interleaved?prepared.interleaved:prepared.native,prepared.tokens,
        prepared.dtype==::tyloi::kFloat16?::tyloi::kFloat16: ::tyloi::kFloat32,use_interleaved)).to(prepared.dtype).reshape(output_shape);

    return result;
}

std::vector<::tyloi::DenseTensor> project_group(const std::vector<GgmlQuantizedTensor>& weights,::tyloi::DenseTensor input) {
    if(weights.empty())return {};
    if (!(input.dim() >= 1 && weights.front().neuron_len > 0 &&
        input.size(-1) == weights.front().neuron_len))
        throw std::runtime_error("GGUF projection group input width mismatch");
    if (!input.is_cuda() || input.numel() / input.size(-1) > 32) {
        GgufPrefillInputScope scope(input.reshape({-1, weights.front().neuron_len}));
        std::vector<DenseTensor> result;
        result.reserve(weights.size());
        for (const auto& weight : weights) result.push_back(project(weight, input));
        return result;
    }
    const auto prepared=prepare_input(weights,std::move(input));
    return project_input(prepared,weights);
}

std::vector<::tyloi::DenseTensor> project_input(const ProjectionInput& prepared, const std::vector<GgmlQuantizedTensor>& weights) {
    std::vector<::tyloi::DenseTensor> output;output.reserve(weights.size());

    const bool grouped_small_m=weights.size()>1 && prepared.tokens>=2 && prepared.tokens<=8 &&
        std::all_of(weights.begin(),weights.end(),[&](const auto& weight) {
            return quad_ready(weight) && !original_input(weight) &&
                tyloi::cuda::gguf::supports_quad_small_m(weight.type,prepared.tokens);
        });
    if(grouped_small_m) {
        const MfqCudaGuard guard(weights.front().packed.get_device());
        const auto handle=mfq_get_current_cuda_stream(weights.front().packed.get_device());
        const auto stream=handle.stream();
        for(const auto& packet:{prepared.native,prepared.interleaved,prepared.original})
            if(packet.defined())mfq_cuda_record_stream(packet,handle);
        const auto result_type=prepared.dtype==::tyloi::kFloat16?::tyloi::kFloat16: ::tyloi::kFloat32;
        std::vector<tyloi::cuda::gguf::QuadProjection> quad;
        std::vector<tyloi::cuda::gguf::DenseProjection> descriptors;
        std::vector<std::uint8_t> layouts;
        quad.reserve(weights.size());descriptors.reserve(weights.size());layouts.reserve(weights.size());
        for(const auto& weight:weights) {
            auto result=::tyloi::cuda::empty({prepared.tokens,weight.out},weight.packed.options().dtype(result_type));
            mfq_cuda_record_stream(weight.packed,handle);mfq_cuda_record_stream(result,handle);
            quad.push_back(quad_projection(weight,result.data_ptr()));
            descriptors.push_back(quad.back().projection);
            layouts.push_back(position_major_q8(weight,prepared.tokens));
            output.push_back(std::move(result));
        }
        const auto separate=[&] {
            for(std::size_t i=0;i<quad.size();++i)
                tyloi::cuda::gguf::matvec_quad_small_m(quad[i],weights.front().neuron_len,
                    (layouts[i]?prepared.interleaved:prepared.native).data_ptr(),prepared.tokens,layouts[i]!=0,
                    result_type==::tyloi::kFloat16,stream);
        };
        const auto batch=[&] {
            tyloi::cuda::gguf::matvec_quad_group_small_m(quad.data(),quad.size(),weights.front().neuron_len,
                prepared.native.defined()?prepared.native.data_ptr():nullptr,prepared.interleaved.defined()?prepared.interleaved.data_ptr():nullptr,
                layouts.data(),prepared.tokens,result_type==::tyloi::kFloat16,stream);
        };
        const auto pair=[&] {
            tyloi::cuda::gguf::matvec_quad_pair_small_m(quad.data(),quad.size(),weights.front().neuron_len,
                prepared.native.defined()?prepared.native.data_ptr():nullptr,prepared.interleaved.defined()?prepared.interleaved.data_ptr():nullptr,
                layouts.data(),prepared.tokens,result_type==::tyloi::kFloat16,stream);
        };
        std::size_t selected=0;
        if(prepared.tokens>=3 && quad.size()==2)
            selected=mfq_gguf_dense_schedule::choose_group(descriptors,weights.front().packed.get_device(),
                weights.front().neuron_len,prepared.tokens,result_type==::tyloi::kFloat16,stream,separate,batch,pair);
        else selected=mfq_gguf_dense_schedule::choose_batch(descriptors,weights.front().packed.get_device(),
                weights.front().neuron_len,prepared.tokens,result_type==::tyloi::kFloat16,stream,separate,batch)?1:0;
        if(selected==2)pair();else if(selected)batch();else separate();
        if(const auto* trace=tyloi::cuda::env_lookup("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
            TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
            if(capture==cudaStreamCaptureStatusActive)
                std::fprintf(stderr,"native_gguf_quad_group_use actual_match=1 M=%lld matrices=%zu selected=%s captured=1\n",
                    static_cast<long long>(prepared.tokens),weights.size(),selected==2?"pair":selected?"batch":"separate");
        }
        for(std::size_t i=0;i<weights.size();++i) {
            auto output_shape=prepared.shape;output_shape.back()=weights[i].out;
            output[i]=output[i].to(prepared.dtype).reshape(output_shape);

        }
        return output;
    }
    const bool direct_small_m=std::any_of(weights.begin(),weights.end(),[&](const auto& weight){
        return quad_ready(weight) && tyloi::cuda::gguf::supports_quad_small_m(weight.type,prepared.tokens);
    });
    if(weights.size()<2 || !prepared.tokens || prepared.tokens>4 || direct_small_m) {
        for(const auto& weight:weights)output.push_back(project_prepared(prepared,weight));
        return output;
    }
    if (!(!prepared.aligned_m1 || (prepared.tokens==1 && prepared.native.defined() && prepared.native.is_cuda() &&
        prepared.native.device()==weights.front().packed.device() && prepared.native.is_contiguous() &&
        prepared.native.scalar_type()==::tyloi::kUInt8 && prepared.native.nbytes()==tyloi::cuda::gguf::q8_row_bytes(weights.front().neuron_len))))
        throw std::runtime_error("Aligned GGUF group activation geometry/device/type differs");
    const MfqCudaGuard guard(weights.front().packed.get_device());
    const auto stream=mfq_get_current_cuda_stream(weights.front().packed.get_device());
    for(const auto& packet:{prepared.native,prepared.interleaved,prepared.original})if(packet.defined())mfq_cuda_record_stream(packet,stream);
    std::vector<tyloi::cuda::gguf::DenseProjection> projections;projections.reserve(weights.size());
    std::vector<tyloi::cuda::gguf::QuadProjection> quad_projections;quad_projections.reserve(weights.size());
    std::vector<::tyloi::DenseTensor> canonical_images;canonical_images.reserve(weights.size());
    bool has_quad=false;
    std::vector<std::size_t> projection_indices;projection_indices.reserve(weights.size());
    const auto result_type=prepared.dtype==::tyloi::kFloat16?::tyloi::kFloat16: ::tyloi::kFloat32;
    for(const auto& weight:weights) {
        if(weight.correction || weight.rotation || (weight.type>=GGML_TYPE_PTQ1_0 && weight.type<=GGML_TYPE_TQK4)) {
            output.push_back(project_prepared(prepared,weight));
            continue;
        }
        auto result=::tyloi::cuda::empty({prepared.tokens,weight.out},weight.packed.options().dtype(result_type));
        mfq_cuda_record_stream(weight.packed,stream);mfq_cuda_record_stream(result,stream);
        auto p=quad_projection(weight,result.data_ptr());
        if(weight.quad_storage_rows && (prepared.tokens!=1 || !quad_ready(weight))) {
            const auto image=canonical_packed(weight);mfq_cuda_record_stream(image,stream);
            p.projection.packed=image.data_ptr();p.quad=false;
            canonical_images.push_back(image);
        }
        projections.push_back(p.projection);quad_projections.push_back(p);has_quad|=p.quad;
        projection_indices.push_back(output.size());
        output.push_back(std::move(result));
    }
    if(projections.empty())return output;
    if(has_quad) {
        // The mixed kernel consumes 256-weight/eight-fragment formats, the
        // same set supported by aligned M=1. Legacy block sizes still use
        // their normal projection alongside direct interleaved projections.
        const bool compatible=std::all_of(projections.begin(),projections.end(),[](const auto& p){
            return tyloi::cuda::gguf::supports_m1_aligned(p.type);
        });
        if(compatible)tyloi::cuda::gguf::matvec_quad_group_m1(quad_projections.data(),quad_projections.size(),
            weights.front().neuron_len,prepared.native.data_ptr(),prepared.aligned_m1,result_type==::tyloi::kFloat16,stream.stream());
        else for(const auto& p:quad_projections)tyloi::cuda::gguf::matvec_quad_m1(p,weights.front().neuron_len,
            prepared.native.data_ptr(),prepared.aligned_m1,result_type==::tyloi::kFloat16,nullptr,stream.stream());
        for(const auto i:projection_indices){auto output_shape=prepared.shape;output_shape.back()=weights[i].out;
            output[i]=output[i].to(prepared.dtype).reshape(output_shape);}
        return output;
    }
    const auto separate=[&] {
        for(const auto& p:projections) {
            if(prepared.aligned_m1)tyloi::cuda::gguf::matvec_m1_aligned(p.type,p.packed,p.stride,p.rows,weights.front().neuron_len,
                prepared.native.data_ptr(),p.output,result_type==::tyloi::kFloat16,stream.stream());
            else if(tyloi::cuda::gguf::supports_interleaved_activation(p.type,prepared.tokens))
                tyloi::cuda::gguf::matvec_interleaved(p.type,p.packed,p.stride,p.rows,weights.front().neuron_len,
                    prepared.interleaved.data_ptr(),prepared.tokens,p.output,result_type==::tyloi::kFloat16,stream.stream());
            else if(result_type==::tyloi::kFloat16)
                tyloi::cuda::gguf::matvec_q8_f16(p.type,p.packed,p.stride,p.rows,weights.front().neuron_len,
                    prepared.native.data_ptr(),prepared.tokens,p.output,stream.stream());
            else tyloi::cuda::gguf::matvec_q8(p.type,p.packed,p.stride,p.rows,weights.front().neuron_len,
                    prepared.native.data_ptr(),prepared.tokens,static_cast<float*>(p.output),stream.stream());
        }
    };
    const auto batch=[&] {
        if(prepared.aligned_m1)tyloi::cuda::gguf::matvec_group_m1_aligned(projections.data(),projections.size(),weights.front().neuron_len,
            prepared.native.data_ptr(),result_type==::tyloi::kFloat16,stream.stream());
        else tyloi::cuda::gguf::matvec_group(projections.data(),projections.size(),weights.front().neuron_len,
            prepared.native.defined()?prepared.native.data_ptr():nullptr,prepared.interleaved.defined()?prepared.interleaved.data_ptr():nullptr,
            prepared.tokens,result_type==::tyloi::kFloat16,stream.stream());
    };
    bool selected=false;
    selected=mfq_gguf_dense_schedule::choose_batch(projections,weights.front().packed.get_device(),
        weights.front().neuron_len,prepared.tokens,result_type==::tyloi::kFloat16,stream.stream(),separate,batch,prepared.aligned_m1);
    if(selected)batch();else separate();
    if(const auto* trace=tyloi::cuda::env_lookup("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
        TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(),&status));
        if(status==cudaStreamCaptureStatusActive)
            std::fprintf(stderr,"native_gguf_dense_schedule_use actual_match=1 M=%lld matrices=%zu selected=%s captured=1\n",
                static_cast<long long>(prepared.tokens),weights.size(),selected?"batch":"separate");
    }
    for(const auto i:projection_indices) {
        auto output_shape=prepared.shape;output_shape.back()=weights[i].out;
        output[i]=output[i].to(prepared.dtype).reshape(output_shape);
    }
    return output;
}

::tyloi::DenseTensor project_m1(const GgmlQuantizedTensor& w, const ::tyloi::DenseTensor& activation,::tyloi::ScalarType dtype) {
    if (!(!original_input(w) && tyloi::cuda::gguf::supports_m1_aligned(w.type) &&
        w.packed.is_cuda() && activation.is_cuda() && activation.device()==w.packed.device() &&
        activation.is_contiguous() && activation.scalar_type()==::tyloi::kUInt8 &&
        activation.nbytes()==tyloi::cuda::gguf::q8_row_bytes(w.neuron_len) &&
        (dtype==::tyloi::kFloat16 || dtype==::tyloi::kFloat32)))
        throw std::runtime_error("Aligned GGUF M=1 activation geometry/device/type differs");
    const MfqCudaGuard guard(w.packed.get_device());const auto stream=mfq_get_current_cuda_stream(w.packed.get_device());
    auto output=::tyloi::cuda::empty({1,w.out},w.packed.options().dtype(dtype));
    for(const auto& tensor:{activation,w.packed,output})mfq_cuda_record_stream(tensor,stream);
    if(quad_ready(w))tyloi::cuda::gguf::matvec_quad_m1(quad_projection(w,output.data_ptr()),
        w.neuron_len,activation.data_ptr(),true,dtype==::tyloi::kFloat16,nullptr,stream.stream());
    else {
        const auto image=canonical_packed(w);mfq_cuda_record_stream(image,stream);
        tyloi::cuda::gguf::matvec_m1_aligned(w.type,image.data_ptr(),w.row_bytes,w.out,w.neuron_len,
            activation.data_ptr(),output.data_ptr(),dtype==::tyloi::kFloat16,stream.stream());
    }
    return output;
}

bool position_major_q8(const GgmlQuantizedTensor& w, std::int64_t tokens) {
    // Three/four queries share coordinate words through aligned vector loads.
    // The two-query native packet retains its smaller register working set.
    if(tokens>=3 && tokens<=8 && quad_ready(w) &&
            (w.type==GGML_TYPE_IQ2_XXS || w.type==GGML_TYPE_IQ2_XS || w.type==GGML_TYPE_IQ2_S))return true;
    if(tokens>=5 && tokens<=8 && w.packed.defined() && w.packed.is_cuda() &&
            tyloi::cuda::gguf::supports_dense_split_window_q8(w.type,w.packed.data_ptr(),w.row_bytes,w.out,w.neuron_len,tokens))return true;
    if(tokens>=5 && tokens<=8 && quad_ready(w) &&
            (w.type==GGML_TYPE_IQ1_S || w.type==GGML_TYPE_IQ1_M || w.type==GGML_TYPE_Q2_K))return true;
    return tyloi::cuda::gguf::supports_interleaved_activation(w.type,tokens) &&
        !(quad_ready(w) && tyloi::cuda::gguf::supports_quad_small_m(w.type,tokens));
}


::tyloi::DenseTensor project_q8(const GgmlQuantizedTensor& w, const ::tyloi::DenseTensor& activation,std::int64_t tokens,::tyloi::ScalarType dtype,bool position_major) {
    if (!(!w.rotation && !w.correction))
        throw std::runtime_error("GGUF rotated/residual projection requires its original activation before Q8 quantization");
    const auto bytes_per_row=tyloi::cuda::gguf::q8_row_bytes(w.neuron_len);
    const auto bytes=position_major?tyloi::cuda::gguf::interleaved_activation_bytes(tokens,w.neuron_len)
        :std::uint64_t(tokens)*bytes_per_row;
    if (!(w.packed.is_cuda() && activation.is_cuda() && activation.device()==w.packed.device() &&
        activation.is_contiguous() && activation.scalar_type()==::tyloi::kUInt8 && tokens>=1 &&
        std::uint64_t(tokens)<=std::numeric_limits<std::uint64_t>::max()/bytes_per_row &&
        (dtype==::tyloi::kFloat16 || dtype==::tyloi::kFloat32) &&
        (!position_major || tyloi::cuda::gguf::supports_interleaved_activation(w.type,tokens) || position_major_q8(w,tokens)) &&
        activation.nbytes()==bytes))
        throw std::runtime_error("GGUF prepared activation geometry/device/type differs");
    const MfqCudaGuard guard(w.packed.get_device());const auto stream_handle=mfq_get_current_cuda_stream(w.packed.get_device());
    const auto stream=stream_handle.stream();
    auto result=::tyloi::cuda::empty({tokens,w.out},w.packed.options().dtype(dtype));
    for(const auto& tensor:{activation,w.packed,result})mfq_cuda_record_stream(tensor,stream_handle);
    if(tokens==1 && !position_major && quad_ready(w)) {
        tyloi::cuda::gguf::matvec_quad_m1(quad_projection(w,result.data_ptr()),w.neuron_len,
            activation.data_ptr(),false,dtype==::tyloi::kFloat16,nullptr,stream);return result;
    }
    if(quad_ready(w) && tyloi::cuda::gguf::supports_quad_small_m(w.type,tokens)) {
        tyloi::cuda::gguf::matvec_quad_small_m(quad_projection(w,result.data_ptr()),w.neuron_len,
            activation.data_ptr(),tokens,position_major,dtype==::tyloi::kFloat16,stream);
        return result;
    }
    const auto image=canonical_packed(w);mfq_cuda_record_stream(image,stream_handle);
    if(position_major && tyloi::cuda::gguf::project_dense_split_window_q8(w.type,image.data_ptr(),w.row_bytes,w.out,w.neuron_len,
            activation.data_ptr(),tokens,result.data_ptr(),dtype==::tyloi::kFloat16,stream))return result;
    if(position_major)tyloi::cuda::gguf::matvec_interleaved(w.type,image.data_ptr(),w.row_bytes,w.out,w.neuron_len,
        activation.data_ptr(),tokens,result.data_ptr(),dtype==::tyloi::kFloat16,stream);
    else if(dtype==::tyloi::kFloat16)tyloi::cuda::gguf::matvec_q8_f16(w.type,image.data_ptr(),w.row_bytes,w.out,w.neuron_len,
        activation.data_ptr(),tokens,result.data_ptr(),stream);
    else tyloi::cuda::gguf::matvec_q8(w.type,image.data_ptr(),w.row_bytes,w.out,w.neuron_len,
        activation.data_ptr(),tokens,result.data_ptr<float>(),stream);
    return result;
}
void project_routed(std::span<const GgmlQuantizedTensor> images, const weights::ExpertRows& rows,
        std::size_t first, const DenseTensor& input, DenseTensor& output) {
    const auto& indexes = input.dim() == 3 ? rows.slots : rows.inputs;
    const auto source = input.reshape({-1, input.size(-1)});
    const auto project_one = [&](std::size_t i) {
        const auto& group = rows.groups[first + i];
        auto value = project(images[i], source.index_select(0, indexes.narrow(0, group.begin, group.count)));
        output.index_copy_(0, rows.slots.narrow(0, group.begin, group.count), value);
    };
    if (input.scalar_type() == kFloat16 && rows.tokens > 32 && images.size() > 1 &&
        images.size() <= 65535 && source.is_contiguous() && source.size(1) % 8 == 0 &&
        reinterpret_cast<std::uintptr_t>(source.data_ptr()) % 16 == 0 && grouped_f16_available(input.get_device())) {
        std::vector<GemmJob> jobs;
        std::vector<std::size_t> fallback;
        jobs.reserve(images.size());
        std::int64_t maximum = 0;
        for (std::size_t i = 0; i < images.size(); ++i) {
            const auto& image = images[i];
            const auto& group = rows.groups[first + i];
            if (group.count < 32 || group.count >= 512 || original_input(image) || image.quad_storage_rows) {
                fallback.push_back(i);
                continue;
            }
            jobs.push_back({image.packed.data_ptr(), rows.slots.data_ptr<std::int64_t>() + group.begin, group.count});
            maximum = std::max(maximum, group.count);
        }
        if (jobs.size() > 1) {
            const auto& weight = images.front();
            const auto stream = cuda::current_stream(input.get_device());
            auto plan = cuda::empty({static_cast<std::int64_t>(jobs.size() * sizeof(GemmJob))},
                source.options().dtype(kUInt8));
            auto* device_jobs = static_cast<GemmJob*>(plan.data_ptr());
            stage_grouped_jobs(jobs.data(), device_jobs, jobs.size(), stream.stream());
            grouped_matmul_f16(weight.type, device_jobs, static_cast<int>(jobs.size()), maximum,
                weight.row_bytes, weight.out, weight.neuron_len, source.data_ptr(), rows.routes,
                input.dim() == 3, output.data_ptr(), stream.stream());
            for (const auto i : fallback) project_one(i);
            return;
        }
    }
    for (std::size_t i = 0; i < images.size(); ++i) project_one(i);
}

DenseTensor project_indexed(const GgmlQuantizedTensor& weight, const DenseTensor& addresses,
        const weights::ExpertRows& rows, const DenseTensor& input) {
    if (!addresses.defined() || addresses.numel() != rows.experts || original_input(weight) ||
        input.scalar_type() != kFloat16 || rows.tokens < 1 || rows.tokens > 32 ||
        (rows.tokens > 1 && !rows.device_groups.defined()))
        throw std::invalid_argument("invalid indexed GGML routed projection");
    const auto stream = cuda::current_stream(input.get_device());
    const auto columns = weight.neuron_len;
    const auto source = input.reshape({-1, columns}).contiguous();
    auto activation = cuda::empty({static_cast<std::int64_t>(source.size(0) * q8_row_bytes(columns))},
        source.options().dtype(kUInt8));
    auto output = cuda::empty({rows.tokens, rows.routes, weight.out}, input.options().dtype(kFloat32));
    weight.record_stream(reinterpret_cast<std::uintptr_t>(stream.stream()), stream.lifetime());
    addresses.record_stream(reinterpret_cast<std::uintptr_t>(stream.stream()), stream.lifetime());
    quantize_input_f16(source.data_ptr(), source.size(0), columns, activation.data_ptr(), stream.stream());
    const auto* groups = rows.device_groups.defined() ? rows.device_groups.data_ptr<RouteGroup>() : nullptr;
    grouped_routed_matvec_q8(weight.type, addresses.data_ptr<std::uint64_t>(), weight.row_bytes,
        rows.experts, weight.out, columns, rows.device_ids.data_ptr<std::int64_t>(), groups,
        rows.tokens, rows.routes, activation.data_ptr(), input.dim() == 3, output.data_ptr<float>(), stream.stream());
    return output.to(kFloat16);
}

::tyloi::DenseTensor project(const GgmlQuantizedTensor& w, ::tyloi::DenseTensor input,bool allow_mmq) {
    auto output=project_base(w,input,allow_mmq);
    if(w.correction)output=add_correction(*w.correction,input,std::move(output));
    return output;
}

bool can_prefill_residual(const GgmlQuantizedTensor& w, int64_t tokens,const ::tyloi::DenseTensor& residual) {
    return !original_input(w) && w.packed.defined() &&
        can_prefill_f16_residual(tokens, w.out, w.neuron_len, w.packed.device(), residual);
}

DenseTensor project_prefill_residual(const GgmlQuantizedTensor& weight, DenseTensor input,
        const DenseTensor& residual) {
    return project_base(weight, std::move(input), true, residual);
}

::tyloi::DenseTensor project_base(const GgmlQuantizedTensor& w, ::tyloi::DenseTensor input,bool allow_mmq) {
    return project_base(w,std::move(input),allow_mmq,{});
}

::tyloi::DenseTensor project_base(const GgmlQuantizedTensor& w, ::tyloi::DenseTensor input,bool allow_mmq,const ::tyloi::DenseTensor& prefill_residual) {
    if (input.dim() < 1 || input.size(-1) != w.neuron_len)
        throw std::runtime_error("GGML projection input width mismatch");
    auto shape = input.sizes().vec();
    shape.back() = w.out;
    // CUDA attention and sampling consume FP16/FP32 activations. BF16 norms
    // from a GGUF source must not propagate BF16 logits into the sampler.
    const auto dtype = input.scalar_type() == ::tyloi::kBFloat16 ? ::tyloi::kFloat16 : input.scalar_type();
    auto x = input.reshape({-1, w.neuron_len}).contiguous();
    if(w.rotation)x=apply_rotation(*w.rotation,x);
    if (prefill_residual.defined() && (!allow_mmq || dtype != ::tyloi::kFloat16 ||
        !can_prefill_residual(w, x.size(0), prefill_residual)))
        throw std::runtime_error("invalid GGML prefill residual projection");
    if (!w.packed.is_cuda())
        return ::tyloi::cuda::matmul(x.to(::tyloi::kFloat32), dequantize_stored(w).transpose(0, 1)).to(dtype).reshape(shape);
    MfqCudaGuard guard(w.packed.get_device());
    if (!x.is_cuda() || x.get_device() != w.packed.get_device())
        throw std::runtime_error("GGML weight and input must share a device");
    const auto tokens = x.size(0);
    if(w.quad_storage_rows && tokens<=32) {
        if(tokens==1 || (quad_ready(w) && tyloi::cuda::gguf::supports_quad_small_m(w.type,tokens))) {
            const auto prepared=prepare_input({w},x);
            return project_prepared(prepared,w).to(dtype).reshape(shape);
        }
        auto base=w;base.packed=canonical_packed(w);base.quad_storage_rows=0;base.quad_row_begin=0;
        base.rotation.reset();
        // x already has the input rotation. Preserve only the base projection;
        // The outer projection applies the row correction.
        return project_base(base,x,allow_mmq).to(dtype).reshape(shape);
    }
    const auto output_dtype=dtype==::tyloi::kFloat16?::tyloi::kFloat16: ::tyloi::kFloat32;
    auto output = ::tyloi::cuda::empty({tokens, w.out}, w.packed.options().dtype(output_dtype));
    const auto stream=mfq_get_current_cuda_stream(w.packed.get_device());
    mfq_cuda_record_stream(w.packed,stream); mfq_cuda_record_stream(output,stream);
    if(prefill_residual.defined())mfq_cuda_record_stream(prefill_residual,stream);
    // Trellis weights decode to floating-point codebook entries. Preserve the
    // original activation instead of requantizing it to Q8 for a float dot.
    // Large prefill batches retain the tensor-core path below.
    if (w.type >= GGML_TYPE_TQ2_T && w.type <= GGML_TYPE_TQK4 && tokens <= 32) {
        auto source = x.scalar_type() == ::tyloi::kFloat16 ? x : x.to(::tyloi::kFloat32).contiguous();
        mfq_cuda_record_stream(source,stream);
        tyloi::cuda::gguf::compact_project(tyloi::cuda::gguf::CompactFormat(w.type), w.packed.data_ptr(), w.row_bytes,
            w.out, w.neuron_len, source.data_ptr(), source.scalar_type() == ::tyloi::kFloat16
                ? tyloi::cuda::gguf::CompactInput::Float16 : tyloi::cuda::gguf::CompactInput::Float32,
            tokens, output.data_ptr(), output_dtype == ::tyloi::kFloat16, mfq_current_cuda_stream());
        return output.to(dtype).reshape(shape);
    }
    if (tokens <= 32) {
        auto source=x.scalar_type()==::tyloi::kFloat16?x:x.to(::tyloi::kFloat32).contiguous();
        mfq_cuda_record_stream(source,stream);
        const bool interleaved=position_major_q8(w,tokens);
        const bool aligned_m1=tokens==1 && tyloi::cuda::gguf::prefers_m1_aligned(w.type);
        auto activation = ::tyloi::cuda::empty({static_cast<std::int64_t>(interleaved
                ?tyloi::cuda::gguf::interleaved_activation_bytes(tokens,w.neuron_len)
                :tokens * tyloi::cuda::gguf::q8_row_bytes(w.neuron_len))},
            w.packed.options());
        mfq_cuda_record_stream(activation,stream);
        if(aligned_m1)tyloi::cuda::gguf::quantize_input_m1_aligned(source.data_ptr(),source.scalar_type()==::tyloi::kFloat16,
            w.neuron_len,activation.data_ptr(),mfq_current_cuda_stream());
        else if(interleaved)
            tyloi::cuda::gguf::quantize_input_interleaved(source.data_ptr(),source.scalar_type()==::tyloi::kFloat16,
                tokens,w.neuron_len,activation.data_ptr(),nullptr,mfq_current_cuda_stream());
        else if(source.scalar_type()==::tyloi::kFloat16)
            tyloi::cuda::gguf::quantize_input_f16(source.data_ptr(),tokens,w.neuron_len,activation.data_ptr(),mfq_current_cuda_stream());
        else tyloi::cuda::gguf::quantize_input(source.data_ptr<float>(),tokens,w.neuron_len,activation.data_ptr(),mfq_current_cuda_stream());
        if(aligned_m1)tyloi::cuda::gguf::matvec_m1_aligned(w.type,w.packed.data_ptr(),w.row_bytes,w.out,w.neuron_len,
            activation.data_ptr(),output.data_ptr(),output_dtype==::tyloi::kFloat16,mfq_current_cuda_stream());
        else if(interleaved && tyloi::cuda::gguf::project_dense_split_window_q8(w.type,w.packed.data_ptr(),w.row_bytes,w.out,w.neuron_len,
                activation.data_ptr(),tokens,output.data_ptr(),output_dtype==::tyloi::kFloat16,mfq_current_cuda_stream())) {}
        else if(interleaved)
            tyloi::cuda::gguf::matvec_interleaved(w.type,w.packed.data_ptr(),w.row_bytes,w.out,w.neuron_len,
                activation.data_ptr(),tokens,output.data_ptr(),output_dtype==::tyloi::kFloat16,mfq_current_cuda_stream());
        else if(output_dtype==::tyloi::kFloat16)
            tyloi::cuda::gguf::matvec_q8_f16(w.type,w.packed.data_ptr(),w.row_bytes,w.out,w.neuron_len,
                activation.data_ptr(),tokens,output.data_ptr(),mfq_current_cuda_stream());
        else tyloi::cuda::gguf::matvec_q8(w.type,w.packed.data_ptr(),w.row_bytes,w.out,w.neuron_len,
            activation.data_ptr(),tokens,output.data_ptr<float>(),mfq_current_cuda_stream());
    } else {
        namespace mmq=tyloi::cuda::gguf::mmq;
        const auto* mmq_setting=tyloi::cuda::env_lookup("TYLOINFER_GGUF_DENSE_MMQ");
        const auto* fast_setting=tyloi::cuda::env_lookup("TYLOINFER_GGUF_PREFILL_TENSORCORE");
        // Large prompts amortize decoding over many tokens. Keep MMQ for short
        // matrices and explicit non-MMQ reference calls. An explicit 1/0
        // permits enabling/disabling the reduced-accumulation GEMM mode.
        const bool fast_prefill=((fast_setting && fast_setting[0]=='1') ||
            ((!fast_setting || fast_setting[0]=='a') && tokens>=512 && allow_mmq)) &&
            tyloi::cuda::gguf::grouped_f16_available(w.packed.get_device());
        const auto mmq_rows=fast_prefill?0:tyloi::cuda::prefill_weight_rows(w.out,w.row_bytes,128,w.packed.get_device());
        const auto final_rows=mmq_rows>0?w.out%mmq_rows:0;
        if(allow_mmq && (!mmq_setting || mmq_setting[0]!='0') && mmq_rows>0 && tokens<=INT_MAX && w.neuron_len<=INT_MAX &&
                (dtype==::tyloi::kFloat16 || dtype==::tyloi::kFloat32) && mmq::fits(w.type,mmq_rows,w.packed.get_device()) &&
                (!final_rows || mmq::fits(w.type,final_rows,w.packed.get_device()))) {
            auto input_float=x.to(::tyloi::kFloat32).contiguous();
            ::tyloi::DenseTensor exponents,scaled;
            if(dtype==::tyloi::kFloat32) {
                scaled=::tyloi::cuda::empty_like(input_float);
                exponents=::tyloi::cuda::empty({tokens},w.packed.options().dtype(::tyloi::kInt32));
                tyloi::cuda::gguf::pack_scaled_f32(input_float.data_ptr<float>(),scaled.data_ptr<float>(),
                    exponents.data_ptr<std::int32_t>(),tokens,w.neuron_len,mfq_current_cuda_stream());
                input_float=scaled;
            }
            auto activation=::tyloi::cuda::empty({std::int64_t(mmq::activation_bytes(tokens,w.neuron_len))},w.packed.options());
            mmq::quantize(input_float.data_ptr<float>(),nullptr,activation.data_ptr(),int(w.type),w.neuron_len,tokens,mfq_current_cuda_stream());
            auto result=output_dtype==::tyloi::kFloat32?output: ::tyloi::cuda::empty({tokens,w.out},w.packed.options().dtype(::tyloi::kFloat32));
            const auto guard_bytes=std::size_t(128)*ggml_type_size(w.type);
            auto weight=::tyloi::cuda::empty({std::int64_t(mmq_rows*w.row_bytes+guard_bytes)},w.packed.options());
            for(const auto& tensor:{input_float,exponents,activation,result,weight})if(tensor.defined())mfq_cuda_record_stream(tensor,stream);
            mmq::Context context;
            for(std::int64_t begin=0;begin<w.out;begin+=mmq_rows) {
                const auto count=std::min(mmq_rows,w.out-begin);
                const auto bytes=std::uint64_t(count)*w.row_bytes;
                if(w.quad_storage_rows)tyloi::cuda::gguf::unpack_quad_storage(w.type,w.packed.data_ptr(),
                    w.quad_storage_rows,w.neuron_len,w.quad_row_begin+begin,count,nullptr,count,
                    weight.data_ptr(),w.row_bytes,mfq_current_cuda_stream());
                else TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(weight.data_ptr(),w.packed.data_ptr<std::uint8_t>()+begin*w.row_bytes,
                    bytes,cudaMemcpyDeviceToDevice,mfq_current_cuda_stream()));
                TYLOINFER_CUDA_CHECK(cudaMemsetAsync(weight.data_ptr<std::uint8_t>()+bytes,0,guard_bytes,mfq_current_cuda_stream()));
                context.run({weight.data_ptr(),int(w.type),count,w.neuron_len,std::size_t(bytes),1,activation.data_ptr(),nullptr,nullptr,
                    tokens,tokens,result.data_ptr<float>()+begin,w.out},mfq_current_cuda_stream());
            }
            if(exponents.defined())tyloi::cuda::gguf::rescale_f32(result.data_ptr<float>(),exponents.data_ptr<std::int32_t>(),
                tokens,w.out,mfq_current_cuda_stream());
            if(const auto* trace=tyloi::cuda::env_lookup("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported[GGML_TYPE_COUNT]{};
                if(!reported[int(w.type)].exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                    "native_gguf_dense_mmq actual_match=1 format=%d tokens=%lld rows=%lld columns=%lld scaled_fp32=%d tile_rows=%lld\n",
                    int(w.type),static_cast<long long>(tokens),static_cast<long long>(w.out),static_cast<long long>(w.neuron_len),int(exponents.defined()),static_cast<long long>(mmq_rows));
            }
            return result.to(dtype).reshape(shape);
        }
        // Prefill shares each decoded tile across every token. Temporary weight
        // expansion is bounded and never changes permanent compressed storage.
        const bool half = dtype == ::tyloi::kFloat16;
        const auto* setting=tyloi::cuda::env_lookup("TYLOINFER_GGUF_PREFILL_TENSORCORE");
        const bool scaled_half=!half && dtype==::tyloi::kFloat32 && (fast_prefill || !setting || setting[0]!='0') &&
            tyloi::cuda::gguf::grouped_f16_available(w.packed.get_device());
        x = x.to(half ? ::tyloi::kFloat16 : ::tyloi::kFloat32).contiguous();
        const auto cached_exponents=fast_prefill && scaled_half?prefill_cached_exponents(w): ::tyloi::DenseTensor{};
        const bool fused_scaled=cached_exponents.defined();
        // Include both decoded and optional scaled source images. Direct BLAS
        // output removes the tile result and its strided copy for tensor cores.
        const auto workspace_row_bytes=std::uint64_t(w.neuron_len)*(half || fused_scaled?2:4)+
            (scaled_half && !fused_scaled?std::uint64_t(w.neuron_len)*2:0)+(!half && !scaled_half?std::uint64_t(tokens)*4:0)+(fast_prefill?std::uint64_t(tokens)*2:0);
        const auto tile_rows=tyloi::cuda::prefill_weight_rows(w.out,workspace_row_bytes,32,w.packed.get_device(),true);
        cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
        TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture));
        const bool pipelined=tile_rows<w.out && capture==cudaStreamCaptureStatusNone;
        const auto decoder=pipelined?tyloi::cuda::stream_from_pool(false,w.packed.get_device()):stream;
        const int buffers=pipelined?2:1;
        ::tyloi::DenseTensor tile[2],scaled_weight[2],weight_exponents[2],scaled_x,x_exponents;
        auto tile_output=fast_prefill && !half?::tyloi::cuda::empty({tokens,tile_rows},x.options().dtype(::tyloi::kFloat16)): ::tyloi::DenseTensor{};
        if(tile_output.defined())mfq_cuda_record_stream(tile_output,stream);
        const auto pack_half=fast_prefill?tyloi::cuda::gguf::pack_scaled_f16_unit:tyloi::cuda::gguf::pack_scaled_f16;
        std::optional<tyloi::cuda::Event> ready[2],consumed[2];
        for(int i=0;i<buffers;++i) {
            tile[i]=::tyloi::cuda::empty({tile_rows,w.neuron_len},x.options().dtype(fused_scaled?::tyloi::kFloat16:x.scalar_type()));
            mfq_cuda_record_stream(tile[i],stream);mfq_cuda_record_stream(tile[i],decoder);
            if(scaled_half && !fused_scaled) {
                scaled_weight[i]=::tyloi::cuda::empty({tile_rows,w.neuron_len},x.options().dtype(::tyloi::kFloat16));
                weight_exponents[i]=::tyloi::cuda::empty({tile_rows},x.options().dtype(::tyloi::kInt32));
                for(const auto& tensor:{scaled_weight[i],weight_exponents[i]}) {
                    mfq_cuda_record_stream(tensor,stream);mfq_cuda_record_stream(tensor,decoder);
                }
            }
            if(pipelined){ready[i].emplace();consumed[i].emplace();}
        }
        const auto context=tyloi::cuda::default_context(w.packed.get_device());
        auto& blas=context->blas(mfq_current_cuda_stream());
        if(scaled_half && !(fast_prefill && GgufPrefillInputScope::lookup(x,scaled_x,x_exponents))) {
            scaled_x=::tyloi::cuda::empty(x.sizes(),x.options().dtype(::tyloi::kFloat16));
            x_exponents=::tyloi::cuda::empty({tokens},x.options().dtype(::tyloi::kInt32));
            for(const auto& tensor:{scaled_x,x_exponents})mfq_cuda_record_stream(tensor,stream);
            pack_half(x.data_ptr<float>(),scaled_x.data_ptr(),x_exponents.data_ptr<std::int32_t>(),
                tokens,w.neuron_len,mfq_current_cuda_stream());
        }
        if(pipelined) {
            tyloi::cuda::Event source_ready;source_ready.record(mfq_current_cuda_stream());
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(decoder.stream(),source_ready.get(),0));
            mfq_cuda_record_stream(w.packed,decoder);
        }
        const auto prepare=[&](std::int64_t index) {
            const auto begin=index*tile_rows,count=std::min(tile_rows,w.out-begin);
            const int slot=int(index%buffers);
            if(fused_scaled) {
                const auto* e=cached_exponents.data_ptr<std::int32_t>()+begin;
                if(w.quad_storage_rows)tyloi::cuda::gguf::decode_quad_rows_scaled(w.type,w.packed.data_ptr(),w.quad_storage_rows,
                    w.neuron_len,w.quad_row_begin+begin,count,e,tile[slot].data_ptr(),decoder.stream());
                else tyloi::cuda::gguf::dequantize_rows_scaled(w.type,w.packed.data_ptr<std::uint8_t>()+begin*w.row_bytes,
                    w.row_bytes,count,w.neuron_len,e,tile[slot].data_ptr(),decoder.stream());
                mfq_cuda_record_stream(cached_exponents,decoder);
            } else if(w.quad_storage_rows)tyloi::cuda::gguf::decode_quad_rows(w.type,w.packed.data_ptr(),w.quad_storage_rows,
                w.neuron_len,w.quad_row_begin+begin,count,nullptr,count,tile[slot].data_ptr(),half,decoder.stream());
            else tyloi::cuda::gguf::dequantize_rows(w.type,w.packed.data_ptr<std::uint8_t>()+begin*w.row_bytes,w.row_bytes,
                count,w.neuron_len,tile[slot].data_ptr(),half,decoder.stream());
            if(scaled_half && !fused_scaled)pack_half(tile[slot].data_ptr<float>(),scaled_weight[slot].data_ptr(),
                weight_exponents[slot].data_ptr<std::int32_t>(),count,w.neuron_len,decoder.stream());
            if(pipelined)ready[slot]->record(decoder.stream());
        };
        const auto tiles=(w.out+tile_rows-1)/tile_rows;
        DenseTensor residual_scratch;
        for(int i=0;i<buffers && i<tiles;++i)prepare(i);
        for(std::int64_t index=0;index<tiles;++index) {
            const auto begin=index*tile_rows,count=std::min(tile_rows,w.out-begin);
            const int slot=int(index%buffers);
            if(pipelined)TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(mfq_current_cuda_stream(),ready[slot]->get(),0));
            if(fast_prefill && half) {
                auto* destination=static_cast<std::uint8_t*>(output.data_ptr())+begin*output.element_size();
                if(prefill_residual.defined()) {
                    prefill_f16_residual_tile(x, tile[slot], prefill_residual, output,
                        begin, count, blas.get(), residual_scratch);
                }
                else tyloi::cuda::gguf::prefill_gemm_f16_accum(x.data_ptr(),tile[slot].data_ptr(),nullptr,
                    destination,true,nullptr,nullptr,tokens,count,w.neuron_len,w.out,blas.get(),mfq_current_cuda_stream());
            } else if(fast_prefill) {
                auto* destination=static_cast<std::uint8_t*>(output.data_ptr())+begin*output.element_size();
                tyloi::cuda::gguf::prefill_gemm_f16_accum(scaled_x.data_ptr(),fused_scaled?tile[slot].data_ptr():scaled_weight[slot].data_ptr(),
                    tile_output.data_ptr(),destination,output_dtype==::tyloi::kFloat16,
                    x_exponents.data_ptr<std::int32_t>(),fused_scaled?cached_exponents.data_ptr<std::int32_t>()+begin:weight_exponents[slot].data_ptr<std::int32_t>(),
                    tokens,count,w.neuron_len,w.out,blas.get(),mfq_current_cuda_stream());
            } else if(half || scaled_half) {
                auto* destination=static_cast<std::uint8_t*>(output.data_ptr())+begin*output.element_size();
                tyloi::cuda::gguf::prefill_gemm_f16(scaled_half?scaled_x.data_ptr():x.data_ptr(),
                    scaled_half?scaled_weight[slot].data_ptr():tile[slot].data_ptr(),destination,half,
                    scaled_half?x_exponents.data_ptr<std::int32_t>():nullptr,
                    scaled_half?weight_exponents[slot].data_ptr<std::int32_t>():nullptr,
                    tokens,count,w.neuron_len,w.out,blas.get(),mfq_current_cuda_stream());
            } else output.slice(1,begin,begin+count).copy_(dense_projection(x,tile[slot].narrow(0,0,count)).to(output_dtype));
            if(index+buffers<tiles) {
                if(pipelined) {
                    consumed[slot]->record(mfq_current_cuda_stream());
                    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(decoder.stream(),consumed[slot]->get(),0));
                }
                prepare(index+buffers);
            }
        }
        if(const auto* trace=tyloi::cuda::env_lookup("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<bool> reported[GGML_TYPE_COUNT][2]{};
            if(!reported[int(w.type)][int(pipelined)].exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                "native_gguf_prefill_stream actual_match=1 format=%d M=%lld N=%lld K=%lld tile_rows=%lld buffers=%d direct_output=%d f16_accum=%d fused_scale_decode=%d\n",
                int(w.type),static_cast<long long>(tokens),static_cast<long long>(w.out),static_cast<long long>(w.neuron_len),
                static_cast<long long>(tile_rows),buffers,int(half || scaled_half),int(fast_prefill),int(fused_scaled));
        }
    }
    return output.to(dtype).reshape(shape);
}

} // namespace tyloi::cuda::gguf
