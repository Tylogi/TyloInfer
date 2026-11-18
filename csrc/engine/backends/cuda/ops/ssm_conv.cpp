#include "backend.h"
#include "engine/ops/ops.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/kernels/recurrent/convolution_raw.h"
#include <cstdint>
#include <climits>
#include <cmath>
#include <vector>

::tyloi::DenseTensor ssm_conv_silu_cuda(::tyloi::DenseTensor conv_input, ::tyloi::DenseTensor weight, ::tyloi::DenseTensor bias, int64_t n_tokens)
{
    TYLOINFER_RUNTIME_CHECK(conv_input.is_cuda() && conv_input.is_contiguous() && conv_input.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu: conv_input must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu: weight must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu: bias must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(conv_input.dim() == 3, "ssm_conv_silu: conv_input must be [B,K-1+T,C]");
    int B = (int)conv_input.size(0);
    int C = (int)conv_input.size(2);
    int has_bias = (int)(bias.numel() > 0);
    TYLOINFER_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "ssm_conv_silu: bias must be [C]");
    int K = 0;
    int layout = 0; // 0: channel-major [C,1,K]/[C,K], 1: [K,C]
    if (weight.dim() == 3) {
        TYLOINFER_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1, "ssm_conv_silu: [C,1,K] weight mismatch");
        K = (int)weight.size(2);
        layout = 0;
    } else {
        TYLOINFER_RUNTIME_CHECK(weight.dim() == 2, "ssm_conv_silu: weight must be [C,1,K], [C,K], or [K,C]");
        if (weight.size(0) == C) {
            K = (int)weight.size(1);
            layout = 0;
        } else {
            TYLOINFER_RUNTIME_CHECK(weight.size(1) == C, "ssm_conv_silu: 2D weight must be [C,K] or [K,C]");
            K = (int)weight.size(0);
            layout = 1;
        }
    }
    int T = (int)n_tokens;
    TYLOINFER_RUNTIME_CHECK(T > 0 && conv_input.size(1) == T + K - 1, "ssm_conv_silu: conv_input length must be K-1+T");
    auto out = ::tyloi::cuda::empty({B, T, C}, conv_input.options());
    tyloi::cuda::ssm_conv_raw::Args a;
    a.input=conv_input.data_ptr<float>(); a.weight=weight.data_ptr<float>(); a.bias=bias.data_ptr<float>(); a.output=out.data_ptr<float>();
    a.batch=B; a.tokens=T; a.channels=C; a.kernel=K; a.layout=layout; a.has_bias=has_bias;
    tyloi::cuda::ssm_conv_raw::prefill(a, mfq_current_cuda_stream());
    return out;
}

::tyloi::DenseTensor ssm_conv_silu_decode_cuda(::tyloi::DenseTensor state, ::tyloi::DenseTensor x, ::tyloi::DenseTensor weight, ::tyloi::DenseTensor bias)
{
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu_decode: state must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu_decode: x must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu_decode: weight must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu_decode: bias must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(state.dim() == 3, "ssm_conv_silu_decode: state must be [B,K-1,C]");
    TYLOINFER_RUNTIME_CHECK(x.dim() == 3 && x.size(1) == 1, "ssm_conv_silu_decode: x must be [B,1,C]");
    int B = (int)state.size(0);
    int K = (int)state.size(1) + 1;
    int C = (int)state.size(2);
    TYLOINFER_RUNTIME_CHECK(x.size(0) == B && x.size(2) == C, "ssm_conv_silu_decode: x shape mismatch");
    int has_bias = (int)(bias.numel() > 0);
    TYLOINFER_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "ssm_conv_silu_decode: bias must be [C]");
    int layout = 0;
    if (weight.dim() == 3) {
        TYLOINFER_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "ssm_conv_silu_decode: [C,1,K] weight mismatch");
        layout = 0;
    } else {
        TYLOINFER_RUNTIME_CHECK(weight.dim() == 2,
                    "ssm_conv_silu_decode: weight must be [C,1,K], [C,K], or [K,C]");
        if (weight.size(0) == C && weight.size(1) == K) {
            layout = 0;
        } else {
            TYLOINFER_RUNTIME_CHECK(weight.size(0) == K && weight.size(1) == C,
                        "ssm_conv_silu_decode: 2D weight must be [C,K] or [K,C]");
            layout = 1;
        }
    }
    auto out = ::tyloi::cuda::empty({B, 1, C}, x.options());
    tyloi::cuda::ssm_conv_raw::Args a;
    a.state=state.data_ptr<float>(); a.input=x.data_ptr<float>(); a.weight=weight.data_ptr<float>(); a.bias=bias.data_ptr<float>(); a.output=out.data_ptr<float>();
    a.batch=B; a.channels=C; a.kernel=K; a.layout=layout; a.has_bias=has_bias;
    tyloi::cuda::ssm_conv_raw::decode(a, mfq_current_cuda_stream());
    return out;
}

std::vector<::tyloi::DenseTensor> linear_conv_qkv_gate_decode_cuda(
    ::tyloi::DenseTensor state,::tyloi::DenseTensor qk,::tyloi::DenseTensor v,
    ::tyloi::DenseTensor weight,::tyloi::DenseTensor alpha,::tyloi::DenseTensor beta,
    ::tyloi::DenseTensor dt_bias,::tyloi::DenseTensor a_log,
    int64_t nk,int64_t nv,int64_t dk,int64_t dv,double eps) {
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type()==::tyloi::kFloat32 &&
        state.dim()==3 && state.size(1)>0 && nk>0 && nv>0 && nv%nk==0 && dk>0 && dk<=256 && dv>0,
        "GDN combined decode state/geometry disagrees");
    const int B=int(state.size(0)),K=int(state.size(1))+1,C=int(2*nk*dk+nv*dv);
    const auto same_device=[&](const ::tyloi::DenseTensor& x){return x.is_cuda() && x.device()==state.device();};
    TYLOINFER_RUNTIME_CHECK(same_device(qk) && same_device(v) && qk.is_contiguous() && v.is_contiguous() &&
        qk.scalar_type()==::tyloi::kFloat16 && v.scalar_type()==::tyloi::kFloat16 &&
        qk.sizes().vec()==std::vector<int64_t>({B,1,2*nk*dk}) &&
        v.sizes().vec()==std::vector<int64_t>({B,1,nv*dv}) && state.size(2)==C,
        "GDN combined decode projection shape/dtype disagrees");
    TYLOINFER_RUNTIME_CHECK(same_device(alpha) && same_device(beta) && alpha.scalar_type()==beta.scalar_type() &&
        (alpha.scalar_type()==::tyloi::kFloat16 || alpha.scalar_type()==::tyloi::kFloat32) &&
        alpha.sizes().vec()==std::vector<int64_t>({B,1,nv}) && beta.sizes()==alpha.sizes(),
        "GDN combined decode gates disagree");
    TYLOINFER_RUNTIME_CHECK(same_device(weight) && same_device(dt_bias) && same_device(a_log) &&
        weight.is_contiguous() && dt_bias.is_contiguous() && a_log.is_contiguous() &&
        weight.scalar_type()==::tyloi::kFloat32 && dt_bias.scalar_type()==::tyloi::kFloat32 && a_log.scalar_type()==::tyloi::kFloat32 &&
        dt_bias.numel()==nv && a_log.numel()==nv,"GDN combined decode parameters disagree");
    const bool channels_first=weight.sizes().vec()==std::vector<int64_t>({C,1,K});
    TYLOINFER_RUNTIME_CHECK(channels_first || weight.sizes().vec()==std::vector<int64_t>({K,C}),
        "GDN combined decode convolution shape disagrees");
    MfqCudaGuard guard(state.device());auto options=state.options();
    auto q=::tyloi::cuda::empty({B,nk,1,dk},options),k=::tyloi::cuda::empty_like(q),output=::tyloi::cuda::empty({B,nv,1,dv},options);
    auto decay=::tyloi::cuda::empty({B,nv,1},options),beta_out=::tyloi::cuda::empty_like(decay);
    tyloi::cuda::ssm_conv_raw::Args a;
    a.state=state.data_ptr<float>(); a.qk=qk.data_ptr(); a.v=v.data_ptr(); a.weight=weight.data_ptr<float>();
    a.alpha=alpha.data_ptr(); a.beta=beta.data_ptr(); a.dt_bias=dt_bias.data_ptr<float>(); a.a_log=a_log.data_ptr<float>();
    a.q=q.data_ptr<float>(); a.k=k.data_ptr<float>(); a.output=output.data_ptr<float>(); a.decay=decay.data_ptr<float>(); a.beta_out=beta_out.data_ptr<float>();
    a.batch=B; a.nk=int(nk); a.nv=int(nv); a.dk=int(dk); a.dv=int(dv); a.kernel=K; a.layout=channels_first?0:1; a.eps=float(eps);
    a.half_gate=alpha.scalar_type()==::tyloi::kFloat16; a.a_stride0=alpha.stride(0); a.a_stride1=alpha.stride(2); a.b_stride0=beta.stride(0); a.b_stride1=beta.stride(2);
    tyloi::cuda::ssm_conv_raw::qkv_gate_decode(a, mfq_current_cuda_stream());
    return {q,k,output,decay,beta_out};
}

std::vector<::tyloi::DenseTensor> linear_conv_qkv_decode_cuda(
    ::tyloi::DenseTensor state,
    ::tyloi::DenseTensor qk,
    ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor weight,
    ::tyloi::DenseTensor bias,
    int64_t nk,
    int64_t nv,
    int64_t dk,
    int64_t dv,
    double eps)
{
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == ::tyloi::kFloat32,
                "linear_conv_qkv_decode: state must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(qk.is_cuda() && qk.is_contiguous() && qk.scalar_type() == ::tyloi::kHalf,
                "linear_conv_qkv_decode: qk must be cuda contiguous f16");
    TYLOINFER_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == ::tyloi::kHalf,
                "linear_conv_qkv_decode: v must be cuda contiguous f16");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == ::tyloi::kFloat32,
                "linear_conv_qkv_decode: weight must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == ::tyloi::kFloat32,
                "linear_conv_qkv_decode: bias must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(state.dim() == 3, "linear_conv_qkv_decode: state must be [B,K-1,C]");
    TYLOINFER_RUNTIME_CHECK(qk.dim() == 3 && qk.size(1) == 1, "linear_conv_qkv_decode: qk must be [B,1,2*nk*dk]");
    TYLOINFER_RUNTIME_CHECK(v.dim() == 3 && v.size(1) == 1, "linear_conv_qkv_decode: v must be [B,1,nv*dv]");
    int B = (int)state.size(0);
    int K = (int)state.size(1) + 1;
    int qkC = (int)(2 * nk * dk);
    int vsz = (int)(nv * dv);
    int C = qkC + vsz;
    TYLOINFER_RUNTIME_CHECK(qk.size(0) == B && qk.size(2) == qkC, "linear_conv_qkv_decode: qk shape mismatch");
    TYLOINFER_RUNTIME_CHECK(v.size(0) == B && v.size(2) == vsz, "linear_conv_qkv_decode: v shape mismatch");
    TYLOINFER_RUNTIME_CHECK(state.size(2) == C, "linear_conv_qkv_decode: state width mismatch");
    TYLOINFER_RUNTIME_CHECK(nv % nk == 0, "linear_conv_qkv_decode: nv must be divisible by nk");
    TYLOINFER_RUNTIME_CHECK(dk <= 256, "linear_conv_qkv_decode: dk > 256 is unsupported");
    int has_bias = (int)(bias.numel() > 0);
    TYLOINFER_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "linear_conv_qkv_decode: bias must be [C]");
    int layout = 0;
    if (weight.dim() == 3) {
        TYLOINFER_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "linear_conv_qkv_decode: [C,1,K] weight mismatch");
        layout = 0;
    } else {
        TYLOINFER_RUNTIME_CHECK(weight.dim() == 2 && weight.size(0) == K && weight.size(1) == C,
                    "linear_conv_qkv_decode: weight must be [C,1,K] or [K,C]");
        layout = 1;
    }
    auto opts = state.options();
    auto q = ::tyloi::cuda::empty({B, (int)nk, 1, (int)dk}, opts);
    auto k = ::tyloi::cuda::empty({B, (int)nk, 1, (int)dk}, opts);
    auto vo = ::tyloi::cuda::empty({B, (int)nv, 1, (int)dv}, opts);
    tyloi::cuda::ssm_conv_raw::Args a;
    a.state=state.data_ptr<float>(); a.qk=qk.data_ptr(); a.v=v.data_ptr(); a.weight=weight.data_ptr<float>(); a.bias=bias.data_ptr<float>();
    a.q=q.data_ptr<float>(); a.k=k.data_ptr<float>(); a.output=vo.data_ptr<float>();
    a.batch=B; a.nk=int(nk); a.nv=int(nv); a.dk=int(dk); a.dv=int(dv); a.kernel=K; a.layout=layout; a.has_bias=has_bias; a.eps=float(eps);
    tyloi::cuda::ssm_conv_raw::qkv_decode(a, mfq_current_cuda_stream());
    return {q,k,vo};
}

std::vector<::tyloi::DenseTensor> linear_conv_qkv_prefill_cuda(
    ::tyloi::DenseTensor state,
    ::tyloi::DenseTensor qk,
    ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor weight,
    ::tyloi::DenseTensor bias,
    int64_t nk,
    int64_t nv,
    int64_t dk,
    int64_t dv,
    double eps)
{
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == ::tyloi::kFloat32,
                "linear_conv_qkv_prefill: state must be contiguous CUDA f32");
    TYLOINFER_RUNTIME_CHECK(qk.is_cuda() && qk.is_contiguous() && (qk.scalar_type() == ::tyloi::kHalf || qk.scalar_type() == ::tyloi::kFloat32),
                "linear_conv_qkv_prefill: qk must be contiguous CUDA f16/f32");
    TYLOINFER_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == qk.scalar_type(),
                "linear_conv_qkv_prefill: v must match contiguous CUDA qk dtype");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == ::tyloi::kFloat32,
                "linear_conv_qkv_prefill: weight must be contiguous CUDA f32");
    TYLOINFER_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == ::tyloi::kFloat32,
                "linear_conv_qkv_prefill: bias must be contiguous CUDA f32");
    TYLOINFER_RUNTIME_CHECK(state.dim() == 3 && qk.dim() == 3 && v.dim() == 3,
                "linear_conv_qkv_prefill: state/qk/v must be rank 3");
    int B = (int)qk.size(0);
    int T = (int)qk.size(1);
    int K = (int)state.size(1) + 1;
    int qkC = (int)(2 * nk * dk);
    int vsz = (int)(nv * dv);
    int C = qkC + vsz;
    TYLOINFER_RUNTIME_CHECK(T > 1 && qk.size(2) == qkC && v.size(0) == B && v.size(1) == T && v.size(2) == vsz,
                "linear_conv_qkv_prefill: qk/v shape mismatch");
    TYLOINFER_RUNTIME_CHECK(state.size(0) == B && state.size(2) == C, "linear_conv_qkv_prefill: state shape mismatch");
    TYLOINFER_RUNTIME_CHECK(dk <= 256 && K <= 8, "linear_conv_qkv_prefill: requires dk<=256 and K<=8");
    int layout = 0;
    if (weight.dim() == 3) {
        TYLOINFER_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "linear_conv_qkv_prefill: [C,1,K] weight mismatch");
    } else {
        TYLOINFER_RUNTIME_CHECK(weight.dim() == 2 && weight.size(0) == K && weight.size(1) == C,
                    "linear_conv_qkv_prefill: weight must be [C,1,K] or [K,C]");
        layout = 1;
    }
    int has_bias = (int)(bias.numel() > 0);
    TYLOINFER_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "linear_conv_qkv_prefill: bias shape mismatch");

    auto opts = state.options();
    auto qo = ::tyloi::cuda::empty({B, (int)nk, T, (int)dk}, opts);
    auto ko = ::tyloi::cuda::empty_like(qo);
    auto vo = ::tyloi::cuda::empty({B, (int)nv, T, (int)dv}, opts);
    auto new_state = ::tyloi::cuda::empty_like(state);
    tyloi::cuda::ssm_conv_raw::Args a;
    a.state=state.data_ptr<float>(); a.qk=qk.data_ptr(); a.v=v.data_ptr(); a.weight=weight.data_ptr<float>(); a.bias=bias.data_ptr<float>();
    a.q=qo.data_ptr<float>(); a.k=ko.data_ptr<float>(); a.output=vo.data_ptr<float>(); a.next=new_state.data_ptr<float>();
    a.batch=B; a.tokens=T; a.channels=C; a.nk=int(nk); a.nv=int(nv); a.dk=int(dk); a.dv=int(dv); a.kernel=K; a.layout=layout; a.has_bias=has_bias; a.eps=float(eps);
    a.input_float=qk.scalar_type()==::tyloi::kFloat32;
    tyloi::cuda::ssm_conv_raw::qkv_prefill(a, mfq_current_cuda_stream());
    return {qo,ko,vo,new_state};
}

static std::vector<::tyloi::DenseTensor> linear_conv_qkv_gate_prefill_impl(
    ::tyloi::DenseTensor state,::tyloi::DenseTensor qkv,
    ::tyloi::DenseTensor weight,::tyloi::DenseTensor alpha,
    ::tyloi::DenseTensor beta,::tyloi::DenseTensor dt_bias,
    ::tyloi::DenseTensor a_log,int64_t nk,int64_t nv,int64_t width,double eps,
    bool stable_softplus,::tyloi::DenseTensor row_exponents,::tyloi::DenseTensor column_exponents)
{

    TYLOINFER_RUNTIME_CHECK(qkv.is_cuda() && (qkv.scalar_type()==::tyloi::kFloat16 || qkv.scalar_type()==::tyloi::kFloat32) &&
        qkv.dim()==3 && qkv.size(1)>=2 && qkv.size(1)<=INT_MAX && nk>0 && nv>0 && nv%nk==0 &&
        width>0 && width<=256 && qkv.stride(2)==1 && qkv.stride(1)>=qkv.size(2) &&
        (qkv.size(0)==1 || qkv.stride(0)>=qkv.size(1)*qkv.stride(1)),
        "GDN fused preparation requires row-strided f16/f32 QKV and at least 2 tokens");
    const int B=int(qkv.size(0)),T=int(qkv.size(1)),K=int(state.size(1))+1;
    const int qkC=int(2*nk*width),vsz=int(nv*width),C=qkC+vsz;
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type()==::tyloi::kFloat32 &&
        state.sizes().vec()==std::vector<int64_t>({B,K-1,C}) && K>=2 && K<=8 && qkv.size(2)==C,
        "short GDN convolution state geometry disagrees");
    for(const auto* tensor:{&weight,&dt_bias,&a_log})
        TYLOINFER_RUNTIME_CHECK(tensor->is_cuda() && tensor->is_contiguous() && tensor->scalar_type()==::tyloi::kFloat32 &&
            tensor->device()==qkv.device(),"short GDN parameters require contiguous float on the QKV device");
    TYLOINFER_RUNTIME_CHECK(alpha.is_cuda() && beta.is_cuda() &&
        alpha.device()==qkv.device() && beta.device()==qkv.device() && state.device()==qkv.device() &&
        alpha.sizes().vec()==std::vector<int64_t>({B,T,nv}) && beta.sizes()==alpha.sizes() &&
        alpha.stride(2)==1 && beta.stride(2)==1 && alpha.stride(1)>=nv && beta.stride(1)>=nv &&
        (B==1 || (alpha.stride(0)>=T*alpha.stride(1) && beta.stride(0)>=T*beta.stride(1))) &&
        beta.scalar_type()==alpha.scalar_type() &&
        (alpha.scalar_type()==::tyloi::kFloat16 || alpha.scalar_type()==::tyloi::kFloat32) &&
        dt_bias.numel()==nv && a_log.numel()==nv,"short GDN gate geometry disagrees");
    const bool scaled=row_exponents.defined();
    TYLOINFER_RUNTIME_CHECK(scaled==column_exponents.defined(),"scaled convolution requires both exponent vectors");
    if(scaled)for(const auto* tensor:{&row_exponents,&column_exponents})
        TYLOINFER_RUNTIME_CHECK(tensor->is_cuda() && tensor->device()==qkv.device() && tensor->is_contiguous() &&
            tensor->scalar_type()==::tyloi::kInt32,"scaled convolution exponents must be contiguous device int32");
    TYLOINFER_RUNTIME_CHECK(!scaled || (qkv.scalar_type()==::tyloi::kFloat16 && row_exponents.numel()==int64_t(B)*T &&
        column_exponents.numel()==C),"scaled convolution exponent geometry disagrees");
    const auto* re=scaled?row_exponents.data_ptr<int32_t>():nullptr;
    const auto* ce=scaled?column_exponents.data_ptr<int32_t>():nullptr;
    const int layout=weight.dim()==2?1:0;
    TYLOINFER_RUNTIME_CHECK(layout?weight.sizes().vec()==std::vector<int64_t>({K,C}):
        weight.sizes().vec()==std::vector<int64_t>({C,1,K}),"short GDN convolution weight geometry disagrees");
    auto options=state.options();
    auto q=::tyloi::cuda::empty({B,nk,T,width},options),k=::tyloi::cuda::empty_like(q),v=::tyloi::cuda::empty({B,nv,T,width},options);
    auto next=::tyloi::cuda::empty_like(state),decay=::tyloi::cuda::empty({B,nv,T},options),bt=::tyloi::cuda::empty_like(decay);
    tyloi::cuda::ssm_conv_raw::Args a;
    a.state=state.data_ptr<float>();a.qk=qkv.data_ptr();a.weight=weight.data_ptr<float>();a.alpha=alpha.data_ptr();a.beta=beta.data_ptr();
    a.dt_bias=dt_bias.data_ptr<float>();a.a_log=a_log.data_ptr<float>();a.q=q.data_ptr<float>();a.k=k.data_ptr<float>();a.output=v.data_ptr<float>();
    a.next=next.data_ptr<float>();a.decay=decay.data_ptr<float>();a.beta_out=bt.data_ptr<float>();a.row_exponents=re;a.column_exponents=ce;
    a.batch=B;a.tokens=T;a.channels=C;a.nk=int(nk);a.nv=int(nv);a.dk=int(width);a.dv=int(width);a.kernel=K;a.layout=layout;a.eps=float(eps);
    a.input_float=qkv.scalar_type()==::tyloi::kFloat32;a.half_gate=alpha.scalar_type()==::tyloi::kFloat16;a.stable_softplus=stable_softplus;
    a.q_stride0=qkv.stride(0);a.q_stride1=qkv.stride(1);a.a_stride0=alpha.stride(0);a.a_stride1=alpha.stride(1);a.b_stride0=beta.stride(0);a.b_stride1=beta.stride(1);
    const auto stream = mfq_get_current_cuda_stream(qkv.get_device());
    for (const auto& tensor : {state, qkv, weight, alpha, beta, dt_bias, a_log,
                                row_exponents, column_exponents, q, k, v, next, decay, bt})
        if (tensor.defined()) mfq_cuda_record_stream(tensor, stream);
    tyloi::cuda::ssm_conv_raw::qkv_gate_short(a,stream.stream());
    return {q,k,v,next,decay,bt};
}

std::vector<::tyloi::DenseTensor> linear_conv_qkv_gate_prefill_cuda(
    ::tyloi::DenseTensor state,::tyloi::DenseTensor qkv,
    ::tyloi::DenseTensor weight,::tyloi::DenseTensor alpha,::tyloi::DenseTensor beta,
    ::tyloi::DenseTensor dt_bias,::tyloi::DenseTensor a_log,int64_t nk,int64_t nv,int64_t width,
    double eps,bool stable_softplus) {
    return linear_conv_qkv_gate_prefill_impl(state,qkv,weight,alpha,beta,dt_bias,a_log,nk,nv,width,eps,stable_softplus,{},{});
}

std::vector<::tyloi::DenseTensor> linear_conv_qkv_gate_scaled_prefill_cuda(
    ::tyloi::DenseTensor state,::tyloi::DenseTensor product,
    ::tyloi::DenseTensor row_exponents,::tyloi::DenseTensor column_exponents,
    ::tyloi::DenseTensor weight,::tyloi::DenseTensor alpha,::tyloi::DenseTensor beta,
    ::tyloi::DenseTensor dt_bias,::tyloi::DenseTensor a_log,int64_t nk,int64_t nv,int64_t width,
    double eps,bool stable_softplus) {
    return linear_conv_qkv_gate_prefill_impl(state,product,weight,alpha,beta,dt_bias,a_log,nk,nv,width,eps,stable_softplus,
        row_exponents,column_exponents);
}

namespace tyloi::ops {
std::optional<LinearPrefill> linear_prefill(
    Tensor& convolution, const Tensor& qkv, const ScaledProjection& scaled,
    const Tensor& alpha, const Tensor& beta, const Tensor& conv_weight,
    const Tensor& dt_bias, const Tensor& a_log, std::int64_t nk, std::int64_t nv,
    std::int64_t width, double eps) {
    auto* state = convolution.representation<DenseTensor>();
    const auto* projected = (scaled ? scaled.values : qkv).representation<DenseTensor>();
    const auto* a = alpha.representation<DenseTensor>();
    const auto* b = beta.representation<DenseTensor>();
    const auto* weight = conv_weight.representation<DenseTensor>();
    const auto* dt = dt_bias.representation<DenseTensor>();
    const auto* log = a_log.representation<DenseTensor>();
    if (!state || !projected || !a || !b || !weight || !dt || !log ||
        !state->defined() || !state->is_cuda() || state->is_encoded() || !state->is_contiguous() ||
        state->scalar_type() != kFloat32 || state->dim() != 3 || a->dim() != 3 ||
        nk <= 0 || nv <= 0 || nv % nk || width <= 0 || width > 256 ||
        !std::isfinite(float(eps)) || float(eps) <= 0 ||
        nk > INT_MAX / 2 || nv > INT_MAX - 2 * nk || 2 * nk + nv > INT_MAX / width)
        return std::nullopt;
    const auto batch = a->size(0), tokens = a->size(1), channels = (2 * nk + nv) * width;
    const auto taps = state->size(1) + 1;
    if (batch <= 0 || batch > INT_MAX || tokens <= 1 || tokens > INT_MAX - 15 ||
        taps < 2 || taps > 8 || state->size(0) != batch || state->size(2) != channels ||
        a->size(2) != nv || b->dim() != 3 || b->sizes() != a->sizes())
        return std::nullopt;
    const auto same_device = [&](const DenseTensor& value) {
        return value.defined() && value.is_cuda() && value.device() == state->device() && !value.is_encoded();
    };
    const auto row_strided = [&](const DenseTensor& value, std::int64_t columns) {
        return same_device(value) && value.dim() == 3 && value.size(0) == batch &&
            value.size(1) == tokens && value.size(2) == columns && value.stride(2) == 1 &&
            value.stride(1) >= columns && (batch == 1 || value.stride(0) / tokens >= value.stride(1)) &&
            (value.scalar_type() == kFloat16 || value.scalar_type() == kFloat32);
    };
    if (!row_strided(*a, nv) || !row_strided(*b, nv) || a->scalar_type() != b->scalar_type())
        return std::nullopt;
    for (const auto* value : {weight, dt, log})
        if (!same_device(*value) || !value->is_contiguous() || value->scalar_type() != kFloat32)
            return std::nullopt;
    if (dt->numel() != nv || log->numel() != nv) return std::nullopt;
    const bool channel_major = weight->sizes().vec() == std::vector<int64_t>({channels, taps});
    if (!channel_major && weight->sizes().vec() != std::vector<int64_t>({channels, 1, taps}) &&
        weight->sizes().vec() != std::vector<int64_t>({taps, channels}))
        return std::nullopt;
    const DenseTensor *rows = nullptr, *columns = nullptr;
    DenseTensor input;
    if (scaled) {
        rows = scaled.row_exponents.representation<DenseTensor>();
        columns = scaled.column_exponents.representation<DenseTensor>();
        if (!same_device(*projected) || projected->scalar_type() != kFloat16 || !projected->is_contiguous() ||
            projected->dim() != 2 || projected->size(0) != batch * tokens || projected->size(1) != channels ||
            !rows || !columns || !same_device(*rows) || !same_device(*columns) ||
            !rows->is_contiguous() || !columns->is_contiguous() || rows->scalar_type() != kInt32 ||
            columns->scalar_type() != kInt32 || rows->numel() != batch * tokens || columns->numel() != channels)
            return std::nullopt;
        input = projected->reshape({batch, tokens, channels});
    } else {
        if (!row_strided(*projected, channels)) return std::nullopt;
        input = *projected;
    }
    const auto heads = 2 * nk + nv;
    if (tokens >= 256) {
        const auto tiles = (tokens + 15) / 16;
        if (heads > INT_MAX / tiles || batch > INT_MAX / (heads * tiles)) return std::nullopt;
    } else if (batch * tokens > INT_MAX / (2 * nk) ||
               nv * width > 65535 * 256 - 255 || batch > (INT_MAX - 255) / ((taps - 1) * channels)) {
        return std::nullopt;
    }
    if (!cuda::Executor::current().mode().linear.fused_conv) return std::nullopt;
    MfqCudaGuard guard(state->device());
    auto kernel_weight = channel_major ? weight->reshape({channels, 1, taps}) : *weight;
    auto result = linear_conv_qkv_gate_prefill_impl(*state, input, kernel_weight, *a, *b, *dt, *log,
        nk, nv, width, eps, false, rows ? *rows : DenseTensor{}, columns ? *columns : DenseTensor{});
    state->copy_(result[3]);
    return LinearPrefill{{Tensor(result[0]), Tensor(result[1]), Tensor(result[2])},
                         {Tensor(result[4]), Tensor(result[5])}};
}
} // namespace tyloi::ops

std::vector<::tyloi::DenseTensor> linear_conv_qkv_gate_prefill_cuda(
    ::tyloi::DenseTensor state,::tyloi::DenseTensor qkv,
    ::tyloi::DenseTensor weight,::tyloi::DenseTensor alpha,::tyloi::DenseTensor beta,
    ::tyloi::DenseTensor dt_bias,::tyloi::DenseTensor a_log,int64_t nk,int64_t nv,int64_t width,double eps) {
    return linear_conv_qkv_gate_prefill_cuda(state,qkv,weight,alpha,beta,dt_bias,a_log,nk,nv,width,eps,true);
}

std::vector<::tyloi::DenseTensor> linear_conv_qkv_gate_short_cuda(
    ::tyloi::DenseTensor state,::tyloi::DenseTensor qkv,
    ::tyloi::DenseTensor weight,::tyloi::DenseTensor alpha,::tyloi::DenseTensor beta,
    ::tyloi::DenseTensor dt_bias,::tyloi::DenseTensor a_log,int64_t nk,int64_t nv,int64_t width,double eps) {
    return linear_conv_qkv_gate_prefill_cuda(state,qkv,weight,alpha,beta,dt_bias,a_log,nk,nv,width,eps);
}

std::vector<::tyloi::DenseTensor> linear_conv_qkv_decode_f32_cuda(
        ::tyloi::DenseTensor state,::tyloi::DenseTensor input,
        ::tyloi::DenseTensor weight,::tyloi::DenseTensor bias,
        int64_t nk,int64_t nv,int64_t dk,int64_t dv,double eps) {

    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type()==::tyloi::kFloat32 &&
        state.dim()==3 && state.size(0)>0 && state.size(1)>0 && nk>0 && nv>0 && nv%nk==0 && dk>0 && dk<=256 && dv>0,
        "F32 convolution decode state/geometry disagrees");
    const auto batch=state.size(0),taps=state.size(1)+1,width=2*nk*dk+nv*dv;
    TYLOINFER_RUNTIME_CHECK(batch<=65535 && width<=INT_MAX && taps<=INT_MAX &&
        2*nk+(nv*dv+255)/256<=INT_MAX,"F32 convolution decode exceeds CUDA grid bounds");
    const auto same=[&](const ::tyloi::DenseTensor& x){return x.is_cuda() && x.device()==state.device() &&
        x.is_contiguous() && x.scalar_type()==::tyloi::kFloat32;};
    TYLOINFER_RUNTIME_CHECK(same(input) && input.sizes().vec()==std::vector<int64_t>({batch,1,width}) &&
        state.size(2)==width,"F32 convolution decode projection disagrees");
    TYLOINFER_RUNTIME_CHECK(same(weight) && same(bias) &&
        (bias.numel()==0 || (bias.dim()==1 && bias.numel()==width)),"F32 convolution decode parameters disagree");
    const bool channels_first=weight.sizes().vec()==std::vector<int64_t>({width,1,taps}) ||
        weight.sizes().vec()==std::vector<int64_t>({width,taps});
    TYLOINFER_RUNTIME_CHECK(channels_first || weight.sizes().vec()==std::vector<int64_t>({taps,width}),
        "F32 convolution decode weight geometry disagrees");
    MfqCudaGuard guard(state.device());
    auto q=::tyloi::cuda::empty({batch,nk,1,dk},state.options()),k=::tyloi::cuda::empty_like(q),v=::tyloi::cuda::empty({batch,nv,1,dv},state.options());
    tyloi::cuda::ssm_conv_raw::Args a;
    a.state=state.data_ptr<float>();a.input=input.data_ptr<float>();a.weight=weight.data_ptr<float>();
    a.bias=bias.numel()?bias.data_ptr<float>():nullptr;a.q=q.data_ptr<float>();a.k=k.data_ptr<float>();a.output=v.data_ptr<float>();
    a.batch=int(batch);a.nk=int(nk);a.nv=int(nv);a.dk=int(dk);a.dv=int(dv);a.kernel=int(taps);a.layout=channels_first?0:1;a.eps=float(eps);
    tyloi::cuda::ssm_conv_raw::qkv_decode_f32(a,mfq_current_cuda_stream());
    return {q,k,v};
}
