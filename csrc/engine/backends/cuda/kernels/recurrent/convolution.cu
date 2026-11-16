// Fused causal depthwise SSM conv + SiLU.
// conv_input [B, K - 1 + T, C], weight [C,1,K], [C,K], or [K,C], output [B,T,C].

#include <cuda_runtime.h>
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/native/env.h"
#include "convolution_raw.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../reduce.cuh"
#include "engine/backends/cuda/kernels/projection/gguf/prefill/scale.cuh"

constexpr int SSM_CONV_BD = 256;

// Unified F32 QKV projections can use the same fused decode preparation as
// split half projections. Keep the standalone convolution and L2 reduction
// orders, and update each state channel in exactly one CTA.
__global__ void ssm_conv_qkv_f32_decode_kernel(float* state,const float* input,
        const float* weight,const float* bias,float* q,float* k,float* v,
        int nk,int nv,int dk,int dv,int taps,bool channels_first,float eps) {
    const int tid=threadIdx.x,batch=blockIdx.y,qk_width=2*nk*dk,width=qk_width+nv*dv;
    const bool query_key=blockIdx.x<unsigned(2*nk);
    const int channel=query_key?int(blockIdx.x)*dk+tid:qk_width+int(blockIdx.x-2*nk)*256+tid;
    const bool valid=query_key?tid<dk:channel<width;
    float value=0.f;
    if(valid) {
        const auto base=std::int64_t(batch)*(taps-1)*width+channel;
        float sum=0.f;
        for(int j=0;j<taps-1;++j) {
            const float w=weight[channels_first?std::int64_t(channel)*taps+j:std::int64_t(j)*width+channel];
            sum+=state[base+std::int64_t(j)*width]*w;
        }
        const float current=input[std::int64_t(batch)*width+channel];
        sum+=current*weight[channels_first?std::int64_t(channel)*taps+taps-1:std::int64_t(taps-1)*width+channel];
        if(bias)sum+=bias[channel];
        value=sum/(1.f+expf(-sum));
        for(int j=0;j<taps-2;++j)state[base+std::int64_t(j)*width]=state[base+std::int64_t(j+1)*width];
        state[base+std::int64_t(taps-2)*width]=current;
    }
    if(query_key) {
        const float squared=block_sum<8>(value*value);
        const float inverse=1.f/fmaxf(sqrtf(squared),eps);
        if(valid) {
            const int which=int(blockIdx.x)/nk,head=int(blockIdx.x)%nk;
            (which?k:q)[(std::int64_t(batch)*nk+head)*dk+tid]=value*inverse;
        }
    }else if(valid)v[std::int64_t(batch)*nv*dv+channel-qk_width]=value;
}



__device__ inline float silu_f32(float x)
{
    return x / (1.0f + expf(-x));
}

__global__ void ssm_conv_silu_tiled_kernel(const float* __restrict__ x,const float* __restrict__ w,
    const float* __restrict__ bias,float* __restrict__ out,int B,int T,int C,int K,int weight_layout,int has_bias) {
    constexpr int Tile=4;
    const auto channels=(std::uint64_t(C)+SSM_CONV_BD-1)/SSM_CONV_BD;
    const auto times=(std::uint64_t(T)+Tile-1)/Tile;
    const auto total=std::uint64_t(B)*channels*times;
    for(auto tile=std::uint64_t(blockIdx.x);tile<total;tile+=gridDim.x) {
        const auto c=(tile%channels)*SSM_CONV_BD+threadIdx.x;
        const auto bt=tile/channels,b=bt/times,t=(bt%times)*Tile;
        if(c>=std::uint64_t(C))continue;
        float sums[Tile]={};
        for(int j=0;j<K;++j) {
            const float wv=weight_layout==0?w[c*K+j]:w[std::uint64_t(j)*C+c];
            #pragma unroll
            for(int r=0;r<Tile;++r)if(t+r<std::uint64_t(T))
                sums[r]+=x[(b*std::uint64_t(T+K-1)+t+r+j)*C+c]*wv;
        }
        #pragma unroll
        for(int r=0;r<Tile;++r)if(t+r<std::uint64_t(T)) {
            if(has_bias)sums[r]+=bias[c];
            out[(b*T+t+r)*C+c]=silu_f32(sums[r]);
        }
    }
}



__global__ void ssm_conv_silu_decode_kernel(
    float* __restrict__ state,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B,
    int C,
    int K,
    int weight_layout,
    int has_bias)
{
    int b = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || c >= C) {
        return;
    }
    size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
    float sum = 0.0f;
    for (int j = 0; j < K - 1; ++j) {
        float xv = state[state_base + (size_t)j * (size_t)C];
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    float x_cur = x[(size_t)b * (size_t)C + (size_t)c];
    float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
    sum += x_cur * wv;
    if (has_bias) {
        sum += bias[c];
    }
    out[(size_t)b * (size_t)C + (size_t)c] = silu_f32(sum);
    for (int j = 0; j < K - 2; ++j) {
        state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
    }
    state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
}



__device__ inline float ssm_decode_cur(
    const __half* __restrict__ qk,
    const __half* __restrict__ v,
    int b,
    int c,
    int qkC,
    int vsz)
{
    if (c < qkC) {
        return __half2float(qk[(size_t)b * (size_t)qkC + (size_t)c]);
    }
    return __half2float(v[(size_t)b * (size_t)vsz + (size_t)(c - qkC)]);
}

template <int BD>
__device__ __forceinline__ void ssm_conv_qk_norm_decode(
    float* __restrict__ state,
    const __half* __restrict__ qk,
    const __half* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ q,
    float* __restrict__ k,
    int B,
    int nk,
    int nv,
    int dk,
    int dv,
    int K,
    int weight_layout,
    int has_bias,
    float eps, int row)
{
    int b = row / (2 * nk);
    int rem = row - b * 2 * nk;
    int which = rem / nk;
    int h = rem - which * nk;
    int tid = threadIdx.x;
    int qkC = 2 * nk * dk;
    int vsz = nv * dv;
    int C = qkC + nv * dv;
    int base_c = (which * nk + h) * dk;

    __shared__ float vals[BD];
    __shared__ float sums[BD];
    float val = 0.0f;
    if (tid < dk) {
        int c = base_c + tid;
        size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
        float sum = 0.0f;
        for (int j = 0; j < K - 1; ++j) {
            float xv = state[state_base + (size_t)j * (size_t)C];
            float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
            sum += xv * wv;
        }
        float x_cur = ssm_decode_cur(qk, v, b, c, qkC, vsz);
        float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
        sum += x_cur * wv;
        if (has_bias) {
            sum += bias[c];
        }
        val = silu_f32(sum);
        vals[tid] = val;
        for (int j = 0; j < K - 2; ++j) {
            state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
        }
        state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
    }
    float ssq = (tid < dk) ? val * val : 0.0f;
    sums[tid] = ssq;
    __syncthreads();
    for (int stride = BD / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        __syncthreads();
    }
    float inv = 1.0f / fmaxf(sqrtf(sums[0]), eps);
    if (tid < dk) {
        float outv = vals[tid] * inv;
        float* dst = which == 0 ? q : k;
        dst[((size_t)b * nk + h) * dk + tid] = outv;
    }
}

__device__ __forceinline__ void ssm_conv_v_decode(
    float* __restrict__ state,
    const __half* __restrict__ qk,
    const __half* __restrict__ v_in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ v_out,
    int B,
    int nk,
    int nv,
    int dk,
    int dv,
    int K,
    int weight_layout,
    int has_bias, int b, int idx)
{
    int vsz = nv * dv;
    if (b >= B || idx >= vsz) {
        return;
    }
    int qkC = 2 * nk * dk;
    int C = qkC + vsz;
    int c = qkC + idx;
    size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
    float sum = 0.0f;
    for (int j = 0; j < K - 1; ++j) {
        float xv = state[state_base + (size_t)j * (size_t)C];
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    float x_cur = __half2float(v_in[(size_t)b * (size_t)vsz + (size_t)idx]);
    float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
    sum += x_cur * wv;
    if (has_bias) {
        sum += bias[c];
    }
    v_out[(size_t)b * (size_t)vsz + (size_t)idx] = silu_f32(sum);
    for (int j = 0; j < K - 2; ++j) {
        state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
    }
    state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
}

template<int BD>
__global__ void ssm_conv_qk_norm_decode_kernel(float* state,const __half* qk,const __half* v,
    const float* w,const float* bias,float* q,float* k,int B,int nk,int nv,int dk,int dv,
    int K,int layout,int has_bias,float eps) {
    ssm_conv_qk_norm_decode<BD>(state,qk,v,w,bias,q,k,B,nk,nv,dk,dv,K,layout,has_bias,eps,blockIdx.x);
}
__global__ void ssm_conv_v_decode_kernel(float* state,const __half* qk,const __half* v,
    const float* w,const float* bias,float* output,int B,int nk,int nv,int dk,int dv,
    int K,int layout,int has_bias) {
    ssm_conv_v_decode(state,qk,v,w,bias,output,B,nk,nv,dk,dv,K,layout,has_bias,
        blockIdx.x,blockIdx.y*blockDim.x+threadIdx.x);
}

template<class Gate>
__global__ void ssm_conv_qkv_gate_decode_kernel(float* state,const __half* qk,const __half* v,
    const float* weight,const Gate* alpha,const Gate* beta,const float* dt_bias,const float* a_log,
    float* q,float* k,float* output,float* decay,float* beta_out,
    int B,int nk,int nv,int dk,int dv,int K,int layout,float eps,
    int64_t as0,int64_t as2,int64_t bs0,int64_t bs2) {
    const int v_blocks=(nv*dv+255)/256;
    if(blockIdx.x<unsigned(2*nk)) {
        ssm_conv_qk_norm_decode<256>(state,qk,v,weight,nullptr,q,k,B,nk,nv,dk,dv,K,layout,0,
            eps,blockIdx.y*2*nk+blockIdx.x);
    } else if(blockIdx.x<unsigned(2*nk+v_blocks)) {
        ssm_conv_v_decode(state,qk,v,weight,nullptr,output,B,nk,nv,dk,dv,K,layout,0,
            blockIdx.y,(blockIdx.x-2*nk)*256+threadIdx.x);
    } else {
        const int head=(blockIdx.x-2*nk-v_blocks)*256+threadIdx.x,b=blockIdx.y;
        if(head<nv) {
            const float a=static_cast<float>(alpha[int64_t(b)*as0+int64_t(head)*as2])+dt_bias[head];
            const float sp=fmaxf(a,0.0f)+log1pf(expf(-fabsf(a)));
            decay[int64_t(b)*nv+head]=sp*-expf(a_log[head]);
            const float bv=static_cast<float>(beta[int64_t(b)*bs0+int64_t(head)*bs2]);
            beta_out[int64_t(b)*nv+head]=1.0f/(1.0f+expf(-bv));
        }
    }
}





template <int BD, bool Packed = false, class Input = __half>
__global__ void linear_conv_qk_norm_prefill_kernel(
    const float* __restrict__ state,
    const Input* __restrict__ qk,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ q,
    float* __restrict__ k,
    int B, int T, int nk, int dk, int C, int K,
    int weight_layout, int has_bias, float eps,int64_t ps0=0,int64_t ps1=0,
    const int32_t* row_exponents=nullptr,const int32_t* column_exponents=nullptr)
{
    int row = blockIdx.x;
    int h = row % nk;
    int r = row / nk;
    int which = r & 1;
    r >>= 1;
    int t = r % T;
    int b = r / T;
    int tid = threadIdx.x;
    int qkC = 2 * nk * dk;
    int c = (which * nk + h) * dk + tid;
    float val = 0.0f;
    if (tid < dk) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j >= K) break;
            int token = t + j - (K - 1);
            float xv;
            if (token < 0) {
                xv = state[((size_t)b * (K - 1) + (token + K - 1)) * C + c];
            } else {
                xv = static_cast<float>(qk[Packed && ps1?int64_t(b)*ps0+int64_t(token)*ps1+c:
                    ((size_t)b * T + token) * (Packed ? C : qkC) + c]);
            }
            if(row_exponents && token>=0)
                xv=prefill_ldexp(xv,row_exponents[int64_t(b)*T+token]+column_exponents[c]);
            float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
            sum += xv * wv;
        }
        if (has_bias) sum += bias[c];
        val = silu_f32(sum);
    }
    float ssq = block_sum<BD / 32>(tid < dk ? val * val : 0.0f);
    float inv = 1.0f / fmaxf(sqrtf(ssq), eps);
    if (tid < dk) {
        float* dst = which == 0 ? q : k;
        dst[(((size_t)b * nk + h) * T + t) * dk + tid] = val * inv;
    }
}

template <bool Packed = false, class Gate = float, class Input = __half>
__global__ void linear_conv_v_prefill_kernel(
    const float* __restrict__ state,
    const Input* __restrict__ v_in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ v_out,
    int B, int T, int nk, int nv, int dk, int dv, int C, int K,
    int weight_layout, int has_bias,
    const Gate* alpha = nullptr, const Gate* beta = nullptr,
    const float* dt_bias = nullptr, const float* a_log = nullptr,
    float* gate_out = nullptr, float* beta_out = nullptr,
    int64_t ps0=0,int64_t ps1=0,int64_t as0=0,int64_t as1=0,int64_t bs0=0,int64_t bs1=0,
    bool stable_softplus=true,const int32_t* row_exponents=nullptr,const int32_t* column_exponents=nullptr)
{
    int bt = blockIdx.x;
    int b = bt / T;
    int t = bt - b * T;
    int vi = blockIdx.y * blockDim.x + threadIdx.x;
    int vsz = nv * dv;
    if (b >= B || vi >= vsz) return;
    int qkC = 2 * nk * dk;
    int c = qkC + vi;
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j >= K) break;
        int token = t + j - (K - 1);
        float xv;
        if (token < 0) {
            xv = state[((size_t)b * (K - 1) + (token + K - 1)) * C + c];
        } else {
            xv = static_cast<float>(v_in[Packed && ps1?int64_t(b)*ps0+int64_t(token)*ps1+vi:
                ((size_t)b * T + token) * (Packed ? C : vsz) + vi]);
        }
        if(row_exponents && token>=0)
            xv=prefill_ldexp(xv,row_exponents[int64_t(b)*T+token]+column_exponents[c]);
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    if (has_bias) sum += bias[c];
    int h = vi / dv;
    int d = vi - h * dv;
    v_out[(((size_t)b * nv + h) * T + t) * dv + d] = silu_f32(sum);
    if constexpr(Packed) {
        if(d == 0) {
            const float a = static_cast<float>(alpha[as1?int64_t(b)*as0+int64_t(t)*as1+h:(size_t)bt * nv + h]) + dt_bias[h];
            const float sp = stable_softplus ? fmaxf(a,0.f) + log1pf(expf(-fabsf(a)))
                                             : (a>20.f ? a : log1pf(expf(a)));
            const auto index = ((size_t)b * nv + h) * T + t;
            gate_out[index] = sp * -expf(a_log[h]);
            const float raw = static_cast<float>(beta[bs1?int64_t(b)*bs0+int64_t(t)*bs1+h:(size_t)bt * nv + h]);
            beta_out[index] = 1.f / (1.f + expf(-raw));
        }
    }
}

template <bool Packed = false, class Input = __half>
__device__ __forceinline__ float linear_prefill_input(
    const Input* qk, const Input* v, int b, int t, int c,
    int T, int qkC, int vsz,int64_t ps0=0,int64_t ps1=0)
{
    if (c < qkC) return static_cast<float>(qk[Packed && ps1?int64_t(b)*ps0+int64_t(t)*ps1+c:
        ((size_t)b * T + t) * (Packed ? qkC+vsz : qkC) + c]);
    return static_cast<float>(v[Packed && ps1?int64_t(b)*ps0+int64_t(t)*ps1+(c-qkC):
        ((size_t)b * T + t) * (Packed ? qkC+vsz : vsz) + (c - qkC)]);
}

template <bool Packed = false, class Input = __half>
__global__ void linear_conv_state_prefill_kernel(
    const float* __restrict__ old_state,
    const Input* __restrict__ qk,
    const Input* __restrict__ v,
    float* __restrict__ new_state,
    int B, int T, int qkC, int vsz, int C, int K,int64_t ps0=0,int64_t ps1=0,
    const int32_t* row_exponents=nullptr,const int32_t* column_exponents=nullptr)
{
    size_t total = (size_t)B * (K - 1) * C;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        int c = (int)(idx % C);
        size_t br = idx / C;
        int j = (int)(br % (K - 1));
        int b = (int)(br / (K - 1));
        int combined = T + j;
        if (combined < K - 1) {
            new_state[idx] = old_state[((size_t)b * (K - 1) + combined) * C + c];
        } else {
            int token = combined - (K - 1);
            const auto value=linear_prefill_input<Packed,Input>(qk, v, b, token, c, T, qkC, vsz,ps0,ps1);
            new_state[idx] = row_exponents?prefill_ldexp(value,row_exponents[int64_t(b)*T+token]+column_exponents[c]):value;
        }
    }
}

// A temporal tile restores each projected value once, then shares it across
// every convolution tap. Q/K normalization writes the recurrent head layout.
template<int RowThreads,class Input,class Gate,bool Scaled>
__global__ void packed_conv_prefill_tile(const float* state,const Input* projected,
    const float* weight,const Gate* alpha,const Gate* beta,const float* dt_bias,const float* a_log,
    const int32_t* row_exponents,const int32_t* column_exponents,
    float* q,float* k,float* v,float* next,float* decay,float* bt,
    int T,int nk,int nv,int width,int taps,int layout,float eps,bool stable_softplus,
    int64_t ps0,int64_t ps1,int64_t as0,int64_t as1,int64_t bs0,int64_t bs1) {
    constexpr int Tile=16,Segments=RowThreads/32;
    const int tiles=(T+Tile-1)/Tile,heads=2*nk+nv;
    const int tile=int(blockIdx.x)%tiles,head=(int(blockIdx.x)/tiles)%heads,b=int(blockIdx.x)/(tiles*heads);
    const int begin=tile*Tile,C=heads*width,offset=head*width;
    const bool normalize=head<2*nk;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    extern __shared__ float image[];
    for(int i=threadIdx.x;i<(Tile+taps-1)*width;i+=256) {
        const int column=i%width,token=begin+i/width-(taps-1);float value=0;
        if(token<0)value=state[(int64_t(b)*(taps-1)+token+taps-1)*C+offset+column];
        else if(token<T) {
            value=float(projected[int64_t(b)*ps0+int64_t(token)*ps1+offset+column]);
            if constexpr(Scaled)value=prefill_ldexp(value,row_exponents[int64_t(b)*T+token]+column_exponents[offset+column]);
        }
        image[i]=value;
    }
    __syncthreads();
    // Eight independent query rows share the same immutable temporal image.
    // Each warp keeps every 32-channel segment of its row in registers.
    for(int first=0;first<Tile;first+=8) {
        const int local=first+warp,token=begin+local;
        float value[Segments];
        #pragma unroll
        for(int segment=0;segment<Segments;++segment)value[segment]=0;
        // Interleave channel accumulators and share the tap control/address
        // calculation. The compact loop avoids four expanded tap pipelines.
        #pragma unroll 1
        for(int tap=0;tap<taps;++tap) {
            #pragma unroll
            for(int segment=0;segment<Segments;++segment) {
                const int column=segment*32+lane;
                if(token<T && column<width) {
                    const float w=weight[layout?int64_t(tap)*C+offset+column:int64_t(offset+column)*taps+tap];
                    value[segment]=fmaf(image[(local+tap)*width+column],w,value[segment]);
                }
            }
        }
        #pragma unroll
        for(int segment=0;segment<Segments;++segment)
            if(token<T && segment*32+lane<width)value[segment]=silu_f32(value[segment]);
        float inverse=1;
        if(normalize) {
            float partial=0;
            #pragma unroll
            for(int segment=0;segment<Segments;++segment) {
                const float sum=warp_sum(value[segment]*value[segment]);
                // The original kernel publishes lane zero's partial from
                // each segment. XOR reduction results can vary by lane.
                const float original=__shfl_sync(0xffffffffu,sum,0);
                if(lane==segment)partial=original;
            }
            const float sum=warp_sum(partial);
            if(!lane)inverse=1.f/fmaxf(sqrtf(sum),eps);
            inverse=__shfl_sync(0xffffffffu,inverse,0);
        }
        #pragma unroll
        for(int segment=0;segment<Segments;++segment) {
            const int column=segment*32+lane;
            if(token<T && column<width) {
                if(normalize) {
                    const int h=head<nk?head:head-nk;
                    (head<nk?q:k)[((int64_t(b)*nk+h)*T+token)*width+column]=value[segment]*inverse;
                }else {
                    const int h=head-2*nk;
                    v[((int64_t(b)*nv+h)*T+token)*width+column]=value[segment];
                }
            }
        }
        if(!normalize && token<T && !lane) {
            const int h=head-2*nk;
            const float a=float(alpha[int64_t(b)*as0+int64_t(token)*as1+h])+dt_bias[h];
            const float sp=stable_softplus?fmaxf(a,0.f)+log1pf(expf(-fabsf(a))):(a>20.f?a:log1pf(expf(a)));
            decay[(int64_t(b)*nv+h)*T+token]=sp*-expf(a_log[h]);
            const float raw=float(beta[int64_t(b)*bs0+int64_t(token)*bs1+h]);
            bt[(int64_t(b)*nv+h)*T+token]=1.f/(1.f+expf(-raw));
        }
    }
    if(tile==tiles-1)for(int i=threadIdx.x;i<(taps-1)*width;i+=256) {
        const int tap=i/width,d=i%width;
        next[(int64_t(b)*(taps-1)+tap)*C+offset+d]=image[(T+tap-begin)*width+d];
    }
}



// Keep projected Q/K/V together for prefill and short verification windows.
// The V convolution also prepares recurrent gates, preserving FP32 operands.









namespace tyloi::cuda::ssm_conv_raw {
void prefill(const Args& a, cudaStream_t s) {
    const auto tiles=std::uint64_t(a.batch)*((std::uint64_t(a.tokens)+3)/4)*((std::uint64_t(a.channels)+SSM_CONV_BD-1)/SSM_CONV_BD);
    const auto blocks=unsigned(std::min<std::uint64_t>(65535,tiles));
    ssm_conv_silu_tiled_kernel<<<blocks,SSM_CONV_BD,0,s>>>(a.input,a.weight,a.bias,a.output,a.batch,a.tokens,a.channels,a.kernel,a.layout,a.has_bias);
}
void decode(const Args& a, cudaStream_t s) {
    dim3 blocks(a.batch,(a.channels+255)/256);
    ssm_conv_silu_decode_kernel<<<blocks,256,0,s>>>(a.state,a.input,a.weight,a.bias,a.output,a.batch,a.channels,a.kernel,a.layout,a.has_bias);
}
void qkv_decode_f32(const Args& a,cudaStream_t s) {
    ssm_conv_qkv_f32_decode_kernel<<<dim3(unsigned(2*a.nk+(a.nv*a.dv+255)/256),unsigned(a.batch)),256,0,s>>>(
        a.state,a.input,a.weight,a.bias,a.q,a.k,a.output,a.nk,a.nv,a.dk,a.dv,a.kernel,a.layout==0,a.eps);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());
}
void qkv_decode(const Args& a, cudaStream_t s) {
    auto qk=static_cast<const __half*>(a.qk); auto v=static_cast<const __half*>(a.v);
    ssm_conv_qk_norm_decode_kernel<256><<<a.batch*2*a.nk,256,0,s>>>(a.state,qk,v,a.weight,a.bias,a.q,a.k,a.batch,a.nk,a.nv,a.dk,a.dv,a.kernel,a.layout,a.has_bias,a.eps);
    dim3 blocks(a.batch,(a.nv*a.dv+255)/256);
    ssm_conv_v_decode_kernel<<<blocks,256,0,s>>>(a.state,qk,v,a.weight,a.bias,a.output,a.batch,a.nk,a.nv,a.dk,a.dv,a.kernel,a.layout,a.has_bias);
}
template<class Gate> void gate_decode(const Args& a, cudaStream_t s) {
    dim3 grid(unsigned(2*a.nk+(a.nv*a.dv+255)/256+(a.nv+255)/256),unsigned(a.batch));
    ssm_conv_qkv_gate_decode_kernel<Gate><<<grid,256,0,s>>>(a.state,static_cast<const __half*>(a.qk),static_cast<const __half*>(a.v),a.weight,
        static_cast<const Gate*>(a.alpha),static_cast<const Gate*>(a.beta),a.dt_bias,a.a_log,a.q,a.k,a.output,a.decay,a.beta_out,
        a.batch,a.nk,a.nv,a.dk,a.dv,a.kernel,a.layout,a.eps,a.a_stride0,a.a_stride1,a.b_stride0,a.b_stride1);
}
void qkv_gate_decode(const Args& a, cudaStream_t s) {
    if(a.half_gate) gate_decode<__half>(a,s); else gate_decode<float>(a,s);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());
}
template<class Input> void qkv_prefill_typed(const Args& a,cudaStream_t s) {
    auto qk=static_cast<const Input*>(a.qk); auto v=static_cast<const Input*>(a.v);
    const int qkc=2*a.nk*a.dk, vsz=a.nv*a.dv;
    linear_conv_qk_norm_prefill_kernel<256,false,Input><<<a.batch*a.tokens*2*a.nk,256,0,s>>>(a.state,qk,a.weight,a.bias,a.q,a.k,a.batch,a.tokens,a.nk,a.dk,a.channels,a.kernel,a.layout,a.has_bias,a.eps);
    linear_conv_v_prefill_kernel<false,float,Input><<<dim3(a.batch*a.tokens,(vsz+255)/256),256,0,s>>>(a.state,v,a.weight,a.bias,a.output,a.batch,a.tokens,a.nk,a.nv,a.dk,a.dv,a.channels,a.kernel,a.layout,a.has_bias);
    const int n=a.batch*(a.kernel-1)*a.channels;
    linear_conv_state_prefill_kernel<false,Input><<<(n+255)/256,256,0,s>>>(a.state,qk,v,a.next,a.batch,a.tokens,qkc,vsz,a.channels,a.kernel);
}
void qkv_prefill(const Args& a,cudaStream_t s) {
    if(a.input_float)qkv_prefill_typed<float>(a,s);else qkv_prefill_typed<__half>(a,s);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());
    if(a.input_float) {
        static std::atomic<bool> reported{false};
        if(const auto* trace=tyloi::cuda::env_lookup("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1' && !reported.exchange(true))
            std::fprintf(stderr,"native_gdn_prefill_conv_f32 actual_match=1 tokens=%d\n",a.tokens);
    }
}
void qkv_gate_short(const Args& a,cudaStream_t s) {
    const auto stream=s;
    const int B=a.batch,T=a.tokens,K=a.kernel,C=a.channels,qkC=2*a.nk*a.dk,vsz=a.nv*a.dv;
    const int nk=a.nk,nv=a.nv,width=a.dk,layout=a.layout;
    const float eps=a.eps;
    const bool scaled=a.row_exponents!=nullptr,stable_softplus=a.stable_softplus;
    const auto* re=a.row_exponents; const auto* ce=a.column_exponents;
    const int64_t tiled_blocks=int64_t(B)*(2*nk+nv)*((int64_t(T)+15)/16);
    if(T>=256 && tiled_blocks<=INT_MAX) {
        const auto launch_input=[&]<class Input>() {
            const auto launch_gate=[&]<class Gate>() {
                const auto launch=[&]<int Threads,bool Scaled>() {
                    packed_conv_prefill_tile<Threads,Input,Gate,Scaled><<<unsigned(tiled_blocks),256,
                        size_t(16+K-1)*width*sizeof(float),stream>>>(a.state,static_cast<const Input*>(a.qk),
                        a.weight,static_cast<const Gate*>(a.alpha),static_cast<const Gate*>(a.beta),
                        a.dt_bias,a.a_log,re,ce,a.q,a.k,a.output,
                        a.next,a.decay,a.beta_out,T,int(nk),int(nv),int(width),K,layout,float(eps),stable_softplus,
                        a.q_stride0,a.q_stride1,a.a_stride0,a.a_stride1,a.b_stride0,a.b_stride1);
                };
                const auto choose=[&]<int Threads>() {if(scaled)launch.template operator()<Threads,true>();else launch.template operator()<Threads,false>();};
                if(width<=32)choose.template operator()<32>();else if(width<=64)choose.template operator()<64>();
                else if(width<=128)choose.template operator()<128>();else choose.template operator()<256>();
            };
            if(a.half_gate)launch_gate.template operator()<__half>();else launch_gate.template operator()<float>();
        };
        if(!a.input_float)launch_input.template operator()<__half>();else launch_input.template operator()<float>();
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());
        if(const auto* trace=tyloi::cuda::env_lookup("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<bool> reported{};
            if(!reported.exchange(true))std::fprintf(stderr,"native_gdn_prefill_conv_tile actual_match=1 tokens=%d tile=16 scaled=%d shared_restore_once=1 warp_rows=8 inner_cta_barriers=0\n",T,int(scaled));
        }
        return;
    }
    const auto launch_input=[&]<class Input>() {
        const auto* projected=static_cast<const Input*>(a.qk);
        const auto launch_qk=[&]<int Threads>() {
            linear_conv_qk_norm_prefill_kernel<Threads,true,Input><<<B*T*2*int(nk),Threads,0,stream>>>(
                a.state,projected,a.weight,nullptr,a.q,a.k,
                B,T,int(nk),int(width),C,K,layout,0,float(eps),a.q_stride0,a.q_stride1,re,ce);
        };
        if(width<=32)launch_qk.template operator()<32>();
        else if(width<=64)launch_qk.template operator()<64>();
        else if(width<=128)launch_qk.template operator()<128>();
        else launch_qk.template operator()<256>();
        const auto launch_gate=[&]<class Gate>() {
            linear_conv_v_prefill_kernel<true,Gate,Input><<<dim3(B*T,(vsz+255)/256),256,0,stream>>>(
                a.state,projected+qkC,a.weight,nullptr,a.output,
                B,T,int(nk),int(nv),int(width),int(width),C,K,layout,0,
                static_cast<const Gate*>(a.alpha),static_cast<const Gate*>(a.beta),
                a.dt_bias,a.a_log,a.decay,a.beta_out,
                a.q_stride0,a.q_stride1,a.a_stride0,a.a_stride1,a.b_stride0,a.b_stride1,stable_softplus,re,ce);
        };
        if(a.half_gate)launch_gate.template operator()<__half>();else launch_gate.template operator()<float>();
        const int elements=B*(K-1)*C;
        linear_conv_state_prefill_kernel<true,Input><<<(elements+255)/256,256,0,stream>>>(
            a.state,projected,projected+qkC,a.next,B,T,qkC,vsz,C,K,a.q_stride0,a.q_stride1,re,ce);
    };
    if(!a.input_float)launch_input.template operator()<__half>();else launch_input.template operator()<float>();
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());
}
} // namespace tyloi::cuda::ssm_conv_raw
