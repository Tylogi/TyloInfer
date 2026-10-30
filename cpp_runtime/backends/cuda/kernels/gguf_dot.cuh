#pragma once
#include "ggml-cuda/vecdotq.cuh"
#include <type_traits>

namespace mfq::cuda::gguf {
namespace {
// Q8_1 activation dots use the bundled llama.cpp/ggml implementation (MIT).
// Warp/block assignment and the custom Q2_0 dot follow Strata iq_kernels.cu,
// Copyright (c) 2026 Niko1221 and contributors, MIT: ../../../third_party/strata.LICENSE.
template<int Format> struct Dot;
#define TYLOINFER_GGUF_DOT(Name, Function, Qk, Qi, Step) \
template<> struct Dot<GGML_TYPE_##Name> { \
    static constexpr int qk = Qk, parts = Qi / Step, step = Step; \
    __device__ static float apply(const void* w, const block_q8_1* x, int block, int part) { \
        return Function(w, x, block, part); \
    } \
};
TYLOINFER_GGUF_DOT(Q4_0, vec_dot_q4_0_q8_1, 32, QI4_0, VDR_Q4_0_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(Q4_1, vec_dot_q4_1_q8_1, 32, QI4_1, VDR_Q4_1_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(Q5_0, vec_dot_q5_0_q8_1, 32, QI5_0, VDR_Q5_0_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(Q5_1, vec_dot_q5_1_q8_1, 32, QI5_1, VDR_Q5_1_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(Q8_0, vec_dot_q8_0_q8_1, 32, QI8_0, VDR_Q8_0_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(Q2_K, vec_dot_q2_K_q8_1, 256, QI2_K, VDR_Q2_K_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(Q3_K, vec_dot_q3_K_q8_1, 256, QI3_K, VDR_Q3_K_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(Q4_K, vec_dot_q4_K_q8_1, 256, QI4_K, VDR_Q4_K_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(Q5_K, vec_dot_q5_K_q8_1, 256, QI5_K, VDR_Q5_K_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(Q6_K, vec_dot_q6_K_q8_1, 256, QI6_K, VDR_Q6_K_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(IQ2_XXS, vec_dot_iq2_xxs_q8_1, 256, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(IQ2_XS, vec_dot_iq2_xs_q8_1, 256, QI2_XS, VDR_IQ2_XS_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(IQ3_XXS, vec_dot_iq3_xxs_q8_1, 256, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(IQ4_NL, vec_dot_iq4_nl_q8_1, 32, QI4_NL, VDR_IQ4_NL_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(IQ3_S, vec_dot_iq3_s_q8_1, 256, QI3_S, VDR_IQ3_S_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(IQ2_S, vec_dot_iq2_s_q8_1, 256, QI2_S, VDR_IQ2_S_Q8_1_MMVQ)
TYLOINFER_GGUF_DOT(IQ4_XS, vec_dot_iq4_xs_q8_1, 256, QI4_XS, VDR_IQ4_XS_Q8_1_MMVQ)
#undef TYLOINFER_GGUF_DOT

template<> struct Dot<GGML_TYPE_Q2_0> {
    struct Block { half d; std::uint8_t qs[16]; };
    static_assert(sizeof(Block) == 18);
    static constexpr int qk = 64, parts = 2, step = 1;
    __device__ static float apply(const void* w, const block_q8_1* x, int block, int part) {
        const auto* weight = static_cast<const Block*>(w) + block;
        const auto* values = reinterpret_cast<const std::int16_t*>(weight->qs) + part * 4;
        const auto* activation = x + part;
        int sum = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int codes = values[j];
            const int even = __byte_perm(0x020100FF, 0x020100FF, codes);
            const int odd = __byte_perm(0x020100FF, 0x020100FF, codes >> 2);
            sum = ggml_cuda_dp4a(get_int_b4(activation->qs, j * 2), __byte_perm(even, odd, 0x5140), sum);
            sum = ggml_cuda_dp4a(get_int_b4(activation->qs, j * 2 + 1), __byte_perm(even, odd, 0x7362), sum);
        }
        return __half2float(weight->d) * __low2float(activation->ds) * sum;
    }
};
template<> struct Dot<GGML_TYPE_Q8_K> {
    static constexpr int qk = 256, parts = 8, step = 1;
    __device__ static float apply(const void* w, const block_q8_1* x, int block, int part) {
        const auto* weight = static_cast<const block_q8_K*>(w) + block;
        int sum = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            sum = ggml_cuda_dp4a(get_int_b4(weight->qs + part * 32, i), get_int_b4(x[part].qs, i), sum);
        return weight->d * __low2float(x[part].ds) * sum;
    }
};
#include "gguf_dot_prepared.cuh"

template<int Format>
__device__ float q8_dot(const std::uint8_t* weight, const block_q8_1* x, std::int64_t columns, int lane) {
    using F = Dot<Format>;
    float sum = 0;
    for (auto i = std::int64_t(lane); i < columns / F::qk * F::parts; i += 32) {
        const int block = int(i / F::parts), part = F::step * (i % F::parts);
        sum += F::apply(weight, x + block * (F::qk / 32), block, part);
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
    return sum;
}
template<int Format,int Rows>
__device__ void q8_dot_row_partials(const std::uint8_t* weight,const block_q8_1* x,std::int64_t columns,
        int lane,int rows,float (&sum)[Rows]) {
    using F=PreparedDot<Format>;
    const auto activation_stride=columns/32;
    // Decode a weight fragment once, then apply each activation row.
    for(auto i=std::int64_t(lane);i<columns/F::qk*F::parts;i+=32) {
        const int block=int(i/F::parts),part=F::step*(i%F::parts);
        const auto prepared=F::load(weight,block,part);
        #pragma unroll
        for(int row=0;row<Rows;++row)if(row<rows)
            sum[row]+=F::apply(prepared,x+row*activation_stride+block*(F::qk/32));
    }
    // The caller reduces all outputs together with warp_sums.
}
template<int Format,int Rows>
__device__ __forceinline__ void q8_dot_indexed_rows(const std::uint8_t* weight,const block_q8_1* x,std::int64_t columns,
        int lane,int rows,const std::int32_t* slots,std::int64_t routes,bool routed_input,float (&sum)[Rows]) {
    using F=PreparedDot<Format>;
    const auto stride=columns/32;
    const block_q8_1* activations[Rows];
    #pragma unroll
    for(int row=0;row<Rows;++row)if(row<rows)
        activations[row]=x+std::int64_t(routed_input?slots[row]:slots[row]/int(routes))*stride;
    for(auto i=std::int64_t(lane);i<columns/F::qk*F::parts;i+=32) {
        const int block=int(i/F::parts),part=F::step*(i%F::parts);
        const auto prepared=F::load(weight,block,part);
        #pragma unroll
        for(int row=0;row<Rows;++row)if(row<rows) {
            sum[row]+=F::apply(prepared,activations[row]+block*(F::qk/32));
        }
    }
    #pragma unroll
    for(int row=0;row<Rows;++row)if(row<rows)
        for(int offset=16;offset;offset>>=1)sum[row]+=__shfl_down_sync(0xffffffffu,sum[row],offset);
}
template<int Format,int Lanes>
__device__ __forceinline__ float q8_dot_subwarp(const std::uint8_t* weight,const block_q8_1* input,
        std::int64_t columns,int lane,bool active) {
    using F=PreparedDot<Format>;
    using Weight=typename F::Weight;
    static_assert(sizeof(Weight)%sizeof(std::uint32_t)==0);
    const auto items=columns/F::qk*F::parts;
    const int local_lane=lane%Lanes;
    float sum=0;
    for(std::int64_t begin=0;begin<items;begin+=Lanes) {
        const auto i=begin+local_lane;
        const int block=int(i/F::parts),part=F::step*(i%F::parts);
        Weight prepared{};
        // Only the first token's lanes decode the weight fragment. Broadcast
        // it to the other tokens without keeping an accumulator per token.
        if(lane<Lanes && i<items)prepared=F::load(weight,block,part);
        auto* words=reinterpret_cast<std::uint32_t*>(&prepared);
        #pragma unroll
        for(int word=0;word<int(sizeof(Weight)/sizeof(std::uint32_t));++word)
            words[word]=__shfl_sync(0xffffffffu,words[word],local_lane);
        if(i<items && active)sum+=F::apply(prepared,input+block*(F::qk/32));
    }
    for(int offset=Lanes/2;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,offset,Lanes);
    return sum;
}
}
}
