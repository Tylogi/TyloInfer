#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace mfq_gr_detail {
template<class Input>
__device__ __forceinline__ float4 gr_dense_input4(const Input* input) {
    if constexpr(std::is_same_v<Input,float>)return *reinterpret_cast<const float4*>(input);
    else {
        const uint2 x=*reinterpret_cast<const uint2*>(input);
        return make_float4(__half2float(__ushort_as_half(x.x)),__half2float(__ushort_as_half(x.x>>16)),
            __half2float(__ushort_as_half(x.y)),__half2float(__ushort_as_half(x.y>>16)));
    }
}
__device__ __forceinline__ float4 gr_dense_weight4(const uint16_t* weight) {
    const uint2 w=*reinterpret_cast<const uint2*>(weight);
    return make_float4(__uint_as_float(w.x<<16),__uint_as_float(w.x&0xffff0000u),
        __uint_as_float(w.y<<16),__uint_as_float(w.y&0xffff0000u));
}
__device__ __forceinline__ float gr_dense_dot4(float sum,float4 w,float4 x) {
    sum=fmaf(w.x,x.x,sum);sum=fmaf(w.y,x.y,sum);
    sum=fmaf(w.z,x.z,sum);return fmaf(w.w,x.w,sum);
}
// One row uses the same FMA and reduction order in standalone and mixed grids.
// A mixed grid may supply extra threads; only the original reduction participates.
template<int Rows,class Input>
__device__ __forceinline__ void gr_bf16_activated_row(
        const Input* input,const uint16_t* matrix,float* output,int width,int outputs,
        int streams,int row,bool injecting) {
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    // Preserve the singleton's accumulation order. Multiple independent rows
    // use more warps to shorten each thread's serial projection loop.
    constexpr int threads=Rows==1?128:256,warps=threads/32;
    float sums[Rows]{};
    if(tid<threads && width%4==0 && reinterpret_cast<std::uintptr_t>(input)%(4*sizeof(Input))==0 &&
            reinterpret_cast<std::uintptr_t>(matrix)%8==0)for(int column=tid*4;column<width;column+=threads*4) {
        const float4 w=gr_dense_weight4(matrix+size_t(row)*width+column);
        #pragma unroll
        for(int m=0;m<Rows;++m)sums[m]=gr_dense_dot4(sums[m],w,
            gr_dense_input4(input+size_t(m)*width+column));
    }else if(tid<threads)for(int column=tid;column<width;column+=threads) {
        const float w=__uint_as_float(uint32_t(matrix[size_t(row)*width+column])<<16);
        #pragma unroll
        for(int m=0;m<Rows;++m)sums[m]=fmaf(w,float(input[size_t(m)*width+column]),sums[m]);
    }
    __shared__ float partial[Rows][warps];
    #pragma unroll
    for(int m=0;m<Rows;++m)if(tid<threads) {
        #pragma unroll
        for(int offset=16;offset>0;offset>>=1)sums[m]+=__shfl_down_sync(0xffffffffu,sums[m],offset);
        if(lane==0)partial[m][warp]=sums[m];
    }
    __syncthreads();
    if(tid<Rows) {
        float sum;
        if constexpr(warps==4)sum=(partial[tid][0]+partial[tid][2])+(partial[tid][1]+partial[tid][3]);
        else {
            const float s0=partial[tid][0]+partial[tid][4],s1=partial[tid][1]+partial[tid][5];
            const float s2=partial[tid][2]+partial[tid][6],s3=partial[tid][3]+partial[tid][7];
            sum=(s0+s2)+(s1+s3);
        }
        const float low=__fdiv_rn(sum,float(streams));
        const float gate=1.f/(1.f+expf(-low));
        if(injecting)output[size_t(tid)*outputs+row]=__fmul_rn(2.f,gate);
        else output[size_t(tid)*outputs+row]=__fmul_rn(low,gate);
    }
}

}
