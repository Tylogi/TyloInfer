#pragma once
#define TRELLIS_FIXED_BOOK 1
#define TRELLIS_NORMALIZED_WINDOW 1
#define TRELLIS_PAIR_BOOK 1
#include "tq_decode.cuh"
#include "tq_ops.h"
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <type_traits>
#include <cmath>
#include <algorithm>

template<int DT> using Scalar=std::conditional_t<DT==2,float,uint16_t>;

template<int DT> __device__ __forceinline__ float scalar_float(Scalar<DT> value) {
    if constexpr(DT==2)return value;
    else if constexpr(DT==1)return __bfloat162float(__ushort_as_bfloat16(value));
    else return __half2float(__ushort_as_half(value));
}

template<int DT> __device__ __forceinline__ Scalar<DT> scalar_round(float value) {
    if constexpr(DT==2)return value;
    else if constexpr(DT==1)return __bfloat16_as_ushort(__float2bfloat16_rn(value));
    else return __half_as_ushort(__float2half_rn(value));
}


template<int DT,int V,bool Columns=false> struct DotValues {
    const Scalar<DT>* x;
    int logical;
    const half* columns;
    uint32_t packed_x[V/2];
    float sums[2];
    __device__ __forceinline__ void operator()(int j,float a,float b) {
        if constexpr(Columns) {
            const uint32_t packed=__ldg(reinterpret_cast<const uint32_t*>(columns+j));
            a=__fmul_rn(a,__half2float(__ushort_as_half(uint16_t(packed))));
            b=__fmul_rn(b,__half2float(__ushort_as_half(uint16_t(packed>>16))));
        }
        if constexpr(DT==2) {
            if(j<logical)sums[j%2]=__fmaf_rn(a,__ldg(x+j),sums[j%2]);
            if(j+1<logical)sums[(j+1)%2]=__fmaf_rn(b,__ldg(x+j+1),sums[(j+1)%2]);
        } else {
            sums[j%2]=__fmaf_rn(a,scalar_float<DT>(uint16_t(packed_x[j/2])),sums[j%2]);
            sums[(j+1)%2]=__fmaf_rn(b,scalar_float<DT>(uint16_t(packed_x[j/2]>>16)),sums[(j+1)%2]);
        }
    }
};


// Retain the FP16 Gaussian book. Apply the shared row/group scale after a
// small FP32 dot, avoiding per-weight FP16 rounding. BF16/FP32 keep the
// reconstruction path below to preserve their existing rounding behavior.
template<int DT,int W,bool C,int Bits,int V>
__device__ __forceinline__ float factored_range(TrellisProjectionView m,uint64_t d,int row,
        const Scalar<DT>* x,int begin,int end) {
    const int tid=threadIdx.x%(W*32);
    float acc=0;
    for(int col=begin+tid*V;col<end;col+=W*32*V) {
        DotValues<DT,V,C> consumer{};
        consumer.x=x+col;consumer.logical=m.input_columns-col;
        if constexpr(DT!=2) {
            #pragma unroll
            for(int j=0;j<V/2;++j)consumer.packed_x[j]=0;
            if(col+V<=m.input_columns && (reinterpret_cast<uintptr_t>(x)&(V==4?7:15))==0) {
                if constexpr(V==4) {
                    const uint2 a=__ldg(reinterpret_cast<const uint2*>(x+col));
                    consumer.packed_x[0]=a.x;consumer.packed_x[1]=a.y;
                }
                #pragma unroll
                for(int j=0;j<V/8;++j) {
                    const uint4 a=__ldg(reinterpret_cast<const uint4*>(x+col+8*j));
                    consumer.packed_x[4*j]=a.x;consumer.packed_x[4*j+1]=a.y;
                    consumer.packed_x[4*j+2]=a.z;consumer.packed_x[4*j+3]=a.w;
                }
            } else {
                #pragma unroll
                for(int j=0;j<V;++j)if(col+j<m.input_columns)
                    consumer.packed_x[j/2]|=uint32_t(__ldg(x+col+j))<<((j&1)*16);
            }
        }

        if constexpr(C)consumer.columns=static_cast<const half*>(m.scales)+col;
        visit_values<V,true,C,Bits,DotValues<DT,V,C>,true>(m.words,d,m.scales,
            static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,consumer);
        const float partial=consumer.sums[0]+consumer.sums[1];
        float scale=1.f;
        if constexpr(!C) {
            const int group=row*((m.stored_columns+(1<<m.gslog)-1)>>m.gslog)+(col>>m.gslog);
            const uint8_t selector=__ldg(static_cast<const uint8_t*>(m.scales)+(group>>1));
            scale=__half2float(__ldg(static_cast<const half*>(m.relative)+((selector>>((group&1)*4))&15)));
        }
        acc=__fmaf_rn(scale,partial,acc);
    }
    return __fmul_rn(__half2float(__ushort_as_half(uint16_t(d>>40))),acc);
}

template<int DT,int W,bool C,int Bits,int V>
__device__ __forceinline__ float projection_range(TrellisProjectionView m,uint64_t d,int row,const Scalar<DT>* x,int begin,int end) {
    if constexpr(DT==0)return factored_range<DT,W,C,Bits,V>(m,d,row,x,begin,end);
    const int tid=threadIdx.x%(W*32);
    DotValues<DT,V> consumer{};
    for(int col=begin+tid*V;col<end;col+=W*32*V) {
        consumer.x=x+col;consumer.logical=m.input_columns-col;
        if constexpr(DT!=2) {
            #pragma unroll
            for(int j=0;j<V/2;++j)consumer.packed_x[j]=0;
            if(col+V<=m.input_columns && (reinterpret_cast<uintptr_t>(x)&(V==4?7:15))==0) {
                if constexpr(V==4) {
                    const uint2 a=__ldg(reinterpret_cast<const uint2*>(x+col));
                    consumer.packed_x[0]=a.x;consumer.packed_x[1]=a.y;
                }
                #pragma unroll
                for(int j=0;j<V/8;++j) {
                    const uint4 a=__ldg(reinterpret_cast<const uint4*>(x+col+8*j));
                    consumer.packed_x[4*j]=a.x;consumer.packed_x[4*j+1]=a.y;
                    consumer.packed_x[4*j+2]=a.z;consumer.packed_x[4*j+3]=a.w;
                }
            } else {
                #pragma unroll
                for(int j=0;j<V;++j)if(col+j<m.input_columns)
                    consumer.packed_x[j/2]|=uint32_t(__ldg(x+col+j))<<((j&1)*16);
            }
        }
        visit_values<V,true,C,Bits>(m.words,d,m.scales,static_cast<const half*>(m.relative),nullptr,nullptr,
                                    row,col,m.stored_columns,m.gslog,consumer);
    }
    return consumer.sums[0]+consumer.sums[1];
}



template<int DT,int W,bool C,bool Tail>
__device__ __forceinline__ float width_dispatch(TrellisProjectionView m,int row,const Scalar<DT>* x) {
    const uint64_t d=__ldg(m.descriptors+row);
    constexpr int TILE=W*32*16;
    const int tail=Tail?m.input_columns%TILE:0;
    const bool compact=Tail && tail>0 && tail<=W*32*4;
    const int prefix=compact?m.input_columns-tail:m.input_columns;
    float value=0;
    #define GB(K) case K-1:value=projection_range<DT,W,C,K,16>(m,d,row,x,0,prefix);break
    switch((d>>32)&7){GB(1);GB(2);GB(3);GB(4);GB(5);GB(6);GB(7);GB(8);}
    #undef GB
    if constexpr(Tail) {
        if(compact)value+=projection_range<DT,W,C,0,4>(m,d,row,x,prefix,m.input_columns);
    }
    return value;
}

template<int DT,int W,bool Tail>
__device__ __forceinline__ float matrix_dot(TrellisProjectionView m,int row,const Scalar<DT>* x) {
    if(m.columns)return width_dispatch<DT,W,true,Tail>(m,row,x);
    return width_dispatch<DT,W,false,Tail>(m,row,x);
}
