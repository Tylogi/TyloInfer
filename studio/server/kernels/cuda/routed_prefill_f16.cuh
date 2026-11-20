#pragma once
#include <mma.h>

namespace routed_prefill_f16 {
__global__ void gather(const int64_t* slots,const half* input,half* output,
        int64_t first,int64_t rows,int columns,int routes) {
    const auto i=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)*8;
    if(i>=uint64_t(rows)*columns)return;
    const auto row=i/columns;const auto col=i%columns;
    const auto* source=input+uint64_t(__ldg(slots+first+row)/routes)*columns+col;
    if(col+8<=columns && ((reinterpret_cast<uintptr_t>(source)|reinterpret_cast<uintptr_t>(output+i))&15)==0)
        *reinterpret_cast<uint4*>(output+i)=__ldg(reinterpret_cast<const uint4*>(source));
    else for(int j=0;j<8 && i+j<uint64_t(rows)*columns;++j) {
        const auto at=i+j;
        output[at]=__ldg(input+uint64_t(__ldg(slots+first+at/columns)/routes)*columns+at%columns);
    }
}

// Double-buffered temporary FP16 products retain FP32 accumulation.
// Async shared copies require SM80; earlier WMMA devices use vector loads.
template<int BM,int BN,int BK>
__device__ __forceinline__ void load_stage(half* x,const half* weights,const half* input,int64_t begin,int64_t count,
        int token,int row,int batch,int row_tile,int rows,int columns,int64_t k) {
    constexpr int Stride=BK+8;
    for(int i=threadIdx.x*8;i<(BM+BN)*BK;i+=blockDim.x*8) {
        const int r=i/BK,col=i%BK;const bool activation=r<BM;const int local=activation?r:r-BM;
        const bool valid=(activation?token+local<count:row+local<rows) && k+col<columns;
        const half* source=valid?(activation?input+(begin+token+local)*columns+k+col:
            weights+(uint64_t(batch)*row_tile+row+local)*columns+k+col):input;
        auto* target=x+(activation?local:BM+local)*Stride+col;
#if __CUDA_ARCH__ >= 800
        if((reinterpret_cast<uintptr_t>(source)&15)==0) {
            const int bytes=valid?min(16,int((columns-k-col)*2)):0;
            const auto address=unsigned(__cvta_generic_to_shared(target));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(address),"l"(source),"r"(bytes));
        }else
#endif
        {
            #pragma unroll
            for(int j=0;j<8;++j)target[j]=valid && k+col+j<columns?source[j]:__float2half_rn(0.f);
        }
    }
}
template<int BM,int BN,int BK>
__device__ __forceinline__ void load_stage_fast(half* x,const half* weights,const half* input,int64_t begin,int64_t count,
        int token,int row,int batch,int row_tile,int rows,int columns,int64_t k) {
    constexpr int Stride=BK+8;
    #pragma unroll
    for(int i=threadIdx.x*8;i<BM*BK;i+=128*8) {
        const int r=i/BK,col=i%BK;const bool valid=token+r<count && k+col<columns;
        const auto* source=input+(begin+token+r)*columns+k+col;
        auto* target=x+r*Stride+col;
#if __CUDA_ARCH__ >= 800
        const auto address=unsigned(__cvta_generic_to_shared(target));const int bytes=valid?16:0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"::"r"(address),"l"(source),"r"(bytes));
#else
        *reinterpret_cast<uint4*>(target)=valid?__ldg(reinterpret_cast<const uint4*>(source)):uint4{};
#endif
    }
    #pragma unroll
    for(int i=threadIdx.x*8;i<BN*BK;i+=128*8) {
        const int r=i/BK,col=i%BK;const bool valid=row+r<rows && k+col<columns;
        const auto* source=weights+(uint64_t(batch)*row_tile+row+r)*columns+k+col;
        auto* target=x+(BM+r)*Stride+col;
#if __CUDA_ARCH__ >= 800
        const auto address=unsigned(__cvta_generic_to_shared(target));const int bytes=valid?16:0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"::"r"(address),"l"(source),"r"(bytes));
#else
        *reinterpret_cast<uint4*>(target)=valid?__ldg(reinterpret_cast<const uint4*>(source)):uint4{};
#endif
    }
}
template<int BM,int BN,int BK,bool Fast=false>
__global__ __launch_bounds__(128)
void product(const int64_t* bounds,const half* weights,const half* input,float* output,
        int row_tile,int rows,int columns,int output_columns,int output_first) {
#if __CUDA_ARCH__ >= 700
    constexpr int Stride=BK+8,Stage=(BM+BN)*Stride,WN=4/(BM/16),Fragments=BN/(WN*16);
    __shared__ __align__(32) half storage[2*Stage];
    const int batch=blockIdx.z,token=blockIdx.y*BM,row=blockIdx.x*BN;
    const int64_t begin=__ldg(bounds+batch)-__ldg(bounds),count=__ldg(bounds+batch+1)-__ldg(bounds+batch);
    if(token>=count)return;
    const int warp=threadIdx.x/32,tm=warp/WN*16,rn=warp%WN*Fragments*16;
    namespace wmma=nvcuda::wmma;
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
    wmma::fragment<wmma::accumulator,16,16,16,float> c[Fragments];
    #pragma unroll
    for(int j=0;j<Fragments;++j)wmma::fill_fragment(c[j],0.f);
    if constexpr(Fast)load_stage_fast<BM,BN,BK>(storage,weights,input,begin,count,token,row,batch,row_tile,rows,columns,0);
    else load_stage<BM,BN,BK>(storage,weights,input,begin,count,token,row,batch,row_tile,rows,columns,0);
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;"::);
#endif
    __syncthreads();
    for(int64_t k=0;k<columns;k+=BK) {
        const int stage=int(k/BK)&1;auto* x=storage+stage*Stage;auto* w=x+BM*Stride;
        if(k+BK<columns) {
            if constexpr(Fast)load_stage_fast<BM,BN,BK>(storage+(stage^1)*Stage,weights,input,begin,count,token,row,batch,row_tile,rows,columns,k+BK);
            else load_stage<BM,BN,BK>(storage+(stage^1)*Stage,weights,input,begin,count,token,row,batch,row_tile,rows,columns,k+BK);
#if __CUDA_ARCH__ >= 800
            asm volatile("cp.async.commit_group;"::);
#endif
        }
        #pragma unroll
        for(int ks=0;ks<BK;ks+=16) {
            wmma::load_matrix_sync(a,x+tm*Stride+ks,Stride);
            #pragma unroll
            for(int j=0;j<Fragments;++j) {
                wmma::load_matrix_sync(b,w+(rn+j*16)*Stride+ks,Stride);wmma::mma_sync(c[j],a,b,c[j]);
            }
        }
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group 0;"::);
#endif
        __syncthreads();
    }
    auto* result=reinterpret_cast<float*>(storage);
    #pragma unroll
    for(int j=0;j<Fragments;++j)wmma::store_matrix_sync(result+tm*BN+rn+j*16,c[j],BN,wmma::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<BM*BN;i+=blockDim.x) {
        const int t=token+i/BN,n=row+i%BN;if(t<count && n<rows)output[(begin+t)*output_columns+output_first+n]=result[i];
    }
#endif
}

inline cudaError_t launch_product(const int64_t* bounds,const half* weights,const half* input,float* output,
        int batch,int capacity,int row_tile,int rows,int columns,int output_columns,int first,cudaStream_t stream) {
    const bool aligned=columns%8==0 && ((reinterpret_cast<uintptr_t>(weights)|reinterpret_cast<uintptr_t>(input))&15)==0;
    // Broad token batches amortize weight loads over 64 rows. Small batches
    // retain more independent CTAs with a 32-row tile. Both accept all tails.
    if(batch>=4 && capacity>=64) {
        const dim3 grid((rows+63)/64,(capacity+63)/64,batch);
        if(aligned)product<64,64,64,true><<<grid,128,0,stream>>>(bounds,weights,input,output,row_tile,rows,columns,output_columns,first);
        else product<64,64,64,false><<<grid,128,0,stream>>>(bounds,weights,input,output,row_tile,rows,columns,output_columns,first);
    }else {
        const dim3 grid((rows+31)/32,(capacity+31)/32,batch);
        if(aligned)product<32,32,128,true><<<grid,128,0,stream>>>(bounds,weights,input,output,row_tile,rows,columns,output_columns,first);
        else product<32,32,128,false><<<grid,128,0,stream>>>(bounds,weights,input,output,row_tile,rows,columns,output_columns,first);
    }
    return cudaGetLastError();
}
} // namespace routed_prefill_f16
