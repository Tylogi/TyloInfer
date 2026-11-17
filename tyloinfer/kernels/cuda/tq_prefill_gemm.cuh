#pragma once
#include "routed_prefill_f16.cuh"
// Long per-expert groups decode bounded FP16 weight tiles and reuse them in
// batched GEMM. FP32 intermediates preserve bias-before-half-rounding semantics.
namespace tq_prefill_batch_gemm {
template<int Bits,bool Columns>
__device__ __forceinline__ void pp_decode_striped_row(TrellisProjectionView m,uint64_t d,
        int row,int lane,half* target) {
    for(int col=lane*16;col<m.input_columns;col+=1024)
        tq_decode_tile<Bits,Columns>(m,d,row,col,target+col);
}
template<bool Columns>
__device__ __forceinline__ void pp_decode_striped_dispatch(TrellisProjectionView m,uint64_t d,
        int row,int lane,half* target) {
    #define STRIP(K) case K-1:pp_decode_striped_row<K,Columns>(m,d,row,lane,target);return
    switch((d>>32)&7){STRIP(1);STRIP(2);STRIP(3);STRIP(4);STRIP(5);STRIP(6);STRIP(7);STRIP(8);}
    #undef STRIP
}
template<int Role>
__global__ void pp_batch_decode(const TrellisPrefillMoeJob* jobs,int first,int row_tile,int count,int K,half* output) {
    const int relative=blockIdx.y*2+threadIdx.x/64;if(relative>=count)return;
    const auto job=jobs[blockIdx.z];const int logical=first+relative;
    const auto m=Role==0?(logical<job.gate.rows?job.gate:job.up):Role==1?job.up:job.down;
    const int row=Role==0?(logical<job.gate.rows?logical:logical-job.gate.rows):logical;
    const uint64_t d=__ldg(m.descriptors+row);
    auto* target=output+(uint64_t(blockIdx.z)*row_tile+relative)*K;
    if(m.columns)pp_decode_striped_dispatch<true>(m,d,row,threadIdx.x&63,target);
    else pp_decode_striped_dispatch<false>(m,d,row,threadIdx.x&63,target);
}
__global__ void pp_batch_activate(const TrellisPrefillMoeJob* jobs,const float* gate,
        half* output,int B,int T,int H,float limit) {
    const uint64_t at=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at>=uint64_t(B)*T*H)return;
    const int row=at%H,b=at/(uint64_t(T)*H);
    const auto job=jobs[b];
    const uint64_t pair=(at/H)*(2ull*H)+row;
    float g=gate[pair]+(job.gate.bias?__ldg(job.gate.bias+row):0);
    float u=gate[pair+H]+(job.up.bias?__ldg(job.up.bias+row):0);
    g=__half2float(__float2half_rn(g));u=__half2float(__float2half_rn(u));
    if(limit>0){g=fminf(g,limit);u=fminf(fmaxf(u,-limit),limit);}
    output[at]=__float2half_rn(u*(g/(1.f+expf(-g))));
}
__global__ void pp_batch_scatter(const TrellisPrefillMoeJob* jobs,const int64_t* slots,const float* input,
        half* pairs,int B,int T,int K) {
    const uint64_t at=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at>=uint64_t(B)*T*K)return;
    const int row=at%K,t=(at/K)%T,b=at/(uint64_t(T)*K);
    const auto job=jobs[b];if(t>=job.count)return;
    const float value=input[at]+(job.down.bias?__ldg(job.down.bias+row):0);
    pairs[uint64_t(__ldg(slots+job.first+t))*K+row]=__float2half_rn(value);
}
template<int Role>
int pp_batch_project(const TrellisPrefillMoeJob* jobs,const half* x,float* y,int B,int T,int K,int N,
        void* workspace,uint64_t bytes,cublasHandle_t handle,cudaStream_t stream) {
    int tile=int(std::min(uint64_t(std::min(N,65535)),bytes/(uint64_t(B)*K*2)));
    if(tile>=16)tile=tile/16*16;
    if(tile<1)return -1;
    const float alpha=1,beta=0;
    for(int64_t first=0;first<N;first+=tile) {
        const int count=std::min(tile,int(N-first));
        pp_batch_decode<Role><<<dim3(1,(count+1)/2,B),128,0,stream>>>(jobs,first,tile,count,K,static_cast<half*>(workspace));
        const auto error=cudaGetLastError();if(error)return int(error);
        const auto st=cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,count,T,K,&alpha,
            workspace,CUDA_R_16F,K,int64_t(tile)*K,x,CUDA_R_16F,K,int64_t(T)*K,&beta,
            y+first,CUDA_R_32F,N,int64_t(T)*N,B,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if(st)return 1000+st;
    }
    return 0;
}

} // namespace tq_prefill_batch_gemm

extern "C" int mfq_tq_prefill_moe_gemm_staged(const TrellisPrefillMoeJob* jobs,const int64_t* slots,
        const void* input,void* pairs,int batch,int capacity,int input_width,int hidden_width,
        int routes,float limit,void* workspace,uint64_t workspace_bytes,void* activations,
        uint64_t activation_bytes,void* blas,cudaEvent_t down_ready,cudaStream_t stream) {
    if(!jobs || !slots || !input || !pairs || !workspace || !activations || !blas ||
       batch<1 || batch>65535 || capacity<1 || input_width<1 || input_width>INT_MAX-255 ||
       hidden_width<1 || hidden_width>INT_MAX/2 || routes<1 || !std::isfinite(limit) || limit<0 ||
       (reinterpret_cast<uintptr_t>(workspace)&15) || (reinterpret_cast<uintptr_t>(activations)&15) ||
       workspace==input || workspace==pairs || workspace==activations || activations==input || activations==pairs)return -1;
    const uint64_t rows=uint64_t(batch)*capacity;
    const uint64_t width=uint64_t(input_width>hidden_width?input_width:hidden_width);
    const uint64_t output_width=uint64_t(input_width>2*hidden_width?input_width:2*hidden_width);
    if(rows>(UINT64_MAX-15)/(2*width+4*output_width))return -1;
    const uint64_t input_bytes=(rows*width*2+15)&~uint64_t(15);
    if(input_bytes+rows*output_width*4>activation_bytes || workspace_bytes<uint64_t(batch)*width*2 ||
       (rows*((uint64_t(input_width)+7)/8)+127)/128>INT_MAX ||
       (rows*hidden_width+255)/256>INT_MAX || (rows*input_width+255)/256>INT_MAX)return -1;
    auto handle=static_cast<cublasHandle_t>(blas);
    cudaStream_t previous=nullptr;auto st=cublasGetStream(handle,&previous);if(st)return 1000+st;
    if(previous!=stream){st=cublasSetStream(handle,stream);if(st)return 1000+st;}
    cublasPointerMode_t mode;st=cublasGetPointerMode(handle,&mode);if(st)return 1000+st;
    if(mode!=CUBLAS_POINTER_MODE_HOST)return -1;
    auto* x=static_cast<half*>(activations);
    auto* values=reinterpret_cast<float*>(static_cast<uint8_t*>(activations)+input_bytes);
    const auto vectors=rows*((uint64_t(input_width)+7)/8);
    tq_prefill::pp_batch_gather<<<(vectors+127)/128,128,0,stream>>>(jobs,slots,
        static_cast<const half*>(input),x,batch,capacity,input_width,routes);
    auto error=cudaGetLastError();if(error)return int(error);
    int status=tq_prefill_batch_gemm::pp_batch_project<0>(jobs,x,values,batch,capacity,input_width,2*hidden_width,
        workspace,workspace_bytes,handle,stream);if(status)return status;
    tq_prefill_batch_gemm::pp_batch_activate<<<(rows*hidden_width+255)/256,256,0,stream>>>(
        jobs,values,x,batch,capacity,hidden_width,limit);
    error=cudaGetLastError();if(error)return int(error);
    if(down_ready){error=cudaStreamWaitEvent(stream,down_ready,0);if(error)return int(error);}
    status=tq_prefill_batch_gemm::pp_batch_project<2>(jobs,x,values,batch,capacity,hidden_width,input_width,
        workspace,workspace_bytes,handle,stream);if(status)return status;
    tq_prefill_batch_gemm::pp_batch_scatter<<<(rows*input_width+255)/256,256,0,stream>>>(
        jobs,slots,values,static_cast<half*>(pairs),batch,capacity,input_width);
    return int(cudaGetLastError());
}

extern "C" int mfq_tq_prefill_moe_gemm(const TrellisPrefillMoeJob* jobs,const int64_t* slots,
        const void* input,void* pairs,int batch,int capacity,int input_width,int hidden_width,
        int routes,float limit,void* workspace,uint64_t workspace_bytes,void* activations,
        uint64_t activation_bytes,void* blas,cudaStream_t stream) {
    return mfq_tq_prefill_moe_gemm_staged(jobs,slots,input,pairs,batch,capacity,input_width,hidden_width,
        routes,limit,workspace,workspace_bytes,activations,activation_bytes,blas,nullptr,stream);
}

namespace tq_prefill_compact {
__global__ void activate(const TrellisPrefillMoeJob* jobs,const int64_t* bounds,
        const float* values,half* output,int capacity,int width,float limit) {
    const auto i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto job=jobs[blockIdx.y];const auto token=i/width;const int row=i%width;
    if(token>=job.count)return;
    const auto at=(uint64_t(__ldg(bounds+blockIdx.y)-__ldg(bounds))+token)*width+row;
    const auto pair=(at/width)*(2ull*width)+row;
    float g=values[pair]+(job.gate.bias?__ldg(job.gate.bias+row):0.f);
    float u=values[pair+width]+(job.up.bias?__ldg(job.up.bias+row):0.f);
    g=__half2float(__float2half_rn(g));u=__half2float(__float2half_rn(u));
    if(limit>0){g=fminf(g,limit);u=fminf(fmaxf(u,-limit),limit);}
    output[at]=__float2half_rn(u*(g/(1.f+expf(-g))));
}
__global__ void scatter(const TrellisPrefillMoeJob* jobs,const int64_t* slots,const int64_t* bounds,
        const float* values,half* output,int width) {
    const auto i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto job=jobs[blockIdx.y];const auto token=i/width;const int row=i%width;
    if(token>=job.count)return;
    const auto at=(uint64_t(__ldg(bounds+blockIdx.y)-__ldg(bounds))+token)*width+row;
    const float value=values[at]+(job.down.bias?__ldg(job.down.bias+row):0.f);
    output[uint64_t(__ldg(slots+job.first+token))*width+row]=__float2half_rn(value);
}
template<int Role>
int project(const TrellisPrefillMoeJob* jobs,const int64_t* bounds,const half* input,float* output,
        int batch,int capacity,int columns,int rows,void* workspace,uint64_t bytes,cudaStream_t stream) {
    int tile=int(std::min(uint64_t(std::min(rows,65535)),bytes/(uint64_t(batch)*columns*2)));
    if(tile>=16)tile=tile/16*16;
    if(tile<1)return -1;
    for(int64_t first=0;first<rows;first+=tile) {
        const int count=std::min(tile,int(rows-first));
        tq_prefill_batch_gemm::pp_batch_decode<Role><<<dim3(1,(count+1)/2,batch),128,0,stream>>>(
            jobs,int(first),tile,count,columns,static_cast<half*>(workspace));
        auto error=cudaGetLastError();if(error)return int(error);
        error=routed_prefill_f16::launch_product(bounds,static_cast<const half*>(workspace),input,output,
            batch,capacity,tile,count,columns,rows,int(first),stream);
        if(error)return int(error);
    }
    return 0;
}
} // namespace tq_prefill_compact

extern "C" int mfq_tq_prefill_moe_compact(const TrellisPrefillMoeJob* jobs,const int64_t* slots,
        const int64_t* bounds,const void* input,void* pairs,int batch,int capacity,int64_t first_slot,
        int64_t rows,int input_width,int hidden_width,int output_width,int routes,float limit,
        void* workspace,uint64_t workspace_bytes,void* activations,uint64_t activation_bytes,
        cudaEvent_t down_ready,cudaStream_t stream) {
    if(!jobs || !slots || !bounds || !input || !pairs || !workspace || !activations ||
            batch<1 || batch>65535 || capacity<1 || capacity>65535*32 || first_slot<0 || rows<batch ||
            uint64_t(rows)>uint64_t(batch)*capacity || input_width<1 || input_width>INT_MAX-255 ||
            hidden_width<1 || hidden_width>INT_MAX/2 ||
            output_width<1 || routes<1 || !std::isfinite(limit) || limit<0 ||
            (reinterpret_cast<uintptr_t>(workspace)&15) || (reinterpret_cast<uintptr_t>(activations)&15) ||
            workspace==input || workspace==pairs || workspace==activations || activations==input || activations==pairs)return -1;
    const auto width=uint64_t(std::max(input_width,hidden_width));
    const auto outputs=uint64_t(std::max(output_width,2*hidden_width));
    if(uint64_t(rows)>(UINT64_MAX-15)/(width*2+outputs*4))return -1;
    const auto input_bytes=(uint64_t(rows)*width*2+15)&~uint64_t(15);
    if(input_bytes+uint64_t(rows)*outputs*4>activation_bytes || workspace_bytes<uint64_t(batch)*width*2 ||
            (uint64_t(rows)*input_width+1023)/1024>INT_MAX ||
            (uint64_t(capacity)*hidden_width+255)/256>INT_MAX ||
            (uint64_t(capacity)*output_width+255)/256>INT_MAX)return -1;
    auto* x=static_cast<half*>(activations);
    auto* values=reinterpret_cast<float*>(static_cast<uint8_t*>(activations)+input_bytes);
    routed_prefill_f16::gather<<<(uint64_t(rows)*input_width+1023)/1024,128,0,stream>>>(
        slots,static_cast<const half*>(input),x,first_slot,rows,input_width,routes);
    auto error=cudaGetLastError();if(error)return int(error);
    int status=tq_prefill_compact::project<0>(jobs,bounds,x,values,batch,capacity,input_width,2*hidden_width,
        workspace,workspace_bytes,stream);if(status)return status;
    tq_prefill_compact::activate<<<dim3((uint64_t(capacity)*hidden_width+255)/256,batch),256,0,stream>>>(
        jobs,bounds,values,x,capacity,hidden_width,limit);
    error=cudaGetLastError();if(error)return int(error);
    if(down_ready){error=cudaStreamWaitEvent(stream,down_ready,0);if(error)return int(error);}
    status=tq_prefill_compact::project<2>(jobs,bounds,x,values,batch,capacity,hidden_width,output_width,
        workspace,workspace_bytes,stream);if(status)return status;
    tq_prefill_compact::scatter<<<dim3((uint64_t(capacity)*output_width+255)/256,batch),256,0,stream>>>(
        jobs,slots,bounds,values,static_cast<half*>(pairs),output_width);
    return int(cudaGetLastError());
}
