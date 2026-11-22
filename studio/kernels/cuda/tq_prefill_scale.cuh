#pragma once
#include "prefill_scale.cuh"

template<bool Columns>
__device__ __forceinline__ void tq_prefill_decode_values(TrellisProjectionView m,
        uint64_t descriptor,int row,int col,half (&values)[16]) {
    #define TQ_SCALE_BITS(B) case B-1:decode_vector<16,true,Columns,B>(m.words,descriptor,m.scales, \
        static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,values);break
    switch((descriptor>>32)&7) {
        TQ_SCALE_BITS(1);TQ_SCALE_BITS(2);TQ_SCALE_BITS(3);TQ_SCALE_BITS(4);
        TQ_SCALE_BITS(5);TQ_SCALE_BITS(6);TQ_SCALE_BITS(7);TQ_SCALE_BITS(8);
    }
    #undef TQ_SCALE_BITS
}

template<bool Columns>
__global__ void tq_prefill_row_exponents_kernel(TrellisProjectionView m,int first,int32_t* exponents) {
    const int row=first+blockIdx.x;
    const auto descriptor=__ldg(m.descriptors+row);
    float maximum=0;
    for(int64_t col=int64_t(threadIdx.x)*16;col<m.input_columns;col+=int64_t(blockDim.x)*16) {
        half values[16];tq_prefill_decode_values<Columns>(m,descriptor,row,int(col),values);
        #pragma unroll
        for(int j=0;j<16;++j)if(col+j<m.input_columns)maximum=fmaxf(maximum,fabsf(__half2float(values[j])));
    }
    for(int step=16;step;step>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,step));
    __shared__ float partial[4];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    if(!lane)partial[warp]=maximum;
    __syncthreads();
    if(!warp) {
        maximum=lane<4?partial[lane]:0;
        for(int step=16;step;step>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,step));
        if(!lane)exponents[blockIdx.x]=prefill_unit_exponent(maximum,m.input_columns);
    }
}

template<bool Columns>
__global__ void tq_prefill_scaled_rows_kernel(TrellisProjectionView m,int first,const int32_t* exponents,half* output) {
    const auto column=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)*16;
    if(column>=uint64_t(m.input_columns))return;
    const int col=int(column),row=first+blockIdx.y;
    const auto descriptor=__ldg(m.descriptors+row);
    half values[16];tq_prefill_decode_values<Columns>(m,descriptor,row,col,values);
    const int exponent=__ldg(exponents+blockIdx.y);
    #pragma unroll
    for(int j=0;j<16;++j)if(col+j<m.input_columns)
        output[std::size_t(blockIdx.y)*m.input_columns+col+j]=
            __float2half_rn(prefill_ldexp(__half2float(values[j]),-exponent));
}

static bool tq_prefill_valid_rows(TrellisProjectionView m,int first,int count) {
    return first>=0 && count>0 && first<=m.rows && count<=m.rows-first &&
        m.input_columns>0 && m.input_columns<=m.stored_columns && m.stored_columns%32==0 &&
        m.words && m.descriptors && m.scales && m.relative &&
        (m.columns || (m.gslog>=4 && m.gslog<=20));
}
extern "C" int mfq_tq_prefill_row_exponents(TrellisProjectionView m,int first,int count,
        int32_t* exponents,cudaStream_t stream) {
    if(!exponents || !tq_prefill_valid_rows(m,first,count))return -1;
    if(m.columns)tq_prefill_row_exponents_kernel<true><<<count,128,0,stream>>>(m,first,exponents);
    else tq_prefill_row_exponents_kernel<false><<<count,128,0,stream>>>(m,first,exponents);
    return int(cudaGetLastError());
}
extern "C" int mfq_tq_prefill_scaled_rows(TrellisProjectionView m,int first,int count,
        const int32_t* exponents,void* output,cudaStream_t stream) {
    if(!exponents || !output || count>65535 || !tq_prefill_valid_rows(m,first,count))return -1;
    const dim3 grid(unsigned((uint64_t(m.input_columns)+511)/512),count);
    if(m.columns)tq_prefill_scaled_rows_kernel<true><<<grid,32,0,stream>>>(m,first,exponents,static_cast<half*>(output));
    else tq_prefill_scaled_rows_kernel<false><<<grid,32,0,stream>>>(m,first,exponents,static_cast<half*>(output));
    return int(cudaGetLastError());
}
