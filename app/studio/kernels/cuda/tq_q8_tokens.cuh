#include "gr_bf16_projection.cuh"

#pragma once
#include "tq_compact.cuh"
#include <climits>
#include <atomic>
#include <cstdio>
#include <cstdlib>

// The fixed book before its final FP16 rounding is affine in the byte sum
// of state*0x83dcd12d. Center at 510: integers fit signed 10 bits. This avoids
// converting weights to FP16 and back to integers for each token tile.
// Q8 activation rounding dominates the omitted final book rounding.
inline constexpr float TQ_Q8_BOOK_INVERSE=0.00676727294921875f;
inline constexpr float TQ_Q8_BOOK_CENTER=-0.0018157958984375f;

template<int Bits,class Consumer>
__device__ __forceinline__ void q8_integer_book(TrellisProjectionView m,uint64_t d,int col,Consumer& consumer) {
    constexpr int V=16,Step=V*Bits,Align=Step&-Step;
    constexpr int MaxShift=Align<32?32-Align:0;
    constexpr int Words=(MaxShift+(V-1)*Bits+16+31)/32;
    constexpr int Normalized=((V-1)*Bits+16+31)/32;
    const int minimum=(15-(col/16)%16)*Step,first=minimum/32;

    uint32_t chunks[Words],normalized[Normalized];
    #pragma unroll
    for(int j=0;j<Words;++j) {
        int at=first+j;if(at>=8*Bits)at-=8*Bits;
        chunks[j]=tq_ring_word(m.words,d,m.stored_columns,col,Bits,at);
    }
    #pragma unroll
    for(int j=0;j<Normalized;++j)normalized[j]=__funnelshift_r(chunks[j],j+1<Words?chunks[j+1]:0,minimum&31);
    #pragma unroll
    for(int j=0;j<V;j+=2) {
        int centered[2];
        #pragma unroll
        for(int pair=0;pair<2;++pair) {
            constexpr uint32_t Multiplier=0x83dcd12du;
            const int pos=(15-j-pair)*Bits,index=pos/32;
            const uint32_t state=__funnelshift_r(normalized[index],index+1<Normalized?normalized[index+1]:0,pos&31)&65535;
            centered[pair]=int(__dp4a(state*Multiplier,0x01010101u,0u))-510;
        }
        consumer(j,centered[0],centered[1]);
    }
}
template<int T>struct Q8DotConsumer {
    uint32_t input[T][4];
    int dot[T]{};
    __device__ __forceinline__ void operator()(int j,int qa,int qb) {
        const uint32_t pair=uint32_t(uint16_t(qa))|(uint32_t(uint16_t(qb))<<16);
        #pragma unroll
        for(int t=0;t<T;++t) {
            if(!(j&2))asm("dp2a.lo.s32.s32 %0, %1, %2, %0;" : "+r"(dot[t]) : "r"(pair),"r"(input[t][j/4]));
            else asm("dp2a.hi.s32.s32 %0, %1, %2, %0;" : "+r"(dot[t]) : "r"(pair),"r"(input[t][j/4]));
        }
    }
};
template<int T,int Bits>
__device__ __forceinline__ void q8_tile_terms(TrellisProjectionView m,uint64_t d,int row,const uint32_t* input,
        int col,float (&terms)[T],float& scale) {
    const int tile=col/512,group=(col%512)/16;
    constexpr int tile_groups=32;
    const auto* block=input+size_t(tile)*32*T*6;
    Q8DotConsumer<T> consumer;
    #pragma unroll
    for(int t=0;t<T;++t) {
        const auto v=__ldg(reinterpret_cast<const uint4*>(block+(size_t(t)*tile_groups+group)*4));
        consumer.input[t][0]=v.x;consumer.input[t][1]=v.y;consumer.input[t][2]=v.z;consumer.input[t][3]=v.w;
    }
    q8_integer_book<Bits>(m,d,col,consumer);
    scale=__half2float(__ushort_as_half(uint16_t(d>>40)));
    if(!m.columns) {
        const size_t ng=(size_t(m.stored_columns)+(1<<m.gslog)-1)>>m.gslog;
        const size_t index=size_t(row)*ng+(col>>m.gslog);
        const uint8_t code=__ldg(static_cast<const uint8_t*>(m.scales)+(index>>1));
        scale*=__half2float(__ldg(static_cast<const half*>(m.relative)+((code>>((index&1)*4))&15)));
    }
    #pragma unroll
    for(int t=0;t<T;++t) {
        const auto coefficients=__ldg(reinterpret_cast<const float2*>(block+4*T*tile_groups)+size_t(t)*tile_groups+group);
        terms[t]=__fmaf_rn(coefficients.x,float(consumer.dot[t]),coefficients.y);
    }
}

template<int T,int Bits>
__device__ __forceinline__ void q8_row(TrellisProjectionView m,int row,const uint32_t* input,
        int qcolumns,float (&result)[T]) {
    const auto d=__ldg(m.descriptors+row);const int lane=threadIdx.x&31;
    // One 512-column tile stores coalesced Q8 planes followed by paired
    // gain/offset planes. Constant tile strides avoid retaining three
    // independent input pointers and their token strides in registers.
    for(int col=lane*16;col<m.input_columns;col+=512) {
        float terms[T],scale;q8_tile_terms<T,Bits>(m,d,row,input,col,terms,scale);
        #pragma unroll
        for(int t=0;t<T;++t)result[t]=__fmaf_rn(scale,terms[t],result[t]);
    }
}
template<int T>
__device__ __forceinline__ void q8_dispatch(TrellisProjectionView m,int row,const uint32_t* input,
        int qcolumns,float (&result)[T]) {
    const auto d=__ldg(m.descriptors+row);
    #define CASE(B) case B-1:q8_row<T,B>(m,row,input,qcolumns,result);return
    switch((d>>32)&7){CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);}
    #undef CASE
}

template<int DT,int T,int Activation=0>
__device__ __forceinline__ void q8_contraction_row(TrellisProjectionView m,const uint32_t* input,Scalar<DT>* output,
        int qcolumns,int groups,int streams,int row) {
    constexpr int warps=8,batch=32;
    const int warp=threadIdx.x/32,lane=threadIdx.x&31;
    const int tiles=(m.input_columns+511)/512;
    const auto d=__ldg(m.descriptors+row);
    input+=size_t(row/(m.rows/groups))*T*(qcolumns/16)*6;
    __shared__ float scales[batch][32],partial[T][batch][32];
    float result[T]{};
    for(int base=0;base<tiles;base+=batch) {
        const int count=min(batch,tiles-base);
        for(int tile=warp;tile<count;tile+=warps) {
            const int col=(base+tile)*512+lane*16;
            if(col<m.input_columns) {
                float values[T],scale;
                #define CASE(B) case B-1:q8_tile_terms<T,B>(m,d,row,input,col,values,scale);break
                switch((d>>32)&7){CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);}
                #undef CASE
                scales[tile][lane]=scale;
                #pragma unroll
                for(int t=0;t<T;++t)partial[t][tile][lane]=values[t];
            }
        }
        __syncthreads();
        if(warp==0)for(int tile=0;tile<count;++tile)if((base+tile)*512+lane*16<m.input_columns) {
            const float scale=scales[tile][lane];
            #pragma unroll
            for(int t=0;t<T;++t)result[t]=__fmaf_rn(scale,partial[t][tile][lane],result[t]);
        }
        __syncthreads();
    }
    if(warp==0) {
        #pragma unroll
        for(int t=0;t<T;++t) {
            #pragma unroll
            for(int step=16;step;step/=2)result[t]+=__shfl_down_sync(0xffffffff,result[t],step);
            if(lane==0) {
                if(m.bias)result[t]+=__ldg(m.bias+row);
                if constexpr(Activation) {
                    const float low=__fdiv_rn(result[t],float(streams)),gate=1.f/(1.f+expf(-low));
                    result[t]=__fmul_rn(Activation==2?2.f:low,gate);
                }
                output[size_t(t)*m.rows+row]=scalar_round<DT>(result[t]);
            }
        }
    }
}

template<int DT,int T,int Activation=0>
__global__ __launch_bounds__(256)
void q8_contraction_projection(TrellisProjectionView m,const uint32_t* input,Scalar<DT>* output,
        int qcolumns,int groups,int streams=1) {
    q8_contraction_row<DT,T,Activation>(m,input,output,qcolumns,groups,streams,int(blockIdx.x));
}

template<int T,bool Contraction>
__global__ __launch_bounds__(256)
void q8_residual_down_injection(TrellisProjectionView m,const uint32_t* quant,const float* input,
        const uint16_t* injection,float* output,float* gates,int qc,int streams,int blocks) {
    if(int(blockIdx.x)>=blocks) {
        mfq_gr_detail::gr_bf16_activated_row<T>(input,injection,gates,m.input_columns,
            streams,streams,int(blockIdx.x)-blocks,true);
        return;
    }
    if constexpr(Contraction)q8_contraction_row<2,T,1>(m,quant,output,qc,1,streams,int(blockIdx.x));
    else {
        const int row=int(blockIdx.x)*8+threadIdx.x/32;
        if(row>=m.rows)return;
        float value[T]{};q8_dispatch<T>(m,row,quant,qc,value);
        #pragma unroll
        for(int t=0;t<T;++t) {
            #pragma unroll
            for(int step=16;step;step/=2)value[t]+=__shfl_down_sync(0xffffffff,value[t],step);
            if(!(threadIdx.x&31)) {
                if(m.bias)value[t]+=__ldg(m.bias+row);
                const float low=__fdiv_rn(value[t],float(streams)),gate=1.f/(1.f+expf(-low));
                output[size_t(t)*m.rows+row]=__fmul_rn(low,gate);
            }
        }
    }
}

static int tq_q8_contraction(TrellisProjectionView m,bool& enabled) {
    enabled=false;
    // Eight independent K tiles amortize the CTA-wide handoff. Short K
    // retains the ordinary four-row CTA without a device query.
    if(m.input_columns<8*512)return 0;
    struct Device {int ordinal=-1,sms=0;};
    static thread_local Device last;
    int device=0;auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return int(error);
    if(last.ordinal!=device) {
        int sms=0;error=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);
        if(error!=cudaSuccess)return int(error);
        last={device,sms};
    }
    // The ordinary grid has four rows per CTA. Expand its parallel work only
    // while it supplies fewer than four CTAs per SM, on any supported GPU.
    enabled=(uint64_t(m.rows)+3)/4<uint64_t(last.sms)*4;
    return 0;
}
static void tq_trace_q8_contraction(TrellisProjectionView m,int tokens) {
    const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");
    if(!trace || trace[0]!='1')return;
    static std::atomic<unsigned> seen{0};const auto bit=1u<<tokens;
    if(!(seen.fetch_or(bit,std::memory_order_relaxed)&bit))
        std::fprintf(stderr,"native_tq_q8_contraction actual_match=1 tokens=%d K=%d N=%d warps=8 ordered_accumulation=1\n",
            tokens,m.input_columns,m.rows);
}
template<int DT,bool Parallel>
__global__ void q8_prepare(TrellisProjectionView plain,const TrellisProjectionJob* jobs,
        const Scalar<DT>* x,uint32_t* output,int tokens,int stride,int qcolumns,int groups) {
    auto m=plain;
    if constexpr(Parallel) {
        if(blockIdx.y<unsigned(groups)) {
            m=jobs[blockIdx.y].weight;
            if(!m.columns)return;
        }
    }
    const int lane=threadIdx.x&31;
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const uint64_t group=index/16;const int element=int(index%16);
    const int token=int(group/(qcolumns/16)),col=(group%(qcolumns/16))*16+element;
    float value=0.f;
    if(token<tokens && col<m.input_columns) {
        const auto input=x+size_t(token)*stride+(Parallel?0:size_t(blockIdx.y)*m.input_columns)+col;
        value=scalar_float<DT>(__ldg(input));
        if(m.columns)value*=__half2float(__ldg(static_cast<const half*>(m.scales)+col));
    }
    float peak=fabsf(value);
    #pragma unroll
    for(int delta=8;delta;delta/=2)peak=fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,delta,16));
    const float scale=peak>0.f?peak/127.f:1.f;
    int q=__float2int_rn(value/scale);q=max(-127,min(127,q));
    int sum=q;
    #pragma unroll
    for(int delta=8;delta;delta/=2)sum+=__shfl_xor_sync(0xffffffff,sum,delta,16);
    const uint32_t byte=uint32_t(uint8_t(q));
    const uint32_t packed=byte|(__shfl_down_sync(0xffffffff,byte,1)<<8)|
        (__shfl_down_sync(0xffffffff,byte,2)<<16)|(__shfl_down_sync(0xffffffff,byte,3)<<24);
    if(token<tokens) {
        const uint64_t slot_words=uint64_t(tokens)*(qcolumns/16)*6;
        const int tile=col/512,local=(col%512)/16;
        constexpr int tile_groups=32;
        const uint64_t base=uint64_t(blockIdx.y)*slot_words+uint64_t(tile)*32*tokens*6;
        const int at=token*tile_groups+local;
        if(!(lane&3))output[base+uint64_t(at)*4+element/4]=packed;
        if(!element) {
            auto* coefficients=reinterpret_cast<float*>(output+base+4*tokens*tile_groups+uint64_t(at)*2);
            coefficients[0]=scale*TQ_Q8_BOOK_INVERSE;
            coefficients[1]=float(sum)*TQ_Q8_BOOK_CENTER*scale;
        }
    }
}
template<int DT,int T,bool Parallel,int Activation=0>
__global__ __launch_bounds__(128)
void q8_projection(TrellisProjectionView plain,const TrellisProjectionJob* jobs,int count,
        const uint32_t* input,Scalar<DT>* output,int qcolumns,int groups,
        int joined_count=0,int joined_rows=0,int streams=1) {
    TrellisProjectionJob job{};job.weight=plain;int begin=0,index=0;
    if constexpr(Parallel) {
        int hi=count;
        while(index<hi){const int mid=(index+hi)/2;if(blockIdx.x<unsigned(jobs[mid].block_end))hi=mid;else index=mid+1;}
        if(index==count)return;
        job=jobs[index];begin=index?jobs[index-1].block_end:0;
    }
    const auto m=job.weight;const int row=(blockIdx.x-begin)*4+threadIdx.x/32;
    if(row>=m.rows)return;
    const int projection=index;
    if constexpr(Parallel)index=m.columns?index:count;
    else index=row/(m.rows/groups);
    float value[T]{};
    const uint64_t slot_words=uint64_t(T)*(qcolumns/16)*6;
    q8_dispatch<T>(m,row,input+size_t(index)*slot_words,qcolumns,value);
    #pragma unroll
    for(int t=0;t<T;++t) {
        #pragma unroll
        for(int step=16;step;step/=2)value[t]+=__shfl_down_sync(0xffffffff,value[t],step);
        if((threadIdx.x&31)==0) {
            if(m.bias)value[t]+=__ldg(m.bias+row);
            if constexpr(Activation) {
                const float low=__fdiv_rn(value[t],float(streams));
                const float gate=1.f/(1.f+expf(-low));
                value[t]=__fmul_rn(Activation==2?2.f:low,gate);
            }
            const bool joined=Parallel && projection<joined_count;
            const size_t offset=Parallel?size_t(job.output_offset)*(joined?1:T):0;
            output[offset+size_t(t)*(joined?joined_rows:m.rows)+row]=scalar_round<DT>(value[t]);
        }
    }
}

template<int T>
__global__ __launch_bounds__(128)
void q8_residual_mix(TrellisProjectionView m,const uint32_t* input,const float* normalized,
        float* output,int qcolumns,int streams) {
    const int hidden=m.rows/streams,column=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31;
    if(column>=hidden)return;
    float result[T]{};
    // Keep the singleton projection's dot/reduction tree and stream-order
    // additions. The expanded projection never needs a temporary output.
    for(int stream=0;stream<streams;++stream) {
        const int row=stream*hidden+column;
        float value[T]{};q8_dispatch<T>(m,row,input,qcolumns,value);
        #pragma unroll
        for(int t=0;t<T;++t) {
            #pragma unroll
            for(int step=16;step;step/=2)value[t]+=__shfl_down_sync(0xffffffff,value[t],step);
            if(lane==0) {
                if(m.bias)value[t]+=__ldg(m.bias+row);
                const float gate=1.f/(1.f+expf(-value[t]));
                result[t]=__fadd_rn(result[t],__fmul_rn(gate,normalized[size_t(t)*m.rows+row]));
            }
        }
    }
    if(lane==0) {
        #pragma unroll
        for(int t=0;t<T;++t)output[size_t(t)*hidden+column]=__fdiv_rn(result[t],float(streams));
    }
}

// __grid_constant__ (SM70+) lets every thread address the same parameter
// image without making a local copy. Older targets read the existing device
// plan. 32 descriptors plus other arguments fit the 4-KiB parameter limit.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define TQ_GRID_CONSTANT __grid_constant__
#else
#define TQ_GRID_CONSTANT
#endif
struct TqInlineProjectionPlan {TrellisProjectionJob jobs[32];};
template<int DT,int T>
__global__ __launch_bounds__(128)
void q8_inline_projection(TQ_GRID_CONSTANT const TqInlineProjectionPlan plan,const TrellisProjectionJob* device_jobs,int count,int first,int begin_block,int common_slot,
        const uint32_t* input,Scalar<DT>* output,int qc,int joined,int joined_rows) {
#if __CUDA_ARCH__ >= 700
    const auto& jobs=plan.jobs;
#else
    const auto* jobs=device_jobs+first;
#endif
    const int block=blockIdx.x+begin_block;
    int lo=0,hi=count;
    while(lo<hi){const int mid=(lo+hi)/2;if(block<unsigned(jobs[mid].block_end))hi=mid;else lo=mid+1;}
    if(lo==count)return;
    const auto job=jobs[lo];const auto m=job.weight;
    const int row=(block-(lo?jobs[lo-1].block_end:begin_block))*4+threadIdx.x/32;
    if(row>=m.rows)return;
    const int projection=first+lo;
    const int slot=m.columns?projection:common_slot;
    float value[T]{};
    const uint64_t slot_words=uint64_t(T)*(qc/16)*6;
    q8_dispatch<T>(m,row,input+size_t(slot)*slot_words,qc,value);
    #pragma unroll
    for(int t=0;t<T;++t) {
        #pragma unroll
        for(int step=16;step;step/=2)value[t]+=__shfl_down_sync(0xffffffff,value[t],step);
        if((threadIdx.x&31)==0){if(m.bias)value[t]+=__ldg(m.bias+row);
            const size_t base=size_t(job.output_offset)*(projection<joined?1:T);
            output[base+size_t(t)*(projection<joined?joined_rows:m.rows)+row]=scalar_round<DT>(value[t]);}
    }
}

static_assert(sizeof(TqInlineProjectionPlan)+256<=4096,"projection parameters exceed 4 KiB");
#undef TQ_GRID_CONSTANT

static int tq_prepare_q8_tokens(TrellisProjectionView m,const void* x,void* y,
        int tokens,int groups,int dtype,void* scratch,uint64_t bytes,cudaStream_t stream) {
    if(!x || !y || x==y || tokens<1 || tokens>4 || groups<1 || groups>65535 || (dtype!=0 && dtype!=2) ||
       m.rows<1 || m.rows%groups || m.input_columns<1 || m.input_columns>INT_MAX-511 ||
       m.input_columns>INT_MAX/groups || m.input_columns>m.stored_columns || m.stored_columns%32 ||
       !m.words || !m.descriptors || !m.scales || !m.relative ||
       (!m.columns && (m.gslog<4 || m.gslog>20)))return -1;
    const int qc=int((uint64_t(m.input_columns)+511)/512*512);
    const uint64_t slot_bytes=uint64_t(tokens)*(qc/16)*24;
    const uint64_t required=uint64_t(groups)*slot_bytes;
    if(!scratch || scratch==x || scratch==y || (reinterpret_cast<uintptr_t>(scratch)&15) || bytes<required)return -1;
    auto* quant=static_cast<uint32_t*>(scratch);
    #define PREPARE(D) q8_prepare<D,false><<<dim3((uint64_t(tokens)*qc+127)/128,groups),128,0,stream>>>(m,nullptr,static_cast<const Scalar<D>*>(x),quant,tokens,m.input_columns*groups,qc,groups)
    if(dtype==0){PREPARE(0);}else{PREPARE(2);}
    #undef PREPARE
    return int(cudaGetLastError());
}
extern "C" int mfq_tq_q8_tokens(TrellisProjectionView m,const void* x,void* y,
        int tokens,int groups,int dtype,void* scratch,uint64_t bytes,cudaStream_t stream) {
    const int prepared=tq_prepare_q8_tokens(m,x,y,tokens,groups,dtype,scratch,bytes,stream);
    if(prepared)return prepared;
    bool contraction=false;const int dispatch=tq_q8_contraction(m,contraction);if(dispatch)return dispatch;
    const int qc=int((uint64_t(m.input_columns)+511)/512*512);
    const auto* quant=static_cast<const uint32_t*>(scratch);
    #define RUN(D,T) do { \
        if(contraction)q8_contraction_projection<D,T><<<m.rows,256,0,stream>>>(m,quant,static_cast<Scalar<D>*>(y),qc,groups); \
        else q8_projection<D,T,false><<<(m.rows+3)/4,128,0,stream>>>(m,nullptr,0,quant,static_cast<Scalar<D>*>(y),qc,groups); \
    }while(false)
    #define TOKENS(D) if(tokens==1){RUN(D,1);}else if(tokens==2){RUN(D,2);}else if(tokens==3){RUN(D,3);}else {RUN(D,4);}
    if(dtype==0){TOKENS(0);}else{TOKENS(2);}
    #undef TOKENS
    #undef RUN
    const auto result=cudaGetLastError();
    if(result==cudaSuccess && contraction)tq_trace_q8_contraction(m,tokens);
    return int(result);
}
extern "C" int mfq_tq_q8_residual_tokens(TrellisProjectionView m,const float* x,const float* normalized,
        float* output,int tokens,int streams,bool injection,void* scratch,uint64_t bytes,cudaStream_t stream) {
    if(streams<1 || (normalized && (m.rows%streams || normalized==output || normalized==scratch)))return -1;
    const int prepared=tq_prepare_q8_tokens(m,x,output,tokens,1,2,scratch,bytes,stream);
    if(prepared)return prepared;
    bool contraction=false;
    if(!normalized){const int dispatch=tq_q8_contraction(m,contraction);if(dispatch)return dispatch;}
    const int qc=int((uint64_t(m.input_columns)+511)/512*512);
    const auto* quant=static_cast<const uint32_t*>(scratch);
    #define RUN(T) do { \
        if(normalized)q8_residual_mix<T><<<(uint64_t(m.rows/streams)+3)/4,128,0,stream>>>(m,quant,normalized,output,qc,streams); \
        else if(contraction) { \
            if(injection)q8_contraction_projection<2,T,2><<<m.rows,256,0,stream>>>(m,quant,output,qc,1,streams); \
            else q8_contraction_projection<2,T,1><<<m.rows,256,0,stream>>>(m,quant,output,qc,1,streams); \
        } \
        else if(injection)q8_projection<2,T,false,2><<<(uint64_t(m.rows)+3)/4,128,0,stream>>>(m,nullptr,0,quant,output,qc,1,0,0,streams); \
        else q8_projection<2,T,false,1><<<(uint64_t(m.rows)+3)/4,128,0,stream>>>(m,nullptr,0,quant,output,qc,1,0,0,streams); \
    }while(false)
    if(tokens==1){RUN(1);}else if(tokens==2){RUN(2);}else if(tokens==3){RUN(3);}else{RUN(4);}
    #undef RUN
    const auto result=cudaGetLastError();
    if(result==cudaSuccess && contraction)tq_trace_q8_contraction(m,tokens);
    return int(result);
}
extern "C" int mfq_tq_q8_residual_down(TrellisProjectionView m,const float* input,
        const uint16_t* injection,float* output,float* gates,int tokens,int streams,
        void* scratch,uint64_t bytes,cudaStream_t stream) {
    if(streams<1 || !injection || !gates || gates==input || gates==output || gates==scratch ||
            reinterpret_cast<const void*>(injection)==scratch)return -1;
    const int prepared=tq_prepare_q8_tokens(m,input,output,tokens,1,2,scratch,bytes,stream);
    if(prepared)return prepared;
    bool contraction=false;const int dispatch=tq_q8_contraction(m,contraction);if(dispatch)return dispatch;
    const int qc=int((uint64_t(m.input_columns)+511)/512*512);
    const int blocks=contraction?m.rows:int((uint64_t(m.rows)+7)/8);
    if(streams>INT_MAX-blocks)return -1;
    const auto* quant=static_cast<const uint32_t*>(scratch);
    #define RUN(T) do { \
        if(contraction)q8_residual_down_injection<T,true><<<blocks+streams,256,0,stream>>>(m,quant,input,injection,output,gates,qc,streams,blocks); \
        else q8_residual_down_injection<T,false><<<blocks+streams,256,0,stream>>>(m,quant,input,injection,output,gates,qc,streams,blocks); \
    }while(false)
    if(tokens==1){RUN(1);}else if(tokens==2){RUN(2);}else if(tokens==3){RUN(3);}else{RUN(4);}
    #undef RUN
    return int(cudaGetLastError());
}
extern "C" int mfq_tq_q8_parallel_tokens(const TrellisProjectionJob* host_jobs,const TrellisProjectionJob* jobs,int count,int blocks,
        const void* x,void* y,int tokens,int stride,int dtype,void* scratch,uint64_t bytes,
        int joined_count,int joined_rows,cudaStream_t stream) {
    if(!host_jobs || !jobs || !x || !y || x==y || count<1 || count>=65535 || blocks<1 || tokens<1 || tokens>4 || (dtype!=0 && dtype!=2) ||
       stride<1 || stride>INT_MAX-511 || joined_count<0 || joined_count>count ||
       (joined_count && joined_rows<1))return -1;
    bool column_scaled=false;
    for(int i=0;i<count;++i)column_scaled|=host_jobs[i].weight.columns!=0;
    const int qc=int((uint64_t(stride)+511)/512*512);
    const uint64_t slot_bytes=uint64_t(tokens)*(qc/16)*24;
    const uint64_t required=(column_scaled?uint64_t(count+1):1)*slot_bytes;
    if(!scratch || scratch==x || scratch==y || (reinterpret_cast<uintptr_t>(scratch)&15) || bytes<required)return -1;
    auto* quant=static_cast<uint32_t*>(scratch);
    TrellisProjectionView common{};common.input_columns=stride;
    #define PREPARE(D) do { \
        if(column_scaled) { \
            q8_prepare<D,true><<<dim3((uint64_t(tokens)*qc+127)/128,count+1),128,0,stream>>>(common,jobs,static_cast<const Scalar<D>*>(x),quant,tokens,stride,qc,count); \
        }else { \
            q8_prepare<D,false><<<(uint64_t(tokens)*qc+127)/128,128,0,stream>>>(common,nullptr,static_cast<const Scalar<D>*>(x),quant,tokens,stride,qc,1); \
        } \
    }while(false)
    if(dtype==0){PREPARE(0);}else{PREPARE(2);}
    #undef PREPARE
    const auto prepared=cudaGetLastError();if(prepared!=cudaSuccess)return int(prepared);
    for(int first=0;first<count;first+=32) {
        const int batch=std::min(32,count-first);
        TqInlineProjectionPlan plan{};
        for(int i=0;i<batch;++i)plan.jobs[i]=host_jobs[first+i];
        const int begin=first?host_jobs[first-1].block_end:0;
        const int end=host_jobs[first+batch-1].block_end;
        if(end<=begin || end>blocks)return -1;
        #define RUN(D,T) q8_inline_projection<D,T><<<end-begin,128,0,stream>>>(plan,jobs,batch,first,begin,column_scaled?count:0,quant,static_cast<Scalar<D>*>(y),qc,joined_count,joined_rows)
        #define TOKENS(D) if(tokens==1){RUN(D,1);}else if(tokens==2){RUN(D,2);}else if(tokens==3){RUN(D,3);}else {RUN(D,4);}
        if(dtype==0){TOKENS(0);}else{TOKENS(2);}
        #undef TOKENS
        #undef RUN
        const auto error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    }
    return 0;
}
