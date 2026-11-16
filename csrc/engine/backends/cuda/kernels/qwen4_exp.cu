#include "qwen4_raw.h"
#include "engine/backends/cuda/kernels/kernel_dispatch_trace.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>

namespace {
template<class Value,class Position>
__global__ void live_index_pool_kernel(const Value* index,const Position* positions,
        const int64_t* cache,Value* pooled,int64_t* blocks,Position* block_positions,
        int tokens,int width,int pool,int64_t maximum,int64_t position_entries,
        int64_t row_stride,int64_t column_stride) {
    const int token=blockIdx.x;
    const int64_t block=cache[token]/pool,start=block*pool;
    if(threadIdx.x==0)blocks[token]=block;
    for(int64_t i=threadIdx.x;i<position_entries/tokens;i+=blockDim.x) {
        const int64_t at=i*tokens+token;
        block_positions[at]=Position(int64_t(positions[at])-cache[token]+start);
    }
    for(int column=threadIdx.x;column<width;column+=blockDim.x) {
        float sum=0;
        for(int i=0;i<pool;++i) {
            const auto row=min(max(int64_t(0),start+i),maximum-1);
            sum=__fadd_rn(sum,float(index[row*row_stride+column*column_stride]));
        }
        pooled[int64_t(token)*width+column]=Value(__fdiv_rn(sum,float(pool)));
    }
}

template<class Position>
__global__ void fill_decode_positions_kernel(int64_t* cache,Position* positions,
    int64_t tokens,int64_t entries,int64_t start,int64_t delta) {
    for(int64_t i=threadIdx.x;i<entries;i+=blockDim.x) {
        positions[i]=static_cast<Position>(start+i%tokens+delta);
        if(i<tokens)cache[i]=start+i;
    }
}

template<class Value, bool Injection>
__global__ void gated_residual_activation_kernel(const Value* input,Value* output,
    int64_t count,float streams) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
        const Value low=static_cast<Value>(__fdiv_rn(static_cast<float>(input[i]),streams));
        const Value gate=static_cast<Value>(1.0f/(1.0f+expf(-static_cast<float>(low))));
        const float multiplier=Injection?2.0f:static_cast<float>(low);
        output[i]=static_cast<Value>(__fmul_rn(multiplier,static_cast<float>(gate)));
    }
}


template<class Gate, class Value>
__global__ void gated_residual_mix_kernel(const Gate* projection,const Value* normalized,
    std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>* output,
    int64_t count,int hidden,int streams) {
    using Output=std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t base=(i/hidden)*hidden*streams+i%hidden;
        float sum=0;
        for(int stream=0;stream<streams;++stream) {
            const int64_t at=base+int64_t(stream)*hidden;
            const Gate gate=static_cast<Gate>(1.0f/(1.0f+expf(-static_cast<float>(projection[at]))));
            const Output product=static_cast<Output>(__fmul_rn(static_cast<float>(gate),static_cast<float>(normalized[at])));
            sum=__fadd_rn(sum,static_cast<float>(product));
        }
        output[i]=static_cast<Output>(sum/streams);
    }
}
template<class Branch,class Residual,class Injection>
using ResidualUpdate=std::conditional_t<std::is_same_v<Branch,float> ||
    std::is_same_v<Injection,float>,float,__half>;
template<class Branch,class Residual,class Injection>
using ResidualOutput=std::conditional_t<std::is_same_v<Residual,float> ||
    std::is_same_v<ResidualUpdate<Branch,Residual,Injection>,float>,float,__half>;

template<class Branch,class Residual,class Injection>
__global__ void gated_residual_post_kernel(const Branch* branch,const Residual* residual,
    const Injection* injection,ResidualOutput<Branch,Residual,Injection>* output,int64_t count,int hidden,int streams) {
    using Update=ResidualUpdate<Branch,Residual,Injection>;
    using Output=ResidualOutput<Branch,Residual,Injection>;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t sample=i/(int64_t(hidden)*streams);
        const int stream=int((i/hidden)%streams);
        const Update update=static_cast<Update>(__fmul_rn(static_cast<float>(branch[sample*hidden+i%hidden]),
            static_cast<float>(injection[sample*streams+stream])));
        output[i]=static_cast<Output>(__fadd_rn(static_cast<float>(residual[i]),static_cast<float>(update)));
    }
}

template<class Branch,class Residual,class Injection>
__global__ void gated_residual_post_norm_kernel(const Branch* branch,const Residual* residual,
    const Injection* injection,const float* weight,
    ResidualOutput<Branch,Residual,Injection>* updated,
    ResidualOutput<Branch,Residual,Injection>* normalized,int hidden,int streams,float eps) {
    using Update=ResidualUpdate<Branch,Residual,Injection>;
    using Output=ResidualOutput<Branch,Residual,Injection>;
    const int stream=blockIdx.x%streams;
    const int64_t sample=blockIdx.x/streams,base=int64_t(blockIdx.x)*hidden;
    const float gate=static_cast<float>(injection[sample*streams+stream]);
    float sum=0;
    for(int column=threadIdx.x;column<hidden;column+=blockDim.x) {
        const Update product=static_cast<Update>(__fmul_rn(
            static_cast<float>(branch[sample*hidden+column]),gate));
        const Output value=static_cast<Output>(__fadd_rn(
            static_cast<float>(residual[base+column]),static_cast<float>(product)));
        updated[base+column]=value;
        const float f=static_cast<float>(value);
        sum=__fadd_rn(sum,__fmul_rn(f,f));
    }
    __shared__ float partial[256];partial[threadIdx.x]=sum;__syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2) {
        if(threadIdx.x<stride)partial[threadIdx.x]=__fadd_rn(partial[threadIdx.x],partial[threadIdx.x+stride]);
        __syncthreads();
    }
    const float mean=__fdiv_rn(partial[0],static_cast<float>(hidden));
    const float inverse=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(mean,eps)));
    for(int column=threadIdx.x;column<hidden;column+=blockDim.x) {
        const float value=__fmul_rn(static_cast<float>(updated[base+column]),inverse);
        normalized[base+column]=static_cast<Output>(__fmul_rn(
            value,__fadd_rn(weight[int64_t(stream)*hidden+column],1.0f)));
    }
}


template<class Query,class Key>
__global__ void live_block_scores_kernel(const Query* query,const Key* keys,const int64_t* positions,
        float* output,int pools,int heads,int width,int pool,int64_t qs1,int64_t qs2,int64_t qs3,
        int64_t ks1,int64_t ks2) {
    const int lane=threadIdx.x&31,row=int(blockIdx.x)*4+threadIdx.x/32,token=int(blockIdx.y);
    if(row>=pools)return;
    if(int64_t(row+1)*pool>positions[token]+1) {
        if(lane==0)output[int64_t(token)*pools+row]=-1e30f;
        return;
    }
    float score=0;
    for(int head=0;head<heads;++head) {
        float dot=0;
        for(int d=lane;d<width;d+=32)
            dot=fmaf(float(query[int64_t(token)*qs1+int64_t(head)*qs2+int64_t(d)*qs3]),
                float(keys[int64_t(row)*ks1+int64_t(d)*ks2]),dot);
        for(int delta=16;delta;delta/=2)dot+=__shfl_down_sync(0xffffffff,dot,delta);
        score+=fmaxf(0.f,dot);
    }
    if(lane==0)output[int64_t(token)*pools+row]=score/sqrtf(float(width));
}

__device__ void emit_selected_blocks(const int64_t* ids,int32_t* selected,int64_t position,
        int pool,int count,int budget,int row) {
    const int slots=(budget+2*pool-2)/pool;
    const uint32_t full=pool==32?0xffffffffu:(1u<<pool)-1;
    for(int slot=threadIdx.x;slot<slots;slot+=blockDim.x) {
        int64_t id=-1;uint32_t mask=0;
        if(position+1<=budget) {
            const int64_t live=position+1-int64_t(slot)*pool;
            if(live>0){id=slot;mask=live>=pool?full:(1u<<int(live))-1;}
        }else if(slot<count) {
            const auto block=ids[slot];
            if(block*pool+pool-1<=position){id=block;mask=full;}
        }else if(slot==budget/pool) {
            const int tail=int((position+1)%pool);
            if(tail){id=(position+1)/pool;mask=(1u<<tail)-1;}
        }
        const auto at=(int64_t(row)*slots+slot)*2;
        selected[at]=int32_t(id);selected[at+1]=int32_t(mask);
    }
}
__global__ void selected_blocks_from_ids(const int64_t* ids,const int64_t* positions,int32_t* selected,
        int pool,int count,int budget) {
    const int row=blockIdx.x;
    emit_selected_blocks(ids+int64_t(row)*count,selected,positions[row],pool,count,budget,row);
}
#include "attention/qsa/topk_threshold.cuh"

template<int Items>
__device__ void live_topk_sort(const float* source,const int64_t* ids,float* output,int64_t* out_ids,
        int row,int columns,int begin,int live,int pools,int count,int64_t out,unsigned char* temporary,bool emit_tokens) {
    // Candidate columns are already bounded by INT_MAX at the public API.
    // Keep CUB's internal values compact; widen only when publishing IDs.
    using Sort=cub::BlockRadixSort<float,256,Items,int32_t>;
    float keys[Items];int32_t indices[Items];
#pragma unroll
    for(int i=0;i<Items;++i) {
        const int column=begin+threadIdx.x*Items+i;
        const auto at=int64_t(row)*columns+column;
        keys[i]=column<live?source[at]:-CUDART_INF_F;
        indices[i]=column<live?int32_t(ids?ids[at]:column):pools;
    }
    Sort(*reinterpret_cast<typename Sort::TempStorage*>(temporary)).SortDescending(keys,indices);
    // CUB no longer needs its storage once every thread has its sorted
    // register values. Reuse it for the final token expansion instead of
    // writing and rereading an otherwise unused global candidate list.
    if(emit_tokens)__syncthreads();
    auto* selected_ids=emit_tokens?reinterpret_cast<int64_t*>(temporary):out_ids+out;
#pragma unroll
    for(int i=0;i<Items;++i) {
        const int rank=threadIdx.x*Items+i;
        if(rank<count){if(!emit_tokens)output[out+rank]=keys[i];selected_ids[rank]=indices[i];}
    }
}
template<bool Blocks,int Items=16>
__global__ void live_topk_blocks_kernel(const float* source,const int64_t* ids,
        float* output,int64_t* out_ids,const int64_t* positions,int columns,
        int pools,int pool,int count,int stage,bool compact,int32_t* selected,int budget) {
    constexpr int tile=4096;
    extern __shared__ __align__(16) unsigned char temporary[];
    const int row=blockIdx.y,begin=int(blockIdx.x)*tile,tiles=int((int64_t(columns)+tile-1)/tile);
    int live=int(min(int64_t(pools),max(int64_t(0),(positions[row]+1)/pool)));
    for(int s=0;s<stage;++s)live=live/tile*count+min(count,live%tile);
    const int available=min(tile,max(0,live-begin));
    const int64_t out=(int64_t(row)*tiles+blockIdx.x)*count;
    auto* selected_ids=selected?reinterpret_cast<int64_t*>(temporary):out_ids+out;
    // Entirely unused tiles and already reduced candidate lists need no sort.
    if(available<=count) {
        for(int i=threadIdx.x;i<count;i+=256) {
            const auto at=int64_t(row)*columns+begin+i;
            if(!selected)output[out+i]=i<available?source[at]:-1e30f;
            selected_ids[i]=i<available?(ids?ids[at]:begin+i):pools;
        }
    } else {
    // The graph reserves the maximum context, while each replay has a much
    // shorter live prefix. Keep the reduction tiles and output layout fixed,
    // but sort only the power-of-two range containing this tile's live rows.
    // CUB's stable ordering is unchanged because input column order is kept.
    if constexpr(Blocks)
        live_topk_threshold<Items>(source,ids,output,out_ids,row,columns,begin,available,pools,count,out,temporary,selected!=nullptr);
    else if(compact && available<=256)
        live_topk_sort<1>(source,ids,output,out_ids,row,columns,begin,live,pools,count,out,temporary,selected!=nullptr);
    else if(compact && available<=512)
        live_topk_sort<2>(source,ids,output,out_ids,row,columns,begin,live,pools,count,out,temporary,selected!=nullptr);
    else if(compact && available<=1024)
        live_topk_sort<4>(source,ids,output,out_ids,row,columns,begin,live,pools,count,out,temporary,selected!=nullptr);
    else if(compact && available<=2048)
        live_topk_sort<8>(source,ids,output,out_ids,row,columns,begin,live,pools,count,out,temporary,selected!=nullptr);
    else live_topk_sort<16>(source,ids,output,out_ids,row,columns,begin,live,pools,count,out,temporary,selected!=nullptr);
    }
    if(selected) {
        // Only the final reduction has one CTA per query. Its IDs are now
        // complete; publish them to this CTA before expanding whole pools
        // and the incomplete causal tail into the existing token order.
        __syncthreads();
        const int64_t position=positions[row];
        if constexpr(Blocks) {
            emit_selected_blocks(selected_ids,selected,position,pool,count,budget,row);
            return;
        }
        const int width=budget+pool-1;
        for(int slot=threadIdx.x;slot<width;slot+=blockDim.x) {
            int64_t token=-1;
            if(position+1<=budget) {
                if(slot<=position)token=slot;
            } else if(int64_t(slot)<int64_t(count)*pool) {
                const int64_t block=selected_ids[slot/pool];
                if(block*pool+pool-1<=position)token=block*pool+slot%pool;
            } else if(slot>=budget) {
                const int offset=slot-budget;
                const int64_t tail=(position+1)%pool;
                if(offset<tail)token=position+1-tail+offset;
            }
            selected[int64_t(row)*width+slot]=int32_t(token);
        }
    }
}
}

namespace tyloi::cuda::qwen4_raw {
template<class V, class P>
void pool_t(const void* x,const void* p,const int64_t* cache,void* out,int64_t* blocks,void* pos,
    int tokens,int width,int pool,int64_t maximum,int64_t entries,int64_t rs,int64_t cs,cudaStream_t stream) {
    live_index_pool_kernel<<<tokens,128,0,stream>>>(static_cast<const V*>(x),static_cast<const P*>(p),cache,
        static_cast<V*>(out),blocks,static_cast<P*>(pos),tokens,width,pool,maximum,entries,rs,cs);
}
void live_pool(const void* x,const void* p,const int64_t* cache,void* out,int64_t* blocks,void* pos,
    int tokens,int width,int pool,int64_t maximum,int64_t entries,int64_t rs,int64_t cs,
    NumericType type,bool wide,cudaStream_t stream) {
    if(type==NumericType::f16) {
        if(wide) pool_t<__half,int64_t>(x,p,cache,out,blocks,pos,tokens,width,pool,maximum,entries,rs,cs,stream);
        else pool_t<__half,int32_t>(x,p,cache,out,blocks,pos,tokens,width,pool,maximum,entries,rs,cs,stream);
    } else if(wide) pool_t<float,int64_t>(x,p,cache,out,blocks,pos,tokens,width,pool,maximum,entries,rs,cs,stream);
    else pool_t<float,int32_t>(x,p,cache,out,blocks,pos,tokens,width,pool,maximum,entries,rs,cs,stream);
}
void fill_positions(int64_t* cache,void* pos,int64_t tokens,int64_t entries,int64_t start,int64_t delta,bool wide,cudaStream_t stream) {
    if(wide) fill_decode_positions_kernel<<<1,32,0,stream>>>(cache,static_cast<int64_t*>(pos),tokens,entries,start,delta);
    else fill_decode_positions_kernel<<<1,32,0,stream>>>(cache,static_cast<int32_t*>(pos),tokens,entries,start,delta);
}
void activation(const void* x,void* out,int64_t count,float streams,NumericType type,bool inject,cudaStream_t stream) {
    int blocks=int((count+255)/256);
    if(type==NumericType::f16) {
        if(inject) gated_residual_activation_kernel<__half,true><<<blocks,256,0,stream>>>(static_cast<const __half*>(x),static_cast<__half*>(out),count,streams);
        else gated_residual_activation_kernel<__half,false><<<blocks,256,0,stream>>>(static_cast<const __half*>(x),static_cast<__half*>(out),count,streams);
    } else {
        if(inject) gated_residual_activation_kernel<float,true><<<blocks,256,0,stream>>>(static_cast<const float*>(x),static_cast<float*>(out),count,streams);
        else gated_residual_activation_kernel<float,false><<<blocks,256,0,stream>>>(static_cast<const float*>(x),static_cast<float*>(out),count,streams);
    }
}
template<class G,class V>
void mix_t(const void* gate,const void* x,void* out,int64_t count,int hidden,int streams,cudaStream_t stream) {
    using O=std::conditional_t<std::is_same_v<G,float> || std::is_same_v<V,float>,float,__half>;
    gated_residual_mix_kernel<<<int((count+255)/256),256,0,stream>>>(static_cast<const G*>(gate),static_cast<const V*>(x),static_cast<O*>(out),count,hidden,streams);
}
void mix(const void* gate,const void* x,void* out,int64_t count,int hidden,int streams,NumericType gt,NumericType xt,cudaStream_t stream) {
    if(gt==NumericType::f16) { if(xt==NumericType::f16) mix_t<__half,__half>(gate,x,out,count,hidden,streams,stream); else mix_t<__half,float>(gate,x,out,count,hidden,streams,stream); }
    else { if(xt==NumericType::f16) mix_t<float,__half>(gate,x,out,count,hidden,streams,stream); else mix_t<float,float>(gate,x,out,count,hidden,streams,stream); }
}
template<class B,class R,class I>
void post_t(const void* b,const void* r,const void* i,void* o,int64_t count,int hidden,int streams,cudaStream_t stream) {
    using O=ResidualOutput<B,R,I>;
    gated_residual_post_kernel<B,R,I><<<int((count+255)/256),256,0,stream>>>(static_cast<const B*>(b),static_cast<const R*>(r),static_cast<const I*>(i),static_cast<O*>(o),count,hidden,streams);
    if(dispatch_trace::enabled())dispatch_trace::record(stream,"gr_residual_post",reinterpret_cast<const void*>(gated_residual_post_kernel<B,R,I>),
        "M="+std::to_string(count/(int64_t(hidden)*streams))+" hidden="+std::to_string(hidden)+" streams="+std::to_string(streams)+
        " type_flags="+std::to_string((std::is_same_v<B,float>?4:0)|(std::is_same_v<R,float>?2:0)|(std::is_same_v<I,float>?1:0))+
        " inplace="+std::to_string(o==r));
}
void post(const void* b,const void* r,const void* i,void* o,int64_t count,int hidden,int streams,NumericType bt,NumericType rt,NumericType it,cudaStream_t stream) {
    const int mode=(bt==NumericType::f32?4:0)|(rt==NumericType::f32?2:0)|(it==NumericType::f32?1:0);
    switch(mode) {
    case 0:post_t<__half,__half,__half>(b,r,i,o,count,hidden,streams,stream);break;
    case 1:post_t<__half,__half,float>(b,r,i,o,count,hidden,streams,stream);break;
    case 2:post_t<__half,float,__half>(b,r,i,o,count,hidden,streams,stream);break;
    case 3:post_t<__half,float,float>(b,r,i,o,count,hidden,streams,stream);break;
    case 4:post_t<float,__half,__half>(b,r,i,o,count,hidden,streams,stream);break;
    case 5:post_t<float,__half,float>(b,r,i,o,count,hidden,streams,stream);break;
    case 6:post_t<float,float,__half>(b,r,i,o,count,hidden,streams,stream);break;
    case 7:post_t<float,float,float>(b,r,i,o,count,hidden,streams,stream);break;
    }
}
template<class B,class R,class I>
void post_norm_t(const void* b,const void* r,const void* i,const float* w,void* u,void* n,int64_t count,int hidden,int streams,float eps,cudaStream_t stream) {
    using O=ResidualOutput<B,R,I>; int threads=hidden<=32?32:hidden<=64?64:hidden<=128?128:256;
    gated_residual_post_norm_kernel<B,R,I><<<count/hidden,threads,0,stream>>>(static_cast<const B*>(b),static_cast<const R*>(r),static_cast<const I*>(i),w,static_cast<O*>(u),static_cast<O*>(n),hidden,streams,eps);
}
void post_norm(const void* b,const void* r,const void* i,const float* w,void* u,void* n,int64_t count,int hidden,int streams,float eps,NumericType bt,NumericType rt,NumericType it,cudaStream_t stream) {
    int mode=(bt==NumericType::f32?4:0)|(rt==NumericType::f32?2:0)|(it==NumericType::f32?1:0);
    switch(mode) {
    case 0:post_norm_t<__half,__half,__half>(b,r,i,w,u,n,count,hidden,streams,eps,stream);break;
    case 1:post_norm_t<__half,__half,float>(b,r,i,w,u,n,count,hidden,streams,eps,stream);break;
    case 2:post_norm_t<__half,float,__half>(b,r,i,w,u,n,count,hidden,streams,eps,stream);break;
    case 3:post_norm_t<__half,float,float>(b,r,i,w,u,n,count,hidden,streams,eps,stream);break;
    case 4:post_norm_t<float,__half,__half>(b,r,i,w,u,n,count,hidden,streams,eps,stream);break;
    case 5:post_norm_t<float,__half,float>(b,r,i,w,u,n,count,hidden,streams,eps,stream);break;
    case 6:post_norm_t<float,float,__half>(b,r,i,w,u,n,count,hidden,streams,eps,stream);break;
    case 7:post_norm_t<float,float,float>(b,r,i,w,u,n,count,hidden,streams,eps,stream);break;
    }
}
template<class Q,class K>
void scores_t(const void* q,const void* k,const int64_t* p,float* out,int tokens,int pools,int heads,int width,int pool,int64_t q1,int64_t q2,int64_t q3,int64_t k1,int64_t k2,cudaStream_t stream) {
    live_block_scores_kernel<<<dim3(unsigned((pools+3)/4),unsigned(tokens)),128,0,stream>>>(static_cast<const Q*>(q),static_cast<const K*>(k),p,out,pools,heads,width,pool,q1,q2,q3,k1,k2);
}
void live_scores(const void* q,const void* k,const int64_t* p,float* out,int tokens,int pools,int heads,int width,int pool,int64_t q1,int64_t q2,int64_t q3,int64_t k1,int64_t k2,NumericType qt,NumericType kt,cudaStream_t stream) {
    if(qt==NumericType::f16) { if(kt==NumericType::f16) scores_t<__half,__half>(q,k,p,out,tokens,pools,heads,width,pool,q1,q2,q3,k1,k2,stream); else scores_t<__half,float>(q,k,p,out,tokens,pools,heads,width,pool,q1,q2,q3,k1,k2,stream); }
    else { if(kt==NumericType::f16) scores_t<float,__half>(q,k,p,out,tokens,pools,heads,width,pool,q1,q2,q3,k1,k2,stream); else scores_t<float,float>(q,k,p,out,tokens,pools,heads,width,pool,q1,q2,q3,k1,k2,stream); }
}
cudaError_t prepare_live_topk() {
    constexpr auto bytes=sizeof(cub::BlockRadixSort<float,256,16,int32_t>::TempStorage);
    if constexpr(bytes>49152) return cudaFuncSetAttribute(live_topk_blocks_kernel<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes));
    return cudaSuccess;
}
void selected_blocks(const int64_t* ids,const int64_t* positions,int32_t* selected,int rows,int pool,int count,int budget,cudaStream_t stream) {
    selected_blocks_from_ids<<<unsigned(rows),256,0,stream>>>(ids,positions,selected,pool,count,budget);
}
void live_topk(const float* keys,const int64_t* ids,float* out,int64_t* out_ids,const int64_t* positions,
    int rows,int columns,int pools,int pool,int count,int stage,bool compact,int32_t* selected,int budget,bool blocks,cudaStream_t stream) {
    const int tiles=int((int64_t(columns)+4095)/4096);
    const auto sort_bytes=blocks?std::size_t(0):compact && columns<=1024?sizeof(cub::BlockRadixSort<float,256,4,int32_t>::TempStorage):
        compact && columns<=2048?sizeof(cub::BlockRadixSort<float,256,8,int32_t>::TempStorage):sizeof(cub::BlockRadixSort<float,256,16,int32_t>::TempStorage);
    const auto bytes=std::max(std::max(sort_bytes,blocks?std::size_t(8192):std::size_t(0)),budget?std::size_t(count)*sizeof(int64_t):std::size_t(0));
    const auto launch=[&]<bool Blocks,int Items=16>() {
        live_topk_blocks_kernel<Blocks,Items><<<dim3(tiles,rows),256,bytes,stream>>>(keys,ids,out,out_ids,positions,columns,pools,pool,count,stage,compact,tiles==1?selected:nullptr,budget);
        if(dispatch_trace::enabled())dispatch_trace::record(stream,"qsa_live_topk",reinterpret_cast<const void*>(live_topk_blocks_kernel<Blocks,Items>),
            "M="+std::to_string(rows)+" candidate_columns="+std::to_string(columns)+" count="+std::to_string(count)+
            " selection_fused="+std::to_string(tiles==1 && selected)+" block_output="+std::to_string(Blocks)+
            " selection_algorithm="+std::string(Blocks?"radix_threshold":"radix_sort")+
            " minimum_sort_items="+std::to_string(compact?1:16)+" shared_bytes="+std::to_string(bytes));
    };
    if(blocks) {
        if(columns<=256)launch.template operator()<true,1>();
        else if(columns<=512)launch.template operator()<true,2>();
        else if(columns<=1024)launch.template operator()<true,4>();
        else if(columns<=2048)launch.template operator()<true,8>();
        else launch.template operator()<true,16>();
    } else launch.template operator()<false>();
}
}
