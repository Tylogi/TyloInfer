#include "gguf_w8a8.h"

#include <cuda_fp16.h>
#include <cublasLt.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>

namespace mfq::cuda::gguf::w8a8 {
namespace {
void checked(cudaError_t value) {
    if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));
}
void checked(cublasStatus_t value) {
    if(value!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("W8A8 cuBLASLt error: "+std::to_string(int(value)));
}

__device__ float rotate_lane(float value,int64_t column) {
    unsigned sign=unsigned(column)*0x9e3779b9u;
    sign^=sign>>16;
    if(sign&0x80000000u)value=-value;
    #pragma unroll
    for(int distance=1;distance<32;distance*=2) {
        const float other=__shfl_xor_sync(0xffffffffu,value,distance);
        value=(column&distance)?other-value:value+other;
    }
    return value*0.1767766952966368811f;
}
template<class Value>
__device__ float rotated(const Value* source,int64_t column,int64_t columns) {
    return rotate_lane(column<columns?float(source[column]):0.f,column);
}
template<class Value>
__device__ float rotated_gated(const Value* source,const Value* gate,Value* materialized,
        int64_t column,int64_t columns,bool silu) {
    float value=0.f;
    if(column<columns) {
        const float g=float(gate[column]);
        const Value activation=Value((silu?g:1.f)/(1.f+expf(-g)));
        const Value product=Value(__fmul_rn(float(activation),float(source[column])));
        materialized[column]=product;value=float(product);
    }
    return rotate_lane(value,column);
}

template<class Value,bool Cached=false,bool Gated=false>
__global__ void pack_rows(const Value* source,int64_t columns,int64_t stride,
                         int64_t padded,int8_t* codes,float* scales,
                         const Value* gate=nullptr,Value* materialized=nullptr,bool silu=false) {
    const auto row=int64_t(blockIdx.x);
    source+=row*stride;
    if constexpr(Gated){gate+=row*stride;materialized+=row*columns;}
    __shared__ float maxima[8];
    extern __shared__ float transformed[];
    float maximum=0.f;
    for(int64_t column=threadIdx.x;column<padded;column+=blockDim.x) {
        const float value=Gated?rotated_gated(source,gate,materialized,column,columns,silu):rotated(source,column,columns);
        if constexpr(Cached)transformed[column]=value;
        maximum=fmaxf(maximum,fabsf(value));
    }
    #pragma unroll
    for(int distance=16;distance;distance/=2)
        maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,distance));
    if(!(threadIdx.x&31))maxima[threadIdx.x/32]=maximum;
    __syncthreads();
    if(threadIdx.x<32) {
        maximum=threadIdx.x<8?maxima[threadIdx.x]:0.f;
        #pragma unroll
        for(int distance=16;distance;distance/=2)
            maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,distance));
        if(!threadIdx.x) {
            maxima[0]=maximum;
            scales[row]=maximum/127.f;
        }
    }
    __syncthreads();
    const float inverse=maxima[0]>0.f?127.f/maxima[0]:0.f;
    for(int64_t column=threadIdx.x;column<padded;column+=blockDim.x) {
        // The first gated pass already wrote the correctly rounded product.
        // Wide rows reuse it rather than repeating activation and gate reads.
        const float value=Cached?transformed[column]:rotated(Gated?materialized:source,column,columns);
        codes[row*padded+column]=int8_t(max(-127,min(127,__float2int_rn(value*inverse))));
    }
}

template<class Value,int Threads,int Slots>
__global__ __launch_bounds__(Threads,Threads==512?2:1)
void pack_gated_registers(const Value* source,const Value* gate,bool silu,int columns,int padded,
                          Value* materialized,int8_t* codes,float* scales) {
    const auto row=int64_t(blockIdx.x);
    source+=row*columns;gate+=row*columns;materialized+=row*columns;
    __shared__ float maxima[Threads/32];
    float values[Slots],maximum=0.f;
    #pragma unroll
    for(int i=0;i<Slots;++i) {
        const int column=threadIdx.x+i*Threads;
        float value=0.f;
        // Padding is a multiple of the transform width, so every active
        // transform retains all 32 lanes, including a partial source row.
        if(column<padded)value=rotated_gated(source,gate,materialized,column,columns,silu);
        values[i]=value;maximum=fmaxf(maximum,fabsf(value));
    }
    #pragma unroll
    for(int distance=16;distance;distance/=2)
        maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,distance));
    if(!(threadIdx.x&31))maxima[threadIdx.x/32]=maximum;
    __syncthreads();
    if(threadIdx.x<32) {
        maximum=threadIdx.x<Threads/32?maxima[threadIdx.x]:0.f;
        #pragma unroll
        for(int distance=16;distance;distance/=2)
            maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,distance));
        if(!threadIdx.x){maxima[0]=maximum;scales[row]=maximum/127.f;}
    }
    __syncthreads();
    const float inverse=maxima[0]>0.f?127.f/maxima[0]:0.f;
    #pragma unroll
    for(int i=0;i<Slots;++i) {
        const int column=threadIdx.x+i*Threads;
        if(column<padded)codes[row*padded+column]=int8_t(max(-127,min(127,__float2int_rn(values[i]*inverse))));
    }
}

template<class Value,int Threads>
bool launch_gated_registers(const Value* source,const Value* gate,bool silu,int64_t rows,
        int64_t columns,int64_t padded,Value* materialized,int8_t* codes,float* scales,cudaStream_t stream) {
    // The local array must stay in registers. Select its capacity from the
    // row width and reject compiler spills or insufficient block resources.
    const auto groups=(padded+Threads*8-1)/(Threads*8);
    if(groups>8)return false;
    auto function=pack_gated_registers<Value,Threads,8>;
    switch(groups) {
        case 1:break;
        case 2:function=pack_gated_registers<Value,Threads,16>;break;
        case 3:function=pack_gated_registers<Value,Threads,24>;break;
        case 4:function=pack_gated_registers<Value,Threads,32>;break;
        case 5:function=pack_gated_registers<Value,Threads,40>;break;
        case 6:function=pack_gated_registers<Value,Threads,48>;break;
        case 7:function=pack_gated_registers<Value,Threads,56>;break;
        default:function=pack_gated_registers<Value,Threads,64>;break;
    }
    cudaFuncAttributes attributes{};checked(cudaFuncGetAttributes(&attributes,function));
    if(attributes.localSizeBytes || attributes.maxThreadsPerBlock<Threads)return false;
    int resident=0;
    checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,function,Threads,0));
    if(resident<(Threads==512?2:1))return false;
    function<<<unsigned(rows),Threads,0,stream>>>(source,gate,silu,int(columns),int(padded),materialized,codes,scales);
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static thread_local bool reported=false;
        if(!reported) {
            reported=true;
            std::fprintf(stderr,"native_w8a8_gated_registers actual_match=1 threads=%d slots=%lld columns=%lld registers=%d local_bytes=%zu resident_blocks=%d\n",
                Threads,static_cast<long long>(groups*8),static_cast<long long>(columns),
                attributes.numRegs,attributes.localSizeBytes,resident);
        }
    }
    return true;
}

template<class Value,bool Vector>
__global__ void restore_rows(const int32_t* source,const float* input_scale,
                             const float* weight_scale,Value* output,
                             int64_t tokens,int64_t rows,int64_t stride,const half* residual) {
    constexpr int elements=Vector?4:1;
    const auto column=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)*elements;
    if(column>=rows)return;
    for(auto token=int64_t(blockIdx.y);token<tokens;token+=gridDim.y) {
        const auto offset=token*stride+column;
        const float scale=input_scale[token];
        if constexpr(Vector) {
            const auto values=*reinterpret_cast<const int4*>(source+token*rows+column);
            const auto weights=*reinterpret_cast<const float4*>(weight_scale+column);
            float4 value{float(values.x)*(scale*weights.x),float(values.y)*(scale*weights.y),
                float(values.z)*(scale*weights.z),float(values.w)*(scale*weights.w)};
            if(residual) {
                const auto a=__half22float2(*reinterpret_cast<const half2*>(residual+offset));
                const auto b=__half22float2(*reinterpret_cast<const half2*>(residual+offset+2));
                value.x+=a.x;value.y+=a.y;value.z+=b.x;value.w+=b.y;
            }
            if constexpr(std::is_same_v<Value,half>) {
                *reinterpret_cast<half2*>(output+offset)=__floats2half2_rn(value.x,value.y);
                *reinterpret_cast<half2*>(output+offset+2)=__floats2half2_rn(value.z,value.w);
            }else *reinterpret_cast<float4*>(output+offset)=value;
        }else {
            float value=float(source[token*rows+column])*(scale*weight_scale[column]);
            if(residual)value+=__half2float(residual[offset]);
            output[offset]=Value(value);
        }
    }
}

struct LtDevice {
    int device=0;
    cublasLtHandle_t handle=nullptr;
    ~LtDevice() {
        int previous=device;
        const auto status=cudaGetDevice(&previous);
        if(status==cudaSuccess && previous!=device)(void)cudaSetDevice(device);
        if(handle)(void)cublasLtDestroy(handle);
        if(status==cudaSuccess && previous!=device)(void)cudaSetDevice(previous);
    }
};
struct Implementation {
    std::shared_ptr<LtDevice> owner;
    cublasLtMatmulDesc_t operation=nullptr;
    cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace=0;
    ~Implementation() {
        const int device=owner->device;
        int previous=device;
        const auto status=cudaGetDevice(&previous);
        if(status==cudaSuccess && previous!=device)(void)cudaSetDevice(device);
        if(a)(void)cublasLtMatrixLayoutDestroy(a);
        if(b)(void)cublasLtMatrixLayoutDestroy(b);
        if(c)(void)cublasLtMatrixLayoutDestroy(c);
        if(operation)(void)cublasLtMatmulDescDestroy(operation);
        if(status==cudaSuccess && previous!=device)(void)cudaSetDevice(previous);
    }
};
}

int64_t padded_columns(int64_t columns) {
    // Each signed product is at most 127^2. The INT32 accumulation must fit
    // for every input, independently of the model's observed activation range.
    constexpr int64_t maximum=(INT_MAX/(127*127))/32*32;
    if(columns<1 || columns>maximum)throw std::invalid_argument("W8A8 columns exceed the exact INT32 dot range");
    return (columns+31)/32*32;
}

void pack(const void* input,bool half_input,int64_t rows,int64_t columns,int64_t stride,
          int8_t* codes,float* scales,cudaStream_t stream) {
    const auto padded=padded_columns(columns);
    if(rows<0 || rows>INT_MAX || stride<columns || stride>INT64_MAX/std::max(int64_t(1),rows) ||
            (rows && (!input || !codes || !scales)))throw std::invalid_argument("Invalid W8A8 source rows");
    if(!rows)return;
    const auto launch=[&]<class Value>() {
        int device=0,available=0,active=0;
        checked(cudaGetDevice(&device));
        checked(cudaDeviceGetAttribute(&available,cudaDevAttrMaxSharedMemoryPerBlock,device));
        const auto bytes=size_t(padded)*sizeof(float);
        // Retain rotated FP32 values only when shared memory still permits
        // multiple resident blocks. Wider rows keep the two-pass low-storage
        // kernel instead of sacrificing occupancy or requiring opt-in memory.
        if(bytes+8*sizeof(float)<=size_t(available))
            checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,pack_rows<Value,true>,256,bytes));
        if(active>=2)pack_rows<Value,true><<<unsigned(rows),256,bytes,stream>>>(static_cast<const Value*>(input),columns,stride,padded,codes,scales);
        else pack_rows<Value,false><<<unsigned(rows),256,0,stream>>>(static_cast<const Value*>(input),columns,stride,padded,codes,scales);
    };
    if(half_input)launch.template operator()<half>();else launch.template operator()<float>();
    checked(cudaGetLastError());
}

void pack_gated(const void* input,const void* gate,bool half_input,bool silu,
        int64_t rows,int64_t columns,void* materialized,int8_t* codes,float* scales,cudaStream_t stream) {
    const auto padded=padded_columns(columns);
    if(rows<0 || rows>INT_MAX || columns>INT64_MAX/std::max(int64_t(1),rows) ||
        (rows && (!input || !gate || !materialized || !codes || !scales)))
        throw std::invalid_argument("Invalid gated W8A8 rows");
    if(!rows)return;
    const auto launch=[&]<class Value>() {
        int device=0,available=0,active=0;
        checked(cudaGetDevice(&device));
        checked(cudaDeviceGetAttribute(&available,cudaDevAttrMaxSharedMemoryPerBlock,device));
        const auto bytes=size_t(padded)*sizeof(float);
        if(bytes+8*sizeof(float)<=size_t(available))
            checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,pack_rows<Value,true,true>,256,bytes));
        if(active>=2)pack_rows<Value,true,true><<<unsigned(rows),256,bytes,stream>>>(static_cast<const Value*>(input),
            columns,columns,padded,codes,scales,static_cast<const Value*>(gate),static_cast<Value*>(materialized),silu);
        else if(launch_gated_registers<Value,512>(static_cast<const Value*>(input),static_cast<const Value*>(gate),
                silu,rows,columns,padded,static_cast<Value*>(materialized),codes,scales,stream) ||
                launch_gated_registers<Value,1024>(static_cast<const Value*>(input),static_cast<const Value*>(gate),
                silu,rows,columns,padded,static_cast<Value*>(materialized),codes,scales,stream))return;
        else pack_rows<Value,false,true><<<unsigned(rows),256,0,stream>>>(static_cast<const Value*>(input),
            columns,columns,padded,codes,scales,static_cast<const Value*>(gate),static_cast<Value*>(materialized),silu);
    };
    if(half_input)launch.template operator()<half>();else launch.template operator()<float>();
    checked(cudaGetLastError());
}

Plan prepare(int64_t tokens,int64_t rows,int64_t columns,size_t limit,cudaStream_t stream) {
    const auto padded=padded_columns(columns);
    if(tokens<1 || tokens>INT_MAX || rows<1 || rows>INT_MAX)
        throw std::invalid_argument("Invalid W8A8 GEMM geometry");
    int device=0;
    checked(cudaGetDevice(&device));
    using Key=std::tuple<int,int64_t,int64_t,int64_t,size_t>;
    struct Entry {Key key;Plan plan;};
    static thread_local std::list<Entry> cache;
    const Key key{device,tokens,rows,padded,limit};
    for(auto i=cache.begin();i!=cache.end();++i)if(i->key==key) {
        auto result=i->plan;
        cache.splice(cache.begin(),cache,i);
        return result;
    }
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    checked(cudaStreamIsCapturing(stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone)return {};
    static thread_local std::unordered_map<int,std::weak_ptr<LtDevice>> owners;
    auto owner=owners[device].lock();
    if(!owner) {
        owner=std::make_shared<LtDevice>();owner->device=device;
        const auto created=cublasLtCreate(&owner->handle);
        if(created==CUBLAS_STATUS_NOT_SUPPORTED || created==CUBLAS_STATUS_ALLOC_FAILED)return {};
        checked(created);owners[device]=owner;
    }
    auto plan=std::make_shared<Implementation>();plan->owner=std::move(owner);
    checked(cublasLtMatmulDescCreate(&plan->operation,CUBLAS_COMPUTE_32I,CUDA_R_32I));
    const cublasOperation_t transpose=CUBLAS_OP_T;
    checked(cublasLtMatmulDescSetAttribute(plan->operation,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
    checked(cublasLtMatrixLayoutCreate(&plan->a,CUDA_R_8I,padded,rows,padded));
    checked(cublasLtMatrixLayoutCreate(&plan->b,CUDA_R_8I,padded,tokens,padded));
    checked(cublasLtMatrixLayoutCreate(&plan->c,CUDA_R_32I,rows,tokens,rows));
    struct Preference {
        cublasLtMatmulPreference_t value=nullptr;
        ~Preference(){if(value)(void)cublasLtMatmulPreferenceDestroy(value);}
    } preference;
    checked(cublasLtMatmulPreferenceCreate(&preference.value));
    checked(cublasLtMatmulPreferenceSetAttribute(preference.value,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit)));
    const unsigned alignment=16;
    for(auto attribute:{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
        checked(cublasLtMatmulPreferenceSetAttribute(preference.value,attribute,&alignment,sizeof(alignment)));
    cublasLtMatmulHeuristicResult_t choices[16]{};
    int count=0;
    const auto status=cublasLtMatmulAlgoGetHeuristic(plan->owner->handle,plan->operation,plan->a,plan->b,plan->c,plan->c,
                                                   preference.value,16,choices,&count);
    if(status==CUBLAS_STATUS_NOT_SUPPORTED || status==CUBLAS_STATUS_ALLOC_FAILED)return {};
    checked(status);
    for(int i=0;i<count;++i) {
        if(choices[i].state!=CUBLAS_STATUS_SUCCESS || choices[i].workspaceSize>limit)continue;
        uint64_t flags=0;
        size_t written=0;
        checked(cublasLtMatmulAlgoCapGetAttribute(&choices[i].algo,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                                                 &flags,sizeof(flags),&written));
        if(!(flags&CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA))continue;
        plan->algorithm=choices[i].algo;
        plan->workspace=choices[i].workspaceSize;
        const auto bytes=plan->workspace;
        Plan result{std::move(plan),bytes};
        cache.push_front({key,result});
        if(cache.size()>64)cache.pop_back();
        return result;
    }
    return {};
}

void multiply(const Plan& prepared,const int8_t* input,const int8_t* weight,int32_t* accumulator,
              void* workspace,size_t bytes,cudaStream_t stream) {
    if(!prepared || !input || !weight || !accumulator || bytes<prepared.workspace_bytes ||
            ((reinterpret_cast<uintptr_t>(input)|reinterpret_cast<uintptr_t>(weight)|reinterpret_cast<uintptr_t>(accumulator))&15u) ||
            (prepared.workspace_bytes && (!workspace || (reinterpret_cast<uintptr_t>(workspace)&255u))))
        throw std::invalid_argument("Invalid W8A8 operands or workspace");
    const auto plan=std::static_pointer_cast<Implementation>(prepared.implementation);
    int device=0;
    checked(cudaGetDevice(&device));
    if(device!=plan->owner->device)throw std::invalid_argument("W8A8 plan belongs to another CUDA device");
    const int32_t one=1,zero=0;
    checked(cublasLtMatmul(plan->owner->handle,plan->operation,&one,weight,plan->a,input,plan->b,&zero,
                          accumulator,plan->c,accumulator,plan->c,&plan->algorithm,workspace,bytes,stream));
}

void restore(const int32_t* accumulator,const float* input_scales,const float* weight_scales,
             void* output,bool half_output,int64_t tokens,int64_t rows,int64_t stride,cudaStream_t stream,const void* residual) {
    if(tokens<0 || tokens>INT_MAX || rows<1 || rows>INT_MAX || stride<rows ||
            stride>INT64_MAX/std::max(int64_t(1),tokens) ||
            (tokens && (!accumulator || !input_scales || !weight_scales || !output)))
        throw std::invalid_argument("Invalid W8A8 output rows");
    if(!tokens)return;
    const bool vector=rows%4==0 && stride%4==0 &&
        !(reinterpret_cast<std::uintptr_t>(accumulator)&15) && !(reinterpret_cast<std::uintptr_t>(weight_scales)&15) &&
        !(reinterpret_cast<std::uintptr_t>(output)&(half_output?3:15)) &&
        (!residual || !(reinterpret_cast<std::uintptr_t>(residual)&3));
    const dim3 grid(unsigned((rows+(vector?1024:256)-1)/(vector?1024:256)),unsigned(std::min(tokens,int64_t(65535))));
    const auto launch=[&]<class Value>() {
        if(vector)restore_rows<Value,true><<<grid,256,0,stream>>>(accumulator,input_scales,weight_scales,
            static_cast<Value*>(output),tokens,rows,stride,static_cast<const half*>(residual));
        else restore_rows<Value,false><<<grid,256,0,stream>>>(accumulator,input_scales,weight_scales,
            static_cast<Value*>(output),tokens,rows,stride,static_cast<const half*>(residual));
    };
    if(half_output)launch.template operator()<half>();else launch.template operator()<float>();
    checked(cudaGetLastError());
}
}
