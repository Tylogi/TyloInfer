// Rotary Position Embedding, rotate-half (HF/GPT-J style, Qwen convention).
// Supports full RoPE, partial RoPE, per-batch positions, and MRoPE sections.
// x [..., T, D], pos [T], [B,T], or [A,T]. rotary_dim <= D.
// freq_j = base^(-2j/rotary_dim). One block per (m, t) row, threads cover j in [0, rotary_dim/2):
//   out[t,j]      = x0*cos - x1*sin
//   out[t,j+half] = x1*cos + x0*sin

#include <cuda_runtime.h>
#include "raw.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/native/env.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <algorithm>
#include <utility>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>
#include "../kernel_dispatch_trace.h"

constexpr int ROPE_BD = 256;

using RotaryProjectionLayout = tyloi::cuda::rope_raw::Layout;
using RotaryNormalizedProjection = tyloi::cuda::rope_raw::Projection;
using RotaryNormalizedGroup = tyloi::cuda::rope_raw::Group;

__device__ int64_t rotary_position(const void* positions, bool wide, int64_t index) {
    return wide ? static_cast<const int64_t*>(positions)[index]
                : static_cast<const int32_t*>(positions)[index];
}

__device__ float2 normalized_rotary_angles(const void* positions,bool wide_positions,
    const float* frequencies,const int32_t* selected_axes,int column,int token,int64_t batch,
    int tokens,int axes,int batches,int maximum) {
    int axis=selected_axes[column];
    if(axis>=axes)axis=0;
    const int64_t index=(int64_t(axis)*batches+(batches==1?0:batch))*tokens+token;
    const int64_t raw_position=rotary_position(positions,wide_positions,index);
    int position=wide_positions?__double2int_rz(double(raw_position)):int(raw_position);
    position=max(0,min(maximum-1,position));
    const float angle=__fmul_rn(float(position),frequencies[column]);
    double sine,cosine;
    ::sincos(double(angle),&sine,&cosine);
    return make_float2(float(cosine),float(sine));
}

template<bool Compact,class Value,bool Affine=false>
__global__ void qsa_pool_rotary_cache_kernel(const Value* index,const float* weight,const void* positions,
        bool wide_positions,const int64_t* cache,const float* frequencies,const int32_t* selected_axes,Value* output,
        int tokens,int width,int pool,int maximum,int rotary_maximum,int pairs,int axes,float eps,
        int64_t row_stride,int64_t column_stride,double attention_factor) {
    const int token=blockIdx.x,column=threadIdx.x;
    const int64_t block=Compact?int64_t(token):cache[token]/pool,start=block*pool;
    // A verify batch can touch one block repeatedly. Only its last query writes
    // the pooled entry; all queries observe the same completed raw cache.
    if constexpr(!Compact)if(token+1<tokens && cache[token+1]/pool==block)return;
    __shared__ float means[256],partial[256];
    float mean=0;
    if(column<width) {
        float sum=0;
        for(int i=0;i<pool;++i) {
            const int64_t row=min(max(int64_t(0),start+i),int64_t(maximum)-1);
            float value;
            if constexpr(Affine) {
                const auto* record=index+row*row_stride;
                value=float(int(record[column])-int(record[width+1]))*__half2float(__ushort_as_half(uint16_t(record[width])));
            }else value=float(index[row*row_stride+column*column_stride]);
            sum=__fadd_rn(sum,value);
        }
        if constexpr(Affine)mean=__fdiv_rn(sum,float(pool));
        else mean=float(Value(__fdiv_rn(sum,float(pool))));
    }
    means[column]=mean;partial[column]=__fadd_rn(0.f,__fmul_rn(mean,mean));
    __syncthreads();
    // Retain the normalized rotary kernel's reduction tree and intermediate
    // FP16 rounding, including the rounding before rotate-half.
    for(int stride=blockDim.x/2;stride>0;stride/=2) {
        if(column<stride)partial[column]=__fadd_rn(partial[column],partial[column+stride]);
        __syncthreads();
    }
    const float inverse=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(__fdiv_rn(partial[0],float(width)),eps)));
    using Normal=std::conditional_t<Affine,float,Value>;
    const auto normalized=[&](int col){return Normal(__fmul_rn(__fmul_rn(means[col],inverse),__fadd_rn(weight[col],1.f)));};
    if constexpr(Affine) {
        float value=0;
        if(column<width) {
            if(column>=2*pairs)value=normalized(column);
            else {
                const int pair=column%pairs;int axis=selected_axes[pair];if(axis>=axes)axis=0;
                int64_t raw=rotary_position(positions,wide_positions,int64_t(axis)*tokens+token);
                if constexpr(!Compact)raw+=start-cache[token];
                const int pos=max(0,min(rotary_maximum-1,int(raw)));
                const float angle=__fmul_rn(float(pos),frequencies[pair]);
                double sine,cosine;::sincos(double(angle),&sine,&cosine);
                const float cs=float(double(float(cosine))*attention_factor),sn=float(double(float(sine))*attention_factor);
                const auto first=float(normalized(pair)),second=float(normalized(pair+pairs));
                value=column<pairs?__fsub_rn(__fmul_rn(first,cs),__fmul_rn(second,sn)):
                    __fadd_rn(__fmul_rn(second,cs),__fmul_rn(first,sn));
            }
        }
        // Finish every reader of the normalized input before reusing its array.
        __syncthreads();means[column]=value;partial[column]=fminf(0.f,value);__syncthreads();
        for(int stride=blockDim.x/2;stride;stride/=2){if(column<stride)partial[column]=fminf(partial[column],partial[column+stride]);__syncthreads();}
        const float lo=partial[0];__syncthreads();partial[column]=fmaxf(0.f,value);__syncthreads();
        for(int stride=blockDim.x/2;stride;stride/=2){if(column<stride)partial[column]=fmaxf(partial[column],partial[column+stride]);__syncthreads();}
        const float step=fmaxf(__fmul_rn(__fsub_rn(partial[0],lo),1.f/65535.f),0x1p-24f);
        auto scale=__float2half_rn(step);if(__half2float(scale)<step)scale=__ushort_as_half(__half_as_ushort(scale)+1);
        const float scale_value=__half2float(scale);
        const int zp=max(-32768,min(32767,-32768-int(nearbyintf(lo/scale_value))));
        auto* record=output+block*(width+2);
        if(!column){record[width]=int16_t(__half_as_ushort(scale));record[width+1]=int16_t(zp);}
        if(column<width)record[column]=int16_t(max(-32768,min(32767,int(nearbyintf(means[column]/scale_value))+zp)));
        return;
    }
    if(column>=width-pairs)return;
    if(column>=pairs){output[block*width+column+pairs]=normalized(column+pairs);return;}
    int axis=selected_axes[column];if(axis>=axes)axis=0;
    int64_t raw=rotary_position(positions,wide_positions,int64_t(axis)*tokens+token);
    if constexpr(!Compact)raw+=start-cache[token];
    int position;
    position=wide_positions?__double2int_rz(double(raw)):int(raw);
    position=max(0,min(rotary_maximum-1,position));
    const float angle=__fmul_rn(float(position),frequencies[column]);
    double wide_sine,wide_cosine;
    ::sincos(double(angle),&wide_sine,&wide_cosine);
    const float cosine=float(double(float(wide_cosine))*attention_factor),sine=float(double(float(wide_sine))*attention_factor);
    const float first=float(normalized(column)),second=float(normalized(column+pairs));
    output[block*width+column]=Value(__fsub_rn(__fmul_rn(first,cosine),__fmul_rn(second,sine)));
    output[block*width+column+pairs]=Value(__fadd_rn(__fmul_rn(second,cosine),__fmul_rn(first,sine)));
}


template<class Value,bool FusedReduction>
__device__ __forceinline__ void rotary_normalized_row(
    const Value* input, const float* weight, const void* positions, bool wide_positions,
    const float* frequencies, const int32_t* selected_axes, Value* output,
    int heads, int tokens, int width, int pairs, int axes, int batches, int maximum,
    float eps, RotaryProjectionLayout layout, __half* key_cache,
    const void* cache_positions, bool wide_cache_positions, int64_t capacity,
    const void* projected_value, bool float_value, RotaryProjectionLayout value_layout,
    __half* value_cache, int64_t row, int active_threads,const float2* rotations,double attention_factor) {
    const int64_t batch = row / (int64_t(heads) * tokens);
    const int head = int((row / tokens) % heads), token = int(row % tokens);
    const int64_t source = batch * layout.batch + token * layout.token + head * layout.head;
    float sum = 0;
    for (int column = threadIdx.x; column < width; column += active_threads) {
        const float x = float(input[source + int64_t(column) * layout.column]);
        sum = __fadd_rn(sum, __fmul_rn(x, x));
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    float inverse;
    if constexpr(FusedReduction) {
        if(threadIdx.x<32) {
            const int lane=threadIdx.x;
            float merged;
            // Preserve the original descending pairwise tree across warps,
            // then finish within one warp. Compute the inverse only once.
            if(active_threads==256) {
                const float a=__fadd_rn(partial[lane],partial[lane+128]);
                const float b=__fadd_rn(partial[lane+32],partial[lane+160]);
                const float c=__fadd_rn(partial[lane+64],partial[lane+192]);
                const float d=__fadd_rn(partial[lane+96],partial[lane+224]);
                merged=__fadd_rn(__fadd_rn(a,c),__fadd_rn(b,d));
            }else if(active_threads==128) {
                merged=__fadd_rn(__fadd_rn(partial[lane],partial[lane+64]),
                    __fadd_rn(partial[lane+32],partial[lane+96]));
            }else if(active_threads==64)merged=__fadd_rn(partial[lane],partial[lane+32]);
            else merged=partial[lane];
            #pragma unroll
            for(int offset=16;offset;offset>>=1)
                merged=__fadd_rn(merged,__shfl_down_sync(0xffffffffu,merged,offset));
            if(!lane) {
                const float mean=__fdiv_rn(merged,float(width));
                partial[0]=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(mean,eps)));
            }
        }
        __syncthreads();
        inverse=partial[0];
    }else {
        for (int stride = active_threads / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride)
                partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]);
            __syncthreads();
        }
        const float mean = __fdiv_rn(partial[0], float(width));
        inverse = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(mean, eps)));
    }
    const auto normalized = [&](int column) {
        const float scaled = __fmul_rn(float(input[source + int64_t(column) * layout.column]), inverse);
        return Value(__fmul_rn(scaled, __fadd_rn(weight[column], 1.f)));
    };
    int64_t cache_base = -1;
    if (key_cache) {
        const auto slot = rotary_position(cache_positions, wide_cache_positions, token);
        if (slot >= 0 && slot < capacity) cache_base = ((batch * heads + head) * capacity + slot) * width;
    }
    const auto store = [&](int column, Value value) {
        if (output) output[row * width + column] = value;
        if (cache_base >= 0) key_cache[cache_base + column] = __float2half_rn(float(value));
    };
    for (int column = threadIdx.x; column < width - pairs; column += active_threads) {
        if (column >= pairs) {
            store(column + pairs, normalized(column + pairs));
            continue;
        }
        const auto rotation=rotations?rotations[((batches==1?0:batch)*tokens+token)*pairs+column]:
            normalized_rotary_angles(positions,wide_positions,frequencies,selected_axes,column,token,batch,
                tokens,axes,batches,maximum);
        const float cosine=float(double(rotation.x)*attention_factor),sine=float(double(rotation.y)*attention_factor);
        const float first = float(normalized(column)), second = float(normalized(column + pairs));
        store(column, Value(__fsub_rn(__fmul_rn(first, cosine), __fmul_rn(second, sine))));
        store(column + pairs, Value(__fadd_rn(__fmul_rn(second, cosine), __fmul_rn(first, sine))));
    }
    if (value_cache && cache_base >= 0) {
        const int64_t value_base = batch * value_layout.batch + token * value_layout.token + head * value_layout.head;
        for (int column = threadIdx.x; column < width; column += active_threads) {
            const int64_t at = value_base + int64_t(column) * value_layout.column;
            const float value = float_value ? static_cast<const float*>(projected_value)[at]
                : float(static_cast<const __half*>(projected_value)[at]);
            value_cache[cache_base + column] = __float2half_rn(value);
        }
    }
}

template<class Value,bool FusedReduction>
__global__ void rotary_normalized_cached_kernel(
    const Value* input, const float* weight, const void* positions, bool wide_positions,
    const float* frequencies, const int32_t* selected_axes, Value* output,
    int heads, int tokens, int width, int pairs, int axes, int batches, int maximum,
    float eps, RotaryProjectionLayout layout, __half* key_cache,
    const void* cache_positions, bool wide_cache_positions, int64_t capacity,
    const void* projected_value, bool float_value, RotaryProjectionLayout value_layout,
    __half* value_cache,const float2* rotations,double attention_factor) {
    rotary_normalized_row<Value,FusedReduction>(input,weight,positions,wide_positions,frequencies,selected_axes,output,
        heads,tokens,width,pairs,axes,batches,maximum,eps,layout,key_cache,
        cache_positions,wide_cache_positions,capacity,projected_value,float_value,value_layout,value_cache,blockIdx.x,blockDim.x,rotations,attention_factor);
}


__global__ void rotary_shared_angles_kernel(RotaryNormalizedProjection p,float2* output,int64_t count) {
    for(int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;index<count;
            index+=int64_t(gridDim.x)*blockDim.x) {
        const int column=int(index%p.pairs),token=int(index/p.pairs%p.tokens);
        const int64_t batch=index/p.pairs/p.tokens;
        output[index]=normalized_rotary_angles(p.positions,p.wide_positions,p.frequencies,p.selected_axes,
            column,token,batch,p.tokens,p.axes,p.batches,p.maximum);
    }
}

bool fused_rotary_reduction_enabled() {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_ROTARY_FUSED_REDUCTION");
    return !option || option[0]!='0';
}

bool shared_rotary_angles_enabled(const void* function,int threads,int64_t rows) {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_ROTARY_SHARED_ANGLES");
    if(option && option[0]=='0')return false;
    if(option && option[0]=='1')return true;
    if(!function)return false;
    // Small windows execute duplicated angles in parallel. Amortize a
    // separate angle launch only beyond two fully resident waves, using the
    // actual kernel's register/shared-memory occupancy on the current device.
    static std::mutex mutex;
    static std::map<std::tuple<int,int,std::uintptr_t>,int64_t> capacities;
    int device=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
    const auto key=std::make_tuple(device,threads,reinterpret_cast<std::uintptr_t>(function));
    std::lock_guard<std::mutex> lock(mutex);
    auto found=capacities.find(key);
    if(found==capacities.end()) {
        int blocks=0,sms=0;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,function,threads,0));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
        found=capacities.emplace(key,int64_t(blocks)*sms).first;
    }
    return rows>2*found->second;
}


template<class Value,bool FusedReduction>
__global__ void rotary_normalized_grouped_kernel(RotaryNormalizedGroup group) {
    int64_t row=blockIdx.x;int selected=0;
    while(selected+1<group.count && row>=group.entries[selected].rows) {
        row-=group.entries[selected].rows;++selected;
    }
    const auto& p=group.entries[selected];
    rotary_normalized_row<Value,FusedReduction>(static_cast<const Value*>(p.input),p.weight,p.positions,p.wide_positions,
        p.frequencies,p.selected_axes,static_cast<Value*>(p.output),p.heads,p.tokens,p.width,p.pairs,
        p.axes,p.batches,p.maximum,p.eps,p.layout,static_cast<__half*>(p.key_cache),p.cache_positions,p.wide_cache_positions,
        p.capacity,p.projected_value,p.float_value,p.value_layout,static_cast<__half*>(p.value_cache),row,p.threads,p.rotations,p.attention_factor);
}
template<class Value>
void launch_rotary_normalized(const RotaryNormalizedProjection& p,cudaStream_t stream) {
    const auto launch=[&](auto fused) {
        constexpr bool Fused=decltype(fused)::value;
        const auto function=rotary_normalized_cached_kernel<Value,Fused>;
        function<<<unsigned(p.rows),p.threads,0,stream>>>(
        static_cast<const Value*>(p.input),p.weight,p.positions,p.wide_positions,
        p.frequencies,p.selected_axes,static_cast<Value*>(p.output),p.heads,p.tokens,p.width,p.pairs,
        p.axes,p.batches,p.maximum,p.eps,p.layout,static_cast<__half*>(p.key_cache),p.cache_positions,p.wide_cache_positions,
        p.capacity,p.projected_value,p.float_value,p.value_layout,static_cast<__half*>(p.value_cache),p.rotations,p.attention_factor);
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());
        if(tyloi::cuda::dispatch_trace::enabled())tyloi::cuda::dispatch_trace::record(stream,
            "rotary_normalized_cached",reinterpret_cast<const void*>(function),
            "fused_reduction="+std::to_string(Fused)+" width="+std::to_string(p.width)+" attention_factor="+std::to_string(p.attention_factor));
    };
    if(fused_rotary_reduction_enabled())launch(std::true_type{});else launch(std::false_type{});
}

template<class Value>
void launch_rotary_normalized_grouped(const RotaryNormalizedGroup& group,unsigned rows,int threads,cudaStream_t stream) {
    const auto launch=[&](auto fused) {
        constexpr bool Fused=decltype(fused)::value;
        const auto function=rotary_normalized_grouped_kernel<Value,Fused>;
        function<<<rows,threads,0,stream>>>(group);
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());
        if(tyloi::cuda::dispatch_trace::enabled())tyloi::cuda::dispatch_trace::record(stream,
            "rotary_normalized_grouped",reinterpret_cast<const void*>(function),
            "fused_reduction="+std::to_string(Fused)+" rows="+std::to_string(rows)+" attention_factor="+std::to_string(group.entries[0].attention_factor));
    };
    if(fused_rotary_reduction_enabled())launch(std::true_type{});else launch(std::false_type{});
}

template<class Value>
const void* rotary_normalized_function(bool grouped) {
    if(fused_rotary_reduction_enabled())return grouped?
        reinterpret_cast<const void*>(rotary_normalized_grouped_kernel<Value,true>):
        reinterpret_cast<const void*>(rotary_normalized_cached_kernel<Value,true>);
    return grouped?reinterpret_cast<const void*>(rotary_normalized_grouped_kernel<Value,false>):
        reinterpret_cast<const void*>(rotary_normalized_cached_kernel<Value,false>);
}


template<typename Value>
__global__ void rotary_embedding_cached_kernel(
    const Value* __restrict__ input,const int32_t* __restrict__ positions,
    const float* __restrict__ frequencies,const int32_t* __restrict__ selected_axes,
    Value* __restrict__ output,int64_t rows,int heads,int tokens,int width,int pairs,
    int axes,int batches,int maximum,double attention_factor) {
    const int columns=width-pairs;
    const int64_t total=rows*columns;
    for(int64_t linear=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
        linear<total;linear+=int64_t(blockDim.x)*gridDim.x) {
        const int64_t row=linear/columns;
        const int column=int(linear-row*columns);
        const size_t offset=size_t(row)*width;
        if(column>=pairs) {
            output[offset+column+pairs]=input[offset+column+pairs];
            continue;
        }
        int axis=selected_axes[column];if(axis>=axes)axis=0;
        const int batch=batches==1?0:int(row/(int64_t(heads)*tokens));
        const int token=int(row%tokens);
        int position=positions[(size_t(axis)*batches+batch)*tokens+token];
        position=max(0,min(maximum-1,position));
        const float angle=__fmul_rn(float(position),frequencies[column]);
        // Match native tensor sin/cos promotion and every stored FP32 product.
        double wide_sine,wide_cosine;
        ::sincos(double(angle),&wide_sine,&wide_cosine);
        const float cosine=float(double(float(wide_cosine))*attention_factor),sine=float(double(float(wide_sine))*attention_factor);
        const float first=float(input[offset+column]),second=float(input[offset+column+pairs]);
        const float a=__fsub_rn(__fmul_rn(first,cosine),__fmul_rn(second,sine));
        const float b=__fadd_rn(__fmul_rn(second,cosine),__fmul_rn(first,sine));
        output[offset+column]=Value(a);output[offset+column+pairs]=Value(b);
    }
}


__device__ int rope_axis_for_pair(int j, int s0, int s1, int s2)
{
    if (s0 <= 0 && s1 <= 0 && s2 <= 0) {
        return 0;
    }
    if (j < s0) {
        return 0;
    }
    if (j < s0 + s1) {
        return 1;
    }
    return 2;
}

__global__ void rope_kernel(const float* __restrict__ x, const float* __restrict__ pos,
                            float* __restrict__ out, int MT, int T, int D, int rotary_dim,
                            int pos_axes, int pos_batches,
                            int s0, int s1, int s2, float base)
{
    int mt = blockIdx.x;
    if (mt >= MT) {
        return;
    }
    int m = mt / T;
    int t = mt % T;
    int half = rotary_dim / 2;
    size_t base0 = ((size_t)m * T + t) * D;
    int tid = threadIdx.x;

    for (int i = tid; i < D; i += ROPE_BD) {
        out[base0 + i] = x[base0 + i];
    }
    __syncthreads();

    for (int j = tid; j < half; j += ROPE_BD) {
        float p;
        if (pos_batches > 0) {
            const int leading_rows = MT / T;
            const int rows_per_batch = leading_rows / pos_batches;
            const int batch = m / rows_per_batch;
            p = pos[(size_t)batch * T + t];
        } else {
            int axis = rope_axis_for_pair(j, s0, s1, s2);
            if (axis >= pos_axes) {
                axis = 0;
            }
            p = pos_axes == 1 ? pos[t] : pos[(size_t)axis * T + t];
        }
        float freq = powf(base, -2.0f * (float)j / (float)rotary_dim);
        float ang = p * freq;
        float cs = cosf(ang);
        float sn = sinf(ang);
        float x0 = x[base0 + j];
        float x1 = x[base0 + j + half];
        out[base0 + j] = x0 * cs - x1 * sn;
        out[base0 + j + half] = x1 * cs + x0 * sn;
    }
}

__global__ void rope_table_kernel(const float* __restrict__ x, const int64_t* __restrict__ pos,
                                  const float* __restrict__ cos, const float* __restrict__ sin,
                                  float* __restrict__ out, int MT, int T, int D, int rotary_dim,
                                  int table_len, int pos_axes, int pos_batches,
                                  int s0, int s1, int s2)
{
    int mt = blockIdx.x;
    if (mt >= MT) {
        return;
    }
    int m = mt / T;
    int t = mt % T;
    int half = rotary_dim / 2;
    size_t base0 = ((size_t)m * T + t) * D;
    int tid = threadIdx.x;

    for (int i = rotary_dim + tid; i < D; i += ROPE_BD) {
        out[base0 + i] = x[base0 + i];
    }

    for (int j = tid; j < half; j += ROPE_BD) {
        int64_t p;
        if (pos_batches > 0) {
            const int leading_rows = MT / T;
            const int rows_per_batch = leading_rows / pos_batches;
            const int batch = m / rows_per_batch;
            p = pos[(size_t)batch * T + t];
        } else {
            int axis = rope_axis_for_pair(j, s0, s1, s2);
            if (axis >= pos_axes) {
                axis = 0;
            }
            p = pos_axes == 1 ? pos[t] : pos[(size_t)axis * T + t];
        }
        if (p < 0) {
            p = 0;
        }
        if (p >= table_len) {
            p = table_len - 1;
        }
        float cs = cos[(size_t)p * half + j];
        float sn = sin[(size_t)p * half + j];
        float x0 = x[base0 + j];
        float x1 = x[base0 + j + half];
        out[base0 + j] = x0 * cs - x1 * sn;
        out[base0 + j + half] = x1 * cs + x0 * sn;
    }
}

__global__ void rope_table_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ pos,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    __nv_bfloat16* __restrict__ out,
    int rows, int T, int D, int rotary_dim, int table_len)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int t = row % T;
    const int half = rotary_dim / 2;
    const size_t offset = (size_t)row * D;
    int64_t p = pos[t];
    p = p < 0 ? 0 : (p >= table_len ? table_len - 1 : p);

    for (int i = rotary_dim + threadIdx.x; i < D; i += ROPE_BD) {
        out[offset + i] = x[offset + i];
    }

    for (int j = threadIdx.x; j < half; j += ROPE_BD) {
        const __nv_bfloat16 cs = __float2bfloat16_rn(
            cos[(size_t)p * half + j]);
        const __nv_bfloat16 sn = __float2bfloat16_rn(
            sin[(size_t)p * half + j]);
        const __nv_bfloat16 x0 = x[offset + j];
        const __nv_bfloat16 x1 = x[offset + j + half];
        const __nv_bfloat16 x0c = __float2bfloat16_rn(
            __bfloat162float(x0) * __bfloat162float(cs));
        const __nv_bfloat16 x1s = __float2bfloat16_rn(
            __bfloat162float(x1) * __bfloat162float(sn));
        const __nv_bfloat16 x1c = __float2bfloat16_rn(
            __bfloat162float(x1) * __bfloat162float(cs));
        const __nv_bfloat16 x0s = __float2bfloat16_rn(
            __bfloat162float(x0) * __bfloat162float(sn));
        out[offset + j] = __float2bfloat16_rn(
            __bfloat162float(x0c) - __bfloat162float(x1s));
        out[offset + j + half] = __float2bfloat16_rn(
            __bfloat162float(x1c) + __bfloat162float(x0s));
    }
}

__global__ void minicpm_bf16_rope_cache_write_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int64_t* __restrict__ rope_pos,
    const int64_t* __restrict__ write_pos,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int Hq, int Hk, int D, int half, int table_len, int max_seq)
{
    const int row = blockIdx.x;
    const int hq = row % Hq;
    const int b = row / Hq;
    const int tid = threadIdx.x;
    int64_t p = rope_pos[0];
    p = p < 0 ? 0 : (p >= table_len ? table_len - 1 : p);
    const int64_t wp = write_pos[0];

    const size_t q_offset = (size_t)row * D;
    if (tid < half) {
        const __nv_bfloat16 cs = __float2bfloat16_rn(
            cos[(size_t)p * half + tid]);
        const __nv_bfloat16 sn = __float2bfloat16_rn(
            sin[(size_t)p * half + tid]);
        const __nv_bfloat16 x0 = q[q_offset + tid];
        const __nv_bfloat16 x1 = q[q_offset + tid + half];
        const __nv_bfloat16 x0c = __float2bfloat16_rn(
            __bfloat162float(x0) * __bfloat162float(cs));
        const __nv_bfloat16 x1s = __float2bfloat16_rn(
            __bfloat162float(x1) * __bfloat162float(sn));
        const __nv_bfloat16 x1c = __float2bfloat16_rn(
            __bfloat162float(x1) * __bfloat162float(cs));
        const __nv_bfloat16 x0s = __float2bfloat16_rn(
            __bfloat162float(x0) * __bfloat162float(sn));
        q_out[q_offset + tid] = __float2bfloat16_rn(
            __bfloat162float(x0c) - __bfloat162float(x1s));
        q_out[q_offset + tid + half] = __float2bfloat16_rn(
            __bfloat162float(x1c) + __bfloat162float(x0s));
    }

    if (hq < Hk && wp >= 0 && wp < max_seq) {
        const size_t src = ((size_t)b * Hk + hq) * D;
        const size_t dst = (((size_t)b * Hk + hq) * max_seq + wp) * D;
        if (tid < half) {
            const __nv_bfloat16 cs = __float2bfloat16_rn(
                cos[(size_t)p * half + tid]);
            const __nv_bfloat16 sn = __float2bfloat16_rn(
                sin[(size_t)p * half + tid]);
            const __nv_bfloat16 x0 = k[src + tid];
            const __nv_bfloat16 x1 = k[src + tid + half];
            const __nv_bfloat16 x0c = __float2bfloat16_rn(
                __bfloat162float(x0) * __bfloat162float(cs));
            const __nv_bfloat16 x1s = __float2bfloat16_rn(
                __bfloat162float(x1) * __bfloat162float(sn));
            const __nv_bfloat16 x1c = __float2bfloat16_rn(
                __bfloat162float(x1) * __bfloat162float(cs));
            const __nv_bfloat16 x0s = __float2bfloat16_rn(
                __bfloat162float(x0) * __bfloat162float(sn));
            k_cache[dst + tid] = __float2bfloat16_rn(
                __bfloat162float(x0c) - __bfloat162float(x1s));
            k_cache[dst + tid + half] = __float2bfloat16_rn(
                __bfloat162float(x1c) + __bfloat162float(x0s));
        }
        if (tid < D) {
            v_cache[dst + tid] = v[src + tid];
        }
    }
}



namespace tyloi::cuda::rope_raw {

void pool(const Pool& p,cudaStream_t stream) {
    const int threads=p.width<=32?32:p.width<=64?64:p.width<=128?128:256;
    const auto launch=[&](auto* source,auto* target) {
        using Value=std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
        constexpr bool Affine=std::is_same_v<Value,int16_t>;
        if(p.compact) {
            qsa_pool_rotary_cache_kernel<true,Value,Affine><<<unsigned(p.tokens),threads,0,stream>>>(
                source,p.weight,p.positions,p.wide_positions,nullptr,p.frequencies,p.selected_axes,target,
                p.tokens,p.width,p.pool,p.maximum,p.rotary_maximum,p.pairs,p.axes,p.eps,
                p.row_stride,p.column_stride,p.attention_factor);
        } else {
            qsa_pool_rotary_cache_kernel<false,Value,Affine><<<unsigned(p.tokens),threads,0,stream>>>(
                source,p.weight,p.positions,p.wide_positions,p.cache_positions,p.frequencies,p.selected_axes,target,
                p.tokens,p.width,p.pool,p.maximum,p.rotary_maximum,p.pairs,p.axes,p.eps,
                p.row_stride,p.column_stride,p.attention_factor);
        }
    };
    if(p.type==PoolType::f16)launch(static_cast<const __half*>(p.input),static_cast<__half*>(p.output));
    else if(p.type==PoolType::affine)launch(static_cast<const int16_t*>(p.input),static_cast<int16_t*>(p.output));
    else launch(static_cast<const float*>(p.input),static_cast<float*>(p.output));
}

bool share_angles(NumericType type,bool grouped,bool uniform,int threads,int64_t rows) {
    const auto function=!uniform?nullptr:type==NumericType::f32?
        rotary_normalized_function<float>(grouped):rotary_normalized_function<__half>(grouped);
    return shared_rotary_angles_enabled(function,threads,rows);
}

void angles(const Projection& p,float2* output,int64_t count,cudaStream_t stream) {
    const auto blocks=unsigned(std::min<int64_t>((count+31)/32,65535));
    rotary_shared_angles_kernel<<<blocks,32,0,stream>>>(p,output,count);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());
    if(tyloi::cuda::dispatch_trace::enabled())tyloi::cuda::dispatch_trace::record(stream,
        "rotary_shared_angles",reinterpret_cast<const void*>(rotary_shared_angles_kernel),
        "tokens="+std::to_string(p.tokens)+" batches="+std::to_string(p.batches)+" pairs="+std::to_string(p.pairs));
}

void normalized(const Projection& p,NumericType type,cudaStream_t stream) {
    if(type==NumericType::f32)launch_rotary_normalized<float>(p,stream);
    else launch_rotary_normalized<__half>(p,stream);
}

void normalized_group(const Group& group,int64_t rows,int threads,NumericType type,cudaStream_t stream) {
    if(type==NumericType::f32)launch_rotary_normalized_grouped<float>(group,unsigned(rows),threads,stream);
    else launch_rotary_normalized_grouped<__half>(group,unsigned(rows),threads,stream);
}

void cached(const void* input,const int32_t* positions,const float* frequencies,
        const int32_t* axes,void* output,int64_t rows,int heads,int tokens,int width,int pairs,
        int position_axes,int batches,int maximum,double attention_factor,NumericType type,cudaStream_t stream) {
    const auto blocks=static_cast<unsigned int>((rows*(width-pairs)+127)/128);
    const auto launch=[&](auto tag) {
        using Value=decltype(tag);
        rotary_embedding_cached_kernel<Value><<<blocks,128,0,stream>>>(
            static_cast<const Value*>(input),positions,frequencies,axes,static_cast<Value*>(output),
            rows,heads,tokens,width,pairs,position_axes,batches,maximum,attention_factor);
    };
    if(type==NumericType::f32)launch(float{});else launch(__half{});
}

void rope(const float* input,const float* positions,float* output,int rows,int tokens,int width,
        int dimension,int axes,int batches,int s0,int s1,int s2,float base,cudaStream_t stream) {
    rope_kernel<<<rows,ROPE_BD,0,stream>>>(input,positions,output,rows,tokens,width,dimension,axes,batches,s0,s1,s2,base);
}

void table(const float* input,const int64_t* positions,const float* cosine,const float* sine,float* output,
        int rows,int tokens,int width,int dimension,int length,int axes,int batches,
        int s0,int s1,int s2,cudaStream_t stream) {
    rope_table_kernel<<<rows,ROPE_BD,0,stream>>>(input,positions,cosine,sine,output,
        rows,tokens,width,dimension,length,axes,batches,s0,s1,s2);
}

void table_bf16(const void* input,const int64_t* positions,const float* cosine,const float* sine,void* output,
        int rows,int tokens,int width,int dimension,int length,cudaStream_t stream) {
    rope_table_bf16_kernel<<<rows,ROPE_BD,0,stream>>>(static_cast<const __nv_bfloat16*>(input),positions,
        cosine,sine,static_cast<__nv_bfloat16*>(output),rows,tokens,width,dimension,length);
}

void minicpm(const void* q,const void* k,const void* v,const int64_t* positions,const int64_t* writes,
        const float* cosine,const float* sine,void* output,void* keys,void* values,
        int batches,int query_heads,int key_heads,int length,int capacity,cudaStream_t stream) {
    minicpm_bf16_rope_cache_write_kernel<<<batches*query_heads,128,0,stream>>>(
        static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),
        positions,writes,cosine,sine,static_cast<__nv_bfloat16*>(output),static_cast<__nv_bfloat16*>(keys),
        static_cast<__nv_bfloat16*>(values),query_heads,key_heads,128,64,length,capacity);
}

} // namespace tyloi::cuda::rope_raw
