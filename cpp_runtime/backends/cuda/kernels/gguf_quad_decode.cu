#include "gguf_quad_decode.h"
#include "gguf_quad_storage.h"
#include "ggml-cuda/common.cuh"
#include "gguf_decode.cuh"
#include "gguf_compact_decode.cuh"
#include "gguf_quad_q2.cuh"
#include "gguf_quad_iq1m.cuh"
#include <algorithm>
#include <climits>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>


#include "gguf_prefill_scale.cuh"

namespace mfq::cuda::gguf {namespace {
#define QUAD_FORMATS(X) \
 X(IQ1_S,block_iq1_s) X(IQ1_M,block_iq1_m) \
 X(IQ2_XXS,block_iq2_xxs) X(IQ2_XS,block_iq2_xs) X(IQ2_S,block_iq2_s) \
 X(IQ3_XXS,block_iq3_xxs) X(IQ3_S,block_iq3_s) X(Q2_K,block_q2_K)
template<int F>struct NativeBlock;
#define BLOCK(F,T) template<>struct NativeBlock<GGML_TYPE_##F>{using type=T;};
QUAD_FORMATS(BLOCK)
#undef BLOCK
template<int F>constexpr bool Iq1=F==GGML_TYPE_IQ1_S||F==GGML_TYPE_IQ1_M;
template<int F>constexpr std::uint64_t Extra=F==GGML_TYPE_IQ2_XS?32:F==GGML_TYPE_IQ2_S?64:F==GGML_TYPE_IQ3_XXS?128:F==GGML_TYPE_IQ3_S?176:0;
__device__ void put_word(void* p,unsigned i,unsigned v){
    auto* q=static_cast<unsigned short*>(p)+2*i;q[0]=v;q[1]=v>>16;
}
__device__ unsigned read_word(const unsigned char* p,std::uint64_t at){return *reinterpret_cast<const unsigned*>(p+at);}

template<int F>__device__ void restore_fragment(const unsigned char* packed,std::uint64_t total,
        std::uint64_t qb,unsigned r,unsigned part,typename NativeBlock<F>::type* b){
    const auto entry=qb*8+part;
    if constexpr(Iq1<F>){
        put_word(b->qs,part,read_word(packed,entry*16+r*4));
        reinterpret_cast<unsigned short*>(b->qh)[part]=reinterpret_cast<const unsigned short*>(packed+total*128)[entry*4+r];
        if(!part){
            if constexpr(F==GGML_TYPE_IQ1_S)b->d=reinterpret_cast<const half*>(packed+total*192)[qb*4+r];
            else restore_quad_iq1m_metadata(packed+total*192+qb*32,r,b);
        }
    }else if constexpr(F==GGML_TYPE_Q2_K){
        restore_quad_q2_fragment(packed,total,qb,r,part,b);
    }else{
        if(!part)b->d=reinterpret_cast<const half*>(packed+total*(256+Extra<F>))[qb*4+r];
        const auto a=read_word(packed,entry*32+r*4),c=read_word(packed,entry*32+16+r*4);
        if constexpr(F==GGML_TYPE_IQ2_S){put_word(b->qs,part,a);put_word(b->qs,8+part,c);}
        else{put_word(b->qs,2*part,a);put_word(b->qs,2*part+1,c);}
        if constexpr(F==GGML_TYPE_IQ2_XS)b->scales[part]=packed[total*256+entry*4+r];
        else if constexpr(F==GGML_TYPE_IQ2_S){const auto meta=reinterpret_cast<const unsigned short*>(packed+total*256)[entry*4+r];b->scales[part]=meta;b->qh[part]=meta>>8;}
        else if constexpr(F==GGML_TYPE_IQ3_XXS)put_word(b->qs,16+part,read_word(packed,total*256+entry*16+r*4));
        else if constexpr(F==GGML_TYPE_IQ3_S){
            put_word(b->signs,part,read_word(packed,total*256+entry*16+r*4));
            b->qh[part]=packed[total*384+entry*4+r];
            if(!(part&1))b->scales[part/2]=packed[total*416+(entry/2)*4+r];
        }
    }
}
__device__ __forceinline__ void rotate_w8a8_fragment(float (&values)[8],std::uint64_t column,unsigned lane) {
    #pragma unroll
    for(int j=0;j<8;++j) {
        unsigned sign=unsigned(column+j)*0x9e3779b9u;
        sign^=sign>>16;
        if(sign&0x80000000u)values[j]=-values[j];
    }
    #pragma unroll
    for(int distance=1;distance<8;distance*=2)
        #pragma unroll
        for(int group=0;group<8;group+=2*distance)
            #pragma unroll
            for(int j=0;j<distance;++j) {
                const float low=values[group+j],high=values[group+j+distance];
                values[group+j]=low+high;values[group+j+distance]=low-high;
            }
    #pragma unroll
    for(int distance=1;distance<4;distance*=2)
        #pragma unroll
        for(int j=0;j<8;++j) {
            const float other=__shfl_xor_sync(0xffffffffu,values[j],distance,4);
            values[j]=(lane&distance)?other-values[j]:values[j]+other;
        }
    #pragma unroll
    for(int j=0;j<8;++j)values[j]*=0.1767766952966368811f;
}

template<int F>__device__ __forceinline__ void native_w8a8_values8(
        const unsigned char* packed,std::uint64_t column,float (&values)[8]) {
    if constexpr(F==GGML_TYPE_PTQ1_0) {
        const auto* block=packed+(column/128)*28;
        const auto scale=compact_detail::scale_at(block+26);
        const int lane=int(column%128)/4;
        const auto a=compact_detail::decode_four<0,0>(block,lane);
        const auto b=compact_detail::decode_four<0,0>(block,lane+1);
        values[0]=__fmul_rn(a.x,scale);values[1]=__fmul_rn(a.y,scale);
        values[2]=__fmul_rn(a.z,scale);values[3]=__fmul_rn(a.w,scale);
        values[4]=__fmul_rn(b.x,scale);values[5]=__fmul_rn(b.y,scale);
        values[6]=__fmul_rn(b.z,scale);values[7]=__fmul_rn(b.w,scale);
    }else gguf_values8<F>(packed,column,values);
}

template<int F,bool Quad=true>__global__ void decode_quad_w8a8(const unsigned char* packed,
        std::int64_t storage_rows,std::uint64_t blocks,std::int64_t begin,
        std::uint64_t row_stride,std::int8_t* codes,float* scales) {
    constexpr unsigned Warps=8,Stride=[] {
        if constexpr(Quad)return unsigned((sizeof(typename NativeBlock<F>::type)+15)&~15u);
        else return 0u;
    }();
    __shared__ __align__(16) unsigned char images[Stride?Warps*Stride:1];
    __shared__ float maxima[Warps];
    extern __shared__ float transformed[];
    const unsigned warp=threadIdx.x/32,lane=threadIdx.x%32;
    auto* image=images+warp*Stride;
    const auto row=std::uint64_t(begin)+blockIdx.x,total=std::uint64_t(storage_rows/4)*blocks;
    const auto columns=blocks*256;
    float maximum=0.f;
    for(auto block=std::uint64_t(warp);block<blocks;block+=Warps) {
        float values[8];
        if constexpr(Quad) {
            using B=typename NativeBlock<F>::type;
            if(row<std::uint64_t(storage_rows/4)*4) {
                if(lane<8)restore_fragment<F>(packed,total,(row/4)*blocks+block,unsigned(row%4),lane,reinterpret_cast<B*>(image));
            }else {
                const auto* source=packed+total*4*sizeof(B)+(row-std::uint64_t(storage_rows/4)*4)*blocks*sizeof(B)+block*sizeof(B);
                for(unsigned j=lane;j<sizeof(B);j+=32)image[j]=source[j];
            }
            __syncwarp();
            gguf_values8<F>(image,lane*8,values);
        }else native_w8a8_values8<F>(packed+row*row_stride,block*256+lane*8,values);
        const auto column=block*256+lane*8;
        rotate_w8a8_fragment(values,column,lane);
        #pragma unroll
        for(int j=0;j<8;++j){transformed[column+j]=values[j];maximum=fmaxf(maximum,fabsf(values[j]));}
        if constexpr(Quad)__syncwarp();
    }
    #pragma unroll
    for(int distance=16;distance;distance/=2)
        maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,distance));
    if(!lane)maxima[warp]=maximum;
    __syncthreads();
    if(!warp) {
        maximum=lane<Warps?maxima[lane]:0.f;
        #pragma unroll
        for(int distance=16;distance;distance/=2)
            maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,distance));
        if(!lane){maxima[0]=maximum;scales[blockIdx.x]=maximum/127.f;}
    }
    __syncthreads();
    const float inverse=maxima[0]>0.f?127.f/maxima[0]:0.f;
    for(auto column=std::uint64_t(threadIdx.x);column<columns;column+=blockDim.x)
        codes[std::uint64_t(blockIdx.x)*columns+column]=std::int8_t(max(-127,min(127,__float2int_rn(transformed[column]*inverse))));
}

// Wide rows retain only their scale reduction between decoding passes.
template<int F,bool Quad=true>__global__ void decode_quad_w8a8_twopass(const unsigned char* packed,
        std::int64_t storage_rows,std::uint64_t blocks,std::int64_t begin,
        std::uint64_t row_stride,std::int8_t* codes,float* scales) {
    constexpr unsigned Warps=8,Stride=[] {
        if constexpr(Quad)return unsigned((sizeof(typename NativeBlock<F>::type)+15)&~15u);
        else return 0u;
    }();
    __shared__ __align__(16) unsigned char images[Stride?Warps*Stride:1];
    __shared__ float maxima[Warps];
    const unsigned warp=threadIdx.x/32,lane=threadIdx.x%32;
    auto* image=images+warp*Stride;
    const auto row=std::uint64_t(begin)+blockIdx.x,total=std::uint64_t(storage_rows/4)*blocks;
    const auto columns=blocks*256;
    float maximum=0.f;
    for(auto block=std::uint64_t(warp);block<blocks;block+=Warps) {
        float values[8];
        if constexpr(Quad) {
            using B=typename NativeBlock<F>::type;
            if(row<std::uint64_t(storage_rows/4)*4) {
                if(lane<8)restore_fragment<F>(packed,total,(row/4)*blocks+block,unsigned(row%4),lane,reinterpret_cast<B*>(image));
            }else {
                const auto* source=packed+total*4*sizeof(B)+(row-std::uint64_t(storage_rows/4)*4)*blocks*sizeof(B)+block*sizeof(B);
                for(unsigned j=lane;j<sizeof(B);j+=32)image[j]=source[j];
            }
            __syncwarp();
            gguf_values8<F>(image,lane*8,values);
        }else native_w8a8_values8<F>(packed+row*row_stride,block*256+lane*8,values);
        const auto column=block*256+lane*8;
        rotate_w8a8_fragment(values,column,lane);
        #pragma unroll
        for(int j=0;j<8;++j){maximum=fmaxf(maximum,fabsf(values[j]));}
        if constexpr(Quad)__syncwarp();
    }
    #pragma unroll
    for(int distance=16;distance;distance/=2)
        maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,distance));
    if(!lane)maxima[warp]=maximum;
    __syncthreads();
    if(!warp) {
        maximum=lane<Warps?maxima[lane]:0.f;
        #pragma unroll
        for(int distance=16;distance;distance/=2)
            maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,distance));
        if(!lane){maxima[0]=maximum;scales[blockIdx.x]=maximum/127.f;}
    }
    __syncthreads();
    const float inverse=maxima[0]>0.f?127.f/maxima[0]:0.f;
    for(auto block=std::uint64_t(warp);block<blocks;block+=Warps) {
        float values[8];
        if constexpr(Quad) {
            using B=typename NativeBlock<F>::type;
            if(row<std::uint64_t(storage_rows/4)*4) {
                if(lane<8)restore_fragment<F>(packed,total,(row/4)*blocks+block,unsigned(row%4),lane,reinterpret_cast<B*>(image));
            }else {
                const auto* source=packed+total*4*sizeof(B)+(row-std::uint64_t(storage_rows/4)*4)*blocks*sizeof(B)+block*sizeof(B);
                for(unsigned j=lane;j<sizeof(B);j+=32)image[j]=source[j];
            }
            __syncwarp();
            gguf_values8<F>(image,lane*8,values);
        }else native_w8a8_values8<F>(packed+row*row_stride,block*256+lane*8,values);
        const auto column=block*256+lane*8;
        rotate_w8a8_fragment(values,column,lane);
        #pragma unroll
        for(int j=0;j<8;++j)
            codes[std::uint64_t(blockIdx.x)*columns+column+j]=std::int8_t(max(-127,min(127,__float2int_rn(values[j]*inverse))));
        if constexpr(Quad)__syncwarp();
    }
}

template<int F,bool Half,bool Scaled=false>__global__ void decode_quad(const unsigned char* packed,
        std::int64_t storage_rows,std::uint64_t blocks,std::int64_t begin,std::int64_t rows,
        const std::int64_t* indices,std::uint64_t count,void* output,const std::int32_t* exponents=nullptr){
    using B=typename NativeBlock<F>::type;
    // A warp owns one quantization block, so reconstruction requires no
    // block-wide barriers. Four independent blocks share a CTA. Their local
    // images remain aligned even when the encoded block size is not.
    constexpr unsigned Warps=4,Stride=(sizeof(B)+15)&~15u;
    __shared__ __align__(16) unsigned char images[Warps*Stride];
    const unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
    auto* image=images+warp*Stride;
    const auto total=std::uint64_t(storage_rows/4)*blocks;
    const bool vector_output=!(reinterpret_cast<uintptr_t>(output)%16);
    for(auto at=std::uint64_t(blockIdx.x)*Warps+warp;at<count*blocks;
            at+=std::uint64_t(gridDim.x)*Warps){
        const auto output_row=at/blocks,block=at%blocks;
        const auto selected=indices?indices[output_row]:std::int64_t(output_row);
        const bool valid=selected>=0 && selected<rows;
        const auto row=valid?std::uint64_t(begin)+std::uint64_t(selected):0;
        if(valid){
            if(row<std::uint64_t(storage_rows/4)*4){
                if(lane<8)restore_fragment<F>(packed,total,(row/4)*blocks+block,
                    unsigned(row%4),lane,reinterpret_cast<B*>(image));
            }else{
                const auto* source=packed+total*4*sizeof(B)+(row-std::uint64_t(storage_rows/4)*4)*blocks*sizeof(B)+block*sizeof(B);
                for(unsigned j=lane;j<sizeof(B);j+=32)image[j]=source[j];
            }
        }
        __syncwarp();
        const auto offset=at*256;
        if(vector_output){
            float values[8]={};
            if(valid)gguf_values8<F>(image,lane*8,values);
            if constexpr(Scaled){
                const auto exponent=-exponents[output_row];
                #pragma unroll
                for(int j=0;j<8;++j)values[j]=prefill_ldexp(values[j],exponent);
            }
            if constexpr(Half)gguf_store8(static_cast<half*>(output)+offset+lane*8,values);
            else gguf_store8(static_cast<float*>(output)+offset+lane*8,values);
        }else{
            // Element-aligned output views retain coalesced scalar stores.
            #pragma unroll
            for(unsigned j=0;j<8;++j){
                const auto k=lane+j*32;
                float value=valid?gguf_value<F>(image,k):0.f;
                if constexpr(Scaled)value=prefill_ldexp(value,-exponents[output_row]);
                if constexpr(Half)static_cast<half*>(output)[offset+k]=__float2half_rn(value);
                else static_cast<float*>(output)[offset+k]=value;
            }
        }
        __syncwarp();
    }
}
template<int F,bool Quad>
int decode_w8a8_mode(const void* packed,std::int64_t columns) {
    int device=0,available=0,active=0;
    auto status=cudaGetDevice(&device);
    if(status==cudaSuccess)status=cudaDeviceGetAttribute(&available,cudaDevAttrMaxSharedMemoryPerBlock,device);
    cudaFuncAttributes attributes{};
    if(status==cudaSuccess)status=cudaFuncGetAttributes(&attributes,decode_quad_w8a8<F,Quad>);
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    const auto bytes=std::size_t(columns)*sizeof(float);
    if(attributes.sharedSizeBytes<std::size_t(available) &&
            std::uint64_t(columns)<=(std::uint64_t(available)-attributes.sharedSizeBytes)/sizeof(float)) {
        status=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,decode_quad_w8a8<F,Quad>,256,bytes);
        if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    }
    if(active>=2)return 1;
    // A wide host image retains the single-read staged decoder.
    cudaPointerAttributes location{};
    status=cudaPointerGetAttributes(&location,packed);
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    if(location.type==cudaMemoryTypeHost)return 0;
    status=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,decode_quad_w8a8_twopass<F,Quad>,256,0);
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    return active>=2?2:0;
}
template<int F,bool Quad>
bool launch_decode_w8a8(const void* packed,std::int64_t storage_rows,std::int64_t columns,
        std::int64_t begin,std::int64_t rows,std::uint64_t stride,std::int8_t* codes,float* scales,cudaStream_t stream) {
    const auto mode=decode_w8a8_mode<F,Quad>(packed,columns);
    if(!mode)return false;
    if(mode==1) {
        const auto bytes=std::size_t(columns)*sizeof(float);
        decode_quad_w8a8<F,Quad><<<unsigned(rows),256,bytes,stream>>>(static_cast<const unsigned char*>(packed),
            storage_rows,std::uint64_t(columns/256),begin,stride,codes,scales);
    } else {
        decode_quad_w8a8_twopass<F,Quad><<<unsigned(rows),256,0,stream>>>(static_cast<const unsigned char*>(packed),
            storage_rows,std::uint64_t(columns/256),begin,stride,codes,scales);
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<bool> reported{};
            if(!reported.exchange(true))std::fprintf(stderr,
                "native_gguf_prefill_w8a8_wide actual_match=1 format=%d quad=%d N=%lld K=%lld native_imma_input=1\n",
                F,int(Quad),(long long)rows,(long long)columns);
        }
    }
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    return true;
}

void trace_decode(bool scaled){
    const auto* setting=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");
    if(!setting || setting[0]!='1')return;
    static std::atomic<unsigned> seen{0};
    const unsigned bit=scaled?2:1;
    if(!(seen.fetch_or(bit,std::memory_order_relaxed)&bit))
        std::fprintf(stderr,"native_gguf_quad_decode actual_match=1 warp_per_block=1 blocks_per_cta=4 shared_codebook=1 scaled=%d\n",int(scaled));
}
}
void decode_quad_rows(ggml_type type,const void* packed,std::int64_t storage_rows,
        std::int64_t columns,std::int64_t begin,std::int64_t rows,const std::int64_t* indices,
        std::int64_t count,void* output,bool half_output,cudaStream_t stream){
    quad_storage_bytes(type,storage_rows,columns);
    if(begin<0||rows<1||begin>storage_rows-rows||count<0||(!indices&&count!=rows)||
       std::uint64_t(count)>UINT64_MAX/std::uint64_t(columns)/(half_output?2:4))
        throw std::invalid_argument("Invalid GGUF compute-storage decode view");
    if(!count)return;
    if(!packed||!output||packed==output||reinterpret_cast<uintptr_t>(packed)%16||
       reinterpret_cast<uintptr_t>(output)%(half_output?2:4)||
       (indices&&reinterpret_cast<uintptr_t>(indices)%8))
        throw std::invalid_argument("Invalid GGUF compute-storage decode buffers");
    const auto blocks=std::uint64_t(columns/256),work=std::uint64_t(count)*blocks;
    trace_decode(false);
    const auto launch=[&]<int F,bool Half>(){
        decode_quad<F,Half><<<unsigned(std::min<std::uint64_t>((work+3)/4,65535)),128,0,stream>>>(
            static_cast<const unsigned char*>(packed),storage_rows,blocks,begin,rows,indices,count,output);
    };
    switch(type){
#define DECODE(F,T) case GGML_TYPE_##F:if(half_output)launch.template operator()<GGML_TYPE_##F,true>();else launch.template operator()<GGML_TYPE_##F,false>();break;
    QUAD_FORMATS(DECODE)
#undef DECODE
    default:break;
    }
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
bool decode_quad_rows_w8a8(ggml_type type,const void* packed,std::int64_t storage_rows,std::int64_t columns,
        std::int64_t begin,std::int64_t rows,std::int8_t* codes,float* scales,cudaStream_t stream) {
    quad_storage_bytes(type,storage_rows,columns);
    if(begin<0 || rows<1 || rows>INT_MAX || begin>storage_rows-rows || !packed || !codes || !scales)
        throw std::invalid_argument("Invalid GGUF W8A8 compute-storage view");
    bool launched=false;
    const auto launch=[&]<int F>() {
        launched=launch_decode_w8a8<F,true>(packed,storage_rows,columns,begin,rows,0,codes,scales,stream);
    };
    switch(type) {
#define W8_QUAD(F,T) case GGML_TYPE_##F:launch.template operator()<GGML_TYPE_##F>();break;
        QUAD_FORMATS(W8_QUAD)
#undef W8_QUAD
        default:break;
    }
    return launched;
}

bool decode_rows_w8a8(ggml_type type,const void* packed,std::uint64_t stride,
        std::int64_t rows,std::int64_t columns,std::int8_t* codes,float* scales,cudaStream_t stream) {
    if(rows<1 || rows>INT_MAX || columns<1 || !packed || !codes || !scales)
        throw std::invalid_argument("Invalid GGUF W8A8 source matrix");
    // This fused decoder covers complete 256-coordinate tiles. Other
    // valid source widths retain the general decoder and padded pack path.
    if(columns%256)return false;
    if(stride<mfq::gguf::row_bytes(type,columns) || stride>UINT64_MAX/std::uint64_t(rows))
        throw std::invalid_argument("Invalid GGUF W8A8 source row stride");
    switch(type) {
#define W8_NATIVE(F) case GGML_TYPE_##F:return launch_decode_w8a8<GGML_TYPE_##F,false>(packed,rows,columns,0,rows,stride,codes,scales,stream);
        W8_NATIVE(Q3_K) W8_NATIVE(Q4_K) W8_NATIVE(Q5_K) W8_NATIVE(Q6_K)
        W8_NATIVE(IQ4_XS) W8_NATIVE(IQ4_NL) W8_NATIVE(Q8_0)
        W8_NATIVE(PTQ1_0)
#undef W8_NATIVE
        default:return false;
    }
}
bool supports_rows_w8a8(ggml_type type,const void* packed,std::int64_t columns,bool quad) {
    if(!packed || columns<=0 || columns%256)return false;
    if(quad) {
        switch(type) {
#define W8_QUAD_SUPPORT(F,T) case GGML_TYPE_##F:return decode_w8a8_mode<GGML_TYPE_##F,true>(packed,columns)!=0;
            QUAD_FORMATS(W8_QUAD_SUPPORT)
#undef W8_QUAD_SUPPORT
            default:return false;
        }
    }
    switch(type) {
#define W8_NATIVE_SUPPORT(F) case GGML_TYPE_##F:return decode_w8a8_mode<GGML_TYPE_##F,false>(packed,columns)!=0;
        W8_NATIVE_SUPPORT(Q3_K) W8_NATIVE_SUPPORT(Q4_K) W8_NATIVE_SUPPORT(Q5_K) W8_NATIVE_SUPPORT(Q6_K)
        W8_NATIVE_SUPPORT(IQ4_XS) W8_NATIVE_SUPPORT(IQ4_NL) W8_NATIVE_SUPPORT(Q8_0) W8_NATIVE_SUPPORT(PTQ1_0)
#undef W8_NATIVE_SUPPORT
        default:return false;
    }
}

void decode_quad_rows_scaled(ggml_type type,const void* packed,std::int64_t storage_rows,std::int64_t columns,
        std::int64_t begin,std::int64_t rows,const std::int32_t* exponents,void* output,cudaStream_t stream) {
    quad_storage_bytes(type,storage_rows,columns);
    if(begin<0 || rows<1 || begin>storage_rows-rows || !exponents || !packed || !output)
        throw std::invalid_argument("Invalid GGUF scaled compute-storage view");
    const auto blocks=std::uint64_t(columns/256),work=std::uint64_t(rows)*blocks;
    trace_decode(true);
    switch(type) {
#define SCALED_QUAD(F,T) case GGML_TYPE_##F:decode_quad<GGML_TYPE_##F,true,true><<<unsigned(std::min<std::uint64_t>((work+3)/4,65535)),128,0,stream>>>(static_cast<const unsigned char*>(packed),storage_rows,blocks,begin,rows,nullptr,rows,output,exponents);break;
        QUAD_FORMATS(SCALED_QUAD)
#undef SCALED_QUAD
        default:throw std::invalid_argument("Unsupported GGUF scaled compute-storage format");
    }
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
}
