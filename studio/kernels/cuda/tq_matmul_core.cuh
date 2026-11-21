#pragma once
// Decode one compressed row and reuse each Gaussian pair across a token tile.
template<int T,int V,bool Columns> struct MultiDotConsumer {
    uint32_t packed[T][V/2];
    float sum[T][2];
    const half* scales;
    int logical;
    __device__ __forceinline__ void operator()(int j,float a,float b) {
        if constexpr(Columns) {
            if(j>=logical)return;
            const uint32_t value=j+1<logical?__ldg(reinterpret_cast<const uint32_t*>(scales+j)):
                uint32_t(__ldg(reinterpret_cast<const uint16_t*>(scales+j)));
            a=__fmul_rn(a,__half2float(__ushort_as_half(value)));
            b=j+1<logical?__fmul_rn(b,__half2float(__ushort_as_half(value>>16))):0.f;
        }
        #pragma unroll
        for(int t=0;t<T;++t) {
            const uint32_t value=packed[t][j/2];
            sum[t][0]=__fmaf_rn(a,__half2float(__ushort_as_half(value)),sum[t][0]);
            sum[t][1]=__fmaf_rn(b,__half2float(__ushort_as_half(value>>16)),sum[t][1]);
        }
    }
};

template<int T,int Bits,bool Columns,bool Indexed=false,int V=16>
__device__ __forceinline__ void multi_row(TrellisProjectionView m,int row,
        const uint16_t* x,int first,int tokens,int xs,float (&value)[T],const int64_t* offsets=nullptr) {
    const uint64_t d=__ldg(m.descriptors+row);
    const int lane=threadIdx.x&31;
    #pragma unroll
    for(int t=0;t<T;++t)value[t]=0;
    for(int col=lane*V;col<m.input_columns;col+=32*V) {
        MultiDotConsumer<T,V,Columns> consumer{};
        consumer.logical=m.input_columns-col;
        if constexpr(Columns)consumer.scales=static_cast<const half*>(m.scales)+col;
        #pragma unroll
        for(int t=0;t<T;++t) {
            const auto* source=x+(Indexed?offsets[t]:std::size_t(first+t)*xs)+col;
            if(first+t<tokens && col+V<=m.input_columns && (reinterpret_cast<uintptr_t>(source)&(V==4?7:15))==0) {
                if constexpr(V==4) {
                    const uint2 a=__ldg(reinterpret_cast<const uint2*>(source));
                    consumer.packed[t][0]=a.x;consumer.packed[t][1]=a.y;
                }else {
                    const uint4 a=__ldg(reinterpret_cast<const uint4*>(source));
                    consumer.packed[t][0]=a.x;consumer.packed[t][1]=a.y;
                    consumer.packed[t][2]=a.z;consumer.packed[t][3]=a.w;
                    if constexpr(V==16) {
                        const uint4 b=__ldg(reinterpret_cast<const uint4*>(source+8));
                        consumer.packed[t][4]=b.x;consumer.packed[t][5]=b.y;
                        consumer.packed[t][6]=b.z;consumer.packed[t][7]=b.w;
                    }
                }
            }else {
                #pragma unroll
                for(int j=0;j<V;++j)if(first+t<tokens && col+j<m.input_columns)
                    consumer.packed[t][j/2]|=uint32_t(__ldg(source+j))<<((j&1)*16);
            }
        }
        visit_values<V,true,Columns,Bits,MultiDotConsumer<T,V,Columns>,true>(
            m.words,d,m.scales,static_cast<const half*>(m.relative),nullptr,nullptr,
            row,col,m.stored_columns,m.gslog,consumer);
        float scale=1;
        if constexpr(!Columns) {
            const int group=row*((m.stored_columns+(1<<m.gslog)-1)>>m.gslog)+(col>>m.gslog);
            const uint8_t selector=__ldg(static_cast<const uint8_t*>(m.scales)+(group>>1));
            scale=__half2float(__ldg(static_cast<const half*>(m.relative)+((selector>>((group&1)*4))&15)));
        }
        #pragma unroll
        for(int t=0;t<T;++t)value[t]=__fmaf_rn(scale,consumer.sum[t][0]+consumer.sum[t][1],value[t]);
    }
    const float anchor=__half2float(__ushort_as_half(uint16_t(d>>40)));
    #pragma unroll
    for(int t=0;t<T;++t)value[t]=__fmul_rn(anchor,value[t]);
}

template<int T,bool Columns,bool Indexed=false,int V=16>
__device__ __forceinline__ void multi_dispatch(TrellisProjectionView m,int row,
        const uint16_t* x,int first,int tokens,int xs,float (&value)[T],const int64_t* offsets=nullptr) {
    const uint64_t d=__ldg(m.descriptors+row);
    #define MC(K) case K-1:multi_row<T,K,Columns,Indexed,V>(m,row,x,first,tokens,xs,value,offsets);return
    switch((d>>32)&7){MC(1);MC(2);MC(3);MC(4);MC(5);MC(6);MC(7);MC(8);}
    #undef MC
}
