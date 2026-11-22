#pragma once

__global__ void tq_prepare_resident_indexed_kernel(const TrellisProjectionView* gate,const TrellisProjectionView* up,
        const TrellisProjectionView* down,const int64_t* ids,TrellisIndexedProjectionJob* gu,TrellisIndexedProjectionJob* dn,
        int slots,int routes,int experts,int input,int hidden,int output,int tile) {
    const auto slot=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(slot>=slots)return;
    TrellisIndexedProjectionJob g{},d{};
    const auto expert=ids[slot];
    if(expert>=0 && expert<experts && gate[expert].words && up[expert].words && down[expert].words) {
        int preceding=0;
        for(int64_t i=0;i<slot;++i)preceding+=ids[i]==expert;
        if(preceding%tile==0) {
            g.weight=gate[expert];g.second=up[expert];d.weight=down[expert];
            for(int64_t i=slot;i<slots && g.count<tile;++i)if(ids[i]==expert) {
                const auto at=g.count++;
                g.input_offsets[at]=(i/routes)*input;g.output_offsets[at]=i*hidden;
                d.input_offsets[at]=i*hidden;d.output_offsets[at]=i*output;
            }
            d.count=g.count;
        }
    }
    gu[slot]=g;dn[slot]=d;
}
extern "C" int mfq_tq_prepare_resident_indexed(const TrellisProjectionView* gate,const TrellisProjectionView* up,
        const TrellisProjectionView* down,const int64_t* ids,TrellisIndexedProjectionJob* gu,TrellisIndexedProjectionJob* dn,
        int slots,int routes,int experts,int input,int hidden,int output,int tile,cudaStream_t stream) {
    if(!gate || !up || !down || !ids || !gu || !dn || gu==dn || slots<1 || routes<1 || slots%routes ||
       experts<1 || input<1 || hidden<1 || output<1 || tile<1 || tile>4)return -1;
    tq_prepare_resident_indexed_kernel<<<(uint64_t(slots)+127)/128,128,0,stream>>>(
        gate,up,down,ids,gu,dn,slots,routes,experts,input,hidden,output,tile);
    return int(cudaGetLastError());
}

__global__ void tq_snapshot_resident_kernel(const TrellisProjectionView* gate,const TrellisProjectionView* up,
        const TrellisProjectionView* down,const int64_t* ids,TrellisProjectionView* selected,int routes,int experts) {
    for(int slot=threadIdx.x;slot<routes;slot+=blockDim.x) {
        const auto expert=ids[slot];
        TrellisProjectionView g{},u{},d{};
        if(expert>=0 && expert<experts) {
            g=gate[expert];u=up[expert];d=down[expert];
            if(!g.words || !u.words || !d.words){g={};u={};d={};}
        }
        selected[slot]=g;selected[routes+slot]=u;selected[2*routes+slot]=d;
    }
}
extern "C" int mfq_tq_snapshot_resident(const TrellisProjectionView* gate,const TrellisProjectionView* up,
        const TrellisProjectionView* down,const int64_t* ids,TrellisProjectionView* selected,
        int routes,int experts,cudaStream_t stream) {
    if(!gate || !up || !down || !ids || !selected || routes<1 || experts<1)return -1;
    tq_snapshot_resident_kernel<<<1,128,0,stream>>>(gate,up,down,ids,selected,routes,experts);
    return int(cudaGetLastError());
}

template<int DT,bool Guarded=false,int Partition=0>
__global__ __launch_bounds__(256)
void tq_routed_gu_kernel(const TrellisProjectionView* gate,const TrellisProjectionView* up,
        const int64_t* ids,const Scalar<DT>* x,Scalar<DT>* hidden,float limit,int stride,const uint32_t* cancelled=nullptr,
        const TrellisProjectionView* resident_mask=nullptr,int experts=0) {
    if constexpr(Guarded)if(cancelled && *cancelled)return;
    const int warp=threadIdx.x/32,role=warp&1,slot=warp/2;
    const auto expert=ids[blockIdx.y];
    if constexpr(Partition) {
        if(expert<0 || expert>=experts)return;
        const bool resident=resident_mask[blockIdx.y].words!=nullptr;
        if(resident!=(Partition==1))return;
    }
    const auto at=Partition==1 || Partition==3?int64_t(blockIdx.y):expert;
    const auto m=role?up[at]:gate[at];
    const int row=blockIdx.x*4+slot;
    float value=row<m.rows?matrix_dot<DT,1,false>(m,row,x):0.f;
    #pragma unroll
    for(int step=16;step;step/=2)value+=__shfl_down_sync(0xffffffff,value,step);
    __shared__ float joined[4][2];
    if((threadIdx.x&31)==0) {
        if(row<m.rows && m.bias)value+=__ldg(m.bias+row);
        joined[slot][role]=scalar_float<DT>(scalar_round<DT>(value));
    }
    __syncthreads();
    if(!role && row<m.rows && (threadIdx.x&31)==0) {
        float g=joined[slot][0],u=joined[slot][1];
        if(limit>0){g=fminf(g,limit);u=fmaxf(-limit,fminf(u,limit));}
        hidden[std::size_t(blockIdx.y)*stride+row]=scalar_round<DT>(u*(g/(1.f+expf(-g))));
    }
}

template<int DT,bool Guarded=false,int Partition=0>
__global__ __launch_bounds__(128)
void tq_routed_down_kernel(const TrellisProjectionView* weights,const int64_t* ids,
        const Scalar<DT>* hidden,Scalar<DT>* output,int xs,int ys,const uint32_t* cancelled=nullptr,
        const TrellisProjectionView* resident_mask=nullptr,int experts=0) {
    if constexpr(Guarded)if(cancelled && *cancelled)return;
    const auto expert=ids[blockIdx.y];
    if constexpr(Partition) {
        if(expert<0 || expert>=experts)return;
        const bool resident=resident_mask[blockIdx.y].words!=nullptr;
        if(resident!=(Partition==1))return;
    }
    const auto m=weights[Partition==1 || Partition==3?int64_t(blockIdx.y):expert];
    const int row=blockIdx.x*4+threadIdx.x/32;
    if(row>=m.rows)return;
    float value=matrix_dot<DT,1,true>(m,row,hidden+std::size_t(blockIdx.y)*xs);
    #pragma unroll
    for(int step=16;step;step/=2)value+=__shfl_down_sync(0xffffffff,value,step);
    if((threadIdx.x&31)==0) {
        if(m.bias)value+=__ldg(m.bias+row);
        output[std::size_t(blockIdx.y)*ys+row]=scalar_round<DT>(value);
    }
}

extern "C" int mfq_tq_routed_swiglu(const TrellisProjectionView* gate,const TrellisProjectionView* up,
        const int64_t* ids,const void* x,void* hidden,int routes,int rows,int dtype,float limit,cudaStream_t stream) {
    if(!gate || !up || !ids || !x || !hidden || x==hidden || routes<1 || routes>65535 || rows<1 ||
       !std::isfinite(limit) || limit<0)return -1;
    #define ROUTED_GU(D) tq_routed_gu_kernel<D><<<dim3((uint64_t(rows)+3)/4,routes),256,0,stream>>>( \
        gate,up,ids,static_cast<const Scalar<D>*>(x),static_cast<Scalar<D>*>(hidden),limit,rows)
    if(dtype==0){ROUTED_GU(0);}else if(dtype==1){ROUTED_GU(1);}else if(dtype==2){ROUTED_GU(2);}else return -1;
    #undef ROUTED_GU
    return int(cudaGetLastError());
}

extern "C" int mfq_tq_routed_down(const TrellisProjectionView* down,const int64_t* ids,
        const void* hidden,void* output,int routes,int rows,int width,int dtype,cudaStream_t stream) {
    if(!down || !ids || !hidden || !output || hidden==output || routes<1 || routes>65535 || rows<1 || width<1)return -1;
    #define ROUTED_DOWN(D) tq_routed_down_kernel<D><<<dim3((uint64_t(rows)+3)/4,routes),128,0,stream>>>( \
        down,ids,static_cast<const Scalar<D>*>(hidden),static_cast<Scalar<D>*>(output),width,rows)
    if(dtype==0){ROUTED_DOWN(0);}else if(dtype==1){ROUTED_DOWN(1);}else if(dtype==2){ROUTED_DOWN(2);}else return -1;
    #undef ROUTED_DOWN
    return int(cudaGetLastError());
}

extern "C" int mfq_tq_routed_guarded_swiglu(const TrellisProjectionView* gate,const TrellisProjectionView* up,
        const int64_t* ids,const void* x,void* hidden,int routes,int rows,int dtype,float limit,
        const uint32_t* cancelled,cudaStream_t stream) {
    if(!gate || !up || !ids || !x || !hidden || !cancelled || x==hidden || routes<1 || routes>65535 || rows<1 ||
       !std::isfinite(limit) || limit<0)return -1;
    #define GUARDED_GU(D) tq_routed_gu_kernel<D,true><<<dim3((uint64_t(rows)+3)/4,routes),256,0,stream>>>( \
        gate,up,ids,static_cast<const Scalar<D>*>(x),static_cast<Scalar<D>*>(hidden),limit,rows,cancelled)
    if(dtype==0){GUARDED_GU(0);}else if(dtype==1){GUARDED_GU(1);}else if(dtype==2){GUARDED_GU(2);}else return -1;
    #undef GUARDED_GU
    return int(cudaGetLastError());
}
extern "C" int mfq_tq_routed_guarded_down(const TrellisProjectionView* down,const int64_t* ids,
        const void* hidden,void* output,int routes,int rows,int width,int dtype,
        const uint32_t* cancelled,cudaStream_t stream) {
    if(!down || !ids || !hidden || !output || !cancelled || hidden==output || routes<1 || routes>65535 || rows<1 || width<1)return -1;
    #define GUARDED_DOWN(D) tq_routed_down_kernel<D,true><<<dim3((uint64_t(rows)+3)/4,routes),128,0,stream>>>( \
        down,ids,static_cast<const Scalar<D>*>(hidden),static_cast<Scalar<D>*>(output),width,rows,cancelled)
    if(dtype==0){GUARDED_DOWN(0);}else if(dtype==1){GUARDED_DOWN(1);}else if(dtype==2){GUARDED_DOWN(2);}else return -1;
    #undef GUARDED_DOWN
    return int(cudaGetLastError());
}

static int tq_partition_swiglu(const TrellisProjectionView* gate,const TrellisProjectionView* up,
        const TrellisProjectionView* resident_mask,const int64_t* ids,const void* x,void* hidden,
        int routes,int experts,int rows,int dtype,float limit,bool resident,const uint32_t* cancelled,cudaStream_t stream,
        bool slot_views) {
    if(!gate || !up || !resident_mask || !ids || !x || !hidden || x==hidden || routes<1 || routes>65535 ||
       experts<1 || rows<1 || !std::isfinite(limit) || limit<0 || (!resident && !cancelled))return -1;
    #define PARTITION_GU(D,P) tq_routed_gu_kernel<D,true,P><<<dim3((uint64_t(rows)+3)/4,routes),256,0,stream>>>( \
        gate,up,ids,static_cast<const Scalar<D>*>(x),static_cast<Scalar<D>*>(hidden),limit,rows,cancelled,resident_mask,experts)
    #define SELECT_GU(D) if(resident){PARTITION_GU(D,1);}else if(slot_views){PARTITION_GU(D,3);}else{PARTITION_GU(D,2);}
    if(dtype==0){SELECT_GU(0);}else if(dtype==1){SELECT_GU(1);}else if(dtype==2){SELECT_GU(2);}else return -1;
    #undef SELECT_GU
    #undef PARTITION_GU
    return int(cudaGetLastError());
}
static int tq_partition_down(const TrellisProjectionView* down,const TrellisProjectionView* resident_mask,
        const int64_t* ids,const void* hidden,void* output,int routes,int experts,int rows,int width,int dtype,
        bool resident,const uint32_t* cancelled,cudaStream_t stream,bool slot_views) {
    if(!down || !resident_mask || !ids || !hidden || !output || hidden==output || routes<1 || routes>65535 ||
       experts<1 || rows<1 || width<1 || (!resident && !cancelled))return -1;
    #define PARTITION_DOWN(D,P) tq_routed_down_kernel<D,true,P><<<dim3((uint64_t(rows)+3)/4,routes),128,0,stream>>>( \
        down,ids,static_cast<const Scalar<D>*>(hidden),static_cast<Scalar<D>*>(output),width,rows,cancelled,resident_mask,experts)
    #define SELECT_DOWN(D) if(resident){PARTITION_DOWN(D,1);}else if(slot_views){PARTITION_DOWN(D,3);}else{PARTITION_DOWN(D,2);}
    if(dtype==0){SELECT_DOWN(0);}else if(dtype==1){SELECT_DOWN(1);}else if(dtype==2){SELECT_DOWN(2);}else return -1;
    #undef SELECT_DOWN
    #undef PARTITION_DOWN
    return int(cudaGetLastError());
}
extern "C" int mfq_tq_partition_swiglu(const TrellisProjectionView* gate,const TrellisProjectionView* up,
        const TrellisProjectionView* resident_mask,const int64_t* ids,const void* x,void* hidden,
        int routes,int experts,int rows,int dtype,float limit,bool resident,const uint32_t* cancelled,cudaStream_t stream) {
    return tq_partition_swiglu(gate,up,resident_mask,ids,x,hidden,routes,experts,rows,dtype,limit,resident,cancelled,stream,false);
}
extern "C" int mfq_tq_partition_down(const TrellisProjectionView* down,const TrellisProjectionView* resident_mask,
        const int64_t* ids,const void* hidden,void* output,int routes,int experts,int rows,int width,int dtype,
        bool resident,const uint32_t* cancelled,cudaStream_t stream) {
    return tq_partition_down(down,resident_mask,ids,hidden,output,routes,experts,rows,width,dtype,resident,cancelled,stream,false);
}
extern "C" int mfq_tq_partition_slot_swiglu(const TrellisProjectionView* gate,const TrellisProjectionView* up,
        const TrellisProjectionView* resident_mask,const int64_t* ids,const void* x,void* hidden,
        int routes,int experts,int rows,int dtype,float limit,const uint32_t* cancelled,cudaStream_t stream) {
    return tq_partition_swiglu(gate,up,resident_mask,ids,x,hidden,routes,experts,rows,dtype,limit,false,cancelled,stream,true);
}
extern "C" int mfq_tq_partition_slot_down(const TrellisProjectionView* down,const TrellisProjectionView* resident_mask,
        const int64_t* ids,const void* hidden,void* output,int routes,int experts,int rows,int width,int dtype,
        const uint32_t* cancelled,cudaStream_t stream) {
    return tq_partition_down(down,resident_mask,ids,hidden,output,routes,experts,rows,width,dtype,false,cancelled,stream,true);
}
