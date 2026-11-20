#pragma once

__global__ void tq_prepare_resident_indexed_kernel(const TrellisProjectionView* gate,const TrellisProjectionView* up,
        const TrellisProjectionView* down,const int64_t* ids,TrellisIndexedProjectionJob* gate_up,
        TrellisIndexedProjectionJob* output_jobs,int slots,int routes,int experts,int input,int hidden,int output,int tile) {
    const auto slot=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(slot>=slots)return;
    TrellisIndexedProjectionJob g{},d{};
    const auto expert=ids[slot];
    if(expert>=0 && expert<experts) {
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
    gate_up[slot]=g;output_jobs[slot]=d;
}

extern "C" int mfq_tq_prepare_resident_indexed(const TrellisProjectionView* gate,const TrellisProjectionView* up,
        const TrellisProjectionView* down,const int64_t* ids,TrellisIndexedProjectionJob* gate_up,
        TrellisIndexedProjectionJob* output_jobs,int slots,int routes,int experts,int input,int hidden,int output,
        int tile,cudaStream_t stream) {
    if(!gate || !up || !down || !ids || !gate_up || !output_jobs || gate_up==output_jobs ||
        slots<1 || routes<1 || slots%routes || experts<1 || input<1 || hidden<1 || output<1 || tile<1 || tile>4)
        return -1;
    tq_prepare_resident_indexed_kernel<<<(uint64_t(slots)+127)/128,128,0,stream>>>(gate,up,down,ids,
        gate_up,output_jobs,slots,routes,experts,input,hidden,output,tile);
    return int(cudaGetLastError());
}

template<int DT>
__global__ __launch_bounds__(256)
void tq_routed_gu_kernel(const TrellisProjectionView* gate,const TrellisProjectionView* up,
        const int64_t* ids,const Scalar<DT>* x,Scalar<DT>* hidden,float limit,int stride) {
    const int warp=threadIdx.x/32,role=warp&1,slot=warp/2;
    const auto expert=ids[blockIdx.y];
    const auto m=role?up[expert]:gate[expert];
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

template<int DT>
__global__ __launch_bounds__(128)
void tq_routed_down_kernel(const TrellisProjectionView* weights,const int64_t* ids,
        const Scalar<DT>* hidden,Scalar<DT>* output,int xs,int ys) {
    const auto m=weights[ids[blockIdx.y]];
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
