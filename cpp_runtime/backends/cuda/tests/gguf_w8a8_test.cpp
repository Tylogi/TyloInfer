#include "../kernels/gguf_w8a8.h"
#include "../kernels/gguf.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace w8=mfq::cuda::gguf::w8a8;
void check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
template<class T>struct Device {
    T* data=nullptr;
    explicit Device(size_t count){check(cudaMalloc(&data,std::max(size_t(1),count)*sizeof(T)));}
    ~Device(){cudaFree(data);}
    Device(const Device&)=delete;
};

void run(int tokens,int rows,int columns,bool half_input,bool half_output,bool with_residual) {
    const auto padded=w8::padded_columns(columns);
    const int source_stride=columns+3,output_stride=rows+7;
    std::vector<float> x(tokens*source_stride),w(rows*source_stride);
    std::mt19937 generator(17);
    std::normal_distribution<float> normal;
    for(auto& value:x)value=normal(generator)*.3f;
    for(auto& value:w)value=normal(generator)*.2f;
    std::fill_n(x.data(),columns,0.f);
    std::vector<half> hx(x.size()),hw(w.size());
    if(half_input) {
        for(size_t i=0;i<x.size();++i){hx[i]=__float2half(x[i]);x[i]=__half2float(hx[i]);}
        for(size_t i=0;i<w.size();++i){hw[i]=__float2half(w[i]);w[i]=__half2float(hw[i]);}
    }
    Device<float> dx(x.size()),dw(w.size()),sx(tokens),sw(rows),out(tokens*output_stride);
    Device<half> dhx(x.size()),dhw(w.size()),hout(tokens*output_stride);
    Device<int8_t> qx(tokens*padded),qw(rows*padded);
    Device<int32_t> accumulator(tokens*rows);
    std::vector<half> residual(tokens*output_stride);
    for(auto& value:residual)value=__float2half(normal(generator)*.1f);
    Device<half> dr(residual.size());
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaMemcpyAsync(dx.data,x.data(),x.size()*4,cudaMemcpyHostToDevice,stream));
    check(cudaMemcpyAsync(dw.data,w.data(),w.size()*4,cudaMemcpyHostToDevice,stream));
    check(cudaMemcpyAsync(dr.data,residual.data(),residual.size()*2,cudaMemcpyHostToDevice,stream));
    if(half_input) {
        check(cudaMemcpyAsync(dhx.data,hx.data(),hx.size()*2,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(dhw.data,hw.data(),hw.size()*2,cudaMemcpyHostToDevice,stream));
    }
    auto plan=w8::prepare(tokens,rows,columns,32ULL<<20,stream);
    if(!plan)throw std::runtime_error("No native INT8 Tensor Core plan for the test geometry");
    Device<uint8_t> scratch(plan.workspace_bytes);
    const auto invoke=[&] {
        w8::pack(half_input?static_cast<void*>(dhx.data):dx.data,half_input,tokens,columns,source_stride,qx.data,sx.data,stream);
        w8::pack(half_input?static_cast<void*>(dhw.data):dw.data,half_input,rows,columns,source_stride,qw.data,sw.data,stream);
        w8::multiply(plan,qx.data,qw.data,accumulator.data,scratch.data,plan.workspace_bytes,stream);
        w8::restore(accumulator.data,sx.data,sw.data,half_output?static_cast<void*>(hout.data):out.data,
                    half_output,tokens,rows,output_stride,stream,with_residual?dr.data:nullptr);
    };
    invoke();
    check(cudaStreamSynchronize(stream));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    invoke();
    check(cudaStreamEndCapture(stream,&graph));
    check(cudaGraphInstantiate(&exec,graph,0));
    // Replaying after an operand changes catches captured host staging or
    // stale activation packets without requiring a second graph capture.
    for(int changed=0;changed<2;++changed) {
        if(changed) {
            x[source_stride+columns/2]+=.25f;
            if(half_input) {
                hx[source_stride+columns/2]=__float2half(x[source_stride+columns/2]);
                x[source_stride+columns/2]=__half2float(hx[source_stride+columns/2]);
                check(cudaMemcpyAsync(dhx.data,hx.data(),hx.size()*2,cudaMemcpyHostToDevice,stream));
            } else check(cudaMemcpyAsync(dx.data,x.data(),x.size()*4,cudaMemcpyHostToDevice,stream));
        }
        check(cudaGraphLaunch(exec,stream));
        check(cudaStreamSynchronize(stream));
        std::vector<int8_t> ix(tokens*padded),iw(rows*padded);
        std::vector<int32_t> integer(tokens*rows);
        std::vector<float> actual(tokens*output_stride);
        check(cudaMemcpy(ix.data(),qx.data,ix.size(),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(iw.data(),qw.data,iw.size(),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(integer.data(),accumulator.data,integer.size()*4,cudaMemcpyDeviceToHost));
        if(half_output) {
            std::vector<half> values(actual.size());
            check(cudaMemcpy(values.data(),hout.data,values.size()*2,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<actual.size();++i)actual[i]=__half2float(values[i]);
        } else check(cudaMemcpy(actual.data(),out.data,actual.size()*4,cudaMemcpyDeviceToHost));
        double error=0,energy=0;
        for(int t=0;t<tokens;++t)for(int r=0;r<rows;++r) {
            int64_t dot=0;
            for(int k=0;k<padded;++k)dot+=int(ix[t*padded+k])*int(iw[r*padded+k]);
            if(dot!=integer[t*rows+r])throw std::runtime_error("Native INT8/INT32 GEMM disagrees with exact integer oracle");
            double reference=0;
            for(int k=0;k<columns;++k)reference+=double(x[t*source_stride+k])*double(w[r*source_stride+k]);
            if(with_residual)reference+=__half2float(residual[t*output_stride+r]);
            const double delta=actual[t*output_stride+r]-reference;
            error+=delta*delta;
            energy+=reference*reference;
            if(t==0 && actual[r]!=(with_residual?__half2float(residual[r]):0.f))
                throw std::runtime_error("Zero activation row did not preserve the residual");
        }
        const double nmse=error/std::max(energy,1.e-30);
        std::printf("W8A8 check M=%d N=%d K=%d input_half=%d output_half=%d residual=%d changed=%d integer_exact=1 nmse=%.9g\n",
                    tokens,rows,columns,int(half_input),int(half_output),int(with_residual),changed,nmse);
        if(!std::isfinite(nmse) || nmse>5.e-4) {
            std::vector<float> ax(tokens),aw(rows);
            check(cudaMemcpy(ax.data(),sx.data,ax.size()*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(aw.data(),sw.data,aw.size()*4,cudaMemcpyDeviceToHost));
            std::fprintf(stderr,"diagnostic int=%d scale_x=%.9g scale_w=%.9g actual=%.9g energy=%.9g\n",
                         integer[rows],ax[1],aw[0],actual[output_stride],energy);
            throw std::runtime_error("Rotated W8A8 projection exceeds NMSE bound");
        }
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

void check_gated_pack() {
    int cases=0;
    for(bool half_input:{false,true})for(bool silu:{false,true})for(int columns:{31,65,131,256,5120,5137,17408,28673}) {
        constexpr int rows=7;const auto padded=w8::padded_columns(columns);
        std::vector<float> input(rows*columns),gate(input.size());
        std::vector<half> hx(input.size()),hg(input.size());
        for(size_t i=0;i<input.size();++i){input[i]=3.f*std::sin(float(i)*.071f);gate[i]=32.f*std::cos(float(i)*.013f);}
        Device<float> x(input.size()),g(input.size()),expected(input.size()),actual(input.size()),scales(rows),new_scales(rows);
        Device<half> xh(input.size()),gh(input.size()),expected_half(input.size()),actual_half(input.size());
        Device<int8_t> codes(rows*padded),new_codes(rows*padded);
        cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        const auto up=half_input?static_cast<void*>(xh.data):x.data;
        const auto gates=half_input?static_cast<void*>(gh.data):g.data;
        const auto reference=half_input?static_cast<void*>(expected_half.data):expected.data;
        const auto output=half_input?static_cast<void*>(actual_half.data):actual.data;
        const auto invoke=[&]{w8::pack_gated(up,gates,half_input,silu,rows,columns,output,new_codes.data,new_scales.data,stream);};
        cudaGraph_t graph;cudaGraphExec_t executable;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));invoke();
        check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
        for(int changed=0;changed<2;++changed) {
            input[columns/2]+=.3f;
            if(half_input) {
                for(size_t i=0;i<input.size();++i){hx[i]=__float2half(input[i]);hg[i]=__float2half(gate[i]);}
                check(cudaMemcpyAsync(up,hx.data(),hx.size()*2,cudaMemcpyHostToDevice,stream));
                check(cudaMemcpyAsync(gates,hg.data(),hg.size()*2,cudaMemcpyHostToDevice,stream));
            } else {
                check(cudaMemcpyAsync(up,input.data(),input.size()*4,cudaMemcpyHostToDevice,stream));
                check(cudaMemcpyAsync(gates,gate.data(),gate.size()*4,cudaMemcpyHostToDevice,stream));
            }
            mfq::cuda::gguf::materialize_gated_input(up,gates,half_input,input.size(),silu,reference,stream);
            w8::pack(reference,half_input,rows,columns,columns,codes.data,scales.data,stream);
            check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
            const auto same=[&](const void* a,const void* b,size_t bytes) {
                std::vector<uint8_t> left(bytes),right(bytes);
                check(cudaMemcpy(left.data(),a,bytes,cudaMemcpyDeviceToHost));check(cudaMemcpy(right.data(),b,bytes,cudaMemcpyDeviceToHost));
                if(std::memcmp(left.data(),right.data(),bytes))throw std::runtime_error("Gated W8A8 fusion changed operands or dtype rounding");
            };
            same(reference,output,input.size()*(half_input?2:4));same(codes.data,new_codes.data,rows*padded);same(scales.data,new_scales.data,rows*4);
            ++cases;
        }
        check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
    }
    std::printf("Gated W8A8 exact codes/scales/materialization, FP16/FP32, sigmoid/SiLU, long/tail rows and changing Graph inputs PASS cases=%d\n",cases);
}
int main()try {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices){std::puts("SKIP: CUDA unavailable");return 77;}
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties,0));
    // Integer Tensor Cores start with the Turing instruction set.
    if(properties.major<7 || (properties.major==7 && properties.minor<5)) {
        std::puts("SKIP: native INT8 Tensor Cores unavailable");return 77;
    }
    for(bool input:{false,true})for(bool output:{false,true})
        for(bool residual:{false,true})for(int columns:{31,65,256,5120})run(5,12,columns,input,output,residual);
    check_gated_pack();
    bool overflow_rejected=false;
    try{(void)w8::padded_columns(INT32_MAX);}catch(const std::invalid_argument&){overflow_rejected=true;}
    if(!overflow_rejected)throw std::runtime_error("INT32 overflow geometry was accepted");
    std::puts("W8A8 rotation, integer oracle, strided tails, zero rows and CUDA Graph checks PASS");
    return 0;
}catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
