#pragma once
#include "float_projection.h"
#include "quant_linear.h"
#include "runtime/decode_window.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mfq::cuda::test {
inline mfq_tensor_backend::Tensor f32_projection_matrix(int rows,int columns,int seed,int shift=0) {
    namespace tb=mfq_tensor_backend;
    std::vector<float> values(std::size_t(rows)*columns+shift);
    for(std::size_t i=0;i<values.size();++i)values[i]=float(std::sin(i*.173+seed)*.13);
    return tb::tensor(values).to(tb::Device(tb::kCUDA,mfq_current_cuda_device()))
        .narrow(0,shift,std::int64_t(rows)*columns).reshape({rows,columns});
}

inline void check_dense_f32_windows() {
    namespace tb=mfq_tensor_backend;
    const auto context=default_context(mfq_current_cuda_device());const auto stream=mfq_current_cuda_stream();
    context->begin_graph_pool(stream);
    struct End {std::shared_ptr<Context> context;cudaStream_t stream;~End(){context->end_graph_pool(stream);}} end_pool{context,stream};
    int cases=0,graphs=0;double worst=0;
    const auto check=[&](const tb::Tensor& input,const tb::Tensor& weight,const tb::Tensor& output) {
        if(!output.defined() || output.scalar_type()!=tb::kFloat32 ||
                output.sizes().vec()!=std::vector<std::int64_t>{input.size(0),weight.size(0)})
            throw std::runtime_error("FP32 window projection rejected its geometry or changed shape/precision");
        const auto actual=output.cpu(),x=input.cpu(),w=weight.cpu();
        for(int row=0;row<input.size(0);++row) {
            const auto single=try_dense_f32_projection_cuda(input.narrow(0,row,1),weight).cpu();
            if(std::memcmp(actual.data_ptr<float>()+row*weight.size(0),single.data_ptr(),single.nbytes()))
                throw std::runtime_error("FP32 projection changed a token when grouped with other tokens");
            for(int column:{0,int(weight.size(0)/2),int(weight.size(0)-1)}) {
                double expected=0,absolute=0;
                for(int k=0;k<input.size(1);++k) {
                    const double term=double(x.data_ptr<float>()[row*input.size(1)+k])*
                        w.data_ptr<float>()[column*input.size(1)+k];
                    expected+=term;absolute+=std::abs(term);
                }
                const double value=actual.data_ptr<float>()[row*weight.size(0)+column];
                const double scaled=std::abs(value-expected)/std::max(absolute,1e-30);
                if(!std::isfinite(value) || scaled>2e-6)
                    throw std::runtime_error("FP32 window projection exceeded its independent FP64 rounding bound");
                worst=std::max(worst,scaled);
            }
        }
        ++cases;
    };
    for(int width:{1,33,128,511,5120})for(int outputs:{1,17,128,512,641})for(int shift:{0,1}) {
        auto weight=f32_projection_matrix(outputs,width,3,shift);
        for(int rows=1;rows<=8;++rows) {
            auto input=f32_projection_matrix(rows,width,7,shift);
            check(input,weight,dense_projection(input,weight));
            if((width==33 && outputs==17 && rows==3 && shift==1) ||
                    (width==5120 && outputs==641 && rows==4 && !shift)) {
                tb::Tensor result;DecodeWindow graph(mfq_current_cuda_stream());
                graph.capture([&]{result=dense_projection(input,weight);},[]{});
                for(float value:{0.f,-.03125f,.0625f}) {
                    input.fill_(value);weight.fill_(value*.5f);graph.run();check(input,weight,result);++graphs;
                }
            }
        }
    }
    for(const auto dtype:{tb::kFloat16,tb::kBFloat16})for(int rows=1;rows<=8;++rows) {
        auto input=f32_projection_matrix(rows,257,11).to(dtype),weight=f32_projection_matrix(513,257,13);
        check(input.to(tb::kFloat32),weight,dense_projection(input,weight));
    }
    auto input=f32_projection_matrix(9,33,13),weight=f32_projection_matrix(17,33,17);
    if(try_dense_f32_projection_cuda(input,weight).defined() ||
            try_dense_f32_projection_cuda(input.narrow(0,0,4).to(tb::kFloat16),weight).defined() ||
            try_dense_f32_projection_cuda(input.narrow(0,0,4),weight.cpu()).defined())
        throw std::runtime_error("FP32 window projection accepted unsupported rows or storage");
    std::cout<<"FP32 projection windows: "<<cases<<" FP64/singleton exact checks, "<<graphs
        <<" changing graphs, aligned/unaligned tails, M1-8; worst_scaled_error="<<worst<<" PASS\n";
}

inline void benchmark_dense_f32_windows(int rows,int outputs,int width,int iterations) {
    namespace tb=mfq_tensor_backend;
    if(rows<1 || rows>8 || outputs<1 || width<1 || iterations<1)
        throw std::invalid_argument("FP32 projection benchmark requires positive dimensions and 1..8 rows");
    const auto input=f32_projection_matrix(rows,width,19),weight=f32_projection_matrix(outputs,width,23);
    const auto context=default_context(mfq_current_cuda_device());const auto stream=mfq_current_cuda_stream();
    context->begin_graph_pool(stream);
    struct End {std::shared_ptr<Context> context;cudaStream_t stream;~End(){context->end_graph_pool(stream);}} end_pool{context,stream};
    std::array<tb::Tensor,2> result;DecodeWindow before(stream),after(stream);
    before.capture([&] {
        if(rows==1 || rows>6)result[0]=tb::matmul(input,weight.transpose(0,1));
        else {
            std::vector<tb::Tensor> separate;
            for(int row=0;row<rows;++row)separate.push_back(tb::matmul(input.narrow(0,row,1),weight.transpose(0,1)));
            result[0]=tb::cat(separate,0);
        }
    },[]{});
    after.capture([&] {
        result[1]=try_dense_f32_projection_cuda(input,weight);
        if(!result[1].defined())throw std::runtime_error("FP32 benchmark missed its native projection");
    },[]{});
    std::array<std::vector<float>,2> times;
    for(int trial=0;trial<5;++trial)for(int order=0;order<2;++order) {
        const int mode=(trial+order)%2;auto& graph=mode?after:before;
        for(int i=0;i<16;++i)graph.enqueue_if_device_only();
        Event begin(cudaEventDefault),end(cudaEventDefault);begin.record(stream);
        for(int i=0;i<iterations;++i)if(!graph.enqueue_if_device_only())throw std::runtime_error("FP32 benchmark has a host task");
        end.record(stream);end.synchronize();float ms=0;
        TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));times[mode].push_back(ms*1000/iterations);
    }
    for(auto& values:times)std::sort(values.begin(),values.end());
    const auto a=result[0].cpu(),b=result[1].cpu();double error=0,energy=0;
    for(int64_t i=0;i<a.numel();++i) {
        const double x=a.data_ptr<float>()[i],y=b.data_ptr<float>()[i];
        error+=(x-y)*(x-y);energy+=x*x;
    }
    if(error>energy*1e-10+1e-12)throw std::runtime_error("FP32 native projection differs excessively from BLAS");
    std::cout<<"FP32 projection M="<<rows<<" N="<<outputs<<" K="<<width
        <<" blas_reference_us="<<times[0][2]<<" native_us="<<times[1][2]<<" actual_match=1\n";
}
} // namespace mfq::cuda::test
