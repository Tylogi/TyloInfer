#pragma once
#include "engine/backends/cuda/executor/recurrent_copy_batch.h"
#include <chrono>
#include <cstring>
#include <iostream>

inline void check_recurrent_copy_batch() {

    using tyloi::cuda::RecurrentCopyBatch;
    using tyloi::cuda::copy_recurrent_state;
    const auto context=tyloi::cuda::default_context(0);
    const auto stream=mfq_get_stream_from_pool();
    const MfqCudaStreamGuard stream_guard(stream);
    const auto options=::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kFloat32);
    const auto check=[](const ::tyloi::DenseTensor& a,const ::tyloi::DenseTensor& b) {
        const auto x=a.contiguous().cpu(),y=b.contiguous().cpu();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() ||
                std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes()))
            throw std::runtime_error("batched recurrent state differs");
    };
    for(bool enabled:{false,true}) {
        std::vector<::tyloi::DenseTensor> results,expected;
        std::size_t submitted=0;
        {
            RecurrentCopyBatch copies(enabled);
            // Cross the packet boundary, with byte and element tails. Sources
            // leave their local scope before submission to check ownership.
            for(int i=0;i<137;++i) {
                auto source=::tyloi::cuda::full({7+i*3},float(i-63),options);
                auto result=::tyloi::cuda::empty_like(source);
                expected.push_back(source.cpu());
                copy_recurrent_state(result,source);results.push_back(result);
            }
            copies.flush();submitted=copies.batched_copies();
        }
        for(std::size_t i=0;i<results.size();++i)check(results[i],expected[i]);
        const auto* option=tyloi::cuda::env_lookup("TYLOINFER_RECURRENT_COPY_BATCH");
        if(enabled && (!option || option[0]!='0') && submitted!=results.size())
            throw std::runtime_error("recurrent copy batch was not dispatched");
        auto a=::tyloi::cuda::full({32},3.f,options),b=::tyloi::cuda::zeros_like(a),c=::tyloi::cuda::zeros_like(a);
        {
            RecurrentCopyBatch copies(enabled);
            copy_recurrent_state(b,a);copy_recurrent_state(c,b);
            copies.flush();
        }
        check(c,a);
        auto destination=::tyloi::cuda::zeros({24},options),ones=::tyloi::cuda::ones({16},options),twos=::tyloi::cuda::full({16},2.f,options);
        auto first=destination.narrow(0,0,16),last=destination.narrow(0,8,16);
        {
            RecurrentCopyBatch copies(enabled);
            copy_recurrent_state(first,ones);copy_recurrent_state(last,twos);copies.flush();
        }
        check(destination.narrow(0,0,8),ones.narrow(0,0,8));check(last,twos);
        // A noncontiguous/converting copy flushes queued producers first.
        auto matrix=::tyloi::cuda::zeros({4,8},options),view=matrix.transpose(0,1);
        auto converted=::tyloi::cuda::empty({32},options.dtype(::tyloi::kFloat16));
        auto host=::tyloi::cuda::empty({32},options.device(::tyloi::kCPU));
        {
            RecurrentCopyBatch copies(enabled);
            copy_recurrent_state(b,a);copy_recurrent_state(view,b.reshape({8,4}));
            copy_recurrent_state(converted,b);copy_recurrent_state(host,b);copies.flush();
        }
        check(view,b.reshape({8,4}));check(converted.to(::tyloi::kFloat32),a);check(host,a);
        // Nested scopes submit the outer packet before an inner consumer.
        {
            RecurrentCopyBatch outer(enabled);copy_recurrent_state(b,a);
            {RecurrentCopyBatch inner(enabled);copy_recurrent_state(c,b);inner.flush();}
            outer.flush();
        }
        check(c,a);
        for(int offset:{0,1,3})for(int bytes:{65535,65536,65537,2097159}) {
            auto host_source=::tyloi::cuda::empty({bytes+offset},options.device(::tyloi::kCPU).dtype(::tyloi::kUInt8));
            for(int64_t i=0;i<host_source.numel();++i)host_source.data_ptr<std::uint8_t>()[i]=std::uint8_t(i%251);
            auto source=host_source.to(::tyloi::kCUDA).narrow(0,offset,bytes);
            auto storage=::tyloi::cuda::full({bytes+offset+7},253,source.options());
            auto target=storage.narrow(0,offset,bytes);
            {
                RecurrentCopyBatch copies(enabled);copy_recurrent_state(target,source);copies.flush();
            }
            check(target,source);
            const auto result=storage.cpu();
            for(int i=0;i<offset;++i)if(result.data_ptr<std::uint8_t>()[i]!=253)
                throw std::runtime_error("state copy wrote before its view");
            for(int64_t i=bytes+offset;i<result.numel();++i)if(result.data_ptr<std::uint8_t>()[i]!=253)
                throw std::runtime_error("state copy wrote beyond its tail");
        }
        cudaGraph_t graph=nullptr;cudaGraphExec_t execution=nullptr;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(stream.stream(),cudaStreamCaptureModeGlobal));
        {RecurrentCopyBatch copies(enabled);copy_recurrent_state(b,a);copies.flush();}
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamEndCapture(stream.stream(),&graph));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphInstantiate(&execution,graph,0));
        a.fill_(5.f);
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(execution,stream.stream()));
        check(b,a);
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphExecDestroy(execution));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphDestroy(graph));
    }
    std::cout<<"recurrent-copy ownership/dependencies/tails/conversion/host/nesting/CUDA-Graph PASS\n";
}

inline void bench_recurrent_copy_batch(int count,int large_bytes,int small_bytes) {

    const auto context=tyloi::cuda::default_context(0);
    const auto stream=mfq_get_stream_from_pool();
    const MfqCudaStreamGuard stream_guard(stream);
    const auto options=::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kUInt8);
    std::vector<::tyloi::DenseTensor> source,destination;
    for(int i=0;i<count;++i) {
        const int bytes=i%3==1?large_bytes:small_bytes;
        if(bytes<1)throw std::invalid_argument("copy benchmark bytes must be positive");
        source.push_back(::tyloi::cuda::ones({bytes},options));destination.push_back(::tyloi::cuda::empty_like(source.back()));
    }
    constexpr int iterations=80;
    for(bool batch:{false,true}) {
        const auto run=[&] {
            tyloi::cuda::RecurrentCopyBatch copies(batch);
            for(int i=0;i<count;++i)tyloi::cuda::copy_recurrent_state(destination[i],source[i]);
            copies.flush();return copies.batched_copies();
        };
        for(int i=0;i<4;++i)run();
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        const auto begin=std::chrono::steady_clock::now();
        std::size_t dispatched=0;
        for(int i=0;i<iterations;++i)dispatched+=run();
        const auto submitted=std::chrono::steady_clock::now();
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        const auto end=std::chrono::steady_clock::now();
        const auto us=[](auto t){return std::chrono::duration<double,std::micro>(t).count()/iterations;};
        std::cout<<"recurrent_copy_bench batch="<<batch<<" copies="<<count
            <<" enqueue_us="<<us(submitted-begin)<<" total_us="<<us(end-begin)
            <<" batched="<<dispatched/iterations<<'\n';
    }
}
