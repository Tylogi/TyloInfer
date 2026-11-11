#include "engine/backends/cuda/kernels/mfq_cuda_attention_ops.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/executor/decode_window.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>


struct Scratch {
    std::shared_ptr<tyloi::cuda::Context> context=tyloi::cuda::default_context(mfq_current_cuda_device());
    cudaStream_t stream=mfq_current_cuda_stream();
    Scratch(){context->begin_graph_pool(stream);}
    ~Scratch(){context->end_graph_pool(stream);}
};
int main()try {
    std::cout.setf(std::ios::unitbuf);
    std::cout<<"GQA MMA test entry\n";
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,0);
    // The existing FP16 MMA instruction path requires Volta or newer.
    if(prop.major<7)return 77;
    const auto options=::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kFloat32);
    int cases=0;
    for(int width:{256,512})for(int ratio:{1,2,3,4,6,8,12})for(int length:{1,65,4097}) {
        constexpr int batch=2,kv_heads=2;
        const int heads=ratio*kv_heads,capacity=(length+127)/64*64;
        auto host_q=::tyloi::cuda::empty({batch,heads,1,width},options.device(::tyloi::kCPU));
        auto host_k=::tyloi::cuda::empty({batch,kv_heads,capacity,width},host_q.options()),host_v=::tyloi::cuda::empty_like(host_k);
        for(int64_t i=0;i<host_q.numel();++i)host_q.data_ptr<float>()[i]=std::sin(float(i)*.317f)*.7f;
        for(int64_t i=0;i<host_k.numel();++i){host_k.data_ptr<float>()[i]=std::sin(float(i)*.119f)*.5f;host_v.data_ptr<float>()[i]=std::cos(float(i)*.071f)*.3f;}
        auto qh=host_q.to(::tyloi::kCUDA,::tyloi::kFloat16),q=qh.to(::tyloi::kFloat32);
        auto k=host_k.to(::tyloi::kCUDA,::tyloi::kFloat16),v=host_v.to(::tyloi::kCUDA,::tyloi::kFloat16);
        auto lens=::tyloi::cuda::tensor(std::vector<int64_t>{length,std::max(1,length-31)},options.dtype(::tyloi::kInt64));
        auto mask=::tyloi::cuda::empty({batch,capacity},options.dtype(::tyloi::kFloat16)),kv_max=::tyloi::cuda::empty({batch},options.dtype(::tyloi::kInt32));
        auto meta=::tyloi::cuda::empty({int64_t(batch)*kv_heads*((ratio+7)/8)*((length+63)/64)*8*(2+width/2)*2},options);
        auto po=::tyloi::cuda::empty({batch*heads,64,width},options),pm=::tyloi::cuda::empty({batch*heads,64},options),pl=::tyloi::cuda::empty_like(pm);
        auto reference=attention_cache_decode_split_cuda(qh,k,v,lens,1./16,po,pm,pl,32).to(::tyloi::kCPU,::tyloi::kFloat32).contiguous();
        ::tyloi::DenseTensor output;
        auto body=[&]{output=width==256
            ?mfq_attention_mma256_decode_cuda(q,k,v,lens,1./16,length,mask,kv_max,meta)
            :mfq_attention_mma512_decode_cuda(q,k,v,lens,1./16,length,mask,kv_max,meta);};
        Scratch scratch;tyloi::cuda::DecodeWindow graph(mfq_current_cuda_stream());
        body();graph.capture(body,[]{},[&]{output={};});graph.run();
        auto actual=output.to(::tyloi::kCPU,::tyloi::kFloat32).contiguous();
        float worst=0;
        for(int64_t i=0;i<actual.numel();++i) {
            const float error=std::abs(actual.data_ptr<float>()[i]-reference.data_ptr<float>()[i]);
            if(!std::isfinite(error))throw std::runtime_error("nonfinite GQA result");worst=std::max(worst,error);
        }
        if(worst>3.e-4f){std::cerr<<"ratio="<<ratio<<" length="<<length<<" max_error="<<worst<<'\n';throw std::runtime_error("masked GQA MMA differs from reference");}
        ++cases;
    }
    // The shared-K/V MLA path uses the same logical-length contract and a
    // different value width. Compare against its existing masked cached path.
    {
        constexpr int batch=1,heads=64,capacity=256,query_width=576;
        auto host_q=::tyloi::cuda::empty({batch,heads,1,query_width},options.device(::tyloi::kCPU));
        auto host_kv=::tyloi::cuda::empty({batch,1,capacity,query_width},host_q.options());
        for(int64_t i=0;i<host_q.numel();++i)host_q.data_ptr<float>()[i]=std::sin(float(i)*.317f)*.7f;
        for(int64_t i=0;i<host_kv.numel();++i)host_kv.data_ptr<float>()[i]=std::cos(float(i)*.119f)*.3f;
        auto q=host_q.to(::tyloi::kCUDA,::tyloi::kFloat16).to(::tyloi::kFloat32),kv=host_kv.to(::tyloi::kCUDA,::tyloi::kFloat16);
        auto lens=::tyloi::cuda::ones({batch},options.dtype(::tyloi::kInt64));
        auto mask=::tyloi::cuda::empty({batch,capacity},options.dtype(::tyloi::kFloat16)),kv_max=::tyloi::cuda::empty({batch},options.dtype(::tyloi::kInt32));
        auto meta=::tyloi::cuda::empty({8*1024*1024},options),reference_meta=::tyloi::cuda::empty_like(meta);
        Scratch scratch;::tyloi::DenseTensor output;tyloi::cuda::DecodeWindow graph(mfq_current_cuda_stream());
        graph.capture([&]{output=attention_glm_mla576_decode_cuda(q,kv,lens,1./16,capacity,mask,kv_max,meta);},[]{},[&]{output={};});
        for(int length:{1,7,31,32,33,63,64,65,129}) {
            lens.fill_(length);graph.run();auto actual=output.to(::tyloi::kCPU,::tyloi::kFloat32).contiguous();
            auto expected=attention_glm_mla576_cached_cuda(q,kv,length,mask,kv_max,reference_meta,1./16)
                .to(::tyloi::kCPU,::tyloi::kFloat32).contiguous();
            if(actual.sizes()!=expected.sizes())throw std::runtime_error("MLA prefix shape differs");
            for(int64_t i=0;i<actual.numel();++i) {
                const float error=std::abs(actual.data_ptr<float>()[i]-expected.data_ptr<float>()[i]);
                if(!std::isfinite(error)||error>3.e-4f)throw std::runtime_error("MLA logical prefix differs from masked reference");
            }
            ++cases;
        }
    }
    std::cout<<"GQA MMA partial query tiles, multi-batch lengths, CUDA Graph: "<<cases<<" cases PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
