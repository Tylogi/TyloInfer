#include "runtime/layer_weight_stream.h"
#include "mfq_cuda_context.h"
#include <array>
#include <iostream>
#include <stdexcept>
namespace tb=mfq_tensor_backend;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main()try{
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    const MfqCudaGuard guard(0);const auto context=mfq::cuda::default_context(0);
    context->begin_graph_pool(mfq_current_cuda_stream());
    struct End{std::shared_ptr<mfq::cuda::Context> c;~End(){c->end_graph_pool(mfq_current_cuda_stream());}}end{context};
    std::array<std::array<tb::Tensor,2>,5> weights,expected;
    mfq::cuda::LayerWeightStream pipeline(0);std::size_t maximum=0,host_bytes=0;
    for(int layer=0;layer<5;++layer){
        weights[layer][0]=tb::full({layer==0?1048579:32771+layer*333},layer+1.25,tb::TensorOptions().dtype(tb::kFloat32));
        weights[layer][1]=tb::full({17+layer},layer+11,tb::TensorOptions().dtype(tb::kUInt8));
        for(int field=0;field<2;++field)expected[layer][field]=weights[layer][field].clone();
        const auto bytes=((weights[layer][0].nbytes()+255)&~std::size_t(255))+weights[layer][1].nbytes();
        maximum=std::max(maximum,bytes);host_bytes+=bytes;
        pipeline.add(&weights[layer],{{&weights[layer][0],{}},{&weights[layer][1],{}}});
    }
    pipeline.prepare();require(pipeline.device_bytes()==2*maximum,"weight streaming did not use two bounded layer buffers");
    require(pipeline.host_bytes()==host_bytes,"weight streaming retained duplicate host images");
    for(int repeat=0;repeat<9;++repeat){
        std::array<std::array<tb::Tensor,2>,5> observed;
        const std::array<int,5> forward{0,1,2,3,4},reverse{4,3,2,1,0};
        for(int layer:repeat%2?reverse:forward){
            auto lease=pipeline.enter(&weights[layer]);
            bool nested=false;try{auto invalid=pipeline.enter(&weights[layer]);}catch(const std::logic_error&){nested=true;}
            require(nested,"nested layer leases were accepted");
            for(int field=0;field<2;++field){
                require(weights[layer][field].is_cuda(),"streamed weights did not move to the GPU");
                observed[layer][field]=weights[layer][field].clone();
                if(field==0 && repeat%3)mfq::cuda::prefetch_active_layer_weights();
            }
        }
        for(int layer=0;layer<5;++layer)for(int field=0;field<2;++field){
            auto value=observed[layer][field].cpu();
            require(std::memcmp(value.data_ptr(),expected[layer][field].data_ptr(),value.nbytes())==0,
                "a GPU layer buffer was overwritten before its consumer completed");
        }
    }
    try{auto lease=pipeline.enter(&weights[0]);auto consume=weights[0][0].clone();throw 1;}catch(int){}
    {auto lease=pipeline.enter(&weights[4]);auto consume=weights[4][0].clone();
        require(consume.cpu().data_ptr<float>()[0]==5.25f,"exception recovery changed streamed weights");}
    std::cout<<"double-buffer layers: mixed dtypes, repeated/partial/reversed windows, exception recovery PASS\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
