#include "channel_fp8.h"
#include "quant_linear.h"
#include "tyloinfer/channel_fp8.h"
#include "tyloinfer/hf_model_source.h"
#include "storage/weight_loader.h"
#include "runtime/decode_window.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace tb=mfq_tensor_backend;
namespace {
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
void close(tb::Tensor actual,tb::Tensor expected) {
    actual=actual.to(tb::kFloat32).cpu();expected=expected.to(tb::kFloat32).cpu();
    require(actual.sizes()==expected.sizes(),"FP8 result shape changed");
    for(std::int64_t i=0;i<actual.numel();++i) {
        const auto a=actual.data_ptr<float>()[i],b=expected.data_ptr<float>()[i];
        if(!std::isfinite(a) || std::abs(a-b)>std::max(.00003f,std::abs(b)*.008f))
            throw std::runtime_error("FP8 projection exceeded the CPU reference tolerance at element "+std::to_string(i)+": actual="+std::to_string(a)+" expected="+std::to_string(b));
    }
}
double worst_projection_relative_l2=0;
void close_projection(tb::Tensor actual,tb::Tensor expected,bool half_accum) {
    if(!half_accum){close(actual,expected);return;}
    actual=actual.to(tb::kFloat32).cpu();expected=expected.to(tb::kFloat32).cpu();
    require(actual.sizes()==expected.sizes(),"FP8 result shape changed");
    double error=0,norm=0;
    for(std::int64_t i=0;i<actual.numel();++i) {
        const double a=actual.data_ptr<float>()[i],b=expected.data_ptr<float>()[i];
        require(std::isfinite(a),"FP8 projection produced a nonfinite result");
        error+=(a-b)*(a-b);norm+=b*b;
    }
    const double relative_l2=std::sqrt(error/std::max(norm,1.e-30));
    worst_projection_relative_l2=std::max(worst_projection_relative_l2,relative_l2);
    // Use the same whole-projection FP16 accumulation bound as GGUF prefill;
    // per-element relative error is unstable when a dot product cancels to zero.
    require(relative_l2<=.01,"FP8 FP16 accumulation relative L2 error exceeds one percent");
}
std::vector<std::uint8_t> image(int rows,int width,bool dynamic,mfq::channel_fp8::ScaleKind kind) {
    namespace fp=mfq::channel_fp8;
    const auto layout=fp::layout(rows,width,kind,dynamic);const auto header=fp::prefix(layout);
    std::vector<std::uint8_t> bytes(layout.bytes);std::memcpy(bytes.data(),header.data(),header.size());
    for(int n=0;n<rows;++n) {
        for(int k=0;k<width;++k)bytes[fp::header_bytes+std::int64_t(n)*width+k]=
            std::uint8_t(((n*17+k*3)%126)|((k%3==0)?128:0));
        const float scale=n==0?0.f:float(n%7+1)/2048.f;
        auto* target=bytes.data()+layout.scales+n*fp::scale_bytes(kind);
        if(kind==fp::ScaleKind::F32)std::memcpy(target,&scale,4);
        else if(kind==fp::ScaleKind::F16) {const mfq_half value(scale);std::memcpy(target,&value,2);}
        else {std::uint32_t bits;std::memcpy(&bits,&scale,4);const auto value=std::uint16_t(bits>>16);std::memcpy(target,&value,2);}
    }
    return bytes;
}
void check_weight(const ChannelFp8Weight& cpu,bool graph_check) {
    auto device=cpu;device.values=cpu.values.to(tb::kCUDA);device.scales=cpu.scales.to(tb::kCUDA);
    CudaExecutionContext execution;QuantLinear layer;layer.kind=QuantLinearKind::ChannelFp8;layer.channel_fp8=device;
    require(layer.out()==cpu.out && layer.neuron_len()==cpu.neuron_len,"FP8 dimensions missed common dispatch");
    close(quant_linear_reference_weight(layer),cpu.dequant());
    for(const auto type:{tb::kFloat32,tb::kFloat16,tb::kBFloat16})for(int tokens:{1,2,4,8,9,33,129}) {
        auto input=tb::empty({1,tokens,cpu.neuron_len},tb::TensorOptions().dtype(tb::kFloat32));
        for(std::int64_t i=0;i<input.numel();++i)input.data_ptr<float>()[i]=float(std::sin(i*.113)*.5);
        input=input.to(type);const auto saved=input.clone();
        auto expected=cpu.forward(input);close(input,saved);
        auto x=input.to(tb::kCUDA),actual=layer.forward(execution,x);
        require(actual.scalar_type()==type,"FP8 output dtype changed");
        try{close_projection(actual,expected,cpu.dynamic_activations && tokens>8);}catch(const std::exception& error){
            throw std::runtime_error(std::string(error.what())+" M="+std::to_string(tokens)+" N="+std::to_string(cpu.out)+" K="+std::to_string(cpu.neuron_len));
        }
        if(cpu.dynamic_activations && tokens==33 && type==tb::kFloat16) {
            QuantLinearGroup group;group.layers={layer,layer};group.outs={int(cpu.out),int(cpu.out)};
            auto outputs=group.forward(execution,x);
            require(outputs.size()==2,"shared FP8 input changed the projection count");
            auto reference=actual.cpu();
            for(auto& part:outputs) {
                auto host=part.cpu();
                require(std::memcmp(host.data_ptr(),reference.data_ptr(),reference.nbytes())==0,
                    "shared FP8 input changed a projection result");
            }
            auto prepared=prepare_channel_fp8_input(x);
            bool rejected=false;
            try{device.forward(x.clone(),nullptr,&prepared);}catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"shared FP8 input accepted a different source allocation");
            close(input,saved);
        }
        if(graph_check && type==tb::kFloat32 && (tokens==4 || tokens==33)) {
            mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
            graph.capture([&]{actual=layer.forward(execution,x);},[]{});
            for(float value:{0.f,.0625f,-.125f}) {
                x.fill_(value);input.fill_(value);graph.run();close_projection(actual,cpu.forward(input),cpu.dynamic_activations && tokens>8);
            }
        }
    }
    auto ids=tb::tensor(std::vector<std::int64_t>{cpu.out-1,0,0});
    close(quant_embedding_lookup(layer,ids.to(tb::kCUDA)),cpu.rows(ids));
}
void check_activation_rounding() {
    constexpr int width=1024;
    ChannelFp8Weight cpu;cpu.out=cpu.neuron_len=width;cpu.dynamic_activations=true;
    cpu.values=tb::zeros({width,width},tb::TensorOptions().dtype(tb::kUInt8));
    cpu.scales=tb::full({width},1,tb::TensorOptions().dtype(tb::kFloat32));
    for(int i=0;i<width;++i)cpu.values.data_ptr<unsigned char>()[i*width+i]=56;
    auto input=tb::zeros({1,width},tb::TensorOptions().dtype(tb::kFloat32));
    int cursor=0;
    for(int code=0;code<126;++code)for(float sign:{1.f,-1.f}) {
        const float midpoint=(mfq::channel_fp8::decode(code)+mfq::channel_fp8::decode(code+1))*.5f;
        for(float value:{std::nextafter(midpoint,0.f),midpoint,std::nextafter(midpoint,448.f)})
            input.data_ptr<float>()[cursor++]=sign*value;
    }
    input.data_ptr<float>()[width-1]=448.f;
    auto gpu=cpu;gpu.values=cpu.values.to(tb::kCUDA);gpu.scales=cpu.scales.to(tb::kCUDA);
    auto expected=cpu.forward(input),actual=gpu.forward(input.to(tb::kCUDA)).cpu();
    require(std::memcmp(expected.data_ptr(),actual.data_ptr(),expected.nbytes())==0,
        "FP8 activation rounding changed a midpoint or adjacent FP32 value");
}
void check_source(const std::filesystem::path& root) {
    mfq::HfModelSource source(root);CudaExecutionContext execution;int count=0;
    for(const auto& tensor:source.tensors())if(tensor.dtype==mfq::channel_fp8::dtype) {
        const auto bytes=source.read(tensor.name);
        const std::vector<std::uint8_t> payload(reinterpret_cast<const std::uint8_t*>(bytes.data()),
            reinterpret_cast<const std::uint8_t*>(bytes.data())+bytes.size());
        auto cpu=load_channel_fp8(payload,false,-1);
        auto layer=load_quant_linear(execution,source,tensor.name);
        require(layer.is_channel_fp8() && layer.channel_fp8.values.element_size()==1 &&
            layer.channel_fp8.values.numel()==cpu.out*cpu.neuron_len,"HF FP8 expanded its persistent weight");
        const auto stop=mfq::channel_fp8::header_bytes+3;
        const auto partial=source.read(tensor.name,stop-5,13);
        require(std::equal(partial.begin(),partial.end(),bytes.begin()+stop-5),"FP8 segmented read changed bytes");
        check_weight(cpu,false);++count;
        for(int tokens:{3,33}) {
            const auto prefix=root/(tensor.name+"-m"+std::to_string(tokens));
            if(std::filesystem::exists(prefix.string()+".input.f32")) {
                auto input=tb::empty({tokens,cpu.neuron_len},tb::TensorOptions().dtype(tb::kFloat32));
                auto expected=tb::empty({tokens,cpu.out},input.options());
                std::ifstream in(prefix.string()+".input.f32",std::ios::binary),ref(prefix.string()+".expected.f32",std::ios::binary);
                require(bool(in.read(reinterpret_cast<char*>(input.data_ptr()),input.nbytes())) &&
                    bool(ref.read(reinterpret_cast<char*>(expected.data_ptr()),expected.nbytes())),"external FP8 oracle is truncated");
                close_projection(layer.forward(execution,input.to(tb::kCUDA)),expected,cpu.dynamic_activations && tokens>8);
                std::cout<<"independent FP8 oracle M="<<tokens<<" PASS\n";
            }
        }
        std::cout<<"HF FP8 projection "<<tensor.name<<" "<<cpu.out<<"x"<<cpu.neuron_len<<" PASS\n";
    }
    require(count>0,"HF source contains no channel FP8 projections");
}
void source_fixture(const std::filesystem::path& root) {
    using json=nlohmann::json;
    namespace fp=mfq::channel_fp8;
    std::filesystem::create_directories(root);
    const json scheme{{"num_bits",8},{"type","float"},{"strategy","channel"},{"dynamic",false},{"symmetric",true}};
    const json input{{"num_bits",8},{"type","float"},{"strategy","token"},{"dynamic",true},{"symmetric",true}};
    json quant{{"quant_method","compressed-tensors"},{"format","float-quantized"},
        {"config_groups",{{"group",{{"targets",{"Linear"}},{"weights",scheme},{"input_activations",input}}}}}};
    const auto write_config=[&]{std::ofstream(root/"config.json")<<json{{"model_type","qwen3_5"},
        {"num_hidden_layers",1},{"quantization_config",quant}}.dump();};
    write_config();
    const auto blob=image(7,33,true,fp::ScaleKind::Bf16);const auto layout=fp::parse(
        reinterpret_cast<const std::byte*>(blob.data()),blob.size());
    const auto weight_bytes=7*33;
    json header{{"language_model.layers.0.mlp.gate_proj.weight",{{"dtype","F8_E4M3"},{"shape",{7,33}},
            {"data_offsets",{0,weight_bytes}}}},
        {"language_model.layers.0.mlp.gate_proj.weight_scale",{{"dtype","BF16"},{"shape",{7,1}},
            {"data_offsets",{weight_bytes,weight_bytes+14}}}},
        {"language_model.embed_tokens.weight",{{"dtype","BF16"},{"shape",{1,2}},
            {"data_offsets",{weight_bytes+14,weight_bytes+18}}}}};
    {
        std::ofstream file(root/"model.safetensors",std::ios::binary);const auto text=header.dump();
        const std::uint64_t length=text.size();file.write(reinterpret_cast<const char*>(&length),8);file<<text;
        file.write(reinterpret_cast<const char*>(blob.data()+fp::header_bytes),blob.size()-fp::header_bytes);
        const std::uint32_t embedding=0;file.write(reinterpret_cast<const char*>(&embedding),4);
    }
    check_source(root);
    mfq::ModelSource::TensorReader retained;
    {
        mfq::HfModelSource source(root);
        require(source.find_tensor("model.block.0.mlp.gate.weight_scale")==nullptr,"consumed FP8 scale leaked into weights");
        retained=source.tensor_reader("model.block.0.mlp.gate.weight");
    }
    std::vector<std::byte> copy(blob.size());retained(0,copy.data(),copy.size());
    require(std::memcmp(copy.data(),blob.data(),blob.size())==0,"retained FP8 reader lost its source lifetime");
    retained={};
    quant["config_groups"]["group"]["input_activations"]["dynamic"]=false;write_config();
    bool rejected=false;
    try{mfq::HfModelSource invalid(root);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"unsupported FP8 activation contract was silently accepted");
}
}
int main(int argc,char** argv) try {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0)return 77;
    const auto context=mfq::cuda::default_context(0);context->begin_graph_pool(mfq_current_cuda_stream());
    struct End{std::shared_ptr<mfq::cuda::Context> c;~End(){c->end_graph_pool(mfq_current_cuda_stream());}} end{context};
    if(argc==3 && std::string(argv[1])=="--source") {
        check_source(argv[2]);std::cout<<"FP8 worst projection relative L2="<<worst_projection_relative_l2<<'\n';return 0;
    }
    namespace fp=mfq::channel_fp8;
    {
        ChannelFp8Weight cpu;cpu.out=254;cpu.neuron_len=1;
        cpu.values=tb::empty({254,1},tb::TensorOptions().dtype(tb::kUInt8));
        cpu.scales=tb::full({254},1,tb::TensorOptions().dtype(tb::kFloat32));
        for(int i=0;i<254;++i)cpu.values.data_ptr<unsigned char>()[i]=i<127?i:i+1;
        auto gpu=cpu;gpu.values=cpu.values.to(tb::kCUDA);gpu.scales=cpu.scales.to(tb::kCUDA);
        auto expected=cpu.dequant(),actual=gpu.dequant().cpu();
        require(std::memcmp(expected.data_ptr(),actual.data_ptr(),expected.nbytes())==0,
            "FP8 bit conversion changed finite codes or signed zero");
    }
    for(auto kind:{fp::ScaleKind::Bf16,fp::ScaleKind::F16,fp::ScaleKind::F32})for(bool dynamic:{false,true})
        for(const auto shape:{std::pair{7,33},std::pair{17,256}})
            check_weight(load_channel_fp8(image(shape.first,shape.second,dynamic,kind),false,-1),true);
    check_activation_rounding();
    const auto root=std::filesystem::temp_directory_path()/(
        "channel-fp8-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup{std::filesystem::path path;~Cleanup(){std::error_code error;std::filesystem::remove_all(path,error);}} cleanup{root};
    source_fixture(root);
    std::cout<<"channel FP8 source/CPU/CUDA/graph/row selection PASS relative_l2="<<worst_projection_relative_l2<<'\n';return 0;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
