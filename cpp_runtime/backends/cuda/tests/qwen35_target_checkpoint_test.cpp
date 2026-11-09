#include "storage/model_loader.h"
#include "storage/text_session_cache.h"
#include "models/qwen35/mtp_graph.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>

namespace tb=mfq_tensor_backend;
using mfq::cuda::qwen35::LinearAttentionBlock;
static void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
static void exact(const tb::Tensor& a,const tb::Tensor& b,const char* message) {
    const auto x=a.contiguous().cpu(),y=b.contiguous().cpu();
    require(x.sizes()==y.sizes() && x.scalar_type()==y.scalar_type() &&
        std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes())==0,message);
}
int main(int argc,char** argv)try {
    const bool key_only=argc==3 && std::string_view(argv[1])=="--prefix-key-check";
    if(argc!=2 && !key_only){std::cout<<"Supply a dense Qwen GGUF checkpoint\n";return 77;}
    constexpr int capacity=128;
    const auto* checkpoint=argv[key_only?2:1];
    auto source=std::make_shared<mfq::GgufModelSource>(checkpoint);
    const auto legacy_key=mfq::cuda::internal::cuda_prefix_cache_compatibility_key(*source,capacity);
    const auto precise_key=mfq::cuda::internal::cuda_prefix_cache_compatibility_key(*source,capacity,"precise");
    const auto fast_key=mfq::cuda::internal::cuda_prefix_cache_compatibility_key(*source,capacity,"fast");
    require(legacy_key==precise_key && fast_key!=precise_key,
        "Fast prefill reused the precise prefix namespace or invalidated precise caches");
    if(key_only){std::cout<<"Precise prefix compatibility and fast prefill namespace isolation PASS\n";return 0;}
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    std::cout<<std::unitbuf;tb::NoGradGuard no_grad;
    const auto context=mfq::cuda::default_context();
    CudaExecutionContext execution;
    execution.kl_mmq.kv_cache_capacity=capacity;
    auto model=mfq::cuda::load_causal_lm<mfq::cuda::Qwen35CausalLm>(execution,
        checkpoint,"",capacity,true,false,source,true);
    model.config.legacy_tensor_layout.norm_weight_offset=0.;
    model.config.legacy_tensor_layout.linear_attention_a_is_log=false;
    model.config.legacy_tensor_layout.qwen_gdn_gguf_layout=true;
    model.metadata.norm_weight_offset=0.;
    model.adapter_load_final_state(*source,model.output_norm);model.adapter_prepare_blocks(*source);
    model.hidden_capture=std::make_shared<mfq::cuda::Qwen35HiddenCapture>();
    model.hidden_capture->layers={0};
    const auto options=tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt64);
    auto ids=tb::tensor(std::vector<int64_t>{17,29,53,71,83,97,107,113},options).reshape({1,8}).to(tb::kCUDA);
    model.reset(1);model.hidden_forward(ids.narrow(1,0,4));
    struct State {LinearAttentionBlock* block;tb::Tensor convolution,recurrent;};
    std::vector<State> base;
    for(const auto& block:model.blocks)if(auto* b=dynamic_cast<LinearAttentionBlock*>(block.get()))
        base.push_back({b,b->mixer.convolution.clone(),b->mixer.recurrent.clone()});
    require(!base.empty(),"Target checkpoint test requires recurrent attention");
    const auto restore=[&] {
        for(const auto& saved:base) {
            saved.block->clear_speculative();
            saved.block->mixer.convolution.copy_(saved.convolution);
            saved.block->mixer.recurrent.copy_(saved.recurrent);
        }
        model.cache_pos=4;model.speculative_start=-1;model.speculative_confirmed=0;
    };
    mfq::cuda::qwen35::MtpTargetGraphs graphs(model);
    unsigned cases=0;
    for(int rows:{2,4,8,3,6,2,8,4}) {
        std::vector<int> retained{1,(rows+1)/2,rows};
        std::sort(retained.begin(),retained.end());
        retained.erase(std::unique(retained.begin(),retained.end()),retained.end());
        for(const auto keep:retained) {
            const auto input=ids.narrow(1,0,rows);
            restore();tb::Tensor expected_raw;bool reference_initialized=false;
            // An independent owner retains only this width. The candidate
            // must reproduce it while reusing storage across earlier widths.
            mfq::cuda::qwen35::MtpTargetGraphs reference(model);
            const auto reference_hidden=reference.run(input,1,&expected_raw,nullptr,&reference_initialized,capacity);
            require(reference_hidden.has_value(),"Reference graph rejected a valid verification window");
            const auto expected=reference_hidden->cpu();
            expected_raw=expected_raw.cpu();
            if(keep==rows)model.commit_speculative();else require(reference.rollback(keep-1),"Reference rollback was unavailable");
            std::vector<std::array<tb::Tensor,2>> expected_state;
            for(const auto& saved:base)expected_state.push_back({saved.block->mixer.convolution.cpu(),saved.block->mixer.recurrent.cpu()});
            restore();tb::Tensor actual_raw;bool initialized=false;
            const auto actual=graphs.run(input,1,&actual_raw,nullptr,&initialized,capacity);
            require(actual.has_value(),"Target graph rejected a valid verification window");
            exact(*actual,expected,"Target hidden changed after sharing recurrent checkpoints");
            exact(actual_raw,expected_raw,"Target raw features changed after sharing recurrent checkpoints");
            if(keep==rows)model.commit_speculative();else require(graphs.rollback(keep-1),"Target graph rollback was unavailable");
            require(model.cache_pos==4+keep,"Target graph commit position changed");
            for(std::size_t i=0;i<base.size();++i) {
                exact(base[i].block->mixer.convolution,expected_state[i][0],"Target convolution changed after cross-width rollback");
                exact(base[i].block->mixer.recurrent,expected_state[i][1],"Target recurrence changed after cross-width rollback");
            }
            ++cases;
        }
    }
    require(graphs.replays()>graphs.captures(),"Target checkpoint test did not reuse captured widths");
    std::cout<<"Dense target shared checkpoints: hidden/features/full commit/partial rollback cases="<<cases<<" PASS\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
