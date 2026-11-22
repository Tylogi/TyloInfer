#include "runtime/routed_cache_resources.h"
#include "cuda_runtime_config.h"
#include "cuda_execution.h"
#include "generation.h"
#include "decision_readout.h"
#include "core/full_block.h"
#include "models/qwen35/linear_attention.h"
#include "models/qwen35/mtp.h"
#include "storage/model_loader.h"
#include "storage/weight_loader.h"
#include "storage/moe_expert_cache.h"
#include "runtime/kv_offload.h"
#include "runtime/qsa_ram_cache.h"
#include "runtime/kv_expert_loan.h"
#include "runtime/layer_weight_stream.h"
#include "mfq_cuda_resident_allocations.h"
#include "mfq_cuda_process_memory.h"
#include "runtime/execution_options.h"
#include "cuda_batching.h"
#include "storage/text_session_cache.h"
#include "mtp_metrics.h"
#include "mtp_probability_policy.h"
#include "request_executor.h"
#include "generation_flow.h"
#include "mfq_weight_memory.h"
#include "tyloinfer/host_memory.h"

#include <cuda_runtime_api.h>
#include "mfq_cuda_kv_quantization.h"
#include <algorithm>
#include <limits>
#include <variant>
#include <set>

namespace mfq::cuda {
using namespace mfq::engine;
using namespace mfq::cuda::internal;

namespace engine_detail {
template <typename Model>
struct CudaEngineState {
    std::shared_ptr<WeightMemoryLedger> weight_memory;
    std::shared_ptr<CudaExecutionContext> execution;
    CudaRuntimeConfig config;
    Model model;
    RuntimeComponents<Model> components;
    Model& language;
    DecodeGraphCache graph;
    TextSessionCache cache;
    std::shared_ptr<void> ram_prefix_reclaimer,ram_expert_reclaimer,ram_duplicate_reclaimer,ram_budget_reserver,ram_pinned_reclaimer;
    std::unique_ptr<ContinuousBatch<CudaBatchOperations<Model>>> batching;
    const qwen4_exp::Qwen4ExpMtp* batch_predictor = nullptr;
    bool duplex_active = false;
    std::optional<mfq::DecisionModelSpec> decision_model;
    QuantLinear decision_readout;
    std::size_t resident_budget=0, weight_budget=0, initial_expert_budget=0, allocation_ceiling=0;
    std::shared_ptr<KvExpertLoan> graph_expert_loan;
    mfq_tensor_backend::Tensor graph_expert_storage;
    std::optional<std::size_t> graph_expert_region;
    struct WorkspaceExpertRegion {mfq_tensor_backend::Tensor storage;std::size_t region;};
    std::vector<WorkspaceExpertRegion> workspace_expert_regions;
    std::uint64_t workspace_expert_reclaimed_used_bytes=0;
    int graph_draft_depth=3;
    bool preparing_graph_depth=true;
    bool planning_mtp=true;

    std::size_t workspace_expert_capacity() const {
        std::size_t bytes=0;for(const auto& part:workspace_expert_regions)bytes+=part.storage.nbytes();return bytes;
    }
    std::size_t workspace_expert_used() const {
        std::size_t bytes=0;
        if(graph_expert_loan)for(const auto& part:workspace_expert_regions)bytes+=graph_expert_loan->region_bytes(part.region);
        return bytes;
    }
    void release_graph_expert_space(std::size_t required_available=std::numeric_limits<std::size_t>::max()) {
        if(!graph_expert_region && workspace_expert_regions.empty())return;
        finish_routed_expert_exchanges(*execution);
        TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
        auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
        const bool all=required_available==std::numeric_limits<std::size_t>::max();
        while(!workspace_expert_regions.empty() && (all || context->available_memory()<required_available)) {
            const auto& part=workspace_expert_regions.back();
            const auto bytes=part.storage.nbytes();
            const auto used=graph_expert_loan->region_bytes(part.region);
            workspace_expert_reclaimed_used_bytes+=used;
            graph_expert_loan->remove(part.region);
            workspace_expert_regions.pop_back();
            context->reclaim_idle_memory();
            std::cerr<<"runtime_prefill_workspace_cache action=reclaim bytes="<<bytes<<" used_bytes="<<used<<'\n';
        }
        if(graph_expert_region && (all || context->available_memory()<required_available)) {
            const auto bytes=graph_expert_storage.nbytes();
            graph_expert_loan->remove(*graph_expert_region);
            graph_expert_region.reset();graph_expert_storage={};
            std::cerr<<"runtime_mtp_workspace_cache action=reclaim bytes="<<bytes<<'\n';
        }
        context->reclaim_idle_memory();
    }

    void restore_mtp_expert_space() {
        // These regions hold legacy paged expert fields. TQ and GGUF retain ordinary
        // compressed tensor images and cannot consume those page addresses.
        if(execution->tq_expert_cache || execution->gguf_expert_cache)return;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            if(preparing_graph_depth || graph_expert_region || !graph_expert_loan || !components.mtp)return;
            const auto maximum=qwen4_mtp_graph_reserve(language,components.mtp.get());
            const auto active=qwen4_mtp_graph_reserve(language,components.mtp.get(),graph_draft_depth);
            if(active>=maximum)return;
            auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
            const auto available=context->available_memory(),headroom=std::size_t(256)<<20;
            auto bytes=std::min(maximum-active,available>headroom?available-headroom:0);
            bytes=bytes/KvExpertLoan::page*KvExpertLoan::page;
            if(!bytes)return;
            // An ordinary resident tensor supplies revocable expert pages;
            // no decode graph captures an address into this allocation.
            ResidentAllocationScope resident;
            try {
                graph_expert_storage=mfq_tensor_backend::empty({int64_t(bytes)},
                    mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kUInt8));
            } catch(const std::exception& error) {
                const std::string message=error.what();
                if(message.find("VRAM limit")==std::string::npos && message.find("out of memory")==std::string::npos)throw;
                std::cerr<<"runtime_mtp_workspace_cache action=skip reason=memory_pressure\n";
                return;
            }
            graph_expert_region=graph_expert_loan->add(graph_expert_storage,int64_t(bytes),1,true,1,true,true);
            std::cerr<<"runtime_mtp_workspace_cache action=lend depth="<<graph_draft_depth<<" bytes="<<bytes<<'\n';
        }
    }

    void restore_graph_expert_space() {
        if(preparing_graph_depth)return;
        const bool trace=runtime_options::layer_timings();
        const auto start=trace?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        std::optional<std::size_t> transient;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
            if(!batching)transient=qwen4_decode_transient_reserve(language,planning_mtp?components.mtp.get():nullptr,graph_draft_depth);
        restore_routed_decode_workspace(*execution,
            execution->kv_memory?execution->kv_memory->growth_reserve():0,transient);
        if(execution->gguf_expert_cache || execution->tq_expert_cache) {
            if(trace) {
                std::cerr<<"runtime_routed_decode_ready restore_ms="<<
                    std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                for(const auto& [name,value]:routed_expert_memory_metrics(*execution))
                    if(name=="expert_logical_transfer_bytes" || name=="expert_host_fetch_ms" ||
                        name=="expert_batched_cpu_copy_ms" || name=="expert_batched_stage_wait_ms" ||
                        name=="expert_allocation_ms" || name=="expert_writeback_bytes" ||
                        name=="resident_expert_cache_bytes" || name=="expert_cache_budget_bytes" ||
                        name=="expert_batched_upload_packets" || name=="gguf_grouped_prefill_calls" ||
                        name=="gguf_prefill_resident_experts" || name=="gguf_prefill_rebalance_ms")
                        std::cerr<<' '<<name<<'='<<value;
                std::cerr<<'\n';
            }
            return;
        }
        mfq::cuda::default_context(execution->layer_placement.primary_device())->reclaim_idle_memory();
        restore_mtp_expert_space();
        if(!graph_expert_loan)return;
        bool ssd=false;std::size_t transfer=0,streamed=0;
        const auto expert_metrics=routed_expert_memory_metrics(*execution);
        for(const auto& [name,value]:expert_metrics) {
            if(name=="ssd_expert_enabled")ssd=value!=0;
            if(name=="expert_decode_stage_budget_bytes")transfer=std::size_t(value);
            if(name=="ram_expert_payload_bytes" || name=="ssd_expert_payload_bytes")streamed+=std::size_t(value);
        }
        if(!ssd)return;
        if(const auto* profile=std::getenv("TYLOINFER_PROFILE_SSD_EXPERTS");profile && profile[0]=='1' && language.source) {
            const auto io=language.source->file_read_stats();
            std::cerr<<"runtime_model_io_stats boundary=prefill_end calls="<<io.calls
                <<" logical_bytes="<<io.logical_bytes<<" physical_bytes="<<io.physical_bytes
                <<" read_ms="<<double(io.read_nanoseconds)/1e6<<'\n';
            std::cerr<<"runtime_moe_pipeline_stats boundary=prefill_end";
            for(const auto& [name,value]:expert_metrics)
                if(name=="expert_route_wait_ms" || name=="expert_plan_ms" ||
                   name=="expert_host_fetch_ms" || name=="expert_logical_transfer_bytes" ||
                   name=="ssd_expert_read_wait_ms")std::cerr<<' '<<name<<'='<<value;
            std::cerr<<'\n';
        }
        // Spare device memory can hold revocable copies of streamed experts.
        // Primary arenas retain their configured budget. Prefill, KV growth
        // and online resident budgets return these chunks before allocation.
        const auto reserve=runtime_options::workspace_reserve_bytes();
        const auto decode=(std::size_t(1280)<<20)+transfer;
        const auto existing=workspace_expert_capacity();
        const auto target=std::max(streamed,reserve>decode?reserve-decode:0);
        if(existing>=target)return;
        auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
        const auto available=context->available_memory(),headroom=std::size_t(256)<<20;
        auto bytes=std::min(target-existing,available>headroom?available-headroom:0);
        bytes=bytes/KvExpertLoan::page*KvExpertLoan::page;
        if(!bytes)return;
        ResidentAllocationScope resident;
        // Small KV or prefix growth returns only the required chunks, keeping
        // the rest of the decoded expert working set available to MTP.
        while(bytes) {
            const auto chunk=std::min(bytes,std::size_t(256)<<20);
            mfq_tensor_backend::Tensor storage;
            try {
                storage=mfq_tensor_backend::empty({int64_t(chunk)},
                    mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kUInt8));
            }catch(const std::exception& error) {
                const std::string message=error.what();
                if(message.find("VRAM limit")==std::string::npos && message.find("out of memory")==std::string::npos)throw;
                break;
            }
            workspace_expert_regions.reserve(workspace_expert_regions.size()+1);
            const auto region=graph_expert_loan->add(storage,int64_t(chunk),1,true,1,true,true);
            workspace_expert_regions.push_back({std::move(storage),region});
            bytes-=chunk;
        }
        std::cerr<<"runtime_prefill_workspace_cache action=lend bytes="<<workspace_expert_capacity()-existing
            <<" capacity_bytes="<<workspace_expert_capacity()<<" reserve_bytes="<<reserve<<'\n';
    }

    void prepare_graph_depth(const InferenceRequest& input) {
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            if(!components.mtp)return;
            const auto* warmup=std::getenv("TYLOINFER_QWEN4_WARMUP_GRAPHS");
            if(!language.reuse_decode_graphs || !execution->config.moe_pipeline || (warmup && warmup[0]=='0'))return;
            const auto& sampling=input.sampling;
            const bool wide=sampling.temperature<=0 || sampling.top_k==1 ||
                (sampling.top_k>0 && sampling.top_k<=64);
            const int requested=sampling.enable_mtp && wide
                ?std::clamp(int(sampling.mtp_max_draft_tokens),1,components.mtp->maximum_draft_depth()):1;
            // Short prefill shares the existing four-row graphs. Only draft
            // caps above three need additional verification storage.
            const int depth=std::max(3,requested);
            if(depth==graph_draft_depth && language.graph_registry)return;
            const auto started=std::chrono::steady_clock::now();
            preparing_graph_depth=true;
            try {
                release_graph_expert_space();
                graph.invalidate();
                qwen4_release_decode_graphs(language,components.mtp.get());
                release_moe_decode_buffers(execution->moe_expert_cache);
                language.reset(1);components.mtp->reset(1);
                auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
                context->reclaim_idle_memory();
                reclaim_prefix_workspace((std::size_t(2)<<30)+qwen4_mtp_graph_reserve(language,components.mtp.get(),depth));
                qwen4_warmup_decode_graphs(language,components.mtp.get(),depth);
                graph_draft_depth=depth;
                preparing_graph_depth=false;
                restore_graph_expert_space();
            } catch(...) {
                graph_draft_depth=0;preparing_graph_depth=false;
                throw;
            }
            std::cerr<<"runtime_mtp_depth_plan requested_depth="<<requested<<" prepared_depth="<<depth
                <<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<<'\n';
        }
    }

    bool online_budget_supported() const {
        return execution->layer_placement.devices.size()<=1 && !execution->tensor_parallel.enabled()
            && !execution->expert_parallel.enabled();
    }

    void reclaim_prefix_workspace(std::size_t bytes) {
        reserve_routed_workspace(*execution,bytes);
        auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
        const auto available=context->available_memory();
        if(available>=bytes)return;
        if(graph_expert_region || !workspace_expert_regions.empty()) {
            std::cerr<<"runtime_workspace_cache_reclaim reason=workspace requested_bytes="<<bytes
                <<" available_bytes="<<available<<'\n';
            release_graph_expert_space(bytes);
        }
        if(context->available_memory()>=bytes)return;
        std::size_t hot=0;
        for(const auto& [name,value]:cache.metrics())
            if(name=="prefix_cache_hot_bytes")hot=std::size_t(value);
        const auto shortage=bytes-context->available_memory();
        const auto released=cache.trim_hot(hot>shortage?hot-shortage:0);
        context->reclaim_idle_memory();
        if(released)std::cerr<<"runtime_workspace_reclaim requested_bytes="<<bytes
            <<" available_bytes="<<context->available_memory()<<" prefix_released_bytes="<<released<<'\n';
    }

    void restore_dense_prefill_weights() {
        auto& borrowed=execution->borrowed_prefill_weights;
        if(borrowed.empty())return;
        if constexpr(std::is_same_v<Model,Qwen35CausalLm>)
            qwen35_release_decode_graphs(model,components.mtp.get(),graph);
        const auto device=execution->layer_placement.primary_device();
        const MfqCudaGuard guard(device);
        const auto context=mfq::cuda::default_context(device);
        context->reclaim_idle_memory();
        const auto floor=execution->prefill_restore_headroom;
        std::size_t remaining=0;
        for(const auto* weight:borrowed)remaining+=weight->nbytes();
        reclaim_prefix_workspace(remaining+floor);
        std::size_t restored=0;
        for(auto it=borrowed.begin();it!=borrowed.end();) {
            try {
                if(const auto moved=restore_prefill_weight(**it,*context,floor)) {
                    restored+=moved;
                    it=borrowed.erase(it);
                    continue;
                }
            } catch(const MfqResourceExhausted&) {
                (void)cudaGetLastError();
                context->reclaim_idle_memory();
            }
            ++it;
        }
        std::cerr<<"runtime_dense_prefill_weight_restore restored="<<restored
            <<" ram_bytes="<<remaining-restored<<'\n';
    }

    CudaEngineState(std::shared_ptr<CudaExecutionContext> owner, Model loaded,
            RuntimeComponents<Model> optional, CudaRuntimeConfig runtime_config,
            std::shared_ptr<WeightMemoryLedger> weights)
        : weight_memory(std::move(weights)), execution(std::move(owner)), config(std::move(runtime_config)),
          model(std::move(loaded)), components(std::move(optional)),
          language(components.language(model)), graph(language.max_position_embeddings()),
          cache(config.session_cache, config.prefix_cache,
              make_cuda_paged_prefix_cache(*language.source, language.max_position_embeddings(),
                  language.supports_paged_text_session_state(), config.prefix_cache),
                language.supports_text_session_state(), language.supports_text_session_state() ? 0 : 1,
                cuda_has_separate_memory()) {
        const auto trace_load=[](const char* phase) {
            if(std::getenv("TYLOINFER_TRACE_ENGINE_LOAD"))std::cerr<<"cuda_engine_load phase="<<phase<<'\n';
        };
        trace_load("resources");
        decision_model=mfq::decision_model_spec(*language.source);
        if(decision_model) {
            if constexpr(std::is_same_v<Model,Qwen35CausalLm>) {
                if(!decision_model->vocabulary_readout)
                    decision_readout=load_decision_readout(*language.source,language.hidden_size(),
                        mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA,execution->layer_placement.primary_device()));
            }
            else throw std::invalid_argument("trained D3 decisions require a Qwen3.5/Qwen3.8 backbone");
        }
        ram_prefix_reclaimer=register_qsa_ram_reclaimer(execution->layer_placement.primary_device(),
            mfq_current_cuda_stream(),cache.ram_reclaimer());
        ram_duplicate_reclaimer=register_qsa_ram_reclaimer(execution->layer_placement.primary_device(),
            mfq_current_cuda_stream(),[weak=std::weak_ptr<CudaExecutionContext>(execution)](std::size_t bytes) {
                auto owner=weak.lock();if(!owner)return std::size_t(0);
                return reclaim_routed_expert_duplicates(*owner,bytes);
            });
        ram_budget_reserver=register_qsa_ram_budget_reserver(execution->layer_placement.primary_device(),
            mfq_current_cuda_stream(),[weak=std::weak_ptr<CudaExecutionContext>(execution)](std::size_t bytes) {
                auto owner=weak.lock();if(!owner)return std::size_t(0);
                return share_routed_expert_host(*owner,bytes);
            });
        ram_expert_reclaimer=register_qsa_ram_capacity_reclaimer(execution->layer_placement.primary_device(),
            mfq_current_cuda_stream(),[weak=std::weak_ptr<CudaExecutionContext>(execution)](std::size_t bytes) {
                auto owner=weak.lock();if(!owner)return std::size_t(0);
                const auto free=mfq::sample_host_memory().available_host_bytes();
                const auto maximum=std::numeric_limits<std::size_t>::max();
                return reclaim_routed_optional_host(*owner,bytes>maximum-free?maximum:free+bytes);
            });
        ram_pinned_reclaimer=register_qsa_pinned_reclaimer(execution->layer_placement.primary_device(),
            mfq_current_cuda_stream(),[weak=std::weak_ptr<CudaExecutionContext>(execution)](std::size_t bytes) {
                auto owner=weak.lock();if(!owner)return std::size_t(0);
                return reclaim_routed_expert_pinned(*owner,bytes);
            });
        if constexpr (std::is_same_v<Model,InklingCausalLm>)
            place_inkling_experts(language,config.continuous_batch.max_sequences);
        initial_expert_budget=routed_expert_cache_budget_bytes(*execution);
        if (execution->qsa_kv_store)
            execution->qsa_kv_store->set_ram_reservation([this](std::size_t bytes) {
                cache.reserve_ram(bytes);reclaim_routed_optional_host(*execution,bytes);
            });
        allocation_ceiling=mfq::cuda::default_context(execution->layer_placement.primary_device())->memory_stats().limit;
        if (config.continuous_batch.max_sequences) {
            if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
                if(const auto* option=std::getenv("TYLOINFER_CUDA_BATCH_MTP");option && option[0]=='1')
                    batch_predictor=dynamic_cast<const qwen4_exp::Qwen4ExpMtp*>(components.mtp.get());
                batching = std::make_unique<ContinuousBatch<CudaBatchOperations<Model>>>(
                    config.generation.prefill_chunk_size, config.continuous_batch.prefill_token_budget,
                    language, *execution, config.continuous_batch, batch_predictor);
            } else if constexpr (std::is_same_v<Model,Qwen35CausalLm>)
                batching = std::make_unique<ContinuousBatch<CudaBatchOperations<Model>>>(
                    config.generation.prefill_chunk_size, config.continuous_batch.prefill_token_budget,
                    language, *execution, config.continuous_batch,
                    language.config.sensenova_loop?dynamic_cast<const Qwen35Mtp*>(components.mtp.get()):nullptr);
            else if constexpr (std::is_same_v<Model,InklingCausalLm>)
                batching = std::make_unique<ContinuousBatch<CudaBatchOperations<Model>>>(
                    config.generation.prefill_chunk_size,config.continuous_batch.prefill_token_budget,
                    language,*execution,config.continuous_batch,components.mtp.get());
            else if constexpr (std::is_same_v<Model,Hy4CausalLm>) {
                if (hy4::can_batch(language))
                    batching = std::make_unique<ContinuousBatch<CudaBatchOperations<Model>>>(
                        config.generation.prefill_chunk_size,config.continuous_batch.prefill_token_budget,
                        language,*execution,config.continuous_batch,components.mtp.get());
            }
            if constexpr (std::is_same_v<Model,Qwen35CausalLm> || std::is_same_v<Model,Qwen4CausalLm> ||
                          std::is_same_v<Model,InklingCausalLm> || std::is_same_v<Model,Hy4CausalLm>)
                if (batching) batching->set_prefix_cache(cache);
        }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if(has_routed_expert_cache(*execution))
            language.restore_decode_workspace=[this]{restore_graph_expert_space();};
        if constexpr (std::is_same_v<Model,Qwen35CausalLm>)
            if(!language.config.num_experts) {
                language.restore_decode_workspace=[this]{restore_dense_prefill_weights();};
                language.reclaim_decode_workspace=[this](std::size_t bytes){reclaim_prefix_workspace(bytes);};
            }
        if(has_routed_expert_cache(*execution) && components.mtp)
            components.mtp->decode_workspace_changed=[this]{restore_graph_expert_space();};
        if(execution->moe_expert_cache) {
            graph_expert_loan=execution->kv_memory?execution->kv_memory->expert_loan():nullptr;
            if(!graph_expert_loan && (components.mtp || moe_expert_cache_ssd_experts(execution->moe_expert_cache))) {
                graph_expert_loan=std::make_shared<KvExpertLoan>();
                set_moe_kv_expert_loan(execution->moe_expert_cache,graph_expert_loan);
            }
            set_moe_workspace_reservation(execution->moe_expert_cache,
                [this](std::size_t bytes) {reclaim_prefix_workspace(bytes);},
                [this](std::size_t bytes) {
                    cache.reserve_ram(bytes);reclaim_routed_optional_host(*execution,bytes);
                });
        }
        if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
            if(!language.config.experts)language.restore_decode_workspace=[this] {
                qwen4_restore_dense_prefill_weights(language,components.mtp.get());
            };
            language.reserve_kv_workspace = [this](std::size_t bytes) {
                reserve_routed_workspace(*execution,bytes);
                const auto device = execution->layer_placement.primary_device();
                auto context = mfq::cuda::default_context(device);
                if (context->available_memory() >= bytes) return;
                if(graph_expert_region || !workspace_expert_regions.empty()) {
                    std::cerr<<"runtime_workspace_cache_reclaim reason=kv_growth requested_bytes="<<bytes
                        <<" available_bytes="<<context->available_memory()<<'\n';
                    release_graph_expert_space(bytes);
                }
                if(context->available_memory()>=bytes)return;
                std::size_t hot = 0;
                for (const auto& [name,value] : cache.metrics())
                    if (name == "prefix_cache_hot_bytes") hot = std::size_t(value);
                const auto shortage = bytes - context->available_memory();
                cache.trim_hot(hot > shortage ? hot - shortage : 0);
                context->reclaim_idle_memory();
                if (context->available_memory() < bytes)
                    throw InferenceInputError(InferenceInputErrorCode::ContextLength,
                        "insufficient VRAM for context growth; reduce concurrent requests or prompt length");
            };
            if(execution->kv_memory)execution->kv_memory->set_reservation(language.reserve_kv_workspace);
        }
#endif
        if(const auto bytes=runtime_options::resident_budget_bytes()) {
            SessionCommand command;
            command.kind=SessionCommand::Kind::memory_budget;
            command.bytes=bytes;
            session(command);
        }
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            mfq::cuda::RoutedCacheLearningScope learning(*execution,false);
            // Bootstrap uses the same persistent host state as a real request.
            const PinnedAllocationScope pinned(execution->layer_placement.primary_device(),mfq_current_cuda_stream(),
                [this](std::size_t bytes) {
                    if(DecodeWindow::recording())return std::size_t(0);
                    return reclaim_routed_expert_pinned(*execution,bytes);
                });
            trace_load("graph_warmup");
            qwen4_warmup_decode_graphs(language,components.mtp.get());
            trace_load("graph_warmup_done");
            preparing_graph_depth=false;
            restore_graph_expert_space();
            // Exercise the same draft/verify/commit loop as a real request.
            // Graph-only warmup cannot cover its history replay, sampling and
            // expert exchange boundaries. No synthetic policy/history survives.
            const auto* warmup=std::getenv("TYLOINFER_QWEN4_WARMUP_MTP");
            if(language.graph_registry && components.mtp &&
                    language.max_position_embeddings()>=48 && (!warmup || warmup[0]!='0')) {
                auto& predictor=*components.mtp;
                const auto saved_policy=predictor.policy_state;
                const auto start=std::chrono::steady_clock::now();
                InferenceRequest request;
                request.chat=false;
                request.sampling.temperature=0.7;
                request.sampling.top_p=0.8;
                request.sampling.top_k=20;
                // Explicit confidence policies consume the greedy proposal's
                // probability. Warm that requested path instead of rejecting
                // its settings during the internal stochastic warmup.
                if(mfq::engine::mtp::DraftProbabilityOptions::from_environment().enabled()) {
                    request.sampling.temperature=0.;
                    request.sampling.top_p=1.;
                    request.sampling.top_k=0;
                }
                request.sampling.seed=1234;
                request.sampling.mtp_max_draft_tokens=3;
                // One full draft/verification window exercises these paths;
                // policy and history are discarded immediately after warmup.
                request.sampling.max_tokens=static_cast<int32_t>(std::min<int64_t>(
                    request.sampling.mtp_max_draft_tokens+1,language.max_position_embeddings()-16));
                for(int i=0;i<16;++i)request.prompt.push_back((i*7919+17)%language.vocab_size());
                InferenceOutput output(request,nullptr,"cuda-mtp-warmup");
                auto scope=language.execution_scope();
                auto generation=run_mtp_generation(language,predictor,request,output,16);
                while(generation.next()) {}
                finish_routed_expert_exchanges(*execution);
                predictor.reset(1);
                language.reset(1);
                predictor.policy_state=saved_policy;
                predictor.last_stats={};
                predictor.last_cycles=predictor.last_accepted=predictor.last_rejected=0;
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
                std::cerr<<"runtime_qwen4_mtp_warmup tokens="<<output.result.completion_tokens
                    <<" ms="<<std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-start).count()<<'\n';
                release_moe_prefill_buffers(execution->moe_expert_cache);
                mfq::cuda::default_context(execution->layer_placement.primary_device())->reclaim_idle_memory();
            }
        }
        preparing_graph_depth=false;
        trace_load("decode_workspace");
        restore_graph_expert_space();
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            if(language.source && execution->gguf_expert_cache && !model_parallel_enabled(*execution) &&
                    !execution->layer_placement.enabled() && !execution->dense_cpu_layer_count) {
                trace_load("expert_profile");
                seed_gguf_expert_cache(execution->gguf_expert_cache,*language.source);
            }
        } else if constexpr(std::is_same_v<Model,Hy4CausalLm>) prepare_hy4_expert_cache(language);
        trace_load("ready");
    }
    bool can_batch(const EngineRequest& request) const {
        if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
            const auto& input=request.input;
            if(!batching || input.media || !input.cache_plan.session_id.empty())return false;
            if(!input.sampling.enable_mtp || !components.mtp)return true;
            return batch_predictor && config.continuous_batch.cuda_graph && mfq_cuda_graph_capture_supported() &&
                FlashNextBatchOperations::supports_mtp(input.sampling,{}) &&
                (!input.chat || !request.token_ids.empty() || (input.chat_input && input.chat_input->preformatted_prompt &&
                    input.chat_input->template_inputs.json_schema.empty()));
        }
        if constexpr (std::is_same_v<Model,InklingCausalLm> || std::is_same_v<Model,Hy4CausalLm>) {
            const auto& input=request.input;
            if (!batching || input.media || !input.cache_plan.session_id.empty()) return false;
            if (!input.sampling.enable_mtp || !components.mtp) return true;
            const auto& sampling=input.sampling;
            return config.continuous_batch.cuda_graph && mfq_cuda_graph_capture_supported() &&
                (!input.chat_input || input.chat_input->template_inputs.json_schema.empty()) &&
                sampling.presence_penalty==0 && sampling.frequency_penalty==0 && sampling.repetition_penalty==1 &&
                (sampling.temperature==0 || (sampling.top_k>0 && sampling.top_k<=64));
        }
        if constexpr (std::is_same_v<Model,Qwen35CausalLm>) {
            const auto& input=request.input;
            const auto& sampling=input.sampling;
            if (language.config.sensenova_loop) {
                if (!batching || input.media || !input.cache_plan.session_id.empty()) return false;
                if (!sampling.enable_mtp || !components.mtp) return true;
                return qwen_continuous_batch_cuda_graph_enabled(language,config.continuous_batch) &&
                    (!input.chat_input || input.chat_input->template_inputs.json_schema.empty()) &&
                    sampling.presence_penalty==0 && sampling.frequency_penalty==0 && sampling.repetition_penalty==1 &&
                    (sampling.temperature==0 || (sampling.top_k>0 && sampling.top_k<=64));
            }
        }
        return batching && batch_compatible(request, cache.persistent_prefix_enabled(), bool(components.mtp));
    }
    ~CudaEngineState() {
        ram_pinned_reclaimer.reset();ram_budget_reserver.reset();ram_duplicate_reclaimer.reset();ram_expert_reclaimer.reset();ram_prefix_reclaimer.reset();
        language.restore_decode_workspace={};
        language.reclaim_decode_workspace={};
        if(components.mtp)components.mtp->decode_workspace_changed={};
        try {release_graph_expert_space();}
        catch(const std::exception& error) {std::cerr<<"runtime_mtp_workspace_cache cleanup_error="<<error.what()<<'\n';}
        if(execution->kv_memory)execution->kv_memory->set_reservation({});
        set_moe_workspace_reservation(execution->moe_expert_cache,{});
        if (execution->qsa_kv_store) execution->qsa_kv_store->set_ram_reservation({});
    }
    SessionResult session(const SessionCommand& command) {
        if(command.kind==SessionCommand::Kind::trim && cuda_has_separate_memory())
            return {cache.trim_ram(command.bytes),{}};
        if(command.kind==SessionCommand::Kind::budget && cuda_has_separate_memory()) {
            auto budget=command;
            budget.bytes=0;
            return control_session(cache,budget);
        }
        if(command.kind!=SessionCommand::Kind::memory_budget)return control_session(cache,command);
        if(!online_budget_supported())throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
        const auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
        const auto dense=weight_memory->resident_bytes();
        const auto minimum=moe_expert_cache_minimum_bytes(execution->moe_expert_cache);
        const auto requested_weights=command.weight_bytes.value_or(weight_budget);
        if(requested_weights && requested_weights<dense+minimum)
            throw std::invalid_argument("weight budget is below dense weights and minimum expert working set");
        auto limit=command.bytes ? allocation_ceiling ? std::min<std::size_t>(command.bytes,allocation_ceiling) : command.bytes : allocation_ceiling;
        if(limit) {
            std::size_t free=0,total=0;
            TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&free,&total));
            limit=std::min(limit,std::min(total,free+context->resident_memory_usage()));
        }
        if(limit && limit<dense+minimum)
            throw std::invalid_argument("resident memory budget is below non-offloadable weights");
        const auto previous_limit=context->memory_stats().limit;
        const auto previous_experts=routed_expert_cache_budget_bytes(*execution);
        const auto before=context->resident_memory_usage();
        release_graph_expert_space();
        finish_routed_expert_exchanges(*execution);
        graph.invalidate();
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
            qwen4_release_decode_graphs(language,components.mtp.get());
        if constexpr(std::is_same_v<Model,Hy4CausalLm>)
            clear_hy4_decode_graph(language);
        if constexpr(std::is_same_v<Model,InklingCausalLm>)
            clear_inkling_graphs(language);
        if(batching)batching->release_idle_buffers();
        if(components.mtp)components.mtp->reset(1);
        language.reset(1);
        release_moe_decode_buffers(execution->moe_expert_cache);
        context->reclaim_idle_memory();
        auto target=requested_weights ? requested_weights-dense : initial_expert_budget;
        std::size_t hot=0,expert_bytes=0;
        for(const auto& [name,value]:cache.metrics())if(name=="prefix_cache_hot_bytes")hot=std::size_t(value);
        for(const auto& [name,value]:routed_expert_memory_metrics(*execution))
            if(name=="resident_expert_cache_bytes")expert_bytes=std::size_t(value);
        auto used=context->resident_memory_usage();
        if(limit && used>limit) {
            cache.trim_hot(hot>used-limit ? hot-(used-limit) : 0);
            context->reclaim_idle_memory();used=context->resident_memory_usage();
        }
        const auto fixed=used>expert_bytes?used-expert_bytes:0;
        if(limit) {
            if(fixed>limit || minimum>limit-fixed)throw std::invalid_argument("resident memory budget is below live KV and workspace requirements");
            target=std::min(target,limit-fixed);
        }
        try {
            if(!limit || limit>previous_limit)context->set_memory_limit(limit);
            resize_routed_expert_cache(*execution,std::max(target,minimum));
            context->set_memory_limit(limit);
        } catch(...) {
            context->set_memory_limit(0);
            resize_routed_expert_cache(*execution,previous_experts);
            context->set_memory_limit(previous_limit);
            throw;
        }
        resident_budget=command.bytes;weight_budget=requested_weights;
        const auto after=context->resident_memory_usage();
        return {before>after?before-after:0,{}};
    }
    bool exclusive() const { return duplex_active; }
    bool mtp_available() const { return bool(components.mtp); }
    void execute(const std::vector<RequestId>& eligible) {
        if (batching) batching->step(eligible);
    }
    using Prepared = CudaPreparedPrompt;
    mfq::StepSequence<Prepared> prepare(InferenceRequest& input) {
        if (components.composite) return components.composite->prepare(input.prompt, *input.vision);
        if (components.grid_vision) return components.grid_vision->prepare(language, input.prompt, *input.vision);
        if constexpr(std::is_same_v<Model,DeepseekV41CausalLm>)
            if(components.deepseek_vision)return components.deepseek_vision->prepare(language,input.prompt,*input.vision);
        if (components.radio_vision) return components.radio_vision->prepare(language, input.prompt, *input.vision);
        throw std::invalid_argument("model has no multimodal component");
    }
    auto advance_preparation(mfq::StepSequence<Prepared>& preparation) {
        PrefillCudaTimer timer;
        auto step = preparation.next();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
        return std::pair{std::move(step), timer.elapsed_ms()};
    }
    Generation generate_text(InferenceRequest& input, InferenceOutput& output, std::optional<Prepared> prepared,
                             const RequestId& id, bool batched) {
        planning_mtp=!batched && input.sampling.enable_mtp && bool(components.mtp);
        if(!batched)prepare_graph_depth(input);
        return internal::generate(language, graph, cache, config,
                                  input, output, batched ? nullptr : components.mtp.get(), std::move(prepared),
                                  batched ? batching.get() : nullptr, id);
    }
    Generation generate_exclusive(Generation sequence) {
        const auto next=[&] {
            // Scope recovery to each resume, so a suspended request cannot
            // reclaim pinned expert pages on behalf of another operation.
            const PinnedAllocationScope pinned(execution->layer_placement.primary_device(),mfq_current_cuda_stream(),
                [this](std::size_t bytes) {
                    if(DecodeWindow::recording())return std::size_t(0);
                    return reclaim_routed_expert_pinned(*execution,bytes);
                });
            return sequence.next();
        };
        while(auto step=next())co_yield std::move(step);
    }
    Generation generate(const RequestId& id, ExecutionRequest& request) {
        if (request.batched) {
            graph.invalidate();
            if constexpr (std::is_same_v<Model,InklingCausalLm>) clear_inkling_graphs(language);
            if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
                // Exclusive MTP generation has ended before a batch starts.
                // Its graph consumers must retire before idle batch buffers.
                qwen4_release_decode_graphs(language,components.mtp.get());
                if (components.mtp) components.mtp->reset(1);
                qwen4_release_idle_qsa(language,components.mtp.get(),true);
            }
        }
        auto sequence=generate_prepared(*this, request.input, request.output, id, request.batched);
        if(request.batched)return sequence;
        return generate_exclusive(std::move(sequence));
    }
    template <class T> ControlResult control(T value) {
        if constexpr(std::is_same_v<T,DecisionTokens>) {
            if constexpr(std::is_same_v<Model,Qwen35CausalLm> || std::is_same_v<Model,Qwen4CausalLm>) {
                if(value.reads.empty() || value.reads.size()>64)
                    throw std::invalid_argument("decision requests require 1 to 64 reads");
                const auto vocabulary=language.vocab_size();
                std::size_t total=0;
                for(const auto& read:value.reads) {
                    if(read.prompt.empty() || read.prompt.size()>std::size_t(language.max_position_embeddings()) ||
                            read.labels.empty() || read.labels.size()>255)
                        throw std::invalid_argument("invalid decision prompt or label count");
                    if(decision_model && decision_model->max_tokens && read.prompt.size()>*decision_model->max_tokens)
                        throw std::invalid_argument("decision prompt exceeds the model input limit");
                    if(decision_model && decision_model->attention==mfq::DecisionAttention::prefix_bidirectional &&
                        (!read.state_tokens || read.state_tokens>=read.prompt.size()))
                        throw std::invalid_argument("decision state boundary is outside the prompt");
                    for(const auto token:read.prompt)if(token<0 || token>=vocabulary)
                        throw std::invalid_argument("decision prompt token is outside the vocabulary");
                    std::unordered_set<int64_t> unique;
                    for(const auto token:read.labels) {
                        if(token<0 || token>=vocabulary || !unique.insert(token).second)
                            throw std::invalid_argument("decision labels must be distinct vocabulary tokens");
                        if(decision_model && !decision_model->vocabulary_readout &&
                            std::find(decision_model->token_ids.begin(),decision_model->token_ids.end(),token)==decision_model->token_ids.end())
                            throw std::invalid_argument("decision label is absent from trained codebook");
                    }
                    if(read.prompt.size()>1048576-total)
                        throw std::invalid_argument("decision request exceeds the aggregate token limit");
                    total+=read.prompt.size();
                }
                auto common=value.reads.front().prompt.size();
                for(const auto& read:value.reads) {
                    common=std::min(common,read.prompt.size());
                    std::size_t i=0;
                    while(i<common && read.prompt[i]==value.reads.front().prompt[i])++i;
                    common=i;
                }
                MfqCudaGuard guard(execution->layer_placement.primary_device());
                graph.invalidate();
                if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
                    qwen4_release_decode_graphs(language,components.mtp.get());
                if(components.mtp)components.mtp->reset(1);
                const bool previous_mtp=planning_mtp;planning_mtp=false;
                bool cleaned=false;
                const auto cleanup=[&] {
                    if(std::exchange(cleaned,true))return;
                    language.reset(1);
                    if(language.restore_decode_workspace)language.restore_decode_workspace();
                    planning_mtp=previous_mtp;
                };
                try {
                    DecisionResult result;
                    for(const auto& read:value.reads) {
                        DecisionAttentionScope attention_scope(*execution,decision_model?&*decision_model:nullptr,read.state_tokens);
                        auto prepared=prefill_readout(language,graph,cache,config,read.prompt,common,
                            "decision:"+value.cache_salt,value.prefix_cache);
                        std::vector<double> selected;
                        std::optional<double> normalizer;
                        std::optional<int64_t> argmax;
                        if(decision_model) {
                            auto logits=(decision_model->vocabulary_readout
                                ? language.lm_head.forward_rows(*execution,prepared.hidden,read.labels)
                                : decision_readout.forward(*execution,prepared.hidden.to(mfq_tensor_backend::kFloat32)))
                                    .to(mfq_tensor_backend::kFloat32).reshape({-1}).cpu().contiguous();
                            if(logits.numel()!=int64_t(decision_model->vocabulary_readout?read.labels.size():decision_model->codes.size()))
                                throw std::runtime_error("trained decision readout size changed");
                            for(std::size_t i=0;i<read.labels.size();++i) {
                                const auto token=read.labels[i];
                                const auto found=std::find(decision_model->token_ids.begin(),decision_model->token_ids.end(),token);
                                if(!decision_model->vocabulary_readout && found==decision_model->token_ids.end())
                                    throw std::invalid_argument("decision label is absent from trained codebook");
                                const auto row=decision_model->vocabulary_readout?i:std::size_t(found-decision_model->token_ids.begin());
                                selected.push_back(logits.template data_ptr<float>()[row] /
                                    decision_model->temperature_for(read.type, value.reads.size()));
                            }
                        } else if(value.full_vocabulary) {
                            auto logits=language.logits_from_hidden(prepared.hidden).to(mfq_tensor_backend::kFloat32)
                                .reshape({-1}).cpu().contiguous();
                            if(logits.numel()!=vocabulary)throw std::runtime_error("decision head vocabulary changed");
                            std::vector<double> all(logits.numel());
                            for(int64_t i=0;i<logits.numel();++i)all[i]=logits.template data_ptr<float>()[i];
                            normalizer=decision_logsumexp(all);
                            argmax=std::max_element(all.begin(),all.end())-all.begin();
                            for(const auto id:read.labels)selected.push_back(all[id]);
                        } else {
                            auto logits=language.apply_final_logit_softcap(language.lm_head.forward_rows(
                                *execution,prepared.hidden,read.labels)).to(mfq_tensor_backend::kFloat32)
                                .reshape({-1}).cpu().contiguous();
                            for(int64_t i=0;i<logits.numel();++i)selected.push_back(logits.template data_ptr<float>()[i]);
                        }
                        auto answer=normalize_decision(read.labels,selected,normalizer,argmax);
                        answer.prompt_tokens=read.prompt.size();answer.cached_tokens=prepared.cached_tokens;
                        answer.prefill_ms=prepared.prefill_ms;
                        result.reads.push_back(std::move(answer));
                    }
                    cleanup();return result;
                }catch(...) {
                    auto failure=std::current_exception();
                    try{cleanup();}catch(...){planning_mtp=previous_mtp;graph.invalidate();throw;}
                    planning_mtp=previous_mtp;
                    std::rethrow_exception(failure);
                }
            } else throw std::invalid_argument("decision reads are unavailable for this model architecture");
        } else if constexpr (std::is_same_v<T, RuntimeMetrics>) {
            const auto batch_metrics = batching ? batching->metrics() : Metrics{};
            const auto capabilities = components.state();
            constexpr bool supports_standard_kv_quantization=
                std::is_same_v<Model,Qwen4CausalLm> || std::is_same_v<Model,Qwen35CausalLm>;
            const auto kv_precision=[&]{
                if constexpr (std::is_same_v<Model,Hy4CausalLm>) return double(language.cold_kv_bits());
                else return supports_standard_kv_quantization?cuda_kv_quantization().bits:0.0;
            }();
            Metrics metrics{
                {"mtp_graph_prepared_depth",double(graph_draft_depth)},
                {"mtp_flexible_expert_capacity_bytes",double(graph_expert_storage.defined()?graph_expert_storage.nbytes():0)},
                {"mtp_flexible_expert_used_bytes",double(graph_expert_loan && graph_expert_region?
                    graph_expert_loan->region_bytes(*graph_expert_region):0)},
                {"prefill_workspace_expert_capacity_bytes",double(workspace_expert_capacity())},
                {"prefill_workspace_expert_used_bytes",double(workspace_expert_used())},
                {"prefill_workspace_expert_reclaimed_used_bytes",double(workspace_expert_reclaimed_used_bytes)},
                {"kv_memory_supported",std::is_same_v<Model,Qwen4CausalLm> || std::is_same_v<Model,Hy4CausalLm>?1.0:0.0},
                {"kv_quantization_supported",supports_standard_kv_quantization || std::is_same_v<Model,Hy4CausalLm>?1.0:0.0},
                {"kv_quantization_bits",kv_precision},
                {"vision_declared", double(capabilities.vision_declared)}, {"vision_supported", double(capabilities.vision_supported)},
                {"vision_available", double(capabilities.vision_available)}, {"mtp_declared", double(capabilities.mtp_declared)},
                {"mtp_supported", double(capabilities.mtp_supported)}, {"mtp_available", double(capabilities.mtp_available)}};
            auto experts = routed_expert_memory_metrics(*execution);
            metrics.insert(metrics.end(), experts.begin(), experts.end());
            if(!has_routed_expert_cache(*execution))
                metrics.insert(metrics.end(),{{"ram_expert_enabled",0.0},{"ram_expert_payload_bytes",0.0},
                    {"ssd_expert_enabled",0.0},{"ssd_expert_payload_bytes",0.0}});
            if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
                auto phases=qwen4_decode_graph_profile_metrics(language,components.mtp.get());
                metrics.insert(metrics.end(),phases.begin(),phases.end());
            }
            if(language.source) {
                const auto io=language.source->file_read_stats();
                metrics.insert(metrics.end(),{{"model_io_direct",io.mode==mfq::FileReadMode::Direct?1.0:0.0},
                    {"model_io_read_calls",double(io.calls)},{"model_io_logical_bytes",double(io.logical_bytes)},
                    {"model_io_physical_bytes",double(io.physical_bytes)},{"model_io_read_ms",double(io.read_nanoseconds)/1e6},
                    {"model_io_staging_bytes",double(io.staging_bytes)},{"model_io_cached_staging_bytes",double(io.staging_cached_bytes)},
                    {"model_io_staging_allocations",double(io.staging_allocations)}});
            }
            if(execution->kv_memory) {
                auto kv=execution->kv_memory->metrics();
                metrics.insert(metrics.end(),kv.begin(),kv.end());
            }
            std::size_t resident_weights=weight_memory->resident_bytes();
            for(const auto& metric:experts)
                if(metric.first=="resident_expert_cache_bytes")resident_weights+=std::size_t(metric.second);
            const auto embedding=weight_memory->embedding_bytes();
            const bool jev=bool(decision_model);
            auto streamed_weights=(language.layer_weights?language.layer_weights->host_bytes():0)+(jev?embedding:0);
            std::size_t loaned_weights=0;
            for(const auto* weight:execution->borrowed_prefill_weights)loaned_weights+=weight->nbytes();
            if constexpr (std::is_same_v<Model,Qwen4CausalLm>)
                loaned_weights+=qwen4_borrowed_dense_weight_bytes(language);
            streamed_weights+=loaned_weights;
            resident_weights-=std::min(resident_weights,loaned_weights);
            metrics.insert(metrics.end(),{{"ram_embedding_bytes",double(jev?0:embedding)},
                {"ram_weight_bytes",double(streamed_weights)},
                {"jev_weights_streamed",language.layer_weights?1.0:0.0}});
            const auto ple=weight_memory->ple_bytes();
            metrics.insert(metrics.end(),{{"resident_weight_bytes",double(resident_weights)},
                {"ssd_ple_enabled",ple>0?1.0:0.0},{"ssd_ple_payload_bytes",double(ple)}});
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                auto cache_memory=qwen4_cache_memory_metrics(language,components.mtp.get());
                for (auto& [name,value] : cache_memory)
                    for (const auto& [batch_name,batch_value] : batch_metrics) {
                        if (name == "kv_cache_bytes" && batch_name == "continuous_batching_request_kv_bytes")
                            value += batch_value;
                        if (name == "kv_cache_contexts" && batch_name == "continuous_batching_request_contexts")
                            value += batch_value;
                    }
                metrics.insert(metrics.end(),cache_memory.begin(),cache_memory.end());
            }
            const auto paged_kv_bytes=std::is_same_v<Model,Qwen35CausalLm> ? [&] {
                for(const auto& [key,value]:batch_metrics)
                    if(key=="paged_kv_reserved_bytes")return std::size_t(value);
                return std::size_t(0);
            }() : std::size_t(0);
            const bool gqa_metrics=std::is_same_v<Model,Qwen35CausalLm> || jev;
            if(gqa_metrics && !std::is_same_v<Model,Qwen4CausalLm>) {
                std::map<std::pair<int,const void*>,std::size_t> allocations;
                const auto add=[&](const mfq_tensor_backend::Tensor& tensor) {
                    if(!tensor.defined() || !tensor.is_cuda())return;
                    const auto storage=std::static_pointer_cast<mfq::cuda::TensorStorage>(tensor.weak_storage_owner().lock());
                    if(storage)allocations[{tensor.get_device(),storage->base}]=storage->bytes;
                };
                const auto add_block=[&](const auto& block) {
                    if(const auto* full=dynamic_cast<const FullBlock*>(block.get()))
                        for(const auto* tensor:{&full->cache.k,&full->cache.v,&full->cache.k_chunk_ptrs,
                            &full->cache.v_chunk_ptrs,&full->cache.page_table})add(*tensor);
                    if(const auto* linear=dynamic_cast<const qwen35::LinearAttentionBlock*>(block.get());jev && linear)
                        for(const auto* tensor:{&linear->mixer.convolution,&linear->mixer.recurrent,
                            &linear->mixer.speculative_convolution,&linear->mixer.speculative_recurrent})add(*tensor);
                };
                for(const auto& block:language.blocks)add_block(block);
                if(const auto* predictor=dynamic_cast<const Qwen35Mtp*>(components.mtp.get()))
                    for(const auto& block:predictor->blocks)add_block(block);
                std::map<int,std::size_t> devices;
                std::size_t bytes=paged_kv_bytes;
                for(const auto& [address,size]:allocations){bytes+=size;devices[address.first]+=size;}
                metrics.emplace_back("kv_cache_bytes",double(bytes));
                metrics.emplace_back("kv_cache_contexts",language.cache_pos>0?1.0:0.0);
                for(const auto& [device,size]:devices)
                    metrics.emplace_back("gpu_"+std::to_string(device)+"_live_kv_bytes",double(size));
            }
            const auto metric_bytes = [&](const std::string& key) {
                for(const auto& [name,value]:metrics)if(name==key)return std::size_t(value);
                return std::size_t(0);
            };
            std::set<int> devices(execution->layer_placement.devices.begin(),execution->layer_placement.devices.end());
            devices.insert(execution->tensor_parallel.devices.begin(),execution->tensor_parallel.devices.end());
            devices.insert(execution->expert_parallel.devices.begin(),execution->expert_parallel.devices.end());
            const auto primary=execution->layer_placement.primary_device();
            devices.insert(primary);
            auto prefix_bytes=cache.device_hot_bytes();
            std::size_t allocated=0,reserved=0,free_bytes=0,total_bytes=0,buffers=0;
            for(const int device:devices) {
                MfqCudaGuard guard(device);
                size_t free=0,total=0;
                TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&free,&total));
                const auto memory=mfq_cuda_memory_stats(device);
                const auto external=device==primary?metric_bytes("kv_memory_vmm_bytes"):0;
                const auto resident=std::max(memory.reserved_bytes,memory.allocated_bytes)+external;
                const auto stem="gpu_"+std::to_string(device)+"_";
                auto weights=weight_memory->resident_bytes(device);
                if(device==primary) {
                    weights+=metric_bytes("resident_expert_cache_bytes");
                    weights-=std::min(weights,loaned_weights);
                }
                const auto hot=prefix_bytes[device];
                auto live=metric_bytes(stem+"live_kv_bytes");
                if(device==primary && devices.size()==1)live+=paged_kv_bytes;
                if(device==primary)for(const auto& [name,value]:batch_metrics)
                    if(name=="continuous_batching_request_kv_bytes")live+=std::size_t(value);
                metrics.insert(metrics.end(),{{stem+"resident_bytes",double(resident)},
                    {stem+"resident_weight_bytes",double(weights)}, {stem+"prefix_cache_bytes",double(hot)},
                    {stem+"memory_total_bytes",double(total)}, {stem+"memory_available_bytes",double(free)}});
                if(const auto physical=mfq::cuda::physical_device_memory_stats(device))
                    metrics.insert(metrics.end(),{{stem+"physical_memory_total_bytes",double(physical->total)},
                        {stem+"physical_memory_available_bytes",double(physical->free)},
                        {stem+"physical_memory_used_bytes",double(physical->used)}});
                if(const auto process=mfq::cuda::process_device_memory_stats(device))
                    metrics.insert(metrics.end(),{{stem+"process_memory_budget_bytes",double(process->budget)},
                        {stem+"process_memory_usage_bytes",double(process->usage)},
                        {stem+"process_memory_reservation_bytes",double(process->reservation)},
                        {stem+"process_shared_memory_budget_bytes",double(process->shared_budget)},
                        {stem+"process_shared_memory_usage_bytes",double(process->shared_usage)}});
                if(std::is_same_v<Model,Qwen4CausalLm> || (gqa_metrics && (!paged_kv_bytes || devices.size()==1))) {
                    const auto categorized=weights+live+hot;
                    const auto buffer=resident>categorized?resident-categorized:0;
                    metrics.emplace_back(stem+"resident_buffer_bytes",double(buffer));
                    metrics.emplace_back(stem+"kv_bytes",double(live+hot));
                    buffers+=buffer;
                }
                allocated+=memory.allocated_bytes+external;reserved+=memory.reserved_bytes+external;free_bytes+=free;total_bytes+=total;
            }
            metrics.insert(metrics.end(),{{"device_free_bytes",double(free_bytes)}, {"device_total_bytes",double(total_bytes)},
                {"cuda_allocated_bytes",double(allocated)}, {"cuda_reserved_bytes",double(reserved)}});
            if(std::is_same_v<Model,Qwen4CausalLm> || (gqa_metrics && (!paged_kv_bytes || devices.size()==1)))
                metrics.emplace_back("resident_buffer_bytes",double(buffers));
            const auto expert_bytes=metric_bytes("resident_expert_cache_bytes");
            const auto minimum=moe_expert_cache_minimum_bytes(execution->moe_expert_cache);
            metrics.insert(metrics.end(),{{"resident_memory_dynamic_budget",online_budget_supported()?1.0:0.0},
                {"resident_weight_dynamic_budget",online_budget_supported()?1.0:0.0},
                {"resident_memory_budget_bytes",double(resident_budget)}, {"resident_weight_budget_bytes",double(weight_budget)},
                {"resident_memory_used_bytes",double(std::max(allocated,reserved))},
                {"resident_reclaimable_weight_bytes",double(expert_bytes>minimum?expert_bytes-minimum:0)}});
            metrics.emplace_back("qsa_kv_offload_supported",
                std::is_same_v<Model,Qwen4CausalLm> || std::is_same_v<Model,Hy4CausalLm> ? 1.0 : 0.0);
            if (execution->qsa_kv_store) {
                auto kv = execution->qsa_kv_store->metrics();
                metrics.insert(metrics.end(), kv.begin(), kv.end());
            } else {
                const auto ram=metric_bytes("qsa_adaptive_ram_allocated_bytes");
                const auto cold=metric_bytes("qsa_adaptive_ssd_bytes");
                const auto index=metric_bytes("qsa_adaptive_index_archive_ssd_bytes");
                const auto disk=cold+index;
                const auto enabled=metric_bytes("qsa_adaptive_ram_enabled");
                metrics.insert(metrics.end(),{{"qsa_kv_offload_enabled",enabled?1.:0.},
                    {"qsa_kv_ssd_enabled",metric_bytes("qsa_adaptive_ssd_enabled")?1.:0.},
                    {"qsa_kv_ram_enabled",enabled?1.:0.},{"qsa_kv_ram_bytes",double(ram)},
                    {"qsa_kv_ssd_bytes",double(disk)},
                    {"qsa_kv_cold_ssd_bytes",double(cold)},
                    {"qsa_kv_index_archive_ssd_bytes",double(index)}});
            }
            if (components.mtp) {
                mtp::append_generation_metrics(metrics, components.mtp->last_stats);
                metrics.emplace_back("mtp_predictor_graph_replays",double(components.mtp->graph_replay_count()));
            }
            if (batching) {
                metrics.insert(metrics.end(), batch_metrics.begin(), batch_metrics.end());
            }
            return metrics;
        } else {
            if (!components.composite) throw std::invalid_argument("model has no duplex component");
            if constexpr (std::is_same_v<T, MfqDuplexSessionParams>) {
                components.composite->start(value); duplex_active = true;
            } else if constexpr (std::is_same_v<T, MfqDuplexStepInput>) {
                return components.composite->step(value);
            } else { components.composite->stop(); duplex_active = false; }
            return std::monostate{};
        }
    }

};
using State = std::variant<
    std::unique_ptr<CudaEngineState<Qwen35CausalLm>>,
    std::unique_ptr<CudaEngineState<MiniCPMO45CausalLm>>,
    std::unique_ptr<CudaEngineState<MiniCPMOTtsCausalLm>>,
    std::unique_ptr<CudaEngineState<Gemma4CausalLm>>,
    std::unique_ptr<CudaEngineState<GlmDsaCausalLm>>,
    std::unique_ptr<CudaEngineState<Glm5CausalLm>>,
    std::unique_ptr<CudaEngineState<Qwen4CausalLm>>,
    std::unique_ptr<CudaEngineState<DeepseekV4CausalLm>>,
    std::unique_ptr<CudaEngineState<DeepseekV41CausalLm>>,
    std::unique_ptr<CudaEngineState<NemotronHCausalLm>>,
    std::unique_ptr<CudaEngineState<InklingCausalLm>>,
    std::unique_ptr<CudaEngineState<Hy4CausalLm>>>;

struct CudaBackend {
    using Options = CudaEngineOptions;
    Options options;
    State state;
    template <class F> decltype(auto) visit(F&& run) {
        return std::visit([&](auto& state) -> decltype(auto) {
            if (!state) throw std::runtime_error("CUDA engine is unloaded");
            MfqCudaGuard guard(state->execution->layer_placement.primary_device());
            mfq_tensor_backend::NoGradGuard no_grad;
            return run(*state);
        }, state);
    }
    std::pair<EngineInfo, std::unique_ptr<TextProcessor>> load() {
        if (!options.context_size) options.context_size = 32768;
        auto execution = std::make_shared<CudaExecutionContext>();
        setup_cuda_load(options, *execution);
        execution->profiler.enabled = false;
        auto weight_memory=std::make_shared<WeightMemoryLedger>();
        const WeightMemoryLoadScope memory_load(*weight_memory);
        mfq_tensor_backend::NoGradGuard no_grad;
        auto config = resolve_cuda_runtime_config(options);
        // Discrete workers retain paged prefixes in host memory and on disk.
        if(cuda_has_separate_memory())config.prefix_cache.hot_bytes=0;
        const auto kv_options = runtime_options::qsa_kv_offload();
        execution->kv_memory=std::make_shared<KvMemoryManager>();
        if (kv_options.gpu_budget_bytes) {
            execution->kv_memory.reset();
            KvOffloadConfig kv;
            kv.gpu_budget_bytes = kv_options.gpu_budget_bytes;
            kv.ram_budget_bytes = cuda_has_separate_memory() ? kv_options.ram_budget_bytes : 0;
            kv.buffer_bytes = std::min<std::size_t>(kv.buffer_bytes, kv.gpu_budget_bytes / 4);
            kv.directory = kv_options.directory;
            execution->qsa_kv_store = std::make_shared<KvOffloadStore>(std::move(kv));
        }
        state = with_loaded_cuda_model(*execution, options, true,
            [&](auto& model, auto& components, auto, auto) -> State {
                using Model = std::decay_t<decltype(model)>;
                if constexpr (!std::is_same_v<Model, Qwen4CausalLm> && !std::is_same_v<Model, Hy4CausalLm>)
                    if (execution->qsa_kv_store) throw std::invalid_argument("CUDA streamed KV requires sparse attention");
                if(std::getenv("TYLOINFER_TRACE_ENGINE_LOAD"))std::cerr<<"cuda_engine_load phase=model_loaded\n";
                return std::make_unique<CudaEngineState<Model>>(execution, std::move(model), std::move(components), config,weight_memory);
            });
        return visit([&](auto& state) {
            EngineInfo info;
            info.max_requests = state.batching ? config.continuous_batch.max_sequences : 1;
            info.max_context = state.language.max_position_embeddings();
            info.vocab_size = static_cast<int32_t>(state.language.vocab_size());
            info.model_type = state.language.model_type();
            using Language=std::decay_t<decltype(state.language)>;
            info.decisions=std::is_same_v<Language,Qwen35CausalLm> || std::is_same_v<Language,Qwen4CausalLm>;
            info.multimodal = state.components.grid_vision.has_value() ||
                state.components.deepseek_vision.has_value() ||
                state.components.radio_vision.has_value() || bool(state.components.composite);
            info.reload = true;
            const auto caps = cuda_runtime_capabilities(state.components.graph, state.components.plan,
                state.components.state(), info.multimodal, bool(state.components.composite));
            info.duplex = caps.full_duplex;
            info.capabilities = {state.components.graph.architecture, caps.text,
                caps.image_input, caps.video_input, caps.audio_input, caps.audio_output,
                caps.full_duplex, caps.mtp, "model-graph+cuda-adapters"};
            auto text = TextProcessor::load(*state.language.source, options.tokenizer_model,
                info.vocab_size, info.model_type);
            info.stop_token_ids = model_stop_token_ids(
                load_model_config_json(*state.language.source, options.config_path), info.vocab_size);
            return std::pair{std::move(info), std::move(text)};
        });
    }
    bool exclusive() const {
        return std::visit([](const auto& state) { return state && state->exclusive(); }, state);
    }
    void unload() {
        state = std::unique_ptr<CudaEngineState<Qwen35CausalLm>>{};
    }
};
} // namespace engine_detail

std::unique_ptr<Engine> load_cuda_engine(CudaEngineOptions options) {
    return std::make_unique<EngineInstance<engine_detail::CudaBackend>>(std::move(options));
}
} // namespace mfq::cuda
