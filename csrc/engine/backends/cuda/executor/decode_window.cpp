#include "engine/backends/cuda/native/env.h"
#include "decode_window.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/native/memory.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/backends/cuda/kernels/moe/pipeline.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/graph.h"
#include "engine/backends/cuda/native/device.h"
#include "components/error.h"
#include "engine/backends/cpu/executor/expert_pool.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <exception>
#include <iostream>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace tyloi::cuda {
namespace {
thread_local cudaStream_t flushed_window_stream=nullptr;
thread_local bool has_flushed_window=false;
struct WindowSubmission {
    cudaStream_t previous=flushed_window_stream;
    bool previously_flushed=has_flushed_window;
    ~WindowSubmission(){flushed_window_stream=previous;has_flushed_window=previously_flushed;}
};
void report_window_error(const char* operation,const char* phase) noexcept {
    try {
        std::rethrow_exception(std::current_exception());
    } catch(const std::exception& error) {
        std::fprintf(stderr,"decode_window_error operation=%s phase=%s error=%s\n",operation,phase,error.what());
    } catch(...) {
        std::fprintf(stderr,"decode_window_error operation=%s phase=%s error=unknown\n",operation,phase);
    }
    std::fflush(stderr);
}
}
void check_stream_progress(cudaStream_t stream) {
    const auto status=cudaStreamQuery(stream);
    if(status!=cudaSuccess && status!=cudaErrorNotReady)TYLOINFER_NATIVE_CUDA_CHECK(status);
}
// Adapted from Strata's session_run_token (src/core/session.cpp).
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// MIT license: LICENSES/Strata-MIT.txt
// One WDDM flush submits the complete token graph. Layer waits keep their
// bounded fallback polling; standalone publications still flush immediately.
void check_route_submission(cudaStream_t stream) {
    if(!has_flushed_window || stream!=flushed_window_stream)check_stream_progress(stream);
}
void publish_mapped_flag(uint32_t* flag) noexcept {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *static_cast<volatile uint32_t*>(flag)=1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}
void wait_route_publication(const uint32_t* flag,cudaStream_t stream,int timeout_ms) {
    using Clock=std::chrono::steady_clock;
    const auto start=Clock::now();auto flush=start;
    const auto poll_interval=std::chrono::microseconds(::tyloi::cuda::options::route_poll_interval_us());
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MOE_ROUTE_SPIN");
    const bool spin=!option || option[0]!='0';
    uint32_t spins=0;
    if(*static_cast<const volatile uint32_t*>(flag)<1)check_route_submission(stream);
    while(*static_cast<const volatile uint32_t*>(flag)<1) {
        if(spin) {
            // Strata checks its mapped doorbell after pause and consults the
            // clock every 1024 spins (src/core/session.cpp, MIT license above).
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            _mm_pause();
#else
            std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
            if((++spins&1023u)!=0)continue;
        }
        const auto now=Clock::now();
        if(now-flush>=poll_interval) {
            flush=now;const auto q=cudaStreamQuery(stream);
            if(q!=cudaSuccess && q!=cudaErrorNotReady)TYLOINFER_NATIVE_CUDA_CHECK(q);
            if(q==cudaSuccess && *static_cast<const volatile uint32_t*>(flag)<1)
                throw std::runtime_error("MFQ graph finished before publishing its next layer");
        }
        if(now-start>std::chrono::milliseconds(timeout_ms))
            throw std::runtime_error("MFQ graph route publication stalled");
        if(!spin)std::this_thread::yield();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}
namespace { thread_local DecodeWindow* active=nullptr; }
struct DecodeWindow::LayerPoint {
    int layer;
    int tile;
    std::string phase;
    cudaEvent_t event=nullptr;
    uint64_t samples=0;
    int64_t total_ns=0,min_ns=0,max_ns=0;
    LayerPoint(int l,int t,const char* p,bool clock):layer(l),tile(t),phase(p) {
        if(!clock)TYLOINFER_NATIVE_CUDA_CHECK(cudaEventCreate(&event));
    }
    ~LayerPoint(){if(event)cudaEventDestroy(event);}
};
DecodeWindow* DecodeWindow::recording() noexcept { return active; }
struct DecodeWindow::ProfileClock {
    static constexpr std::size_t capacity=32768;
    uint64_t* device=nullptr;
    std::vector<uint64_t> host=std::vector<uint64_t>(capacity);
    ProfileClock(){TYLOINFER_NATIVE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device),capacity*sizeof(uint64_t)));}
    ~ProfileClock(){if(device)(void)cudaFree(device);}
};
struct DecodeStream::Impl {
    StreamHandle stream=current_stream();
    std::unique_ptr<Event> ready;
};
DecodeStream::DecodeStream():impl_(std::make_unique<Impl>()) {}
DecodeStream::~DecodeStream() = default;
void DecodeStream::require_current() const {
    const auto current=current_stream();
    if(current.device_index()!=impl_->stream.device_index() || current.stream()!=impl_->stream.stream())
        throw std::runtime_error("MFQ model window changed execution stream");
}
std::unique_ptr<tyloi::Executor::Scope> DecodeStream::enter() {
    if(!impl_->ready)impl_->ready=std::make_unique<Event>();
    impl_->ready->record(current_stream().stream());
    struct Scope final : tyloi::Executor::Scope {
        StreamGuard guard;
        Scope(const StreamHandle& stream,const Event& ready):guard(stream) {
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),ready.get(),0));
        }
    };
    return std::make_unique<Scope>(impl_->stream,*impl_->ready);
}
DecodeWindow::DecodeWindow(bool layer_profile,bool independent_sequences)
    :DecodeWindow(current_stream().stream(),layer_profile,independent_sequences) {}
DecodeWindow::DecodeWindow(cudaStream_t stream,bool layer_profile,bool independent_sequences)
    :stream_(stream),graphs_(stream),layer_profile_(layer_profile),independent_sequences_(independent_sequences) {
    // External CUDA timing events split Windows graph submissions. Device
    // timestamps preserve the graph's submission and use the same clock as
    // Strata's verifier profiler. This storage exists only when tracing.
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_TRACE_LAYER_TIMINGS_CLOCK");
    if(layer_profile_ && (!option || option[0]!='0'))profile_clock_=std::make_unique<ProfileClock>();
}
void DecodeWindow::mark(int layer,const char* phase) {
    profile_layer_=layer;
    if(!layer_profile_)return;
    if(const auto* scope=tyloi::cuda::env_lookup("TYLOINFER_TRACE_LAYER_TIMINGS");scope && std::strcmp(scope,"4")==0 &&
        std::strncmp(phase,"qsa_cache_",10)!=0 && std::strcmp(phase,"window_begin")!=0 && std::strcmp(phase,"window_end")!=0)return;
    // Scope 2 keeps the MFE phase boundaries. Timing every small operation
    // measurably perturbs Windows graph replay; ordinary execution adds none.
    if(const auto* scope=tyloi::cuda::env_lookup("TYLOINFER_TRACE_LAYER_TIMINGS");scope &&
            (std::strcmp(scope,"2")==0 || std::strcmp(scope,"3")==0)) {
        constexpr const char* phases[]={"window_begin","window_end","hot_begin","hot_end",
            "dma_end","transfer_gate_up_end","cpu_wait_begin","transfer_end"};
        // Scope 3 separates the attention/residual work, route publication and
        // input preparation (including the mapped host-plan wait). Scope 2
        // retains its existing event count for historical comparisons.
        constexpr const char* boundaries[]={"attention_core_begin","attention_core_end",
            "ffn_pre_end","expert_begin","shared_end","ple_begin","ple_end","gate_up_end","down_end",
            "cpu_merge_begin","cpu_merge_end","resident_gate_up_end","resident_down_end","stream_copy_join","stream_gate_up_join","stream_down_join",
            "gdn_input_end","gdn_recurrence_end","qsa_input_end","qsa_attention_end",
            "cpu_input_publish_end","route_publish_end","input_fusion_end","head_begin","head_norm_end","head_projection_end"};
        if(std::none_of(std::begin(phases),std::end(phases),
                [&](const char* allowed){return std::strcmp(phase,allowed)==0;}) &&
            (std::strcmp(scope,"3")!=0 || std::none_of(std::begin(boundaries),std::end(boundaries),
                [&](const char* allowed){return std::strcmp(phase,allowed)==0;})))return;
    }
    if(active!=this)throw std::logic_error("MFQ layer timing outside window recording");
    const auto key=std::make_tuple(layer,profile_tile_,std::string(phase));
    auto found=layer_point_index_.find(key);
    if(found==layer_point_index_.end()) {
        if(profile_clock_ && layer_points_.size()>=ProfileClock::capacity)
            throw std::runtime_error("GPU stage profile exceeds timestamp capacity");
        auto point=std::make_unique<LayerPoint>(layer,profile_tile_,phase,bool(profile_clock_));
        found=layer_point_index_.emplace(key,point.get()).first;layer_points_.push_back(std::move(point));
    }
    if(profile_clock_) {
        const auto index=std::find_if(layer_points_.begin(),layer_points_.end(),
            [&](const auto& point){return point.get()==found->second;})-layer_points_.begin();
        record_gpu_timestamp(profile_clock_->device,index,stream_);return;
    }
    cudaStreamCaptureStatus capture;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream_,&capture));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecordWithFlags(found->second->event,stream_,
        capture==cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault));
}
void DecodeWindow::observe_layers() {
    if(profile_clock_ && !layer_points_.empty()) {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(profile_clock_->host.data(),profile_clock_->device,
            layer_points_.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream_));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_));
        // A bad diagnostic sample must not cancel an otherwise completed
        // decode, or partly increment the phase totals for this replay.
        for(std::size_t i=1;i<layer_points_.size();++i) {
            const auto before=profile_clock_->host[i-1],after=profile_clock_->host[i];
            if(!before || after<before) {
                ++dropped_profiles_;
                std::cerr<<"native_layer_gpu_invalid role="<<profile_role_<<" rows="<<profile_rows_
                    <<" index="<<i<<" from_layer="<<layer_points_[i-1]->layer<<" layer="<<layer_points_[i]->layer
                    <<" from="<<layer_points_[i-1]->phase<<" to="<<layer_points_[i]->phase
                    <<" before="<<before<<" after="<<after<<std::endl;
                return;
            }
        }
    }
    for(std::size_t i=1;i<layer_points_.size();++i) {
        auto& current=*layer_points_[i];float ms=0;
        int64_t ns;
        if(profile_clock_) {
            const auto before=profile_clock_->host[i-1],after=profile_clock_->host[i];
            ns=static_cast<int64_t>(after-before);
        } else {
            TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,layer_points_[i-1]->event,current.event));
            ns=static_cast<int64_t>(ms*1e6);
        }
        current.total_ns+=ns;current.min_ns=current.samples ? std::min(current.min_ns,ns) : ns;
        current.max_ns=std::max(current.max_ns,ns);++current.samples;
    }
}
std::vector<DecodeWindow::GpuPhase> DecodeWindow::gpu_phases() const {
    std::vector<GpuPhase> phases;
    for(std::size_t i=1;i<layer_points_.size();++i) {
        const auto& previous=*layer_points_[i-1];const auto& current=*layer_points_[i];
        phases.push_back({previous.layer,current.layer,previous.phase,current.phase,
            current.samples,current.total_ns,current.min_ns,current.max_ns,previous.tile,current.tile});
    }
    return phases;
}
struct DecodeWindow::BranchPool {
    StreamHandle stream;
    std::shared_ptr<Context> context;
    bool owned=false;
    BranchPool(StreamHandle value,GraphAllocationPlan* plan):stream(std::move(value)),context(default_context(stream.device_index())) {
        if(!context->has_graph_pool(stream.stream())) {context->begin_graph_pool(stream.stream());owned=true;}
        else if(plan)owned=plan->release_pool(stream.device_index(),stream.stream());
    }
    ~BranchPool(){if(owned)context->end_graph_pool(stream.stream());}
};
void DecodeWindow::prepare_branch_stream(const StreamHandle& stream,std::shared_ptr<void>& owner) {
    if(active!=this)throw std::logic_error("parallel stream outside window recording");
    if(!stream || stream.stream()==stream_)throw std::invalid_argument("parallel stream must differ from its parent");
    auto pool=std::static_pointer_cast<BranchPool>(owner);
    if(pool && (pool->stream.stream()!=stream.stream() || pool->stream.device_index()!=stream.device_index()))
        throw std::logic_error("parallel stream pool owner changed");
    if(pool && std::find(branch_pools_.begin(),branch_pools_.end(),pool)!=branch_pools_.end())return;
    if(recording_segments_)throw std::logic_error("parallel stream was not prepared before graph capture");
    if(!pool){pool=std::make_shared<BranchPool>(stream,allocation_plan_.get());owner=pool;}
    branch_pools_.push_back(std::move(pool));
    prepare_recording_branch(branch_pools_.back()->context,branch_pools_.back()->stream);
}
struct DecodeWindow::Signal::Impl {
    DecodeWindow& window;
    HostBuffer flags{128,true};
    uint32_t* device=nullptr;
    bool profile=false,mapped_rows=false;
    uint64_t calls=0,route_wait_ns=0,upload_ns=0,notify_ns=0;
    std::chrono::steady_clock::time_point point;
    explicit Impl(DecodeWindow& value):window(value) {
        const auto timing=options::trace_ple_timings();profile=timing && *timing;
        charge_tensor_host_bytes(128);
        TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device),flags.data(),0));
        std::memset(flags.data(),0,128);publish_mapped_flag(host()+16);
    }
    ~Impl() {
        if(profile)std::cerr<<"ple_stage_host mapped_rows="<<mapped_rows<<" calls="<<calls
            <<" route_wait_ns="<<route_wait_ns<<" upload_ns="<<upload_ns<<" notify_ns="<<notify_ns<<'\n';
        tensor_host_bytes.fetch_sub(128);
    }
    void sample(uint64_t& total) {
        if(!profile)return;
        const auto now=std::chrono::steady_clock::now();
        total+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-point).count();point=now;
    }
    uint32_t* host() {return static_cast<uint32_t*>(flags.data());}
};
DecodeWindow::Signal::Signal(DecodeWindow& window):impl_(std::make_unique<Impl>(window)) {}
DecodeWindow::Signal::~Signal() = default;
void DecodeWindow::Signal::reset() {std::memset(impl_->flags.data(),0,128);}
void DecodeWindow::Signal::wait_route() {
    if(impl_->profile)impl_->point=std::chrono::steady_clock::now();
    wait_route_publication(impl_->host(),impl_->window.stream_);
    impl_->sample(impl_->route_wait_ns);
}
void DecodeWindow::Signal::wait_device() {
    signal_mapped_flag(impl_->device,impl_->window.stream_);
    wait_mapped_flag(impl_->device+16,impl_->window.stream_);
}
void DecodeWindow::Signal::set_mapped_rows(bool mapped) noexcept {impl_->mapped_rows=mapped;}
void DecodeWindow::Signal::cancel() noexcept {publish_mapped_flag(impl_->host()+16);}
void DecodeWindow::Signal::publish(bool mapped) {
    impl_->sample(impl_->upload_ns);
    if(mapped)publish_mapped_flag(impl_->host()+16);
    else {
        const auto stream=impl_->window.transfer_->get();
        TYLOINFER_NATIVE_CUDA_CHECK(cudaLaunchHostFunc(stream,[](void* p){publish_mapped_flag(static_cast<uint32_t*>(p));},impl_->host()+16));
        check_stream_progress(stream);
    }
    impl_->sample(impl_->notify_ns);
    if(impl_->profile)++impl_->calls;
}
std::unique_ptr<DecodeWindow::Signal> DecodeWindow::signal() {
    if(!transfer_) {
        int device=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
        transfer_=std::make_shared<Stream>(device);
    }
    return std::unique_ptr<Signal>(new Signal(*this));
}
std::unique_ptr<tyloi::Executor::Scope> DecodeWindow::transfer_scope() {
    if(!transfer_)throw std::logic_error("graph transfer stream is unavailable");
    struct Scope final : tyloi::Executor::Scope {
        StreamGuard guard;
        explicit Scope(const std::shared_ptr<Stream>& stream):guard(StreamHandle(stream->device(),stream->get(),stream)) {}
    };
    return std::make_unique<Scope>(transfer_);
}
struct DecodeWindow::Sampling {
    HostBuffer host{128};
    Tensor control=cuda::zeros({1},TensorOptions().device(kCUDA).dtype(kInt32));
    int previous=-1;
    Sampling() {charge_tensor_host_bytes(host.size());}
    ~Sampling() {tensor_host_bytes.fetch_sub(host.size());}
};
struct DecodeWindow::Ownership {
    std::vector<Tensor> owners;
    std::vector<decode_graph_detail::Layout> layouts;
};
DecodeWindow::~DecodeWindow() {
    // Mapped flags and task owners must survive the last consumer, including
    // a graph released by a failed CPU or I/O phase.
    for(auto& task:tasks_)if(task.cancel)task.cancel();
    (void)cudaStreamSynchronize(stream_);
    if(transfer_)(void)cudaStreamSynchronize(transfer_->get());
    graphs_.clear();
    for(auto& task:tasks_)if(task.cleanup)task.cleanup();
    if((replays_ || async_replays_) && tyloi::cuda::env_lookup("TYLOINFER_TRACE_NATIVE_CUDA_GRAPH"))
        std::cerr<<"native_cuda_window tasks="<<tasks_.size()<<" replays="<<replays_
            <<" async_replays="<<async_replays_<<" capture_ns="<<capture_ns_<<" replay_ns="<<replay_ns_
            <<" reset_ns="<<reset_ns_<<" launch_ns="<<launch_ns_<<" serve_ns="<<serve_ns_
            <<" wait_ns="<<wait_ns_<<" finish_ns="<<finish_ns_<<" profile_ns="<<profile_ns_
            <<" role="<<profile_role_<<" rows="<<profile_rows_
            <<" clock="<<(profile_clock_?1:0)<<" profile_dropped="<<dropped_profiles_<<std::endl;
    for(const auto& phase:gpu_phases())if(phase.samples)
        std::cerr<<"native_layer_gpu from_layer="<<phase.from_layer<<" layer="<<phase.layer
            <<" from="<<phase.from<<" to="<<phase.to<<" samples="<<phase.samples
            <<" total_ns="<<phase.total_ns<<" min_ns="<<phase.min_ns<<" max_ns="<<phase.max_ns
            <<" from_tile="<<phase.from_tile<<" tile="<<phase.tile
            <<" kind="<<(profile_kinds_.count(phase.layer)?profile_kinds_.at(phase.layer):"unknown")<<std::endl;
}
void DecodeWindow::enroll(Task task) {
    if(active!=this)throw std::logic_error("MFQ task outside window capture");
    if(std::any_of(tasks_.begin(),tasks_.end(),[&](const auto& t){return t.identity==task.identity;}))
        throw std::logic_error("one MFQ operator cell was recorded twice in a window");
    tasks_.push_back(std::move(task));
}
std::size_t DecodeWindow::nodes() const {
    std::size_t result=0;
    for(std::size_t i=0;i<segments();++i)if(const auto* graph=graphs_.find(i,1))result+=graph->nodes();
    return result;
}
void DecodeWindow::set_kernel_rows(std::size_t segment,cudaGraphNode_t node,unsigned rows) {
    if(!valid())throw std::logic_error("graph kernel update requires a captured window");
    const auto* graph=graphs_.find(segment,1);
    if(!graph)throw std::out_of_range("graph kernel update has no matching segment");
    graph->set_kernel_rows(node,rows);
}
void DecodeWindow::host_barrier(std::function<void()> serve) {
    if(active!=this)throw std::logic_error("host I/O barrier outside window capture");
    if(recording_segments_)graphs_.end_segment(barriers_.size(),1);
    barriers_.push_back({tasks_.size(),std::move(serve)});
    if(recording_segments_)graphs_.begin(barriers_.size(),1);
}
struct DecodeWindow::Capture::Impl {
    DecodeWindow& window;
    std::unique_ptr<HostRegistrationBlock> registration;
    GraphAllocationPlan& plan;
    StreamHandle stream=current_stream();
    std::shared_ptr<Context> primary=default_context(stream.device_index());
    std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<GraphWarmupScope>> warmup;
    std::unique_ptr<tyloi::Executor::Scope> ffn_scope;
    bool* ffn_serial=nullptr;
    bool previous_ffn_serial=false;
    std::size_t branch_captures=0, warm_tasks=0, warm_barriers=0;
    int pass=-1;
    bool finished=false, complete=false, paused=false, retrying=false;

    explicit Impl(DecodeWindow& owner):window(owner),plan(*owner.allocation_plan_) {
        if(active)throw std::logic_error("nested MFQ window capture");
        registration=std::make_unique<HostRegistrationBlock>();
        active=&window;
    }
    void release_ffn() noexcept {
        if (ffn_serial) *ffn_serial = previous_ffn_serial;
        ffn_serial = nullptr;
        ffn_scope.reset();
    }
    void synchronize() {
        const auto wait=[](const StreamHandle& participant) {
            DeviceGuard guard(participant.device_index());
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(participant.stream()));
        };
        wait(stream);
        for(const auto& pool:window.branch_pools_)wait(pool->stream);
        for(const auto& [context,participant]:plan.participants())wait(participant);
    }
    void end_branches() noexcept {
        while(branch_captures) {
            const auto& pool=window.branch_pools_[--branch_captures];
            pool->context->end_graph_capture(pool->stream.stream());
        }
    }
    ~Impl() {
        release_ffn();
        plan.end();plan.end_warmup();warmup.clear();end_branches();
        primary->graph_strict_warmup(stream.stream(),false);
        for(const auto& pool:window.branch_pools_)pool->context->graph_strict_warmup(pool->stream.stream(),false);
        for(const auto& [context,participant]:plan.participants())context->graph_strict_warmup(participant.stream(),false);
        if(!complete) {
            window.recording_segments_=false;
            for(auto& task:window.tasks_)if(task.cancel)task.cancel();
            window.graphs_.clear();
            (void)cudaStreamSynchronize(window.stream_);
            if(window.transfer_)(void)cudaStreamSynchronize(window.transfer_->get());
            window.failed_=true;
        }
        if(active==&window)active=nullptr;
    }
};
void DecodeWindow::prepare_recording_branch(const std::shared_ptr<Context>& context,const StreamHandle& stream) {
    if(!capture_)return;
    auto& state=*capture_->impl_;
    if(state.pass!=1 && state.pass!=2)return;
    for(const auto& [planned,participant]:state.plan.participants())
        if(participant.device_index()==stream.device_index() && participant.stream()==stream.stream())return;
    context->graph_strict_warmup(stream.stream(),true);
    state.warmup.push_back(std::make_unique<GraphWarmupScope>(context,stream.stream()));
}
DecodeWindow::Capture::Capture(DecodeWindow& window):impl_(std::make_unique<Impl>(window)) {window.capture_=this;}
DecodeWindow::Capture::~Capture() {impl_->window.capture_=nullptr;}
std::vector<std::pair<Device,std::size_t>> DecodeWindow::fixed_growth(
        Device device,bool layer_profile,bool compact_sampling,std::size_t signals) {
    std::size_t gpu=compact_sampling?sizeof(int32_t):0;
    if(signals>SIZE_MAX/128)throw std::overflow_error("CUDA graph signal growth overflows");
    std::size_t host=signals*128;
    if(compact_sampling) {
        if(host>SIZE_MAX-128)throw std::overflow_error("CUDA graph signal growth overflows");
        host+=128;
    }
    const auto* option=env_lookup("TYLOINFER_TRACE_LAYER_TIMINGS_CLOCK");
    if(layer_profile && (!option || option[0]!='0')) {
        constexpr std::size_t clock_bytes=ProfileClock::capacity*sizeof(uint64_t);
        if(host>SIZE_MAX-clock_bytes)throw std::overflow_error("CUDA graph profiling growth overflows");
        gpu+=clock_bytes;host+=clock_bytes;
    }
    std::vector<std::pair<Device,std::size_t>> result;
    if(gpu)result.emplace_back(device,gpu);
    if(host)result.emplace_back(Device{kCPU,0},host);
    return result;
}
void DecodeWindow::set_compact_sampling(bool enabled,const char* role,int rows) {
    if(!sampling_)sampling_=std::make_unique<Sampling>();
    auto& state=*sampling_;
    if(state.previous==int(enabled))return;
    *static_cast<int32_t*>(state.host.data())=enabled?1:0;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(state.control.dense().data_ptr(),state.host.data(),sizeof(int32_t),
        cudaMemcpyHostToDevice,stream_));
    std::cerr<<"runtime_mtp_compact_control role="<<role<<" rows="<<rows<<" enabled="<<int(enabled)<<'\n';
    state.previous=int(enabled);
}
void DecodeWindow::sample(const Tensor& logits,const Tensor& greedy_host,const Tensor& compact_host,int rows) {
    auto flat=logits.dense().reshape({rows,-1}).contiguous();
    if(rows>1 && logits.size(-1)>=64) {
        if(!sampling_)throw std::logic_error("compact sampling was not configured");
        auto compact=sample_top_k_compact_if_cuda(flat,64,sampling_->control.dense().data_ptr<int32_t>());
        TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(compact_host.dense().data_ptr(),compact.data_ptr(),
            rows*64*sizeof(int64_t),cudaMemcpyDeviceToHost,stream_));
    }
    auto greedy=sample_greedy_cuda(flat);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(greedy_host.dense().data_ptr(),greedy.data_ptr(),
        rows*sizeof(int64_t),cudaMemcpyDeviceToHost,stream_));
}
Tensor DecodeWindow::buffer(const std::vector<int64_t>& shape,const TensorOptions& options,bool zero) {
    Tensor result=zero ? cuda::zeros(shape,options) : cuda::empty(shape,options);
    buffers_.push_back(result);
    return result;
}
void DecodeWindow::retain_result(Tensor value) {
    if(value)buffers_.push_back(std::move(value));
}
void DecodeWindow::retain(std::span<const Tensor> owners) {
    auto value=std::make_unique<Ownership>();
    value->owners.assign(owners.begin(),owners.end());
    const auto scratch=Executor::current().storage();
    value->owners.insert(value->owners.end(),scratch.begin(),scratch.end());
    for(const auto& tensor:value->owners)value->layouts.push_back(decode_graph_detail::snapshot_layout(tensor));
    ownership_=std::move(value);
}
bool DecodeWindow::matches(std::span<const Tensor> owners) const {
    if(!ownership_)return false;
    std::vector<Tensor> values(owners.begin(),owners.end());
    const auto scratch=Executor::current().storage();
    values.insert(values.end(),scratch.begin(),scratch.end());
    if(values.size()!=ownership_->layouts.size())return false;
    for(std::size_t i=0;i<values.size();++i)
        if(!decode_graph_detail::same_layout(ownership_->layouts[i],values[i]))return false;
    return true;
}
void DecodeWindow::prepare_pool() {
    pool_=std::make_unique<Graph>();
    pool_->prepare_memory();
    int device=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
    default_context(device)->end_graph_warmup(stream_);
}
std::unique_ptr<DecodeWindow::Capture> DecodeWindow::begin_capture() {
    if(valid())throw std::logic_error("MFQ window is already captured");
    if(!allocation_plan_)allocation_plan_=std::make_unique<GraphAllocationPlan>();
    if(!default_context(current_stream().device_index())->has_graph_pool(stream_))prepare_pool();
    return std::unique_ptr<Capture>(new Capture(*this));
}
bool DecodeWindow::Capture::next() {
    auto& state=*impl_;auto& window=state.window;
    if(state.paused)throw std::logic_error("paused MFQ capture");
    if(state.pass>=0 && !state.finished)throw std::logic_error("unfinished MFQ capture pass");
    if(state.pass==3)return false;
    if(state.retrying) {
        window.tasks_.clear();window.barriers_.clear();state.retrying=false;
    } else {
        if(state.pass==0 || state.pass==1) {window.tasks_.clear();window.barriers_.clear();}
        if(state.pass==2) {
            state.warm_tasks=window.tasks_.size();state.warm_barriers=window.barriers_.size();
            window.tasks_.clear();window.barriers_.clear();
        }
        ++state.pass;
    }
    state.finished=false;
    if(state.pass==0)state.plan.begin();
    if(state.pass==1 || state.pass==2) {
        int device=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
        auto primary=default_context(device);
        primary->graph_strict_warmup(window.stream_,true);
        state.warmup.push_back(std::make_unique<GraphWarmupScope>(primary,window.stream_));
        for(const auto& pool:window.branch_pools_) {
            pool->context->graph_strict_warmup(pool->stream.stream(),true);
            state.warmup.push_back(std::make_unique<GraphWarmupScope>(pool->context,pool->stream.stream()));
        }
        for(const auto& [context,stream]:state.plan.participants()) {
            if(stream.device_index()==state.stream.device_index() && stream.stream()==state.stream.stream())continue;
            if(std::any_of(window.branch_pools_.begin(),window.branch_pools_.end(),[&](const auto& pool) {
                return pool->stream.device_index()==stream.device_index() && pool->stream.stream()==stream.stream();
            }))continue;
            context->graph_strict_warmup(stream.stream(),true);
            state.warmup.push_back(std::make_unique<GraphWarmupScope>(context,stream.stream()));
        }
        state.plan.begin_warmup();
    } else if(state.pass==3) {
        std::vector<StreamHandle> participants;
        for(const auto& [context,stream]:state.plan.participants()) {
            if(stream.device_index()==state.stream.device_index() && stream.stream()==state.stream.stream())continue;
            if(std::any_of(window.branch_pools_.begin(),window.branch_pools_.end(),[&](const auto& pool) {
                return pool->stream.device_index()==stream.device_index() && pool->stream.stream()==stream.stream();
            }))continue;
            participants.push_back(stream);
        }
        window.graphs_.participants(std::move(participants));
        for(const auto& pool:window.branch_pools_) {
            pool->context->begin_graph_capture(pool->stream.stream());++state.branch_captures;
        }
        window.recording_segments_=true;window.graphs_.begin(0,1);
    }
    auto& executor = Executor::current();
    auto mode = Executor::current_ffn();
    mode.serial = true;
    state.ffn_scope = executor.enter(mode);
    state.ffn_serial = &executor.mode().ffn.serial;
    state.previous_ffn_serial = *state.ffn_serial;
    *state.ffn_serial = true;
    return true;
}
bool DecodeWindow::Capture::finish() {
    auto& state=*impl_;auto& window=state.window;
    if(state.pass<0 || state.finished)throw std::logic_error("invalid MFQ capture pass");
    state.release_ffn();
    if(state.pass==3) {
        window.graphs_.end_segment(window.barriers_.size(),1);
        window.recording_segments_=false;state.end_branches();
        if(!window.nodes())throw Error("CUDA decode window body produced no work");
        if(window.tasks_.size()!=state.warm_tasks || window.barriers_.size()!=state.warm_barriers)
            throw std::logic_error("MFQ window task order changed during capture");
    } else {
        state.plan.end_warmup();state.warmup.clear();
        state.synchronize();
    }
    state.finished=true;
    return state.pass!=3;
}
bool DecodeWindow::Capture::retry(std::exception_ptr failure) {
    auto& state=*impl_;auto& window=state.window;
    state.release_ffn();
    try {if(failure)std::rethrow_exception(failure);return false;}
    catch(const GraphAllocationPlan::Miss& miss) {
        if(state.pass!=1 && state.pass!=2)return false;
        state.plan.end_warmup();state.warmup.clear();
        for(auto& task:window.tasks_)if(task.cancel)task.cancel();
        state.synchronize();
        if(window.transfer_)window.transfer_->synchronize();
        state.plan.add(miss);
        state.finished=true;state.retrying=true;
        return true;
    } catch(...) {return false;}
}
void DecodeWindow::Capture::pause() {
    auto& state=*impl_;
    if(state.paused || !state.finished || active!=&state.window)
        throw std::logic_error("invalid MFQ capture pause");
    state.release_ffn();
    state.plan.end();
    if(state.pass==0)state.plan.freeze();
    state.primary->graph_strict_warmup(state.stream.stream(),false);
    for(const auto& pool:state.window.branch_pools_)pool->context->graph_strict_warmup(pool->stream.stream(),false);
    for(const auto& [context,stream]:state.plan.participants())context->graph_strict_warmup(stream.stream(),false);
    active=nullptr;state.registration.reset();state.paused=true;
}
std::vector<std::pair<Device,std::size_t>> DecodeWindow::Capture::growth() const {
    if(!impl_->paused)throw std::logic_error("MFQ capture growth requires pause");
    return impl_->plan.growth();
}
void DecodeWindow::Capture::resume() {
    auto& state=*impl_;
    if(!state.paused || active)throw std::logic_error("invalid MFQ capture resume");
    const auto current=current_stream();
    if(current.device_index()!=state.stream.device_index() || current.stream()!=state.stream.stream())
        throw std::logic_error("MFQ capture resumed on a different stream");
    state.plan.materialize();
    state.registration=std::make_unique<HostRegistrationBlock>();active=&state.window;state.paused=false;
}
void DecodeWindow::Capture::release_warm_outputs() {
    for(auto& task:impl_->window.tasks_)if(task.release_warm_output)task.release_warm_output();
}
void DecodeWindow::Capture::complete() {
    auto& state=*impl_;auto& window=state.window;
    if(state.pass!=3 || !state.finished)throw std::logic_error("incomplete MFQ window capture");
    state.synchronize();
    window.capture_ns_=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-state.started).count();
    state.complete=true;
}
bool DecodeWindow::enqueue_if_device_only() {
    if(!tasks_.empty() || !barriers_.empty() || layer_profile_)return false;
    if(!valid() || failed_)throw std::logic_error("MFQ device graph is not ready");
    std::string error;
    if(!graphs_.find(0,1)->launch(stream_,error)) {
        failed_=true;throw std::runtime_error(error);
    }
    ++async_replays_;
    return true;
}

void DecodeWindow::run() {
    if(failed_ || !valid())throw std::runtime_error("MFQ window is unavailable after a failed execution");
    const HostRegistrationBlock registration;
    const tyloi::cpu::ScopedHostAffinity placement(::tyloi::cuda::options::decode_host_pin()
        ?tyloi::cpu::planned_host_core():-1);
    const auto* trace_option=tyloi::cuda::env_lookup("TYLOINFER_TRACE_MOE_SERVE");
    const bool trace=trace_option && trace_option[0] && trace_option[0]!='0' && (replays_<2 || replays_%64==0);
    // Value 2 adds one summary per replay, including later slow tokens that
    // the sampled verbose trace would otherwise omit. Ordinary runs add no I/O.
    const bool trace_steps=trace_option && trace_option[0]=='2';
    const char* phase="reset";
    const auto checkpoint=[&](const char* next) {
        phase=next;
        if(trace) {
            std::fprintf(stderr,"decode_window operation=replay replay=%llu phase=%s tasks=%zu\n",
                static_cast<unsigned long long>(replays_),phase,tasks_.size());
            std::fflush(stderr);
        }
    };
    try {
        WindowSubmission submission;
        using Clock=std::chrono::steady_clock;
        const auto reset_before=reset_ns_,launch_before=launch_ns_,serve_before=serve_ns_,wait_before=wait_ns_,finish_before=finish_ns_;
        auto previous=Clock::now();const auto started=previous;
        const auto sample=[&](int64_t& total) {
            const auto now=Clock::now();
            total+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-previous).count();previous=now;
        };
        checkpoint("reset");
        for(auto& task:tasks_)task.reset();
        sample(reset_ns_);
        std::string error;
        std::size_t task_begin=0;
        for(std::size_t segment=0;segment<segments();++segment) {
            const auto* graph=graphs_.find(segment,1);
            checkpoint("launch");
            if(!graph->launch(stream_,error))throw std::runtime_error(error);
            checkpoint("submit");
            if(::tyloi::cuda::options::single_window_flush()) {
                check_stream_progress(stream_);
                flushed_window_stream=stream_;has_flushed_window=true;
            }
            sample(launch_ns_);
            checkpoint("serve");
            const auto task_end=segment<barriers_.size()?barriers_[segment].task_end:tasks_.size();
            for(;task_begin<task_end;++task_begin)tasks_[task_begin].serve();
            sample(serve_ns_);
            checkpoint("wait");
            if(!graph->wait_ms(60000))throw std::runtime_error("MFQ model window did not finish");
            sample(wait_ns_);
            if(segment<barriers_.size()) {
                checkpoint("host_io");barriers_[segment].serve();sample(serve_ns_);
            }
        }
        // Only after all GPU consumers finish may adaptation overwrite L1.
        checkpoint("finish");
        for(auto& task:tasks_)if(task.finish)task.finish();
        sample(finish_ns_);
        observe_layers();sample(profile_ns_);
        replay_ns_+=std::chrono::duration_cast<std::chrono::nanoseconds>(previous-started).count();++replays_;
        const auto elapsed=std::chrono::duration<double,std::milli>(previous-started).count();
        if(trace_steps) {
            double gpu_ms=0;
            if(profile_clock_ && !layer_points_.empty()) {
                const auto first=profile_clock_->host.front();
                const auto last=profile_clock_->host[layer_points_.size()-1];
                if(first && last>=first)gpu_ms=double(last-first)/1e6;
            }
            std::fprintf(stderr,"runtime_decode_window_step role=%s rows=%d replay=%llu ms=%.6f reset_ms=%.6f "
                "launch_ms=%.6f serve_ms=%.6f wait_ms=%.6f finish_ms=%.6f gpu_ms=%.6f\n",
                profile_role_.c_str(),profile_rows_,static_cast<unsigned long long>(replays_),elapsed,
                (reset_ns_-reset_before)/1e6,(launch_ns_-launch_before)/1e6,(serve_ns_-serve_before)/1e6,
                (wait_ns_-wait_before)/1e6,(finish_ns_-finish_before)/1e6,gpu_ms);
        }
        if(trace && replays_<=64 && elapsed>=45.0)
            std::cerr<<"runtime_decode_window_slow role="<<profile_role_<<" rows="<<profile_rows_
                <<" replay="<<replays_<<" ms="<<elapsed<<" reset_ms="<<(reset_ns_-reset_before)/1e6
                <<" launch_ms="<<(launch_ns_-launch_before)/1e6<<" serve_ms="<<(serve_ns_-serve_before)/1e6
                <<" wait_ms="<<(wait_ns_-wait_before)/1e6<<" finish_ms="<<(finish_ns_-finish_before)/1e6<<'\n';
        checkpoint("complete");
    } catch(...) {
        report_window_error("replay",phase);
        failed_=true;
        for(auto& task:tasks_)if(task.cancel)task.cancel();
        (void)cudaStreamSynchronize(stream_);throw;
    }
}
}
