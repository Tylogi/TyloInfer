#pragma once
#include "engine/backends/cuda/executor/graph/registry.h"
#include "engine/tensor/tensor.h"
#include "engine/executor.h"
#include <span>
#include <cuda_runtime_api.h>
#include <functional>
#include <exception>
#include <memory>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace tyloi::cuda {
class StreamHandle;
class Context;
class Graph;
class GraphAllocationPlan;
void wait_route_publication(const uint32_t* flag,cudaStream_t stream,int timeout_ms=60000);
void publish_mapped_flag(uint32_t* flag) noexcept;
void check_stream_progress(cudaStream_t stream);
void check_route_submission(cudaStream_t stream);
// A graph stream follows the current work stream on entry and keeps its event private.
class DecodeStream {
public:
    DecodeStream();
    ~DecodeStream();
    void require_current() const;
    std::unique_ptr<tyloi::Executor::Scope> enter();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Resident and codec-free streamed KV share a model graph with host I/O tasks.
// Reads requiring a foreground device codec retain explicit host boundaries;
// expert operators keep their GPU body and host tasks within each submission.
class DecodeWindow {
public:
    struct Task {
        const void* identity;
        std::function<void()> reset, serve, finish, cancel;
        std::function<void()> cleanup;
        std::function<void()> release_warm_output;
    };
    struct GpuPhase {
        int from_layer,layer;
        std::string from,to;
        uint64_t samples;
        int64_t total_ns,min_ns,max_ns;
        int from_tile=-1,tile=-1;
    };
    explicit DecodeWindow(cudaStream_t stream,bool layer_profile=false,bool independent_sequences=false);
    explicit DecodeWindow(bool layer_profile,bool independent_sequences=false);
    bool independent_sequences() const noexcept {return independent_sequences_;}
    bool layer_profiling() const noexcept {return layer_profile_;}
    ~DecodeWindow();
    DecodeWindow(const DecodeWindow&)=delete;
    DecodeWindow& operator=(const DecodeWindow&)=delete;
    // Each pass is synchronous: submit the model body, finish it, restore
    // recurrent state, then release warm outputs before the next pass.
    class Capture {
    public:
        ~Capture();
        bool next();
        bool finish();
        bool retry(std::exception_ptr failure);
        void pause();
        std::vector<std::pair<Device,std::size_t>> growth() const;
        void resume();
        void release_warm_outputs();
        void complete();
    private:
        friend class DecodeWindow;
        explicit Capture(DecodeWindow& window);
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    std::unique_ptr<Capture> begin_capture();
    static std::vector<std::pair<Device,std::size_t>> fixed_growth(
        Device device,bool layer_profile,bool compact_sampling,std::size_t signals);
    void prepare_pool();
    Tensor buffer(const std::vector<int64_t>& shape,const TensorOptions& options,bool zero=false);
    void retain(std::span<const Tensor> owners);
    void retain_result(Tensor value);
    bool matches(std::span<const Tensor> owners) const;
    void set_compact_sampling(bool enabled,const char* role,int rows);
    Tensor sample(const Tensor& logits,const Tensor& greedy_host,const Tensor& compact_host,int rows);
    class Signal {
    public:
        ~Signal();
        void reset();
        void wait_route();
        void wait_device();
        void set_mapped_rows(bool mapped) noexcept;
        void publish(bool mapped);
        void cancel() noexcept;
    private:
        friend class DecodeWindow;
        explicit Signal(DecodeWindow& window);
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    std::unique_ptr<Signal> signal();
    std::unique_ptr<tyloi::Executor::Scope> transfer_scope();
    void enroll(Task task);
    // The executor retains this stream's pool across windows; each captured
    // window also owns it until its final GPU submission has completed.
    void prepare_branch_stream(const StreamHandle& stream,std::shared_ptr<void>& owner);
    // Finish the current GPU segment before running host I/O. During graph
    // preparation only the supplied placeholder buffers are consumed.
    void host_barrier(std::function<void()> serve);
    // Update the next segment at a completed host barrier, before its launch.
    // Zero disables a kernel while preserving the captured dependency edges.
    void set_kernel_rows(std::size_t segment,cudaGraphNode_t node,unsigned rows);
    template<class T,class Factory> std::shared_ptr<T> workspace(const void* identity,Factory create) {
        auto& owner=workspaces_[identity];if(!owner)owner=create();return std::static_pointer_cast<T>(owner);
    }
    void run();
    // Enqueue pure-device graphs without a host wait. Host tasks, I/O
    // boundaries or window timing require run() and return false here.
    bool enqueue_if_device_only();
    bool can_enqueue_device_only() const {
        return valid() && !failed_ && tasks_.empty() && barriers_.empty() && !layer_profile_;
    }
    void mark(int layer,const char* phase);
    void mark_current(const char* phase) {mark(profile_layer_,phase);}
    int set_profile_tile(int tile) noexcept {const int previous=profile_tile_;profile_tile_=tile;return previous;}
    void set_profile_geometry(const char* role,int rows) {profile_role_=role;profile_rows_=rows;}
    void set_profile_layer_kind(int layer,const char* kind) {if(layer_profile_)profile_kinds_[layer]=kind;}
    std::vector<GpuPhase> gpu_phases() const;
    double capture_ms() const {return capture_ns_/1e6;}
    std::size_t nodes() const;
    std::size_t segments() const {return barriers_.size()+1;}
    bool valid() const { const auto* graph=graphs_.find(0,1);return graph && graph->valid() && !recording_segments_; }
    static DecodeWindow* recording() noexcept;
private:
    cudaStream_t stream_;
    Capture* capture_=nullptr;
    void prepare_recording_branch(const std::shared_ptr<Context>& context,const StreamHandle& stream);
    struct BranchPool;
    std::vector<std::shared_ptr<BranchPool>> branch_pools_;
    GraphRegistry graphs_;
    std::unique_ptr<GraphAllocationPlan> allocation_plan_;
    std::unique_ptr<Graph> pool_;
    struct Ownership;
    std::unique_ptr<Ownership> ownership_;
    std::vector<Tensor> buffers_;
    std::shared_ptr<Stream> transfer_;
    struct Sampling;
    std::unique_ptr<Sampling> sampling_;
    struct LayerPoint;
    struct ProfileClock;
    std::unique_ptr<ProfileClock> profile_clock_;
    std::vector<std::unique_ptr<LayerPoint>> layer_points_;
    std::map<std::tuple<int,int,std::string>,LayerPoint*> layer_point_index_;
    int profile_tile_=-1;
    int profile_layer_=-1;
    bool layer_profile_;
    bool independent_sequences_;
    std::string profile_role_="unknown";
    int profile_rows_=1;
    std::map<int,std::string> profile_kinds_;
    void observe_layers();
    std::vector<Task> tasks_;
    std::map<const void*,std::shared_ptr<void>> workspaces_;
    struct Barrier {std::size_t task_end;std::function<void()> serve;};
    std::vector<Barrier> barriers_;
    bool recording_segments_=false;
    bool failed_=false;
    int64_t capture_ns_=0,replay_ns_=0,reset_ns_=0,launch_ns_=0,serve_ns_=0,wait_ns_=0,finish_ns_=0,profile_ns_=0;
    uint64_t replays_=0,async_replays_=0;
    uint64_t dropped_profiles_=0;
};
}
