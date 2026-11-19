#pragma once

#include "engine/ops/ops.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace tyloi::state::qsa { struct Session; }

namespace tyloi::ops {

/** Immutable projection and normalization weights for pooled sparse attention. */
struct QsaWeights {
    std::vector<Tensor> input_projection;
    Tensor output, query_norm, key_norm, index_query_norm, index_key_norm;
    bool has_gate = false;
};

/** Geometry and rotary policy for pooled sparse attention. */
struct QsaConfig {
    std::int64_t heads, kv_heads, width, index_heads, index_width, pool, budget, maximum;
    double eps;
    std::int64_t rotary;
    double rope_base;
    std::vector<std::int64_t> sections;
    bool interleaved;
    YarnScaling yarn;
    /** Execution target, independent of host-resident projection weights. */
    Device device = Device(kCUDA, 0);
    state::KvQuantization quantization{};
};

/** Execute pooled sparse attention using an independently owned request state. */
class Qsa {
public:
    /** Own attention cache and backend resources for one request. */
    class State {
    public:
        explicit State(const Qsa& model);
        ~State();
        State(State&&) noexcept;
        State& operator=(State&&) noexcept;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        friend class Qsa;
    };

    /** Keep host staging capacity borrowed during prefill, within the supplied states' lifetime. */
    class HostLoan {
    public:
        HostLoan(std::span<State* const> layers, std::size_t base_bytes, std::int64_t end, std::int64_t rows);
        ~HostLoan();
        HostLoan(const HostLoan&) = delete;
        HostLoan& operator=(const HostLoan&) = delete;
        std::size_t bytes() const noexcept { return bytes_; }
        /** Reclaim host memory before allocating the input staging buffers. */
        void prepare();
        /** Retire completed GPU images before returning the expert workspace. */
        void collect();
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        std::size_t bytes_;
    };

    /** Plan and prepare graph cache layouts for transient request states. */
    class Graph {
    public:
        /** Describe one request and its attention layers in execution order. */
        struct Request {
            std::vector<std::pair<const Qsa*, State*>> layers;
            std::int64_t position = 0, maximum = 0, reserve = 0;
        };
        /** Describe a request's capacity, private unmanaged growth and pending append kind. */
        struct Row { std::int64_t capacity = 0; std::size_t growth = 0; int pending_kind = 0; };
        /** Borrow the supplied operators and states for this plan's lifetime. */
        Graph(std::vector<Request> requests, bool independent);
        ~Graph();
        Graph(Graph&&) noexcept;
        Graph& operator=(Graph&&) noexcept;
        Graph(const Graph&) = delete;
        Graph& operator=(const Graph&) = delete;
        /** Report graph eligibility under the active execution policy. */
        bool eligible(bool allow_pending = false) const;
        /** Report whether a single-request graph has reusable managed backing. */
        bool reusable() const;
        /** Report whether startup warmup has eligible managed attention backing. */
        bool warmup_supported() const;
        /** Plan the window without allocating backend backing. */
        void admit(std::int64_t tokens = 1, std::int64_t previous_capacity = 0, bool hidden = false);
        /** Pending append identity for the admitted window, or zero for dense caches. */
        int pending_kind() const;
        /** Inspect a pending append and its captured storage before advancing the cache. */
        std::pair<int, std::vector<Tensor>> pending(std::int64_t tokens, std::int64_t offset = 0) const;
        /** Report per-device backing required by this plan without allocating it. */
        std::vector<std::pair<Device, std::size_t>> growth() const;
        /** Materialize the managed backing after its growth has been admitted. */
        void materialize();
        /** Report whether single-request graph storage must be retired. */
        bool clear_required() const;
        /** Complete the numeric layout after any previous graph is retired. */
        void layout();
        /** Return the planned capacity and QSA growth for each request. */
        const std::vector<Row>& rows() const;
        /** Owning handles for fixed pending cache storage referenced by capture. */
        std::vector<Tensor> storage() const;
        /** Prepare one attention layer at its original execution boundary. */
        void prepare(std::size_t request_index, std::size_t qsa_index, const Tensor& positions) const;
        /** Prepare all attention layers of a request in order. */
        void prepare(std::size_t request_index, const Tensor& positions) const;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    /** Own the attention state in a session snapshot. */
    struct Snapshot { Tensor keys, values, index; std::shared_ptr<const tyloi::state::qsa::Session> ram; };

    /** Construct attention from its immutable weights and geometry. */
    Qsa(QsaWeights weights, QsaConfig config);
    ~Qsa();
    Qsa(Qsa&&) noexcept;
    Qsa& operator=(Qsa&&) noexcept;
    Qsa(const Qsa&) = delete;
    Qsa& operator=(const Qsa&) = delete;

    /** Return the committed request cache position. */
    std::int64_t position(const State&) const;
    /** Report whether managed backing is active for the request. */
    bool managed_memory(const State&) const;
    /** Return the graph-compatible managed cache capacity. */
    std::int64_t graph_capacity(const State&) const;
    /** Borrow stable graph state slots from the request. */
    std::vector<Tensor*> graph_state(State&) const;
    /** Activate already admitted managed cache storage. */
    void activate_memory(State&) const;
    /** Estimate additional cache backing for a token span. */
    std::size_t growth_bytes(const State&, std::int64_t tokens) const;
    /** Plan cache allocation for a batch window, including a pending batch reset. */
    std::size_t allocation_bytes(const State&, std::int64_t batch, std::int64_t tokens, bool reset) const;
    /** Plan temporary copies needed when leaving an unmanaged graph cache. */
    std::size_t exit_growth(const State&, std::int64_t tokens) const;
    /** Plan copies needed to restore the current unmanaged graph cache. */
    std::size_t graph_exit_growth_bytes(const State&) const;
    /** Estimate backing needed for the requested graph capacity. */
    std::size_t graph_growth_bytes(const State&, std::int64_t capacity) const;
    /** Report whether this request supports session snapshots. */
    bool supports_session(const State&) const;
    /** Validate a snapshot against the request and target device. */
    void validate_session(const State&, const Snapshot&, std::int64_t count, Device device) const;
    /** Hold the RAM target until all restored layers are hydrated. */
    std::shared_ptr<void> reserve_session(State&, const tyloi::state::qsa::Session&) const;
    /** Estimate fresh backing needed by a staged session restore. */
    std::size_t fresh_session_growth(const State&, std::int64_t count) const;
    /** Stage a validated snapshot into an independent request state. */
    void stage_session(State&, const Snapshot&, std::int64_t count, Device device) const;
    /** Return the logical snapshot bytes for the given prefix. */
    std::size_t session_bytes(const State&, std::int64_t count) const;
    /** Capture a committed prefix and add its bytes to the caller's total. */
    void capture_session(const State&, Snapshot&, std::int64_t count, std::size_t& bytes) const;
    /** Restore a validated snapshot into request cache storage. */
    void restore_session(State&, const Snapshot&, std::int64_t count) const;
    /** Enumerate unmanaged resident tensors for request telemetry. */
    std::vector<Tensor> resident_cache_tensors(const State&) const;
    /** Enumerate request and backend cache backings for graph keep-alive. */
    std::vector<Tensor> storage(const State&) const;
    /** Return graph cache to its ordinary sequence representation. */
    void leave_graph(State&) const;
    /** Reset the request cache and graph views. */
    void reset(State&) const;
    /** Prepare graph cache storage and pooled positions. */
    void prepare_graph(State&, const Tensor& positions, std::int64_t capacity,
        std::int64_t pending_tokens = 0) const;
    /** Advance the request after a graph replay. */
    void advance_graph(State&, std::int64_t tokens, std::int64_t attention_rows = -1) const;
    /** Truncate request cache to the accepted prefix. */
    void truncate(State&, std::int64_t keep) const;
    /** Project input rows under the configured grouped policy. */
    std::vector<Tensor> project_inputs(const Tensor&) const;
    /** Project attention output to model width. */
    Tensor project_output(const Tensor&) const;
    /** Execute attention using request-owned cache and optional projected inputs. */
    Tensor forward(State&, const Tensor& hidden, const Tensor& positions, const Tensor& full_positions,
        bool cache, std::vector<Tensor>* selection_trace = nullptr, std::vector<Tensor> projections = {},
        bool project_output = true, bool history_only = false) const;
    /** Execute attention using prepared graph cache storage. */
    Tensor forward_graph(State&, const Tensor& hidden, const Tensor& positions, const Tensor& cache_positions,
        std::vector<Tensor> projections = {}, bool project_output = true, bool history_only = false,
        bool query_tail = false) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tyloi::ops
