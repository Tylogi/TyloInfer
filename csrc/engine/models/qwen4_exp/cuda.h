#pragma once

#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/qwen4_exp/state.h"
#include <memory>
#include <optional>
#include <vector>

namespace tyloi::models::qwen4_exp {
struct Qwen4DecodeGraph;
class Qwen4BatchDecodeGraph;
// A request owns recurrent, positional and attention state; weights stay shared.
class Qwen4BatchRequestState {
  public:
    explicit Qwen4BatchRequestState(Model&);
    ~Qwen4BatchRequestState();
    Qwen4BatchRequestState(const Qwen4BatchRequestState&) = delete;
    Qwen4BatchRequestState& operator=(const Qwen4BatchRequestState&) = delete;
    int64_t position() const;
    std::size_t resident_bytes() const;
    std::size_t initial_growth_bytes() const;
    void initialize();
    std::size_t prefill_growth_bytes(int64_t tokens) const;
  private:
    struct Impl;
    std::unique_ptr<Impl> state_;
    friend class Qwen4BatchDecodeGraph;
    friend tyloi::StepSequence<std::vector<::tyloi::Tensor>> qwen4_batch_prefill(
        Model&,const std::vector<Qwen4BatchRequestState*>&,
        const std::vector<::tyloi::Tensor>&, bool);
    friend ::tyloi::Tensor qwen4_batch_decode(
        Model&, const std::vector<Qwen4BatchRequestState*>&,
        ::tyloi::Tensor, const ::tyloi::Tensor&);
};
tyloi::StepSequence<std::vector<::tyloi::Tensor>> qwen4_batch_prefill(
    Model&,const std::vector<Qwen4BatchRequestState*>&,
    const std::vector<::tyloi::Tensor>&, bool);
::tyloi::Tensor qwen4_batch_decode(
    Model&, const std::vector<Qwen4BatchRequestState*>&,
    ::tyloi::Tensor ids, const ::tyloi::Tensor& host_ids);
class Qwen4BatchDecodeGraph {
  public:
    struct Plan {
        std::vector<int64_t> capacities;
        std::vector<std::pair<Device, std::size_t>> growth;
        std::unique_ptr<tyloi::ops::Qsa::Graph> attention;
        bool rebuild = false;
    };
    explicit Qwen4BatchDecodeGraph(Model&);
    ~Qwen4BatchDecodeGraph();
    tyloi::StepSequence<Plan> plan(std::vector<Qwen4BatchRequestState*>, bool greedy);
    tyloi::StepSequence<::tyloi::Tensor> run(
        const std::vector<Qwen4BatchRequestState*>&,
        const ::tyloi::Tensor&,const ::tyloi::Tensor&,bool greedy,const Plan&,bool prepared);
    void clear();
    int64_t captures() const;
    int64_t replays() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> graph_;
};

} // namespace tyloi::models::qwen4_exp
