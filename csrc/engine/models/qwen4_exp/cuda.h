#pragma once

#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/qwen4_exp/state.h"
#include <memory>
#include <optional>
#include <vector>

namespace tyloi::models::qwen4_exp {
struct Qwen4DecodeGraph;
class Qwen4BatchDecodeGraph;
class Qwen4BatchDecodeGraph {
  public:
    struct Plan {
        std::vector<int> signature;
        std::vector<int64_t> capacities;
        std::vector<std::pair<Device, std::size_t>> growth;
        std::unique_ptr<tyloi::ops::Qsa::Graph> attention;
        bool rebuild = false;
    };
    explicit Qwen4BatchDecodeGraph(Model&);
    ~Qwen4BatchDecodeGraph();
    tyloi::StepSequence<Plan> plan(std::vector<State*>, bool greedy);
    tyloi::StepSequence<::tyloi::Tensor> run(
        const std::vector<State*>&,
        const ::tyloi::Tensor&,const ::tyloi::Tensor&,bool greedy,const Plan&,bool prepared);
    void clear();
    int64_t captures() const;
    int64_t replays() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> graph_;
};

} // namespace tyloi::models::qwen4_exp
