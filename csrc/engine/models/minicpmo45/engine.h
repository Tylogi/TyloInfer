#pragma once

#include "engine/engine.h"

#include <memory>

namespace tyloi::cuda::minicpmo45 {

class Duplex;

// MiniCPM owns its duplex session beside the shared text engine.
class Engine final : public tyloi::engine::Engine {
public:
    Engine(std::unique_ptr<models::CausalModel> model, std::unique_ptr<tyloi::Executor> executor,
        tyloi::engine::Config config, std::shared_ptr<const ModelSource> source,
        std::shared_ptr<weights::Memory> memory, std::unique_ptr<weights::Cache> weights,
        std::unique_ptr<models::State> state, std::unique_ptr<tyloi::state::Sessions> sessions,
        std::unique_ptr<tyloi::engine::TextProcessor> text, models::ForwardOptions forward,
        std::unique_ptr<tyloi::Executor::Graph> graph, std::unique_ptr<tyloi::engine::Batch> batch = {},
        std::unique_ptr<tyloi::Executor> predictor_execution = {});
    ~Engine() override;

protected:
    tyloi::engine::ControlResult control_command(tyloi::engine::ControlRequest request) override;
    bool exclusive() const override;
    void unload_model() override;

private:
    std::unique_ptr<Duplex> duplex_;
    bool duplex_active_ = false;
};

} // namespace tyloi::cuda::minicpmo45
