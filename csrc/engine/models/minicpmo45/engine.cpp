#include "engine/models/minicpmo45/engine.h"
#include "engine/models/minicpmo45/duplex.h"
#include "engine/models/minicpmo45/text/model.h"
#include "engine/requests/processor.h"
#include "engine/state/session/sessions.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <variant>

namespace tyloi::cuda::minicpmo45 {

Engine::Engine(std::unique_ptr<models::CausalModel> model, std::unique_ptr<tyloi::Executor> executor,
        tyloi::engine::Config config, std::shared_ptr<const ModelSource> source,
        std::shared_ptr<weights::Memory> memory, std::unique_ptr<weights::Cache> weights,
        std::unique_ptr<models::State> state, std::unique_ptr<tyloi::state::Sessions> sessions,
        std::unique_ptr<tyloi::engine::TextProcessor> text, models::ForwardOptions forward,
        std::unique_ptr<tyloi::Executor::Graph> graph, std::unique_ptr<tyloi::engine::Batch> batch,
        std::unique_ptr<tyloi::Executor> predictor_execution)
    : tyloi::engine::Engine(std::move(model), std::move(executor), std::move(config), std::move(source),
        std::move(memory), std::move(weights), std::move(state), std::move(sessions), std::move(text),
        std::move(forward), std::move(graph), std::move(batch), std::move(predictor_execution)) {}

Engine::~Engine() {
    try { shutdown(); }
    catch (const std::exception& error) {
        std::cerr << "runtime_unload_cleanup error=" << error.what() << '\n';
        std::terminate();
    } catch (...) {
        std::cerr << "runtime_unload_cleanup error=unknown\n";
        std::terminate();
    }
}

tyloi::engine::ControlResult Engine::control_command(tyloi::engine::ControlRequest request) {
    const bool start = std::holds_alternative<MfqDuplexSessionParams>(request);
    const bool step = std::holds_alternative<MfqDuplexStepInput>(request);
    const bool stop = std::holds_alternative<tyloi::engine::StopDuplex>(request);
    if (!start && !step && !stop)
        return tyloi::engine::Engine::control_command(std::move(request));
    if (!ready()) throw std::runtime_error("engine is unloaded");
    auto scope = enter();
    auto* model = dynamic_cast<const text::Model*>(model_.get());
    if (!model) throw std::invalid_argument("model has no MiniCPM-o duplex component");
    if (!duplex_) duplex_ = std::make_unique<Duplex>(*model);
    drain();
    if (start) {
        const auto& value = std::get<MfqDuplexSessionParams>(request);
        budget_->reserve(duplex_->growth(value, forward_options_));
        duplex_->start(value, forward_options_);
        duplex_active_ = true;
        return std::monostate{};
    }
    if (step) {
        const auto& value = std::get<MfqDuplexStepInput>(request);
        budget_->reserve(duplex_->growth(value, forward_options_));
        return duplex_->step(value, forward_options_);
    }
    duplex_->stop();
    duplex_active_ = false;
    return std::monostate{};
}

bool Engine::exclusive() const { return duplex_active_; }

void Engine::unload_model() {
    if (duplex_) {
        auto scope = enter();
        drain();
        duplex_->stop();
        duplex_.reset();
        duplex_active_ = false;
    }
    tyloi::engine::Engine::unload_model();
}

} // namespace tyloi::cuda::minicpmo45
