#pragma once

#include "mlx_generation_job.h"
#include "request_executor.h"

namespace tyloi::metal {

// Backend-native resources remain behind the engine, not the communication API.
class MlxEngineModel {
public:
    virtual ~MlxEngineModel() = default;
    virtual void generate(tyloi::engine::InferenceRequest&, MlxGenerationJob&) = 0;
    virtual std::size_t request_capacity() const { return 1; }
    virtual bool can_batch(const tyloi::engine::EngineRequest&) const { return false; }
    virtual void execute(const std::vector<tyloi::engine::RequestId>&) {}
    virtual tyloi::engine::Generation generate_batch(const tyloi::engine::RequestId&,
        tyloi::engine::InferenceRequest&, tyloi::engine::InferenceOutput&) {
        throw std::invalid_argument("continuous batching is unavailable for this Metal model");
        co_return;
    }
    virtual std::int64_t reload(std::int64_t context) = 0;
    virtual tyloi::engine::SessionResult session(const tyloi::engine::SessionCommand&) = 0;
    virtual tyloi::engine::ControlResult control(tyloi::engine::ControlRequest) = 0;
};

class MlxEngine final : public tyloi::engine::Engine {
public:
    MlxEngine(std::unique_ptr<MlxEngineModel> model,
        std::unique_ptr<tyloi::engine::TextProcessor> text,
        tyloi::engine::EngineInfo info);
    ~MlxEngine() override;
    tyloi::engine::EngineInfo info() const override { return info_; }
    tyloi::engine::EngineStatus status() const override;
    tyloi::engine::Admission admit(tyloi::engine::EngineRequest) override;
    void cancel(const tyloi::engine::RequestId&) override;
    tyloi::engine::EngineStepResult step(const std::vector<tyloi::engine::RequestId>&) override;
    tyloi::engine::SessionResult session(const tyloi::engine::SessionCommand&) override;
    tyloi::engine::ControlResult control(tyloi::engine::ControlRequest) override;
    std::int64_t reload(std::int64_t context) override;
    void shutdown() override;

    bool can_batch(const tyloi::engine::EngineRequest& request) const { return model_->can_batch(request); }
    bool exclusive() const { return duplex_active_; }
    bool mtp_available() const { return info_.capabilities.mtp; }
    void execute(const std::vector<tyloi::engine::RequestId>& eligible) { model_->execute(eligible); }
    tyloi::engine::Generation generate(const tyloi::engine::RequestId&,
        tyloi::engine::ExecutionRequest& request);

private:
    std::unique_ptr<MlxEngineModel> model_;
    std::unique_ptr<tyloi::engine::TextProcessor> text_;
    tyloi::engine::EngineInfo info_;
    tyloi::engine::RequestExecutor requests_;
    bool duplex_active_ = false;
};

} // namespace tyloi::metal
