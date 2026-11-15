#pragma once

#include "engine/models/common/mtp.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/qwen4_exp/config.h"

namespace tyloi::weights { struct Loader; }
namespace tyloi::models { struct LoadOptions; }

namespace tyloi::ops { class Residual; }

namespace tyloi::models::qwen4_exp {
struct Layer;
struct LayerState;
struct Predictor;

struct PredictorState final : tyloi::models::MtpState {
    explicit PredictorState(const Predictor& owner, bool initialize_layers = true);
    ~PredictorState() override;
    int64_t position() const noexcept override;
    std::vector<Tensor> storage() const override;
    std::unique_ptr<tyloi::models::Restore> prepare_restore(const MtpSessionState&) override;

    const Predictor& owner;
    std::vector<std::unique_ptr<LayerState>> layer_states;
    std::vector<::tyloi::Tensor> positions;
    std::vector<int64_t> lengths;
    int64_t batch = 0;
};

struct Predictor final : tyloi::models::Mtp {
    tyloi::models::qwen4_exp::Config config;
    ::tyloi::Tensor embedding, output;
    double softcap = 0;
    ::tyloi::Device device;
    ::tyloi::Tensor embedding_norm, hidden_norm;
    ::tyloi::Tensor embedding_fusion, hidden_fusion;
    std::unique_ptr<tyloi::ops::Residual> final_mixer;
    std::vector<std::unique_ptr<Layer>> layers;

    Predictor();
    ~Predictor() override;
    Predictor(Predictor&&) noexcept;
    Predictor& operator=(Predictor&&) noexcept;

    static std::optional<Predictor> load_if_present(const tyloi::weights::Loader& weights,
                                                      const tyloi::models::LoadOptions& options,
                                                      const tyloi::models::qwen4_exp::Config &main,
                                                      ::tyloi::Tensor embedding, ::tyloi::Tensor output,
                                                      double softcap);

    Tensor embed(Tensor ids) const;
    Tensor logits(Tensor hidden) const;

    std::unique_ptr<tyloi::models::MtpState> make_state() const override;
    void reset(tyloi::models::MtpState&, int64_t next_batch = 1) const override;
    std::vector<std::pair<Device, std::size_t>> growth(const tyloi::models::MtpState&,
        const Tensor& hidden, const Tensor& ids, const Tensor& positions,
        const tyloi::models::ForwardOptions&) const override;

    std::pair<Tensor, Tensor> evaluate(tyloi::models::MtpState&, const Tensor &hidden,
                                       const Tensor &ids, int64_t depth = 0, bool cache = true,
                                       const Tensor &supplied_positions = {},
                                       const Tensor &supplied_embeddings = {},
                                       bool history_only = false) const;

    tyloi::models::MtpStep step(tyloi::models::MtpState&, const tyloi::Tensor& hidden, const tyloi::Tensor& ids,
                                const tyloi::Tensor& positions = {},
                                const tyloi::models::ForwardOptions& = {}) const override;
    void history_step(tyloi::models::MtpState&, Tensor hidden, Tensor ids, Tensor positions = {},
                      bool scalar_chain = false) const;

    void trim_cache_to(tyloi::models::MtpState&, int64_t position) const override;

    bool teacher_forced_prompt_prime() const noexcept override;

    bool target_bootstrap_decode() const noexcept override;

    bool preserve_output_dtype() const noexcept override;
    std::int64_t prefill_chunk_size(std::int64_t requested) const noexcept override {
        return std::min<std::int64_t>(1024, requested);
    }
    bool stream_prompt_history(std::int64_t tokens) const override;
    bool host_prompt_history(std::int64_t tokens, std::size_t available,
                             std::size_t raw_row_bytes) const override;
    bool bootstrap_before_publication() const override;
    std::size_t predictor_history_merge_limit() const override { return 128; }
    std::size_t predictor_history_flush_threshold() const override;
    std::size_t predictor_parked_history_flush_threshold() const override;
    std::size_t prefill_workspace_bytes(std::int64_t tokens) const override;
    tyloi::models::MtpStep prefill(tyloi::models::MtpState&, const tyloi::Tensor& hidden,
        const tyloi::Tensor& ids, const tyloi::Tensor& positions,
        const tyloi::models::ForwardOptions& options) const override;
    std::unique_ptr<tyloi::engine::mtp::DepthPolicy> depth_policy(
        int maximum_depth, tyloi::engine::mtp::PolicyState& state) const override;
    bool supports_session_state(const tyloi::models::MtpState&) const noexcept override;
    MtpSessionState capture_session_state(const tyloi::models::MtpState&, int64_t,const tyloi::Tensor&) const override;
};

} // namespace tyloi::models::qwen4_exp
