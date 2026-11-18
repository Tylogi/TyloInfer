#pragma once

#include "engine/models/common/mtp.h"

#include <filesystem>

namespace tyloi::models::dflash2 {

struct Config {
    int hidden, vocabulary, layers, intermediate, heads, kv_heads, dimension;
    int block, window, group, rank, candidates, mask;
    float theta, epsilon;
    std::vector<std::int64_t> target_layers;
};

struct Matrix {
    DenseTensor tiles;
    int rows, columns, padded_rows;
    bool q4;
};

struct Layer {
    DenseTensor input_norm, attention_base, q_norm, k_norm, post_norm, mlp_base;
    Matrix dynamic_attention, q, k, v, output, dynamic_mlp, gate, up, down;
};

class Draft;
struct DraftGraph;

struct State final : MtpState {
    explicit State(const Draft& model);
    std::int64_t position() const noexcept override { return pos; }
    std::vector<Tensor> storage() const override;
    std::unique_ptr<Restore> prepare_restore(const MtpSessionState&) override;
    void release_graphs() noexcept override { graph.reset(); }

    const Draft& model;
    std::vector<DenseTensor> keys, values;
    DenseTensor compact, selector;
    DenseTensor successor_rows;
    std::shared_ptr<DraftGraph> graph;
    std::int64_t pos = 0;
    int row = 0;
};

class Draft final : public Mtp {
public:
    Draft(const CausalModel& target, Config config, Device device)
        : target(target), config(std::move(config)), device(device) {}

    std::unique_ptr<MtpState> make_state() const override;
    std::vector<std::pair<Device, std::size_t>> reset_growth(const MtpState&, std::int64_t) const override;
    void reset(MtpState&, std::int64_t) const override;
    MtpStep step(MtpState&, const Tensor&, const Tensor&, const Tensor&, const ForwardOptions&) const override;
    void trim_cache_to(MtpState&, std::int64_t) const override;
    bool teacher_forced_prompt_prime() const noexcept override { return false; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return true; }
    bool blockwise_drafting() const noexcept override { return true; }
    bool retains_partial_target_prefix() const noexcept override { return true; }
    bool supports_session_state(const MtpState&) const noexcept override { return true; }
    bool session_boundary(std::int64_t target, std::int64_t predictor) const noexcept override {
        return target > 0 && target == predictor;
    }
    int maximum_draft_depth() const noexcept override;
    std::span<const std::int64_t> target_layers() const noexcept override { return config.target_layers; }
    std::int64_t target_context_window() const noexcept override { return config.window; }
    MtpSessionState capture_session_state(const MtpState&, std::int64_t, const Tensor&) const override;
    void seed_target_context(MtpState&, const Tensor&, std::int64_t) const override;
    void append_target_context(MtpState&, const Tensor&, std::int64_t) const override;
    StepSequence<Tensor> draft_block(MtpState&, const Tensor&, int, bool graph) const override;
    Tensor draft_next(MtpState&, const Tensor&, const Tensor&) const override;
    StepSequence<MtpDraft> greedy_draft(const CausalModel&, MtpRow, int, const ForwardOptions&) const override;

    const CausalModel& target;
    const Config config;
    const Device device;
    std::string cache_key;
    bool compact_kv = false;
    Matrix context_projection, selector_projection;
    DenseTensor hidden_norm, final_norm, predecessor, successor, mask_embedding;
    std::vector<Layer> layers;
};

std::unique_ptr<Draft> load(const CausalModel&, const std::filesystem::path&,
    Device, std::int64_t hidden, std::int64_t vocabulary, std::int64_t target_layers);

} // namespace tyloi::models::dflash2
