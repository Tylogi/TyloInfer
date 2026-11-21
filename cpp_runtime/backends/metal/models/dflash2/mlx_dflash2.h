#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>
#include <mlx/mlx.h>

namespace mfq::metal {

class MlxQwen4CausalLm;
class MlxQwen35CausalLm;
struct MlxSamplingParams;

struct MlxDFlash2Config {
    int hidden = 0, vocabulary = 0, layers = 0, intermediate = 0;
    int query_heads = 0, kv_heads = 0, head_dimension = 0;
    int block_size = 0, window = 0, convolution_group = 0, selector_rank = 0, candidates = 0;
    int mask_token = -1;
    float rope_theta = 0, epsilon = 0;
    std::vector<int> target_layers;
    static MlxDFlash2Config load(const std::filesystem::path& directory);
};

struct MlxDFlash2Context {
    struct Layer {
        std::optional<mlx::core::array> keys, values;
    };
    int position = 0;
    const void* owner = nullptr;
    std::vector<Layer> layers;
    std::size_t bytes() const noexcept;
};

struct MlxDFlash2Proposal {
    mlx::core::array tokens;
    mlx::core::array confidence;
};

struct MlxDFlash2TreeInput {
    int depth, rows;
    std::vector<std::int32_t> lookup;
    std::vector<std::array<float, 4>> lookup_rates;
};

struct MlxDFlash2Tree {
    std::vector<std::int64_t> tokens;
    std::vector<int> parents;
    std::vector<float> reach;
};

struct MlxDFlash2GenerationStats {
    std::size_t cycles = 0, proposed = 0, verified = 0, accepted = 0;
    std::array<std::uint64_t, 8> depth_cycles{};
    std::array<std::uint64_t, 7> position_verified{}, position_accepted{};
    std::size_t standard_tokens = 0;
    double prefill_ms = 0;
    double target_ms = 0, head_ms = 0, rollback_ms = 0;
    std::array<double, 8> measured_depth_ms{};
};

class MlxDFlash2 {
public:
    static MlxDFlash2 load(const std::filesystem::path& directory);
    const MlxDFlash2Config& config() const noexcept;
    std::size_t weight_bytes() const noexcept;
    void track_weights() const;
    void prepare(const std::function<mlx::core::array(const mlx::core::array&)>& output_head,
        bool tree_batch = false) const;
    void reset(MlxDFlash2Context& context) const;
    void append_context(MlxDFlash2Context& context, const mlx::core::array& target_hidden,
        int start_position) const;
    struct ContextCommit {
        MlxDFlash2Context* context;
        mlx::core::array features;
        int position;
    };
    void append_context_batch(std::span<const ContextCommit> commits) const;
    MlxDFlash2Proposal propose(const MlxDFlash2Context& context, int anchor,
        const mlx::core::array& embeddings,
        const std::function<mlx::core::array(const mlx::core::array&)>& output_head,
        const std::function<void(std::string_view, const mlx::core::array&)>& trace = {}) const;
    // Returns [lanes, max_depth] with only the requested prefix valid in each lane.
    // With no explicit depths, returns a full proposal for every lane.
    MlxDFlash2Proposal propose_batch(std::span<const MlxDFlash2Context* const> contexts,
        std::span<const std::int32_t> anchors, const mlx::core::array& embeddings,
        const std::function<mlx::core::array(const mlx::core::array&)>& output_head,
        std::span<const int> depths = {},
        const std::function<void(std::string_view, const mlx::core::array&)>& trace = {}) const;
    std::vector<MlxDFlash2Tree> propose_tree_batch(std::span<const MlxDFlash2Context* const> contexts,
        std::span<const std::int32_t> anchors, const mlx::core::array& embeddings,
        const std::function<mlx::core::array(const mlx::core::array&)>& output_head,
        std::span<const MlxDFlash2TreeInput> inputs) const;
    std::int32_t generate(MlxQwen4CausalLm& target,
        const std::vector<std::int64_t>& prompt, const MlxSamplingParams& sampling, std::int32_t max_tokens,
        const std::function<bool(std::int64_t)>& callback = {},
        const std::function<void(std::size_t, double)>& prefill_callback = {},
        int prefill_chunk_size = 4096, std::optional<int> stop_token = std::nullopt,
        MlxDFlash2GenerationStats* stats = nullptr, int maximum_drafts = 7) const;
    std::int32_t generate_greedy(MlxQwen4CausalLm& target,
        const std::vector<std::int64_t>& prompt, std::int32_t max_tokens,
        const std::function<bool(std::int64_t)>& callback = {},
        const std::function<void(std::size_t, double)>& prefill_callback = {},
        int prefill_chunk_size = 4096, std::optional<int> stop_token = std::nullopt,
        MlxDFlash2GenerationStats* stats = nullptr, int maximum_drafts = 7) const;
    std::int32_t generate(MlxQwen35CausalLm& target,
        const std::vector<std::int64_t>& prompt, const MlxSamplingParams& sampling, std::int32_t max_tokens,
        const std::function<bool(std::int64_t)>& callback = {},
        const std::function<void(std::size_t, double)>& prefill_callback = {},
        int prefill_chunk_size = 4096, std::optional<int> stop_token = std::nullopt,
        MlxDFlash2GenerationStats* stats = nullptr, int maximum_drafts = 7) const;

private:
    struct TreeProposals;
    TreeProposals tree_proposals(std::span<const MlxDFlash2Context* const> contexts,
        std::span<const std::int32_t> anchors, const mlx::core::array& embeddings,
        const std::function<mlx::core::array(const mlx::core::array&)>& output_head,
        std::span<const MlxDFlash2TreeInput> inputs) const;
    struct ProposalLattice;
    ProposalLattice proposal_lattice(std::span<const MlxDFlash2Context* const> contexts,
        std::span<const std::int32_t> anchors, const mlx::core::array& embeddings,
        const std::function<mlx::core::array(const mlx::core::array&)>& output_head,
        std::span<const int> depths,
        const std::function<void(std::string_view, const mlx::core::array&)>& trace) const;
    struct Impl;
    explicit MlxDFlash2(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<Impl> impl_;
};

}
