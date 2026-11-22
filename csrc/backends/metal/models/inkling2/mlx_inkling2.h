#pragma once

#include "mlx_text_backbone_runtime.h"
#include "mlx_transformer.h"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace mfq::metal {

class MfqContainer;

struct MlxInkling2VisitCache {
    std::optional<mlx::core::array> window;
    std::optional<MlxKvCache> global;
    std::array<std::optional<mlx::core::array>, 4> convolution;
};

struct MlxInkling2Cache {
    int position = 0;
    int batch = 0;
    std::vector<MlxInkling2VisitCache> visits;
};

struct MlxInkling2TextSessionState {
    struct Visit {
        std::optional<mlx::core::array> window;
        std::optional<MlxKvCacheSnapshot> global;
        std::array<mlx::core::array, 4> convolution;
    };
    std::vector<std::int64_t> tokens;
    std::vector<Visit> visits;
    int window = 0;
    std::optional<mlx::core::array> last_logits;
    std::size_t bytes = 0;
};

class MlxInkling2Model {
public:
    using Cache = MlxInkling2Cache;
    using TextSessionState = MlxInkling2TextSessionState;
    static MlxInkling2Model load(const MfqContainer& model, int context);

    mlx::core::array forward(const mlx::core::array& tokens, MlxInkling2Cache& cache,
        bool last_logit_only = false) const;
    mlx::core::array decode_batch(std::span<Cache* const> requests, const mlx::core::array& tokens) const;
    TextSessionState capture(const Cache& state, const std::vector<std::int64_t>& tokens,
        bool detached) const;
    Cache restore(const TextSessionState& state) const;
    int context() const noexcept;
    int vocabulary() const noexcept;

private:
    mlx::core::array project_output(const mlx::core::array& hidden, const mlx::core::array& blocks) const;
    struct Weights;
    explicit MlxInkling2Model(std::shared_ptr<const Weights> weights)
        : weights_(std::move(weights)) {}
    std::shared_ptr<const Weights> weights_;
};

using MlxInkling2Runtime = MlxTextBackboneRuntime<MlxInkling2Model>;

}  // namespace mfq::metal
