#pragma once

#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxHyperConnectionConfig {
    int streams = 0;
    int sinkhorn_iterations = 0;
    float norm_epsilon = 1e-6f;
    float gate_epsilon = 0.0f;
    float pre_scale = 1.0f;
    float post_scale = 2.0f;
};

struct MlxHyperConnectionState {
    mlx::core::array branch;
    mlx::core::array post;
    std::optional<mlx::core::array> combination;
};

class MlxHyperConnection {
public:
    MlxHyperConnection(mlx::core::array weight, mlx::core::array bias,
        MlxHyperConnectionConfig config,
        std::optional<mlx::core::array> projection_scale = std::nullopt);

    MlxHyperConnectionState collapse(const mlx::core::array& residual) const;
    static mlx::core::array expand(const mlx::core::array& residual,
        const mlx::core::array& branch, const MlxHyperConnectionState& state);

private:
    mlx::core::array weight_;
    mlx::core::array bias_;
    MlxHyperConnectionConfig config_;
    std::optional<mlx::core::array> projection_scale_;
};

}  // namespace mfq::metal
