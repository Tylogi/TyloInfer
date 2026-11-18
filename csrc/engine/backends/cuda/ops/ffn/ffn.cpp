#include "engine/ops/ffn.h"

#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/graph/window.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/ops/ffn/options.h"
#include "engine/backends/cuda/ops/ffn/experts/launch.h"
#include "engine/backends/cuda/ops/ffn/experts/moe.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/tensor.h"
#include "engine/weights/moe/batch_internal.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::ops {
namespace {

using DenseTensor = ::tyloi::DenseTensor;

DenseTensor project_rows(const DenseTensor& input, const Tensor& weight,
                         std::int64_t rows) {
    if (rows <= 0 || input.size(0) <= rows)
        return ::tyloi::ops::project(input, weight).dense();
    std::vector<DenseTensor> parts;
    parts.reserve(static_cast<std::size_t>((input.size(0) + rows - 1) / rows));
    for (std::int64_t start = 0; start < input.size(0); start += rows) {
        const auto count = std::min(rows, input.size(0) - start);
        parts.push_back(::tyloi::ops::project(input.narrow(0, start, count), weight).dense());
    }
    return ::tyloi::cuda::cat(parts, 0);
}

template<class Work>
DenseTensor tiled_rows(const DenseTensor& source, Work work) {
    auto& executor = cuda::Executor::current();
    const auto option = cuda::options::continuous_ffn_tile();
    const auto rows = option ? *option : 8;
    TYLOINFER_RUNTIME_CHECK(rows == 8 || rows == 16,
                            "continuous FFN tile must be 8 or 16 rows");
    auto* window = cuda::DecodeWindow::recording();
    const int previous_profile_tile = window ? window->set_profile_tile(-1) : -1;
    auto& mode = executor.mode().ffn;
    struct TileScope {
        cuda::FfnMode& mode;
        int tile, tiles;
        cuda::DecodeWindow* window;
        int profile_tile;
        ~TileScope() {
            mode.tile = tile;
            mode.tiles = tiles;
            if (window) window->set_profile_tile(profile_tile);
        }
    } scope{mode, mode.tile, mode.tiles, window, previous_profile_tile};
    mode.tiles = static_cast<int>((source.size(0) + rows - 1) / rows);
    std::vector<DenseTensor> output;
    output.reserve(static_cast<std::size_t>(mode.tiles));
    for (std::int64_t start = 0; start < source.size(0); start += rows) {
        mode.tile = static_cast<int>(start / rows);
        if (window) window->set_profile_tile(mode.tile);
        auto tile_mode = cuda::Executor::current_ffn();
        tile_mode.tile = mode.tile;
        tile_mode.tiles = mode.tiles;
        auto tile_scope = executor.enter(tile_mode);
        const auto count = std::min(rows, source.size(0) - start);
        output.push_back(work(source.narrow(0, start, count).contiguous()));
    }
    return ::tyloi::cuda::cat(output, 0);
}

} // namespace

ScalarType Ffn::input_type(const Tensor& bank, ScalarType input) {
    (void)input;
    if (!bank) throw std::invalid_argument("CUDA FFN input type requires an expert bank");
    // CUDA expert kernels consume their selected input in F16.  The model
    // performs this conversion before constructing ExpertBatch.
    return kFloat16;
}

Tensor Ffn::project(const Tensor& input_tensor, const Tensor& weight,
                    std::int64_t projection_rows) {
    const auto& input = input_tensor.dense();
    if (!input.defined() || !weight || input.dim() < 2 || input.size(0) <= 0 || input.size(-1) <= 0)
        throw std::invalid_argument("CUDA FFN projection requires a nonempty input with a feature axis");
    std::optional<MfqCudaGuard> device_guard;
    device_guard.emplace(input.device());
    auto project = [&](const DenseTensor& rows) {
        return project_rows(rows, weight, projection_rows);
    };
    const auto& mode = cuda::Executor::current_ffn();
    DenseTensor result = mode.batch_serial && input.size(0) > 1
        ? tiled_rows(input, project) : project(input);
    return std::make_shared<DenseTensor>(std::move(result));
}

SelectedExperts hash_route(const Tensor& logits_tensor, const Tensor& table_tensor,
                           const Tensor& token_ids_tensor, double scale) {
    const auto& logits = logits_tensor.dense();
    const auto& table = table_tensor.dense();
    const auto& token_ids = token_ids_tensor.dense();
    if (!logits.defined() || !table.defined() || !token_ids.defined() || !logits.is_cuda() ||
        (token_ids.scalar_type() != kInt32 && token_ids.scalar_type() != kInt64) ||
        token_ids.numel() <= 0 || table.device() != logits.device() ||
        token_ids.device() != logits.device())
        throw std::invalid_argument("hash routing requires CUDA logits and token ids");
    std::optional<MfqCudaGuard> device_guard;
    device_guard.emplace(logits.device());
    auto ids = table.index_select(0, token_ids.reshape({-1}).to(logits.device(), kInt64))
                   .to(kInt32).contiguous();
    auto weights = moe_sqrtsoftplus_weights_cuda(logits.contiguous(), ids,
                                                  1e-20, scale);
    return {std::make_shared<DenseTensor>(std::move(ids)),
            std::make_shared<DenseTensor>(std::move(weights))};
}

Tensor Ffn::routed_shared(const Tensor& input_tensor, const Tensor& route_weights_tensor,
                          weights::ExpertBatch& batch,
                          std::span<const Tensor> shared_gate_up,
                          const Tensor& shared_down, const Tensor& gate_input_tensor,
                          const Tensor& gate_weight, const ExpertSpec& spec,
                          std::int64_t projection_rows) {
    if (!batch.impl_) throw std::invalid_argument("expert batch has been moved");
    if (spec.activation != MlpActivation::silu_gated || spec.swiglu_limit != 0)
        throw std::invalid_argument("CUDA routed shared fusion supports only SwiGLU");
    if (shared_gate_up.size() != 2 || !shared_gate_up[0] || !shared_gate_up[1] ||
        !shared_down || !gate_weight)
        throw std::invalid_argument("incomplete routed shared projections");
    const auto& input = input_tensor.dense();
    const auto& route_weights = route_weights_tensor.dense();
    const auto& gate_input = gate_input_tensor.dense();
    auto& b = *batch.impl_;
    const auto& ids = b.ids.dense();
    if (!input.is_cuda() || input.dim() != 2 || input.scalar_type() != kFloat16 ||
        !input.is_contiguous() || !gate_input.is_cuda() || gate_input.dim() != 2 ||
        gate_input.scalar_type() != kFloat16 || !input_tensor.same_view(gate_input_tensor) ||
        gate_input.size(0) != input.size(0) || gate_input.size(1) != input.size(1) ||
        route_weights.dim() != 2 || route_weights.scalar_type() != kFloat32 ||
        route_weights.sizes() != ids.sizes() || ids.dim() != 2 ||
        ids.scalar_type() != kInt32 || ids.size(0) != input.size(0) ||
        shared_down.dim() != 2 || shared_gate_up[0].dim() != 2 ||
        shared_gate_up[1].dim() != 2 || gate_weight.dim() != 2 ||
        gate_weight.size(0) != 1 || gate_weight.size(1) != gate_input.size(1))
        throw std::invalid_argument("routed shared geometry mismatch");
    // This fusion uses the same F16 source for shared projections and the scalar gate.
    if (b.gate_up_weights.empty() || b.gate_up_weights.size() > 2 || !b.down_weight)
        throw std::invalid_argument("routed shared fusion requires Gate/Up and Down banks");
    std::optional<MfqCudaGuard> device_guard;
    device_guard.emplace(input.device());

    auto& executor = cuda::Executor::current();
    auto run = [&](weights::ExpertBatch::Impl& selected, const DenseTensor& source,
                   const DenseTensor& weights, const DenseTensor& shared_source,
                   const DenseTensor& selected_ids) {
        // Enqueue shared projections first to preserve overlap with selected experts.
        auto shared = cuda::MoeSharedExpert(shared_gate_up[0], shared_gate_up[1],
                                            shared_down, gate_weight)
                          .forward(shared_source, projection_rows);
        selected.ready();
        auto project = executor.projector(cuda::Executor::current_ffn());
        DenseTensor routed;
        if (selected.tq.gate) {
            routed = executor.profiler().measure("tq.moe", [&] {
                return project.run(selected.tq, source, selected_ids, weights);
            });
        } else {
            const auto expert = [&](const Tensor& bank, DenseTensor value) {
                if (selected.down) return project.run(bank, std::move(value), selected.route);
                return project.run(bank, std::move(value), selected_ids);
            };
            const auto width = selected.down_weight.size(2);
            auto gu = expert(selected.gate_up_weights[0], source);
            if (selected.gate_up_weights.size() == 2)
                gu = cuda::cat({gu, expert(selected.gate_up_weights[1], source)}, -1);
            auto gate = gu.narrow(-1, 0, width), up = gu.narrow(-1, width, width);
            auto pairs = expert(selected.down_weight, (gate * cuda::sigmoid(gate)) * up);
            routed = cuda::zeros({source.size(0), source.size(1)}, source.options().dtype(kFloat32));
            for (std::int64_t route = 0; route < selected_ids.size(1); ++route)
                routed = routed + pairs.select(1, route).to(kFloat32) * weights.select(1, route).unsqueeze(-1);
            routed = routed.to(pairs.scalar_type());
        }
        selected.retire();
        return (routed + shared).contiguous();
    };

    const auto& mode = cuda::Executor::current_ffn();
    if (!mode.batch_serial || input.size(0) <= 1) {
        return std::make_shared<DenseTensor>(run(b, input, route_weights, gate_input, ids));
    }
    const auto option = cuda::options::continuous_ffn_tile();
    const auto rows = option ? *option : 8;
    TYLOINFER_RUNTIME_CHECK(rows == 8 || rows == 16,
                            "continuous FFN tile must be 8 or 16 rows");
    auto* window = cuda::DecodeWindow::recording();
    const int previous_profile_tile = window ? window->set_profile_tile(-1) : -1;
    auto& ffn_mode = executor.mode().ffn;
    struct TileScope {
        cuda::FfnMode& mode;
        int tile, tiles;
        cuda::DecodeWindow* window;
        int profile_tile;
        ~TileScope() {
            mode.tile = tile;
            mode.tiles = tiles;
            if (window) window->set_profile_tile(profile_tile);
        }
    } scope{ffn_mode, ffn_mode.tile, ffn_mode.tiles, window, previous_profile_tile};
    ffn_mode.tiles = static_cast<int>((input.size(0) + rows - 1) / rows);
    std::vector<DenseTensor> outputs;
    outputs.reserve(static_cast<std::size_t>(ffn_mode.tiles));
    for (std::int64_t start = 0; start < input.size(0); start += rows) {
        ffn_mode.tile = static_cast<int>(start / rows);
        if (window) window->set_profile_tile(ffn_mode.tile);
        auto tile_mode = cuda::Executor::current_ffn();
        tile_mode.tile = ffn_mode.tile;
        tile_mode.tiles = ffn_mode.tiles;
        auto tile_scope = executor.enter(tile_mode);
        const auto count = std::min(rows, input.size(0) - start);
        auto tile_ids = ids.narrow(0, start, count).contiguous();
        std::optional<weights::ExpertBatch> tile;
        if (b.down)
            tile.emplace(b.gate_up_weights, b.down_weight, Tensor{tile_ids},
                         b.route.row_offset + start, b.route.row_total, b.activation);
        outputs.push_back(run(tile ? *tile->impl_ : b, input.narrow(0, start, count).contiguous(),
                              route_weights.narrow(0, start, count).contiguous(),
                              gate_input.narrow(0, start, count).contiguous(),
                              tile_ids));
    }
    return std::make_shared<DenseTensor>(::tyloi::cuda::cat(outputs, 0));
}

} // namespace tyloi::ops
