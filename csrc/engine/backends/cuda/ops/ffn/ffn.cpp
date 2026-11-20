#include "engine/ops/ffn.h"

#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/graph/window.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "engine/backends/cuda/ops/ffn/options.h"
#include "engine/backends/cuda/ops/ffn/experts/launch.h"
#include "engine/backends/cuda/ops/ffn/experts/moe.h"
#include "engine/backends/cuda/ops/gguf/projection.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/tensor.h"
#include "engine/weights/moe/batch_internal.h"
#include "engine/weights/moe/ggml_bank.h"
#include "engine/weights/moe/routed_groups.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::ops {
namespace {

using DenseTensor = ::tyloi::DenseTensor;

using ExpertImages = std::array<weights::PreparedExperts, 3>;

struct StreamedExperts {
    std::array<const GgmlExpertBank*, 3> banks{};
    std::array<DenseTensor, 3> addresses, host_addresses;
    DenseTensor host_ids;
    std::shared_ptr<ExpertImages> images;
    std::vector<std::int64_t> selected;
    std::size_t count = 0;

    void load_gate_up() {
        *images = {};
        const auto ids = host_ids.to(kInt64);
        weights::group_routed_experts({ids.data_ptr<std::int64_t>(), std::size_t(ids.numel())},
            banks[0]->experts, selected);
        for (std::size_t i = 0; i < count - 1; ++i) load(i);
    }

    void load(std::size_t i) const {
        if (!host_addresses[i].defined()) return;
        const auto stream = cuda::current_stream(addresses[0].get_device());
        auto& projection = (*images)[i];
        projection = weights::prepare_indexed_projection(*banks[i], selected);
        auto* table = host_addresses[i].data_ptr<std::uint64_t>();
        std::fill_n(table, banks[i]->experts, 0);
        assert(projection.ggml.size() == selected.size());
        for (std::size_t j = 0; j < selected.size(); ++j)
            table[selected[j]] = reinterpret_cast<std::uintptr_t>(projection.ggml[j].packed.data_ptr());
        projection.wait();
        projection.retire();
        TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(addresses[i].data_ptr(), table,
            addresses[i].nbytes(), cudaMemcpyHostToDevice, stream.stream()));
    }
};

std::shared_ptr<StreamedExperts> stage_experts(cuda::DecodeWindow& window,
        std::span<const Tensor> gate_up, const Tensor& down, const DenseTensor& ids) {
    std::array<const GgmlExpertBank*, 3> banks{
        gate_up[0].representation<GgmlExpertBank>(),
        (gate_up.size() == 2 ? gate_up[1] : down).representation<GgmlExpertBank>(),
        down.representation<GgmlExpertBank>()};
    const auto selected_banks = std::span{banks}.first(gate_up.size() + 1);
    if (std::all_of(selected_banks.begin(), selected_banks.end(),
            [](const auto* bank) { return bank->resident_addresses.defined(); })) return {};

    static const char window_images{};
    auto images = window.workspace<ExpertImages>(&window_images, [] { return std::make_shared<ExpertImages>(); });
    auto stage = window.workspace<StreamedExperts>(banks[0], [&] {
        const cuda::ResidentAllocationScope fixed;
        auto result = std::make_shared<StreamedExperts>();
        result->banks = banks;
        result->count = selected_banks.size();
        result->images = images;
        const auto host_options = TensorOptions{}.device(kCPU).pinned_memory(true);
        result->host_ids = cuda::empty({1, ids.size(1)}, host_options.dtype(kInt32));
        for (std::size_t i = 0; i < result->count; ++i) {
            const auto& bank = *banks[i];
            result->addresses[i] = bank.resident_addresses;
            if (result->addresses[i].defined()) continue;
            result->addresses[i] = cuda::empty({bank.experts}, ids.options().dtype(kInt64));
            result->host_addresses[i] = cuda::empty({bank.experts}, host_options.dtype(kInt64));
        }
        return result;
    });
    const auto stream = cuda::current_stream(ids.get_device());
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(stage->host_ids.data_ptr(), ids.data_ptr(),
        ids.nbytes(), cudaMemcpyDeviceToHost, stream.stream()));
    // The preceding segment completes before the next layer can replace the images.
    window.host_barrier([stage] { stage->load_gate_up(); });
    if (capture == cudaStreamCaptureStatusNone) {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        stage->load_gate_up();
    }
    window.enroll({stage.get(), [] {}, [stage] { stage->load(stage->count - 1); }, [stage] {
        *stage->images = {};
        weights::observe_ggml_routes(std::span{stage->banks}.first(stage->count),
            stage->host_ids, 0, 1, false);
    }, [] {}, {}, [images] { *images = {}; }});
    return stage;
}

void stage_down(cuda::DecodeWindow& window, const StreamedExperts& stage) {
    if (!stage.host_addresses[stage.count - 1].defined()) return;
    const auto stream = cuda::current_stream(stage.addresses[0].get_device());
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture));
    // The host task submits the Down upload while the Gate/Up segment runs.
    window.host_barrier([] {});
    if (capture == cudaStreamCaptureStatusNone) stage.load(stage.count - 1);
}

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
    if (!(rows == 8 || rows == 16))
        throw std::runtime_error("continuous FFN tile must be 8 or 16 rows");
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
        std::shared_ptr<StreamedExperts> streamed;
        std::optional<weights::ExpertRows> indexed_rows;
        auto* window = cuda::DecodeWindow::recording();
        if (window && selected.ggml && !selected.route.prefill && selected_ids.size(0) == 1) {
            streamed = stage_experts(*window, selected.gate_up_weights, selected.down_weight, selected_ids);
            if (streamed) indexed_rows = weights::group_expert_rows(selected_ids, streamed->banks[0]->experts, true);
        }
        if (!streamed) selected.ready();
        auto project = executor.projector(cuda::Executor::current_ffn());
        DenseTensor routed;
        if (selected.tq.gate) {
            routed = executor.profiler().measure("tq.moe", [&] {
                return project.run(selected.tq, source, selected_ids, weights, 0,
                    selected.route.row_offset, selected.route.row_total,
                    cuda::Executor::current_ffn().batch_serial);
            });
        } else {
            const auto expert = [&](const Tensor& bank, DenseTensor value, std::size_t index) {
                if (streamed)
                    return cuda::gguf::project_indexed(streamed->banks[index]->image,
                        streamed->addresses[index], *indexed_rows, value);
                if (selected.ggml_rows) return project.run(bank, std::move(value), *selected.ggml_rows);
                if (selected.down) return project.run(bank, std::move(value), selected.route);
                return project.run(bank, std::move(value), selected_ids);
            };
            const auto width = selected.down_weight.size(2);
            auto gu = expert(selected.gate_up_weights[0], source, 0);
            if (selected.gate_up_weights.size() == 2)
                gu = cuda::cat({gu, expert(selected.gate_up_weights[1], source, 1)}, -1);
            auto gate = gu.narrow(-1, 0, width), up = gu.narrow(-1, width, width);
            auto activation = (gate * cuda::sigmoid(gate)) * up;
            if (streamed) stage_down(*window, *streamed);
            auto pairs = expert(selected.down_weight, std::move(activation), selected.gate_up_weights.size());
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
    if (!(rows == 8 || rows == 16))
        throw std::runtime_error("continuous FFN tile must be 8 or 16 rows");
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
        if (b.down || b.tq.gate || b.ggml)
            tile.emplace(b.gate_up_weights, b.down_weight, Tensor{tile_ids},
                         b.route.row_offset + start, b.route.row_total, b.activation, b.route.prefill);
        outputs.push_back(run(tile ? *tile->impl_ : b, input.narrow(0, start, count).contiguous(),
                              route_weights.narrow(0, start, count).contiguous(),
                              gate_input.narrow(0, start, count).contiguous(),
                              tile_ids));
    }
    return std::make_shared<DenseTensor>(::tyloi::cuda::cat(outputs, 0));
}

} // namespace tyloi::ops
