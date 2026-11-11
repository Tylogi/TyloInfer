#include "engine/ops.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#include "engine/weights/moe/batch_internal.h"
#include "engine/weights/moe/experts.h"
#include "engine/backends/cuda/executor/tq_moe.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/ops/ffn/moe.h"
#include "engine/backends/cuda/ops/ffn/mlp.h"
#include "engine/backends/cuda/ops/projection/parallel.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "diagnostics/route_stats.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/norm/layer.h"
#include "engine/backends/cuda/ops/vision/attention.h"
#include "engine/backends/cuda/ops/ffn/dense.h"
#include "engine/backends/cuda/ops/ffn/routed.h"
#include "engine/backends/cuda/ops/ffn/residual.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/tensor.h"
#include "engine/backends/cuda/native/tensor/tensor.h"
#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/attention/kv.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/ops/rope/tail.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#endif

namespace tyloi::ops {
#ifdef TYLOINFER_NATIVE_CUDA
namespace {
const DenseTensor& dense(const Tensor& value) {
    const auto* result = value.representation<DenseTensor>();
    if (!result || result->is_encoded()) throw std::invalid_argument("operation requires a dense tensor");
    return *result;
}
DenseTensor& dense(Tensor& value) {
    auto* result = value.representation<DenseTensor>();
    if (!result || result->is_encoded()) throw std::invalid_argument("operation requires a dense tensor");
    return *result;
}
Tensor owner(DenseTensor value) {
    return std::make_shared<DenseTensor>(std::move(value));
}
std::vector<Tensor> owners(std::vector<DenseTensor> values) {
    std::vector<Tensor> result;
    result.reserve(values.size());
    for (auto& value : values) result.push_back(owner(std::move(value)));
    return result;
}
void device(Device value) {
    if (!value.is_cpu() && !value.is_cuda())
        throw std::invalid_argument("unsupported tensor device");
    if (value.is_cuda()) (void)cuda::Executor::current();
}
void input(const Tensor& value_tensor) {
    const auto& value = dense(value_tensor);
    if (!value.defined()) throw std::invalid_argument("undefined tensor");
    value.require_dense();
    device(value.device());
}
void compatible(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    input(left);
    input(right);
    if (left.device() != right.device())
        throw std::invalid_argument("tensor devices do not match");
}
void residual_inputs(const Tensor& residual, const Tensor& value) {
    compatible(residual, value);
    if (residual.dim() != 3 || residual.sizes() != value.sizes())
        throw std::invalid_argument("residual inputs must have matching [batch,tokens,width] shapes");
}
void require_cuda(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    if (!value.device().is_cuda()) throw std::invalid_argument("operation requires a CUDA tensor");
}
bool encoded(std::initializer_list<const Tensor> values) {
    for (const auto& value : values)
        if (value.defined() && value.is_encoded()) return true;
    return false;
}
}

AttentionResult attention(const AttentionInput& request, const state::KvView& kv,
                          const AttentionConfig& config) {
    compatible(request.query, request.key);
    compatible(request.query, request.value);
    input(request.positions);
    input(request.write_positions);
    if (request.sequence_lengths.defined()) input(request.sequence_lengths);
    if (request.attention_mask.defined()) input(request.attention_mask);
    if (request.query.dim() != 4 || request.key.dim() != 4 || request.value.dim() != 4 ||
        dense(request.key).sizes() != dense(request.value).sizes() ||
        request.query.size(0) != request.key.size(0) ||
        request.query.size(2) != request.key.size(2) ||
        request.query.size(3) != request.key.size(3) || request.key.size(1) <= 0 ||
        request.query.size(1) <= 0 || request.query.size(3) <= 0)
        throw std::invalid_argument("attention query, key and value geometry is incompatible");
    for (const auto& tensor : {kv.visible_key, kv.visible_value, kv.physical_key,
                                kv.physical_value, kv.key_chunk_ptrs,
                                kv.value_chunk_ptrs, kv.page_table})
        if (tensor.defined()) compatible(request.query, tensor);
    require_cuda(request.query);
    return cuda::ops::attention(request, kv, config);
}

std::int64_t attention_parts(std::int64_t planned_len, Device target) {
    device(target);
    if (!target.is_cuda()) throw std::invalid_argument("attention parts require a CUDA device");
    return cuda::ops::attention_parts(planned_len);
}

Tensor attention(const Tensor& query_tensor, const Tensor& key_tensor, const Tensor& value_tensor,
                 const Tensor& mask, double scale) {
    const auto& query = dense(query_tensor);
    const auto& key = dense(key_tensor);
    const auto& value = dense(value_tensor);
    compatible(query, key);
    compatible(query, value);
    if (mask.defined()) compatible(query, mask);
    if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4 ||
        query.size(3) != key.size(3) || key.size(2) != value.size(2))
        throw std::invalid_argument("attention query, key and value geometry is incompatible");
    const auto broadcasts = [](std::int64_t a, std::int64_t b) {
        return a == b || a == 1 || b == 1;
    };
    for (int axis = 0; axis < 2; ++axis) {
        const auto q = query.size(axis), k = key.size(axis), v = value.size(axis);
        if (!broadcasts(q, k) || !broadcasts(q > k ? q : k, v))
            throw std::invalid_argument("attention query, key and value geometry is incompatible");
    }
    require_cuda(query);
    const std::optional<DenseTensor> dense_mask = mask ? std::optional<DenseTensor>(dense(mask)) : std::nullopt;
    return owner(cuda::ops::attention(query, key, value, dense_mask, scale));
}

Tensor sparse_attention(const Tensor& query_tensor, const Tensor& cache_tensor,
                        const Tensor& indices_tensor, const Tensor& mask_tensor,
                        const Tensor& sinks_tensor, double scale_value) {
    const auto& query = dense(query_tensor);
    const auto& cache = dense(cache_tensor);
    const auto& indices = dense(indices_tensor);
    const auto& mask = dense(mask_tensor);
    const auto& sinks = dense(sinks_tensor);
    compatible(query, cache);
    compatible(query, indices);
    compatible(query, mask);
    compatible(query, sinks);
    require_cuda(query);
    if (query.dim() != 4 || cache.dim() != 3 || indices.dim() != 3 ||
        mask.sizes() != indices.sizes() || sinks.dim() != 1 ||
        query.size(0) != cache.size(0) || query.size(0) != indices.size(0) ||
        query.size(2) != indices.size(1) || query.size(3) != cache.size(2))
        throw std::invalid_argument("sparse attention geometry is incompatible");
    return owner(cuda::ops::sparse_attention(
        query, cache, indices, mask, sinks, scale_value));
}

Tensor mla_attention(const Tensor& query, const Tensor& cache, const Tensor& lengths,
                     const Tensor& indices, std::int64_t length, std::int64_t planned, double scale) {
    compatible(query, cache);
    require_cuda(query);
    if (query.dim() != 4 || cache.dim() != 4 || query.size(0) != cache.size(0) ||
        cache.size(1) != 1 || query.size(3) != 576 || cache.size(3) != 576 ||
        length <= 0 || length > cache.size(2) || planned < 0 ||
        (planned > 0 && (planned < length || planned > cache.size(2))))
        throw std::invalid_argument("MLA attention geometry is incompatible");
    if (lengths) {
        compatible(query, lengths);
        if (lengths.dim() != 1 || lengths.size(0) != query.size(0))
            throw std::invalid_argument("MLA lengths must have shape [batch]");
    }
    if (indices) {
        compatible(query, indices);
        if (indices.dim() != 3 || indices.size(0) != query.size(0) || indices.size(1) != query.size(2))
            throw std::invalid_argument("MLA indices must have shape [batch,tokens,selected]");
    }
    return owner(cuda::ops::mla_attention(dense(query), dense(cache),
        lengths ? dense(lengths) : DenseTensor{}, indices ? dense(indices) : DenseTensor{},
        length, planned, scale));
}

TailRopeTable tail_rope_table(std::int64_t max_positions, double base, std::int64_t compress_ratio,
        Device target, double compress_base, std::int64_t original_positions,
        double factor, double beta_fast, double beta_slow) {
    device(target);
    if (!target.is_cuda() || target.index < 0)
        throw std::invalid_argument("tail RoPE table requires a CUDA device");
    return cuda::ops::tail_rope_table(max_positions, base, compress_ratio, target,
        compress_base, original_positions, factor, beta_fast, beta_slow);
}

Tensor tail_rope(const Tensor& value, const Tensor& positions,
        const TailRopeTable& table, bool inverse) {
    const auto& x = dense(value);
    const auto& pos = dense(positions);
    device(x.device());
    if (!x.device().is_cuda() || x.device().index < 0)
        throw std::invalid_argument("tail RoPE requires a CUDA device");
    if (!table.cos || !(inverse ? table.negative_sin : table.sin))
        throw std::invalid_argument("tail RoPE table requires cosine and sine");
    return owner(cuda::ops::tail_rope(x, pos, table, inverse));
}

RopeTable rope_table(std::int64_t positions, std::int64_t dimension, double base, Device target,
                     std::int64_t frequency_dim, std::int64_t active_pairs) {
    device(target);
    if (!target.is_cuda()) throw std::invalid_argument("RoPE table requires a CUDA device");
    if (positions <= 0 || dimension <= 0 || dimension % 2 != 0 ||
        !std::isfinite(base) || base <= 0.0 || active_pairs > dimension / 2)
        throw std::invalid_argument("invalid RoPE table parameters");
    return cuda::ops::rope_table(positions, dimension, base, target, frequency_dim, active_pairs);
}

Tensor rope(const Tensor& value_tensor, const Tensor& positions_tensor, const RopeTable& table, bool bf16, bool grid) {
    const auto& value = dense(value_tensor);
    const auto& positions = dense(positions_tensor);
    if (!table.cos || !table.sin) throw std::invalid_argument("RoPE table requires cosine and sine");
    compatible(value, table.cos);
    compatible(value, table.sin);
    for (const auto& item : {table.sections, table.empty_sections,
                             table.interleaved_order, table.interleaved_inverse})
        if (item.defined()) (void)dense(item);
    input(positions);
    if (value.dim() == 0 || table.rotary_dim <= 0 || table.rotary_dim % 2 != 0 ||
        table.rotary_dim > value.size(-1))
        throw std::invalid_argument("invalid RoPE rotary dimension");
    require_cuda(value);
    return owner(cuda::ops::rope(value, positions, table, bf16, grid));
}
Tensor norm_rope_kv(const AttentionInput& request, const state::KvView& kv,
    const RopeTable& table, const Tensor& q_norm, const Tensor& k_norm, double eps, double offset) {
    if (!request.query.is_cuda()) return {};
    if (encoded({request.query, request.key, request.value, request.positions, request.write_positions,
                 table.cos, table.sin, table.sections, table.empty_sections, table.interleaved_order,
                 table.interleaved_inverse, q_norm, k_norm, kv.visible_key, kv.visible_value,
                 kv.physical_key, kv.physical_value, kv.key_chunk_ptrs, kv.value_chunk_ptrs, kv.page_table,
                 request.sequence_lengths, request.attention_mask})) return {};
    auto result = cuda::ops::norm_rope_kv(request, kv, table, dense(q_norm), dense(k_norm), eps, offset);
    return result ? owner(std::move(*result)) : Tensor{};
}
Tensor rope_kv(const AttentionInput& request, const state::KvView& kv,
                               const RopeTable& table) {
    if (!request.query.is_cuda()) return {};
    if (encoded({request.query, request.key, request.value, request.positions, request.write_positions,
                 table.cos, table.sin, table.sections, table.empty_sections, table.interleaved_order,
                 table.interleaved_inverse, kv.visible_key, kv.visible_value, kv.physical_key,
                 kv.physical_value, kv.key_chunk_ptrs, kv.value_chunk_ptrs, kv.page_table,
                 request.sequence_lengths, request.attention_mask})) return {};
    auto result = cuda::ops::rope_kv(request, kv, table);
    return result ? owner(std::move(*result)) : Tensor{};
}

std::pair<Tensor, Tensor> append_kv(state::KvView kv, const Tensor& key_tensor, const Tensor& value_tensor,
                                    const Tensor& positions_tensor, std::int64_t start,
                                    bool contiguous_prefill_prefix) {
    const auto& key = dense(key_tensor);
    const auto& value = dense(value_tensor);
    const auto& positions = dense(positions_tensor);
    compatible(key, value);
    compatible(key, positions);
    if (key.dim() != 4 || value.dim() != 4 || key.sizes() != value.sizes())
        throw std::invalid_argument("KV write key and value geometry is incompatible");
    for (const auto& tensor : {kv.physical_key, kv.physical_value, kv.key_chunk_ptrs,
                                kv.value_chunk_ptrs, kv.page_table})
        if (tensor.defined()) compatible(key, tensor);
    // A zero batch skips the unpaged, unquantized index writes.
    if (kv.paged() || kv.quantization.enabled() || positions.dim() != 2 || positions.size(0) != 0)
        require_cuda(key);
    auto result = cuda::ops::append_kv(std::move(kv), key, value, positions, start, contiguous_prefill_prefix);
    return {owner(std::move(result.first)), owner(std::move(result.second))};
}

Tensor add(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::add(left, right));
}
Tensor multiply(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::multiply(left, right));
}
Tensor matmul(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::matmul(left, right));
}
Tensor transpose(const Tensor& value_tensor, std::int64_t first, std::int64_t second) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::transpose(value, first, second));
}
Tensor add_residual(const Tensor& residual_tensor, const Tensor& value_tensor, ResidualMode mode) {
    const auto& residual = ::tyloi::ops::dense(residual_tensor);
    const auto& value = ::tyloi::ops::dense(value_tensor);
    compatible(residual, value);
    require_cuda(residual);
    return owner(cuda::ops::add_residual(residual, value, mode == ResidualMode::configured));
}

std::array<Tensor, 2> residual_norm(const Tensor& residual, const Tensor& value,
        const Tensor& weight, double eps, double offset, bool bf16) {
    residual_inputs(residual, value);
    compatible(residual, weight);
    if (weight.numel() != residual.size(2))
        throw std::invalid_argument("residual norm weight has the wrong width");
    auto result = cuda::ops::residual_norm(dense(residual), dense(value), weight, eps, offset, bf16);
    return {owner(std::move(result[0])), owner(std::move(result[1]))};
}
std::array<Tensor, 2> recurrent_norm(const Tensor& residual, const Tensor& value,
        const Tensor& weight, double eps, double offset) {
    residual_inputs(residual, value);
    compatible(residual, weight);
    if (weight.numel() != residual.size(2))
        throw std::invalid_argument("residual norm weight has the wrong width");
    auto result = cuda::ops::recurrent_norm(dense(residual), dense(value), weight, eps, offset);
    return {owner(std::move(result[0])), owner(std::move(result[1]))};
}
Tensor residual_add(const Tensor& residual, const Tensor& value) {
    residual_inputs(residual, value);
    return owner(cuda::ops::residual_add(dense(residual), dense(value)));
}
Tensor recurrent_add(const Tensor& residual, const Tensor& value) {
    residual_inputs(residual, value);
    return owner(cuda::ops::recurrent_add(dense(residual), dense(value)));
}
std::optional<ResidualNorms> residual_norms(const Tensor& residual_tensor, const Tensor& attention_tensor, const Tensor& attention_weight_tensor,
                                            const Tensor& dense_weight_tensor, const Tensor& router_weight_tensor,
                                            const Tensor& expert_weight_tensor, double eps) {
    if (!residual_tensor.is_cuda() ||
        encoded({residual_tensor, attention_tensor, attention_weight_tensor,
                 dense_weight_tensor, router_weight_tensor, expert_weight_tensor})) return std::nullopt;
    const auto& residual = ::tyloi::ops::dense(residual_tensor);
    const auto& attention = ::tyloi::ops::dense(attention_tensor);
    const auto& attention_weight = ::tyloi::ops::dense(attention_weight_tensor);
    const auto& dense_weight = ::tyloi::ops::dense(dense_weight_tensor);
    const auto& router_weight = ::tyloi::ops::dense(router_weight_tensor);
    const auto& expert_weight = ::tyloi::ops::dense(expert_weight_tensor);
    return cuda::ops::residual_norms(residual, attention, attention_weight,
                                     dense_weight, router_weight, expert_weight, eps);
}
Tensor merge_norms(const Tensor& dense_tensor, const Tensor& experts_tensor, const Tensor& residual_tensor, const Tensor& dense_weight_tensor,
                   const Tensor& expert_weight_tensor, const Tensor& output_weight_tensor,
                   const Tensor& scale_tensor, double eps) {
    const auto& dense = ::tyloi::ops::dense(dense_tensor);
    const auto& experts = ::tyloi::ops::dense(experts_tensor);
    const auto& residual = ::tyloi::ops::dense(residual_tensor);
    const auto& dense_weight = ::tyloi::ops::dense(dense_weight_tensor);
    const auto& expert_weight = ::tyloi::ops::dense(expert_weight_tensor);
    const auto& output_weight = ::tyloi::ops::dense(output_weight_tensor);
    const auto& scale = ::tyloi::ops::dense(scale_tensor);
    for (const DenseTensor* value : std::initializer_list<const DenseTensor*>{
             &experts, &residual, &dense_weight, &expert_weight, &output_weight, &scale})
        compatible(dense, *value);
    require_cuda(dense);
    return owner(cuda::ops::merge_norms(dense, experts, residual,
                                  dense_weight, expert_weight, output_weight, scale, eps));
}
struct ProfileRange::Impl {
    std::optional<cuda::DeviceGuard> guard;
    std::optional<CudaProfiler::Range> range;
    Impl(CudaProfiler& profiler, const char* name, std::optional<Device> device) {
        if (device && device->is_cuda()) guard.emplace(device->index);
        range.emplace(profiler, name);
    }
};
ProfileRange::ProfileRange(const char* name, std::optional<Device> device) {
    auto& profiler = cuda::Executor::current().profiler();
    if (profiler.enabled && profiler.selected(name))
        impl_ = std::make_unique<Impl>(profiler, name, device);
}
ProfileRange::~ProfileRange() = default;
void ProfileRange::finish() { if (impl_) impl_->range->finish(); }
struct ProfileCapture::Impl {
    CudaProfiler& profiler;
    bool previous, enabled, selected;
    std::string report_name;
    Impl(CudaProfiler& profiler, bool selected, std::string report_name)
        : profiler(profiler), previous(profiler.enabled), enabled(selected || profiler.enabled),
          selected(selected), report_name(std::move(report_name)) {
        if (selected) {
            profiler.reset();
            profiler.enabled = true;
        }
    }
    ~Impl() {
        if (selected && !report_name.empty() && std::uncaught_exceptions() == 0) {
            try {
                profiler.report(report_name);
                profiler.reset();
            } catch (const std::exception& error) {
                std::cerr << "prefill_profile_error " << error.what() << '\n';
            }
        }
        profiler.enabled = previous;
    }
};
ProfileCapture::ProfileCapture(bool selected)
    : impl_(selected ? std::make_unique<Impl>(cuda::Executor::current().profiler(), true, std::string{}) : nullptr) {}
ProfileCapture::ProfileCapture(bool selected, const std::string& report_name)
    : impl_(std::make_unique<Impl>(cuda::Executor::current().profiler(), selected, report_name)) {}
ProfileCapture::~ProfileCapture() = default;
void ProfileCapture::report(const std::string& name) {
    if (impl_ && impl_->selected) impl_->profiler.report(name);
}
void ProfileCapture::resume() {
    if (impl_) impl_->profiler.enabled = impl_->enabled;
}
SelectedExperts topk(const Tensor& logits_tensor, std::int64_t count, const MoeRouting& routing,
                     const Tensor& bias) {
    const auto& logits = ::tyloi::ops::dense(logits_tensor);
    input(logits);
    require_cuda(logits);
    const auto* selection_bias = bias.defined() ? &dense(bias) : nullptr;
    if (selection_bias) compatible(logits, bias);
    return cuda::ops::topk(logits, count, routing, selection_bias);
}
void expert_scale(Tensor& weights_tensor, const Tensor& ids_tensor, const Tensor& scales_tensor) {
    auto& weights = dense(weights_tensor);
    const auto& ids = dense(ids_tensor);
    const auto& scales = dense(scales_tensor);
    compatible(weights, ids);
    compatible(weights, scales);
    require_cuda(weights);
    cuda::ops::expert_scale(weights, ids, scales);
}
Tensor weighted_reduce(const Tensor& pairs_tensor, const Tensor& weights_tensor) {
    const auto& pairs = ::tyloi::ops::dense(pairs_tensor);
    const auto& weights = ::tyloi::ops::dense(weights_tensor);
    compatible(pairs, weights);
    require_cuda(pairs);
    return owner(cuda::ops::weighted_reduce(pairs, weights));
}
RoutedExperts routed(const Tensor& input_tensor, const Tensor& input_f32_tensor,
                     const Tensor& weights_tensor, weights::ExpertBatch& batch, const ExpertSpec& spec) {
    if (!batch.impl_) throw std::invalid_argument("expert batch has been moved");
    auto& b = *batch.impl_;
    const auto& x = dense(input_tensor);
    const auto& xf32 = dense(input_f32_tensor);
    const auto& weights = dense(weights_tensor);
    const auto& ids = dense(b.ids);
    compatible(input_tensor, input_f32_tensor);
    compatible(input_tensor, weights_tensor);
    compatible(input_tensor, b.ids);
    compatible(input_tensor, spec.shared_gate);
    require_cuda(input_tensor);
    if (x.dim() != 2 || x.scalar_type() != kFloat16 || !x.is_contiguous() ||
        xf32.sizes() != x.sizes() || xf32.scalar_type() != kFloat32 || !xf32.is_contiguous() ||
        ids.dim() != 2 || ids.scalar_type() != kInt32 || ids.size(0) != x.size(0) ||
        ids.size(1) <= 0 || weights.sizes() != ids.sizes() || weights.scalar_type() != kFloat32 ||
        !ids.is_contiguous() || !weights.is_contiguous() || x.size(0) <= 0 || x.size(1) <= 0 ||
        spec.shared_gate.dim() != 2 || spec.shared_gate.size(0) != 1 ||
        spec.shared_gate.size(1) != x.size(1) || dense(spec.shared_gate).scalar_type() != kFloat32)
        throw std::invalid_argument("routed expert input geometry mismatch");
    const auto& gate_bank = b.gate_up_weights;
    if (gate_bank.empty() || !b.down_weight.defined())
        throw std::invalid_argument("routed expert banks are missing");
    const auto& gate = gate_bank[0];
    const auto& up = b.split ? gate_bank[1] : gate_bank[0];
    const auto& down = b.down_weight;
    const bool dense_routed = bool(b.dense_gate) || bool(b.dense_up) || bool(b.dense_down);
    const bool tq_routed = bool(b.tq.gate || b.tq.up || b.tq.down);
    if ((spec.latent_in.defined() && (spec.latent_in.dim() != 2 || spec.latent_in.size(1) != x.size(1))) ||
        (spec.latent_out.defined() && spec.latent_out.dim() != 2))
        throw std::invalid_argument("routed expert latent geometry mismatch");
    const auto input_width = spec.latent_in.defined() && (dense_routed || tq_routed || b.sharded)
        ? spec.latent_in.size(0) : x.size(1);
    if (gate.dim() != 3 || up.dim() != 3 || down.dim() != 3 ||
        gate.size(0) <= 0 || gate.size(0) != up.size(0) || gate.size(0) != down.size(0) ||
        gate.size(2) != input_width || up.size(2) != input_width ||
        gate.size(1) <= 0 || up.size(1) <= 0 || down.size(1) <= 0 ||
        (b.split ? gate.size(1) != up.size(1) || down.size(2) != up.size(1)
                 : gate.size(1) % 2 != 0 || down.size(2) != gate.size(1) / 2) ||
        (spec.latent_out.defined() && (dense_routed || tq_routed || b.sharded) &&
         (spec.latent_out.size(1) != down.size(1) || spec.latent_out.size(0) != x.size(1))) ||
        ((!spec.latent_out.defined() || (!dense_routed && !tq_routed && !b.sharded)) && down.size(1) != x.size(1)))
        throw std::invalid_argument("routed expert bank geometry mismatch");
    auto& executor = cuda::Executor::current();
    const auto& mode = cuda::Executor::current_ffn();
    auto project = executor.projector(mode);
    auto& profiler = executor.profiler();
    auto& stats = executor.routes();
    const auto& options = executor.options().ffn;
    const auto& projection = executor.options().projection;
    DenseTensor logits;
    auto shared_gate = [&] {
        if (!logits.defined())
            logits = profiler.measure("moe.shared_gate", [&] {
                return cuda::matmul(xf32, dense(spec.shared_gate).transpose(0, 1)).contiguous();
            });
    };
    if (!b.split) shared_gate();
    b.ready();
    const bool latent_routed = !dense_routed && spec.latent_in.defined();
    if (latent_routed && !spec.latent_out.defined())
        throw std::runtime_error("MoE latent output projection is missing");
    auto routed_input = x;
    if (latent_routed)
        routed_input = profiler.measure("moe.latent_down", [&] { return project.run(spec.latent_in, x); });
    auto finish_latent = [&](DenseTensor value) {
        if (latent_routed)
            return profiler.measure("moe.latent_up", [&] { return project.run(spec.latent_out, value); });
        return value;
    };
    DenseTensor result;
    bool reduced = false;
    if (b.tq.gate || b.tq.up || b.tq.down) {
        if (!b.tq.up || !b.tq.down || !b.tq.gate)
            throw std::runtime_error("TQ routed FFN requires matching Gate, Up and Down projections");
        result = profiler.measure("moe.tq", [&] {
            return project.run(b.tq, routed_input, ids, weights, static_cast<float>(spec.swiglu_limit));
        });
        result = finish_latent(std::move(result));
        reduced = true;
    } else if (dense_routed) {
        if (!b.dense_gate || !b.dense_up || !b.dense_down)
            throw std::runtime_error("gated dense routed MoE requires Gate, Up and Down banks");
        auto expert_input = x;
        if (spec.latent_in.defined())
            expert_input = profiler.measure("moe.latent_down", [&] { return project.run(spec.latent_in, x); });
        auto up_value = profiler.measure("moe.dense_up", [&] { return ::dense_routed_projection(*b.dense_up, expert_input, ids); });
        auto gate_value = profiler.measure("moe.dense_gate", [&] { return ::dense_routed_projection(*b.dense_gate, expert_input, ids); });
        auto hidden = profiler.measure("moe.dense_activation", [&] {
            return cuda::expert_activation(MlpActivation::silu_gated, std::move(gate_value), std::move(up_value), spec.swiglu_limit);
        });
        auto pairs = profiler.measure("moe.dense_down", [&] { return ::dense_routed_projection(*b.dense_down, hidden, ids); });
        result = profiler.measure("moe.dense_reduce", [&] { return moe_weighted_reduce_cuda(pairs, weights); });
        if (spec.latent_out.defined())
            result = profiler.measure("moe.latent_up", [&] { return project.run(spec.latent_out, result); });
        stats.record(spec.layer, ids, weights, pairs, static_cast<int>(b.dense_up->size(0)));
        reduced = true;
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    else if (b.sharded) {
        auto& collectives = executor.collectives();
        const size_t count = b.gate_up->shards.size();
        std::vector<DenseTensor> partials(count), down_partials;
        const bool energy = stats.energy();
        if (energy) down_partials.resize(count);
        result = profiler.measure("moe.sharded_experts", [&] {
            for (size_t pos = 0; pos < count; ++pos) {
                const auto index = ::parallel_index(projection.peer_first, pos, count);
                const auto& gs = b.gate_up->shards[index];
                const auto& ds = b.down->shards[index];
                MfqCudaGuard guard(gs.device);
                auto local_x = collectives.copy(routed_input, gs.device);
                auto local_weights = collectives.copy(weights, gs.device);
                const auto& route = ::moe_route_to_device(collectives, b.route, gs.device);
                DenseTensor pairs;
                if (spec.swiglu_limit <= 0 && !mode.materialized && cuda::supports_projection_glu_epilogue(*gs.weight)) {
                    auto hidden = project.glu(*gs.weight, local_x, route, false);
                    pairs = project.run(*ds.weight, hidden, route);
                } else {
                    auto gu = project.run(*gs.weight, local_x, route);
                    const bool fusion = !mode.materialized && options.swiglu_fusion &&
                        (gu.size(0) == 1 || (gu.size(0) <= 4 && options.small_heterogeneous));
                    if (fusion && spec.swiglu_limit <= 0)
                        pairs = project.gated(*ds.weight, gu, route, false);
                    else if (fusion && cuda::supports_clamped_swiglu(*ds.weight))
                        pairs = project.clamped(*ds.weight, gu, route, spec.swiglu_limit);
                    else {
                        DenseTensor hidden;
                        if (spec.swiglu_limit <= 0) hidden = moe_swiglu_split_cuda(gu);
                        else {
                            const auto width = gu.size(-1) / 2;
                            auto g = cuda::clamp_max(gu.narrow(-1, 0, width).to(kFloat32), spec.swiglu_limit);
                            auto u = cuda::clamp(gu.narrow(-1, width, width).to(kFloat32), -spec.swiglu_limit, spec.swiglu_limit);
                            hidden = (cuda::silu(g) * u).to(kFloat16).contiguous();
                        }
                        pairs = project.run(*ds.weight, hidden, route);
                    }
                }
                if (energy) down_partials[index] = pairs;
                partials[index] = moe_weighted_reduce_cuda(pairs, local_weights);
            }
            if (stats.enabled()) {
                DenseTensor complete;
                if (energy) complete = collectives.sum(std::move(down_partials), project.primary_device(), projection.reduce_to_primary, projection.fp16_reduce);
                stats.record(spec.layer, b.route.ids, weights, complete, b.gate_up->n_experts);
            }
            return collectives.sum(std::move(partials), project.primary_device(), projection.reduce_to_primary, projection.fp16_reduce);
        });
        if (latent_routed && result.dim() == 3) result = moe_weighted_reduce_cuda(result, weights);
        result = finish_latent(std::move(result));
        reduced = true;
    } else {
        // shortcut: local MFQ retains its legacy latent behavior until numerical validation.
        auto active_gate_up = b.gate_up, active_gate = b.gate, active_up = b.up;
        DenseTensor gu, hidden;
        if (b.split) {
            if (b.cpu_gate) active_gate = b.stage(b.cpu_gate, "moe.cpu_offload_gate_h2d");
            if (b.cpu_up) active_up = b.stage(b.cpu_up, "moe.cpu_offload_up_h2d");
            gu = profiler.measure("moe.gate_up_split", [&] {
                const bool reuse = mode.reuse_input && executor.mmq().mode == KlMmqMode::Default && x.size(0) <= 8 &&
                    active_gate && active_up && cuda::can_reuse_activation_for(*active_gate, *active_up);
                TYLOINFER_RUNTIME_CHECK(active_gate, "MFE input must be contiguous CUDA f16 with exact K");
                auto g = project.run(*active_gate, x, b.route);
                if (!b.bundle_prefetched && active_up) cuda::prefetch_experts(*active_up, b.route);
                shared_gate();
                TYLOINFER_RUNTIME_CHECK(active_up, "MFE input must be contiguous CUDA f16 with exact K");
                auto u = reuse ? project.run(*active_up, x, b.route, true) : project.run(*active_up, x, b.route);
                return cuda::cat({g, u}, -1).contiguous();
            });
        } else {
            if (b.cpu_gate_up) active_gate_up = b.stage(b.cpu_gate_up, "moe.cpu_offload_gate_up_h2d");
            if (spec.swiglu_limit <= 0 && !mode.materialized && active_gate_up && cuda::supports_projection_glu_epilogue(*active_gate_up))
                hidden = profiler.measure("moe.gate_up_swiglu", [&] { return project.glu(*active_gate_up, x, b.route, false); });
            else
                gu = profiler.measure("moe.gate_up", [&] {
                    TYLOINFER_RUNTIME_CHECK(active_gate_up, "MFE input must be contiguous CUDA f16 with exact K");
                    return project.run(*active_gate_up, x, b.route);
                });
        }
        active_gate_up.reset(); active_gate.reset(); active_up.reset();
        if (!b.bundle_prefetched && b.down) cuda::prefetch_experts(*b.down, b.route);
        auto active_down = b.down;
        if (b.cpu_down) active_down = b.stage(b.cpu_down, "moe.cpu_offload_down_h2d");
        if (hidden.defined()) {
            result = profiler.measure("moe.down", [&] {
                TYLOINFER_RUNTIME_CHECK(active_down, "MFE input must be contiguous CUDA f16 with exact K");
                return project.run(*active_down, hidden, b.route);
            });
        } else {
            const bool fusion = !mode.materialized && options.swiglu_fusion &&
                (gu.size(0) == 1 || (gu.size(0) <= 4 && options.small_heterogeneous));
            if (fusion && spec.swiglu_limit <= 0)
                result = profiler.measure("moe.swiglu_down", [&] {
                    TYLOINFER_RUNTIME_CHECK(active_down, "fused MFE GLU input has an unsupported layout");
                    return project.gated(*active_down, gu, b.route, false);
                });
            else if (fusion && active_down && cuda::supports_clamped_swiglu(*active_down))
                result = profiler.measure("moe.swiglu_down", [&] { return project.clamped(*active_down, gu, b.route, spec.swiglu_limit); });
            else {
                hidden = profiler.measure("moe.swiglu", [&] {
                    if (spec.swiglu_limit <= 0) return moe_swiglu_split_cuda(gu);
                    const auto width = gu.size(-1) / 2;
                    auto g = cuda::clamp_max(gu.slice(-1, 0, width).to(kFloat32), spec.swiglu_limit);
                    auto u = cuda::clamp(gu.slice(-1, width, 2 * width).to(kFloat32), -spec.swiglu_limit, spec.swiglu_limit);
                    return (cuda::silu(g) * u).to(kFloat16).contiguous();
                });
                result = profiler.measure("moe.down", [&] {
                    TYLOINFER_RUNTIME_CHECK(active_down, "MFE input must be contiguous CUDA f16 with exact K");
                    return project.run(*active_down, hidden, b.route);
                });
            }
        }
        stats.record(spec.layer, ids, weights, result, b.split ? b.gate->n_experts : b.gate_up->n_experts);
        active_down.reset();
    }
#else
    else throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
    shared_gate();
    return {owner(std::move(result)), owner(std::move(logits)), reduced};
}

Tensor combine_experts(const RoutedExperts& experts, const Tensor& weights_tensor, const Tensor& shared_tensor) {
    const auto& values = dense(experts.values);
    const auto& weights = dense(weights_tensor);
    const auto& shared = dense(shared_tensor);
    const auto& logits = dense(experts.shared_logits);
    compatible(experts.values, weights_tensor);
    compatible(experts.values, shared_tensor);
    compatible(experts.values, experts.shared_logits);
    require_cuda(experts.values);
    if (shared.dim() != 2 || shared.scalar_type() != kFloat16 || !shared.is_contiguous() ||
        weights.dim() != 2 || weights.scalar_type() != kFloat32 || !weights.is_contiguous() ||
        logits.dim() != 2 || logits.size(0) != shared.size(0) || logits.size(1) != 1 ||
        logits.scalar_type() != kFloat32 || !logits.is_contiguous() ||
        (experts.reduced ? values.dim() != 2 || values.sizes() != shared.sizes() :
         values.dim() != 3 || values.size(0) != shared.size(0) || values.size(1) != weights.size(1) || values.size(2) != shared.size(1)) ||
        weights.size(0) != shared.size(0))
        throw std::invalid_argument("expert combination geometry mismatch");
    auto& executor = cuda::Executor::current();
    auto& profiler = executor.profiler();
    if (!experts.reduced && !cuda::Executor::current_ffn().unfused && executor.options().ffn.reduce_fusion && values.size(0) <= 8)
        return owner(profiler.measure("moe.reduce_combine", [&] {
            return moe_weighted_reduce_shared_gate_cuda(values, weights, shared, logits);
        }));
    auto reduced = experts.reduced ? values : profiler.measure("moe.reduce", [&] { return moe_weighted_reduce_cuda(values, weights); });
    return owner(profiler.measure("moe.combine", [&] { return moe_add_shared_gate_cuda(reduced.contiguous(), shared, logits); }));
}

RoutedGeglu routed_geglu(const Tensor& value_tensor, const Tensor& ids_tensor,
                         const Tensor& gate_up, const Tensor& down,
                         bool trace) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& ids = ::tyloi::ops::dense(ids_tensor);
    compatible(value, ids);
    if (!gate_up.defined() || !down.defined() || gate_up.dim() != 3 || down.dim() != 3 ||
        value.dim() != 2 || ids.dim() != 2 || value.size(0) != ids.size(0) ||
        value.scalar_type() != kFloat16 || ids.scalar_type() != kInt32 ||
        value.size(0) <= 0 || value.size(1) <= 0 || ids.size(1) <= 0 ||
        gate_up.size(0) <= 0 || gate_up.size(1) <= 0 || down.size(1) <= 0 ||
        gate_up.size(0) != down.size(0) || gate_up.size(2) != value.size(-1) ||
        gate_up.size(1) % 2 != 0 || gate_up.size(1) / 2 != down.size(2))
        throw std::invalid_argument("routed GEGLU input, routes, and expert banks are incompatible");
    require_cuda(value);
    return cuda::ops::routed_geglu(value, ids, gate_up, down, trace);
}

Tensor rms_norm(const Tensor& value_tensor, const Tensor& weight_tensor, double eps, double offset, ScalarType output_type) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& weight = ::tyloi::ops::dense(weight_tensor);
    input(value);
    input(weight);
    if (value.device() != weight.device() || value.dim() == 0 || weight.dim() == 0 ||
        value.size(-1) <= 0 || weight.numel() != value.size(-1) ||
        weight.size(-1) != value.size(-1))
        throw std::invalid_argument("rms_norm input and weight are incompatible");
    if (output_type != ScalarType::float32 && output_type != ScalarType::float16 &&
        output_type != ScalarType::bfloat16)
        throw std::invalid_argument("unsupported rms_norm output type");
    if (!value.is_cuda()) {
        auto normalized = qwen_rms_norm(value.reshape({-1, value.size(-1)}).to(kFloat32),
                                        weight_tensor, eps, offset);
        return owner(normalized.to(output_type).contiguous().reshape(value.sizes()));
    }
    return owner(cuda::ops::rms_norm(value, weight, eps, offset, output_type));
}
Tensor layer_norm(const Tensor& value_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor, double epsilon) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& weight = ::tyloi::ops::dense(weight_tensor);
    const auto& bias = ::tyloi::ops::dense(bias_tensor);
    compatible(value, weight);
    compatible(value, bias);
    if (value.dim() == 0 || weight.dim() != 1 || bias.dim() != 1 ||
        weight.size(0) != value.size(-1) || bias.size(0) != value.size(-1))
        throw std::invalid_argument("layer_norm input and weights are incompatible");
    require_cuda(value);
    return owner(cuda::ops::layer_norm(value, weight, bias, epsilon));
}
Tensor gelu(const Tensor& value_tensor, bool approximate) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::gelu(value, approximate));
}
Tensor silu(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::silu(value));
}
Tensor relu_squared(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::relu_squared(value));
}
Tensor axial_rope(const Tensor& value_tensor, const std::vector<std::int32_t>& positions, double theta) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::axial_rope(value, positions, theta));
}
Tensor segmented_attention(const Tensor& query_tensor, const Tensor& key_tensor, const Tensor& value_tensor,
                           const std::vector<std::int32_t>& lengths) {
    const auto& query = ::tyloi::ops::dense(query_tensor);
    const auto& key = ::tyloi::ops::dense(key_tensor);
    const auto& value = ::tyloi::ops::dense(value_tensor);
    compatible(query, key);
    compatible(query, value);
    require_cuda(query);
    return owner(cuda::ops::segmented_attention(query, key, value, lengths));
}
std::array<Tensor, 3> qkv_norm(std::array<Tensor, 3> values, std::array<Tensor, 3> weights,
                               double eps, double offset, ScalarType output_type) {
    if (output_type != ScalarType::float32 && output_type != ScalarType::bfloat16)
        throw std::invalid_argument("unsupported qkv_norm output type");
    std::array<const DenseTensor*, 3> dense_weights{};
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!values[i].defined()) throw std::invalid_argument("qkv_norm requires all values");
        input(values[i]);
        if (weights[i].defined()) {
            dense_weights[i] = &dense(weights[i]);
            compatible(values[i], weights[i]);
            if (values[i].dim() == 0 || weights[i].dim() == 0 ||
                values[i].size(-1) <= 0 || weights[i].numel() != values[i].size(-1))
                throw std::invalid_argument("qkv_norm input and weight are incompatible");
        }
    }
    for (std::size_t i = 0; i < values.size(); ++i)
        if (weights[i].defined()) require_cuda(values[i]);
    (void)cuda::Executor::current();
    std::array<DenseTensor, 3> dense_values{dense(values[0]), dense(values[1]), dense(values[2])};
    auto result = cuda::ops::qkv_norm(std::move(dense_values), dense_weights, eps, offset, output_type);
    return {owner(std::move(result[0])), owner(std::move(result[1])), owner(std::move(result[2]))};
}
Tensor softcap(const Tensor& value_tensor, double cap) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    if (cap > 0.0) require_cuda(value);
    return owner(cuda::ops::softcap(value, cap));
}
Tensor scale(const Tensor& value_tensor, double factor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::scale(value, factor));
}
Tensor divide(const Tensor& value_tensor, double divisor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::divide(value, divisor));
}
Tensor embedding(const Tensor& ids_tensor, const ::tyloi::Tensor& weight) {
    const auto& ids = ::tyloi::ops::dense(ids_tensor);
    input(ids);
    return owner(cuda::ops::embedding(ids, weight));
}
Tensor project(const Tensor& value_tensor, const ::tyloi::Tensor& weight) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::project(value, weight));
}
Tensor head_project(const Tensor& value, const Tensor& weight) {
    input(value);
    require_cuda(value);
    if (!weight || !weight.defined() || value.dim() != 4 || weight.dim() != 3 ||
        value.numel() <= 0 || value.size(2) <= 0 || value.size(2) > std::numeric_limits<int>::max() ||
        value.size(3) <= 0 || value.size(2) != weight.size(0) || value.size(3) != weight.size(2))
        throw std::invalid_argument("head-wise projection geometry is incompatible");
    return owner(cuda::ops::head_project(dense(value), weight));
}

Tensor project(const Tensor& value, const Tensor& weight, const Tensor& ids) {
    compatible(value, ids);
    return owner(cuda::ops::project(dense(value), weight, dense(ids)));
}
Tensor low_rank(const Tensor& value_tensor, const Tensor& output_a, const Tensor& output_b,
                std::int64_t groups, bool groupwise, bool profile) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::low_rank(value, output_a, output_b, groups, groupwise, profile));
}
Tensor linear(const Tensor& value_tensor, const ::tyloi::Tensor& weight, const Tensor& bias) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    const auto* bias_dense = bias.defined() ? &::tyloi::ops::dense(bias) : nullptr;
    if (bias_dense) input(*bias_dense);
    return owner(cuda::ops::linear(value, weight, bias_dense));
}
Tensor affine(const Tensor& value_tensor, const ::tyloi::Tensor& weight, const Tensor& bias) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    const auto* bias_dense = bias.defined() ? &::tyloi::ops::dense(bias) : nullptr;
    if (bias_dense) input(*bias_dense);
    return owner(cuda::ops::affine(value, weight, bias_dense));
}
Tensor mlp(const Tensor& value_tensor, std::span<const Tensor> gate_up, const ::tyloi::Tensor& down,
           MlpActivation activation, double swiglu_limit) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::mlp(value, gate_up, down, activation, swiglu_limit));
}
Tensor bf16_mlp(const Tensor& value, std::span<const Tensor> gate_up, const Tensor& down,
        MlpActivation activation, double swiglu_limit) {
    input(value);
    return owner(cuda::ops::bf16_mlp(dense(value), gate_up, down, activation, swiglu_limit));
}
Tensor additive_mlp(const Tensor& value,
        std::span<const Tensor> gate_up, const Tensor& down, MlpActivation activation, double limit,
        std::span<const Tensor> extra_gate_up, const Tensor& extra_down,
        MlpActivation extra_activation, double extra_limit) {
    input(value);
    return owner(cuda::ops::additive_mlp(dense(value), gate_up, down, activation, limit,
        extra_gate_up, extra_down, extra_activation, extra_limit));
}
std::optional<Tensor> mlp_residual(const Tensor& value, const Tensor& residual,
        std::span<const Tensor> gate_up, const Tensor& down, MlpActivation activation,
        double swiglu_limit, bool recurrent) {
    compatible(value, residual);
    if (value.dim() != 2 || value.sizes() != residual.sizes())
        throw std::invalid_argument("fused MLP residual inputs must have matching [rows,width] shapes");
    auto result = cuda::ops::mlp_residual(dense(value), dense(residual), gate_up, down,
        activation, swiglu_limit, recurrent);
    if (!result) return std::nullopt;
    return owner(std::move(*result));
}
std::vector<Tensor> project_group(const Tensor& value_tensor, std::span<const Tensor> group) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owners(cuda::ops::project_group(value, group));
}
bool grouped(std::span<const Tensor> weights) {
    return cuda::ops::grouped(weights);
}
std::vector<Tensor> project_group(const Tensor& value_tensor, std::span<const Tensor> weights,
                                  std::size_t prefix) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owners(cuda::ops::project_group(value, weights, prefix));
}
std::array<Tensor, 2> linear_gates(const Tensor& alpha, const Tensor& beta, const Tensor& dt_bias, const Tensor& a_log) {
    compatible(alpha, beta); compatible(alpha, dt_bias); compatible(alpha, a_log);
    const auto& a = dense(alpha); const auto& b = dense(beta);
    if (a.dim() != 3 || a.sizes() != b.sizes() || dt_bias.numel() != a.size(2) || a_log.numel() != a.size(2))
        throw std::invalid_argument("linear gate geometry mismatch");
    if (a.is_cuda()) {
        auto result = linear_gate_beta_cuda(a, b, dense(dt_bias), dense(a_log));
        return {owner(result[0]), owner(result[1])};
    }
    auto gate = cuda::softplus(a.to(kFloat32) + dense(dt_bias).to(kFloat32).reshape({1, 1, a.size(2)})) *
        (-cuda::exp(dense(a_log).to(kFloat32))).reshape({1, 1, a.size(2)});
    return {owner(gate.transpose(1, 2).contiguous()),
            owner(cuda::sigmoid(b.to(kFloat32)).transpose(1, 2).contiguous())};
}

std::array<Tensor, 3> linear_conv(Tensor& convolution, const Tensor& qkv_or_qk, const Tensor& value,
        const Tensor& weight, const Tensor& bias, std::int64_t nk, std::int64_t nv,
        std::int64_t dk, std::int64_t dv, double eps, bool replay) {
    compatible(convolution, qkv_or_qk); compatible(convolution, weight);
    if (value.defined()) compatible(convolution, value);
    if (bias.defined()) compatible(convolution, bias);
    auto& state = dense(convolution); const auto& x = dense(qkv_or_qk); const auto& w = dense(weight);
    const auto ksz = nk * dk, vsz = nv * dv, channels = 2 * ksz + vsz;
    if (nk <= 0 || nv <= 0 || dk <= 0 || dv <= 0 || nv % nk || x.dim() != 3 || state.dim() != 3 ||
        x.size(0) != state.size(0) || state.size(2) != channels || x.size(1) <= 0 ||
        x.size(2) != (value.defined() ? 2 * ksz : channels) ||
        (value.defined() && (value.dim() != 3 || value.size(0) != x.size(0) || value.size(1) != x.size(1) || value.size(2) != vsz)))
        throw std::invalid_argument("linear convolution geometry mismatch");
    const auto B = x.size(0), T = x.size(1), kernel = state.size(1) + 1;
    const bool valid_weight = (w.dim() == 3 && w.size(1) == 1 &&
        ((w.size(0) == channels && w.size(2) == kernel) ||
         (!state.is_cuda() && w.size(0) != channels && w.size(0) == kernel && w.size(2) == channels))) ||
        (w.dim() == 2 && ((w.size(0) == channels && w.size(1) == kernel) ||
                          (w.size(0) != channels && w.size(1) == channels && w.size(0) == kernel)));
    if (!valid_weight || (bias.defined() && bias.numel() != 0 &&
                          (bias.dim() != 1 || bias.size(0) != channels)))
        throw std::invalid_argument("linear convolution weight or bias geometry mismatch");
    if (state.is_cuda() && (!state.is_contiguous() || state.scalar_type() != kFloat32 ||
        !w.is_contiguous() || w.scalar_type() != kFloat32 ||
        (bias.defined() && (!dense(bias).is_contiguous() || dense(bias).scalar_type() != kFloat32))))
        throw std::invalid_argument("CUDA linear convolution state, weight and bias must be contiguous float32");
    auto empty_bias = bias.defined() ? dense(bias) : cuda::empty({0}, TensorOptions().device(state.device()).dtype(kFloat32));
    if (!replay && state.is_cuda() && value.defined() && (T == 1 || (T >= 256 && cuda::Executor::current().mode().linear.fused_conv))) {
        auto result = T == 1 ? linear_conv_qkv_decode_cuda(state, x.contiguous(), dense(value).contiguous(), w, empty_bias, nk, nv, dk, dv, eps)
            : linear_conv_qkv_prefill_cuda(state, x.contiguous(), dense(value).contiguous(), w, empty_bias, nk, nv, dk, dv, eps);
        if (T != 1) state.copy_(result[3]);
        return {owner(result[0]), owner(result[1]), owner(result[2])};
    }
    auto qkv = value.defined() ? cuda::cat({x.to(kFloat32), dense(value).to(kFloat32)}, -1) : x.to(kFloat32);
    DenseTensor conv;
    if (!replay && state.is_cuda() && T == 1) conv = ssm_conv_silu_decode_cuda(state, qkv, w, empty_bias);
    else {
        auto conv_input = cuda::cat({state, qkv}, 1);
        state.copy_(conv_input.narrow(1, conv_input.size(1) - (kernel - 1), kernel - 1));
        if (state.is_cuda()) conv = ssm_conv_silu_cuda(conv_input, w, empty_bias, T);
        else {
            auto wf = w.to(kFloat32);
            if (wf.dim() == 3) wf = wf.squeeze(1);
            if (wf.size(0) != channels && wf.size(1) == channels) wf = wf.transpose(0, 1).contiguous();
            conv = (conv_input.unfold(1, kernel, 1) * wf.reshape({1, 1, channels, kernel})).sum(-1);
            if (bias.defined() && bias.numel() != 0) conv = conv + dense(bias).to(kFloat32).reshape({1, 1, channels});
            conv = cuda::silu(conv);
        }
    }
    auto q = conv.narrow(2, 0, ksz).reshape({B,T,nk,dk}).transpose(1,2);
    auto k = conv.narrow(2, ksz, ksz).reshape({B,T,nk,dk}).transpose(1,2);
    auto v = conv.narrow(2, 2*ksz, vsz).reshape({B,T,nv,dv}).transpose(1,2);
    if (state.is_cuda()) {
        q = l2_norm_cuda(q.contiguous().reshape({-1, dk}), eps).reshape_as(q);
        k = l2_norm_cuda(k.contiguous().reshape({-1, dk}), eps).reshape_as(k);
    } else {
        q = q.contiguous(); k = k.contiguous(); v = v.contiguous();
        q = q / cuda::sqrt(q.square().sum(-1, true)).clamp_min(eps);
        k = k / cuda::sqrt(k.square().sum(-1, true)).clamp_min(eps);
    }
    return {owner(q), owner(k), owner(v)};
}

Tensor linear_recur(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& gate,
        const Tensor& beta, Tensor& recurrent, bool transposed, bool tiled) {
    compatible(query, key); compatible(query, value); compatible(query, gate); compatible(query, beta); compatible(query, recurrent);
    auto q = dense(query), k = dense(key), v = dense(value); auto& state = dense(recurrent);
    const auto& g = dense(gate); const auto& b = dense(beta);
    if (q.dim() != 4 || q.sizes() != k.sizes() || v.dim() != 4 || g.dim() != 3 || g.sizes() != b.sizes() ||
        state.dim() != 4 || q.size(0) != v.size(0) || q.size(2) != v.size(2) || q.size(1) <= 0 ||
        v.size(1) % q.size(1) || g.size(0) != v.size(0) || g.size(1) != v.size(1) || g.size(2) != v.size(2) ||
        state.size(0) != v.size(0) || state.size(1) != v.size(1) ||
        state.size(2) != (transposed ? v.size(3) : q.size(3)) || state.size(3) != (transposed ? q.size(3) : v.size(3)))
        throw std::invalid_argument("linear recurrence geometry mismatch");
    if (q.is_cuda()) {
        std::vector<DenseTensor> result;
        if (transposed) result = tiled ? gdn_inplace_transposed_tiled_cuda(q.contiguous(), k.contiguous(), v.contiguous(), g, b, state)
                                       : gdn_inplace_transposed_cuda(q.contiguous(), k.contiguous(), v.contiguous(), g, b, state);
        else result = tiled ? gdn_inplace_tiled_cuda(q.contiguous(), k.contiguous(), v.contiguous(), g, b, state)
                            : gdn_inplace_cuda(q.contiguous(), k.contiguous(), v.contiguous(), g, b, state);
        recurrent = owner(result[1]);
        return owner(result[0]);
    }
    if (q.size(1) != v.size(1)) {
        auto repeat = v.size(1) / q.size(1);
        q = tiled ? q.repeat({1, repeat, 1, 1}) : q.repeat_interleave(repeat, 1);
        k = tiled ? k.repeat({1, repeat, 1, 1}) : k.repeat_interleave(repeat, 1);
    }
    auto r = transposed ? state.transpose(-1,-2) : state;
    std::vector<DenseTensor> outputs;
    for (std::int64_t t = 0; t < v.size(2); ++t) {
        auto qt = q.select(2,t), kt = k.select(2,t), vt = v.select(2,t);
        r = r * cuda::exp(g.select(2,t)).reshape({v.size(0),v.size(1),1,1});
        auto delta = (vt - cuda::matmul(r.transpose(-1,-2), kt.unsqueeze(-1)).squeeze(-1)) * b.select(2,t).unsqueeze(-1);
        r = r + kt.unsqueeze(-1) * delta.unsqueeze(-2);
        outputs.push_back(cuda::matmul(r.transpose(-1,-2), qt.unsqueeze(-1)).squeeze(-1) * (1.0 / std::sqrt(static_cast<double>(q.size(3)))));
    }
    recurrent = owner((transposed ? r.transpose(-1,-2) : r).contiguous());
    return owner(cuda::stack(outputs, 2));
}

std::optional<std::vector<Tensor>> linear_shared(const Tensor& input_tensor, std::span<const Tensor> qkv,
        const Tensor& z, std::span<const Tensor> ab) {
    const auto& value = dense(input_tensor);
    input(value);
    if (!value.is_cuda() || !cuda::Executor::current().mode().linear.shared_input || qkv.size() != 2)
        return std::nullopt;
    std::vector<Tensor> projections{qkv[0], qkv[1], z};
    if (ab.size() == 2 && ab[0].representation<ShardedTensor>() && ab[1].representation<ShardedTensor>()) {
        projections.push_back(ab[0]);
        projections.push_back(ab[1]);
    }
    if (!tensor_parallel_output_projections_compatible(projections)) return std::nullopt;
    return measure("linear.split_projections", [&] {
        return owners(cuda::Executor::current().projector(cuda::Executor::current_ffn()).grouped(projections, value));
    });
}

Tensor silu_project(const Tensor& input_tensor, const Tensor& gate_tensor, const Tensor& weight) {
    compatible(input_tensor, gate_tensor);
    const auto& value = dense(input_tensor);
    const auto& gate = dense(gate_tensor);
    if (const auto* w = weight.representation<DenseTensor>(); w && !w->is_encoded()) {
        compatible(input_tensor, weight);
        if (value.dim() == 0 || value.sizes() != gate.sizes() || w->dim() != 2 || value.size(-1) != w->size(1))
            throw std::invalid_argument("dense gated projection geometry mismatch");
        auto shape = value.sizes().vec();
        shape.back() = w->size(0);
        auto gated = value.to(w->scalar_type()) * cuda::silu(gate.to(w->scalar_type()));
        return owner(dense_projection(gated.reshape({-1, value.size(-1)}), *w).reshape(shape));
    }
    return owner(cuda::Executor::current().projector(cuda::Executor::current_ffn()).gated(
        weight, value, gate, InputGateMode::Silu));
}
Tensor sigmoid_project(const Tensor& value_tensor, const Tensor& gate_tensor, const ::tyloi::Tensor& weight) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& gate = ::tyloi::ops::dense(gate_tensor);
    compatible(value, gate);
    return owner(cuda::ops::sigmoid_project(value, gate, weight));
}
Tensor tensor(std::span<const float> values, Device target) {
    device(target);
    return owner(cuda::ops::tensor(values, target));
}
Tensor tensor(std::span<const std::int32_t> values, Device target) {
    device(target);
    return owner(cuda::ops::tensor(values, target));
}
Tensor tensor(std::span<const std::int64_t> values, Device target) {
    device(target);
    return owner(cuda::ops::tensor(values, target));
}
Tensor tensor(std::span<const std::uint8_t> values, Device target) {
    device(target);
    return owner(cuda::ops::tensor(values, target));
}
Tensor from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options) {
    device(options.target_device());
    return owner(cuda::ops::from_blob(data, shape, options));
}
Tensor copy(const Tensor& value_tensor, Device target) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    device(target);
    return owner(cuda::ops::copy(value, target));
}
void copy_into(Tensor destination_tensor, const Tensor& source_tensor) {
    auto& destination = ::tyloi::ops::dense(destination_tensor);
    const auto& source = ::tyloi::ops::dense(source_tensor);
    input(destination);
    input(source);
    cuda::ops::copy_into(destination, source);
}
Tensor cast(const Tensor& value_tensor, ScalarType type) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::cast(value, type));
}
Tensor cast(const Tensor& value_tensor, ScalarType type, Device target) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    device(target);
    return owner(cuda::ops::cast(value, type, target));
}
Tensor contiguous(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::contiguous(value));
}
Tensor clone(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::clone(value));
}
Tensor reshape(const Tensor& value_tensor, std::span<const std::int64_t> shape) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::reshape(value, shape));
}
Tensor expand(const Tensor& value_tensor, std::span<const std::int64_t> shape) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::expand(value, shape));
}
Tensor cat(std::span<const Tensor> values, std::int64_t dim) {
    if (values.empty()) throw std::invalid_argument("cat requires at least one tensor");
    std::vector<DenseTensor> dense_values;
    dense_values.reserve(values.size());
    for (const auto& value : values) {
        if (!value.defined()) throw std::invalid_argument("cat requires defined tensors");
        if (dense_values.empty()) input(value);
        else compatible(dense_values.front(), value);
        dense_values.push_back(dense(value));
    }
    return owner(cuda::ops::cat(dense_values, dim));
}
Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
    device(options.target_device());
    return owner(cuda::ops::zeros(shape, options));
}
Tensor empty(std::span<const std::int64_t> shape, const TensorOptions& options) {
    device(options.target_device());
    return owner(cuda::empty(shape, options));
}
Tensor sample_greedy(const Tensor& logits_tensor) {
    const auto& logits = dense(logits_tensor);
    input(logits);
    require_cuda(logits);
    return owner(::sample_greedy_cuda(logits));
}

void decode_commit(const Tensor& next, Tensor& generated, Tensor& step,
                   Tensor& input_tensor, Tensor& position, Tensor& length) {
    input(next);
    require_cuda(next);
    const std::array<const Tensor*, 6> operands{&next, &generated, &step, &input_tensor, &position, &length};
    for (const auto* value : operands) compatible(next, *value);
    for (const auto* value : operands) {
        const auto& item = dense(*value);
        if (item.scalar_type() != kInt64 || !item.is_contiguous())
            throw std::invalid_argument("decode commit requires contiguous int64 tensors");
    }
    if (dense(next).numel() != 1 || dense(generated).numel() < 1 || dense(step).numel() != 1 ||
        dense(input_tensor).numel() != 1 || dense(position).numel() != 1 || dense(length).numel() != 1)
        throw std::invalid_argument("invalid decode commit geometry");
    cuda::ops::decode_commit(dense(next), dense(generated), dense(step), dense(input_tensor), dense(position), dense(length));
}
Tensor sample_stochastic(const Tensor& logits_tensor, const Tensor& random_tensor,
                                          double temperature, std::int64_t top_k, double top_p) {
    const auto& logits = dense(logits_tensor);
    const auto& random = dense(random_tensor);
    compatible(logits, random);
    require_cuda(logits);
    if (top_k > 0)
        return owner(::sample_top_k_top_p_cuda(logits, random, temperature, top_k, top_p));
    return owner(::sample_softmax_cuda(logits, random, temperature));
}
void add_token_counts(Tensor& counts_tensor, const Tensor& ids_tensor) {
    auto& counts = dense(counts_tensor);
    const auto& ids = dense(ids_tensor);
    compatible(counts, ids);
    require_cuda(counts);
    ::sample_token_counts_add_cuda(counts, ids);
}
void apply_penalties(Tensor& logits_tensor, const Tensor& counts_tensor,
                     double presence, double frequency, double repetition) {
    auto& logits = dense(logits_tensor);
    const auto& counts = dense(counts_tensor);
    compatible(logits, counts);
    require_cuda(logits);
    ::sample_apply_penalties_cuda(logits, counts, presence, frequency, repetition);
}
Tensor constrain(const Tensor& logits_tensor, const MfqTokenConstraintPtr& constraint) {
    const auto& logits = dense(logits_tensor);
    input(logits);
    auto masked = logits.to(kCPU, kFloat32).contiguous();
    constraint->apply(masked.data_ptr<float>(), static_cast<std::size_t>(masked.numel()));
    return owner(std::move(masked));
}
std::int64_t item(const Tensor& value_tensor) {
    const auto& value = dense(value_tensor);
    input(value);
    return value.item<std::int64_t>();
}
Tensor select(const Tensor& value_tensor, std::int64_t dim, std::int64_t index) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::select(value, dim, index));
}
Tensor unsqueeze(const Tensor& value_tensor, std::int64_t dim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::unsqueeze(value, dim));
}
Tensor slice(const Tensor& value_tensor, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::slice(value, dim, start, end, step));
}
Tensor index_select(const Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    compatible(value, indices);
    require_cuda(value);
    return owner(cuda::ops::index_select(value, dim, indices));
}
Tensor sum(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::sum(value, dim, keepdim));
}
void index_copy(Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor, const Tensor& source_tensor) {
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    const auto& source = ::tyloi::ops::dense(source_tensor);
    compatible(value, indices);
    compatible(value, source);
    require_cuda(value);
    cuda::ops::index_copy(value, dim, indices, source);
}
std::vector<Tensor> chunk(const Tensor& value_tensor, std::int64_t count, std::int64_t dim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owners(tyloi::chunk(value, count, dim));
}
Tensor arange(std::int64_t first, std::int64_t last, Device target) {
    device(target);
    return owner(cuda::ops::arange(first, last, target));
}
Tensor full(std::span<const std::int64_t> shape, std::int64_t value, Device target) {
    device(target);
    return owner(cuda::ops::full(shape, value, target));
}
Tensor add(const Tensor& value_tensor, std::int64_t offset) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::add(value, offset));
}
Tensor mean(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::mean(value, dim, keepdim));
}
Tensor amax(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::amax(value, dim, keepdim));
}
Tensor argmax(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::argmax(value, dim, keepdim));
}
Tensor all(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::all(value, dim, keepdim));
}
Tensor any(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::any(value, dim, keepdim));
}
Tensor abs(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::abs(value));
}
Tensor square(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::square(value));
}
Tensor exp(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::exp(value));
}
Tensor sqrt(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::sqrt(value));
}
Tensor logical_not(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::logical_not(value));
}
Tensor clamp(const Tensor& value_tensor, double minimum, double maximum) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::clamp(value, minimum, maximum));
}
Tensor clamp_min(const Tensor& value_tensor, double minimum) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::clamp_min(value, minimum));
}
Tensor clamp_max(const Tensor& value_tensor, double maximum) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::clamp_max(value, maximum));
}
Tensor remainder(const Tensor& value_tensor, double divisor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::remainder(value, divisor));
}
Tensor eq(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::eq(left, right));
}
Tensor eq(const Tensor& value_tensor, double other) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::eq(value, other));
}
Tensor ne(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::ne(left, right));
}
Tensor ne(const Tensor& value_tensor, double other) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::ne(value, other));
}
bool equal(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return cuda::ops::equal(left, right);
}
Tensor masked_select(const Tensor& value_tensor, const Tensor& mask_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& mask = ::tyloi::ops::dense(mask_tensor);
    compatible(value, mask);
    require_cuda(value);
    return owner(cuda::ops::masked_select(value, mask));
}
Tensor gather(const Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    compatible(value, indices);
    require_cuda(value);
    return owner(cuda::ops::gather(value, dim, indices));
}
Tensor repeat(const Tensor& value_tensor, std::span<const std::int64_t> repeats) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::repeat(value, repeats));
}
Tensor repeat_interleave(const Tensor& value_tensor, std::int64_t repeats, std::int64_t dim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::repeat_interleave(value, repeats, dim));
}
Tensor tril(const Tensor& value_tensor, std::int64_t diagonal) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::tril(value, diagonal));
}
Tensor masked_fill(const Tensor& value_tensor, const Tensor& mask_tensor, double fill_value) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& mask = ::tyloi::ops::dense(mask_tensor);
    compatible(value, mask);
    require_cuda(value);
    return owner(cuda::ops::masked_fill(value, mask, fill_value));
}
void masked_fill_inplace(Tensor& value_tensor, const Tensor& mask_tensor, double fill_value) {
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& mask = ::tyloi::ops::dense(mask_tensor);
    compatible(value, mask);
    require_cuda(value);
    cuda::ops::masked_fill_inplace(value, mask, fill_value);
}
void fill(Tensor& value_tensor, double fill_value) {
    auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    if (value.is_cuda()) {
        cuda::ops::fill(value, fill_value);
        return;
    }
    if (!value.is_contiguous())
        throw std::invalid_argument("CPU fill requires a contiguous tensor");
    if (fill_value == 0.0) {
        std::memset(value.data_ptr(), 0, value.nbytes());
        return;
    }
    switch (value.scalar_type()) {
        case ScalarType::boolean:
            std::fill_n(value.data_ptr<bool>(), value.numel(), fill_value != 0.0);
            break;
        case ScalarType::uint8:
            std::fill_n(value.data_ptr<std::uint8_t>(), value.numel(), static_cast<std::uint8_t>(fill_value));
            break;
        case ScalarType::int8:
            std::fill_n(value.data_ptr<std::int8_t>(), value.numel(), static_cast<std::int8_t>(fill_value));
            break;
        case ScalarType::int16:
            std::fill_n(value.data_ptr<std::int16_t>(), value.numel(), static_cast<std::int16_t>(fill_value));
            break;
        case ScalarType::int32:
            std::fill_n(value.data_ptr<std::int32_t>(), value.numel(), static_cast<std::int32_t>(fill_value));
            break;
        case ScalarType::int64:
            std::fill_n(value.data_ptr<std::int64_t>(), value.numel(), static_cast<std::int64_t>(fill_value));
            break;
        case ScalarType::float16:
            std::fill_n(value.data_ptr<std::uint16_t>(), value.numel(),
                cuda::encode_float16(static_cast<float>(fill_value)));
            break;
        case ScalarType::bfloat16:
            std::fill_n(value.data_ptr<std::uint16_t>(), value.numel(),
                cuda::encode_bfloat16(static_cast<float>(fill_value)));
            break;
        case ScalarType::float32:
            std::fill_n(value.data_ptr<float>(), value.numel(), static_cast<float>(fill_value));
            break;
        case ScalarType::float64:
            std::fill_n(value.data_ptr<double>(), value.numel(), fill_value);
            break;
        default:
            throw std::invalid_argument("CPU fill for this scalar type is not implemented yet");
    }
}
void scatter_add(Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor, const Tensor& source_tensor) {
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    const auto& source = ::tyloi::ops::dense(source_tensor);
    compatible(value, indices);
    compatible(value, source);
    require_cuda(value);
    cuda::ops::scatter_add(value, dim, indices, source);
}
void scatter(Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor, const Tensor& source_tensor) {
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    const auto& source = ::tyloi::ops::dense(source_tensor);
    compatible(value, indices);
    compatible(value, source);
    require_cuda(value);
    cuda::ops::scatter(value, dim, indices, source);
}
void index_fill(Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor, double fill_value) {
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    compatible(value, indices);
    require_cuda(value);
    cuda::ops::index_fill(value, dim, indices, fill_value);
}
#endif
} // namespace tyloi::ops
