#include "quant_linear.h"
#include "engine/backends/cuda/ops/embedding.h"
#include "engine/backends/cuda/ops/projection/quant.h"
#include "engine/backends/cuda/ops/projection/channel_fp8.h"
#include "engine/backends/cuda/ops/gguf/projection.h"
#include "engine/backends/cuda/ops/gguf/prefill_fusion.h"
#include "engine/backends/cuda/kernels/projection/dense.h"
#include "engine/backends/cuda/kernels/projection/gguf/linear.h"
#include "engine/backends/cuda/ops/projection/prefill.h"
#include "components/error.h"
#include "engine/weights/encoding/sq/mxfp4_decode.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tyloi::cuda {

bool can_prefill_f16_residual(std::int64_t tokens, std::int64_t rows,
        std::int64_t columns, Device device, const DenseTensor& residual) {
    const auto* setting = env_lookup("TYLOINFER_GGUF_PREFILL_TENSORCORE");
    if (tokens <= 32 || tokens > INT_MAX || (setting && setting[0] != '1' && setting[0] != 'a') ||
        ((!setting || setting[0] == 'a') && tokens < 512) || !device.is_cuda() ||
        rows < 1 || rows > INT_MAX || columns < 1 || columns > INT_MAX ||
        !residual.defined() || residual.device() != device || !residual.is_contiguous() ||
        residual.scalar_type() != kFloat16 || residual.dim() < 1 || residual.size(-1) != rows ||
        residual.numel() / rows != tokens) return false;
    const DeviceGuard guard(device.index);
    return gguf::grouped_f16_available(device.index);
}

void prefill_f16_residual_tile(const DenseTensor& input, const DenseTensor& weights,
        const DenseTensor& residual, const DenseTensor& output, std::int64_t first,
        std::int64_t count, void* blas, DenseTensor& scratch) {
    if (input.dim() != 2 || weights.dim() != 2 || output.dim() != 2 ||
        weights.size(1) != input.size(1) || output.size(0) != input.size(0) ||
        count <= 0 || count > weights.size(0) || first < 0 || first > output.size(1) ||
        count > output.size(1) - first || residual.numel() != output.numel() ||
        weights.device() != input.device() || weights.scalar_type() != kFloat16 || !weights.is_contiguous())
        throw std::runtime_error("invalid decoded tile for FP16 prefill residual projection");
    const auto stream = current_stream(input.get_device());
    const auto offset = first * output.element_size();
    const auto* addend = static_cast<const std::byte*>(residual.data_ptr()) + offset;
    auto* destination = static_cast<std::byte*>(output.data_ptr()) + offset;
    auto plan = gguf::prepare_prefill_residual_gemm(input.data_ptr(), weights.data_ptr(),
        addend, destination, input.size(0), count, input.size(1), output.size(1), stream.stream());
    if (plan.workspace_bytes && (!scratch.defined() || scratch.nbytes() < plan.workspace_bytes)) {
        try {
            scratch = empty({static_cast<std::int64_t>(plan.workspace_bytes)}, input.options().dtype(kUInt8));
        } catch (const MfqResourceExhausted&) {
            (void)cudaGetLastError();
            plan = {};
        }
    }
    gguf::prefill_gemm_f16_residual(input.data_ptr(), weights.data_ptr(), addend, destination,
        input.size(0), count, input.size(1), output.size(1), plan,
        scratch.defined() ? scratch.data_ptr() : nullptr, scratch.defined() ? scratch.nbytes() : 0,
        blas, stream.stream());
}

} // namespace tyloi::cuda

namespace {
void check_width(int64_t width, int64_t expected, bool padded) {
    if (width <= 0 || expected <= 0 || (padded ? width > expected : width != expected))
        throw std::invalid_argument("projection input width does not match weight");
}

}  // namespace

void check_fields(const ::tyloi::TensorImpl& weight, ::tyloi::Device device, bool check_device) {
    for (const auto* field : weight.buffers()) {
        field->require_dense();
        if (!check_device || !field->defined() || field->numel() == 0) continue;
        if (field->device() != device)
            throw std::invalid_argument("projection weight fields and input devices do not match");
    }
}

// NINT and NVQ pack input groups, so a narrower activation is accepted.
bool width_is_padded(const ::tyloi::Tensor& weight) {
    return weight.representation<tyloi::NintTensor>() != nullptr ||
           weight.representation<tyloi::NvqTensor>() != nullptr;
}

void check_projection_width(const ::tyloi::Tensor& weight, int64_t width) {
    const auto* sharded = weight.representation<tyloi::ShardedTensor>();
    if (!sharded) {
        check_width(width, weight.size(1), width_is_padded(weight));
        return;
    }
    if (sharded->axis == TensorParallelAxis::Input) {
        check_width(width, weight.size(1), false);
        for (const auto& shard : sharded->shards) {
            if (shard.input_begin < 0 || shard.input_end <= shard.input_begin || shard.input_end > width)
                throw std::invalid_argument("projection input does not cover tensor-parallel shard");
            check_width(shard.input_end - shard.input_begin, shard.weight.size(1),
                        width_is_padded(shard.weight));
        }
    } else if (sharded->axis == TensorParallelAxis::Output) {
        for (const auto& shard : sharded->shards)
            check_width(width, shard.weight.size(1), width_is_padded(shard.weight));
    } else {
        throw std::invalid_argument("tensor-parallel linear has an invalid axis");
    }
}

void check_projection_device(const ::tyloi::Tensor& weight, ::tyloi::Device device) {
    const auto* sharded = weight.representation<tyloi::ShardedTensor>();
    if (!sharded) {
        tyloi::weights::check_placement(projection_placement(weight), device);
        const bool cpu_tq = weight.representation<tyloi::encoding::TqTensor>() && !weight.is_cuda();
        check_fields(weight, device, !cpu_tq);
        return;
    }
    for (const auto& shard : sharded->shards) {
        if (!shard.weight || !shard.weight.defined() || !shard.weight.is_cuda() ||
            shard.weight.get_device() != shard.device)
            throw std::invalid_argument("tensor-parallel weight is not prepared on its device");
        check_fields(shard.weight, ::tyloi::Device(::tyloi::kCUDA, shard.device));
    }
}

void check_projection(const ::tyloi::Tensor& weight, const ::tyloi::DenseTensor& input) {
    input.require_dense();
    if (!input.defined() || input.dim() < 1)
        throw std::invalid_argument("projection input must have a last dimension");
    check_projection_device(weight, input.device());
    check_projection_width(weight, input.size(-1));
}

void check_projection(const QuantLinearGroup& group, const ::tyloi::DenseTensor& input) {
    input.require_dense();
    if (!input.defined() || input.dim() < 1)
        throw std::invalid_argument("projection input must have a last dimension");
    if (group.outs.empty() || group.cols.size() != group.outs.size() ||
        group.placements.size() != group.outs.size())
        throw std::invalid_argument("projection group is not prepared");
    for (const auto out : group.outs)
        if (out <= 0) throw std::invalid_argument("invalid projection group output width");
    if (!group.layers.empty()) {
        if (group.layers.size() != group.outs.size())
            throw std::invalid_argument("projection group has invalid layers");
        for (const auto& layer : group.layers) check_projection(layer, input);
        return;
    }
    if (!group.nint_grouped || !input.is_cuda())
        throw std::invalid_argument("projection group is not prepared for input");
    for (size_t i = 0; i < group.outs.size(); ++i) {
        check_width(input.size(-1), group.cols[i], true);
        tyloi::weights::check_placement(group.placements[i], input.device());
    }
    const auto check_nint_weight = [&](const tyloi::NintTensor& weight) {
        if (!weight.q_packed.defined())
            throw std::invalid_argument("projection weight is not prepared");
        check_width(input.size(-1), weight.neuron_len, true);
        check_device(weight, input);
    };
    if (group.nint.split_w.empty()) check_nint_weight(group.nint.w);
    for (const auto& weight : group.nint.projection_w) check_nint_weight(weight);
    for (const auto& weight : group.nint.split_w) check_nint_weight(weight);
}

void check_device(const ::tyloi::DenseTensor& weight, const ::tyloi::DenseTensor& input) {
    weight.require_dense(); input.require_dense();
    if (input.is_cuda() && weight.defined() && weight.numel() != 0 && weight.device() != input.device())
        throw std::invalid_argument("projection weight and input devices do not match");
}

void check_device(const ::tyloi::TensorImpl& weight, const ::tyloi::DenseTensor& input) {
    check_fields(weight, input.device(), input.is_cuda());
}


#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "linear_dispatch.inc"

::tyloi::DenseTensor dequant_nint_dense_f32(const tyloi::NintTensor & w) {
    if (w.q8_zero) {
        return nint8_zero_dequant_f32_cuda(
            w.q_packed, w.q8_zero_scale, w.neuron_len).contiguous();
    }
    return nint_decode_f32_cuda(
        w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
        w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
        w.neuron_len, w.gs).contiguous();
}
#endif

tyloi::cuda::gguf::GgufPrefillProduct tyloi::cuda::mfq_prefill_product(
        const Tensor& weight, const DenseTensor& input, const DenseTensor& input_exponents) {
    gguf::GgufPrefillProduct product;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* nint = weight.representation<NintTensor>())
        scaled_prefill(*nint, input, &product, input_exponents);
    else if (const auto* nvq = weight.representation<NvqTensor>())
        scaled_prefill(*nvq, input, &product, input_exponents);
#else
    (void)weight;
    (void)input;
    (void)input_exponents;
#endif
    return product;
}

::tyloi::DenseTensor quant_embedding_lookup(
        const ::tyloi::Tensor & weight,
        ::tyloi::DenseTensor token_ids) {
    if (!(weight.representation<tyloi::ShardedTensor>() == nullptr))
        throw std::runtime_error("quantized embedding does not support tensor-parallel shards");
    if (const auto* dense = weight.representation<::tyloi::DenseTensor>()) {
        auto output_shape = token_ids.sizes().vec();
        output_shape.push_back(dense->size(1));
        return dense->index_select(0, token_ids.reshape({-1})).reshape(output_shape);
    }
    if (const auto* tq = weight.representation<tyloi::encoding::TqTensor>())
        return tyloi::cuda::tq::embedding(*tq, token_ids);
    if (const auto* fp8 = weight.representation<tyloi::ChannelFp8Tensor>()) {
        check_device(*fp8, token_ids);
        return tyloi::cuda::channel_fp8_rows(*fp8, std::move(token_ids));
    }
    if (const auto* gguf = weight.representation<tyloi::GgmlQuantizedTensor>())
        return tyloi::cuda::gguf::embedding(*gguf, std::move(token_ids));
    if (const auto* nvq = weight.representation<tyloi::NvqTensor>()) {
        if (!tyloi::mfq_compat_enabled)
            throw std::runtime_error("NVQ embedding weight is not enabled");
#if TYLOINFER_ENABLE_MFQ_COMPAT
        check_device(*nvq, token_ids);
        return nvq_embedding(*nvq, token_ids);
#endif
    }
    if (const auto* nint = weight.representation<tyloi::NintTensor>()) {
        if (!tyloi::mfq_compat_enabled)
            throw std::runtime_error("NINT embedding weight is not enabled");
#if TYLOINFER_ENABLE_MFQ_COMPAT
        check_device(*nint, token_ids);
        if (nint->q8_zero)
            return nint8_zero_embedding_lookup_cuda(
                nint->q_packed, nint->q8_zero_scale, token_ids, nint->neuron_len);
        return nint_embedding_cuda(
            nint->q_packed, nint->row_q_bits, nint->row_q_bit_offsets,
            nint->sub_scale, nint->sub_min, nint->neuron_scale, nint->neuron_min,
            token_ids, nint->neuron_len, nint->gs);
#endif
    }
    if (const auto* mx4 = weight.representation<tyloi::Mxfp4Tensor>()) {
        check_device(*mx4, token_ids);
        return mxfp4_embedding_lookup_cuda(mx4->values, mx4->scales, token_ids);
    }
    if (const auto* mx8 = weight.representation<tyloi::Mxfp8Tensor>()) {
        check_device(*mx8, token_ids);
        return mxfp8_embedding_lookup_cuda(mx8->values, mx8->scales, token_ids);
    }
    if (!(weight.representation<tyloi::Mxfp4SqTensor>() == nullptr &&
            weight.representation<tyloi::Fp8SqTensor>() == nullptr))
        throw std::runtime_error("SQ tensors do not support embedding lookup");
    throw std::runtime_error("unsupported quantized embedding weight");
}

bool tensor_parallel_output_projections_compatible(
        const std::vector<::tyloi::Tensor>& projections) {
    if (projections.size() < 2) return false;
    const auto* first = projections.front().representation<tyloi::ShardedTensor>();
    if (!first || first->axis != TensorParallelAxis::Output || first->shards.size() < 2)
        return false;
    for (const auto& projection : projections) {
        const auto* sharded = projection.representation<tyloi::ShardedTensor>();
        if (!sharded || sharded->axis != TensorParallelAxis::Output ||
            sharded->shards.size() != first->shards.size()) return false;
        for (size_t i = 0; i < first->shards.size(); ++i) {
            const auto& candidate = sharded->shards[i];
            const auto& reference = first->shards[i];
            if (candidate.device != reference.device ||
                candidate.input_begin != reference.input_begin ||
                candidate.input_end != reference.input_end) return false;
        }
    }
    return true;
}

::tyloi::DenseTensor quant_linear_reference_weight(
        const ::tyloi::Tensor& weight) {
    if (!(weight.representation<tyloi::ShardedTensor>() == nullptr))
        throw std::runtime_error("linear reference requires local weights");
    if (const auto* fp8 = weight.representation<tyloi::ChannelFp8Tensor>())
        return tyloi::cuda::dequant_channel_fp8(*fp8);
    if (const auto* gguf = weight.representation<tyloi::GgmlQuantizedTensor>())
        return tyloi::cuda::gguf::dequantize(*gguf);
    if (const auto* tq = weight.representation<tyloi::encoding::TqTensor>()) {
        const auto ids = ::tyloi::cuda::arange(tq->size(0),
            ::tyloi::TensorOptions{}.dtype(::tyloi::kInt64).device(tq->device()));
        return tyloi::cuda::tq::embedding(*tq, ids);
    }
    if (const auto* nint = weight.representation<tyloi::NintTensor>()) {
        if (!tyloi::mfq_compat_enabled)
            throw std::runtime_error("NINT linear weight is not enabled");
#if TYLOINFER_ENABLE_MFQ_COMPAT
        return nint->q8_zero
            ? nint8_zero_dequant_cuda(nint->q_packed, nint->q8_zero_scale, nint->neuron_len)
            : nint_decode_cuda(nint->q_packed, nint->row_q_bits, nint->row_q_bit_offsets,
                               nint->sub_scale, nint->sub_min, nint->neuron_scale,
                               nint->neuron_min, nint->neuron_len, nint->gs);
#endif
    }
    if (const auto* nvq = weight.representation<tyloi::NvqTensor>()) {
        if (!tyloi::mfq_compat_enabled)
            throw std::runtime_error("NVQ linear weight is not enabled");
#if TYLOINFER_ENABLE_MFQ_COMPAT
        return nvq_dequant(*nvq);
#endif
    }
    if (const auto* mx4 = weight.representation<tyloi::Mxfp4Tensor>())
        return mxfp4_dequant_cuda(mx4->values, mx4->scales);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* sq4 = weight.representation<tyloi::Mxfp4SqTensor>()) {
        if (!tyloi::mfq_compat_enabled)
            throw std::runtime_error("MXFP4-SQ linear weight is not enabled");
        if (!sq4->blob.is_cuda()) return dequant_mxfp4_sq_cpu(*sq4).to(::tyloi::kFloat16);
        return mxfp4_sq_dequant_cuda(
            sq4->blob, sq4->row_q, sq4->row_symbol_byte_offsets, sq4->row_auxiliary,
            sq4->bits, sq4->out, sq4->neuron_len, sq4->matrix_scale_base,
            sq4->q_sum, sq4->sq4_rows, false);
    }
#endif
    if (const auto* sq8 = weight.representation<tyloi::Fp8SqTensor>()) {
        if (!tyloi::mfq_compat_enabled)
            throw std::runtime_error("FP8-SQ linear weight is not enabled");
#if TYLOINFER_ENABLE_MFQ_COMPAT
        return dequant_fp8_sq(*sq8, false);
#endif
    }
    if (const auto* mx8 = weight.representation<tyloi::Mxfp8Tensor>())
        return mxfp8_cpu_reference(*mx8);
    if (const auto* dense = weight.representation<::tyloi::DenseTensor>()) return *dense;
    throw std::runtime_error("unsupported linear reference format");
}

::tyloi::DenseTensor prefill_linear_weight(const ::tyloi::Tensor& weight) {
    if (const auto* mx8 = weight.representation<tyloi::Mxfp8Tensor>())
        return mxfp8_dequant_cuda(mx8->values, mx8->scales);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* nint = weight.representation<tyloi::NintTensor>())
        return dequant_nint_dense_f32(*nint).to(tyloi::kFloat16).contiguous();
    if (const auto* sq4 = weight.representation<tyloi::Mxfp4SqTensor>())
        return mxfp4_sq_dequant_cuda(sq4->blob, sq4->row_q, sq4->row_symbol_byte_offsets,
            sq4->row_auxiliary, sq4->bits, sq4->out, sq4->neuron_len, sq4->matrix_scale_base,
            sq4->q_sum, sq4->sq4_rows, true).to(tyloi::kFloat16).contiguous();
    if (const auto* sq8 = weight.representation<tyloi::Fp8SqTensor>())
        return dequant_fp8_sq(*sq8, true).to(tyloi::kFloat16).contiguous();
#endif
    return quant_linear_reference_weight(weight).to(tyloi::kFloat16).contiguous();
}

::tyloi::DenseTensor dense_projection(
        ::tyloi::DenseTensor input, const ::tyloi::DenseTensor& weight) {
    const auto activation_type = input.scalar_type();
    // CUDA residual and sampling ops consume FP16 or FP32 activations.
    const auto activation_output = [activation_type](::tyloi::DenseTensor value) {
        return value.scalar_type() == tyloi::kBFloat16
            ? value.to(activation_type == tyloi::kFloat32 ? tyloi::kFloat32 : tyloi::kFloat16)
            : value;
    };
    if (auto out = try_dense_f32_projection_cuda(input, weight); out.defined()) return out;
    if (auto out = try_dense_bf16_projection_cuda(input, weight); out.defined()) return out;
    input = input.to(weight.scalar_type());
    if (activation_type != tyloi::kFloat32 && weight.scalar_type() == tyloi::kFloat32) {
        if (auto out = try_dense_f32_projection_cuda(input, weight); out.defined()) return out;
    }
    const int64_t rows = input.numel() / input.size(-1);
    if (rows > 1 && rows <= 6) {
        auto shape = input.sizes().vec();
        shape.back() = weight.size(0);
        auto flat = input.reshape({rows, input.size(-1)});
        std::vector<::tyloi::DenseTensor> outputs;
        outputs.reserve(rows);
        // A strided-batched GEMM may select different accumulation from M=1.
        // Keep short target/predictor windows on the decode projection path.
        for (int64_t row = 0; row < rows; ++row)
            outputs.push_back(dense_projection(flat.narrow(0, row, 1), weight));
        return activation_output(tyloi::cuda::cat(outputs, 0).reshape(shape));
    }
    return activation_output(tyloi::cuda::matmul(input, weight.transpose(0, 1)));
}

tyloi::weights::Placement projection_placement(const ::tyloi::Tensor& weight, std::size_t index) {
    if (index != 0) throw std::out_of_range("matrix index out of range");
    tyloi::weights::Placement result;
    if (const auto* sharded = weight.representation<tyloi::ShardedTensor>()) {
        for (const auto& shard : sharded->shards)
            result.devices.emplace_back(tyloi::DeviceType::cuda, shard.device);
    } else {
        result.devices.push_back(weight.device());
        if (weight.representation<tyloi::encoding::TqTensor>() && !weight.is_cuda())
            result.caller = tyloi::DeviceType::cuda;
    }
    return result;
}

std::vector<::tyloi::Tensor> QuantLinearGroup::tensor_parallel_output_projections() const {
    std::vector<::tyloi::Tensor> projections;
    projections.reserve(layers.size());
    for (const auto& layer : layers) projections.push_back(layer);
    return projections;
}

bool QuantLinearGroup::tensor_parallel_output_compatible() const {
    return tensor_parallel_output_projections_compatible(
        tensor_parallel_output_projections());
}
