#include "mlx_tensor.h"
#include "mlx_kernel_prepare.h"
#include "mlx_moe_ops.h"
#include "engine/backends/metal/native/env.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/backends/metal/ops/tensor.h"
#include "engine/backends/metal/ops/projection/dense.h"
#include "engine/backends/metal/ops/rope/adjacent.h"
#include "engine/weights/memory.h"
#include "engine/backends/metal/ops/gguf/native/mlx_native_gguf.h"
#include "engine/backends/metal/ops/gguf/mlx_gguf_rotation.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tyloi::metal {
namespace {

using mlx::core::Dtype;
using mlx::core::Shape;
using mlx::core::array;

std::atomic_bool g_predequantize_fp16{false};
thread_local unsigned int g_fp16_weight_scopes = 0;

template <typename Variant>
array materialize_weight_fp16(
    const Variant& weight,
    int output_size) {
    const auto row_ids = mlx::core::arange(
        0,
        output_size,
        1,
        mlx::core::int32);
    auto dense = std::visit(
        [&](const auto& value) -> array {
            using Weight = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Weight, tyloi::Tensor>) {
                return tyloi::metal::array(
                    tyloi::metal::ops::cast(value, tyloi::ScalarType::float16));
            } else if constexpr (
                std::is_same_v<Weight, MlxMxWeight>
            ) {
                return value.dequantize(mlx::core::float16);
            } else {
                return value.embedding(
                    row_ids,
                    mlx::core::float16);
            }
        },
        weight);
    dense.eval();
    return dense;
}

template <typename Variant>
std::optional<array> unpack_quantized_weight(const Variant& weight, int output_size) {
    if (const auto* gguf = std::get_if<MlxGgufWeight>(&weight)) return gguf->dequantize();
    if (const auto* affine = std::get_if<MlxAffineWeight>(&weight)) return affine->dequantize();
    if (const auto* tq = std::get_if<MlxTqWeight>(&weight))
        return tq->embedding(mlx::core::arange(output_size, mlx::core::int32), mlx::core::float16);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* packed = std::get_if<MlxNintWeight>(&weight)) return packed->embedding(mlx::core::arange(output_size, mlx::core::int32), mlx::core::float16);
    if (const auto* packed = std::get_if<MlxNint8ZeroWeight>(&weight)) return packed->embedding(mlx::core::arange(output_size, mlx::core::int32), mlx::core::float16);
    if (const auto* packed = std::get_if<MlxVqWeight>(&weight)) return packed->embedding(mlx::core::arange(output_size, mlx::core::int32), mlx::core::float16);
    if (const auto* packed = std::get_if<MlxFp8SqWeight>(&weight)) return packed->embedding(mlx::core::arange(output_size, mlx::core::int32), mlx::core::float16);
    if (const auto* packed = std::get_if<MlxMxfp4SqWeight>(&weight)) return packed->embedding(mlx::core::arange(output_size, mlx::core::int32), mlx::core::float16);
#endif
    return std::nullopt;
}

array dense_reference_matmul(
    const array& dense,
    const array& input) {
    auto source = input.dtype() == mlx::core::float16
        ? input
        : mlx::core::astype(
              input,
              mlx::core::float16);
    return mlx::core::matmul(source, mlx::core::transpose(dense));
}


} // namespace

void set_mlx_predequantize_fp16(bool enabled) noexcept {
    g_predequantize_fp16.store(enabled, std::memory_order_relaxed);
}

bool mlx_predequantize_fp16_enabled() noexcept {
    return g_fp16_weight_scopes > 0 || g_predequantize_fp16.load(std::memory_order_relaxed);
}

MlxFp16WeightScope::MlxFp16WeightScope(bool enabled) noexcept : enabled_(enabled) {
    if (enabled_) ++g_fp16_weight_scopes;
}
MlxFp16WeightScope::~MlxFp16WeightScope() {
    if (enabled_) --g_fp16_weight_scopes;
}



#if TYLOINFER_ENABLE_MFQ_COMPAT
MlxLinear::MlxLinear(MlxNintWeight weight)
    : weight_(weight), input_size_(weight.input_size()), output_size_(weight.output_size()) {}
MlxLinear::MlxLinear(MlxNint8ZeroWeight weight)
    : weight_(weight), input_size_(weight.input_size()), output_size_(weight.output_size()) {}
MlxLinear::MlxLinear(MlxVqWeight weight)
    : weight_(weight), input_size_(weight.input_size()), output_size_(weight.output_size()) {}
MlxLinear::MlxLinear(MlxFp8SqWeight weight)
    : weight_(weight), input_size_(weight.input_size()), output_size_(weight.output_size()) {}
MlxLinear::MlxLinear(MlxMxfp4SqWeight weight)
    : weight_(weight), input_size_(weight.input_size()), output_size_(weight.output_size()) {}
#endif

namespace {
using StorageList = std::vector<std::shared_ptr<tyloi::TensorStorage>>;
void append_backings(StorageList& result, const tyloi::Tensor& value) {
    auto items = value.backings();
    result.insert(result.end(), items.begin(), items.end());
}
void append_backings(StorageList& result, const array& value) {
    auto items = tyloi::metal::backings(value);
    result.insert(result.end(), items.begin(), items.end());
}
}

std::vector<std::shared_ptr<tyloi::TensorStorage>> MlxLinear::backings() const {
    return std::visit([](const auto& weight) { return weight.backings(); }, weight_);
}
std::vector<std::shared_ptr<tyloi::TensorStorage>> MlxTqWeight::backings() const {
    StorageList result;
    append_backings(result, words_); append_backings(result, descriptors_); append_backings(result, scales_);
    return result;
}
std::vector<std::shared_ptr<tyloi::TensorStorage>> MlxGgufWeight::backings() const {
    StorageList result;
    append_backings(result, bytes_);
    if (planes_) for (const auto& plane : *planes_) append_backings(result, plane);
    if (native_) for (const auto& plane : native_->planes) append_backings(result, plane);
    if (mixed_layout_) append_backings(result, *mixed_layout_);
    if (rotation_) append_backings(result, rotation_->signs);
    return result;
}
std::vector<std::shared_ptr<tyloi::TensorStorage>> MlxAffineWeight::backings() const {
    StorageList result;
    append_backings(result, values_); append_backings(result, scales_); append_backings(result, biases_);
    if (native_) for (const auto& plane : native_->planes) append_backings(result, plane);
    return result;
}
#if TYLOINFER_ENABLE_MFQ_COMPAT
std::vector<std::shared_ptr<tyloi::TensorStorage>> MlxNintWeight::backings() const {
    StorageList result;
    for (const auto& item : {q_packed_, sub_scale_, sub_min_, row_metadata_}) append_backings(result, item);
    return result;
}
std::vector<std::shared_ptr<tyloi::TensorStorage>> MlxNint8ZeroWeight::backings() const {
    StorageList result;
    append_backings(result, q_); append_backings(result, scales_);
    return result;
}
std::vector<std::shared_ptr<tyloi::TensorStorage>> MlxVqWeight::backings() const {
    StorageList result;
    for (const auto& item : {indices_packed_, state_packed_, aux_packed_, anchors_, codebooks_,
            scale_lut_, state_to_codebank_, bank_ids_, rotation_signs_, parameters_,
            residual_codebook_, residual_first_, residual_second_}) append_backings(result, item);
    return result;
}
std::vector<std::shared_ptr<tyloi::TensorStorage>> MlxFp8SqWeight::backings() const {
    StorageList result;
    for (const auto& item : {blob_, row_q_, row_symbol_byte_offsets_, palette_values_, palette_quads_, palette_pairs_})
        append_backings(result, item);
    return result;
}
std::vector<std::shared_ptr<tyloi::TensorStorage>> MlxMxfp4SqWeight::backings() const {
    StorageList result;
    for (const auto& item : {blob_, row_q_, row_symbol_byte_offsets_, row_auxiliary_}) append_backings(result, item);
    return result;
}
#endif

MlxLinear::MlxLinear(MlxTqWeight weight)
    : weight_(weight), input_size_(weight.input_size()), output_size_(weight.output_size()) {}
MlxLinear::MlxLinear(MlxGgufWeight weight)
    : weight_(weight), input_size_(weight.input_size()), output_size_(weight.output_size()) {}
MlxLinear::MlxLinear(MlxAffineWeight weight)
    : weight_(weight), input_size_(weight.input_size()), output_size_(weight.output_size()) {
    std::get<MlxAffineWeight>(weight_).prepare_native();
}


MlxLinear::MlxLinear(MlxMxWeight weight)
    : input_size_(weight.input_size()),
      output_size_(weight.output_size()),
      weight_(std::move(weight)) {}


MlxLinear::MlxLinear(array weight)
    : weight_(tyloi::metal::tensor(std::move(weight))) {
    const auto& dense = std::get<tyloi::Tensor>(weight_);
    if (dense.dim() != 2) {
        throw std::runtime_error("dense linear weight must have rank two");
    }
    output_size_ = dense.size(0);
    input_size_ = dense.size(1);
}

std::optional<array> MlxLinear::greedy_argmax(
    const array& input) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* weight = std::get_if<MlxNintWeight>(&weight_)) return weight->greedy_argmax(input);
#endif
    return std::nullopt;
}

array MlxLinear::operator()(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size_)
        throw std::runtime_error("linear input width mismatch");
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* weight = std::get_if<MlxNintWeight>(&weight_)) return weight->matmul(input);
    if (const auto* weight = std::get_if<MlxNint8ZeroWeight>(&weight_)) return weight->matmul(input);
    if (const auto* weight = std::get_if<MlxVqWeight>(&weight_)) return weight->matmul(input);
    if (const auto* weight = std::get_if<MlxFp8SqWeight>(&weight_)) return weight->matmul(input);
    if (const auto* weight = std::get_if<MlxMxfp4SqWeight>(&weight_)) return weight->matmul(input);
#endif
    if (const auto* weight = std::get_if<MlxTqWeight>(&weight_)) return weight->matmul(input);
    if (const auto* weight = std::get_if<MlxGgufWeight>(&weight_)) return weight->matmul(input);
    if (const auto* weight = std::get_if<MlxAffineWeight>(&weight_)) return weight->matmul(input);
    if (mlx_reference_enabled())
        if (auto dense = unpack_quantized_weight(weight_, output_size_))
            return dense_reference_matmul(*dense, input);
    if (const auto* weight = std::get_if<MlxMxWeight>(&weight_)) {
        auto result = weight->matmul(input);
        return input.dtype() == mlx::core::bfloat16 && result.dtype() != input.dtype()
            ? mlx::core::astype(result, input.dtype()) : result;
    }
    return tyloi::metal::array(tyloi::metal::ops::dense_project(
        tyloi::metal::tensor(input), std::get<tyloi::Tensor>(weight_)));
}

array MlxLinear::moe_shared(
    const array& input,
    const array& routed_pairs,
    const array& route_weights,
    const array& gate_logits) const {
    return moe_weighted_reduce_shared_gate(
        routed_pairs,
        route_weights,
        (*this)(input),
        gate_logits);
}

void MlxLinear::prepare_grouped_row_matmul(int group_count) const {
    if (!tq_weight_ref() && !gguf_weight_ref()) return;
    if (auto* preparation = MlxKernelPreparation::current()) preparation->collect([&] {
        for (const auto rows : preparation->row_buckets())
            for (auto dtype : {mlx::core::float16, mlx::core::float32})
                preparation->add(grouped_row_matmul(
                    mlx::core::zeros(Shape{rows, group_count, input_size_}, dtype), group_count));
    });
}

array MlxLinear::grouped_row_matmul(
    const array& input,
    int group_count) const {
    if (group_count <= 0 ||
        input.ndim() < 2 ||
        input.shape(-2) != group_count ||
        input.shape(-1) != input_size_ ||
        output_size_ % group_count != 0) {
        throw std::runtime_error(
            "grouped-row linear shape/group mismatch");
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* weight = std::get_if<MlxNintWeight>(&weight_)) {
        if (auto output = weight->grouped_row_matmul(input, group_count)) return *output;
    }
    if (const auto* weight = std::get_if<MlxNint8ZeroWeight>(&weight_)) return weight->grouped_row_matmul(input, group_count);
#endif
    if (const auto* tq = std::get_if<MlxTqWeight>(&weight_)) return tq->grouped_row_matmul(input, group_count);
    if (const auto* gguf = std::get_if<MlxGgufWeight>(&weight_)) return gguf->grouped_row_matmul(input, group_count);
    if (mlx_reference_enabled()) {
        if (auto dense = unpack_quantized_weight(
                weight_, output_size_)) {
            auto source = input.dtype() == mlx::core::float16
                ? input
                : mlx::core::astype(
                      input,
                      mlx::core::float16);
            const auto grouped_weight = mlx::core::reshape(
                *dense,
                Shape{
                    group_count,
                    output_size_ / group_count,
                    input_size_,
                });
            return mlx::core::sum(
                mlx::core::expand_dims(source, -2)
                    * grouped_weight,
                -1);
        }
    }
    if (const auto* packed =
            std::get_if<MlxMxWeight>(&weight_);
        packed != nullptr && packed->bits() == 8) {
        return packed->grouped_row_matmul(
            input,
            group_count);
    }
    if (const auto* weight =
            std::get_if<tyloi::Tensor>(&weight_)) {
        const auto* dense = &tyloi::metal::array(*weight);
        auto source = input;
        if (source.dtype() != dense->dtype()) {
            source = mlx::core::astype(
                source,
                dense->dtype());
        }
        const std::array<std::int64_t, 3> shape{
            group_count, output_size_ / group_count, input_size_};
        const auto grouped = tyloi::metal::ops::reshape(*weight, shape);
        const auto& grouped_weight = tyloi::metal::array(grouped);
        return mlx::core::sum(
            mlx::core::expand_dims(source, -2) *
                grouped_weight,
            -1);
    }

    // The fallback intentionally stays on the original packed representation:
    // project each input group, then keep the output rows assigned to that
    // group. This keeps correctness for uncommon O-LoRA weight formats.
    const auto complete = (*this)(input);
    const int output_per_group =
        output_size_ / group_count;
    std::vector<array> pieces;
    pieces.reserve(
        static_cast<std::size_t>(group_count));
    for (int group = 0; group < group_count; ++group) {
        auto selected = mlx::core::take(
            complete,
            group,
            complete.ndim() - 2);
        Shape starts(
            static_cast<std::size_t>(selected.ndim()),
            0);
        Shape stops = selected.shape();
        starts.back() = group * output_per_group;
        stops.back() = (group + 1) * output_per_group;
        pieces.push_back(
            mlx::core::slice(
                selected,
                starts,
                stops));
    }
    return mlx::core::stack(pieces, input.ndim() - 2);
}

array MlxLinear::grouped_row_matmul_inverse_rope(
    const array& value,
    int group_count,
    const array& cosine,
    const array& sine,
    int head_dimension,
    int rotary_dimension) const {
    if (value.ndim() != 4 ||
        group_count <= 0 ||
        head_dimension <= 0 ||
        rotary_dimension <= 0 ||
        rotary_dimension > head_dimension ||
        rotary_dimension % 2 != 0 ||
        value.shape(-1) != head_dimension ||
        value.shape(-2) * head_dimension != group_count * input_size_) {
        throw std::runtime_error(
            "inverse-RoPE grouped-row linear shape/group mismatch");
    }
    const int batch = value.shape(0);
    const int tokens = value.shape(1);
    const auto grouped_shape = Shape{
        batch,
        tokens,
        group_count,
        input_size_,
    };
    auto grouped = mlx::core::reshape(value, grouped_shape);

    if (const auto* packed = std::get_if<MlxMxWeight>(&weight_)) {
        const auto rows = static_cast<std::int64_t>(batch) * tokens;
        const bool block32_eligible =
            packed->scale_column_block_size() == 32 &&
            rows == 1 &&
            (value.dtype() == mlx::core::float32 || value.dtype() == mlx::core::float16);
        const bool block128_eligible =
            packed->scale_column_block_size() == 128 &&
            rows >= 1 && rows <= 6;
        if (packed->bits() == 8 &&
            cosine.ndim() == 2 &&
            cosine.shape(0) == rows &&
            (block32_eligible || block128_eligible)) {
            return packed->grouped_row_matmul_inverse_rope(
                grouped,
                group_count,
                cosine,
                sine,
                head_dimension,
                rotary_dimension);
        }
    }

    // Adjacent RoPE treats rotary_dimension as a suffix width, matching
    // DeepSeek's decoupled-RoPE layout while preserving the leading content
    // channels in the same kernel dispatch.
    auto unrotated = tyloi::metal::ops::adjacent_rope(
        value,
        rotary_dimension,
        cosine,
        sine,
        true);
    return grouped_row_matmul(
        mlx::core::reshape(std::move(unrotated), grouped_shape),
        group_count);
}

std::optional<MlxGroupedLinearWeightRef>
MlxLinear::grouped_weight_ref() const noexcept {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* weight = std::get_if<MlxNintWeight>(&weight_)) return MlxGroupedLinearWeightRef{weight};
    if (const auto* weight = std::get_if<MlxNint8ZeroWeight>(&weight_)) return MlxGroupedLinearWeightRef{weight};
    if (const auto* weight = std::get_if<MlxVqWeight>(&weight_)) return MlxGroupedLinearWeightRef{weight};
    if (const auto* weight = std::get_if<MlxFp8SqWeight>(&weight_)) return MlxGroupedLinearWeightRef{weight};
    if (const auto* weight = std::get_if<MlxMxfp4SqWeight>(&weight_)) return MlxGroupedLinearWeightRef{weight};
#endif
    if (mlx_reference_enabled()) {
        if (const auto* packed =
                std::get_if<MlxMxWeight>(&weight_)) {
            if (packed->bits() == 8 &&
                (packed->scale_row_block_size() != 128 ||
                 packed->scale_column_block_size() != 128)) {
                return std::nullopt;
            }
            return MlxGroupedLinearWeightRef{packed};
        }
        return std::nullopt;
    }
    if (const auto* packed =
            std::get_if<MlxMxWeight>(&weight_)) {
        return MlxGroupedLinearWeightRef{packed};
    }
    if (const auto* dense =
            std::get_if<tyloi::Tensor>(&weight_)) {
        return MlxGroupedLinearWeightRef{&tyloi::metal::array(*dense)};
    }
    return std::nullopt;
}


const MlxMxWeight* MlxLinear::mx_weight_ref() const noexcept {
    return std::get_if<MlxMxWeight>(&weight_);
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
const MlxNintWeight* MlxLinear::nint_weight_ref() const noexcept {
    return std::get_if<MlxNintWeight>(&weight_);
}
#endif

const MlxTqWeight* MlxLinear::tq_weight_ref() const noexcept {
    return std::get_if<MlxTqWeight>(&weight_);
}

const MlxAffineWeight* MlxLinear::affine_weight_ref() const noexcept {
    return std::get_if<MlxAffineWeight>(&weight_);
}

const MlxGgufWeight* MlxLinear::gguf_weight_ref() const noexcept {
    return std::get_if<MlxGgufWeight>(&weight_);
}

void MlxLinear::prepare_grouped_gguf() {
    if (auto* weight = std::get_if<MlxGgufWeight>(&weight_)) weight->prepare_grouped();
}

const array* MlxLinear::dense_weight_ref() const noexcept {
    const auto* dense = std::get_if<tyloi::Tensor>(&weight_);
    return dense ? &tyloi::metal::array(*dense) : nullptr;
}

void MlxLinear::materialize_fp16() {
    if (std::holds_alternative<MlxAffineWeight>(weight_)) return;
    if (std::holds_alternative<MlxGgufWeight>(weight_)) return;
    if (std::holds_alternative<MlxTqWeight>(weight_)) return;
    if (const auto* dense = std::get_if<tyloi::Tensor>(&weight_);
        dense && (!g_fp16_weight_scopes || tyloi::metal::array(*dense).dtype() == mlx::core::float16)) return;
    auto dense = materialize_weight_fp16(weight_, output_size_);
    weight_ = tyloi::metal::tensor(std::move(dense));
    tyloi::weights::observe_weight(std::get<tyloi::Tensor>(weight_));
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
MlxEmbedding::MlxEmbedding(MlxNintWeight weight)
    : weight_(weight), vocabulary_size_(weight.output_size()), hidden_size_(weight.input_size()) {}
MlxEmbedding::MlxEmbedding(MlxNint8ZeroWeight weight)
    : weight_(weight), vocabulary_size_(weight.output_size()), hidden_size_(weight.input_size()) {}
MlxEmbedding::MlxEmbedding(MlxVqWeight weight)
    : weight_(weight), vocabulary_size_(weight.output_size()), hidden_size_(weight.input_size()) {}
MlxEmbedding::MlxEmbedding(MlxFp8SqWeight weight)
    : weight_(weight), vocabulary_size_(weight.output_size()), hidden_size_(weight.input_size()) {}
MlxEmbedding::MlxEmbedding(MlxMxfp4SqWeight weight)
    : weight_(weight), vocabulary_size_(weight.output_size()), hidden_size_(weight.input_size()) {}
#endif

MlxEmbedding::MlxEmbedding(MlxTqWeight weight)
    : weight_(weight), vocabulary_size_(weight.output_size()), hidden_size_(weight.input_size()) {}
MlxEmbedding::MlxEmbedding(MlxGgufWeight weight)
    : weight_(weight), vocabulary_size_(weight.output_size()), hidden_size_(weight.input_size()) {}
MlxEmbedding::MlxEmbedding(MlxAffineWeight weight)
    : weight_(weight), vocabulary_size_(weight.output_size()), hidden_size_(weight.input_size()) {}


MlxEmbedding::MlxEmbedding(MlxMxWeight weight)
    : vocabulary_size_(weight.output_size()),
      hidden_size_(weight.input_size()),
      weight_(std::move(weight)) {}

MlxEmbedding::MlxEmbedding(array weight)
    : weight_(tyloi::metal::tensor(std::move(weight))) {
    const auto& dense = std::get<tyloi::Tensor>(weight_);
    if (dense.dim() != 2) {
        throw std::runtime_error("dense embedding weight must have rank two");
    }
    vocabulary_size_ = dense.size(0);
    hidden_size_ = dense.size(1);
}

void MlxEmbedding::materialize_fp16() {
    if (std::holds_alternative<MlxAffineWeight>(weight_)) return;
    if (std::holds_alternative<MlxGgufWeight>(weight_)) return;
    if (std::holds_alternative<MlxTqWeight>(weight_)) return;
    if (const auto* dense = std::get_if<tyloi::Tensor>(&weight_);
        dense && (!g_fp16_weight_scopes || tyloi::metal::array(*dense).dtype() == mlx::core::float16)) return;
    auto dense = materialize_weight_fp16(weight_, vocabulary_size_);
    weight_ = tyloi::metal::tensor(std::move(dense));
    tyloi::weights::observe_weight(std::get<tyloi::Tensor>(weight_));
}

Dtype MlxEmbedding::preferred_activation_dtype() const noexcept {
    if (activation_dtype_) return *activation_dtype_;
    if (const auto* weight = std::get_if<MlxAffineWeight>(&weight_)) return weight->activation_dtype();
    return mlx::core::float16;
}

void MlxEmbedding::set_activation_dtype(Dtype dtype) {
    if (dtype != mlx::core::bfloat16 && dtype != mlx::core::float16 && dtype != mlx::core::float32)
        throw std::invalid_argument("embedding activation precision must be floating point");
    activation_dtype_ = dtype;
}

array MlxEmbedding::operator()(
    const array& token_ids,
    Dtype dtype) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* weight = std::get_if<MlxNintWeight>(&weight_)) return weight->embedding(token_ids, dtype);
    if (const auto* weight = std::get_if<MlxNint8ZeroWeight>(&weight_)) return weight->embedding(token_ids, dtype);
    if (const auto* weight = std::get_if<MlxVqWeight>(&weight_)) return weight->embedding(token_ids, dtype);
    if (const auto* weight = std::get_if<MlxFp8SqWeight>(&weight_)) return weight->embedding(token_ids, dtype);
    if (const auto* weight = std::get_if<MlxMxfp4SqWeight>(&weight_)) return weight->embedding(token_ids, dtype);
#endif
    if (const auto* tq = std::get_if<MlxTqWeight>(&weight_)) return tq->embedding(token_ids, dtype);
    if (const auto* gguf = std::get_if<MlxGgufWeight>(&weight_)) return gguf->embedding(token_ids, dtype);
    if (const auto* affine = std::get_if<MlxAffineWeight>(&weight_)) return affine->embedding(token_ids, dtype);
    if (const auto* packed = std::get_if<MlxMxWeight>(&weight_)) {
        auto result = packed->embedding(
            token_ids,
            dtype == mlx::core::bfloat16
                ? mlx::core::float16
                : dtype);
        return result.dtype() == dtype
            ? result
            : mlx::core::astype(result, dtype);
    }
    auto ids = token_ids;
    if (ids.dtype() != mlx::core::int32 && ids.dtype() != mlx::core::int64) {
        ids = mlx::core::astype(ids, mlx::core::int32);
    }
    auto result = tyloi::metal::array(tyloi::metal::ops::dense_embedding(
        tyloi::metal::tensor(std::move(ids)), std::get<tyloi::Tensor>(weight_)));
    return result.dtype() == dtype
        ? result
        : mlx::core::astype(result, dtype);
}

array MlxEmbedding::project(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != hidden_size_) {
        throw std::runtime_error(
            "embedding projection input width mismatch");
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* weight = std::get_if<MlxNintWeight>(&weight_)) return weight->matmul(input);
    if (const auto* weight = std::get_if<MlxNint8ZeroWeight>(&weight_)) return weight->matmul(input);
    if (const auto* weight = std::get_if<MlxVqWeight>(&weight_)) return weight->matmul(input);
    if (const auto* weight = std::get_if<MlxFp8SqWeight>(&weight_)) return weight->matmul(input);
    if (const auto* weight = std::get_if<MlxMxfp4SqWeight>(&weight_)) return weight->matmul(input);
#endif
    if (const auto* tq = std::get_if<MlxTqWeight>(&weight_)) return tq->matmul(input);
    if (const auto* gguf = std::get_if<MlxGgufWeight>(&weight_)) return gguf->matmul(input);
    if (const auto* affine = std::get_if<MlxAffineWeight>(&weight_)) return affine->matmul(input);
    if (mlx_reference_enabled()) {
        if (auto dense = unpack_quantized_weight(
                weight_, vocabulary_size_)) {
            return dense_reference_matmul(*dense, input);
        }
    }
    if (const auto* packed = std::get_if<MlxMxWeight>(&weight_)) {
        return packed->matmul(input);
    }
    return tyloi::metal::array(tyloi::metal::ops::dense_project(
        tyloi::metal::tensor(input), std::get<tyloi::Tensor>(weight_)));
}

} // namespace tyloi::metal
