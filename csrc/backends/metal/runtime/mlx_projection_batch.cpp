#include "mlx_projection_batch.h"
#include "mlx_kernel_prepare.h"
#include "mlx_transformer.h"
#include "mlx_moe_ops.h"
#include "mlx_gguf_staged.h"
#include "engine/backends/metal/ops/gguf/native/mlx_native_gguf.h"

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace tyloi::metal {
using mlx::core::array;
using mlx::core::Dtype;
using mlx::core::Shape;

std::optional<MlxGroupedLinear> mlx_group_linears(
    std::span<const MlxLinear* const> linears) {
    if (linears.size() < 2) return std::nullopt;
    std::vector<MlxGroupedLinearWeightRef> references;
    references.reserve(linears.size());
    for (const auto* linear : linears) {
        if (linear == nullptr) return std::nullopt;
        auto reference = linear->grouped_weight_ref();
        if (!reference) return std::nullopt;
        references.push_back(*reference);
    }
    try {
        return MlxGroupedLinear(std::move(references));
    } catch (const MlxGroupedLinearUnsupported&) {
        return std::nullopt;
    }
}

struct MlxProjectionBatch::Impl {
    struct Segment {
        std::vector<std::size_t> indices;
        std::optional<MlxGroupedLinear> grouped;
        std::vector<MlxTqWeight> tq;
        std::vector<MlxGgufWeight> gguf;
        std::vector<MlxNativeGgufWeight> native;
    };

    explicit Impl(std::vector<const MlxLinear*> sources) {
        if (sources.empty() || std::any_of(
                sources.begin(),
                sources.end(),
                [](const MlxLinear* linear) {
                    return linear == nullptr;
                })) {
            throw std::invalid_argument(
                "projection batch requires non-null projections");
        }
        linears.reserve(sources.size());
        for (const auto* source : sources) {
            linears.push_back(*source);
        }
        bfloat_input = std::all_of(linears.begin(), linears.end(), [](const auto& linear) {
            if (const auto* weight = linear.gguf_weight_ref())
                return !weight->has_hadamard() && weight->native_weight_ref();
            const auto* dense = linear.dense_weight_ref();
            return dense && dense->dtype() == mlx::core::bfloat16;
        });

        std::vector<const MlxLinear*> references;
        references.reserve(linears.size());
        for (const auto& linear : linears) {
            references.push_back(&linear);
        }

        std::size_t begin = 0;
        while (begin < references.size()) {
            const std::size_t remaining = references.size() - begin;
            if (const auto* first = references[begin]->gguf_weight_ref(); first && first->native_weight_ref()) {
                Segment segment;
                for (std::size_t offset = 0; offset < std::min(remaining, std::size_t(3)); ++offset) {
                    const auto* weight = references[begin + offset]->gguf_weight_ref();
                    if (!weight || !weight->native_weight_ref() || weight->input_size() != first->input_size()) break;
                    segment.indices.push_back(begin + offset); segment.native.push_back(*weight->native_weight_ref());
                }
                if (segment.native.size() >= 2) {
                    grouped_projection_count += segment.native.size(); begin += segment.native.size();
                    segments.push_back(std::move(segment)); continue;
                }
            }
            if (const auto* first = references[begin]->affine_weight_ref(); first && first->native_weight_ref()) {
                Segment segment;
                for (std::size_t offset = 0; offset < std::min(remaining, std::size_t(3)); ++offset) {
                    const auto* weight = references[begin + offset]->affine_weight_ref();
                    if (!weight || !weight->native_weight_ref() || weight->input_size() != first->input_size()) break;
                    segment.indices.push_back(begin + offset); segment.native.push_back(*weight->native_weight_ref());
                }
                if (segment.native.size() >= 2) {
                    grouped_projection_count += segment.native.size(); begin += segment.native.size();
                    segments.push_back(std::move(segment)); continue;
                }
            }
            if (const auto* first = references[begin]->gguf_weight_ref(); first && first->has_staged_planes()) {
                Segment segment;
                for (std::size_t offset = 0; offset < std::min(remaining, std::size_t(4)); ++offset) {
                    const auto* weight = references[begin + offset]->gguf_weight_ref();
                    if (!weight || !weight->has_staged_planes() || weight->input_size() != first->input_size()) break;
                    segment.indices.push_back(begin + offset);
                    segment.gguf.push_back(*weight);
                }
                if (segment.gguf.size() >= 2) {
                    grouped_projection_count += segment.gguf.size();
                    begin += segment.gguf.size();
                    segments.push_back(std::move(segment));
                    continue;
                }
            }
            if (const auto* first = references[begin]->tq_weight_ref()) {
                Segment segment;
                for (std::size_t offset = 0; offset < std::min(remaining, std::size_t(8)); ++offset) {
                    const auto* weight = references[begin + offset]->tq_weight_ref();
                    if (!weight || weight->input_size() != first->input_size()) break;
                    segment.indices.push_back(begin + offset);
                    segment.tq.push_back(*weight);
                }
                if (segment.tq.size() >= 2) {
                    grouped_projection_count += segment.tq.size();
                    begin += segment.tq.size();
                    segments.push_back(std::move(segment));
                    continue;
                }
            }
            std::optional<MlxGroupedLinear> selected;
            std::size_t selected_count = 0;
            for (std::size_t count = remaining; count >= 2; --count) {
                // Avoid leaving a lone compatible projection when two
                // balanced groups can cover the same four-member suffix.
                if (remaining - count == 1 && count > 2) {
                    continue;
                }
                auto candidate = mlx_group_linears(
                    std::span<const MlxLinear* const>(
                        references.data() + begin,
                        count));
                if (candidate && candidate->has_projection_fusion()) {
                    selected = std::move(candidate);
                    selected_count = count;
                    break;
                }
            }

            const std::size_t count = selected ? selected_count : 1;
            std::vector<std::size_t> indices;
            indices.reserve(count);
            for (std::size_t offset = 0; offset < count; ++offset)
                indices.push_back(begin + offset);
            segments.push_back(Segment{
                .indices = std::move(indices),
                .grouped = std::move(selected),
            });
            if (segments.back().grouped) {
                grouped_projection_count += count;
            }
            begin += count;
        }
        for (std::size_t first = 0; first < segments.size(); ++first) {
            if (segments[first].grouped || !segments[first].tq.empty() || !segments[first].gguf.empty() || !segments[first].native.empty()) continue;
            for (std::size_t second = first + 1; second < segments.size(); ++second) {
                if (segments[second].grouped || !segments[second].tq.empty() || !segments[second].gguf.empty() || !segments[second].native.empty()) continue;
                const std::array<const MlxLinear*, 2> pair{
                    references[segments[first].indices.front()],
                    references[segments[second].indices.front()]};
                auto candidate = mlx_group_linears(pair);
                if (!candidate || !candidate->has_projection_fusion()) continue;
                segments[first].indices.push_back(segments[second].indices.front());
                segments[first].grouped = std::move(candidate);
                grouped_projection_count += 2;
                segments.erase(segments.begin() + second);
                break;
            }
        }
    }

    std::vector<MlxLinear> linears;
    std::vector<Segment> segments;
    std::size_t grouped_projection_count = 0;
    bool bfloat_input = false;
};

MlxProjectionBatch::MlxProjectionBatch(
    std::vector<const MlxLinear*> linears)
    : impl_(std::make_shared<Impl>(std::move(linears))) {
    if (auto* preparation = MlxKernelPreparation::current(); preparation && !preparation->model_paths()) {
        preparation->collect([&] {
            for (const auto rows : preparation->row_buckets()) {
                for (const auto dtype : {mlx::core::float16, mlx::core::float32}) {
                    const auto input = mlx::core::zeros(
                        Shape{rows, impl_->linears.front().input_size()}, dtype);
                    for (const auto& output : (*this)(input)) preparation->add(output);
                    if (impl_->linears.size() == 2 &&
                        impl_->linears[0].output_size() == impl_->linears[1].output_size())
                        preparation->add(swiglu(input));
                }
            }
        });
    }
}

std::vector<array> MlxProjectionBatch::operator()(
    const array& input) const {
    return project(input, std::nullopt);
}

std::vector<array> MlxProjectionBatch::forward_normalized(
    const array& input, const MlxRmsNorm& norm) const {
    const bool bfloat_consumers = input.dtype() == mlx::core::float16 && input.ndim() >= 2 &&
        input.shape(-1) == norm.width() && input.size() / std::size_t(norm.width()) <= 8 &&
        impl_->bfloat_input;
    if (bfloat_consumers) return project(norm(input, mlx::core::bfloat16), input.dtype());
    return (*this)(norm(input));
}

std::vector<array> MlxProjectionBatch::project(
    const array& input, std::optional<Dtype> native_output_dtype) const {
    const std::size_t rows =
        input.ndim() == 0 || input.shape(-1) <= 0
        ? 0
        : input.size() /
            static_cast<std::size_t>(input.shape(-1));
    std::vector<array> outputs(impl_->linears.size(), input);
    for (const auto& segment : impl_->segments) {
        const bool affine_vector = rows == 1 && std::all_of(segment.indices.begin(), segment.indices.end(),
            [&](const auto index) { return impl_->linears[index].affine_weight_ref() != nullptr; });
        if (!segment.native.empty() && !affine_vector) {
            auto values = mlx_native_gguf_project(segment.native, input, native_output_dtype);
            for (std::size_t offset = 0; offset < segment.indices.size(); ++offset)
                outputs[segment.indices[offset]] = std::move(values[offset]);
            continue;
        }
        if (!segment.gguf.empty() && rows >= 1 && rows <= 8) {
            auto values = MlxGgufStagedPrefill::project_many(segment.gguf, input);
            for (std::size_t offset = 0; offset < segment.indices.size(); ++offset)
                outputs[segment.indices[offset]] = std::move(values[offset]);
            continue;
        }
        if (!segment.tq.empty()) {
            auto values = MlxTqWeight::matmul_many(segment.tq, input);
            for (std::size_t offset = 0; offset < segment.indices.size(); ++offset)
                outputs[segment.indices[offset]] = std::move(values[offset]);
            continue;
        }
        const bool use_grouped = segment.grouped &&
            segment.grouped->supports(input) &&
            (rows > 1 ||
             segment.grouped
                 ->supports_single_row_projection_fusion());
        if (use_grouped) {
            auto values = segment.grouped->matmul(input);
            for (std::size_t offset = 0; offset < segment.indices.size(); ++offset)
                outputs[segment.indices[offset]] = std::move(values[offset]);
            continue;
        }
        for (const auto index : segment.indices) {
            if (native_output_dtype) {
                const auto* weight = impl_->linears[index].gguf_weight_ref();
                if (weight && weight->native_weight_ref()) {
                    outputs[index] = mlx_native_gguf_project(
                        std::span(weight->native_weight_ref(), 1), input, native_output_dtype).front();
                    continue;
                }
            }
            outputs[index] = impl_->linears[index](input);
        }
    }
    return outputs;
}

array MlxProjectionBatch::swiglu(
    const array& input,
    float limit) const {
    if (impl_->linears.size() != 2 ||
        impl_->linears[0].output_size() !=
            impl_->linears[1].output_size()) {
        throw std::logic_error(
            "SwiGLU projection batch requires two equal-width outputs");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "SwiGLU projection limit must be finite and non-negative");
    }
    if (const auto* gate = impl_->linears[0].tq_weight_ref()) {
        if (const auto* up = impl_->linears[1].tq_weight_ref()) {
            const int tokens = input.size() / gate->input_size();
            static const auto single_expert = array({0}, Shape{1, 1}, mlx::core::int32);
            const auto ids = tokens == 1 ? single_expert : mlx::core::zeros(Shape{tokens, 1}, mlx::core::int32);
            auto result = gate->swiglu(*up, input, ids, limit);
            auto shape = input.shape(); shape.back() = gate->output_size();
            return mlx::core::reshape(result, shape);
        }
    }
    if (limit == 0.0f) {
        if (const auto* gate = impl_->linears[0].gguf_weight_ref(); gate && gate->native_weight_ref()) {
            if (const auto* up = impl_->linears[1].gguf_weight_ref(); up && up->native_weight_ref())
                return mlx_native_gguf_swiglu(*gate->native_weight_ref(), *up->native_weight_ref(), input);
        }
        if (const auto* gate = impl_->linears[0].affine_weight_ref(); gate && gate->native_weight_ref()) {
            if (const auto* up = impl_->linears[1].affine_weight_ref(); up && up->native_weight_ref())
                return mlx_native_gguf_swiglu(*gate->native_weight_ref(), *up->native_weight_ref(), input);
        }
        if (const auto* gate = impl_->linears[0].gguf_weight_ref()) {
            if (const auto* up = impl_->linears[1].gguf_weight_ref()) return gate->swiglu(*up, input);
        }
    }
    if (impl_->segments.size() == 1 &&
        impl_->segments.front().indices.size() == 2 &&
        impl_->segments.front().grouped) {
        const auto& grouped = *impl_->segments.front().grouped;
        if (grouped.supports_single_row_swiglu(input)) {
            return grouped.single_row_swiglu(input, limit);
        }
        if (grouped.supports_small_m_swiglu(input)) {
            return grouped.small_m_swiglu(input, limit);
        }
    }
    auto projected = (*this)(input);
    if (limit > 0.0f) {
        return moe_limited_swiglu_pair(
            projected.at(0), projected.at(1), limit);
    }
    auto result = moe_swiglu_pair(projected.at(0), projected.at(1));
    return projected.at(0).dtype() == mlx::core::bfloat16
        ? mlx::core::astype(result, mlx::core::bfloat16) : result;
}

bool MlxProjectionBatch::supports_fused_swiglu(
    const array& input) const noexcept {
    if (impl_->linears.size() == 2 && impl_->segments.size() == 1 &&
        impl_->segments.front().native.size() == 2 && input.ndim() > 1 &&
        impl_->linears[0].output_size() == impl_->linears[1].output_size() &&
        input.shape(-1) == impl_->linears[0].input_size()) return true;
    if (impl_->linears.size() == 2 && impl_->linears[0].tq_weight_ref()
        && impl_->linears[1].tq_weight_ref()) return true;
    if (impl_->linears.size() == 2 && impl_->linears[0].gguf_weight_ref() && impl_->linears[1].gguf_weight_ref()
        && impl_->linears[0].output_size() == impl_->linears[1].output_size() && input.ndim() > 0
        && input.shape(-1) == impl_->linears[0].input_size()
        && input.size() / impl_->linears[0].input_size() <= 6) return true;
    if (impl_->linears.size() != 2 ||
        impl_->segments.size() != 1 ||
        impl_->segments.front().indices.size() != 2 ||
        !impl_->segments.front().grouped) {
        return false;
    }
    const auto& grouped = *impl_->segments.front().grouped;
    return grouped.supports_single_row_swiglu(input) ||
        grouped.supports_small_m_swiglu(input);
}

std::size_t MlxProjectionBatch::projection_count() const noexcept {
    return impl_->linears.size();
}

std::size_t
MlxProjectionBatch::grouped_projection_count() const noexcept {
    return impl_->grouped_projection_count;
}

bool MlxProjectionBatch::projections_share_group(
    std::size_t begin,
    std::size_t count) const noexcept {
    if (count < 2 || begin > impl_->linears.size() ||
        count > impl_->linears.size() - begin) {
        return false;
    }
    return std::any_of(
        impl_->segments.begin(),
        impl_->segments.end(),
        [begin, count](const Impl::Segment& segment) {
            if (!segment.grouped && segment.tq.empty() && segment.gguf.empty() && segment.native.empty()) return false;
            for (std::size_t index = begin; index < begin + count; ++index)
                if (std::find(segment.indices.begin(), segment.indices.end(), index)
                    == segment.indices.end()) return false;
            return true;
        });
}




} // namespace tyloi::metal
