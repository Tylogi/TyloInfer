#include "engine/models/qwen35/model.h"
#include "engine/ops.h"

#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen35 {

::tyloi::Tensor Model::embedding(const tyloi::Tensor& ids, int primary) const {
    if (ids.dim() != 1 && ids.dim() != 2)
        throw std::runtime_error("token IDs must have shape [tokens] or [batch,tokens]");
    auto shaped = ids.dim() == 1 ? tyloi::Tensor(tyloi::ops::unsqueeze(ids, 0)) : tyloi::Tensor{};
    const auto& input = shaped ? shaped : ids;
    if (input.size(0) <= 0 || input.size(1) <= 0)
        throw std::runtime_error("token IDs must not be empty");
    auto normalized = tyloi::ops::contiguous(tyloi::ops::copy(
        tyloi::ops::cast(input, tyloi::ScalarType::int64),
        tyloi::Device{tyloi::DeviceType::cuda, primary}));
    auto result = tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return tyloi::ops::embedding(normalized, embed);
    });
    return tyloi::ops::contiguous(tyloi::ops::cast(result, tyloi::ScalarType::float16));
}

::tyloi::Tensor Model::normalize(const tyloi::Tensor& hidden, std::int64_t batch, std::int64_t tokens) const {
    auto shaped = tyloi::ops::reshape(hidden, {batch, tokens, metadata.hidden_size});
    return tyloi::ops::measure("model.output_norm", shaped.device(), [&] {
        return tyloi::ops::rms_norm(shaped, output_norm, metadata.rms_norm_eps,
                                    metadata.norm_weight_offset);
    });
}

::tyloi::Tensor Model::softcap(const tyloi::Tensor& logits) const {
    return tyloi::ops::softcap(logits, metadata.final_logit_softcapping);
}

::tyloi::Tensor Model::logits(const tyloi::Tensor& hidden, bool capped) const {
    auto result = tyloi::ops::measure("model.lm_head", hidden.device(), [&] {
        return tyloi::ops::project(hidden, lm_head);
    });
    return capped ? softcap(result) : result;
}

} // namespace tyloi::cuda::qwen35
