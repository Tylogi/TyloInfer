#pragma once

#include "mlx_sampling.h"
#include "tyloinfer/runtime.h"

#include <filesystem>
#include <memory>
#include <optional>

namespace mfq::metal {

class MlxQwen4CausalLm;
class MlxQwen35CausalLm;
struct MlxEngineComponents;

inline bool mlx_dflash2_sampling_supported(const MfqSamplingParams& sampling,
    const MfqTokenConstraintPtr& constraint) noexcept {
    return sampling.enable_mtp && sampling.max_tokens > 1 &&
        sampling.presence_penalty == 0 && sampling.frequency_penalty == 0 &&
        sampling.repetition_penalty == 1 && !constraint;
}
void configure_mlx_dflash2(MlxEngineComponents& components,
    std::shared_ptr<std::optional<MlxQwen4CausalLm>> runtime,
    const std::filesystem::path& directory, const MlxSamplingParams& sampling);
void configure_mlx_dflash2(MlxEngineComponents& components,
    std::shared_ptr<std::optional<MlxQwen35CausalLm>> runtime,
    const std::filesystem::path& directory, const MlxSamplingParams& sampling);

}
