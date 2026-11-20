#pragma once

#include "tyloinfer/hf_safetensors_source.h"
#include "nlohmann/json.hpp"

namespace mfq {

struct HfPackedQuantBinding {
    TensorMetadata tensor;
    std::vector<std::string> auxiliaries;
    ModelSource::TensorReader read;
};

std::string hf_packed_tensor_name(std::string name,const nlohmann::json& config);
std::optional<HfPackedQuantBinding> hf_packed_quant_binding(const HfSafetensorsSource& source,
    const nlohmann::json& config,const TensorMetadata& weight);
std::optional<HfPackedQuantBinding> hf_channel_fp8_binding(const HfSafetensorsSource& source,
    const nlohmann::json& config,const TensorMetadata& weight);

} // namespace mfq
