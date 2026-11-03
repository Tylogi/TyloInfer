#pragma once

#include <nlohmann/json.hpp>
#include <cstdint>
#include <optional>
#include <string>

namespace mfq { class ModelSource; }

namespace mfq::server {

// Context sizing math over the model's embedded config, mirroring the Python
// model_context and qsa_kv_policy modules so yarn info and KV offload checks
// answer with the same numbers.
struct YarnLimits {
    std::optional<std::int64_t> native, extended;
    std::optional<double> factor;
};
YarnLimits yarn_limits(const nlohmann::json& config);

struct QsaNeed {
    bool supported = false;
    std::string reason;
    std::int64_t required_index_bytes = 0;
    std::int64_t buffer_bytes = 0;
};
// context is the target context; budget is the offered offload budget when enabled.
QsaNeed qsa_requirement(const ModelSource& source, const nlohmann::json& config,
    std::int64_t context, const std::optional<std::int64_t>& budget,
    const std::optional<std::int64_t>& maximum_context, bool cuda);
} // namespace mfq::server
