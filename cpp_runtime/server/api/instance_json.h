#pragma once
#include "instance.h"
#include "models/instances.h"
#include "protocol.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <nlohmann/json.hpp>

namespace mfq::server {
// The wire shape of one runtime instance; the load job result and runtime observation share it.
inline nlohmann::json instance_json(const InstanceInfo& item) {
    using json = nlohmann::json;
    const auto state = item.state == InstanceInfo::State::loaded ? "loaded" :
        item.state == InstanceInfo::State::unloading ? "unloading" : "failed";
    return {{"instance_id", item.id}, {"model", item.model}, {"artifact_id", item.artifact_id}, {"backend", item.backend},
        {"state", state}, {"context_size", item.capabilities.max_context}, {"max_requests", item.capabilities.max_requests},
        {"active_requests", item.execution.requests}, {"input_bytes", item.execution.input_bytes},
        {"capabilities", mfq::protocol::model_capability_profile_json(item.capabilities.capabilities)},
        {"policy", {{"pin", item.policy.pin}, {"idle_ttl_seconds", item.policy.idle_ttl ? json(item.policy.idle_ttl->count()) : json(nullptr)}}},
        {"metrics_age_ms", std::max<std::int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(
            engine::Clock::now() - item.execution.captured).count())},
        {"metrics", item.execution.runtime}, {"session_metrics", item.execution.session}};
}
}
