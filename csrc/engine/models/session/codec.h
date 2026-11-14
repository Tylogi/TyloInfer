#pragma once

#include "engine/models/session/paged.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tyloi { class ModelSource; }
namespace tyloi::models {
std::string prefix_cache_compatibility_key(const tyloi::ModelSource&, int64_t,
    tyloi::Device, std::string_view kv_fingerprint);
}

size_t paged_decodable_blocks(const std::vector<SessionBlockPayload>&);
size_t paged_payload_tokens(const SessionBlockPayload&);
