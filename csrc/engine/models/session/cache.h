#pragma once

#include "engine/models/session/snapshot.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tyloi {
class ModelSource;
namespace cache {
class PagedPrefixCache;
}
}

namespace tyloi::engine {
struct PrefixCacheConfig;
struct SessionCacheConfig;
}
namespace tyloi::state { class Sessions; }

namespace tyloi::models {

struct SessionOps {
    using Snapshot = TextSessionState;

    static TextSessionState decode(
        const std::vector<SessionBlockPayload>& payloads,
        const std::vector<std::int64_t>& tokens,
        std::size_t block_size);
    static SessionBlockPayload encode(
        const Snapshot& state, std::size_t block_size, std::size_t index);
    static bool paged_mtp(const Snapshot& state);
    static bool partial_blocks(const Snapshot& state);
    static bool paged_media(const Snapshot& state);
    static bool checkpoint_needs_update(
        const SessionBlockPayload& payload, const Snapshot& state);
    static std::size_t decodable_blocks(
        const std::vector<SessionBlockPayload>& payloads);
    static std::size_t token_count(const SessionBlockPayload& payload);
    static void release_host_cache();
};

std::shared_ptr<tyloi::cache::PagedPrefixCache>
make_paged_prefix_cache(
    const tyloi::ModelSource& source,
    std::int64_t max_position_embeddings,
    bool supports_paged_text_session_state,
    const tyloi::engine::PrefixCacheConfig& config,
    tyloi::Device device, bool separate_memory, std::string_view kv_fingerprint);

std::unique_ptr<tyloi::state::Sessions> make_sessions(
    const tyloi::ModelSource& source,
    const tyloi::engine::SessionCacheConfig& session_config,
    const tyloi::engine::PrefixCacheConfig& prefix_config,
    std::int64_t maximum_context, bool supported, bool paged,
    tyloi::Device device, bool separate_memory, std::string_view kv_fingerprint);

} // namespace tyloi::models
