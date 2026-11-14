#pragma once

#include "engine/models/session/snapshot.h"
#include "engine/state/session/cache.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <map>
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

namespace tyloi::models {

struct SessionOps {
    using Snapshot = TextSessionState;
    using Restore = tyloi::state::SessionRestore;
    using StateError = CudaSessionStateError;

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

    template <class State, class Predictor>
    static void reset_predictor(State& state, Predictor* mtp) {
        if (!mtp) return;
        if constexpr (requires { state.predictor; }) {
            if (!state.predictor)
                throw CudaSessionStateError("predictor state is unavailable");
            mtp->reset(*state.predictor, 1);
        } else {
            throw CudaSessionStateError("model state has no predictor state");
        }
    }

    static tyloi::Tensor restore(tyloi::models::State& state, const Snapshot& snapshot,
        tyloi::models::Mtp* mtp, tyloi::Executor& executor,
        tyloi::weights::Cache& weights, tyloi::engine::Budget& budget);

    template <class State, class Predictor>
    static tyloi::Tensor restore(
        State& state, const Snapshot& snapshot, Predictor* mtp,
        const std::function<void()>& drain) {
        // The caller owns the execution scope for the entire restore operation.
        if constexpr (requires { restore_session(state, mtp, snapshot, drain); }) {
            restore_session(state, mtp, snapshot, drain);
        } else {
            if constexpr (requires(decltype(state.prepare(snapshot))& candidate) {
                state.stage(candidate, snapshot);
                { state.publish(std::move(candidate)) } noexcept;
            }) {
                auto candidate = state.prepare(snapshot);
                try {
                    state.stage(candidate, snapshot);
                    drain();
                    state.publish(std::move(candidate));
                } catch (...) {
                    const auto error = std::current_exception();
                    try {
                        drain();
                    } catch (...) {
                    }
                    std::rethrow_exception(error);
                }
            } else if constexpr (requires { state.restore_text_session_state(snapshot); }) {
                state.restore_text_session_state(snapshot);
            } else {
                throw CudaSessionStateError("text session state is unsupported by this model");
            }
            if (mtp) {
                if constexpr (requires { state.predictor; })
                    mtp->restore_session_state(*state.predictor, *snapshot.mtp);
                else
                    throw CudaSessionStateError("model state has no predictor state");
            }
        }
        return mtp && snapshot.mtp ? snapshot.mtp->last_target_hidden : tyloi::Tensor{};
    }
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
