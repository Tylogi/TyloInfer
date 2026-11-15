#pragma once

#include "engine/models/session/snapshot.h"
#include "engine/state/session/paged.h"
#include "engine/state/session/sessions.h"
#include "engine/state/session/snapshot.h"

#include <memory>

namespace tyloi::engine {

struct SessionCacheConfig;
struct PrefixCacheConfig;

class SessionCache final : public tyloi::state::Sessions {
public:
    explicit SessionCache(const SessionCacheConfig& session_config, const PrefixCacheConfig& prefix_config,
        std::shared_ptr<tyloi::cache::PagedPrefixCache> paged_cache = {}, bool supported = true,
        int disabled_reason = 0, bool native_hot_snapshots = false);

    bool persistent_prefix_enabled() const noexcept override;
    tyloi::state::SessionRestore restore_best(tyloi::models::State& state, tyloi::Executor& executor,
        tyloi::weights::Cache& weights, Budget& budget, tyloi::models::Mtp* mtp,
        const std::string& session, const std::vector<std::int64_t>& prompt,
        std::size_t maximum_prefix, const std::string& input_key = {}) override;
    void store(const std::string& session, TextSessionState state) override;
    SessionResult control(const SessionCommand& command) override;
    std::vector<std::pair<std::string, double>> metrics() const override;
    std::map<int, std::size_t> device_hot_bytes() const override;
    std::uint64_t trim_hot(std::uint64_t bytes) override;
    void reserve_ram(std::uint64_t bytes) override;

private:
    tyloi::state::SessionRestore restore_paged(tyloi::models::State& state,
        tyloi::Executor& executor, tyloi::weights::Cache& weights, Budget& budget,
        const std::string& requested_session, const std::vector<std::int64_t>& prompt,
        std::size_t maximum_prefix_tokens, tyloi::models::Mtp* mtp, const std::string& input_key);
    void store_paged(const std::string& session_id, const TextSessionState& state);
    std::size_t fork_session(const std::string& source, const std::string& target);
    std::size_t close_session(const std::string& session);
    std::size_t clear();
    std::uint64_t set_hot_limit(std::uint64_t bytes);
    std::uint64_t set_disk_limit(std::uint64_t bytes);
    std::uint64_t set_ram_limit(std::uint64_t bytes);
    std::uint64_t refresh_disk_index();

    SessionSnapshotCache<TextSessionState> snapshots_;
    std::shared_ptr<tyloi::cache::PagedPrefixCache> paged_cache_;
    PagedSessionBindings paged_bindings_;
    std::uint64_t paged_disk_budget_ = 0;
    std::uint64_t paged_hot_budget_ = 0;
    std::uint64_t paged_ram_budget_ = 0;
    bool trace_ = false;
    bool supported_ = true;
    int disabled_reason_ = 0;
    bool native_hot_snapshots_ = false;
    std::uint64_t anonymous_snapshots_ = 0;
};

} // namespace tyloi::engine
