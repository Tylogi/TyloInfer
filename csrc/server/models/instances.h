#pragma once

#include "catalog.h"
#include "factory.h"
#include "engine/engine.h"
#include "engine/state/session/prefix_blocks.h"

#include <atomic>
#include <filesystem>
#include <future>
#include <memory>
#include <map>
#include <mutex>

namespace tyloi::server {
class Storage;
struct LoadPolicy {
    bool pin = false;
    std::optional<std::chrono::seconds> idle_ttl;
    // The runtime inference policy is applied once, when the instance is created.
    bool mtp_enabled = true;
};
struct InstanceInfo {
    enum class State { loaded, unloading, failed } state;
    std::string id, model, artifact_id, backend;
    engine::EngineInfo capabilities;
    engine::Engine::Snapshot execution;
    LoadPolicy policy;
};

class EngineInstance : public std::enable_shared_from_this<EngineInstance> {
public:
    const std::string id, model, artifact_id, backend;
    EngineInstance(const ModelCatalog::Artifact& artifact, std::string backend, std::unique_ptr<engine::Engine> engine);
    InstanceInfo snapshot() const;
    const MfqSamplingParams& sampling() const { return sampling_; }
    // The returned native outbox keeps this instance alive and cancels on abandonment.
    std::shared_ptr<engine::Output> submit(engine::EngineRequest request);
    bool cancel(const engine::RequestId& id);
private:
    friend class EngineInstances;
    void close(bool force);
    // A partial policy patch: an absent optional keeps the current value and a set-but-empty idle
    // timeout clears it. The result never combines a pin with an idle timeout.
    InstanceInfo apply_policy(std::optional<bool> pin, std::optional<std::optional<std::chrono::seconds>> idle_ttl);
    std::unique_ptr<engine::Engine> engine_;
    MfqSamplingParams sampling_;
    std::atomic<bool> stopping_{false}, close_failed_{false};
    std::mutex admission_;
    bool accepting_ = true;
    LoadPolicy policy_;
    engine::Clock::time_point last_activity_ = engine::Clock::now();
    bool idle_expired(engine::Clock::time_point now);
};

struct LoadControl {
    enum class Phase { resolving, loading, initializing };
    const std::atomic<bool>* cancelled = nullptr;
    std::function<void(Phase)> progress;
    // Native construction is synchronous; cancellation there is observed on return.
    void checkpoint(Phase phase) const;
};

class EngineInstances {
public:
    using Aliases = std::map<std::string, std::string>;
    struct Support { bool supported; std::string reason; };
    EngineInstances(ModelCatalog& catalog, Storage& storage, EngineFactory factory);
    ~EngineInstances();
    Support support(const ModelCatalog::Artifact& artifact) const;
    InstanceInfo load(const std::string& model, LoadOptions options = {}, const LoadControl& control = {}, LoadPolicy policy = {});
    void unload(const std::string& instance_id, bool force = false);
    // Applies a PATCH-style policy update to one instance and returns its new representation.
    InstanceInfo update_policy(const std::string& instance_id, std::optional<bool> pin,
        std::optional<std::optional<std::chrono::seconds>> idle_ttl);
    // Runs a cache/session command on one instance or every loaded instance; returns released units.
    std::size_t cache_control(const std::optional<std::string>& instance_id, const engine::SessionCommand& command);
    // Reclaims on-disk prefix blocks after all loaded instances pass a quiescence barrier.
    cache::PrefixReclaim reclaim_prefix_cache(const std::filesystem::path& directory,
        const std::optional<std::string>& space = {});
    // Re-aims one instance (or every loaded instance) at a context size; the returned snapshots carry the actual size.
    std::vector<InstanceInfo> reload(const std::optional<std::string>& instance_id, std::int64_t context);
    const std::string& backend() const { return factory_.backend; }
    std::vector<InstanceInfo> list() const;
    /** True when a loaded, unloading, or in-flight load selects this artifact or model name. */
    bool busy_for(const std::string& artifact_id, const std::string& name) const;
    std::shared_ptr<EngineInstance> get(const std::string& model) const;
    std::shared_ptr<EngineInstance> acquire(const std::string& model, const std::function<bool()>& cancelled = {});
    Aliases aliases() const;
    Aliases set_aliases(Aliases values);
    std::vector<std::string> names() const;
    void request_stop();
    void shutdown();
    void reap_idle();
private:
    struct Load {
        std::string id;
        std::shared_future<std::shared_ptr<EngineInstance>> result;
    };
    std::vector<std::shared_ptr<EngineInstance>> targets(const std::optional<std::string>& instance_id);
    std::shared_ptr<EngineInstance> open(const ModelCatalog::Artifact& artifact, LoadOptions options,
        const LoadControl& control, LoadPolicy policy, bool reuse, const std::function<bool()>& cancelled);
    std::shared_ptr<EngineInstance> create(const ModelCatalog::Artifact& artifact, LoadOptions options,
        const LoadControl& control, LoadPolicy policy);
    Aliases read_aliases();
    ModelCatalog& catalog_;
    Storage& storage_;
    EngineFactory factory_;
    mutable std::mutex mutex_;
    std::mutex operations_;
    std::map<std::string, std::shared_ptr<EngineInstance>> instances_;
    std::optional<Load> loading_;
    Aliases aliases_;
    std::atomic<bool> stopping_{false};
};
}
