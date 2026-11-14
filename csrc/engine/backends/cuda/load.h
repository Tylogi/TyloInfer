#pragma once
#include "engine/load.h"
#include <chrono>

namespace tyloi::engine { struct Config; }
namespace tyloi::weights { class Cache; }

namespace tyloi::cuda { class Executor; struct CudaLoadConfig; class KvMemoryManager; class KvOffloadStore; }

namespace tyloi::cuda::internal {

bool cuda_has_separate_memory();
tyloi::engine::Config resolve_cuda_config(const tyloi::engine::EngineOptions& options);

void setup_cuda_load(
    const tyloi::engine::LoadOptions& options,
    CudaLoadConfig& config, Executor& executor, tyloi::weights::Cache& weights,
    std::shared_ptr<KvMemoryManager>& kv, std::shared_ptr<KvOffloadStore>& offload);

} // namespace tyloi::cuda::internal

namespace tyloi::cuda {

void finalize_kv(KvMemoryManager* kv, tyloi::weights::Cache& weights,
    std::shared_ptr<KvOffloadStore>& offload);
std::chrono::steady_clock::time_point finish_load(
    KvMemoryManager* kv, tyloi::weights::Cache& weights, bool report_memory,
    std::size_t pending_kv = 0);

} // namespace tyloi::cuda


namespace tyloi::cuda {
std::unique_ptr<engine::Engine> load(engine::EngineOptions options);
void reload(engine::Engine& engine, engine::EngineOptions options);
}
