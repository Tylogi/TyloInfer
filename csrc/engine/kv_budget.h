#pragma once

#include "engine/state/kv_offload.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace tyloi::engine {

// Placement applies to raw K/V. Indexer and kernel scratch remain device-local.
struct KvOptions {
    std::string placement = "auto";
    bool dynamic = false;
    bool budget_policy = false, ram_preallocated = true;
    std::size_t gpu_budget = 0;
    std::size_t gpu_limit = 0, ram_limit = 0;
    bool ram_limit_set = false;
    bool additional_request = false;
    std::int64_t auto_stream_threshold = 65536, resident_cells = 32768;
    std::string directory;
};

struct KvDemand {
    std::size_t raw = 0, index = 0, reserve = 0, per_cell = 0, metadata = 0,
                initial_mapped = 0, stream_buffer = 0, allocations = 0;
    std::int64_t maximum = 0, cells = 0;
    bool discrete = false, dynamic_available = false, admission = false;
};

struct KvPlacement {
    std::size_t gpu = 0;
    std::int64_t cells = 0;
    bool exclusive = false, raw_host_only = false, automatic_ram = false;
    std::optional<state::KvOffloadConfig> offload;
    std::optional<std::size_t> disk_needed;
};

} // namespace tyloi::engine
