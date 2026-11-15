#include "engine/budget.h"
#include "components/host_memory.h"
#include "engine/generation/prefill_budget.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::engine {
namespace {
constexpr const char* insufficient = "kv_memory_insufficient: available VRAM and RAM cannot hold the configured KV cache; choose automatic with SSD KV or reduce context length";

double kv_resident_fraction(std::int64_t context) {
    constexpr double x[]{0,131072,262144,1048576}, y[]{1,.6,.3,.1};
    if (context <= 0) return 1;
    if (context >= 1048576) return .1;
    double h[3], d[3], m[4];
    for (int i = 0; i < 3; ++i) { h[i] = x[i+1]-x[i]; d[i] = (y[i+1]-y[i])/h[i]; }
    m[0] = (3*d[0]-d[1])/2; m[3] = 0;
    for (int i = 1; i < 3; ++i) {
        const double w1 = 2*h[i]+h[i-1], w2 = h[i]+2*h[i-1];
        m[i] = (w1+w2)/(w1/d[i-1]+w2/d[i]);
    }
    int i = 0;
    while (context > x[i+1]) ++i;
    const double t = (context-x[i])/h[i], t2 = t*t, t3 = t2*t;
    return std::clamp((2*t3-3*t2+1)*y[i]+(t3-2*t2+t)*h[i]*m[i]+
        (-2*t3+3*t2)*y[i+1]+(t3-t2)*h[i]*m[i+1], y[i+1], y[i]);
}

std::size_t host_available() {
    return static_cast<std::size_t>(std::min<std::uint64_t>(
        sample_host_memory().available_host_bytes(), std::numeric_limits<std::size_t>::max()));
}
} // namespace

KvPlacement Budget::place_kv(const KvOptions& options, const KvDemand& demand,
        std::size_t available, weights::Cache& cache) {
    if (!options.ram_preallocated && options.placement != "auto_ssd")
        throw std::invalid_argument("kv_ssd_required: disabling RAM preallocation requires SSD KV");
    if (available < demand.reserve || demand.index > available - demand.reserve)
        throw std::runtime_error(insufficient);
    KvPlacement result;
    result.gpu = available - demand.reserve - demand.index;
    if (options.gpu_limit) result.gpu = std::min(result.gpu, options.gpu_limit);
    result.cells = demand.cells;
    const auto ram_available = [&] {
        const auto available_ram = host_available();
        auto ram = available_ram > (std::size_t(1) << 30) ? available_ram - (std::size_t(1) << 30) : 0;
        if (options.ram_limit_set) ram = std::min(ram, options.ram_limit);
        return ram;
    };
    auto ram = ram_available();
    auto& gpu = result.gpu;
    auto& cells = result.cells;
    const auto raw = demand.raw;
    const bool automatic = options.placement == "auto" || options.placement == "auto_ssd";
    const bool tiered = options.budget_policy && demand.discrete;
    if (tiered) {
        if (options.gpu_budget && options.gpu_budget < demand.index)
            throw std::runtime_error("kv_budget_below_indexer: VRAM KV budget must cover the complete target indexer cache");
        const auto desired = options.gpu_budget ? options.gpu_budget - demand.index :
            (options.placement == "vram" || options.dynamic ? std::max(raw, demand.initial_mapped) :
             options.placement == "ram" ? 0 : std::size_t(std::ceil(raw * kv_resident_fraction(demand.maximum))));
        if (!demand.admission && options.gpu_budget && desired > gpu)
            throw std::runtime_error("kv_vram_insufficient: explicit KV VRAM budget exceeds available capacity after weights and workspace");
        gpu = std::min(gpu, desired);
        if (options.placement == "ram") gpu = 0;
        if (!options.dynamic && options.placement != "vram" && gpu < raw) {
            if (!demand.per_cell) throw std::invalid_argument("invalid KV per-cell demand");
            cells = std::min<std::int64_t>(demand.maximum, gpu / demand.per_cell);
            gpu = std::size_t(cells) * demand.per_cell;
            const auto required = raw - gpu;
            if (required > ram) {
                cache.reserve(prefill_budget_add(required, std::size_t(1) << 30));
                ram = ram_available();
            }
            if (options.placement != "auto_ssd" && required > ram) throw std::runtime_error(insufficient);
            if (options.ram_preallocated && required <= ram) {
                result.exclusive = cells > 0;
                result.raw_host_only = cells == 0;
            } else {
                if (options.placement != "auto_ssd") throw std::runtime_error(insufficient);
                const auto disk_needed = options.ram_preallocated ? required - std::min(required, ram) : required;
                auto directory = std::filesystem::path(options.directory);
                if (directory.empty()) throw std::runtime_error("kv_ssd_insufficient: SSD KV directory is required");
                while (!std::filesystem::exists(directory) && directory.has_parent_path()) directory = directory.parent_path();
                if (std::filesystem::space(directory).available < prefill_budget_add(disk_needed, 64ULL << 20))
                    throw std::runtime_error("kv_ssd_insufficient: insufficient disk space for the target cold KV capacity");
                state::KvOffloadConfig config;
                config.buffer_bytes = demand.stream_buffer;
                config.gpu_budget_bytes = prefill_budget_add(gpu, config.buffer_bytes);
                config.ram_budget_bytes = std::min(ram, prefill_budget_add(required, config.buffer_bytes));
                config.preallocate = true; config.ram_preallocated = options.ram_preallocated;
                config.ssd_enabled = true; config.directory = options.directory;
                result.offload = std::move(config);
                result.disk_needed = disk_needed;
            }
        }
    }
    result.automatic_ram = !tiered && automatic && !options.dynamic && options.auto_stream_threshold > 0 &&
        demand.maximum >= options.auto_stream_threshold && raw <= ram && cells > 0 && demand.discrete;
    if (result.automatic_ram) {
        const auto count_bytes = demand.allocations > std::numeric_limits<std::size_t>::max() / 12
            ? std::numeric_limits<std::size_t>::max() : demand.allocations * 12;
        const auto per_cell = prefill_budget_add(demand.per_cell, count_bytes);
        if (gpu <= demand.metadata) result.automatic_ram = false;
        else if (!per_cell) throw std::invalid_argument("invalid KV per-cell demand");
        else cells = std::min<std::int64_t>(cells, (gpu - demand.metadata) / per_cell);
        if (!cells) result.automatic_ram = false;
    }
    if (options.placement == "vram" && (!demand.admission || !options.dynamic) && raw > gpu)
        throw std::runtime_error("kv_vram_insufficient: configured KV cache does not fit in available VRAM");
    if (options.placement == "ram" && raw > ram) throw std::runtime_error(insufficient);
    if (options.placement == "auto" && (!demand.admission || !options.dynamic) && raw > gpu && raw-gpu > ram)
        throw std::runtime_error(insufficient);
    if (options.dynamic && !demand.dynamic_available)
        throw std::runtime_error("kv_dynamic_unavailable: CUDA virtual memory management is unavailable");
    if (options.dynamic && !demand.admission && raw > gpu)
        throw std::runtime_error("kv_dynamic_unavailable: experimental dynamic KV currently requires a GPU-resident cache");
    if (demand.admission && options.dynamic && demand.initial_mapped > gpu)
        throw std::runtime_error("kv_dynamic_unavailable: initial dynamic KV pages exceed available VRAM");
    if (!tiered && !result.automatic_ram && !options.dynamic && options.placement != "ram" && raw > gpu) {
        state::KvOffloadConfig config;
        config.buffer_bytes = std::min<std::size_t>(64ULL << 20, std::max<std::size_t>(4096, raw/16));
        config.gpu_budget_bytes = std::min(raw, gpu);
        if (config.gpu_budget_bytes < config.buffer_bytes*2) throw std::runtime_error(insufficient);
        const auto required_ram = prefill_budget_add(raw - (config.gpu_budget_bytes-config.buffer_bytes), config.buffer_bytes);
        if (options.placement == "auto" && required_ram > ram) throw std::runtime_error(insufficient);
        config.ram_budget_bytes = std::min(ram, required_ram);
        config.directory = options.directory; config.preallocate = true;
        config.ssd_enabled = options.placement == "auto_ssd";
        result.offload = std::move(config);
    }
    return result;
}

} // namespace tyloi::engine
