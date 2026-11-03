#include "handlers.h"
#include "http.h"
#include "instance.h"
#include "instance_json.h"
#include "models/instances.h"
#include "protocol.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

namespace mfq::server {
namespace {
using json = nlohmann::json;

void fields(const json& body, std::initializer_list<const char*> allowed) {
    if (!body.is_object()) throw ApiError(400, "invalid_request", "request must be a JSON object");
    for (const auto& item : body.items())
        if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
            throw ApiError(400, "invalid_request", "unknown model operation parameter: " + item.key());
}

std::int64_t metrics_age_ms(const InstanceInfo& item) {
    return std::max<std::int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(
        engine::Clock::now() - item.execution.captured).count());
}

// The native server has no shared runtime worker, so a status resource is a projection of one instance snapshot.
json status_json(const InstanceInfo& item) {
    const auto state = item.state == InstanceInfo::State::loaded
        ? (item.execution.requests ? "busy" : "ready")
        : item.state == InstanceInfo::State::unloading ? "unloading" : "failed";
    return {{"source", "native-instances"}, {"status", "ok"}, {"instance_id", item.id}, {"model", item.model},
        {"backend", item.backend}, {"model_type", item.capabilities.model_type}, {"runtime_state", state},
        {"max_context", item.capabilities.max_context}, {"context_capacity", item.capabilities.max_context},
        {"vocab_size", item.capabilities.vocab_size}, {"max_requests", item.capabilities.max_requests},
        {"active_requests", item.execution.requests}, {"reloading", item.state == InstanceInfo::State::unloading},
        {"model_capabilities", mfq::protocol::model_capability_profile_json(item.capabilities.capabilities)},
        {"metrics_age_ms", metrics_age_ms(item)}};
}

std::vector<InstanceInfo>::const_iterator find_instance(const std::vector<InstanceInfo>& items, const std::string& id) {
    return std::find_if(items.begin(), items.end(), [&](const auto& item) { return item.id == id; });
}

struct CpuTimes {
    std::int64_t total = 0, idle = 0;
    bool valid = false;
};

CpuTimes cpu_times() {
    CpuTimes result;
#ifdef _WIN32
    FILETIME idle{}, kernel{}, user{};
    if (GetSystemTimes(&idle, &kernel, &user)) {
        const auto ticks = [](const FILETIME& value) {
            return (static_cast<std::uint64_t>(value.dwHighDateTime) << 32) | value.dwLowDateTime;
        };
        result.idle = static_cast<std::int64_t>(ticks(idle));
        result.total = static_cast<std::int64_t>(ticks(kernel) + ticks(user));  // kernel time includes idle
        result.valid = true;
    }
#elif defined(__APPLE__)
    static const mach_port_t host = mach_host_self();
    host_cpu_load_info_data_t loads{};
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(host, HOST_CPU_LOAD_INFO, reinterpret_cast<host_info_t>(&loads), &count) == KERN_SUCCESS) {
        result.idle = static_cast<std::int64_t>(loads.cpu_ticks[CPU_STATE_IDLE]);
        result.total = result.idle + loads.cpu_ticks[CPU_STATE_USER] + loads.cpu_ticks[CPU_STATE_SYSTEM] +
            loads.cpu_ticks[CPU_STATE_NICE];
        result.valid = true;
    }
#else
    std::ifstream input("/proc/stat");
    std::string label;
    std::array<std::int64_t, 10> values{};
    std::size_t fields = 0;
    if (input >> label) while (fields < values.size() && input >> values[fields]) ++fields;
    if (fields >= 4) {
        // guest and guest_nice are already counted inside user and nice; iowait counts as idle.
        result.total = 0;
        for (std::size_t index = 0; index < std::min<std::size_t>(fields, 8); ++index) result.total += values[index];
        result.idle = values[3] + values[4];
        result.valid = true;
    }
#endif
    return result;
}

struct WeightMetric {
    const char* field, *counter, *enabled;
};
// The same native runtime counters the Python resource monitor turns into per-instance byte rates.
constexpr std::array<WeightMetric, 5> weight_metrics{{
    {"expert_read_bytes_per_second", "ssd_expert_bytes_read", "ssd_expert_enabled"},
    {"ple_read_bytes_per_second", "ple_source_bytes_read", "ssd_ple_enabled"},
    {"engram_read_bytes_per_second", "engram_bytes_read", nullptr},
    {"kv_read_bytes_per_second", "qsa_kv_ssd_read_bytes", "qsa_kv_offload_enabled"},
    {"kv_write_bytes_per_second", "qsa_kv_ssd_written_bytes", "qsa_kv_offload_enabled"}}};

std::optional<double> counter(const engine::Metrics& metrics, const char* name) {
    const auto found = std::find_if(metrics.begin(), metrics.end(),
        [&](const auto& item) { return item.first == name; });
    if (found == metrics.end()) return std::nullopt;
    return found->second;
}

struct WeightSample {
    std::chrono::steady_clock::time_point at;
    std::map<std::string, double> counters;
};

class ResourceMonitor {
public:
    json sample(const std::vector<InstanceInfo>& items);

private:
    std::mutex mutex_;
    bool sampled_ = false;
    std::chrono::steady_clock::time_point sampled_at_;
    CpuTimes cpu_;
    std::unordered_map<std::string, WeightSample> weights_;
};

json ResourceMonitor::sample(const std::vector<InstanceInfo>& items) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard guard(mutex_);
    const auto elapsed = std::chrono::duration<double>(now - sampled_at_).count();
    // Rates are only meaningful between samples taken close together, as on the Python side.
    const bool comparable = sampled_ && elapsed > 0 && elapsed <= 10.0;
    const auto current_cpu = cpu_times();
    std::optional<double> utilization;
    if (comparable && current_cpu.valid && cpu_.valid && current_cpu.total > cpu_.total && current_cpu.idle >= cpu_.idle)
        utilization = std::min(100.0, 100.0 * (1.0 - static_cast<double>(current_cpu.idle - cpu_.idle) /
            static_cast<double>(current_cpu.total - cpu_.total)));
    cpu_ = current_cpu;
    sampled_ = true;
    sampled_at_ = now;
    const auto [total, available] = host_memory_bytes();
    auto data = json::array();
    std::unordered_map<std::string, WeightSample> current_weights;
    for (const auto& item : items) {
        if (item.state != InstanceInfo::State::loaded) continue;
        const auto stored = weights_.find(item.id);
        const WeightSample* previous = stored == weights_.end() ? nullptr : &stored->second;
        WeightSample current{now, {}};
        for (const auto& metric : weight_metrics) {
            if (const auto value = counter(item.execution.runtime, metric.counter)) current.counters[metric.field] = *value;
            else if (metric.enabled) {
                const auto enabled = counter(item.execution.runtime, metric.enabled);
                if (enabled && *enabled == 0.0) current.counters[metric.field] = 0.0;
            }
        }
        auto entry = json{{"instance_id", item.id}, {"model", item.model}};
        for (const auto& metric : weight_metrics) {
            json rate = nullptr;
            const auto value = current.counters.find(metric.field);
            if (comparable && previous && value != current.counters.end()) {
                const auto prior = previous->counters.find(metric.field);
                if (prior != previous->counters.end() && value->second >= prior->second)
                    rate = (value->second - prior->second) / elapsed;
            }
            entry[metric.field] = rate;
        }
        current_weights.emplace(item.id, std::move(current));
        data.push_back(std::move(entry));
    }
    weights_ = std::move(current_weights);
    return {{"sampled_at", std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()},
        {"interval_seconds", comparable ? json(elapsed) : json(nullptr)},
        {"cpu_name", nullptr},
        {"cpu_cores", static_cast<int>(std::max(1U, std::thread::hardware_concurrency()))},
        {"cpu_utilization_percent", utilization ? json(*utilization) : json(nullptr)},
        {"gpus", json::array()}, {"disks", json::array()},
        {"memory_total_bytes", total ? json(total) : json(nullptr)},
        {"memory_used_bytes", total && available <= total ? json(total - available) : json(nullptr)},
        {"memory_available_bytes", available ? json(available) : json(nullptr)},
        {"memory_bandwidth_bytes_per_second", nullptr},
        {"memory_bandwidth_limit_bytes_per_second", nullptr},
        {"memory_bandwidth_utilization_percent", nullptr},
        {"weights", std::move(data)}};
}

// The monitor and the serving clock live for the process, like every service in Services.
ResourceMonitor& monitor() {
    static ResourceMonitor instance;
    return instance;
}

} // namespace

void runtime_status(const Services& services, const httplib::Request& request, httplib::Response& response) {
    const auto items = services.instances.list();
    if (const auto id = query_uuid(request, "instance_id")) {
        const auto found = find_instance(items, *id);
        if (found == items.end())
            throw ApiError(404, "runtime_instance_not_found", "runtime instance was not found: " + *id);
        set_json(response, status_json(*found));
        return;
    }
    const auto loaded = std::find_if(items.begin(), items.end(),
        [](const auto& item) { return item.state == InstanceInfo::State::loaded; });
    set_json(response, loaded == items.end()
        ? json{{"source", "native-instances"}, {"status", "ok"}, {"runtime_state", "idle"}}
        : status_json(*loaded));
}

void runtime_instances(const Services& services, const httplib::Request&, httplib::Response& response) {
    auto data = json::array();
    for (const auto& item : services.instances.list()) data.push_back(instance_json(item));
    set_json(response, {{"data", std::move(data)}});
}

void runtime_instance(const Services& services, const httplib::Request& request, httplib::Response& response) {
    const auto id = normalize_uuid(request.matches[1]);
    const auto items = services.instances.list();
    const auto found = find_instance(items, id);
    if (found == items.end())
        throw ApiError(404, "runtime_instance_not_found", "runtime instance was not found: " + id);
    set_json(response, instance_json(*found));
}



void runtime_resources(const Services& services, const httplib::Request&, httplib::Response& response) {
    set_json(response, monitor().sample(services.instances.list()));
}



void runtime_aliases_get(const Services& services, const httplib::Request&, httplib::Response& response) {
    set_json(response, {{"aliases", services.instances.aliases()}});
}

void runtime_aliases_put(const Services& services, const httplib::Request& request, httplib::Response& response) {
    const auto body = parse_json(request);
    fields(body, {"aliases"});
    if (!body.contains("aliases") || !body["aliases"].is_object())
        throw ApiError(400, "invalid_request", "aliases must be an object of strings");
    for (const auto& value : body["aliases"])
        if (!value.is_string()) throw ApiError(400, "invalid_request", "aliases must be an object of strings");
    set_json(response, {{"aliases", services.instances.set_aliases(body["aliases"].get<ModelInstances::Aliases>())}});
}

} // namespace mfq::server
