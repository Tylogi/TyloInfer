#include "runtime.h"
#include "http.h"
#include "instance.h"
#include "instance_json.h"
#include "engine.h"
#include "models/instances.h"
#include "protocol.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
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

std::optional<std::string> query_uuid(const httplib::Request& request, const char* name) {
    if (!request.has_param(name)) return std::nullopt;
    const auto value = request.get_param_value(name);
    if (value.empty()) throw ApiError(400, "invalid_request", std::string(name) + " must be a UUID");
    // normalize_uuid rejects malformed values, which describe_failure reports as 400 invalid_request.
    return normalize_uuid(value);
}

std::int64_t query_integer(const httplib::Request& request, const char* name, std::int64_t fallback,
        std::int64_t minimum, std::int64_t maximum) {
    if (!request.has_param(name)) return fallback;
    const auto value = request.get_param_value(name);
    std::int64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < minimum || result > maximum)
        throw ApiError(400, "invalid_request",
            std::string(name) + " must be between " + std::to_string(minimum) + " and " + std::to_string(maximum));
    return result;
}

// shortcut: only the timestamp prefix is decoded, because the native server keeps no metric history to compare it against.
void valid_timestamp(const httplib::Request& request, const char* name) {
    const auto value = request.get_param_value(name);
    std::tm parts{};
    std::istringstream input(value);
    input >> std::get_time(&parts, "%Y-%m-%dT%H:%M:%S");
    if (input.fail() || (value.size() > 19 && value[19] != '.' && value[19] != '+' && value[19] != 'Z' && value[19] != '-'))
        throw ApiError(400, "invalid_request", std::string(name) + " must be an ISO-8601 timestamp");
}

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

// The native server has no runtime log store and no persisted metric history; both lists stay empty.
json empty_page() { return {{"data", json::array()}}; }
} // namespace

void register_runtime(httplib::Server& http, ModelInstances& instances) {
    const auto started_at = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    auto monitor = std::make_shared<ResourceMonitor>();
    http.Get("/api/v1/runtime/status", [&](const httplib::Request& request, httplib::Response& response) {
        const auto items = instances.list();
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
    });
    http.Get("/api/v1/runtime/instances", [&](const httplib::Request&, httplib::Response& response) {
        auto data = json::array();
        for (const auto& item : instances.list()) data.push_back(instance_json(item));
        set_json(response, {{"data", std::move(data)}});
    });
    http.Get("/api/v1/runtime/instances/([^/]+)", [&](const httplib::Request& request, httplib::Response& response) {
        const auto id = normalize_uuid(request.matches[1]);
        const auto items = instances.list();
        const auto found = find_instance(items, id);
        if (found == items.end())
            throw ApiError(404, "runtime_instance_not_found", "runtime instance was not found: " + id);
        set_json(response, instance_json(*found));
    });
    http.Get("/api/v1/runtime/metrics", [&](const httplib::Request& request, httplib::Response& response) {
        query_uuid(request, "instance_id");
        if (request.has_param("since")) valid_timestamp(request, "since");
        query_integer(request, "limit", 200, 1, 2000);
        set_json(response, empty_page());
    });
    http.Get("/api/v1/runtime/logs", [&](const httplib::Request& request, httplib::Response& response) {
        query_uuid(request, "instance_id");
        if (request.has_param("level")) {
            const auto level = request.get_param_value("level");
            if (level != "debug" && level != "info" && level != "warning" && level != "error")
                throw ApiError(400, "invalid_request", "level must be debug, info, warning, or error");
        }
        query_integer(request, "after", 0, 0, std::numeric_limits<std::int64_t>::max());
        query_integer(request, "limit", 200, 1, 2000);
        set_json(response, empty_page());
    });
    http.Get("/api/v1/runtime/resources", [&, monitor](const httplib::Request&, httplib::Response& response) {
        set_json(response, monitor->sample(instances.list()));
    });
    http.Get("/api/v1/runtime/serving", [&, started_at](const httplib::Request& request, httplib::Response& response) {
        const auto scope = request.has_param("scope") ? request.get_param_value("scope") : std::string("session");
        if (scope != "session" && scope != "alltime")
            throw ApiError(400, "invalid_request", "scope must be session or alltime");
        if (request.has_param("model") && request.get_param_value("model").empty())
            throw ApiError(400, "invalid_request", "model must be a nonempty string");
        // Completed-request totals live in the Python store; the native server reports the empty session instead.
        set_json(response, {{"scope", scope}, {"started_at", started_at},
            {"totals", {{"completed_requests", 0}, {"prompt_tokens", 0}, {"prefill_tokens", 0},
                {"completion_tokens", 0}, {"cached_tokens", 0}, {"prefill_tps", nullptr}, {"decode_tps", nullptr},
                {"ttft_ms", nullptr}, {"cache_hit_rate", nullptr}}},
            {"activity", json::array()}});
    });
    http.Get("/api/v1/runtime/models", [&](const httplib::Request&, httplib::Response& response) {
        auto data = json::array();
        for (const auto& item : instances.list()) {
            if (item.state != InstanceInfo::State::loaded) continue;
            if (std::any_of(data.begin(), data.end(),
                    [&](const auto& entry) { return entry.at("id") == item.model; })) continue;
            data.push_back({{"id", item.model}});
        }
        set_json(response, {{"data", std::move(data)}});
    });
    http.Get("/api/v1/runtime/model-aliases", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, {{"aliases", instances.aliases()}});
    });
    http.Put("/api/v1/runtime/model-aliases", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        fields(body, {"aliases"});
        if (!body.contains("aliases") || !body["aliases"].is_object())
            throw ApiError(400, "invalid_request", "aliases must be an object of strings");
        for (const auto& value : body["aliases"])
            if (!value.is_string()) throw ApiError(400, "invalid_request", "aliases must be an object of strings");
        set_json(response, {{"aliases", instances.set_aliases(body["aliases"].get<ModelInstances::Aliases>())}});
    });
}
} // namespace mfq::server
