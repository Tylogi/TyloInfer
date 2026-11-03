#include "protocol.h"

#include "nlohmann/json.hpp"
#include "chat.h"
#include "mtp_metrics.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <aclapi.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mfq::protocol {
json request_metric_values_json(
        const RequestMetricValues & values,
        const MfqSamplingParams & sampling) {
    return {
        {"prefill_tokens", values.prefill_tokens},
        {"ttft_ms", values.ttft_ms},
        {"prefill_ms", values.prefill_ms},
        {"prefill_tps", values.prefill_tps},
        {"multimodal_ms", values.multimodal_ms},
        {"model_prefill_ms", values.model_prefill_ms},
        {"decode_ms", values.decode_ms},
        {"decode_tps", values.decode_tps},
        {"generation_ms", values.generation_ms},
        {"generation_tps", values.generation_tps},
        {"sampling", {
            {"max_tokens", sampling.max_tokens},
            {"temperature", sampling.temperature},
            {"top_k", sampling.top_k},
            {"top_p", sampling.top_p},
            {"presence_penalty", sampling.presence_penalty},
            {"frequency_penalty", sampling.frequency_penalty},
            {"repetition_penalty", sampling.repetition_penalty},
            {"seed", sampling.seed},
            {"enable_thinking", sampling.enable_thinking},
            {"enable_mtp", sampling.enable_mtp},
            {"mtp_max_draft_tokens", sampling.mtp_max_draft_tokens},
        }},
    };
}
void add_request_runtime_metrics(json& value, const RequestMetrics& metrics) {
    value["mtp_available"] = metrics.mtp.available ? 1.0 : 0.0;
    if (!metrics.mtp.available) return;
    mfq::engine::Metrics values;
    mfq::engine::mtp::append_generation_metrics(values, metrics.mtp);
    for (const auto& [name, number] : values) value[name] = number;
}
void log_request_metrics(const std::string & id, bool chat, bool stream,
                                size_t prompt_tokens, const MfqSamplingParams & sampling,
                                const CompletionResult & result,
                                const RequestMetricValues & values) {
    const char * enabled = std::getenv("TYLOINFER_RUNTIME_REQUEST_METRICS");
    if (enabled != nullptr && std::atoi(enabled) == 0) return;

    const bool penalties = sampling.presence_penalty != 0.0 ||
        sampling.frequency_penalty != 0.0 || sampling.repetition_penalty != 1.0;

    std::ostringstream line;
    line << std::fixed << std::setprecision(3)
         << "request_metrics"
         << " id=" << id
         << " endpoint=" << (chat ? "chat" : "completion")
         << " stream=" << (stream ? 1 : 0)
         << " prompt_tokens=" << prompt_tokens
         << " prefill_tokens=" << values.prefill_tokens
         << " completion_tokens=" << result.completion_tokens
         << " max_tokens=" << sampling.max_tokens
         << " ttft_ms=" << values.ttft_ms
         << " prefill_ms=" << values.prefill_ms
         << " prefill_tps=" << values.prefill_tps
         << " multimodal_ms=" << values.multimodal_ms
         << " model_prefill_ms=" << values.model_prefill_ms
         << " decode_ms=" << values.decode_ms
         << " decode_tps=" << values.decode_tps
         << " generation_ms=" << values.generation_ms
         << " generation_tps=" << values.generation_tps
         << " temperature=" << sampling.temperature
         << " top_k=" << sampling.top_k
         << " top_p=" << sampling.top_p
         << " presence_penalty=" << sampling.presence_penalty
         << " frequency_penalty=" << sampling.frequency_penalty
         << " repetition_penalty=" << sampling.repetition_penalty
         << " mtp=" << (sampling.enable_mtp ? 1 : 0)
         << " mtp_max_draft_tokens=" << sampling.mtp_max_draft_tokens
         << " seed=" << sampling.seed
         << " penalties=" << (penalties ? 1 : 0)
         << " finish_reason=" << result.finish_reason
         << " client_connected=" << (result.client_connected ? 1 : 0);
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cout << line.str() << std::endl;
}
RuntimeRequestMetrics::RuntimeRequestMetrics()
    : started_steady_(std::chrono::steady_clock::now()),
      started_unix_(unix_time_seconds()) {}
void RuntimeRequestMetrics::begin() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++total_requests_;
    ++active_requests_;
}
void RuntimeRequestMetrics::complete(
        const std::string & id,
        bool chat,
        bool stream,
        size_t prompt_tokens,
        const CompletionResult & result,
        const RequestMetricValues & values) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_requests_ > 0) --active_requests_;
    live_requests_.erase(id);
    total_prompt_tokens_ += prompt_tokens;
    total_completion_tokens_ +=
        static_cast<uint64_t>(std::max<int32_t>(result.completion_tokens, 0));
    last_request_ = {
        {"id", id},
        {"endpoint", chat ? "chat" : "completion"},
        {"stream", stream},
        {"prompt_tokens", prompt_tokens},
        {"prefill_tokens", values.prefill_tokens},
        {"completion_tokens", result.completion_tokens},
        {"ttft_ms", values.ttft_ms},
        {"prefill_ms", values.prefill_ms},
        {"prefill_tps", values.prefill_tps},
        {"multimodal_ms", values.multimodal_ms},
        {"model_prefill_ms", values.model_prefill_ms},
        {"decode_ms", values.decode_ms},
        {"decode_tps", values.decode_tps},
        {"generation_ms", values.generation_ms},
        {"generation_tps", values.generation_tps},
        {"finish_reason", result.finish_reason},
        {"client_connected", result.client_connected},
        {"completed_at", unix_time_seconds()},
    };
}
void RuntimeRequestMetrics::progress(const std::string & id, const json & value) {
    std::lock_guard<std::mutex> lock(mutex_);
    live_requests_[id] = value;
}
void RuntimeRequestMetrics::fail(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex_);
    live_requests_.erase(id);
    if (active_requests_ > 0) --active_requests_;
    ++failed_requests_;
}
uint64_t RuntimeRequestMetrics::active_requests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_requests_;
}
json RuntimeRequestMetrics::snapshot(
        const MfqRuntimeTransportConfig & config,
        int64_t max_context,
        bool reloading) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const double uptime_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_steady_).count();
    return {
        {"status", "ok"},
        {"model", config.model_name},
        {"model_type", config.model_type},
        {"max_context", max_context},
        {"context_capacity", config.context_capacity},
        {"reloading", reloading},
        {"vocab_size", config.vocab_size},
        {"started_at", started_unix_},
        {"uptime_seconds", uptime_seconds},
        {"active_requests", active_requests_},
        {"total_requests", total_requests_},
        {"failed_requests", failed_requests_},
        {"total_prompt_tokens", total_prompt_tokens_},
        {"total_completion_tokens", total_completion_tokens_},
        {"last_request", last_request_},
        {"live_requests", live_requests_},
    };
}
ActiveRequest::ActiveRequest(RuntimeRequestMetrics & metrics)
    : metrics_(metrics) {
    metrics_.begin();
}

ActiveRequest::~ActiveRequest() {
    if (!completed_) metrics_.fail(id_);
}
void ActiveRequest::progress(const std::string & id, const json & value) {
    id_ = id;
    metrics_.progress(id, value);
}
void ActiveRequest::complete(
        const std::string & id,
        bool chat,
        bool stream,
        size_t prompt_tokens,
        const CompletionResult & result,
        const RequestMetricValues & values) {
    metrics_.complete(id, chat, stream, prompt_tokens, result, values);
    completed_ = true;
}
} // namespace mfq::protocol
