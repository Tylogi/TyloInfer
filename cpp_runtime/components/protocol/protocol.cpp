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


int64_t unix_time_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string request_id(const char * prefix) {
    static std::atomic<uint64_t> sequence{0};
    const uint64_t n = sequence.fetch_add(1, std::memory_order_relaxed);
    return std::string(prefix) + std::to_string(unix_time_seconds()) + "-" + std::to_string(n);
}

static bool request_enable_thinking(const json & body, bool fallback) {
    if (body.contains("enable_thinking") && !body["enable_thinking"].is_null()) {
        if (!body["enable_thinking"].is_boolean()) {
            throw ApiError(400, "invalid_request_error", "enable_thinking must be boolean", "enable_thinking");
        }
        return body["enable_thinking"].get<bool>();
    }
    if (body.contains("chat_template_kwargs") && !body["chat_template_kwargs"].is_null()) {
        const auto & kwargs = body["chat_template_kwargs"];
        if (!kwargs.is_object()) {
            throw ApiError(400, "invalid_request_error", "chat_template_kwargs must be an object", "chat_template_kwargs");
        }
        if (kwargs.contains("enable_thinking")) {
            if (!kwargs["enable_thinking"].is_boolean()) {
                throw ApiError(400, "invalid_request_error", "enable_thinking must be boolean",
                               "chat_template_kwargs.enable_thinking");
            }
            return kwargs["enable_thinking"].get<bool>();
        }
    }
    return fallback;
}

static common_reasoning_format request_reasoning_format(const json & body) {
    if (!body.contains("reasoning_format") || body["reasoning_format"].is_null()) {
        return COMMON_REASONING_FORMAT_AUTO;
    }
    if (!body["reasoning_format"].is_string()) {
        throw ApiError(
            400, "invalid_request_error",
            "reasoning_format must be a string", "reasoning_format");
    }
    try {
        return common_reasoning_format_from_name(
            body["reasoning_format"].get<std::string>());
    } catch (const std::exception & error) {
        throw ApiError(
            400, "invalid_request_error", error.what(), "reasoning_format");
    }
}

static bool boolean_field(
        const json & body, const char * name, bool fallback) {
    if (!body.contains(name) || body[name].is_null()) return fallback;
    if (!body[name].is_boolean()) {
        throw ApiError(
            400, "invalid_request_error",
            std::string(name) + " must be boolean", name);
    }
    return body[name].get<bool>();
}

static std::string request_json_schema(const json & body) {
    const bool has_direct =
        body.contains("json_schema") && !body["json_schema"].is_null();
    const bool has_response_format =
        body.contains("response_format") &&
        !body["response_format"].is_null();
    if (has_direct && has_response_format) {
        throw ApiError(
            400, "invalid_request_error",
            "json_schema and response_format cannot both be specified",
            "response_format");
    }

    json schema;
    if (has_direct) {
        schema = body["json_schema"];
        if (schema.is_string()) {
            try {
                schema = json::parse(schema.get<std::string>());
            } catch (const std::exception &) {
                throw ApiError(
                    400, "invalid_request_error",
                    "json_schema string must contain valid JSON",
                    "json_schema");
            }
        }
    } else if (has_response_format) {
        const auto & response_format = body["response_format"];
        if (!response_format.is_object() ||
            !response_format.contains("type") ||
            !response_format["type"].is_string()) {
            throw ApiError(
                400, "invalid_request_error",
                "response_format must contain a string type",
                "response_format");
        }
        const std::string type = response_format["type"];
        if (type == "text") return {};
        if (type == "json_object") {
            schema = {{"type", "object"}};
        } else if (type == "json_schema") {
            if (!response_format.contains("json_schema") ||
                !response_format["json_schema"].is_object()) {
                throw ApiError(
                    400, "invalid_request_error",
                    "response_format.json_schema must be an object",
                    "response_format.json_schema");
            }
            const auto & envelope = response_format["json_schema"];
            schema = envelope.contains("schema")
                ? envelope["schema"]
                : envelope;
        } else {
            throw ApiError(
                400, "invalid_request_error",
                "response_format.type must be text, json_object, or json_schema",
                "response_format.type");
        }
    } else {
        return {};
    }

    if (!schema.is_object()) {
        throw ApiError(
            400, "invalid_request_error",
            "structured output JSON schema must be an object",
            has_direct ? "json_schema" : "response_format.json_schema.schema");
    }
    return schema.dump();
}

static std::optional<std::string> request_preformatted_prompt(
    const json & body);

static mfq::engine::ChatInput parse_chat_input(
        const json & body,
        bool enable_thinking_default) {
    if (!body.contains("messages") || !body["messages"].is_array() ||
        body["messages"].empty()) {
        throw ApiError(
            400, "invalid_request_error",
            "messages must be a non-empty array", "messages");
    }

    mfq::engine::ChatInput result;
    auto & inputs = result.template_inputs;
    result.preformatted_prompt = request_preformatted_prompt(body);
    result.reasoning_format = request_reasoning_format(body);
    inputs.json_schema = request_json_schema(body);
    inputs.reasoning_format = result.reasoning_format;
    inputs.enable_thinking = request_enable_thinking(
        body, enable_thinking_default);
    inputs.use_jinja = true;
    inputs.add_generation_prompt =
        boolean_field(body, "add_generation_prompt", true);
    if (result.preformatted_prompt) return result;

    try {
        inputs.messages = common_chat_msgs_parse_oaicompat(body["messages"]);
        if (body.contains("continue_final_message") &&
            !body["continue_final_message"].is_null()) {
            inputs.continue_final_message = common_chat_continuation_parse(
                body["continue_final_message"]);
        }
        if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE &&
            inputs.add_generation_prompt) {
            throw ApiError(
                400, "invalid_request_error",
                "add_generation_prompt and continue_final_message "
                "cannot both be enabled",
                "continue_final_message");
        }
        if (body.contains("parallel_tool_calls") &&
            !body["parallel_tool_calls"].is_null()) {
            result.parallel_tool_calls = boolean_field(
                body, "parallel_tool_calls", false);
        }
        if (body.contains("tools") && !body["tools"].is_null()) {
            inputs.tools = common_chat_tools_parse_oaicompat(body["tools"]);
        }
        const json tool_choice =
            body.contains("tool_choice") && !body["tool_choice"].is_null()
                ? body["tool_choice"] : json("auto");
        if (tool_choice.is_string()) {
            inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(
                tool_choice.get<std::string>());
        } else if (tool_choice.is_object()) {
            if (!tool_choice.contains("type") ||
                tool_choice["type"] != "function" ||
                !tool_choice.contains("function") ||
                !tool_choice["function"].is_object() ||
                !tool_choice["function"].contains("name") ||
                !tool_choice["function"]["name"].is_string()) {
                throw ApiError(
                    400, "invalid_request_error",
                    "named tool_choice must select a function name",
                    "tool_choice");
            }
            const std::string selected_name =
                tool_choice["function"]["name"];
            const auto selected = std::find_if(
                inputs.tools.begin(), inputs.tools.end(),
                [&](const common_chat_tool & tool) {
                    return tool.name == selected_name;
                });
            if (selected == inputs.tools.end()) {
                throw ApiError(
                    400, "invalid_request_error",
                    "named tool_choice does not match any supplied tool",
                    "tool_choice");
            }
            inputs.tools = {*selected};
            inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        } else {
            throw ApiError(
                400, "invalid_request_error",
                "tool_choice must be a string or function selector object",
                "tool_choice");
        }
        if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED &&
            inputs.tools.empty()) {
            throw ApiError(
                400, "invalid_request_error",
                "tool_choice required needs at least one tool",
                "tool_choice");
        }
        if (body.contains("chat_template_kwargs") &&
            !body["chat_template_kwargs"].is_null()) {
            if (!body["chat_template_kwargs"].is_object()) {
                throw ApiError(
                    400, "invalid_request_error",
                    "chat_template_kwargs must be an object",
                    "chat_template_kwargs");
            }
            for (const auto & item : body["chat_template_kwargs"].items()) {
                inputs.chat_template_kwargs[item.key()] = item.value().dump();
            }
        }
    } catch (const ApiError &) {
        throw;
    } catch (const std::exception & error) {
        throw ApiError(
            400, "invalid_request_error", error.what(), "messages");
    }
    return result;
}

static std::optional<std::string> request_preformatted_prompt(
        const json & body) {
    if (!body.contains("mfq_preformatted_prompt") ||
        body["mfq_preformatted_prompt"].is_null()) {
        return std::nullopt;
    }
    if (!body["mfq_preformatted_prompt"].is_string() ||
        body["mfq_preformatted_prompt"].get_ref<
            const std::string&>().empty()) {
        throw ApiError(
            400,
            "invalid_request_error",
            "mfq_preformatted_prompt must be a non-empty string",
            "mfq_preformatted_prompt");
    }
    return body["mfq_preformatted_prompt"].get<std::string>();
}

int64_t integer_field(const json & body, const char * name, int64_t fallback) {
    if (!body.contains(name) || body[name].is_null()) return fallback;
    if (!body[name].is_number_integer()) {
        throw ApiError(400, "invalid_request_error", std::string(name) + " must be an integer", name);
    }
    return body[name].get<int64_t>();
}

double number_field(const json & body, const char * name, double fallback) {
    if (!body.contains(name) || body[name].is_null()) return fallback;
    if (!body[name].is_number()) {
        throw ApiError(400, "invalid_request_error", std::string(name) + " must be a number", name);
    }
    return body[name].get<double>();
}

json sampling_params_json(const MfqSamplingParams & sampling) {
    return {
        {"max_tokens", sampling.max_tokens},
        {"temperature", sampling.temperature},
        {"top_k", sampling.top_k},
        {"top_p", sampling.top_p},
        {"presence_penalty", sampling.presence_penalty},
        {"frequency_penalty", sampling.frequency_penalty},
        {"repetition_penalty", sampling.repetition_penalty},
        {"enable_thinking", sampling.enable_thinking},
        {"enable_vision", sampling.enable_vision},
        {"enable_mtp", sampling.enable_mtp},
        {"mtp_max_draft_tokens", sampling.mtp_max_draft_tokens},
    };
}

static std::vector<std::string> parse_stops(const json & body) {
    std::vector<std::string> stops;
    if (!body.contains("stop") || body["stop"].is_null()) return stops;
    if (body["stop"].is_string()) {
        stops.push_back(body["stop"].get<std::string>());
    } else if (body["stop"].is_array()) {
        if (body["stop"].size() > 16) {
            throw ApiError(400, "invalid_request_error", "stop accepts at most 16 strings", "stop");
        }
        for (const auto & stop : body["stop"]) {
            if (!stop.is_string()) {
                throw ApiError(400, "invalid_request_error", "stop entries must be strings", "stop");
            }
            stops.push_back(stop.get<std::string>());
        }
    } else {
        throw ApiError(400, "invalid_request_error", "stop must be a string or an array of strings", "stop");
    }
    for (const auto & stop : stops) {
        if (stop.empty()) throw ApiError(400, "invalid_request_error", "stop strings cannot be empty", "stop");
    }
    return stops;
}


bool valid_mfq_session_id(const std::string & session_id) {
    return !session_id.empty() && session_id.size() <= 128 &&
        std::all_of(
            session_id.begin(), session_id.end(),
            [](unsigned char value) {
                return std::isalnum(value) != 0 || value == '-' ||
                    value == '_' || value == '.' || value == ':';
            });
}

mfq::engine::CountTokens parse_token_count_request(const json& body) {
    if (!body.is_object() || body.size() != 1 || !body.contains("prompt") ||
        !body["prompt"].is_string() || body["prompt"].get_ref<const std::string&>().empty() ||
        body["prompt"].get_ref<const std::string&>().size() > 262144)
        throw ApiError(400, "invalid_request_error", "token count requires a nonempty prompt of at most 262144 bytes");
    return {body["prompt"].get<std::string>()};
}

json token_count_json(const mfq::engine::TokenCount& result) {
    return {{"prompt_tokens", result.prompt_tokens}, {"context_size", result.context_size},
        {"tokenization", "raw-no-special-tokens"}};
}

mfq::engine::ScoreText parse_score_request(const json& body) {
    if (!body.is_object() || !body.contains("prompt") || !body["prompt"].is_string() ||
        body["prompt"].get_ref<const std::string&>().empty() || body["prompt"].get_ref<const std::string&>().size() > 262144 ||
        !body.contains("continuations") || !body["continuations"].is_array() || body["continuations"].empty() || body["continuations"].size() > 64)
        throw ApiError(400, "invalid_request_error", "score requires a prompt and 1 to 64 continuations");
    if (body.contains("mode") && !body["mode"].is_string())
        throw ApiError(400, "invalid_request_error", "score mode must be a string", "mode");
    for (const auto& item : body.items())
        if (item.key() != "prompt" && item.key() != "continuations" && item.key() != "mode")
            throw ApiError(400, "invalid_request_error", "unknown score parameter", item.key());
    const auto mode = body.value("mode", std::string("continuation"));
    if (mode != "continuation" && mode != "next_token")
        throw ApiError(400, "invalid_request_error", "score mode must be continuation or next_token", "mode");
    mfq::engine::ScoreText request;
    request.prompt = body["prompt"].get<std::string>();
    request.next_token = mode == "next_token";
    for (const auto& item : body["continuations"]) {
        if (!item.is_string() || item.get_ref<const std::string&>().empty() || item.get_ref<const std::string&>().size() > 65536)
            throw ApiError(400, "invalid_request_error", "score continuations must be nonempty bounded strings", "continuations");
        request.continuations.push_back(item.get<std::string>());
    }
    return request;
}

json likelihood_result_json(const mfq::engine::LikelihoodResult& result) {
    json scores = json::array();
    for (const auto& score : result.scores)
        scores.push_back({{"token_ids", score.token_ids}, {"token_logprobs", score.token_logprobs}, {"log_likelihood", score.log_likelihood}});
    return {{"prompt_tokens", result.prompt_tokens}, {"scores", scores}, {"log_base", "e"}, {"tokenization", "raw-no-special-tokens"}};
}

json runtime_generate_body(const json & params) {
    if (!params.is_object()) {
        throw ApiError(
            400, "invalid_request_error",
            "runtime generation parameters must be a JSON object");
    }
    if (!params.contains("input") || !params["input"].is_object()) {
        throw ApiError(
            400, "invalid_request_error",
            "runtime generation requires an input object", "input");
    }
    const auto & input = params["input"];
    json body = json::object();
    if (params.contains("model")) body["model"] = params["model"];
    if (input.contains("messages")) body["messages"] = input["messages"];
    if (input.contains("preformatted_prompt")) {
        body["mfq_preformatted_prompt"] = input["preformatted_prompt"];
    }
    if (input.contains("benchmark_prompt_tokens")) {
        body["mfq_benchmark_prompt_tokens"] = input["benchmark_prompt_tokens"];
    }
    if (params.contains("cache")) {
        if (!params["cache"].is_object()) {
            throw ApiError(400, "invalid_request_error", "cache must be an object", "cache");
        }
        if (params["cache"].contains("enabled"))
            body["mfq_prefix_cache_enabled"] = params["cache"]["enabled"];
    }
    if (params.contains("sampling")) {
        if (!params["sampling"].is_object()) {
            throw ApiError(
                400, "invalid_request_error",
                "sampling must be an object", "sampling");
        }
        const auto & sampling = params["sampling"];
        if (sampling.contains("max_new_tokens")) {
            body["max_tokens"] = sampling["max_new_tokens"];
        }
        constexpr std::array<const char *, 10> fields{
            "temperature", "top_k", "top_p", "presence_penalty",
            "frequency_penalty", "repetition_penalty", "enable_vision",
            "enable_mtp", "mtp_max_draft_tokens", "seed",
        };
        for (const char * field : fields) {
            if (sampling.contains(field)) body[field] = sampling[field];
        }
    }
    if (params.contains("stream")) body["stream"] = params["stream"];
    if (params.contains("include_usage")) {
        body["stream_options"] = {
            {"include_usage", params["include_usage"]},
        };
    }
    if (params.contains("template")) {
        body["chat_template_kwargs"] = params["template"];
    }
    if (params.contains("output")) {
        if (!params["output"].is_object()) {
            throw ApiError(
                400, "invalid_request_error",
                "output must be an object", "output");
        }
        if (params["output"].contains("reasoning_format")) {
            body["reasoning_format"] =
                params["output"]["reasoning_format"];
        }
    }
    constexpr std::array<const char *, 3> constraint_fields{
        "tools", "tool_choice", "response_format",
    };
    for (const char * field : constraint_fields) {
        if (params.contains(field)) body[field] = params[field];
    }
    if (params.contains("session_id")) {
        body["mfq_session_id"] = params["session_id"];
    }
    if (params.contains("media")) body["mfq_multimodal"] = params["media"];
    return body;
}

RequestInput parse_input(
        const json & body,
        bool chat,
        const MfqSamplingParams & defaults) {
    if (!body.is_object()) throw ApiError(400, "invalid_request_error", "request body must be a JSON object");
    if (integer_field(body, "n", 1) != 1) {
        throw ApiError(400, "unsupported_parameter", "only n=1 is supported", "n");
    }
    if (body.contains("logprobs") && !body["logprobs"].is_null()) {
        if (!body["logprobs"].is_boolean()) {
            throw ApiError(400, "invalid_request_error", "logprobs must be boolean", "logprobs");
        }
        if (body["logprobs"].get<bool>()) {
            throw ApiError(400, "unsupported_parameter", "logprobs are not implemented", "logprobs");
        }
    }
    if (number_field(body, "min_p", 0.0) != 0.0) {
        throw ApiError(400, "unsupported_parameter", "min_p is not implemented", "min_p");
    }

    RequestInput work;
    work.chat = chat;
    if (body.contains("stream") && !body["stream"].is_null() && !body["stream"].is_boolean()) {
        throw ApiError(400, "invalid_request_error", "stream must be boolean", "stream");
    }
    work.stream = body.contains("stream") && !body["stream"].is_null()
        ? body["stream"].get<bool>()
        : false;
    if (body.contains("stream_options") && !body["stream_options"].is_null()) {
        if (!body["stream_options"].is_object()) {
            throw ApiError(400, "invalid_request_error", "stream_options must be an object", "stream_options");
        }
        const auto & stream_options = body["stream_options"];
        if (stream_options.contains("include_usage") && !stream_options["include_usage"].is_null() &&
            !stream_options["include_usage"].is_boolean()) {
            throw ApiError(400, "invalid_request_error", "include_usage must be boolean",
                           "stream_options.include_usage");
        }
        work.include_usage = stream_options.contains("include_usage") && !stream_options["include_usage"].is_null()
            ? stream_options["include_usage"].get<bool>()
            : false;
    }

    const int64_t max_tokens = body.contains("max_completion_tokens")
        ? integer_field(body, "max_completion_tokens", defaults.max_tokens)
        : integer_field(body, "max_tokens", defaults.max_tokens);
    if (max_tokens < 1 || max_tokens > std::numeric_limits<int32_t>::max()) {
        throw ApiError(400, "invalid_request_error", "max_tokens must be positive", "max_tokens");
    }
    work.sampling.max_tokens = static_cast<int32_t>(max_tokens);
    work.sampling.temperature = number_field(
        body, "temperature", defaults.temperature);
    work.sampling.top_p = number_field(body, "top_p", defaults.top_p);
    const int64_t top_k = integer_field(body, "top_k", defaults.top_k);
    work.sampling.presence_penalty = number_field(
        body, "presence_penalty", defaults.presence_penalty);
    work.sampling.frequency_penalty = number_field(
        body, "frequency_penalty", defaults.frequency_penalty);
    work.sampling.repetition_penalty = number_field(
        body, "repetition_penalty", defaults.repetition_penalty);
    work.sampling.enable_thinking = request_enable_thinking(
        body, defaults.enable_thinking);
    work.sampling.enable_vision = boolean_field(
        body, "enable_vision", defaults.enable_vision);
    work.sampling.enable_mtp = boolean_field(
        body, "enable_mtp", defaults.enable_mtp);
    const int64_t mtp_max_draft_tokens = integer_field(
        body,
        "mtp_max_draft_tokens",
        defaults.mtp_max_draft_tokens);
    if (work.sampling.temperature < 0.0 || work.sampling.temperature > 10.0) {
        throw ApiError(400, "invalid_request_error", "temperature must be in [0, 10]", "temperature");
    }
    if (work.sampling.top_p <= 0.0 || work.sampling.top_p > 1.0) {
        throw ApiError(400, "invalid_request_error", "top_p must be in (0, 1]", "top_p");
    }
    if (top_k < 0 || top_k > 1024) {
        throw ApiError(400, "invalid_request_error", "top_k must be in [0, 1024]", "top_k");
    }
    if (mtp_max_draft_tokens < 1 || mtp_max_draft_tokens > 7) {
        throw ApiError(
            400,
            "invalid_request_error",
            "mtp_max_draft_tokens must be in [1, 7]",
            "mtp_max_draft_tokens");
    }
    work.sampling.top_k = static_cast<int32_t>(top_k);
    work.sampling.mtp_max_draft_tokens = static_cast<int32_t>(mtp_max_draft_tokens);
    if (work.sampling.temperature > 0.0 && work.sampling.top_k == 0 && work.sampling.top_p < 1.0) {
        throw ApiError(400, "invalid_request_error", "top_p below 1 requires top_k above 0 in this sampler", "top_p");
    }
    if (work.sampling.presence_penalty < -2.0 || work.sampling.presence_penalty > 2.0) {
        throw ApiError(400, "invalid_request_error", "presence_penalty must be in [-2, 2]", "presence_penalty");
    }
    if (work.sampling.frequency_penalty < -2.0 || work.sampling.frequency_penalty > 2.0) {
        throw ApiError(400, "invalid_request_error", "frequency_penalty must be in [-2, 2]", "frequency_penalty");
    }
    if (work.sampling.repetition_penalty <= 0.0 || work.sampling.repetition_penalty > 10.0) {
        throw ApiError(400, "invalid_request_error", "repetition_penalty must be in (0, 10]", "repetition_penalty");
    }
    if (body.contains("seed") && !body["seed"].is_null()) {
        const int64_t seed = integer_field(body, "seed", 0);
        work.sampling.seed = static_cast<uint64_t>(seed);
    } else {
        std::random_device device;
        work.sampling.seed = (static_cast<uint64_t>(device()) << 32) ^ device();
    }

    if (chat) {
        work.chat_input = parse_chat_input(
            body, work.sampling.enable_thinking);
    } else {
        if (!body.contains("prompt") || !body["prompt"].is_string()) {
            throw ApiError(
                400, "invalid_request_error",
                "prompt must be a string", "prompt");
        }
        work.prompt = body["prompt"].get<std::string>();
    }
    if (body.contains("mfq_session_id") && !body["mfq_session_id"].is_null()) {
        if (!body["mfq_session_id"].is_string()) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_session_id must be a string", "mfq_session_id");
        }
        work.cache_plan.session_id =
            body["mfq_session_id"].get<std::string>();
        if (!valid_mfq_session_id(work.cache_plan.session_id)) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_session_id must contain 1 to 128 safe identifier bytes",
                "mfq_session_id");
        }
    }
    work.stops = parse_stops(body);
    work.prefix_cache_enabled = boolean_field(body, "mfq_prefix_cache_enabled", true);
    if (body.contains("mfq_benchmark_prompt_tokens")) {
        const auto count = integer_field(body, "mfq_benchmark_prompt_tokens", 0);
        if (count < 1 || count > 1048576) {
            throw ApiError(400, "invalid_request_error", "benchmark prompt tokens must be in [1, 1048576]", "mfq_benchmark_prompt_tokens");
        }
        if (work.prefix_cache_enabled || !work.chat_input || !work.chat_input->preformatted_prompt) {
            throw ApiError(400, "invalid_request_error", "benchmark token sizing requires a preformatted prompt and disabled prefix cache");
        }
        work.benchmark_prompt_tokens = static_cast<std::size_t>(count);
    }
    return work;
}


RequestMetricValues request_metric_values(
        const CompletionResult & result, const RequestMetrics & metrics) {
    const auto finished = RequestMetrics::Clock::now();
    RequestMetricValues values;
    values.prefill_tokens = metrics.saw_prefill
        ? metrics.prefill_tokens
        : 0;
    values.generation_ms =
        std::chrono::duration<double, std::milli>(finished - metrics.started).count();
    values.ttft_ms = metrics.saw_token
        ? std::chrono::duration<double, std::milli>(
              metrics.first_token - metrics.started).count()
        : values.generation_ms;
    values.prefill_ms = metrics.saw_prefill ? metrics.prefill_ms : 0.0;
    values.multimodal_ms = metrics.saw_prefill
        ? metrics.multimodal_ms
        : 0.0;
    values.model_prefill_ms = metrics.saw_prefill
        ? metrics.model_prefill_ms
        : 0.0;
    values.prefill_tps = metrics.saw_prefill && metrics.prefill_ms > 0.0
        ? 1000.0 * metrics.prefill_tokens / metrics.prefill_ms
        : 0.0;
    values.decode_ms = metrics.saw_token
        ? std::chrono::duration<double, std::milli>(
              metrics.last_token - metrics.first_token).count()
        : 0.0;
    values.generation_tps = values.generation_ms > 0.0
        ? 1000.0 * result.completion_tokens / values.generation_ms
        : 0.0;
    values.decode_tps =
        result.completion_tokens > 1 && values.decode_ms > 0.0
        ? 1000.0 * (result.completion_tokens - 1) / values.decode_ms
        : 0.0;
    return values;
}

json usage_json(size_t prompt_tokens, int32_t completion_tokens) {
    return {
        {"prompt_tokens", prompt_tokens},
        {"completion_tokens", completion_tokens},
        {"total_tokens", prompt_tokens + static_cast<size_t>(completion_tokens)},
    };
}

static json chat_tool_calls_json(
    const std::vector<common_chat_tool_call> & tool_calls);

json runtime_generation_event(
        const std::string & event,
        const std::string & request_id,
        int64_t created,
        const std::string & model) {
    return {
        {"event", event},
        {"request_id", request_id},
        {"created", created},
        {"model", model},
    };
}

json runtime_generation_result(
        const std::string & request_id,
        int64_t created,
        const std::string & model,
        const CompletionResult & result,
        json usage,
        json metrics) {
    json output = {{"text", result.text}};
    if (!result.reasoning_text.empty()) {
        output["reasoning"] = result.reasoning_text;
    }
    if (!result.tool_calls.empty()) {
        output["tool_calls"] = chat_tool_calls_json(result.tool_calls);
    }
    return {
        {"request_id", request_id},
        {"created", created},
        {"model", model},
        {"output", std::move(output)},
        {"finish_reason", result.finish_reason},
        {"usage", std::move(usage)},
        {"metrics", std::move(metrics)},
    };
}

json chat_diff_json(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call = {{"index", diff.tool_call_index}};
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"] = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        if (!diff.tool_call_delta.name.empty() ||
            !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                function["arguments"] =
                    diff.tool_call_delta.arguments;
            }
            tool_call["function"] = std::move(function);
        }
        delta["tool_calls"] =
            json::array({std::move(tool_call)});
    }
    return delta;
}

void apply_tool_call_delta(std::vector<common_chat_tool_call> & calls, const common_chat_msg_diff & diff) {
    if (calls.size() <= diff.tool_call_index) calls.resize(diff.tool_call_index + 1);
    auto & call = calls[diff.tool_call_index];
    if (!diff.tool_call_delta.id.empty()) call.id = diff.tool_call_delta.id;
    if (!diff.tool_call_delta.name.empty()) call.name = diff.tool_call_delta.name;
    call.arguments += diff.tool_call_delta.arguments;
}

std::optional<std::string> identity_fragment(std::string & previous, const std::string & value) {
    if (value.empty()) return std::string{};
    if (value.compare(0, previous.size(), previous) != 0) return {};
    auto fragment = value.substr(previous.size());
    previous = value;
    return fragment;
}

static json chat_tool_calls_json(
        const std::vector<common_chat_tool_call> & tool_calls) {
    json out = json::array();
    for (const auto & tool_call : tool_calls) {
        out.push_back({
            {"id", tool_call.id},
            {"type", "function"},
            {"function", {
                {"name", tool_call.name},
                {"arguments", tool_call.arguments},
            }},
        });
    }
    return out;
}


} // namespace mfq::protocol

MfqSamplingParams default_sampling_params(
        const MfqRuntimeProfile & runtime_profile) {
    MfqSamplingParams defaults;
    const auto & profile = runtime_profile.chat;
    if (profile.max_tokens) defaults.max_tokens = *profile.max_tokens;
    if (profile.temperature) defaults.temperature = *profile.temperature;
    if (profile.top_k) defaults.top_k = *profile.top_k;
    if (profile.top_p) defaults.top_p = *profile.top_p;
    if (profile.presence_penalty) {
        defaults.presence_penalty = *profile.presence_penalty;
    }
    if (profile.frequency_penalty) {
        defaults.frequency_penalty = *profile.frequency_penalty;
    }
    if (profile.repetition_penalty) {
        defaults.repetition_penalty = *profile.repetition_penalty;
    }
    if (profile.enable_thinking) {
        defaults.enable_thinking = *profile.enable_thinking;
    }
    if (profile.enable_vision) defaults.enable_vision = *profile.enable_vision;
    if (profile.enable_mtp) defaults.enable_mtp = *profile.enable_mtp;
    if (profile.mtp_max_draft_tokens) {
        defaults.mtp_max_draft_tokens = *profile.mtp_max_draft_tokens;
    }
    return defaults;
}

namespace mfq::protocol {


CompletionStream::CompletionStream(const MfqScheduler& scheduler, RequestWork work, std::string id)
    : cancel_([&scheduler, id] { scheduler.cancel_request(id); }), streaming_(work.stream),
      request_(scheduler.submit(mfq::engine::EngineRequest{std::move(id), std::move(work)})) {}
CompletionStream::CompletionStream(std::shared_ptr<MfqScheduledRequest> request, bool streaming, std::function<void()> cancel)
    : cancel_(std::move(cancel)), streaming_(streaming), request_(std::move(request)) {}
CompletionStream::~CompletionStream() {
    if (!terminal_) try { cancel(); } catch (...) {}
}
void CompletionStream::cancel() {
    cancel_();
    result.client_connected = false;
}
std::optional<std::vector<common_chat_msg_diff>> CompletionStream::next() {
    if (terminal_) return {};
    std::vector<common_chat_msg_diff> diffs;
    for (auto& event : request_->wait_for(std::chrono::milliseconds(100))) {
        std::visit([&](auto& data) {
            using T = std::decay_t<decltype(data)>;
            if constexpr (std::is_same_v<T, mfq::engine::OutputDelta>) {
                if (!data.token_ids.empty()) {
                    if (!generated_tokens_) first_token_ = std::chrono::steady_clock::now();
                    generated_tokens_ += data.token_ids.size();
                }
                for (const auto& diff : data.diffs) {
                    if (streaming_) continue;
                    result_bytes_ += diff.content_delta.size() + diff.reasoning_content_delta.size() +
                        diff.tool_call_delta.id.size() + diff.tool_call_delta.name.size() + diff.tool_call_delta.arguments.size();
                    if (result_bytes_ > 16 * 1024 * 1024) {
                        cancel();
                        throw ApiError(413, "output_limit", "collected output exceeds 16 MiB; use streaming");
                    }
                    result.text += diff.content_delta;
                    result.reasoning_text += diff.reasoning_content_delta;
                    if (diff.tool_call_index != std::string::npos)
                        apply_tool_call_delta(result.tool_calls, diff);
                }
                diffs.insert(diffs.end(), std::make_move_iterator(data.diffs.begin()),
                             std::make_move_iterator(data.diffs.end()));
            } else if constexpr (std::is_same_v<T, mfq::engine::PrefillProgress>) {
                const auto & timing = data.timing;
                const double interval = timing.llm_ms - prefill_ms_;
                if (interval > 0 && timing.prompt_tokens >= processed_tokens_)
                    prefill_tps_ = 1000.0 * (timing.prompt_tokens - processed_tokens_) / interval;
                processed_tokens_ = timing.prompt_tokens;
                prefill_ms_ = timing.llm_ms;
            } else if constexpr (std::is_same_v<T, mfq::engine::Completed> ||
                                 std::is_same_v<T, mfq::engine::Cancelled>) {
                result.completion_tokens = static_cast<std::int32_t>(data.usage.completion_tokens);
                prompt_tokens = data.usage.prompt_tokens;
                result.cancelled = std::is_same_v<T, mfq::engine::Cancelled>;
                if constexpr (std::is_same_v<T, mfq::engine::Completed>) result.finish_reason = std::move(data.finish_reason);
                else result.finish_reason = "cancelled";
                metrics = data.metrics; terminal_ = true;
            } else if constexpr (std::is_same_v<T, mfq::engine::Failed>) {
                terminal_ = true;
                if (data.code == "invalid_request") throw ApiError(400, "invalid_request_error", data.message);
                if (data.code == "unsupported_input") throw ApiError(501, "unsupported_parameter", data.message);
                if (data.code == "resource_exhausted") throw ApiError(503, "resource_exhausted", data.message);
                throw ApiError(500, data.code, data.message);
            }
        }, event.data);
    }
    return diffs;
}

json CompletionStream::progress() const {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - started_).count();
    const double decode = generated_tokens_ ? std::chrono::duration<double>(now - first_token_).count() : 0;
    return {{"phase", generated_tokens_ ? "generating" : "prefilling"},
        {"processed_tokens", processed_tokens_}, {"generated_tokens", generated_tokens_},
        {"prefill_tps", prefill_tps_}, {"elapsed_seconds", elapsed},
        {"decode_tps", decode > 0 ? (generated_tokens_ - 1) / decode : 0},
        {"updated_at", unix_time_seconds()}};
}

} // namespace mfq::protocol
