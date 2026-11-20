#include "decision.h"

#include <algorithm>
#include <set>
#include <string_view>

namespace tyloi::protocol {
namespace {
std::string text(const json &value, const std::string &field, std::size_t maximum, bool nonempty = false) {
    if (!value.is_string() || value.get_ref<const std::string &>().size() > maximum ||
        (nonempty && value.get_ref<const std::string &>().empty()))
        throw ApiError(400, "invalid_request_error", "invalid decision text field", field);
    return value.get<std::string>();
}
void fields(const json &value, const std::set<std::string> &allowed, const std::string &parent) {
    if (!value.is_object())
        throw ApiError(400, "invalid_request_error", "decision parameters must be objects", parent);
    for (const auto &item : value.items())
        if (!allowed.contains(item.key()))
            throw ApiError(400, "invalid_request_error", "unsupported decision parameter",
                           parent + item.key());
}
} // namespace

DecisionRequest parse_decision_request(const json &body, bool allow_conditional) {
    std::set<std::string> allowed{
        "model", "state", "questions", "instructions", "chat_template_kwargs", "cache_salt", "joint"};
    if (allow_conditional) {
        allowed.insert("full_vocabulary");
        allowed.insert("prefix_cache");
    }
    fields(body, allowed, "");
    if (!body.contains("state") || !body.contains("questions") || !body["questions"].is_object() ||
        body["questions"].empty() || body["questions"].size() > 64)
        throw ApiError(400, "invalid_request_error",
                       "decisions require a state and 1 to 64 named questions");
    if (body.contains("joint") && (!body["joint"].is_boolean() || body["joint"].get<bool>()))
        throw ApiError(400, "invalid_request_error", "joint decisions require a joint-read model", "joint");
    DecisionRequest request;
    if (body.contains("model"))
        request.model = text(body["model"], "model", 512, true);
    if (body.contains("cache_salt"))
        request.input.cache_salt = text(body["cache_salt"], "cache_salt", 256);
    for (const auto *key : {"full_vocabulary", "prefix_cache"})
        if (body.contains(key)) {
            if (!body[key].is_boolean())
                throw ApiError(400, "invalid_request_error", "decision switch must be boolean", key);
            if (std::string_view(key) == "full_vocabulary")
                request.input.full_vocabulary = body[key].get<bool>();
            else
                request.input.prefix_cache = body[key].get<bool>();
        }
    if (body.contains("chat_template_kwargs")) {
        const auto &kwargs = body["chat_template_kwargs"];
        if (!kwargs.is_object() || kwargs.dump().size() > 16384 || kwargs.contains("chat_template"))
            throw ApiError(400, "invalid_request_error", "invalid decision chat template arguments",
                           "chat_template_kwargs");
        for (const auto &item : kwargs.items())
            request.input.chat_template_kwargs[item.key()] = item.value().dump();
    }
    auto state = body["state"].is_string() ? body["state"].get<std::string>() : body["state"].dump();
    if (state.size() > 262144)
        throw ApiError(400, "invalid_request_error", "decision state exceeds the text limit", "state");
    std::string prefix;
    if (body.contains("instructions"))
        prefix = text(body["instructions"], "instructions", 65536) + "\n\n";
    prefix += "State:\n" + state + "\n\n";
    for (const auto &item : body["questions"].items()) {
        if (item.key().empty() || item.key().size() > 256)
            throw ApiError(400, "invalid_request_error",
                           "decision question names must be nonempty bounded strings", "questions");
        const auto field = "questions." + item.key() + ".";
        fields(item.value(), {"type", "instructions", "criteria"}, field);
        const auto &question = item.value();
        if (!question.contains("type"))
            throw ApiError(400, "invalid_request_error", "decision question type is required",
                           field + "type");
        DecisionQuestion descriptor;
        descriptor.id = item.key();
        descriptor.type = text(question["type"], field + "type", 16, true);
        engine::DecisionTextRead read;
        read.prompt = prefix;
        read.state = state;
        read.type = descriptor.type;
        if (question.contains("instructions"))
            read.instructions = text(question["instructions"], field + "instructions", 65536);
        if (question.contains("instructions"))
            read.prompt += "Question:\n" + read.instructions + "\n";
        read.prompt += "Options:\n";
        const auto criteria = question.value("criteria", json(nullptr));
        if (descriptor.type == "choice") {
            if (!criteria.is_object() || criteria.empty() || criteria.size() > 255)
                throw ApiError(400, "invalid_request_error", "choice requires 1 to 255 named options",
                               field + "criteria");
            for (const auto &option : criteria.items()) {
                if (option.key().empty() || option.key().size() > 512)
                    throw ApiError(400, "invalid_request_error",
                                   "choice option names must be nonempty bounded strings",
                                   field + "criteria");
                const auto number = read.labels.size();
                const auto label = number < 26 ? std::string(1, char('A' + number))
                                               : std::string(1, char('A' + (number - 26) / 26)) +
                                                     char('A' + (number - 26) % 26);
                read.labels.push_back(label);
                descriptor.options.push_back(option.key());
                read.prompt += label + ": " + option.key();
                const auto description = option.value().is_null()
                                             ? std::string{}
                                             : text(option.value(), field + "criteria", 16384);
                if (!option.value().is_null())
                    read.prompt += " - " + description;
                read.prompt += '\n';
                read.option_texts.push_back(
                    option.key() + (option.value().is_null() ? std::string{} : ": " + description));
            }
        } else if (descriptor.type == "noul") {
            if (!criteria.is_null())
                fields(criteria, {"true", "false"}, field + "criteria.");
            read.labels = {"yes", "no"};
            descriptor.options = read.labels;
            read.binary = true;
            for (int i = 0; i < 2; ++i) {
                read.prompt += read.labels[i];
                const auto *name = i ? "false" : "true";
                const bool described = criteria.contains(name) && !criteria[name].is_null();
                const auto description =
                    described ? text(criteria[name], field + "criteria." + name, 16384) : std::string{};
                if (described)
                    read.prompt += " - " + description;
                read.prompt += '\n';
                read.option_texts.push_back(described ? description : (i ? "No / false" : "Yes / true"));
            }
        } else if (descriptor.type == "score") {
            if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10)
                throw ApiError(400, "invalid_request_error", "score requires 2 to 10 levels",
                               field + "criteria");
            for (std::size_t i = 0; i < criteria.size(); ++i) {
                const auto label = std::to_string(i),
                           level = text(criteria[i], field + "criteria", 512, true);
                read.labels.push_back(label);
                descriptor.options.push_back(label);
                descriptor.legend.push_back(level);
                read.prompt += label + ": " + level + '\n';
                read.option_texts.push_back(level);
            }
        } else
            throw ApiError(400, "invalid_request_error", "decision type must be choice, noul or score",
                           field + "type");
        read.prompt += "Answer with exactly one option label.";
        if (read.prompt.size() > 262144)
            throw ApiError(400, "invalid_request_error", "decision read exceeds the text limit", field);
        request.input.reads.push_back(std::move(read));
        request.questions.push_back(std::move(descriptor));
    }
    return request;
}

json decision_result_json(const DecisionRequest &request,
                          const std::vector<engine::DecisionReadResult> &reads,
                          const std::string &default_model) {
    if (reads.size() != request.questions.size())
        throw std::runtime_error("decision response read count changed");
    json answers = json::object(), diagnostics = json::object();
    std::size_t tokens = 0, cached = 0;
    for (std::size_t i = 0; i < reads.size(); ++i) {
        const auto &question = request.questions[i];
        const auto &read = reads[i];
        if (read.probabilities.size() != question.options.size() || read.probabilities.empty())
            throw std::runtime_error("decision response option count changed");
        json probabilities = json::object(), answer{{"type", question.type}};
        for (std::size_t j = 0; j < question.options.size(); ++j)
            probabilities[question.options[j]] = read.probabilities[j];
        const auto winner = std::max_element(read.probabilities.begin(), read.probabilities.end()) -
                            read.probabilities.begin();
        if (question.type == "choice")
            answer["choice"] = question.options[winner];
        else if (question.type == "noul")
            answer["noul"] = read.probabilities[0];
        else {
            double score = 0;
            json legend = json::object();
            for (std::size_t j = 0; j < question.options.size(); ++j) {
                score += j * read.probabilities[j];
                legend[question.options[j]] = question.legend[j];
            }
            answer["score"] = score;
            answer["legend"] = std::move(legend);
        }
        answer["probabilities"] = std::move(probabilities);
        answer["confidence"] = read.confidence;
        answers[question.id] = std::move(answer);
        diagnostics[question.id] = {
            {"label_mass", read.label_mass ? json(*read.label_mass) : json(nullptr)},
            {"argmax_is_label", read.argmax_is_label ? json(*read.argmax_is_label) : json(nullptr)}};
        if (!read.label_mass)
            diagnostics[question.id]["normalization"] = "labels";
        diagnostics[question.id]["prefill_ms"] = read.prefill_ms;
        diagnostics[question.id]["prefill_tps"] =
            read.prefill_ms > 0 ? double(read.prompt_tokens - read.cached_tokens) * 1000 / read.prefill_ms
                                : 0.0;
        tokens += read.prompt_tokens;
        cached += read.cached_tokens;
    }
    return {{"id", request_id("decision")},
            {"object", "structured_decision"},
            {"created", unix_time_seconds()},
            {"model", request.model.empty() ? default_model : request.model},
            {"answers", std::move(answers)},
            {"usage",
             {{"input_tokens", tokens},
              {"output_tokens", 0},
              {"input_tokens_details", {{"cached_tokens", cached}}}}},
            {"diagnostics", std::move(diagnostics)}};
}

std::vector<engine::DecisionReadResult> collect_decisions(engine::Output &output,
                                                          const std::function<bool()> &disconnected,
                                                          const std::function<void()> &cancel) {
    std::vector<engine::DecisionReadResult> reads;
    for (;;) {
        if (disconnected && disconnected()) {
            if (cancel)
                cancel();
            throw ApiError(409, "request_cancelled", "decision request was cancelled");
        }
        for (auto &event : output.wait_for(std::chrono::milliseconds(100))) {
            if (auto *read = std::get_if<engine::DecisionReadResult>(&event.data))
                reads.push_back(std::move(*read));
            else if (std::holds_alternative<engine::Completed>(event.data))
                return reads;
            else if (std::holds_alternative<engine::Cancelled>(event.data))
                throw ApiError(409, "request_cancelled", "decision request was cancelled");
            else if (const auto *failure = std::get_if<engine::Failed>(&event.data))
                throw_request_failure(*failure);
        }
    }
}

} // namespace tyloi::protocol
