#pragma once

#include "config.h"
#include <set>

namespace mfq::models::lfm3::d1 {

struct Request {
    std::vector<int> tokens;
    std::vector<int> groups;
    std::vector<int> slots;
    std::vector<int> candidates;
    int state_tokens;
    int iterations;
    int warmup;

    static Request parse(const nlohmann::json& document, const Config& config) {
        if (document.value("use_cache", false) || document.contains("past_key_values"))
            throw std::invalid_argument("d1 native execution does not reuse prefix/KV/conv state");
        Request request{
            document.at("tokens").get<std::vector<int>>(),
            document.at("question_ids").get<std::vector<int>>(),
            document.at("slots").get<std::vector<int>>(),
            document.at("candidate_ids").get<std::vector<int>>(),
            document.at("state_tokens").get<int>(),
            document.value("iterations", 1),
            document.value("warmup", 0),
        };
        const auto count = request.tokens.size();
        if (count == 0 || count > std::size_t(config.context) || count != request.groups.size()
                || request.state_tokens < 0 || std::size_t(request.state_tokens) >= count
                || request.slots.empty() || request.candidates.empty())
            throw std::invalid_argument("invalid native d1 input geometry");
        for (std::size_t i = 0; i < count; ++i) {
            if (request.tokens[i] < 0 || request.tokens[i] >= config.vocabulary
                    || (i >= std::size_t(request.state_tokens) && request.groups[i] <= 0))
                throw std::invalid_argument("invalid native d1 token/group");
        }
        for (int slot : request.slots)
            if (slot < 0 || std::size_t(slot) >= count)
                throw std::invalid_argument("d1 answer slot outside input");
        std::set<int> unique;
        for (int token : request.candidates)
            if (token < 0 || token >= config.vocabulary || !unique.insert(token).second)
                throw std::invalid_argument("invalid native d1 candidates");
        if (request.iterations < 1 || request.iterations > 1000 || request.warmup < 0 || request.warmup > 100)
            throw std::invalid_argument("invalid native d1 iteration count");
        return request;
    }
};

} // namespace mfq::models::lfm3::d1
