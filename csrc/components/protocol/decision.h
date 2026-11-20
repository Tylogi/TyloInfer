#pragma once

#include "protocol.h"

namespace tyloi::protocol {

struct DecisionQuestion {
    std::string id, type;
    std::vector<std::string> options, legend;
};
struct DecisionRequest {
    engine::DecisionText input;
    std::vector<DecisionQuestion> questions;
    std::string model;
};
DecisionRequest parse_decision_request(const json &body, bool allow_conditional = false);
json decision_result_json(const DecisionRequest &request,
                          const std::vector<engine::DecisionReadResult> &reads,
                          const std::string &default_model);

std::vector<engine::DecisionReadResult> collect_decisions(engine::Output &output,
                                                          const std::function<bool()> &disconnected = {},
                                                          const std::function<void()> &cancel = {});

} // namespace tyloi::protocol
