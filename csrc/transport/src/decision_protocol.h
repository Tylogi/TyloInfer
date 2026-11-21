#pragma once

#include "common.h"

namespace mfq::transport_detail {

struct DecisionQuestion {
    std::string id,type;
    std::vector<std::string> options,legend;
};
struct DecisionRequest {
    mfq::engine::DecisionText input;
    std::vector<DecisionQuestion> questions;
    std::string model;
};
DecisionRequest parse_decision_request(const nlohmann::ordered_json& body,bool allow_conditional = false);
json decision_result_json(const DecisionRequest& request,const mfq::engine::DecisionResult& result,
    const std::string& default_model);

} // namespace mfq::transport_detail
