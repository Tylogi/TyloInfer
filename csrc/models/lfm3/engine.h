#pragma once
#include "input.h"
#include "nlohmann/json.hpp"
#include <memory>

namespace mfq::models::lfm3::d1 {
class Engine {
public:
    Engine(Input& input, bool fused_projections, bool stream_weights = false);
    ~Engine();
    nlohmann::json ready() const;
    nlohmann::json evaluate(const nlohmann::json& request);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
