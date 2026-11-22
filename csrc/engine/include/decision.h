#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mfq::engine {

struct DecisionTextRead {
    std::string prompt;
    std::vector<std::string> labels;
    std::string state, instructions;
    std::vector<std::string> option_texts;
    bool binary = false;
    std::string type = "choice";
};
struct DecisionText {
    std::vector<DecisionTextRead> reads;
    std::map<std::string,std::string> chat_template_kwargs;
    std::string cache_salt;
    bool prefix_cache = true;
    bool full_vocabulary = true;
};
struct DecisionTokenRead {
    std::vector<std::int64_t> prompt, labels;
    std::size_t state_tokens = 0;
    std::string type = "choice";
};
struct DecisionTokens {
    std::vector<DecisionTokenRead> reads;
    std::string cache_salt;
    bool prefix_cache = true;
    bool full_vocabulary = true;
};
struct DecisionReadResult {
    std::vector<std::int64_t> token_ids;
    std::vector<double> probabilities;
    std::size_t prompt_tokens = 0, cached_tokens = 0;
    double prefill_ms = 0.0;
    double confidence = 0;
    std::optional<double> label_mass;
    std::optional<bool> argmax_is_label;
};
struct DecisionResult {
    std::vector<DecisionReadResult> reads;
};

// Probabilities are conditional on the supplied labels. A full-vocabulary
// normalizer additionally gives the unconditioned confidence and label mass.
DecisionReadResult normalize_decision(std::span<const std::int64_t> labels,
    std::span<const double> logits,std::optional<double> log_normalizer = {},
    std::optional<std::int64_t> argmax = {});
double decision_logsumexp(std::span<const double> logits);

} // namespace mfq::engine
