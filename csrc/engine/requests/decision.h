#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace tyloi::engine {

struct DecisionTextRead {
    std::string prompt, state, instructions, type;
    std::vector<std::string> labels, option_texts;
    bool binary = false;
};

struct DecisionText {
    std::vector<DecisionTextRead> reads;
    std::map<std::string, std::string> chat_template_kwargs;
    std::string cache_salt;
    bool prefix_cache = true, full_vocabulary = true;
};

struct DecisionTokenRead {
    std::vector<std::int64_t> prompt, labels;
    std::int64_t bidirectional_prefix = 0;
    std::string type;
};

struct DecisionTokens {
    std::vector<DecisionTokenRead> reads;
    std::string cache_salt;
    bool prefix_cache, full_vocabulary;
};

struct DecisionReadResult {
    std::vector<double> probabilities;
    std::size_t prompt_tokens = 0, cached_tokens = 0;
    double prefill_ms = 0.0, confidence = 0.0;
    std::optional<double> label_mass;
    std::optional<bool> argmax_is_label;
};

} // namespace tyloi::engine
