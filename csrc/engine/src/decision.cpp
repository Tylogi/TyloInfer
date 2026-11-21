#include "decision.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace mfq::engine {

double decision_logsumexp(std::span<const double> logits) {
    if(logits.empty())throw std::invalid_argument("decision readout has no logits");
    for(const auto value:logits)if(!std::isfinite(value))
        throw std::runtime_error("decision readout produced a nonfinite logit");
    const auto peak=*std::max_element(logits.begin(),logits.end());
    double total=0;
    for(const auto value:logits)total+=std::exp(value-peak);
    return peak+std::log(total);
}

DecisionReadResult normalize_decision(std::span<const std::int64_t> labels,
        std::span<const double> logits,std::optional<double> log_normalizer,
        std::optional<std::int64_t> argmax) {
    if(labels.empty() || labels.size()!=logits.size())
        throw std::invalid_argument("decision label and logit counts disagree");
    std::unordered_set<std::int64_t> unique;
    for(const auto label:labels)if(label<0 || !unique.insert(label).second)
        throw std::invalid_argument("decision label tokens must be distinct and nonnegative");
    const auto normalizer=decision_logsumexp(logits);
    DecisionReadResult result;result.token_ids.assign(labels.begin(),labels.end());
    const auto peak=*std::max_element(logits.begin(),logits.end());
    double scaled_sum=0;
    for(const auto value:logits) {result.probabilities.push_back(std::exp(value-peak));scaled_sum+=result.probabilities.back();}
    for(auto& probability:result.probabilities)probability/=scaled_sum;
    const auto largest=*std::max_element(result.probabilities.begin(),result.probabilities.end());
    result.confidence=largest;
    if(log_normalizer) {
        if(!std::isfinite(*log_normalizer) || *log_normalizer+1e-7<normalizer)
            throw std::runtime_error("decision vocabulary normalizer is inconsistent with its labels");
        result.label_mass=std::min(1.,std::exp(normalizer-*log_normalizer));
        result.confidence*=*result.label_mass;
    }
    if(argmax)result.argmax_is_label=unique.contains(*argmax);
    return result;
}

} // namespace mfq::engine
