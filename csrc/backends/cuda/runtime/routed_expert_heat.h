#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

namespace mfq::cuda {
inline constexpr std::uint64_t expert_usage_interval=4;
inline constexpr float expert_usage_retention=.7f;
inline constexpr float expert_replacement_margin=1.5f;
inline float expert_usage_decay(std::uint64_t windows) {
    return windows==expert_usage_interval?expert_usage_retention:
        std::pow(expert_usage_retention,float(windows)/float(expert_usage_interval));
}
inline bool routed_admission_due(std::uint64_t windows,bool frequent=false)noexcept {
    return frequent || windows%expert_usage_interval==0;
}
inline std::size_t routed_observed_rows(std::size_t committed,std::size_t executed)noexcept {
    return committed?executed:0;
}
inline float expert_prefill_usage(std::uint64_t uses,std::size_t rows)noexcept {
    return rows?float(uses)*float(expert_usage_interval)/(.3f*float(rows)):0.f;
}

// Default cache learning accounts for all executed verifier rows after a
// successful commit. Explicit legacy modes retain accepted-prefix learning:
// 0 counts tokens, 1 counts unique experts, 2 normalizes the unique heat.
template<class Id>
std::size_t routed_speculative_heat(std::span<const Id> ids,std::size_t rows,
        std::size_t committed,std::span<float> observations,std::span<std::uint64_t> seen,
        std::optional<int> legacy_mode={}) {
    if(committed>rows || (!rows && !ids.empty()) || (rows && ids.size()%rows))
        throw std::invalid_argument("speculative heat rows are not aligned");
    if(legacy_mode && (*legacy_mode<0 || *legacy_mode>2))
        throw std::invalid_argument("invalid speculative heat override");
    const auto learned=legacy_mode?committed:routed_observed_rows(committed,rows);
    const auto count=rows?learned*(ids.size()/rows):0;
    for(std::size_t i=0;i<count;++i)
        if(ids[i]<0 || std::uint64_t(ids[i])>=observations.size())
            throw std::out_of_range("speculative heat expert is out of range");
    const bool unique=legacy_mode && *legacy_mode!=0;
    if(!unique) {
        for(std::size_t i=0;i<count;++i)observations[std::size_t(ids[i])]+=1.f;
        return count;
    }
    if(seen.size()<(observations.size()+63)/64)
        throw std::invalid_argument("speculative heat scratch is too small");
    std::fill(seen.begin(),seen.end(),std::uint64_t(0));
    std::size_t distinct=0;
    for(std::size_t i=0;i<count;++i) {
        const auto expert=std::size_t(ids[i]);auto& word=seen[expert/64];
        const auto bit=std::uint64_t(1)<<(expert%64);
        if(!(word&bit)){word|=bit;++distinct;}
    }
    const float weight=*legacy_mode==2 && distinct?float(count)/float(distinct):1.f;
    for(std::size_t i=0;i<count;++i) {
        const auto expert=std::size_t(ids[i]);auto& word=seen[expert/64];
        const auto bit=std::uint64_t(1)<<(expert%64);
        if(word&bit){word&=~bit;observations[expert]+=weight;}
    }
    return distinct;
}
} // namespace mfq::cuda
