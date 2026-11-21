#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include "expert_route_usage.h"

namespace mfq::cuda {
struct AdaptiveSlotSwap {float gain;std::int32_t layer,in,out;};
struct AdaptiveImageSwap {std::uint64_t incoming;std::vector<std::uint64_t> victims;};
struct RoutedImageCandidate {float score;std::uint64_t key;std::size_t bytes;};
struct AdaptiveImageSwapStats {std::uint64_t proposed=0,capacity_rejected=0,heat_rejected=0,multi_victim=0;};

// Variable-size images may need several cold experts to admit one hot expert.
// Charge their combined heat before committing an atomic byte-budget exchange.
inline std::vector<AdaptiveImageSwap> adaptive_image_swaps(ExpertRouteUsage& usage,
        std::vector<RoutedImageCandidate>& candidates,std::vector<RoutedImageCandidate>& victims,
        std::size_t budget,const RoutedExpertBanks& banks,std::size_t cap=96,AdaptiveImageSwapStats* stats=nullptr) {
    const auto higher=[](const auto& a,const auto& b) {
        return a.score!=b.score?a.score>b.score:a.key<b.key;
    };
    std::sort(candidates.begin(),candidates.end(),higher);
    std::sort(victims.begin(),victims.end(),[&](const auto& a,const auto& b){return higher(b,a);});
    std::vector<AdaptiveImageSwap> swaps;std::vector<std::uint64_t> removed;
    std::size_t first=0;
    for(const auto& candidate:candidates) {
        if(swaps.size()==cap)break;
        if(stats)++stats->proposed;
        removed.clear();float heat=0;
        auto available=budget-std::min(budget,usage.retained_bytes());
        for(auto i=first;i<victims.size() && available<candidate.bytes;++i) {
            heat+=victims[i].score;
            if(candidate.score<heat+expert_replacement_margin)break;
            available+=std::min(victims[i].bytes,SIZE_MAX-available);
            removed.push_back(victims[i].key);
        }
        if(candidate.score<heat+expert_replacement_margin){if(stats)++stats->heat_rejected;continue;}
        if(available<candidate.bytes){if(stats)++stats->capacity_rejected;continue;}
        if(usage.exchange(removed,candidate.key,budget,banks)) {
            first+=removed.size();swaps.push_back({candidate.key,removed});
            if(stats && removed.size()>1)++stats->multi_victim;
        }
    }
    return swaps;
}
template<class Candidates,class Victims,class Emit>
// Fixed-slot ranking from Strata generate.cpp (MIT; see third_party/strata.LICENSE).
// Its selection rule and defaults remain unchanged.
void rank_routed_replacements(Candidates& candidates,Victims& victims,Emit&& emit) {
    if(candidates.empty() || victims.empty())return;
    std::sort(candidates.begin(),candidates.end(),[](auto a,auto b){return a.first>b.first;});
    const auto count=std::min(candidates.size(),victims.size());
    std::partial_sort(victims.begin(),victims.begin()+std::ptrdiff_t(count),victims.end(),
        [](auto a,auto b){return a.first<b.first;});
    for(std::size_t i=0;i<count;++i) {
        if(candidates[i].first<victims[i].first+expert_replacement_margin)break;
        emit(candidates[i].first-victims[i].first,candidates[i].second,victims[i].second);
    }
}
template<class Resident>
std::vector<AdaptiveSlotSwap> adaptive_slot_swaps(
        const std::vector<std::vector<float>>& usage,Resident resident,std::size_t cap=96) {
    std::vector<AdaptiveSlotSwap> swaps;
    std::vector<std::pair<float,std::int32_t>> candidates,victims;
    for(std::size_t layer=0;layer<usage.size();++layer) {
        candidates.clear();victims.clear();
        const auto& counts=usage[layer];
        for(std::size_t expert=0;expert<counts.size();++expert) {
            if(!resident(layer,expert)) {if(counts[expert]>=2.f)candidates.emplace_back(counts[expert],std::int32_t(expert));}
            else victims.emplace_back(counts[expert],std::int32_t(expert));
        }
        rank_routed_replacements(candidates,victims,[&](auto gain,auto incoming,auto victim) {
            swaps.push_back({gain,std::int32_t(layer),incoming,victim});
        });
    }
    std::sort(swaps.begin(),swaps.end(),[](auto a,auto b){return a.gain>b.gain;});
    if(swaps.size()>cap)swaps.resize(cap);
    return swaps;
}
}
