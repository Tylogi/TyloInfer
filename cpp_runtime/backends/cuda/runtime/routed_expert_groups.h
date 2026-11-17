#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mfq::cuda {
// Preserve first-use expert order and original routing slots. Every ID is
// validated, including IDs outside a caller's local expert partition.
inline void group_routed_experts(std::span<const std::int64_t> ids,std::int64_t experts,
        std::vector<std::int64_t>& unique,std::vector<std::vector<std::int64_t>>* slots=nullptr,
        std::int64_t begin=0,std::int64_t end=std::numeric_limits<std::int64_t>::max()) {
    if(end==std::numeric_limits<std::int64_t>::max())end=experts;
    if(experts<1 || begin<0 || begin>end || end>experts || ids.size()>std::uint64_t(INT64_MAX))
        throw std::invalid_argument("invalid routed expert grouping geometry");
    unique.clear();if(slots)slots->clear();
    const auto local=[&](std::int64_t id) {
        if(id<0 || id>=experts)throw std::out_of_range("routed expert ID is out of bounds");
        return id>=begin && id<end;
    };
    if(!slots) {
        for(const auto id:ids)if(local(id) && std::find(unique.begin(),unique.end(),id)==unique.end())
            unique.push_back(id);
        return;
    }
    if(std::uint64_t(experts)<=ids.size()) {
        // Dense counters remain bounded by the route list. Reserving each
        // group once avoids reallocating long prefill groups on every append.
        std::vector<std::size_t> positions(std::size_t(experts),0);
        for(const auto id:ids)if(local(id) && !positions[id]++)unique.push_back(id);
        slots->resize(unique.size());
        for(std::size_t i=0;i<unique.size();++i) {
            auto& count=positions[unique[i]];(*slots)[i].reserve(count);count=i;
        }
        for(std::size_t i=0;i<ids.size();++i)if(ids[i]>=begin && ids[i]<end)
            (*slots)[positions[ids[i]]].push_back(std::int64_t(i));
    }else {
        std::unordered_map<std::int64_t,std::size_t> positions;
        positions.reserve(ids.size());
        for(std::size_t i=0;i<ids.size();++i)if(local(ids[i])) {
            const auto [position,inserted]=positions.emplace(ids[i],unique.size());
            if(inserted){unique.push_back(ids[i]);slots->emplace_back();}
            (*slots)[position->second].push_back(std::int64_t(i));
        }
    }
}
} // namespace mfq::cuda
