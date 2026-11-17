#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace mfq::cuda {
// Transfer cost and resident bytes grow together. Reuse frequency therefore
// ranks saved transfer time per occupied byte; size enforces the capacity.
template<class Candidates,class Score,class Key,class Bytes,class Retain>
void select_routed_experts(Candidates& candidates,std::size_t budget,Score score,Key key,Bytes bytes,Retain retain) {
    auto minimum=std::numeric_limits<std::size_t>::max();
    for(const auto& item:candidates)minimum=std::min(minimum,bytes(item));
    const auto lower=[&](const auto& a,const auto& b) {
        return score(a)!=score(b)?score(a)<score(b):key(a)>key(b);
    };
    if(budget>=minimum)std::make_heap(candidates.begin(),candidates.end(),lower);
    while(!candidates.empty() && budget>=minimum) {
        std::pop_heap(candidates.begin(),candidates.end(),lower);
        auto item=std::move(candidates.back());candidates.pop_back();
        const auto count=bytes(item);
        if(count && count<=budget){budget-=count;retain(std::move(item));}
    }
    candidates.clear();
}
} // namespace mfq::cuda
