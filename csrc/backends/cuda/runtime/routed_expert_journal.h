#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

struct RoutedExpertTouch {
    std::int64_t expert=0;
    std::uint64_t uses=0;
    bool missed=false;
};

// Residency must stay fixed during collect(). Reverse traversal records each
// expert's last access first and evaluates its miss state only once. Reversing
// the result when committing preserves the original final LRU order.
class RoutedExpertJournal {
    struct Index {std::uint64_t epoch=0;std::size_t offset=0;};
    std::vector<Index> index_;
    std::uint64_t epoch_=0;
    template<class Id,class Missed>
    void collect_ids(std::span<const Id> ids,std::size_t experts,
            std::vector<RoutedExpertTouch>& touches,Missed&& missed) {
        if(index_.size()<experts)index_.resize(experts);
        if(!++epoch_) {
            std::fill(index_.begin(),index_.end(),Index{});
            epoch_=1;
        }
        touches.clear();touches.reserve(std::min(ids.size(),experts));
        for(auto it=ids.rbegin();it!=ids.rend();++it) {
            const auto expert=*it;
            auto offset=touches.size();
            if(expert>=0 && std::uint64_t(expert)<experts) {
                auto& entry=index_[std::size_t(expert)];
                if(entry.epoch==epoch_)offset=entry.offset;
                else {entry.epoch=epoch_;entry.offset=offset;}
            } else {
                // Preserve arbitrary-ID behavior without indexing scratch
                // outside the bank's actual expert geometry.
                const auto found=std::find_if(touches.begin(),touches.end(),
                    [&](const auto& item){return item.expert==expert;});
                if(found!=touches.end())offset=std::size_t(found-touches.begin());
            }
            if(offset==touches.size())touches.push_back({expert,1,missed(expert)});
            else ++touches[offset].uses;
        }
    }
public:
    template<class Missed>
    void collect(std::span<const std::int64_t> ids,std::size_t experts,
            std::vector<RoutedExpertTouch>& touches,Missed&& missed) {
        collect_ids(ids,experts,touches,std::forward<Missed>(missed));
    }
    template<class Missed>
    void collect(std::span<const std::int32_t> ids,std::size_t experts,
            std::vector<RoutedExpertTouch>& touches,Missed&& missed) {
        collect_ids(ids,experts,touches,std::forward<Missed>(missed));
    }
};

// A journal can be replayed without repeating route-to-expert lookups. Keep
// individual float increments for token heat, including its rounding at large
// scores. Explicit unique/normalized modes apply once per collected expert.
inline std::size_t apply_routed_expert_heat(std::span<const RoutedExpertTouch> touches,
        std::span<float> observations,std::optional<int> legacy_mode={}) {
    if(legacy_mode && (*legacy_mode<0 || *legacy_mode>2))
        throw std::invalid_argument("invalid routed journal heat override");
    std::size_t uses=0;
    for(const auto& touch:touches) {
        if(touch.expert<0 || std::uint64_t(touch.expert)>=observations.size())
            throw std::out_of_range("routed journal heat expert is out of range");
        if(touch.uses>std::numeric_limits<std::size_t>::max()-uses)
            throw std::overflow_error("routed journal heat use count overflow");
        uses+=std::size_t(touch.uses);
    }
    const bool unique=legacy_mode && *legacy_mode!=0;
    const float weight=legacy_mode==2 && !touches.empty()?float(uses)/float(touches.size()):1.f;
    for(const auto& touch:touches) {
        auto& value=observations[std::size_t(touch.expert)];
        if(unique)value+=weight;
        else for(std::uint64_t i=0;i<touch.uses;++i)value+=1.f;
    }
    return unique?touches.size():uses;
}
