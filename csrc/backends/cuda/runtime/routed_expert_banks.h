#pragma once
#include <array>
#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mfq::cuda {
// A projection shared by different expert groups has independent ownership.
// Ordinary disjoint Gate/Up/Down groups admit and evict as one expert.
class RoutedExpertBanks {
public:
    using Group=std::array<std::uint64_t,3>;
private:
    std::unordered_map<std::uint64_t,Group> owners_;
    std::unordered_set<std::uint64_t> shared_;
public:
    RoutedExpertBanks()=default;
    explicit RoutedExpertBanks(std::span<const Group> groups){for(const auto& group:groups)add(group);}
    void add(const Group& group) {
        for(const auto source:group)if(source>UINT32_MAX)throw std::out_of_range("routed source exceeds encoded identifier range");
        std::array<std::uint64_t,3> added{},marked{};unsigned additions=0,marks=0;
        try {
            for(const auto source:group) {
                const auto [owner,inserted]=owners_.try_emplace(source,group);
                if(inserted)added[additions++]=source;
                else if(owner->second!=group && shared_.insert(source).second)marked[marks++]=source;
            }
        }catch(...) {
            while(marks)shared_.erase(marked[--marks]);
            while(additions)owners_.erase(added[--additions]);
            throw;
        }
    }
    const Group* find(std::uint64_t source)const {
        const auto owner=owners_.find(source);
        if(owner==owners_.end())return nullptr;
        for(const auto member:owner->second)if(shared_.count(member))return nullptr;
        return &owner->second;
    }
    std::vector<Group> groups()const {
        std::vector<Group> result;
        for(const auto& [source,group]:owners_)if(source==group[0] && find(source))result.push_back(group);
        std::sort(result.begin(),result.end());return result;
    }
    template<class Visit> void visit_expert(std::uint64_t image,Visit&& visit)const {
        const auto* banks=find(image>>32);
        if(!banks){visit(image);return;}
        for(unsigned role=0;role<banks->size();++role) {
            bool repeated=false;
            for(unsigned prior=0;prior<role;++prior)repeated|=(*banks)[prior]==(*banks)[role];
            if(!repeated)visit(((*banks)[role]<<32)|std::uint32_t(image));
        }
    }
};
} // namespace mfq::cuda
