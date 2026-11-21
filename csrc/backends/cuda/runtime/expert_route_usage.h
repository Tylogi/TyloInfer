#pragma once
#include "routed_expert_selection.h"
#include "routed_expert_banks.h"
#include "routed_expert_heat.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mfq::cuda {
class ExpertRouteUsage {
    struct Entry {std::uint64_t key;float uses=0;std::size_t bytes=0;bool retained=false,resident=false;};
    struct Candidate {std::uint64_t key;float uses;std::size_t bytes;std::array<Entry*,3> entries{};unsigned count=0;};
    std::unordered_map<std::uint64_t,std::size_t> indices_;
    // Selection, decay and source resets traverse compact entries rather than
    // chasing one hash-table allocation per observed expert image.
    std::vector<Entry> entries_;
    std::vector<Candidate> ranked_;
    mutable std::vector<std::uint64_t> selected_;
    std::size_t selected_bytes_=0,selected_count_=0;
    mutable bool selected_dirty_=false;
    Entry& entry(std::uint64_t key) {
        const auto [index,inserted]=indices_.try_emplace(key,entries_.size());
        if(inserted) {
            try {entries_.push_back({key});}
            catch(...) {indices_.erase(index);throw;}
        }
        return entries_[index->second];
    }
public:
    void image(std::uint64_t key,std::size_t bytes) {entry(key).bytes=bytes;}
    void resident(std::uint64_t key,bool value) {entry(key).resident=value;}
    void prefill(std::uint64_t key,std::size_t uses,std::size_t rows) {
        if(!rows)return;
        // Convert prompt counts to the steady mass of the decay window.
        entry(key).uses=expert_prefill_usage(uses,rows);
    }
    void reset_prefill(std::uint64_t source) {
        for(auto& entry:entries_)if((entry.key>>32)==source)entry.uses=0;
    }
    void observe(std::uint64_t key) {entry(key).uses+=1.f;}
    float score(std::uint64_t key)const {
        const auto found=indices_.find(key);return found==indices_.end()?0.f:entries_[found->second].uses;
    }
    void decay(std::uint64_t windows) {
        const auto factor=expert_usage_decay(windows);
        for(auto& entry:entries_)entry.uses*=factor;
    }
    bool retained(std::uint64_t key)const {
        const auto found=indices_.find(key);return found!=indices_.end() && entries_[found->second].retained;
    }
    std::size_t retained_images()const noexcept {return selected_count_;}
    std::size_t retained_bytes()const noexcept {return selected_bytes_;}
    std::size_t measured_retained_bytes()const noexcept {
        std::size_t bytes=0;for(const auto& entry:entries_)if(entry.retained)bytes+=entry.bytes;return bytes;
    }
    const std::vector<std::uint64_t>& retained_keys()const {
        if(selected_dirty_) {
            selected_.clear();selected_.reserve(selected_count_);
            for(const auto& entry:entries_)if(entry.retained)selected_.push_back(entry.key);
            selected_dirty_=false;
        }
        return selected_;
    }
    std::size_t retained_resident_images()const noexcept {
        return std::count_if(entries_.begin(),entries_.end(),[](const auto& entry) {
            return entry.retained && entry.resident;
        });
    }
    std::pair<double,double> retained_route_mass()const noexcept {
        double total=0,retained=0;
        for(const auto& entry:entries_) {
            total+=entry.uses;
            if(entry.retained)retained+=entry.uses;
        }
        return {retained,total};
    }
    double unretained_route_mass_below(float limit)const noexcept {
        double mass=0;for(const auto& entry:entries_)if(!entry.retained && entry.uses<limit)mass+=entry.uses;
        return mass;
    }
    bool replace(std::uint64_t victim,std::uint64_t incoming,std::size_t budget,const RoutedExpertBanks& banks) {
        return (victim>>32)==(incoming>>32) && exchange(victim,incoming,budget,banks);
    }
    bool exchange(std::uint64_t victim,std::uint64_t incoming,std::size_t budget,const RoutedExpertBanks& banks) {
        return exchange(std::span<const std::uint64_t>(&victim,1),incoming,budget,banks);
    }
    std::size_t expert_bytes(std::uint64_t image,const RoutedExpertBanks& banks)const {
        std::size_t bytes=0;bool complete=true;
        banks.visit_expert(image,[&](auto key) {
            const auto found=indices_.find(key);
            if(found==indices_.end() || !entries_[found->second].bytes){complete=false;return;}
            const auto size=entries_[found->second].bytes;
            if(size>SIZE_MAX-bytes)throw std::overflow_error("routed expert image byte count");
            bytes+=size;
        });
        return complete?bytes:0;
    }
    bool exchange(std::span<const std::uint64_t> victims,std::uint64_t incoming,std::size_t budget,
            const RoutedExpertBanks& banks) {
        std::vector<Entry*> old;old.reserve(victims.size()*3);
        std::array<Entry*,3> next{};unsigned next_count=0;
        bool complete=true;std::size_t released=0,required=0;
        for(const auto victim:victims)banks.visit_expert(victim,[&](auto key) {
            const auto found=indices_.find(key);
            if(found==indices_.end() || !entries_[found->second].retained){complete=false;return;}
            auto* entry=&entries_[found->second];
            if(std::find(old.begin(),old.end(),entry)!=old.end()){complete=false;return;}
            if(entry->bytes>SIZE_MAX-released){complete=false;return;}
            old.push_back(entry);released+=entry->bytes;
        });
        banks.visit_expert(incoming,[&](auto key) {
            const auto found=indices_.find(key);
            if(found==indices_.end() || !entries_[found->second].bytes || entries_[found->second].retained) {
                complete=false;return;
            }
            auto* entry=&entries_[found->second];next[next_count++]=entry;
            if(entry->bytes>SIZE_MAX-required){complete=false;return;}required+=entry->bytes;
        });
        if(!complete || selected_bytes_<released || selected_bytes_-released>budget ||
                required>budget-(selected_bytes_-released))return false;
        for(auto* entry:old)entry->retained=false;
        for(unsigned i=0;i<next_count;++i)next[i]->retained=true;
        selected_bytes_=selected_bytes_-released+required;
        selected_count_=selected_count_-old.size()+next_count;selected_dirty_=true;
        return true;
    }
    void select(std::size_t budget,float resident_margin=0,const RoutedExpertBanks* banks=nullptr) {
        ranked_.clear();ranked_.reserve(entries_.size());
        selected_.clear();selected_.reserve(entries_.size());
        selected_bytes_=0;selected_count_=0;selected_dirty_=false;
        for(auto& entry:entries_)entry.retained=false;
        for(auto& entry:entries_) {
            if(!(entry.uses>0) || !entry.bytes)continue;
            const auto* group=banks?banks->find(entry.key>>32):nullptr;
            if(!group) {
                ranked_.push_back({entry.key,entry.uses+(entry.resident?resident_margin:0),entry.bytes,{&entry},1});
                continue;
            }
            if((entry.key>>32)!=(*group)[0])continue;
            // Admission buys a complete expert, matching the GGUF cache unit.
            // Encoded projection sizes remain independent and are not padded.
            Candidate candidate{entry.key,entry.uses,0};
            bool complete=true,resident=true;
            for(const auto source:*group) {
                const auto key=(source<<32)|std::uint32_t(entry.key);
                const auto found=indices_.find(key);
                if(found==indices_.end() || !entries_[found->second].bytes){complete=false;break;}
                auto* member=&entries_[found->second];
                if(std::find(candidate.entries.begin(),candidate.entries.begin()+candidate.count,member)!=
                        candidate.entries.begin()+candidate.count)continue;
                if(member->bytes>SIZE_MAX-candidate.bytes)throw std::overflow_error("routed expert group byte count");
                candidate.bytes+=member->bytes;resident&=member->resident;
                candidate.entries[candidate.count++]=member;
            }
            if(complete) {
                if(resident)candidate.uses+=resident_margin;
                ranked_.push_back(candidate);
            }
        }
        select_routed_experts(ranked_,budget,[](const auto& item){return item.uses;},
            [](const auto& item){return item.key;},[](const auto& item){return item.bytes;},
            [&](const auto& item){for(unsigned i=0;i<item.count;++i) {
                item.entries[i]->retained=true;selected_.push_back(item.entries[i]->key);
                selected_bytes_+=item.entries[i]->bytes;++selected_count_;
            }});
    }
};
}
