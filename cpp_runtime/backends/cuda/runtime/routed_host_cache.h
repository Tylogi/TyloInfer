#pragma once
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cstring>
#include <map>
#include <utility>
#include <chrono>

namespace mfq::cuda {
struct RoutedHostCacheState {};
struct RoutedHostEvacuationBlock {
    bool& flag;bool previous;
    explicit RoutedHostEvacuationBlock(bool& value):flag(value),previous(std::exchange(value,true)){}
    ~RoutedHostEvacuationBlock(){flag=previous;}
};
inline std::size_t routed_shared_host_limit(std::size_t ordinary,std::size_t used,
        std::size_t available,std::size_t requested) {
    if(!requested)return ordinary;
    const auto ceiling=available>=requested?
        used+std::min(ordinary-std::min(ordinary,used),available-requested):
        used-std::min(used,requested-available);
    return std::min(ordinary,ceiling);
}
inline std::size_t routed_host_evacuation_capacity(std::size_t limit,std::size_t used,
        std::size_t available,std::size_t reserve)noexcept {
    return std::min(limit-std::min(limit,used),available-std::min(available,reserve));
}
// Cache indices own encoded images. Transfer leases independently retain the
// same image, so eviction cannot invalidate an asynchronous DMA source.
template<class Image,class Key=std::uint64_t,class Hash=std::hash<Key>,class State=RoutedHostCacheState>
class RoutedHostCache {
protected:
    struct HostEntry:State {
        std::shared_ptr<Image> image;
        typename std::list<Key>::iterator recent;
        std::optional<typename std::list<Key>::iterator> duplicate,pinned_duplicate;
        HostEntry(std::shared_ptr<Image> value,typename std::list<Key>::iterator position,
                std::optional<typename std::list<Key>::iterator> copy={},
                std::optional<typename std::list<Key>::iterator> pinned={})
            :image(std::move(value)),recent(position),duplicate(copy),pinned_duplicate(pinned) {}
    };
    std::unordered_map<Key,HostEntry,Hash> host_;
    std::list<Key> host_recent_,host_duplicates_,pinned_duplicates_;
    std::size_t host_used_=0;
    std::chrono::steady_clock::time_point last_host_allocator_trim_{};
    // Recover idle allocator/row-cache pages before discarding live expert
    // images. Re-sample real headroom; a logical cache release is not credit.
    template<class Release,class Probe>
    std::size_t host_headroom_after_allocator_trim(std::size_t requested,std::size_t available,
            Release&& release,Probe&& probe,
            std::chrono::steady_clock::time_point now=std::chrono::steady_clock::now()) {
        if(available>=requested || now-last_host_allocator_trim_<std::chrono::seconds(1))return available;
        last_host_allocator_trim_=now;release();return probe();
    }
    struct HostAllocationVictim {
        const void* allocation=nullptr;
        std::size_t unique=0,payload=0,allocated=0,age=0;
        std::unordered_set<const void*> blocks;
        bool busy=false;
        std::vector<Key> keys;
    };
    template<class Eligible,class Ready>
    std::vector<Key> host_evacuation_candidates(const void* allocation,Eligible&& eligible,Ready&& ready)const {
        std::vector<Key> candidates;
        for(const auto& [key,entry]:host_) {
            const auto& image=entry.image;
            if(entry.duplicate || !image->block || image.use_count()!=1 ||
                    image->block->allocation_id()!=allocation || !eligible(key))continue;
            if(!ready(*image))continue;
            candidates.push_back(key);
        }
        return candidates;
    }
    template<class Arenas>
    std::vector<HostAllocationVictim> host_allocation_victims(const Arenas& arenas)const {
        std::unordered_map<const void*,HostAllocationVictim> allocations;
        std::size_t age=0;
        for(auto it=host_recent_.rbegin();it!=host_recent_.rend();++it,++age) {
            const auto& entry=host_.at(*it);const auto& image=entry.image;
            if(image->snapshot)continue;
            const auto id=image->block?image->block->allocation_id():static_cast<const void*>(image.get());
            auto& victim=allocations[id];victim.allocation=id;
            if(victim.keys.empty())victim.age=age;victim.keys.push_back(*it);
            if(image->block && victim.blocks.insert(image->block.get()).second)victim.allocated+=image->block->bytes();
            victim.payload+=image->size();
            if(!entry.duplicate)victim.unique+=image->size();
            victim.busy|=image.use_count()!=1;
        }
        for(const auto& arena:arenas)if(const auto found=allocations.find(arena->allocation_id());found!=allocations.end())
            found->second.busy|=arena->used_bytes()>found->second.allocated;
        std::vector<HostAllocationVictim> order;order.reserve(allocations.size());
        for(auto& [_,victim]:allocations)if(!victim.busy)order.push_back(std::move(victim));
        // Fully duplicated allocations precede unique RAM data. Whole
        // allocations avoid leaving unusable holes in many pinned arenas.
        std::stable_sort(order.begin(),order.end(),[](const auto& a,const auto& b) {
            if(a.unique!=b.unique)return a.unique<b.unique;
            if(a.payload!=b.payload)return a.payload<b.payload;
            return a.age<b.age;
        });
        return order;
    }
    // Caller holds the cache lock and excludes active graph/admission users.
    // Raw block pointers preserve the original exclusivity counts: only idle
    // leases wholly represented by this cache may move inside an arena.
    template<class Arenas,class Index,class Copied>
    bool compact_host_interval(const Arenas& arenas,const Index& keys_by_allocation,
            std::size_t bytes,Copied& copied) {
        using Arena=typename Arenas::value_type::element_type;
        using Block=typename decltype(std::declval<Image&>().block)::element_type;
        struct Lease {Block* block;std::size_t owners=0,references=0;bool idle=true;};
        std::vector<std::pair<std::size_t,Arena*>> candidates;
        for(const auto& arena:arenas) {
            const auto free=arena->bytes()-arena->used_bytes();
            if(free>=bytes)candidates.emplace_back(free,arena.get());
        }
        std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.first>b.first;});
        // Prefer an arena with spare capacity, then minimize moved bytes.
        std::vector<Block*> best;
        std::size_t best_bytes=SIZE_MAX;
        for(const auto& [free,arena]:candidates) {
            const auto keys=keys_by_allocation.find(arena->allocation_id());
            if(keys==keys_by_allocation.end())continue;
            std::unordered_map<Block*,Lease> leases;
            for(const auto key:keys->second) {
                const auto& image=host_.at(key).image;
                auto* block=image->block.get();
                auto [found,inserted]=leases.try_emplace(block,Lease{block});
                ++found->second.owners;
                found->second.references=image->block.use_count();
                found->second.idle&=image.use_count()==1;
            }
            std::vector<Lease*> blocks;blocks.reserve(leases.size());
            for(auto& [block,lease]:leases)blocks.push_back(&lease);
            std::size_t accounted=0;
            for(const auto* lease:blocks)accounted+=lease->block->bytes();
            if(accounted!=arena->used_bytes())continue;
            std::sort(blocks.begin(),blocks.end(),[](const auto* a,const auto* b){return a->block->offset()<b->block->offset();});
            std::vector<std::size_t> gaps;gaps.reserve(blocks.size()+1);
            std::size_t end=0;
            for(auto* lease:blocks) {
                gaps.push_back(lease->block->offset()-end);
                end=lease->block->offset()+lease->block->bytes();
                lease->idle&=lease->owners==lease->references;
            }
            gaps.push_back(arena->bytes()-end);
            std::size_t begin=0;
            while(begin<blocks.size()) {
                while(begin<blocks.size() && !blocks[begin]->idle)++begin;
                if(begin==blocks.size())break;
                auto finish=begin;
                while(finish<blocks.size() && blocks[finish]->idle)++finish;
                std::size_t left=begin,free=0,cost=0;
                for(auto right=begin;right<=finish;++right) {
                    free+=gaps[right];
                    if(right>begin)cost+=blocks[right-1]->block->bytes();
                    while(left<right && free-gaps[left]>=bytes) {
                        free-=gaps[left];cost-=blocks[left]->block->bytes();++left;
                    }
                    if(free>=bytes && cost<best_bytes) {
                        best_bytes=cost;best.clear();
                        for(auto i=left;i<right;++i)best.push_back(blocks[i]->block);
                    }
                }
                begin=finish;
            }
            if(!best.empty())break;
        }
        if(best.empty())return false;
        for(auto* block:best)copied+=block->compact_left();
        return true;
    }
    // A surviving image may own only part of an originally shared lease.
    // Idle, exclusive ownership preserves its address while releasing padding
    // and deleted siblings to this same arena's free-interval index.
    template<class Keys>
    std::size_t trim_host_image_ranges(const void* allocation,const Keys& keys) {
        std::size_t released=0;
        for(const auto key:keys)if(const auto found=host_.find(key);found!=host_.end()) {
            auto& image=found->second.image;
            if(image.use_count()!=1 || !image->block || image->block.use_count()!=1 || image->block_offset%16 ||
                    image->block->allocation_id()!=allocation)continue;
            if(const auto bytes=image->block->retain_range(image->block_offset,image->size())) {
                image->block_offset=0;released+=bytes;
            }
        }
        return released;
    }
    template<class Arenas,class Keys,class Moved>
    std::size_t relocate_host_images(const Arenas& arenas,const void* allocation,const Keys& keys,
            std::unordered_set<const void*>& retained,Moved&& moved) {
        using Arena=typename Arenas::value_type::element_type;
        using Block=typename decltype(std::declval<Image&>().block)::element_type;
        struct Move {Key key;Image* image;std::shared_ptr<Block> target;};
        std::vector<Move> moves;
        std::vector<std::pair<Key,Image*>> images;
        std::multimap<std::size_t,Arena*> targets;
        std::size_t available=0,required=0;
        for(const auto& arena:arenas)if(arena->allocation_id()!=allocation) {
            if(const auto largest=arena->largest_free_bytes())targets.emplace(largest,arena.get());
            available+=arena->bytes()-arena->used_bytes();
        }
        // Reserve only existing registered holes. A failed plan leaves all
        // source pointers and bytes unchanged and releases temporary leases.
        for(const auto key:keys) {
            const auto found=host_.find(key);if(found==host_.end())continue;
            const auto& image=found->second.image;
            if(!image->block || image->block->allocation_id()!=allocation)continue;
            if(image.use_count()!=1 || image->size()>SIZE_MAX-15)return 0;
            const auto bytes=(image->size()+15)&~std::size_t(15);
            if(targets.empty() || bytes>targets.rbegin()->first || bytes>available-required)return 0;
            required+=bytes;images.emplace_back(key,image.get());
        }
        // Place larger images first to preserve the smallest usable holes.
        std::sort(images.begin(),images.end(),[](const auto& a,const auto& b){return a.second->size()>b.second->size();});
        moves.reserve(images.size());
        for(const auto& [key,image]:images) {
            const auto found=targets.lower_bound((image->size()+15)&~std::size_t(15));
            if(found==targets.end())return 0;
            auto* arena=found->second;targets.erase(found);
            auto target=arena->allocate(image->size());
            if(!target)return 0;
            if(const auto largest=arena->largest_free_bytes())targets.emplace(largest,arena);
            moves.push_back({key,image,std::move(target)});
        }
        std::size_t copied=0;
        for(auto& move:moves) {
            std::memcpy(move.target->data(),move.image->data(),move.image->size());
            copied+=move.image->size();
            retained.insert(move.target->allocation_id());
            moved(move.key,allocation,move.target->allocation_id());
            move.image->block=std::move(move.target);move.image->block_offset=0;
        }
        return copied;
    }
    Key host_index_victim()const {
        if(host_recent_.empty())throw std::logic_error("empty host expert cache has no victim");
        return host_duplicates_.empty()?host_recent_.back():host_duplicates_.back();
    }
    void host_index_touch(Key key) {
        auto& entry=host_.at(key);
        host_recent_.splice(host_recent_.begin(),host_recent_,entry.recent);
        if(entry.duplicate)host_duplicates_.splice(host_duplicates_.begin(),host_duplicates_,*entry.duplicate);
        if(entry.pinned_duplicate)pinned_duplicates_.splice(pinned_duplicates_.begin(),pinned_duplicates_,*entry.pinned_duplicate);
    }
    void host_index_duplicate(Key key,bool duplicate) {
        const auto found=host_.find(key);if(found==host_.end())return;
        auto& entry=found->second;
        if(duplicate) {
            if(!entry.duplicate){host_duplicates_.push_front(key);entry.duplicate=host_duplicates_.begin();}
            const bool pinned=[&] {
                if constexpr(requires {entry.image->pinned();})return bool(entry.image->pinned());
                else return false;
            }();
            if(pinned && !entry.pinned_duplicate) {
                pinned_duplicates_.push_front(key);entry.pinned_duplicate=pinned_duplicates_.begin();
            }else if(!pinned && entry.pinned_duplicate) {
                pinned_duplicates_.erase(*entry.pinned_duplicate);entry.pinned_duplicate.reset();
            }
        }else {
            if(entry.duplicate){host_duplicates_.erase(*entry.duplicate);entry.duplicate.reset();}
            if(entry.pinned_duplicate){pinned_duplicates_.erase(*entry.pinned_duplicate);entry.pinned_duplicate.reset();}
        }
    }
    void host_index_insert(Key key,const std::shared_ptr<Image>& image,bool duplicate) {
        if(!image || image->size()>SIZE_MAX-host_used_)throw std::overflow_error("host expert cache byte count");
        const auto recent=host_recent_.insert(host_recent_.begin(),key);
        try {
            if(!host_.emplace(key,HostEntry{image,recent,{},{}}).second)
                throw std::logic_error("duplicate encoded expert host image");
        }catch(...) {host_recent_.erase(recent);throw;}
        host_used_+=image->size();
        try{host_index_duplicate(key,duplicate);}
        catch(...){host_index_erase(key);throw;}
    }
    void host_index_erase(Key key) {
        const auto found=host_.find(key);if(found==host_.end())return;
        auto& entry=found->second;
        host_used_-=entry.image->size();host_recent_.erase(entry.recent);
        if(entry.duplicate)host_duplicates_.erase(*entry.duplicate);
        if(entry.pinned_duplicate)pinned_duplicates_.erase(*entry.pinned_duplicate);
        host_.erase(found);
    }
};
} // namespace mfq::cuda
