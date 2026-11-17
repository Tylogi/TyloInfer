#pragma once
#include "storage/moe_pinned_expert_arena.h"
#include <iterator>
#include <algorithm>
#include <string>
#include <utility>
#include <chrono>

// Callers hold the cache lock. Only idle GPU-backed duplicates may be
// released or moved to ordinary RAM; transfer leases keep in-flight data alive.
template<class Arenas,class Duplicates,class Idle,class Release>
std::shared_ptr<MoePinnedExpertArena::Block> reuse_pinned_expert_interval(
        const Arenas& arenas,Duplicates& duplicates,std::size_t bytes,Idle&& idle,Release&& release) {
    const auto allocate=[&]() -> std::shared_ptr<MoePinnedExpertArena::Block> {
        for(const auto& arena:arenas)if(auto block=arena->allocate(bytes))return block;
        return {};
    };
    if(auto block=allocate())return block;
    for(auto next=duplicates.end();next!=duplicates.begin();) {
        const auto previous=std::prev(next);
        if(!idle(*previous)){next=previous;continue;}
        const auto key=*previous;
        release(key);
        if(auto block=allocate())return block;
    }
    return {};
}

template<class Arenas,class Duplicates,class Idle,class Release,class Available,class Registered>
std::shared_ptr<MoePinnedExpertArena::Block> allocate_routed_pinned(
        Arenas& arenas,Duplicates& duplicates,std::size_t bytes,std::size_t limit,bool allow_growth,
        std::size_t& reserved,std::size_t& ceiling,std::uint64_t& failures,
        Idle&& idle,Release&& release,Available&& available_bytes,Registered&& registered,double* registration_ms=nullptr) {
    if(auto block=reuse_pinned_expert_interval(arenas,duplicates,bytes,
            std::forward<Idle>(idle),std::forward<Release>(release)))return block;
    if(!allow_growth)return {};
    auto registration=mfq::cuda::try_host_registration();
    if(!registration.owns_lock())return {};
    const auto capacity_limit=std::min(limit,ceiling);
    const auto remaining=capacity_limit-std::min(capacity_limit,reserved);
    if(bytes>remaining)return {};
    const auto available=available_bytes();
    const auto capacity=std::min(remaining,std::max(bytes,std::min(std::size_t(256)<<20,available/8)));
    if(capacity<bytes || capacity>available)return {};
    struct Timer {
        double* value;
        std::chrono::steady_clock::time_point start;
        ~Timer(){if(value)*value+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}
    } timer{registration_ms,registration_ms?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{}};
    try {
        auto arena=std::make_shared<MoePinnedExpertArena>(capacity);
        auto block=arena->allocate(bytes);arenas.push_back(std::move(arena));
        reserved+=capacity;registered(capacity);return block;
    }catch(const std::exception& error) {
        if(!std::string(error.what()).starts_with("CUDA pinned allocation failed:"))throw;
        ++failures;ceiling=reserved;(void)cudaGetLastError();return {};
    }
}
