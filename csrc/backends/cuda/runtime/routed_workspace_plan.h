#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>

namespace mfq::cuda {
// Fixed source footprints keep resource shares stable across cache resizing.
// Allocate each adapter's real minimum before redistributing the remainder.
template<std::size_t N>
std::array<std::size_t,N> routed_budget_shares(std::size_t bytes,
        const std::array<std::size_t,N>& weights,const std::array<std::size_t,N>& minima={}) {
    auto available=bytes;
    for(const auto minimum:minima) {
        if(minimum>available)throw std::runtime_error("routed cache budget is below its live minimum");
        available-=minimum;
    }
    std::array<std::size_t,N> result{};
    std::array<bool,N> fixed{};
    auto remaining=bytes;
    for(std::size_t i=0;i<N;++i)if(!weights[i]) {
        result[i]=minima[i];remaining-=minima[i];fixed[i]=true;
    }
    for(;;) {
        long double total=0;
        std::size_t last=N;
        for(std::size_t i=0;i<N;++i)if(!fixed[i]){total+=weights[i];last=i;}
        if(last==N)return result;
        auto unassigned=remaining;
        for(std::size_t i=0;i<N;++i)if(!fixed[i]) {
            const auto proportional=static_cast<long double>(remaining)*weights[i]/total;
            result[i]=i==last || proportional>=static_cast<long double>(unassigned)?unassigned:
                static_cast<std::size_t>(proportional);
            unassigned-=result[i];
        }
        bool constrained=false;
        for(std::size_t i=0;i<N;++i)if(!fixed[i] && result[i]<minima[i]) {
            result[i]=minima[i];remaining-=minima[i];fixed[i]=true;constrained=true;
        }
        if(!constrained)return result;
    }
}

inline std::size_t routed_saturating_add(std::size_t a,std::size_t b)noexcept {
    const auto maximum=std::numeric_limits<std::size_t>::max();return b>maximum-a?maximum:a+b;
}

// Captured allocations are already charged. Price transient operator storage,
// driver residency and any additional transfer scratch using the GGUF plan.
inline std::size_t routed_decode_workspace_reserve(std::size_t growth,std::optional<std::size_t> transient,
        std::optional<std::size_t> configured,std::size_t transfer_scratch=0)noexcept {
    const auto base=configured.value_or(routed_saturating_add(
        std::max(transient.value_or(std::size_t(1280)<<20),std::size_t(700)<<20),transfer_scratch));
    return routed_saturating_add(base,growth);
}

inline std::size_t routed_cache_capacity(std::size_t physical,std::size_t reusable,std::size_t available,
        std::size_t reserve)noexcept {
    const auto total=routed_saturating_add(routed_saturating_add(physical,reusable),available);
    return total-std::min(total,reserve);
}
} // namespace mfq::cuda
