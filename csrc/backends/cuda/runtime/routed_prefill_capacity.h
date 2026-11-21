#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace mfq::cuda {
// Frequency-ordered groups never pad a member below half the leading count.
// The operator's adapter supplies its actual image capacity.
template<class Count>
std::vector<std::size_t> routed_prefill_group_ends(std::size_t size,std::size_t capacity,Count count) {
    if(!capacity)throw std::invalid_argument("prefill group capacity must be positive");
    std::vector<std::size_t> ends;
    for(std::size_t begin=0;begin<size;) {
        auto end=begin+std::min(capacity,size-begin);
        const auto leading=count(begin);
        const auto minimum=leading/2+leading%2;
        for(auto next=begin+1;next<end;++next)if(count(next)<minimum){end=next;break;}
        ends.push_back(end);begin=end;
    }
    return ends;
}

// Common reusable image limits. Zero decoded bytes means the operator consumes
// encoded weights directly and never materializes a decoded weight image.
inline std::size_t routed_prefill_group_capacity(std::size_t experts,long double encoded,long double decoded,
        std::size_t requested=0,std::size_t host_bytes=std::numeric_limits<std::size_t>::max()) {
    if(!experts || !std::isfinite(encoded) || !std::isfinite(decoded) || encoded<=0 || decoded<0)
        throw std::invalid_argument("invalid prefill image capacity");
    auto capacity=std::max<std::size_t>(1,std::size_t(std::min({65535.L,static_cast<long double>(experts),
        (64ULL<<20)/encoded,decoded>0?(256ULL<<20)/decoded:static_cast<long double>(experts)})));
    if(requested)capacity=std::min(capacity,requested);
    // Two CPU payloads may be in use by the corresponding upload stages.
    if(host_bytes!=std::numeric_limits<std::size_t>::max())capacity=std::size_t(std::min(
        static_cast<long double>(capacity),std::max(1.L,host_bytes/2/encoded)));
    return capacity;
}

// Sum projection/cohort fields for the same logical expert before finding the
// peak image. Separate instances keep backbone and predictor budgets independent.
struct RoutedPrefillImageGeometry {
    std::map<int,std::vector<long double>> layers;
    std::size_t experts=0;
    long double encoded=0;
    void add(int layer,std::size_t expert,long double bytes) {
        if(!std::isfinite(bytes) || bytes<0 || expert==std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("invalid routed prefill expert geometry");
        auto& images=layers[layer];
        if(images.size()<=expert)images.resize(expert+1);
        images[expert]+=bytes;
        experts=std::max(experts,images.size());encoded=std::max(encoded,images[expert]);
    }
};

// Adapters provide their actual, monotonic arena layout, including all stages,
// alignment and any dummy cohort slots. Do not price unused decoded images.
template<class WorkspaceBytes>
std::size_t routed_prefill_workspace_capacity(std::size_t capacity,std::size_t available,WorkspaceBytes bytes) {
    std::size_t low=0,high=capacity;
    while(low<high) {
        const auto span=high-low;
        const auto candidate=low+span/2+span%2;
        const auto required=bytes(candidate);
        bool fits=false;
        if constexpr(std::is_integral_v<decltype(required)>) {
            if constexpr(std::is_signed_v<decltype(required)>)if(required<0)
                throw std::overflow_error("invalid routed prefill workspace size");
            fits=static_cast<std::uintmax_t>(required)<=static_cast<std::uintmax_t>(available);
        } else {
            if(!std::isfinite(required) || required<0)
                throw std::overflow_error("invalid routed prefill workspace size");
            fits=required<=static_cast<long double>(available);
        }
        if(fits)low=candidate;
        else high=candidate-1;
    }
    return low;
}
} // namespace mfq::cuda
