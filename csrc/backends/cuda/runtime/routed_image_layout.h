#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mfq::cuda {
inline std::size_t routed_aligned_bytes(std::size_t bytes,std::size_t alignment=16) {
    if(!alignment || (alignment&(alignment-1)) || bytes>std::size_t(INT64_MAX)-(alignment-1))
        throw std::overflow_error("routed image alignment exceeds tensor limits");
    return (bytes+alignment-1)&~(alignment-1);
}
struct RoutedExpertImageLayout {
    std::array<std::size_t,3> offsets{},sizes{};
    std::size_t bytes=0;
    RoutedExpertImageLayout(const std::array<std::uint64_t,3>& sources,
            const std::array<std::size_t,3>& encoded,std::size_t alignment=16) {
        for(unsigned role=0;role<3;++role) {
            sizes[role]=routed_aligned_bytes(encoded[role],alignment);
            bool alias=false;
            for(unsigned previous=0;previous<role;++previous)if(sources[previous]==sources[role]) {
                if(encoded[previous]!=encoded[role])throw std::invalid_argument("aliased expert image sizes differ");
                offsets[role]=offsets[previous];alias=true;break;
            }
            if(alias)continue;
            offsets[role]=bytes;
            if(sizes[role]>std::size_t(INT64_MAX)-bytes)throw std::overflow_error("routed expert image exceeds tensor limits");
            bytes+=sizes[role];
        }
    }
};
}
