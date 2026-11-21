#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mfq::cuda {
// Encoding-independent compact order shared by transfer and compute plans.
// Bounds address valid routes only; original slots preserve reduction order.
struct RoutedPrefillRoutes {
    std::vector<std::int64_t> bounds{0},slots;
    template<class Group>
    RoutedPrefillRoutes(std::size_t count,Group group) {
        bounds.reserve(count+1);
        for(std::size_t i=0;i<count;++i) {
            const auto& values=group(i);
            if(values.empty() || values.size()>std::uint64_t(INT64_MAX-bounds.back()))
                throw std::invalid_argument("invalid compact prefill route count");
            for(const auto slot:values) {
                if(slot<0)throw std::out_of_range("negative compact prefill slot");
                slots.push_back(slot);
            }
            bounds.push_back(std::int64_t(slots.size()));
        }
    }
    std::int64_t rows(std::size_t begin,std::size_t end)const {
        if(begin>end || end>=bounds.size())throw std::out_of_range("compact prefill group bounds");
        return bounds[end]-bounds[begin];
    }
};
} // namespace mfq::cuda
