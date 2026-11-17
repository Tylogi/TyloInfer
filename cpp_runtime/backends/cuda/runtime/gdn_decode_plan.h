#pragma once
#include <cstdint>
#include <stdexcept>

namespace mfq::cuda {
// Two CTAs per SM hide state-load latency. Additional column splits repeat
// convolution, normalization and tile publication without adding useful
// parallelism once independent heads already cover that working set.
inline int gdn_decode_tiles(std::uint64_t heads,int processors) {
    if(!heads || processors<=0)throw std::invalid_argument("invalid GDN parallel work geometry");
    const auto target=std::uint64_t(processors)*2;
    int tiles=4;
    while(tiles>1 && heads>=(target+tiles/2-1)/(tiles/2))tiles/=2;
    return tiles;
}
} // namespace mfq::cuda
