#pragma once

#include <cstdint>

namespace mfq::cpu {

void qsa_indexer_topk(const float* queries, const std::int64_t* positions, int query_count, int heads,
                      int width, const std::int16_t* codes, const std::uint16_t* scales,
                      const std::int16_t* zero, int pools, int budget, int threads, std::int32_t* selected);

} // namespace mfq::cpu
