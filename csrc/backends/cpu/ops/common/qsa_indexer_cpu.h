#pragma once

#include <cstdint>

namespace mfq::cpu {

void qsa_indexer_topk(const float* queries, const std::int64_t* positions, int query_count, int heads,
                      int width, const std::int16_t* codes, const std::uint16_t* scales,
                      const std::int16_t* zero, int pools, int budget, int threads, std::int32_t* selected);

// Records store signed codes followed by FP16 scale bits and a signed zero point.
void index_pool_affine16_topk(const float* queries,const float* head_weights,const std::int64_t* positions,
    int query_count,int heads,int width,const std::int16_t* records,int pools,int pool_size,
    bool reserve_tail,int count,int threads,std::int32_t* selected);

} // namespace mfq::cpu
