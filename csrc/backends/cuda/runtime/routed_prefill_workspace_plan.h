#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mfq::cuda {
namespace routed_prefill_workspace_detail {
inline std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if(b>std::numeric_limits<std::uint64_t>::max()-a)
        throw std::overflow_error("prefill workspace byte count overflow");
    return a+b;
}
inline std::uint64_t multiply(std::uint64_t a,std::uint64_t b) {
    if(a && b>std::numeric_limits<std::uint64_t>::max()/a)
        throw std::overflow_error("prefill workspace byte count overflow");
    return a*b;
}
inline std::uint64_t align16(std::uint64_t bytes) {return add(bytes,15)&~std::uint64_t(15);}
} // namespace routed_prefill_workspace_detail

// FP16 projection geometry and transient buffers are independent of the
// encoded expert image. Output width may differ from the input width.
struct RoutedPrefillWorkspaceShape {
    std::uint64_t input,hidden,output;
    std::uint64_t activation_bytes(std::uint64_t rows,bool gemm)const {
        using namespace routed_prefill_workspace_detail;
        if(!input || !hidden || !output)throw std::invalid_argument("empty prefill projection geometry");
        const auto width=std::max(input,hidden);
        const auto input_bytes=align16(multiply(multiply(rows,width),2));
        if(gemm)return add(input_bytes,multiply(multiply(rows,std::max(output,multiply(hidden,2))),4));
        // The short verification operator has two aligned FP16 intermediates.
        return add(add(input_bytes,align16(multiply(multiply(rows,hidden),2))),multiply(multiply(rows,width),2));
    }
    std::uint64_t decoded_expert_bytes()const {
        using namespace routed_prefill_workspace_detail;
        if(!input || !hidden || !output)throw std::invalid_argument("empty prefill projection geometry");
        return multiply(std::max(multiply(multiply(input,hidden),2),multiply(hidden,output)),2);
    }
};

struct RoutedPrefillComputeTile {
    std::size_t begin,end;
    std::uint64_t capacity,rows;
    bool gemm,compact;
};
struct RoutedPrefillWorkspacePlan {
    std::vector<RoutedPrefillComputeTile> tiles;
    std::vector<std::size_t> group_tile_ends;
    std::uint64_t activation_bytes=0,decoded_bytes=0,peak_rows=0;
    std::size_t peak_batch=0,peak_image_batch=0;
};

// Plan only batches whose compressed images are simultaneously available.
// Budgets are soft for one indivisible expert's activations, as in the GGUF
// schedule. Decoded weights always retain the bounded row-tile fallback.
template<class Count>
RoutedPrefillWorkspacePlan routed_prefill_workspace_plan(
        const std::vector<std::size_t>& group_ends,std::size_t gemm_groups,Count count,
        const RoutedPrefillWorkspaceShape& shape,std::uint64_t activation_budget,
        std::uint64_t decoded_budget,bool compact_gemm,std::size_t gemm_batch_limit=16) {
    using namespace routed_prefill_workspace_detail;
    RoutedPrefillWorkspacePlan plan;
    if(group_ends.empty()) {
        if(gemm_groups)throw std::invalid_argument("GEMM groups without prefill images");
        return plan;
    }
    if(!gemm_batch_limit || gemm_groups>group_ends.back())
        throw std::invalid_argument("invalid prefill compute batch limit");
    const auto full_image=shape.decoded_expert_bytes();
    const auto minimum_decoded=multiply(std::max(shape.input,shape.hidden),2);
    const auto decode_limit=std::max(minimum_decoded,decoded_budget);
    const auto decoded_batch_limit=std::max<std::uint64_t>(1,decode_limit/full_image);
    std::size_t begin=0;
    for(const auto group_end:group_ends) {
        if(group_end<=begin)throw std::invalid_argument("invalid prefill image group ends");
        plan.peak_image_batch=std::max(plan.peak_image_batch,group_end-begin);
        for(auto first=begin;first<group_end;) {
            const bool gemm=first<gemm_groups;
            const auto boundary=gemm?std::min(group_end,gemm_groups):group_end;
            const auto batch_limit=gemm?std::min<std::uint64_t>({65535,gemm_batch_limit,decoded_batch_limit}):65535;
            const auto budget=gemm?activation_budget:std::min<std::uint64_t>(activation_budget,16ULL<<20);
            std::uint64_t valid_rows=0,capacity=0,bytes=0,rows=0;
            bool compact=false;
            auto end=first;
            for(;end<boundary && end-first<batch_limit;++end) {
                const auto selected=std::uint64_t(count(end));
                if(!selected || selected>std::uint64_t(std::numeric_limits<int>::max()))
                    throw std::invalid_argument("invalid prefill expert row count");
                const auto next_capacity=std::max(capacity,align16(selected));
                if(next_capacity>std::uint64_t(std::numeric_limits<int>::max()))
                    throw std::overflow_error("prefill capacity exceeds the operator limit");
                // Compact WMMA has at most 65535 blocks of 32 token rows.
                const bool next_compact=gemm && compact_gemm && next_capacity<=65535ULL*32;
                const auto next_valid=add(valid_rows,selected);
                const auto next_rows=next_compact?next_valid:multiply(next_capacity,end-first+1);
                const auto next_bytes=shape.activation_bytes(next_rows,gemm);
                if(end>first && next_bytes>budget)break;
                valid_rows=next_valid;capacity=next_capacity;rows=next_rows;bytes=next_bytes;compact=next_compact;
            }
            if(end==first)throw std::logic_error("empty prefill compute tile");
            plan.tiles.push_back({first,end,capacity,rows,gemm,compact});
            plan.activation_bytes=std::max(plan.activation_bytes,bytes);
            plan.peak_rows=std::max(plan.peak_rows,rows);
            plan.peak_batch=std::max(plan.peak_batch,end-first);
            if(gemm)plan.decoded_bytes=std::max(plan.decoded_bytes,
                std::min(decode_limit,multiply(full_image,end-first)));
            first=end;
        }
        plan.group_tile_ends.push_back(plan.tiles.size());begin=group_end;
    }
    return plan;
}
} // namespace mfq::cuda
