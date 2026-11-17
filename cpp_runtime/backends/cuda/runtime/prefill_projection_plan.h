#pragma once
#include "mfq_cuda_context.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <tuple>

namespace mfq::cuda {
inline const char* prefill_tensorcore_setting() {
    const auto* value=std::getenv("TYLOINFER_PREFILL_TENSORCORE");
    return value?value:std::getenv("TYLOINFER_GGUF_PREFILL_TENSORCORE");
}

// Decode complete matrices when live workspace permits it. Captures retain
// the tile selected during warmup, independently of the encoded row format.
inline std::int64_t prefill_weight_rows(std::int64_t rows,std::uint64_t bytes_per_row,
        std::int64_t alignment,int device,bool double_buffer=false) {
    const auto context=default_context(device);
    using Key=std::tuple<int,std::int64_t,std::uint64_t,std::int64_t,bool>;
    static thread_local std::map<Key,std::int64_t> prepared;
    const Key key{device,rows,bytes_per_row,alignment,double_buffer};
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture));
    if(capture!=cudaStreamCaptureStatusNone) {
        if(const auto found=prepared.find(key);found!=prepared.end())return found->second;
        return std::min(rows,alignment);
    }
    const auto physical=context->available_memory(),reusable=context->reusable_workspace_bytes();
    const auto available=physical>SIZE_MAX-reusable?SIZE_MAX:physical+reusable;
    const auto loan=context->workspace_available();
    const auto budget=available>SIZE_MAX-loan?SIZE_MAX:available+loan;
    auto count=std::min<std::uint64_t>(rows,(budget/2)/bytes_per_row);
    if(count<std::uint64_t(rows) && double_buffer)count/=2;
    const auto selected=count>=std::uint64_t(rows)?rows:count>=std::uint64_t(alignment)?
        std::int64_t(count/alignment*alignment):std::max<std::int64_t>(1,std::int64_t(count));
    prepared[key]=selected;return selected;
}
} // namespace mfq::cuda
