#include "qsa_tq.h"
#include <climits>

namespace tyloinfer::qsa {
namespace {
__global__ void accumulate_recall_bits(const SelectedBlock* selected,const std::int64_t* positions,
    std::uint32_t* recalled,std::int64_t total,int count,int blocks) {
    const auto item=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(item>=total)return;
    const auto block=selected[item];
    if(block.id<0 || block.id>=blocks)return;
    const auto first=std::int64_t(block.id)*4,position=positions[item/count];
    if(position<first)return;
    const auto visible=position-first;
    const auto mask=visible>=3?15u:(1u<<unsigned(visible+1))-1;
    if(block.token_mask&mask)atomicOr(recalled+unsigned(block.id)/32,1u<<(unsigned(block.id)%32));
}
}

int tq_accumulate_recall_bits(const SelectedBlock* selected,const std::int64_t* positions,
    std::uint32_t* recalled,int queries,int count,int blocks,cudaStream_t stream) {
    const auto total=std::int64_t(queries)*count;
    if(!selected || !positions || !recalled || queries<=0 || count<=0 || blocks<=0 ||
        (total+255)/256>INT_MAX)return int(cudaErrorInvalidValue);
    accumulate_recall_bits<<<unsigned((total+255)/256),256,0,stream>>>(selected,positions,recalled,total,count,blocks);
    return int(cudaGetLastError());
}
}
