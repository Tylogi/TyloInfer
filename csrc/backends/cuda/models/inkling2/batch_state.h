#pragma once

#include "core/kv_cache.h"
#include "models/inkling2/ops.h"

#include <array>
#include <span>

namespace mfq::cuda::inkling {

struct LayerCache {
    KVCache attention;
    std::array<mfq_tensor_backend::Tensor,4> convolution;
};

struct BatchCache {
    std::vector<LayerCache> layers;
};

struct ConvolutionVerification {
    int64_t start=0;
    std::array<mfq_tensor_backend::Tensor,4> previous,inputs;
};
using BatchVerification=std::vector<ConvolutionVerification>;

BatchCache take_cache(InklingCausalLm&);
void bind_cache(InklingCausalLm&,BatchCache&&,int64_t position,bool decoding);
void reserve_slots(InklingCausalLm&,BatchCache&,const BatchCache& prototype,int64_t slots,int64_t positions,
                   int64_t draft_depth,std::span<const int64_t> live_lengths);
void copy_to_slot(const InklingCausalLm&,BatchCache&,const BatchCache& source,int64_t slot,int64_t length);
BatchCache save_convolution(const InklingCausalLm&,const mfq_tensor_backend::Tensor& slots);
void restore_convolution(InklingCausalLm&,const BatchCache&,const mfq_tensor_backend::Tensor& slots);
void set_batch_verification(InklingCausalLm&,bool enabled);
BatchVerification take_batch_verification(InklingCausalLm&);
void retain_batch_prefix(InklingCausalLm&,const BatchVerification&,const mfq_tensor_backend::Tensor& accepted);

}  // namespace mfq::cuda::inkling
