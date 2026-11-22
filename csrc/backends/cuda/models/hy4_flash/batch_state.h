#pragma once

#include "models/deepseek_v41/engram.h"
#include "models/hy4_flash/ops.h"
#include "runtime/kv_offload.h"

#include <span>

namespace mfq::cuda::hy4 {

struct DraftKvHistory {
    int64_t position=0;
    mfq_tensor_backend::Tensor rows,indices;
};

struct LayerCache {
    mfq_tensor_backend::Tensor kv,records,pending_values,pending_logits;
    std::vector<std::unique_ptr<KvOffloadSequence>> sequences;
    mfq_tensor_backend::Tensor last_selection;
    std::vector<mfq_tensor_backend::Tensor> batch_selections;
    std::vector<DraftKvHistory> batch_draft_kv;
};

struct BatchCache {
    std::vector<LayerCache> layers;
    std::vector<mfq::models::deepseek_v41::EngramHashSnapshot> hashes;
};

BatchCache take_cache(Hy4CausalLm&);
void bind_cache(Hy4CausalLm&,BatchCache&&,int64_t position,bool decoding);
void reserve_slots(Hy4CausalLm&,BatchCache&,const BatchCache& prototype,int64_t slots,int64_t positions);
void copy_to_slot(const Hy4CausalLm&,BatchCache&,BatchCache& source,int64_t slot,int64_t length);
void release_slot(Hy4CausalLm&,BatchCache&,int64_t slot,bool bound);
BatchCache save_pending_groups(const Hy4CausalLm&,const mfq_tensor_backend::Tensor& slots);
void restore_pending_groups(Hy4CausalLm&,const BatchCache&,const mfq_tensor_backend::Tensor& slots);
bool can_batch(const Hy4CausalLm&);
using EngramRows=std::vector<std::pair<mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor>>;
void batch_engram_rows(Hy4CausalLm&,std::span<const int64_t> ids,std::span<const int64_t> positions,
                       std::span<const uint8_t> active,EngramRows& output,int64_t width=1,bool speculative=false);
void commit_batch_engram(Hy4CausalLm&,std::span<const int64_t> ids,std::span<const int64_t> positions,
                        std::span<const int64_t> kept,int64_t width);
void commit_batch_selections(Hy4CausalLm&,std::span<const uint8_t> active);
struct Verification {
    int64_t start=0;
    mfq_tensor_backend::Tensor pending_values,pending_logits,values,logits,positions,selections;
    mfq_tensor_backend::Tensor gathered_kv,gathered_indices;
};
using BatchVerification=std::vector<Verification>;
void set_batch_verification(Hy4CausalLm&,bool);
BatchVerification take_batch_verification(Hy4CausalLm&,bool rollback=true);
void retain_batch_prefix(Hy4CausalLm&,const BatchVerification&,const mfq_tensor_backend::Tensor& lengths,
                        std::span<const int64_t> positions,std::span<const int64_t> kept,int64_t width);
void set_staged_engram(Hy4CausalLm&,const EngramRows*);

}  // namespace mfq::cuda::hy4
