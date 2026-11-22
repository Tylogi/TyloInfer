#pragma once

#include "mfq_tensor_backend.h"

namespace mfq::cuda {

// Projected values/logits contain complete groups [groups, group_size, width].
// The rotary rows encode the first token of each group, in interleaved order.
mfq_tensor_backend::Tensor index_pool_keys(
    const mfq_tensor_backend::Tensor& values, const mfq_tensor_backend::Tensor& logits,
    const mfq_tensor_backend::Tensor& position_bias, const mfq_tensor_backend::Tensor& norm,
    const mfq_tensor_backend::Tensor& cosine, const mfq_tensor_backend::Tensor& sine, float epsilon);

struct IndexPoolAppend {
    mfq_tensor_backend::Tensor keys;
    mfq_tensor_backend::Tensor record_positions;
};

// Emit the groups completed by [batch, tokens, width] and advance the pending
// group in place. Incomplete candidates carry record position -1.
IndexPoolAppend index_pool_append(const mfq_tensor_backend::Tensor& values,
    const mfq_tensor_backend::Tensor& logits, const mfq_tensor_backend::Tensor& positions,
    mfq_tensor_backend::Tensor& pending_values, mfq_tensor_backend::Tensor& pending_logits,
    const mfq_tensor_backend::Tensor& position_bias, const mfq_tensor_backend::Tensor& norm,
    const mfq_tensor_backend::Tensor& cosine, const mfq_tensor_backend::Tensor& sine,
    int capacity, float epsilon);

// Retain the target selection once per request. The new draft KV is kept in
// a separate, short causal suffix and never overwrites the target's history.
// Invalid or padded positions are represented by -1.
mfq_tensor_backend::Tensor index_pool_draft_history(
    const mfq_tensor_backend::Tensor& selected, const mfq_tensor_backend::Tensor& confirmed_lengths,
    int recent_window);

void index_pool_retain_prefix(const mfq_tensor_backend::Tensor& values,
    const mfq_tensor_backend::Tensor& logits,const mfq_tensor_backend::Tensor& positions,
    const mfq_tensor_backend::Tensor& lengths,const mfq_tensor_backend::Tensor& saved_values,
    const mfq_tensor_backend::Tensor& saved_logits,mfq_tensor_backend::Tensor& pending_values,
    mfq_tensor_backend::Tensor& pending_logits);

}  // namespace mfq::cuda
