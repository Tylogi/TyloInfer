#pragma once

#include <cstddef>
#include <optional>
#include <span>

#include <mlx/mlx.h>

namespace mfq::metal {

mlx::core::array mlx_sparse_indexer_dot_scores(
    const mlx::core::array& query, const mlx::core::array& pooled_keys);

mlx::core::array mlx_sparse_indexer_head_scores(
    const mlx::core::array& query, const mlx::core::array& pooled_keys);

mlx::core::array mlx_sparse_indexer_weighted_scores(const mlx::core::array& query,
    const mlx::core::array& pooled_keys, const mlx::core::array& weights);

// Exact FP32 top-512 selection adapted from DeepSelect. The operator and its
// device/shape policy are model-neutral; architecture adapters only provide
// score tensors and valid-key counts.
mlx::core::array mlx_deepselect_topk512(
    const mlx::core::array& scores,
    const std::optional<mlx::core::array>& valid_keys = std::nullopt);

bool mlx_deepselect_topk512_preferred(int width, int rows) noexcept;

mlx::core::array mlx_sparse_indexer_topk512(
    const mlx::core::array& head_scores,
    int query_offset,
    int block_size);

// Select 511 historical groups of four tokens, then append the causal tail.
mlx::core::array mlx_sparse_indexer_token_ids(const mlx::core::array& scores, int position);

mlx::core::array mlx_sparse_block_gqa_prefill_tiles(const mlx::core::array& query,
    const mlx::core::array& key, const mlx::core::array& value,
    const std::optional<mlx::core::array>& selected, int query_offset, float scale,
    mlx::core::Dtype output_dtype);

mlx::core::array mlx_qsa_prefill_attention(
    const mlx::core::array& query,
    const mlx::core::array& key,
    const mlx::core::array& value,
    const mlx::core::array& index_query,
    const mlx::core::array& pooled_index_keys,
    int query_offset);

// Common selected-block sparse-attention seam for Metal runtimes. Model
// adapters own index construction and cache semantics; this operator owns the
// direct indexed GQA execution. Blocks identify fixed-width, chronological
// cache blocks. The incomplete causal tail is appended inside the kernel.
mlx::core::array mlx_sparse_block_gqa_attention(
    const mlx::core::array& query,
    const mlx::core::array& key,
    const mlx::core::array& value,
    const mlx::core::array& selected_blocks,
    int query_offset,
    int block_size,
    std::optional<float> scale = std::nullopt);

// Common selected-token sparse-MLA seam.  The adapter owns index selection,
// masks, and cache updates. The value may use only the latent prefix of the
// cached key. The remaining key coordinates can carry rotary features.
mlx::core::array mlx_sparse_selected_mla_attention(
    const mlx::core::array& query,
    const mlx::core::array& kv_cache,
    const mlx::core::array& selected_indices,
    const mlx::core::array& selected_mask,
    const mlx::core::array& sinks,
    std::optional<float> scale = std::nullopt, int value_dimension = 0);

mlx::core::array mlx_gather_selected_mla_rows(const mlx::core::array& kv_cache,
    const mlx::core::array& selected_indices);

void mlx_prepare_indexed_mla_prefill();

// Attend directly to independently stored FP16 KV for a packed batch of short indexed queries.
std::optional<mlx::core::array> mlx_batched_resident_mla_attention(const mlx::core::array& query,
    std::span<const mlx::core::array> caches, std::span<const mlx::core::array> selections,
    const mlx::core::array& sinks, float scale);

// The same latent attention over rows already gathered from paged storage.
mlx::core::array mlx_gathered_mla_attention(const mlx::core::array& query,
    const mlx::core::array& gathered, const mlx::core::array& mask,
    const mlx::core::array& sinks, float scale, int value_dimension, bool packed_queries = false);

// FP16 resident KV shared by the queries and heads of each request.
mlx::core::array mlx_masked_dense_mla_attention(const mlx::core::array& query,
    const mlx::core::array& rows, const mlx::core::array& mask,
    const mlx::core::array& sinks, float scale, int value_dimension,
    const std::optional<mlx::core::array>& kv_lanes = std::nullopt);

// Multi-query sparse MLA over an already chronological local window and a
// capacity-backed compressed pool. This is the oMLX-style M>1 path: causal
// local and pooled visibility are derived in the kernel, so adapters do not
// need to concatenate the pool or materialize an indices/mask plan.
mlx::core::array mlx_sparse_circular_mla_attention(
    const mlx::core::array& query,
    const mlx::core::array& local_kv,
    const std::optional<mlx::core::array>& pooled_kv,
    int pool_len,
    const mlx::core::array& topk,
    const mlx::core::array& sinks,
    int query_offset,
    int pool_ratio,
    int local_window,
    std::optional<float> scale = std::nullopt);

// Single-token sparse-MLA specialization over a circular local cache plus an
// optional compressed pool.  Keeping this in the common backend lets DSA and
// sparse-MLA adapters share execution while retaining their own index policy.
mlx::core::array mlx_sparse_circular_mla_decode_attention(
    const mlx::core::array& query,
    const mlx::core::array& local_kv,
    const std::optional<mlx::core::array>& pooled_kv,
    int pool_len,
    const mlx::core::array& topk,
    const mlx::core::array& sinks,
    int sequence_length,
    int pool_ratio,
    int local_window,
    std::optional<float> scale = std::nullopt);

} // namespace mfq::metal
