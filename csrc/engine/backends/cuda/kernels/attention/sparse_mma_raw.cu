#include "sparse_mma_raw.h"
#include "sparse_mma_f16.cuh"
#include "engine/backends/cuda/kernels/kernel_dispatch_trace.h"

#include <algorithm>
#include <climits>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <stdexcept>
#include <string>

namespace tyloi::cuda::sparse_mma_raw {
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template<int DKQ, int DV, bool view, bool hot, int ncols2, bool resolved>
Plan make_plan(int B, int M, int heads, int kv_heads, int max_seq, int selected, bool native_selection) {
    constexpr int ncols = ncols2;
    constexpr int ncols1 = 1;
    auto kernel = mfq_sparse_attn_mma::sparse_attention_kernel<DKQ, DV, ncols1, ncols2, view, hot, resolved>;
    int device = 0;
    cudaDeviceProp properties{};
    check(cudaGetDevice(&device), "cudaGetDevice");
    check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    const int cc = properties.major * 100 + properties.minor * 10;
    if (!turing_mma_available(cc)) throw std::invalid_argument("attention requires Turing-class tensor cores or newer");
    const auto base = ampere_mma_available(cc)
        ? ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols)
        : ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
    const auto config = native_selection ? mfq_fattn_mma_select_config(base, DKQ, DV, ncols, true) : base;
    const int threads = config.nthreads;
    if (threads <= 0 || config.nbatch_fa <= 0 || config.nbatch_K2 <= 0 ||
        config.nbatch_V2 <= 0 || config.nbatch_combine <= 0)
        throw std::invalid_argument("unsupported MMA geometry");
    const int warps = threads / 32;
    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const size_t shared_kv = size_t(config.nbatch_fa) * std::max(config.nbatch_K2 + 4, config.nbatch_V2 + 4) * sizeof(half2);
    const size_t shared_q = size_t(ncols) * (DKQ / 2 + 4) * sizeof(half2);
    const size_t shared_mask = size_t(ncols1) * (config.nbatch_fa / 2 + 4) * sizeof(half2);
    const size_t shared_combine = size_t(warps) * cols_per_warp * (config.nbatch_combine + 4) * sizeof(half2);
    const size_t shared = std::max(shared_combine, config.Q_in_reg ? std::max(shared_q, shared_kv + shared_mask) : shared_q + shared_kv + shared_mask);
    {
        static std::mutex mutex;
        static std::unordered_map<int, size_t> devices;
        std::lock_guard<std::mutex> lock(mutex);
        auto found = devices.find(device);
        if (found == devices.end() || found->second < shared) {
            check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)), "attention shared-memory attribute");
            devices[device] = shared;
        }
    }
    const int ratio = heads / kv_heads;
    const int iter_j = M;
    const int iter_z = int((int64_t(ratio) + ncols2 - 1) / ncols2);
    int64_t tiles64 = B;
    for (const int factor : {kv_heads, iter_z, iter_j}) {
        if (tiles64 > int64_t(INT_MAX) / factor)
            throw std::invalid_argument("attention CUDA grid is too large");
        tiles64 *= factor;
    }
    const int64_t ntiles_kv = (int64_t(selected) + config.nbatch_fa - 1) / config.nbatch_fa;
    const int tiles = int(tiles64);
    int max_blocks = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, threads, shared), "attention occupancy");
    if (max_blocks <= 0) throw std::runtime_error("attention occupancy is zero");
    const int64_t resident = int64_t(max_blocks) * properties.multiProcessorCount;
    const int64_t total_tiles = tiles64 * ntiles_kv;
    if constexpr (ncols2 == 16 && DKQ != 576) {
        if (native_selection) {
            const auto small_base = ampere_mma_available(cc)
                ? ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, 8)
                : ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, 8);
            const auto small = mfq_fattn_mma_select_config(small_base, DKQ, DV, 8, true);
            if (small.nthreads >= 128 && small.nbatch_fa > 0 && selected % small.nbatch_fa == 0 &&
                (ratio <= 8 || total_tiles < resident))
                return make_plan<DKQ, DV, view, hot, 8, resolved>(B, M, heads, kv_heads, max_seq, selected, true);
        }
    }
    const int64_t raw = std::min(resident, total_tiles);
    const int64_t blocks64 = std::max(tiles64, (raw / tiles64) * tiles64);
    if (blocks64 > INT_MAX) throw std::invalid_argument("attention CUDA grid is too large");
    const int blocks = int(blocks64);
    const int parts = blocks / tiles;
    size_t required = 1;
    if (parts > 1) {
        required = size_t(blocks);
        for (const size_t factor : {size_t(ncols), size_t(2 + DV / 2), size_t(2)}) {
            if (required > SIZE_MAX / factor) throw std::invalid_argument("attention workspace is too large");
            required *= factor;
        }
        if (required > size_t(std::numeric_limits<int64_t>::max()) || required > SIZE_MAX / sizeof(float))
            throw std::invalid_argument("attention workspace is too large");
    }
    return {B, M, heads, kv_heads, max_seq, selected, DKQ, DV, blocks, parts, tiles,
        iter_j, iter_z, threads, device, shared, required * sizeof(float), hot, ncols2, resolved, config.nbatch_fa, native_selection};
}

template<int DKQ, int DV, bool view, bool hot, int ncols2, bool resolved>
void run(const Plan& p, const void* q, const void* k, const void* v,
    const int32_t* indices, const void* mask, const float* sinks, const void* positions,
    const int64_t strides[4], bool half_query, bool query_contiguous, bool wide_positions, int causal_columns,
    Hot cache, float scale, float* out, void* workspace, bool gated, cudaStream_t stream) {
    constexpr int ncols1 = 1;
    const tyloi::cuda::HotKvView hot_view{cache.keys, cache.values, cache.mapping, cache.cells,
                                          cache.cold_mapping, cache.cold_cells};
    const mfq_sparse_attn_mma::RuntimeInput input{positions, strides[0], strides[1], strides[2], strides[3], causal_columns,
        half_query, positions && wide_positions,
        positions || half_query || !query_contiguous, hot_view};
    mfq_sparse_attn_mma::sparse_attention_kernel<DKQ, DV, ncols1, ncols2, view, hot, resolved>
        <<<p.blocks, dim3(32, p.threads / 32, 1), p.shared_bytes, stream>>>(
            q, static_cast<const half*>(k), static_cast<const half*>(v), indices,
            static_cast<const half*>(mask), sinks, out, static_cast<float2*>(workspace), scale,
            p.batch, p.tokens, p.heads, p.kv_heads, p.max_seq, p.selected,
            mfq_sparse_attn_mma::sparse_fastdiv_values(uint32_t(p.tokens)), input);
    check(cudaGetLastError(), "attention launch");
    if (p.native_selection && tyloi::cuda::dispatch_trace::enabled()) tyloi::cuda::dispatch_trace::record(
        stream, "selected_attention_mma",
        reinterpret_cast<const void*>(mfq_sparse_attn_mma::sparse_attention_kernel<DKQ, DV, ncols1, ncols2, view, hot, resolved>),
        "M="+std::to_string(p.tokens)+" heads="+std::to_string(p.heads)+
        " kv_heads="+std::to_string(p.kv_heads)+" width="+std::to_string(DKQ)+
        " query_heads_per_tile="+std::to_string(ncols2)+
        " tile_keys="+std::to_string(p.key_tile)+
        " selected="+std::to_string(p.selected)+" parts="+std::to_string(p.parts)+
        " blocks="+std::to_string(p.blocks)+" shared_bytes="+std::to_string(p.shared_bytes)+
        " resolved_indices="+std::to_string(resolved)+
        " direct_query="+std::to_string(input.direct_query)+
        " causal_live_tiles="+std::to_string(positions != nullptr));
    if (!gated && p.parts > 1) {
        const auto fd0 = mfq_sparse_attn_mma::sparse_fastdiv_values(uint32_t(int64_t(p.iter_j) * p.iter_z_gqa * p.kv_heads));
        const auto fd1 = mfq_sparse_attn_mma::sparse_fastdiv_values(uint32_t(int64_t(p.iter_j) * p.iter_z_gqa));
        const auto fd2 = mfq_sparse_attn_mma::sparse_fastdiv_values(uint32_t(p.iter_j));
        flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>
            <<<dim3(p.tiles, ncols1, ncols2), DV, 0, stream>>>(out, static_cast<const float2*>(workspace),
                p.tokens, p.heads, p.kv_heads, p.blocks, p.heads / p.kv_heads, p.parts, fd0, fd1, fd2);
        check(cudaGetLastError(), "attention fixup");
    }
}
}

int selection_alignment(int width, int value_width, int device) {
    return mfq_sparse_attn_mma::selection_alignment(width, value_width, 16, device);
}

Plan plan(int B, int M, int heads, int kv_heads, int max_seq, int selected,
          int width, int value_width, bool hot, bool native_selection, bool resolved_hot) {
    if (resolved_hot && (!hot || !native_selection)) throw std::invalid_argument("invalid resolved hot attention plan");
    if (B <= 0 || M <= 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads || max_seq <= 0 || selected <= 0)
        throw std::invalid_argument("invalid attention geometry");
    if (width == 256 && value_width == 256)
        return hot ? (resolved_hot ? make_plan<256, 256, false, true, 16, true>(B, M, heads, kv_heads, max_seq, selected, native_selection)
                    : make_plan<256, 256, false, true, 16, false>(B, M, heads, kv_heads, max_seq, selected, native_selection))
                   : make_plan<256, 256, false, false, 16, false>(B, M, heads, kv_heads, max_seq, selected, native_selection);
    if (width == 512 && value_width == 512 && !hot)
        return make_plan<512, 512, true, false, 16, false>(B, M, heads, kv_heads, max_seq, selected, native_selection);
    if (width == 576 && value_width == 512 && !hot)
        return make_plan<576, 512, true, false, 16, false>(B, M, heads, kv_heads, max_seq, selected, native_selection);
    throw std::invalid_argument("unsupported attention MMA variant");
}

void launch(const Plan& p, const void* q, const void* k, const void* v, const int32_t* indices,
    const void* mask, const float* sinks, const void* positions, const int64_t strides[4], bool half_query, bool query_contiguous,
    bool wide_positions, int causal_columns, Hot hot, float scale, float* out, void* workspace,
    std::size_t bytes, bool gated, cudaStream_t stream) {
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    if (device != p.device || !q || !k || !v || !strides || !out || !workspace || bytes < p.workspace_bytes ||
        p.hot != (hot.mapping != nullptr) || (positions ? (causal_columns <= 0 || indices || mask) : !indices))
        throw std::invalid_argument("invalid attention MMA launch inputs");
    #define RUN(W,V,VIEW,HOT,COLS,RESOLVED) run<W,V,VIEW,HOT,COLS,RESOLVED>(p, q, k, v, indices, mask, sinks, positions, strides, half_query, query_contiguous, wide_positions, causal_columns, hot, scale, out, workspace, gated, stream)
    if (p.width == 256 && p.value_width == 256 && p.hot) {
        if (p.resolved_hot) {
            if (p.head_columns == 8) RUN(256,256,false,true,8,true); else RUN(256,256,false,true,16,true);
        } else {
            if (p.head_columns == 8) RUN(256,256,false,true,8,false); else RUN(256,256,false,true,16,false);
        }
    } else if (p.width == 256 && p.value_width == 256 && !p.hot) {
        if (p.head_columns == 8) RUN(256,256,false,false,8,false); else RUN(256,256,false,false,16,false);
    } else if (p.width == 512 && p.value_width == 512 && !p.hot) {
        if (p.head_columns == 8) RUN(512,512,true,false,8,false); else RUN(512,512,true,false,16,false);
    } else if (p.width == 576 && p.value_width == 512 && !p.hot) {
        RUN(576,512,true,false,16,false);
    } else throw std::invalid_argument("unsupported attention MMA variant");
    #undef RUN
}
}
