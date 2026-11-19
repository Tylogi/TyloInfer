#include "gguf.h"
#include "tyloinfer/gguf_quant.h"
#include "ggml-cuda/common.cuh"
#include "gguf_decode.cuh"
#include <mma.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace mfq::cuda::gguf {
namespace {
template<int Format>
__global__ void decode_group(const GemmJob* jobs,std::uint64_t stride,std::int64_t rows,std::int64_t columns,
        std::int64_t batch_stride,half* output) {
    const auto i=(std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x)*8;
    if(i>=rows*columns)return;
    const auto* packed=static_cast<const std::uint8_t*>(jobs[blockIdx.y].packed)+(i/columns)*stride;
    float values[8];
    gguf_values8<Format>(packed,i%columns,values);
    gguf_store8(output+std::int64_t(blockIdx.y)*batch_stride+i,values);
}
__global__ void gather_group(const GemmJob* jobs,std::int64_t tokens,std::int64_t columns,
    const half* input,std::int64_t routes,half* output) {
    const auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=tokens*columns)return;
    const auto job=jobs[blockIdx.y];const auto row=i/columns;
    output[std::int64_t(blockIdx.y)*tokens*columns+i]=row<job.count?
        input[(job.slots[row]/routes)*columns+i%columns]:__float2half_rn(0.f);
}
__global__ void scatter_group(const GemmJob* jobs,std::int64_t tokens,std::int64_t columns,
    const half* input,half* output) {
    const auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto job=jobs[blockIdx.y];
    if(i<job.count*columns)output[job.slots[i/columns]*columns+i%columns]=input[std::int64_t(blockIdx.y)*tokens*columns+i];
}
__global__ void scatter(const std::uint8_t* packet) {
    const auto chunk = reinterpret_cast<const UploadChunk*>(packet)[blockIdx.y];
    auto* output = reinterpret_cast<std::uint8_t*>(chunk.destination);
    const auto* input = packet + chunk.offset;
    const auto begin = std::uint64_t(blockIdx.x) * 65536;
    const auto end = min(chunk.bytes, begin + 65536);
    for (auto i = begin + std::uint64_t(threadIdx.x) * 16; i < end; i += std::uint64_t(blockDim.x) * 16) {
        if (i + 16 <= end) *reinterpret_cast<uint4*>(output+i) = *reinterpret_cast<const uint4*>(input+i);
        else for (auto tail=i;tail<end;++tail) output[tail]=input[tail];
    }
}
// Strata's grouped prompt path shares one decoded weight tile across token
// rows. Keep that organization with runtime dimensions and original GGUF
// blocks: FP16 uses four warps for a 32x32 tile; compensated BF16 uses
// eight warps for a 64x64 tile, including incomplete experts.
template<class Value> __device__ unsigned short storage_word(Value x) {
    if constexpr (std::is_same_v<Value, half>) return __half_as_ushort(x);
    else return __bfloat16_as_ushort(x);
}
template<int Format, class Value = half, int TokenTile = std::is_same_v<Value, half> ? 32 : 64,
    int WeightTile = 64, int Warps = 8, bool VectorDecode = false, bool PlanarWeight = false, int Padding = 0, int KBlock = 128, bool SinglePartial = false, bool AsyncInput = false, bool DelayedCompensation = false>
__global__ void grouped(const GemmJob* jobs, std::uint64_t stride, std::int64_t rows,
    std::int64_t columns, const Value* input, std::int64_t routes, bool routed_input, Value* output) {
#if __CUDA_ARCH__ >= 700
    if constexpr (std::is_same_v<Value, half> || __CUDA_ARCH__ >= 800) {
    constexpr bool bf16 = !std::is_same_v<Value, half>;
    constexpr int Tile = bf16 ? WeightTile : 32, KTile = KBlock, SharedK = KTile + Padding;
    constexpr int RowWarps = Tile / 16, TokenWarps = Warps / RowWarps;
    constexpr int Fragments = bf16 ? TokenTile / (16 * TokenWarps) : 1;
    static_assert(!bf16 || (Tile % 16 == 0 && Warps % RowWarps == 0 && TokenTile % (16 * TokenWarps) == 0));
    const auto job = jobs[blockIdx.z];
    const auto token_begin = std::int64_t(blockIdx.y) * TokenTile;
    const auto row_begin = std::int64_t(blockIdx.x) * Tile;
    if (token_begin >= job.count) return;
    // BF16 output is stored only after the final operand barrier, so its
    // shared buffer can reuse operands without increasing the tile footprint.
    constexpr int OperandValues = (TokenTile + Tile) * SharedK;
    constexpr int OutputValues = TokenTile * Tile * sizeof(float) / sizeof(Value);
    constexpr int SharedValues = bf16 ? (OperandValues > OutputValues ? OperandValues : OutputValues) : OperandValues + OutputValues;
    __shared__ __align__(32) Value storage[SharedValues];
    Value* x = storage;
    Value* w = storage + TokenTile * SharedK;
    float* result = reinterpret_cast<float*>(bf16 ? storage : storage + OperandValues);
    using namespace nvcuda;
    const int warp = threadIdx.x / 32;
    const int token_tile = bf16 ? warp / RowWarps * (TokenTile / TokenWarps) : warp / 2 * 16;
    const int row_tile = bf16 ? warp % RowWarps * 16 : warp % 2 * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, Value, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, Value, wmma::col_major> b;
    using Accumulator = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    constexpr int PartialFragments = bf16 && SinglePartial ? 1 : Fragments;
    Accumulator c[PartialFragments];
    for (int f = 0; f < PartialFragments; ++f) wmma::fill_fragment(c[f], 0.f);
    float precise[Fragments][Accumulator::num_elements]{}, correction[Fragments][Accumulator::num_elements]{};
    for (std::int64_t begin = 0; begin < columns; begin += KTile) {
        if constexpr (bf16 && AsyncInput) {
            for (int i = threadIdx.x * 8; i < TokenTile * KTile; i += blockDim.x * 8) {
                const auto column = begin + i % KTile, token = token_begin + i / KTile;
                const auto shared = (i / KTile) * SharedK + i % KTile;
                if (token < job.count && column + 7 < columns) {
                    const auto slot = job.slots[token];
                    const auto* source = input + (routed_input ? slot : slot / routes) * columns + column;
                    const auto address = static_cast<unsigned>(__cvta_generic_to_shared(x + shared));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(address), "l"(source));
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) x[shared + j] = token < job.count && column + j < columns ?
                        input[(routed_input ? job.slots[token] : job.slots[token] / routes) * columns + column + j] : Value(0.f);
                }
            }
            asm volatile("cp.async.commit_group;");
        }
        // Adjacent values share the GGUF codeword and scaling metadata. Decode
        // eight together, and amortize synchronization over eight MMA steps.
        for (int i = threadIdx.x * 8; i < Tile * KTile; i += blockDim.x * 8) {
            const auto row = row_begin + i / KTile, column = begin + i % KTile;
            const auto* packed = static_cast<const std::uint8_t*>(job.packed) + row * stride;
            uint4 decoded{};
            auto* words=reinterpret_cast<std::uint32_t*>(&decoded);
            if constexpr (bf16 && Format == GGML_TYPE_Q8_0 && VectorDecode) {
                // Whole Q8_0 blocks guarantee a two-byte-aligned code pair.
                // Share its scale and bounds check across eight adjacent values.
                if (row < rows && column < columns) {
                    const auto* block = reinterpret_cast<const block_q8_0*>(packed) + column / 32;
                    const auto* plane = static_cast<const std::uint8_t*>(job.packed);
                    float scale;
                    uint2 eight{};
                    if constexpr (PlanarWeight) {
                        const auto at = row * columns + column;
                        scale = __half2float(reinterpret_cast<const half*>(plane + stride)[at / 32]);
                        eight = *reinterpret_cast<const uint2*>(plane + at);
                    } else scale = __half2float(block->d);
                    #pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        std::uint16_t codes;
                        if constexpr (PlanarWeight) codes = (pair < 2 ? eight.x : eight.y) >> (16 * (pair % 2));
                        else codes = *reinterpret_cast<const std::uint16_t*>(block->qs + column % 32 + pair * 2);
                        const float lo = float(static_cast<std::int8_t>(codes & 255));
                        const float hi = float(static_cast<std::int8_t>(codes >> 8));
                        const auto values = __floats2bfloat162_rn(__fmul_rn(scale, lo), __fmul_rn(scale, hi));
                        words[pair] = std::uint32_t(__bfloat16_as_ushort(values.x)) | (std::uint32_t(__bfloat16_as_ushort(values.y)) << 16);
                    }
                }
            } else {
                #pragma unroll
                for(int pair=0;pair<4;++pair) {
                    const auto k=column+pair*2;
                    const auto a=Value(row<rows && k<columns?gguf_value<Format>(packed,k):0.f);
                    const auto b=Value(row<rows && k+1<columns?gguf_value<Format>(packed,k+1):0.f);
                    words[pair]=std::uint32_t(storage_word(a))|(std::uint32_t(storage_word(b))<<16);
                }
            }
            *reinterpret_cast<uint4*>(w + (i / KTile) * SharedK + i % KTile)=decoded;
        }
        if constexpr (!bf16 || !AsyncInput) {
            for (int i = threadIdx.x * 8; i < TokenTile * KTile; i += blockDim.x * 8) {
                const auto column = begin + i % KTile;
                const auto token=token_begin+i/KTile;
                const auto shared = (i / KTile) * SharedK + i % KTile;
                if(token<job.count && column+7<columns) {
                    const auto slot=job.slots[token];
                    *reinterpret_cast<uint4*>(x+shared)=*reinterpret_cast<const uint4*>(
                        input+(routed_input?slot:slot/routes)*columns+column);
                } else {
                    #pragma unroll
                    for(int j=0;j<8;++j) x[shared+j]=token<job.count && column+j<columns?
                        input[(routed_input?job.slots[token]:job.slots[token]/routes)*columns+column+j]:Value(0.f);
                }
            }
        }
        if constexpr (bf16 && AsyncInput) asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < KTile; k += 16) {
            wmma::load_matrix_sync(b, w + row_tile * SharedK + k, SharedK);
            #pragma unroll
            for (int f = 0; f < Fragments; ++f) {
                const int ci = bf16 && SinglePartial ? 0 : f;
                wmma::load_matrix_sync(a, x + (token_tile + f * 16) * SharedK + k, SharedK);
                if constexpr (bf16) wmma::fill_fragment(c[ci], 0.f);
                wmma::mma_sync(c[ci], a, b, c[ci]);
                // Compensate BF16 block sums before final activation rounding.
                if constexpr (bf16 && !DelayedCompensation) {
                    for (int e = 0; e < c[ci].num_elements; ++e) {
                        const float add = __fsub_rn(c[ci].x[e], correction[f][e]);
                        const float sum = __fadd_rn(precise[f][e], add);
                        correction[f][e] = __fsub_rn(__fsub_rn(sum, precise[f][e]), add); precise[f][e] = sum;
                    }
                }
            }
            // Read independent MMA fragments together; each output still
            // compensates every 16-product partial in the original order.
            if constexpr (bf16 && DelayedCompensation) {
                static_assert(!SinglePartial);
                #pragma unroll
                for (int f = 0; f < Fragments; ++f) {
                    #pragma unroll
                    for (int e = 0; e < c[f].num_elements; ++e) {
                        const float add = __fsub_rn(c[f].x[e], correction[f][e]);
                        const float sum = __fadd_rn(precise[f][e], add);
                        correction[f][e] = __fsub_rn(__fsub_rn(sum, precise[f][e]), add);
                        precise[f][e] = sum;
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int f = 0; f < Fragments; ++f) {
        const int ci = bf16 && SinglePartial ? 0 : f;
        if constexpr (bf16) {
            for (int e = 0; e < c[ci].num_elements; ++e) c[ci].x[e] = precise[f][e];
        }
        wmma::store_matrix_sync(result + (token_tile + f * 16) * Tile + row_tile, c[ci], Tile, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TokenTile * Tile; i += blockDim.x) {
        const auto token = token_begin + i / Tile, row = row_begin + i % Tile;
        if (token < job.count && row < rows) output[job.slots[token] * rows + row] = Value(result[i]);
    }
    } else {
        asm volatile("trap;");
    }
#else
    asm volatile("trap;");
#endif
}
// Reorder quantized weight bytes only; scale bit patterns are unchanged.
__global__ void q8_weight_columns(const std::uint8_t* input, std::uint64_t stride,
    std::int64_t rows, std::int64_t columns, std::uint8_t* output) {
    const auto group = (std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (group >= rows * (columns / 8)) return;
    const auto row = group % rows, column = (group / rows) * 8;
    const auto* block = input + row * stride + (column / 32) * 34;
    const auto* values = reinterpret_cast<const std::uint16_t*>(block + 2 + column % 32);
    const uint2 packed{std::uint32_t(values[0]) | (std::uint32_t(values[1]) << 16),
        std::uint32_t(values[2]) | (std::uint32_t(values[3]) << 16)};
    *reinterpret_cast<uint2*>(output + group * 8) = packed;
    if (column % 32 == 0)
        reinterpret_cast<std::uint16_t*>(output + rows * columns)[(column / 32) * rows + row] = *reinterpret_cast<const std::uint16_t*>(block);
}
// Native MMA avoids WMMA's duplicated operand registers and shared readout.
// Both paths compensate the same 16-product FP32 partials.
template<bool AsyncInput = true, bool DenseSlots = false, int MinimumBlocks = 1, int ReductionTile = 64, int OutputFragments = 1, bool ColumnPlanes = false, class Index = std::uint64_t>
__global__ __launch_bounds__(256, MinimumBlocks) void q8_bf16_mma(const GemmJob* jobs, GemmJob dense_job, Index stride, Index rows,
    Index columns, const __nv_bfloat16* input, std::int64_t routes,
    bool routed_input, __nv_bfloat16* output) {
#if __CUDA_ARCH__ >= 800
    constexpr int Tile = 64, KTile = ReductionTile, SharedK = KTile + 8;
    static_assert(KTile % 16 == 0 && OutputFragments > 0 && 4 % OutputFragments == 0);
    constexpr int MFragments = 4 / OutputFragments, RowWarps = 8 / OutputFragments;
    const auto job = DenseSlots ? dense_job : jobs[blockIdx.z];
    const Index tokens = Index(job.count);
    const Index token_begin = Index(blockIdx.y) * Tile;
    const Index row_begin = Index(blockIdx.x) * Tile;
    if (token_begin >= tokens) return;
    __shared__ __align__(32) __nv_bfloat16 x[Tile * SharedK], w[Tile * SharedK];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int token_tile = (warp / RowWarps) * MFragments * 16;
    const int row_tile = (warp % RowWarps) * OutputFragments * 8;
    float precise[4][4]{}, correction[4][4]{};
    for (Index begin = 0; begin < columns; begin += KTile) {
        for (int i = threadIdx.x * 8; i < Tile * KTile; i += blockDim.x * 8) {
            const auto column = begin + i % KTile, token = token_begin + i / KTile;
            const auto shared = (i / KTile) * SharedK + i % KTile;
            if (token < tokens && column + 7 < columns) {
                const auto slot = DenseSlots ? token : job.slots[token];
                const auto* source = input + std::uint64_t(DenseSlots ? token : (routed_input ? slot : slot / routes)) * columns + column;
                if constexpr (AsyncInput) {
                    const auto address = static_cast<unsigned>(__cvta_generic_to_shared(x + shared));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(address), "l"(source));
                } else *reinterpret_cast<uint4*>(x + shared) = *reinterpret_cast<const uint4*>(source);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) x[shared + j] = token < tokens && column + j < columns ?
                    input[std::uint64_t(DenseSlots ? token : (routed_input ? job.slots[token] : job.slots[token] / routes)) * columns + column + j] : __nv_bfloat16(0.f);
            }
        }
        if constexpr (AsyncInput) asm volatile("cp.async.commit_group;");
        for (int i = threadIdx.x * 8; i < Tile * KTile; i += blockDim.x * 8) {
            const auto local_row = ColumnPlanes ? (i / 8) % Tile : i / KTile;
            const auto local_column = ColumnPlanes ? ((i / 8) / Tile) * 8 : i % KTile;
            const auto row = row_begin + local_row, column = begin + local_column;
            uint4 decoded{};
            auto* words = reinterpret_cast<std::uint32_t*>(&decoded);
            if (row < rows && column < columns) {
                const auto* packed = static_cast<const std::uint8_t*>(job.packed);
                const auto* block = reinterpret_cast<const block_q8_0*>(packed + std::uint64_t(row) * stride) + column / 32;
                float scale;
                uint2 eight{};
                if constexpr (ColumnPlanes) {
                    eight = *reinterpret_cast<const uint2*>(packed + (std::uint64_t(column / 8) * rows + row) * 8);
                    scale = __half2float(reinterpret_cast<const half*>(packed + std::uint64_t(rows) * columns)[std::uint64_t(column / 32) * rows + row]);
                } else scale = __half2float(block->d);
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    std::uint16_t codes;
                    if constexpr (ColumnPlanes) codes = (pair < 2 ? eight.x : eight.y) >> (16 * (pair % 2));
                    else codes = *reinterpret_cast<const std::uint16_t*>(block->qs + column % 32 + pair * 2);
                    const float lo = float(static_cast<std::int8_t>(codes & 255));
                    const float hi = float(static_cast<std::int8_t>(codes >> 8));
                    const auto values = __floats2bfloat162_rn(__fmul_rn(scale, lo), __fmul_rn(scale, hi));
                    words[pair] = std::uint32_t(__bfloat16_as_ushort(values.x)) | (std::uint32_t(__bfloat16_as_ushort(values.y)) << 16);
                }
            }
            *reinterpret_cast<uint4*>(w + local_row * SharedK + local_column) = decoded;
        }
        if constexpr (AsyncInput) asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < KTile; k += 16) {
            std::uint32_t bparts[OutputFragments][2];
            #pragma unroll
            for (int n = 0; n < OutputFragments; ++n) {
                const auto wb = static_cast<unsigned>(__cvta_generic_to_shared(w + (row_tile + n * 8 + lane % 8) * SharedK + k + ((lane / 8) % 2) * 8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(bparts[n][0]), "=r"(bparts[n][1]) : "r"(wb));
            }
            #pragma unroll
            for (int m = 0; m < MFragments; ++m) {
                std::uint32_t a0, a1, a2, a3;
                const auto xa = static_cast<unsigned>(__cvta_generic_to_shared(x + (token_tile + m * 16 + lane % 16) * SharedK + k + (lane / 16) * 8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(xa));
                #pragma unroll
                for (int n = 0; n < OutputFragments; ++n) {
                    const int f = m * OutputFragments + n;
                    float dot[4];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                        : "=f"(dot[0]), "=f"(dot[1]), "=f"(dot[2]), "=f"(dot[3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bparts[n][0]), "r"(bparts[n][1]), "f"(0.f));
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float add = __fsub_rn(dot[e], correction[f][e]);
                        const float sum = __fadd_rn(precise[f][e], add);
                        correction[f][e] = __fsub_rn(__fsub_rn(sum, precise[f][e]), add);
                        precise[f][e] = sum;
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int f = 0; f < 4; ++f) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const auto token = token_begin + token_tile + (f / OutputFragments) * 16 + lane / 4 + half * 8;
            const auto row = row_begin + row_tile + (f % OutputFragments) * 8 + (lane % 4) * 2;
            if (token < tokens && row < rows) {
                const auto offset = std::uint64_t(DenseSlots ? token : job.slots[token]) * rows + row;
                const auto values = __floats2bfloat162_rn(precise[f][half * 2], precise[f][half * 2 + 1]);
                if (row + 1 < rows && reinterpret_cast<std::uintptr_t>(output + offset) % 4 == 0) {
                    *reinterpret_cast<std::uint32_t*>(output + offset) = std::uint32_t(__bfloat16_as_ushort(values.x)) | (std::uint32_t(__bfloat16_as_ushort(values.y)) << 16);
                } else {
                    output[offset] = values.x;
                    if (row + 1 < rows) output[offset + 1] = values.y;
                }
            }
        }
    }
#else
    asm volatile("trap;");
#endif
}
template<class Index>
__device__ __forceinline__ void prefetch_q8_eight(const GemmJob& job, Index stride,
    Index rows, Index row, Index column,
    std::uint16_t& scale, std::uint16_t (&codes)[4]) {
    scale = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) codes[i] = 0;
    if (row < rows) {
        const auto* block = static_cast<const std::uint8_t*>(job.packed) + std::uint64_t(row) * stride + std::uint64_t(column / 32) * 34;
        const auto* values = block + 2 + column % 32;
        asm volatile("ld.global.u16 %0,[%5]; ld.global.u16 %1,[%6]; ld.global.u16 %2,[%6+2]; ld.global.u16 %3,[%6+4]; ld.global.u16 %4,[%6+6];"
            : "=h"(scale), "=h"(codes[0]), "=h"(codes[1]), "=h"(codes[2]), "=h"(codes[3]) : "l"(block), "l"(values));
    }
}
__device__ __forceinline__ void store_q8_eight(__nv_bfloat16* output, std::uint16_t scale_bits,
    const std::uint16_t (&codes)[4]) {
    uint4 decoded{};
    auto* words = reinterpret_cast<std::uint32_t*>(&decoded);
    const float scale = __half2float(__ushort_as_half(scale_bits));
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const auto values = __floats2bfloat162_rn(__fmul_rn(scale, float(static_cast<std::int8_t>(codes[pair] & 255))),
            __fmul_rn(scale, float(static_cast<std::int8_t>(codes[pair] >> 8))));
        words[pair] = std::uint32_t(__bfloat16_as_ushort(values.x)) | (std::uint32_t(__bfloat16_as_ushort(values.y)) << 16);
    }
    *reinterpret_cast<uint4*>(output) = decoded;
}
template<class Index>
__device__ __forceinline__ void copy_bf16_tile32(const __nv_bfloat16* input, __nv_bfloat16* output,
    Index columns, Index tokens, Index token_begin, Index begin) {
#if __CUDA_ARCH__ >= 800
    const auto token = token_begin + threadIdx.x / 4;
    const int column = (threadIdx.x % 4) * 8;
    const auto address = static_cast<unsigned>(__cvta_generic_to_shared(output + (threadIdx.x / 4) * 40 + column));
    const auto* source = input + std::uint64_t(token < tokens ? token : 0) * columns + begin + column;
    const int bytes = token < tokens ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(address), "l"(source), "r"(bytes));
#endif
}
template<class Index>
__device__ __forceinline__ void prefetch_q8_columns(const GemmJob& job, Index rows, Index columns,
    Index row, Index column, std::uint16_t& scale, uint2& codes) {
    scale = 0; codes = uint2{};
    if (row < rows) {
        const auto* base = static_cast<const std::uint8_t*>(job.packed);
        const auto* values = base + (std::uint64_t(column / 8) * rows + row) * 8;
        const auto* metadata = base + std::uint64_t(rows) * columns + (std::uint64_t(column / 32) * rows + row) * 2;
        asm volatile("ld.global.v2.u32 {%0,%1},[%3]; ld.global.u16 %2,[%4];"
            : "=r"(codes.x), "=r"(codes.y), "=h"(scale) : "l"(values), "l"(metadata));
    }
}
__device__ __forceinline__ void store_q8_columns(__nv_bfloat16* output, std::uint16_t scale_bits, uint2 packed) {
    uint4 decoded{};
    auto* words = reinterpret_cast<std::uint32_t*>(&decoded);
    const float scale = __half2float(__ushort_as_half(scale_bits));
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const auto codes = std::uint16_t((pair < 2 ? packed.x : packed.y) >> (16 * (pair % 2)));
        const auto values = __floats2bfloat162_rn(__fmul_rn(scale, float(static_cast<std::int8_t>(codes & 255))),
            __fmul_rn(scale, float(static_cast<std::int8_t>(codes >> 8))));
        words[pair] = std::uint32_t(__bfloat16_as_ushort(values.x)) | (std::uint32_t(__bfloat16_as_ushort(values.y)) << 16);
    }
    *reinterpret_cast<uint4*>(output) = decoded;
}
// Accumulation: 0 compensated partials, 1 ordinary RN FP32 partials.
template<int MinimumBlocks = 4, class Index = std::uint64_t, bool ColumnPlanes = false, int OutputFragments = 2, int Accumulation = 0>
__global__ __launch_bounds__(256, MinimumBlocks) void q8_bf16_pipeline(GemmJob job, Index stride,
    Index rows, Index columns, const __nv_bfloat16* input, __nv_bfloat16* output) {
#if __CUDA_ARCH__ >= 800
    constexpr int Tile = 64, KTile = 32, SharedK = 40, Plane = Tile * SharedK;
    static_assert(OutputFragments > 0 && 4 % OutputFragments == 0);
    static_assert(Accumulation == 0 || Accumulation == 1);
    constexpr int MFragments = 4 / OutputFragments, RowWarps = 8 / OutputFragments;
    __shared__ __align__(32) __nv_bfloat16 shared[4 * Plane];
    const Index token_begin = Index(blockIdx.y) * Tile, row_begin = Index(blockIdx.x) * Tile;
    const Index tokens = Index(job.count);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int token_tile = (warp / RowWarps) * MFragments * 16;
    const int row_tile = (warp % RowWarps) * OutputFragments * 8;
    const int tile_row = ColumnPlanes ? threadIdx.x % Tile : threadIdx.x / 4;
    const int tile_column = ColumnPlanes ? (threadIdx.x / Tile) * 8 : (threadIdx.x % 4) * 8;
    copy_bf16_tile32<Index>(input, shared, columns, tokens, token_begin, 0);
    asm volatile("cp.async.commit_group;");
    std::uint16_t initial_scale, initial_codes[4];
    uint2 initial_wide{};
    if constexpr (ColumnPlanes) {
        prefetch_q8_columns<Index>(job, rows, columns, row_begin + tile_row, tile_column, initial_scale, initial_wide);
        store_q8_columns(shared + Plane + tile_row * SharedK + tile_column, initial_scale, initial_wide);
    } else {
        prefetch_q8_eight<Index>(job, stride, rows, row_begin + tile_row, tile_column, initial_scale, initial_codes);
        store_q8_eight(shared + Plane + tile_row * SharedK + tile_column, initial_scale, initial_codes);
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    float precise[4][4]{}, correction[4][4]{};
    for (Index begin = 0; begin < columns; begin += KTile) {
        const int current = int((begin / KTile) % 2) * 2 * Plane, next = 2 * Plane - current;
        const bool has_next = begin + KTile < columns;
        std::uint16_t scale, codes[4];
        uint2 wide{};
        if (has_next) {
            if constexpr (ColumnPlanes) prefetch_q8_columns<Index>(job, rows, columns, row_begin + tile_row, begin + KTile + tile_column, scale, wide);
            else prefetch_q8_eight<Index>(job, stride, rows, row_begin + tile_row, begin + KTile + tile_column, scale, codes);
            copy_bf16_tile32<Index>(input, shared + next, columns, tokens, token_begin, begin + KTile);
            asm volatile("cp.async.commit_group;");
        }
        #pragma unroll
        for (int k = 0; k < KTile; k += 16) {
            std::uint32_t bparts[OutputFragments][2];
            #pragma unroll
            for (int n = 0; n < OutputFragments; ++n) {
                const auto wb = static_cast<unsigned>(__cvta_generic_to_shared(shared + current + Plane + (row_tile + n * 8 + lane % 8) * SharedK + k + ((lane / 8) % 2) * 8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(bparts[n][0]), "=r"(bparts[n][1]) : "r"(wb));
            }
            #pragma unroll
            for (int m = 0; m < MFragments; ++m) {
                std::uint32_t a0, a1, a2, a3;
                const auto xa = static_cast<unsigned>(__cvta_generic_to_shared(shared + current + (token_tile + m * 16 + lane % 16) * SharedK + k + (lane / 16) * 8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(xa));
                #pragma unroll
                for (int n = 0; n < OutputFragments; ++n) {
                    const int f = m * OutputFragments + n;
                    float dot[4];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                        : "=f"(dot[0]), "=f"(dot[1]), "=f"(dot[2]), "=f"(dot[3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bparts[n][0]), "r"(bparts[n][1]), "f"(0.f));
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        if constexpr (Accumulation == 0) {
                            const float add = __fsub_rn(dot[e], correction[f][e]);
                            const float sum = __fadd_rn(precise[f][e], add);
                            correction[f][e] = __fsub_rn(__fsub_rn(sum, precise[f][e]), add);
                            precise[f][e] = sum;
                        } else {
                            precise[f][e] = __fadd_rn(precise[f][e], dot[e]);
                        }
                    }
                }
            }
        }
        if (has_next) {
            if constexpr (ColumnPlanes) store_q8_columns(shared + next + Plane + tile_row * SharedK + tile_column, scale, wide);
            else store_q8_eight(shared + next + Plane + tile_row * SharedK + tile_column, scale, codes);
        }
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
    }
    #pragma unroll
    for (int f = 0; f < 4; ++f) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const auto token = token_begin + token_tile + (f / OutputFragments) * 16 + lane / 4 + half * 8;
            const auto row = row_begin + row_tile + (f % OutputFragments) * 8 + (lane % 4) * 2;
            if (token < tokens && row < rows) {
                const auto offset = std::uint64_t(token) * rows + row;
                const auto values = __floats2bfloat162_rn(precise[f][half * 2], precise[f][half * 2 + 1]);
                if (row + 1 < rows && reinterpret_cast<std::uintptr_t>(output + offset) % 4 == 0)
                    *reinterpret_cast<std::uint32_t*>(output + offset) = std::uint32_t(__bfloat16_as_ushort(values.x)) | (std::uint32_t(__bfloat16_as_ushort(values.y)) << 16);
                else {
                    output[offset] = values.x;
                    if (row + 1 < rows) output[offset + 1] = values.y;
                }
            }
        }
    }
#else
    asm volatile("trap;");
#endif
}
// A small output-row tile remains in shared memory while token tiles stream.
// Global storage is still the original Q8_0 blocks, with no decoded cache.
__global__ __launch_bounds__(1024, 1) void q8_bf16_cached_rows(GemmJob job, std::uint64_t stride,
    std::int64_t rows, std::int64_t columns, const __nv_bfloat16* input, __nv_bfloat16* output) {
#if __CUDA_ARCH__ >= 800
    constexpr int Rows = 16, Tokens = 256, KTile = 16, XStride = 24;
    const auto row_begin = std::int64_t(blockIdx.x) * Rows;
    const auto wstride = columns + 8;
    extern __shared__ __align__(32) __nv_bfloat16 cache[];
    auto* w = cache;
    auto* x0 = cache + Rows * wstride;
    auto* x1 = x0 + Tokens * XStride;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int token_tile = (warp / 2) * 16, row_tile = (warp % 2) * 8;
    for (std::int64_t i = std::int64_t(threadIdx.x) * 8; i < Rows * columns; i += std::int64_t(blockDim.x) * 8) {
        const auto row = row_begin + i / columns, column = i % columns;
        uint4 decoded{};
        auto* words = reinterpret_cast<std::uint32_t*>(&decoded);
        if (row < rows) {
            const auto* block = reinterpret_cast<const block_q8_0*>(static_cast<const std::uint8_t*>(job.packed) + row * stride) + column / 32;
            const float scale = __half2float(block->d);
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const auto codes = *reinterpret_cast<const std::uint16_t*>(block->qs + column % 32 + pair * 2);
                const auto values = __floats2bfloat162_rn(__fmul_rn(scale, float(static_cast<std::int8_t>(codes & 255))),
                    __fmul_rn(scale, float(static_cast<std::int8_t>(codes >> 8))));
                words[pair] = std::uint32_t(__bfloat16_as_ushort(values.x)) | (std::uint32_t(__bfloat16_as_ushort(values.y)) << 16);
            }
        }
        *reinterpret_cast<uint4*>(w + (i / columns) * wstride + column) = decoded;
    }
    __syncthreads();
    for (std::int64_t token_begin = 0; token_begin < job.count; token_begin += Tokens) {
        float precise[4]{}, correction[4]{};
        // The producer and MMA phases use disjoint activation buffers.
        for (int i = threadIdx.x * 8; i < Tokens * KTile; i += blockDim.x * 8) {
            const auto token = token_begin + i / KTile;
            const auto address = static_cast<unsigned>(__cvta_generic_to_shared(x0 + (i / KTile) * XStride + i % KTile));
            const auto* source = input + (token < job.count ? token : 0) * columns + i % KTile;
            const int bytes = token < job.count ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address), "l"(source), "r"(bytes));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        for (std::int64_t k = 0; k < columns; k += KTile) {
            auto* current = (k / KTile) % 2 ? x1 : x0;
            auto* next = (k / KTile) % 2 ? x0 : x1;
            if (k + KTile < columns) {
                for (int i = threadIdx.x * 8; i < Tokens * KTile; i += blockDim.x * 8) {
                    const auto token = token_begin + i / KTile;
                    const auto address = static_cast<unsigned>(__cvta_generic_to_shared(next + (i / KTile) * XStride + i % KTile));
                    const auto* source = input + (token < job.count ? token : 0) * columns + k + KTile + i % KTile;
                    const int bytes = token < job.count ? 16 : 0;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address), "l"(source), "r"(bytes));
                }
                asm volatile("cp.async.commit_group;");
            }
            std::uint32_t a0, a1, a2, a3, b0, b1;
            const auto xa = static_cast<unsigned>(__cvta_generic_to_shared(current + (token_tile + lane % 16) * XStride + (lane / 16) * 8));
            const auto wb = static_cast<unsigned>(__cvta_generic_to_shared(w + (row_tile + lane % 8) * wstride + k + ((lane / 8) % 2) * 8));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(xa));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(wb));
            float dot[4];
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                : "=f"(dot[0]), "=f"(dot[1]), "=f"(dot[2]), "=f"(dot[3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float add = __fsub_rn(dot[e], correction[e]);
                const float sum = __fadd_rn(precise[e], add);
                correction[e] = __fsub_rn(__fsub_rn(sum, precise[e]), add);
                precise[e] = sum;
            }
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const auto token = token_begin + token_tile + lane / 4 + half * 8;
            const auto row = row_begin + row_tile + (lane % 4) * 2;
            if (token < job.count && row < rows) {
                const auto offset = token * rows + row;
                const auto values = __floats2bfloat162_rn(precise[half * 2], precise[half * 2 + 1]);
                if (row + 1 < rows && reinterpret_cast<std::uintptr_t>(output + offset) % 4 == 0)
                    *reinterpret_cast<std::uint32_t*>(output + offset) = std::uint32_t(__bfloat16_as_ushort(values.x)) | (std::uint32_t(__bfloat16_as_ushort(values.y)) << 16);
                else {
                    output[offset] = values.x;
                    if (row + 1 < rows) output[offset + 1] = values.y;
                }
            }
        }
    }
#else
    asm volatile("trap;");
#endif
}
__global__ void activate(const std::int64_t* slots, std::int64_t count, std::int64_t width,
    const half* gate, const half* up, half* hidden, float limit) {
    const auto i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count * width) return;
    const auto offset = slots[i / width] * width + i % width;
    float g = __half2float(gate[offset]), u = __half2float(up[offset]);
    if (limit > 0) {
        g = fminf(g, limit); u = fminf(limit, fmaxf(-limit, u));
        hidden[offset] = __float2half_rn(g / (1.f + expf(-g)) * u);
    } else {
        const auto sigmoid = __float2half_rn(1.f / (1.f + expf(-g)));
        const auto silu = __float2half_rn(g * __half2float(sigmoid));
        hidden[offset] = __float2half_rn(__half2float(silu) * u);
    }
}
__global__ void activate_gate_up(const half* gate_up,std::int64_t count,std::int64_t width,
        half* hidden,float limit) {
    const auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count*width)return;
    const auto at=i/width*(2*width)+i%width;
    float g=__half2float(gate_up[at]),u=__half2float(gate_up[at+width]);
    if(limit>0) {
        g=fminf(g,limit);u=fminf(limit,fmaxf(-limit,u));
        const float sigmoid=1.f/(1.f+expf(-g));
        hidden[i]=__float2half_rn(__fmul_rn(__fmul_rn(g,sigmoid),u));
    } else {
        const auto sigmoid=__float2half_rn(1.f/(1.f+expf(-g)));
        const auto silu=__float2half_rn(__fmul_rn(g,__half2float(sigmoid)));
        hidden[i]=__float2half_rn(__fmul_rn(__half2float(silu),u));
    }
}
void checked() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
bool grouped_f16_available(int device) {
    int major = 0;
    const auto status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    // WMMA FP16 instructions were introduced with compute capability 7.0.
    return major >= 7;
}
void grouped_decode_f16(ggml_type type,const GemmJob* jobs,int experts,std::uint64_t stride,
    std::int64_t rows,std::int64_t columns,void* output,cudaStream_t stream,std::int64_t batch_stride) {
    if(!experts || !rows || !columns)return;
    if(!batch_stride)batch_stride=rows*columns;
    if(batch_stride<rows*columns)throw std::invalid_argument("GGUF decoded batch stride overlaps matrices");
    const dim3 grid(unsigned((rows*columns+2047)/2048),unsigned(experts));
    switch(type) {
#define TYLOINFER_GGUF_DECODE_GROUP(name) case GGML_TYPE_##name: \
        decode_group<GGML_TYPE_##name><<<grid,256,0,stream>>>(jobs,stride,rows,columns,batch_stride,static_cast<half*>(output));break;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_DECODE_GROUP)
#undef TYLOINFER_GGUF_DECODE_GROUP
        default:throw std::invalid_argument("unsupported CUDA GGUF grouped decode");
    }
    checked();
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported[GGML_TYPE_COUNT]{};
        if(!reported[type].exchange(true,std::memory_order_relaxed))
            std::fprintf(stderr,"native_gguf_prefill_decode format=%d values_per_thread=8 four_value_groups=%d rows=%lld columns=%lld\n",
                int(type),int(type>=GGML_TYPE_PTQ1_0 && type<=GGML_TYPE_TQK4?2:0),
                static_cast<long long>(rows),static_cast<long long>(columns));
    }
}
void gate_up_swiglu_f16(const void* gate_up,std::int64_t tokens,std::int64_t width,void* hidden,float limit,cudaStream_t stream) {
    if(!tokens || !width)return;
    activate_gate_up<<<unsigned((tokens*width+255)/256),256,0,stream>>>(
        static_cast<const half*>(gate_up),tokens,width,static_cast<half*>(hidden),limit);checked();
}
void grouped_gather_f16(const GemmJob* jobs,int experts,std::int64_t tokens,std::int64_t columns,
    const void* input,std::int64_t routes,void* output,cudaStream_t stream) {
    if(!experts || !tokens || !columns)return;
    gather_group<<<dim3(unsigned((tokens*columns+255)/256),unsigned(experts)),256,0,stream>>>(
        jobs,tokens,columns,static_cast<const half*>(input),routes,static_cast<half*>(output));checked();
}
void grouped_scatter_f16(const GemmJob* jobs,int experts,std::int64_t tokens,std::int64_t columns,
    const void* input,void* output,cudaStream_t stream) {
    if(!experts || !tokens || !columns)return;
    scatter_group<<<dim3(unsigned((tokens*columns+255)/256),unsigned(experts)),256,0,stream>>>(
        jobs,tokens,columns,static_cast<const half*>(input),static_cast<half*>(output));checked();
}
void scatter_upload(const void* packet, int chunks, std::uint64_t maximum_bytes, cudaStream_t stream) {
    if (!chunks) return;
    scatter<<<dim3(unsigned((maximum_bytes+65535)/65536),unsigned(chunks)),256,0,stream>>>(static_cast<const std::uint8_t*>(packet));
    checked();
}
void grouped_matmul_f16(ggml_type type, const GemmJob* jobs, int count, std::int64_t max_tokens,
    std::uint64_t stride, std::int64_t rows, std::int64_t columns,
    const void* input, std::int64_t routes, bool routed_input, void* output, cudaStream_t stream) {
    if (!count || !max_tokens || !rows) return;
    const dim3 grid(unsigned((rows + 31) / 32), unsigned((max_tokens + 31) / 32), unsigned(count));
    switch (type) {
#define TYLOINFER_GGUF_GEMM(name) case GGML_TYPE_##name: \
        grouped<GGML_TYPE_##name><<<grid, 128, 0, stream>>>(jobs, stride, rows, columns, \
            static_cast<const half*>(input), routes, routed_input, static_cast<half*>(output)); break;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_GEMM)
#undef TYLOINFER_GGUF_GEMM
        default: throw std::invalid_argument("unsupported CUDA GGUF grouped GEMM");
    }
    checked();
}
void grouped_matmul_bf16(ggml_type type, const GemmJob* jobs, int count, std::int64_t max_tokens,
    std::uint64_t stride, std::int64_t rows, std::int64_t columns,
    const void* input, std::int64_t routes, bool routed_input, void* output, cudaStream_t stream) {
    if (!count || !max_tokens || !rows) return;
    if (type != GGML_TYPE_Q8_0 || columns <= 0 || columns % 32 || rows < 0 || max_tokens < 0 || routes <= 0)
        throw std::invalid_argument("packed BF16 grouped GEMM currently requires Q8_0 whole-block rows");
    int device = 0, major = 0;
    if (cudaGetDevice(&device) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
        throw std::runtime_error("cannot query packed BF16 GEMM capability");
    if (major < 8) throw std::invalid_argument("BF16 WMMA requires compute capability 8.0 or newer");
    const dim3 grid(unsigned((rows + 63) / 64), unsigned((max_tokens + 63) / 64), unsigned(count));
    grouped<GGML_TYPE_Q8_0, __nv_bfloat16, 64, 64, 8, true, false, 8, 64><<<grid, 256, 0, stream>>>(jobs, stride, rows, columns,
        static_cast<const __nv_bfloat16*>(input), routes, routed_input, static_cast<__nv_bfloat16*>(output));
    checked();
}
void matmul_q8_bf16(const void* packed, std::uint64_t stride, std::int64_t rows,
    std::int64_t columns, std::int64_t tokens, const void* input, void* output, cudaStream_t stream) {
    if (!tokens || !rows) return;
    if (tokens < 0 || rows < 0 || columns <= 0 || columns % 32 ||
        stride < std::uint64_t(columns / 32) * sizeof(block_q8_0) || stride % 2 ||
        !packed || !input || !output || reinterpret_cast<std::uintptr_t>(packed) % 2 ||
        reinterpret_cast<std::uintptr_t>(input) % 16 || reinterpret_cast<std::uintptr_t>(output) % 2)
        throw std::invalid_argument("dense Q8_0 BF16 GEMM requires whole blocks and aligned tile operands");
    int device = 0, major = 0;
    if (cudaGetDevice(&device) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
        throw std::runtime_error("cannot query packed BF16 GEMM capability");
    // BF16 MMA and asynchronous shared copies require compute capability 8.0.
    if (major < 8) throw std::invalid_argument("BF16 MMA requires compute capability 8.0 or newer");
    const GemmJob job{packed, nullptr, tokens};
    const dim3 grid(unsigned((rows + 63) / 64), unsigned((tokens + 63) / 64));
    q8_bf16_mma<true, true, 4, 64, 1, false, std::uint64_t><<<grid, 256, 0, stream>>>(nullptr, job, stride, rows, columns,
        static_cast<const __nv_bfloat16*>(input), 1, true, static_cast<__nv_bfloat16*>(output));
    checked();
}
void repack_q8_bf16_columns(const void* packed, std::uint64_t stride, std::int64_t rows,
    std::int64_t columns, void* output, cudaStream_t stream) {
    if (!rows || !columns) return;
    if (rows < 0 || columns < 0 || columns % 32 || rows > std::numeric_limits<std::int64_t>::max() / columns ||
        stride < std::uint64_t(columns / 32) * sizeof(block_q8_0) || stride % 2 ||
        !packed || !output || packed == output || reinterpret_cast<std::uintptr_t>(packed) % 2 ||
        reinterpret_cast<std::uintptr_t>(output) % 8)
        throw std::invalid_argument("Q8_0 weight column repack requires separate aligned whole-block matrices");
    const auto groups = std::uint64_t(rows) * (columns / 8);
    q8_weight_columns<<<unsigned((groups + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(packed), stride, rows, columns, static_cast<std::uint8_t*>(output));
    checked();
}
Q8Bf16Kernel matmul_q8_bf16_columns(const void* packed, std::int64_t rows, std::int64_t columns,
    std::int64_t tokens, const void* input, void* output, cudaStream_t stream) {
    if (!tokens || !rows) return Q8Bf16Kernel::None;
    if (tokens < 0 || rows < 0 || columns <= 0 || columns % 32 || rows > std::numeric_limits<std::int64_t>::max() / columns ||
        !packed || !input || !output || reinterpret_cast<std::uintptr_t>(packed) % 8 ||
        reinterpret_cast<std::uintptr_t>(input) % 16 || reinterpret_cast<std::uintptr_t>(output) % 2)
        throw std::invalid_argument("column Q8_0 BF16 GEMM requires aligned whole-block matrices");
    int device = 0, major = 0;
    if (cudaGetDevice(&device) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
        throw std::runtime_error("cannot query packed BF16 GEMM capability");
    if (major < 8) throw std::invalid_argument("BF16 MMA and asynchronous copies require compute capability 8.0 or newer");
    const GemmJob job{packed, nullptr, tokens};
    const dim3 grid(unsigned((rows + 63) / 64), unsigned((tokens + 63) / 64));
    // Leave room for tile-tail additions; all address products remain 64-bit.
    constexpr auto narrow_limit = std::uint64_t(std::numeric_limits<std::uint32_t>::max()) - 64;
    if (std::uint64_t(rows) <= narrow_limit && std::uint64_t(columns) <= narrow_limit && std::uint64_t(tokens) <= narrow_limit) {
        q8_bf16_pipeline<4, std::uint32_t, true, 2, 1><<<grid, 256, 0, stream>>>(job, 0, std::uint32_t(rows), std::uint32_t(columns),
            static_cast<const __nv_bfloat16*>(input), static_cast<__nv_bfloat16*>(output));
        checked();
        return Q8Bf16Kernel::Pipeline32Plain;
    }
    q8_bf16_mma<true, true, 4, 64, 2, true, std::uint64_t><<<grid, 256, 0, stream>>>(nullptr, job, 0, rows, columns,
        static_cast<const __nv_bfloat16*>(input), 1, true, static_cast<__nv_bfloat16*>(output));
    checked();
    return Q8Bf16Kernel::Native64;
}
void grouped_swiglu_f16(const std::int64_t* slots, std::int64_t count, std::int64_t width,
    const void* gate, const void* up, void* hidden, float limit, cudaStream_t stream) {
    if (!count) return;
    activate<<<unsigned((count * width + 255) / 256), 256, 0, stream>>>(slots, count, width,
        static_cast<const half*>(gate), static_cast<const half*>(up), static_cast<half*>(hidden), limit);
    checked();
}
}
