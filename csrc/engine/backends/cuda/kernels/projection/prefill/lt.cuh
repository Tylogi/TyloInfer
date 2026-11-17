#pragma once

#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/device.h"

#include <cublasLt.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace tyloi::cuda::prefill {

struct GemmDevice {
    int device = 0;
    cublasLtHandle_t handle = nullptr;

    ~GemmDevice() {
        int previous = device;
        const auto status = cudaGetDevice(&previous);
        if (status == cudaSuccess && previous != device) (void)cudaSetDevice(device);
        if (handle) (void)cublasLtDestroy(handle);
        if (status == cudaSuccess && previous != device) (void)cudaSetDevice(previous);
    }
};

struct GemmPlan {
    std::shared_ptr<GemmDevice> owner;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    std::size_t workspace = 0;
    bool half_accum = false;
    std::int64_t tokens = 0, rows = 0, columns = 0, stride = 0;

    ~GemmPlan() {
        if (a) (void)cublasLtMatrixLayoutDestroy(a);
        if (b) (void)cublasLtMatrixLayoutDestroy(b);
        if (c) (void)cublasLtMatrixLayoutDestroy(c);
        if (operation) (void)cublasLtMatmulDescDestroy(operation);
    }
};

inline void check_lt(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("prefill GEMM planning failed: " + std::to_string(int(status)));
}

inline unsigned alignment(const void* pointer) {
    unsigned value = 256;
    while (reinterpret_cast<std::uintptr_t>(pointer) % value) value /= 2;
    return value;
}

namespace detail {
inline std::shared_ptr<GemmPlan> prepare(const void* input, const void* weight,
        const void* residual, void* output, std::int64_t tokens, std::int64_t rows,
        std::int64_t columns, std::int64_t stride, cudaStream_t stream, bool half_accum) {
    int device = 0;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
    const auto aa = alignment(weight), ab = alignment(input), ac = alignment(residual), ad = alignment(output);
    using Key = std::tuple<int, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        unsigned, unsigned, unsigned, unsigned, bool>;
    struct Entry { Key key; std::shared_ptr<GemmPlan> plan; };
    static thread_local std::list<Entry> cache;
    static thread_local std::unordered_map<int, std::weak_ptr<GemmDevice>> devices;
    const Key key{device, tokens, rows, columns, stride, aa, ab, ac, ad, half_accum};
    for (auto entry = cache.begin(); entry != cache.end(); ++entry) {
        if (entry->key == key) {
            auto plan = entry->plan;
            cache.splice(cache.begin(), cache, entry);
            return plan;
        }
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) return {};
    const auto remember = [&](std::shared_ptr<GemmPlan> plan) {
        cache.push_front({key, plan});
        if (cache.size() > 64) cache.pop_back();
        return plan;
    };
    auto owner = devices[device].lock();
    if (!owner) {
        owner = std::make_shared<GemmDevice>();
        owner->device = device;
        const auto status = cublasLtCreate(&owner->handle);
        if (status == CUBLAS_STATUS_ALLOC_FAILED || status == CUBLAS_STATUS_NOT_SUPPORTED) return {};
        check_lt(status);
        devices[device] = owner;
    }
    auto plan = std::make_shared<GemmPlan>();
    plan->owner = std::move(owner);
    plan->half_accum = half_accum;
    plan->tokens = tokens;
    plan->rows = rows;
    plan->columns = columns;
    plan->stride = stride;
    check_lt(cublasLtMatmulDescCreate(&plan->operation,
        half_accum ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
        half_accum ? CUDA_R_16F : CUDA_R_32F));
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    check_lt(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    check_lt(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    check_lt(cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_16F, columns, rows, columns));
    check_lt(cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_16F, columns, tokens, columns));
    check_lt(cublasLtMatrixLayoutCreate(&plan->c, half_accum ? CUDA_R_16F : CUDA_R_32F,
        rows, tokens, stride));
    struct Preference {
        cublasLtMatmulPreference_t value = nullptr;
        ~Preference() { if (value) (void)cublasLtMatmulPreferenceDestroy(value); }
    } preference;
    check_lt(cublasLtMatmulPreferenceCreate(&preference.value));
    const std::size_t limit = 32u << 20;
    check_lt(cublasLtMatmulPreferenceSetAttribute(preference.value,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
    for (auto item : {std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, aa},
                      std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, ab},
                      std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, ac},
                      std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, ad}})
        check_lt(cublasLtMatmulPreferenceSetAttribute(preference.value, item.first, &item.second, sizeof(item.second)));
    cublasLtMatmulHeuristicResult_t algorithm{};
    int count = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(plan->owner->handle, plan->operation,
        plan->a, plan->b, plan->c, plan->c, preference.value, 1, &algorithm, &count);
    if (status == CUBLAS_STATUS_ALLOC_FAILED || status == CUBLAS_STATUS_NOT_SUPPORTED) return {};
    check_lt(status);
    if (!count || algorithm.state != CUBLAS_STATUS_SUCCESS || algorithm.workspaceSize > limit)
        return remember({});
    plan->algorithm = algorithm.algo;
    plan->workspace = algorithm.workspaceSize;
    return remember(std::move(plan));
}
} // namespace detail

// Metadata is reusable across immutable weight buffers. Scratch remains owned by
// the caller so this module never retains a tensor allocation.
inline std::shared_ptr<GemmPlan> prepare_gemm(const void* input, const void* weight, void* output,
        std::int64_t tokens, std::int64_t rows, std::int64_t columns, std::int64_t stride,
        cudaStream_t stream, bool half_accum = true) {
    return detail::prepare(input, weight, output, output, tokens, rows, columns, stride, stream, half_accum);
}

inline void gemm(const GemmPlan& plan, const void* input, const void* weight, void* output,
        void* workspace, cudaStream_t stream) {
    const float alpha = 1.f, beta = 0.f;
    const __half half_alpha = __float2half(1.f), half_beta = __float2half(0.f);
    check_lt(cublasLtMatmul(plan.owner->handle, plan.operation,
        plan.half_accum ? static_cast<const void*>(&half_alpha) : &alpha, weight, plan.a, input, plan.b,
        plan.half_accum ? static_cast<const void*>(&half_beta) : &beta,
        output, plan.c, output, plan.c, &plan.algorithm, workspace, plan.workspace, stream));
}

// GGUF residual prefill uses the same canonical descriptor cache with a distinct
// C alignment and a beta=1 epilogue.
inline std::shared_ptr<GemmPlan> prepare_residual(const void* input, const void* weight,
        const void* residual, void* output, std::int64_t tokens, std::int64_t rows,
        std::int64_t columns, std::int64_t stride, cudaStream_t stream) {
    return detail::prepare(input, weight, residual, output, tokens, rows, columns, stride, stream, true);
}

} // namespace tyloi::cuda::prefill
