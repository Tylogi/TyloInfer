#pragma once
#include "mfq_tensor_backend.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

class PrefillInputScope {
    struct Impl;std::unique_ptr<Impl> impl_;
    PrefillInputScope();
public:
    explicit PrefillInputScope(const mfq_tensor_backend::Tensor& input);
    ~PrefillInputScope();
    static bool lookup(const mfq_tensor_backend::Tensor& input,
        mfq_tensor_backend::Tensor& half,mfq_tensor_backend::Tensor& exponents);
    static std::unique_ptr<PrefillInputScope> prepare_rms(
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& weight,
        double eps,double offset,mfq_tensor_backend::Tensor& normalized);
};

struct PrefillProduct {
    mfq_tensor_backend::Tensor values,row_exponents,column_exponents;
    bool defined()const{return values.defined();}
};

struct PrefillWeightImage {
    mfq_tensor_backend::Tensor values,column_exponents;
    bool defined()const{return values.defined() && column_exponents.defined();}
};

// Immutable row scales share ownership and cross-stream readiness rules.
mfq_tensor_backend::Tensor prefill_weight_exponents(
    const mfq_tensor_backend::Tensor& storage,std::int64_t rows,std::int64_t columns,
    const std::vector<std::uintptr_t>& layout,
    const std::function<void(const mfq_tensor_backend::Tensor&)>& prepare);

// An encoding supplies its bounded decoder. Input preparation, GEMM and
// deferred exponent restoration use the same schedule for every encoding.
struct PrefillProjectionWeight {
    mfq_tensor_backend::Tensor storage;
    std::int64_t rows=0,columns=0;
    std::uint64_t decode_row_bytes=0;
    std::function<PrefillWeightImage()> decode;
    std::function<mfq_tensor_backend::Tensor()> decode_f16;
    explicit operator bool()const{return storage.defined() && bool(decode);}
};
struct TqWeight;
struct GgufWeight;
struct NintWeight;
struct NvqWeight;
PrefillProjectionWeight prefill_decoded_projection(const mfq_tensor_backend::Tensor& storage,
    std::int64_t rows,std::int64_t columns,
    std::function<mfq_tensor_backend::Tensor()> decode);
PrefillProjectionWeight prefill_projection_weight(const TqWeight& weight);
PrefillProjectionWeight prefill_projection_weight(const GgufWeight& weight);
PrefillProjectionWeight prefill_projection_weight(const NintWeight& weight);
PrefillProjectionWeight prefill_projection_weight(const NvqWeight& weight);
bool can_prefill_residual(const mfq_tensor_backend::Tensor& storage,
    std::int64_t rows,std::int64_t columns,std::int64_t tokens,
    const mfq_tensor_backend::Tensor& residual);
bool can_prefill_residual(const PrefillProjectionWeight& weight,std::int64_t tokens,
    const mfq_tensor_backend::Tensor& residual);
mfq_tensor_backend::Tensor prefill_residual_projection(const PrefillProjectionWeight& weight,
    const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& residual);
// Encoders provide decoded tiles; scratch allocation and residual epilogues
// share the same bounds, stream lifetime and allocation-failure behavior.
void prefill_residual_gemm(const mfq_tensor_backend::Tensor& input,
    const mfq_tensor_backend::Tensor& weight,const mfq_tensor_backend::Tensor& residual,
    const mfq_tensor_backend::Tensor& output,std::int64_t first,std::int64_t count,
    void* blas,mfq_tensor_backend::Tensor& workspace);

PrefillProduct prefill_product(const PrefillProjectionWeight& weight,
    const mfq_tensor_backend::Tensor& input);
PrefillProduct prefill_rms_gated_product(const PrefillProjectionWeight& weight,
    const mfq_tensor_backend::Tensor& attended,const mfq_tensor_backend::Tensor& gate,
    const mfq_tensor_backend::Tensor& norm,double eps,bool silu);
PrefillProduct prefill_rms_gated_product(const PrefillProjectionWeight& weight,
    const mfq_tensor_backend::Tensor& attended,const PrefillProduct& gate,
    const mfq_tensor_backend::Tensor& norm,double eps,bool silu);
mfq_tensor_backend::Tensor prefill_materialize(const PrefillProduct& product);
std::vector<mfq_tensor_backend::Tensor> prefill_acc_rms(const PrefillProduct& product,
    const mfq_tensor_backend::Tensor& residual,const mfq_tensor_backend::Tensor& norm,
    double eps,double offset);

using GgufPrefillInputScope=PrefillInputScope;
using GgufPrefillProduct=PrefillProduct;
