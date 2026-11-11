#pragma once
#include "engine/backends/cuda/native/tensor/backend.h"
#include <cuda_runtime_api.h>
struct GgufWeight;
#include <memory>
#include <vector>
class GgufPrefillInputScope {
    struct Impl;std::unique_ptr<Impl> impl_;
    GgufPrefillInputScope();
public:
    explicit GgufPrefillInputScope(const ::tyloi::DenseTensor& input);
    ~GgufPrefillInputScope();
    static bool lookup(const ::tyloi::DenseTensor& input,
        ::tyloi::DenseTensor& half,::tyloi::DenseTensor& exponents);
    static std::unique_ptr<GgufPrefillInputScope> prepare_rms(
        const ::tyloi::DenseTensor& input,const ::tyloi::DenseTensor& weight,
        double eps,double offset,::tyloi::DenseTensor& normalized);
};
::tyloi::DenseTensor gguf_rms_gated_prefill(const GgufWeight& weight,
    const ::tyloi::DenseTensor& attended,const ::tyloi::DenseTensor& gate,
    const ::tyloi::DenseTensor& norm,double eps,bool silu);

struct GgufPrefillProduct {
    ::tyloi::DenseTensor values, row_exponents, column_exponents;
    bool defined() const { return values.defined(); }
};
// A following operator may restore values at the point of consumption.
// Empty means the ordinary streamed/reference projection should be used.
GgufPrefillProduct gguf_prefill_product(const GgufWeight& weight,
    const ::tyloi::DenseTensor& input);

GgufPrefillProduct gguf_rms_gated_product(const GgufWeight& weight,
    const ::tyloi::DenseTensor& attended,const ::tyloi::DenseTensor& gate,
    const ::tyloi::DenseTensor& norm,double eps,bool silu);
GgufPrefillProduct gguf_rms_gated_product(const GgufWeight& weight,
    const ::tyloi::DenseTensor& attended,const GgufPrefillProduct& gate,
    const ::tyloi::DenseTensor& norm,double eps,bool silu);
// Materialize a scaled [tokens, columns] product for an ordinary FP32 consumer.
::tyloi::DenseTensor gguf_prefill_materialize(const GgufPrefillProduct& product);
std::vector<::tyloi::DenseTensor> gguf_prefill_acc_rms(
    const GgufPrefillProduct& product,const ::tyloi::DenseTensor& residual,
    const ::tyloi::DenseTensor& norm,double eps,double offset);
