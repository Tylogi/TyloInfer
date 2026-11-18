#pragma once
#include "prefill_projection.h"
#include <cuda_runtime_api.h>
struct GgufWeight;
#include <memory>
#include <vector>
// A following operator may restore values at the point of consumption.
// Empty means the ordinary streamed/reference projection should be used.
GgufPrefillProduct gguf_prefill_product(const GgufWeight& weight,
    const mfq_tensor_backend::Tensor& input);

GgufPrefillProduct gguf_rms_gated_product(const GgufWeight& weight,
    const mfq_tensor_backend::Tensor& attended,const mfq_tensor_backend::Tensor& gate,
    const mfq_tensor_backend::Tensor& norm,double eps,bool silu);
GgufPrefillProduct gguf_rms_gated_product(const GgufWeight& weight,
    const mfq_tensor_backend::Tensor& attended,const GgufPrefillProduct& gate,
    const mfq_tensor_backend::Tensor& norm,double eps,bool silu);
// Materialize a scaled [tokens, columns] product for an ordinary FP32 consumer.
mfq_tensor_backend::Tensor gguf_prefill_materialize(const GgufPrefillProduct& product);
std::vector<mfq_tensor_backend::Tensor> gguf_prefill_acc_rms(
    const GgufPrefillProduct& product,const mfq_tensor_backend::Tensor& residual,
    const mfq_tensor_backend::Tensor& norm,double eps,double offset);
