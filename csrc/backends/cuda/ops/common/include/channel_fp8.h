#pragma once
#include "mfq_tensor_backend.h"
#include <cstdint>
#include <vector>

struct CudaProfiler;
struct ChannelFp8Input {
    mfq_tensor_backend::Tensor source,values,scales;
};
ChannelFp8Input prepare_channel_fp8_input(const mfq_tensor_backend::Tensor& input,CudaProfiler* profiler=nullptr);
struct ChannelFp8Weight {
    mfq_tensor_backend::Tensor values, scales;
    std::int64_t out = 0, neuron_len = 0;
    bool dynamic_activations = false;
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor input,CudaProfiler* profiler=nullptr,
        const ChannelFp8Input* prepared=nullptr) const;
    mfq_tensor_backend::Tensor rows(mfq_tensor_backend::Tensor ids) const;
    mfq_tensor_backend::Tensor dequant() const;
};
ChannelFp8Weight load_channel_fp8(const std::vector<std::uint8_t>& payload, bool cuda, int device);
