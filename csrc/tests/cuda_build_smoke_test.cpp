#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/kernels/mfq_cuda_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/models/deepseek_v4/model.h"
#include "engine/ops.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    tyloi::cuda::Executor executor({}, 0);
    auto scope = executor.enter();
    const auto options = tyloi::TensorOptions().device(tyloi::Device(tyloi::kCUDA, 0)).dtype(tyloi::kFloat32);
    const auto close = [](const tyloi::DenseTensor& actual, const std::vector<float>& expected) {
        auto host = actual.to(tyloi::kCPU, tyloi::kFloat32).contiguous();
        assert(host.numel() == static_cast<std::int64_t>(expected.size()));
        for (std::size_t i = 0; i < expected.size(); ++i)
            assert(std::abs(host.data_ptr<float>()[i] - expected[i]) < 1e-5f);
    };

    auto logits = tyloi::cuda::tensor(std::vector<float>{1, 3, 2}, options).reshape({1, 3});
    assert(sample_greedy_cuda(logits).cpu().item<std::int64_t>() == 1);
    auto q = tyloi::cuda::ones({1, 1, 1, 32}, options);
    auto gate = tyloi::cuda::zeros({1, 1, 1}, options);
    auto beta = tyloi::cuda::ones({1, 1, 1}, options);
    auto recurrent = gdn_cuda(q, q, q, gate, beta, std::nullopt);
    close(recurrent[0], std::vector<float>(32, std::sqrt(32.f)));
    close(recurrent[1], std::vector<float>(32 * 32, 1.f));

    auto input = tyloi::cuda::tensor(std::vector<float>{1, 3, 5, 7}, options).reshape({1, 1, 1, 4});
    auto positions = tyloi::cuda::zeros({1}, options.dtype(tyloi::kInt32));
    auto frequencies = tyloi::cuda::ones({2}, options);
    auto axes = tyloi::cuda::zeros({2}, options.dtype(tyloi::kInt32));
    close(rotary_embedding_cached_cuda(input, positions, frequencies, axes, 8), {1, 3, 5, 7});

    tyloi::cuda::deepseek_v4::Model model;
    model.metadata.hidden_size = model.config.hidden_size = 2;
    model.metadata.hc_mult = model.config.hc_mult = 2;
    model.metadata.rms_norm_eps = model.config.rms_norm_eps = 0;
    model.config.hc_eps = 0;
    model.output_head.function = tyloi::cuda::zeros({2, 4}, options);
    model.output_head.scale = tyloi::cuda::ones({2}, options);
    model.output_head.base = tyloi::cuda::zeros({2}, options);
    model.output_norm = tyloi::cuda::ones({2}, options);
    model.embed = tyloi::cuda::tensor(std::vector<float>{1, 0, 0, 1}, options).reshape({2, 2});
    model.lm_head = model.embed;
    const tyloi::models::CausalModel& language = model;
    auto ids = tyloi::cuda::ones({1}, options.dtype(tyloi::kInt64));
    close(language.embedding(ids).dense(), {0, 1});
    const std::vector<float> expected{3 / std::sqrt(17.f), 5 / std::sqrt(17.f)};
    auto hidden = language.normalize(input.reshape({1, 1, 2, 2}), 1, 1);
    close(hidden.dense(), expected);
    auto collapsed = tyloi::cuda::tensor(std::vector<float>{3, 5}, options).reshape({1, 1, 2});
    close(language.normalize(collapsed, 1, 1).dense(), expected);
    close(language.logits(hidden).dense(), expected);
    executor.drain();
    std::cout << "CUDA build smoke passed: sampling, GDN state, RoPE, DeepSeek V4 output\n";
}
