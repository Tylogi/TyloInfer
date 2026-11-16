#pragma once

#include "engine/models/minicpmo45/media.h"
#include "engine/ops/ops.h"
#include "vision.h"

struct MiniCPMO45WhisperAttention {
    tyloi::weights::BiasWeight q;
    tyloi::weights::BiasWeight k;
    tyloi::weights::BiasWeight v;
    tyloi::weights::BiasWeight output;
    int64_t heads = 16;
    int64_t head_dim = 64;

    static MiniCPMO45WhisperAttention load(const tyloi::weights::Loader& weights, const std::string &prefix) {
        MiniCPMO45WhisperAttention result;
        result.q = weights.biased(prefix + ".query.weight", prefix + ".query.bias");
        result.k = weights.biased(prefix + ".key.weight");
        result.v = weights.biased(prefix + ".value.weight", prefix + ".value.bias");
        result.output = weights.biased(prefix + ".output.weight", prefix + ".output.bias");
        return result;
    }

    ::tyloi::Tensor forward(::tyloi::Tensor input_value, ::tyloi::Tensor mask_value,
                            ::tyloi::Tensor *key_cache = nullptr,
                            ::tyloi::Tensor *value_cache = nullptr) const {
        auto input = input_value.dense();
        const int64_t batch = input.size(0);
        const int64_t tokens = input.size(1);
        auto query = tyloi::ops::linear(input_value, q.weight, q.bias).dense()
            .reshape({batch, tokens, heads, head_dim}).transpose(1, 2).contiguous();
        auto key = tyloi::ops::linear(input_value, k.weight, k.bias).dense()
            .reshape({batch, tokens, heads, head_dim}).transpose(1, 2).contiguous();
        auto value = tyloi::ops::linear(input_value, v.weight, v.bias).dense()
            .reshape({batch, tokens, heads, head_dim}).transpose(1, 2).contiguous();
        if (key_cache && value_cache) {
            if (key_cache->defined()) {
                key = ::tyloi::cuda::cat({key_cache->dense(), key}, 2).contiguous();
                value = ::tyloi::cuda::cat({value_cache->dense(), value}, 2).contiguous();
            }
            *key_cache = key;
            *value_cache = value;
        }
        std::optional<::tyloi::DenseTensor> attention_mask = std::nullopt;
        if (mask_value.defined()) attention_mask = mask_value.dense().to(query.scalar_type());
        auto attended = mfq_scaled_dot_product_attention(query, key, value, attention_mask, 0.0,
                                                         false, std::nullopt, false)
                            .contiguous().transpose(1, 2)
                            .reshape({batch, tokens, heads * head_dim});
        return tyloi::ops::linear(attended, output.weight, output.bias);
    }
};

struct MiniCPMO45WhisperLayer {
    MiniCPMO45WhisperAttention attention;
    ::tyloi::Tensor attention_norm_weight, attention_norm_bias;
    ::tyloi::Tensor final_norm_weight, final_norm_bias;
    tyloi::weights::BiasWeight fc1;
    tyloi::weights::BiasWeight fc2;
    struct State {
        ::tyloi::Tensor key_cache, value_cache;
    };

    static MiniCPMO45WhisperLayer load(const tyloi::weights::Loader& weights,
                                       int index) {
        const std::string prefix = "audio.block." + std::to_string(index);
        MiniCPMO45WhisperLayer result;
        result.attention = MiniCPMO45WhisperAttention::load(weights, prefix + ".attention");
        result.attention_norm_weight = weights.native(prefix + ".attention.norm.weight");
        result.attention_norm_bias = weights.native(prefix + ".attention.norm.bias");
        result.final_norm_weight = weights.native(prefix + ".mlp.norm.weight");
        result.final_norm_bias = weights.native(prefix + ".mlp.norm.bias");
        result.fc1 = weights.biased(prefix + ".mlp.up.weight", prefix + ".mlp.up.bias");
        result.fc2 = weights.biased(prefix + ".mlp.down.weight", prefix + ".mlp.down.bias");
        return result;
    }

    ::tyloi::Tensor forward(::tyloi::Tensor input_value, ::tyloi::Tensor mask, State& state, bool use_cache) const {
        auto input = input_value.dense();
        auto normalized = minicpmo45_layer_norm(input, attention_norm_weight, attention_norm_bias, 1e-5);
        auto branch = attention.forward(normalized, mask, use_cache ? &state.key_cache : nullptr,
                                        use_cache ? &state.value_cache : nullptr).dense();
        input = (input + branch).contiguous();
        normalized = minicpmo45_layer_norm(input, final_norm_weight, final_norm_bias, 1e-5);
        branch = tyloi::ops::linear(normalized, fc1.weight, fc1.bias).dense();
        branch = tyloi::ops::linear(::tyloi::cuda::gelu(branch), fc2.weight, fc2.bias).dense();
        return (input + branch).contiguous();
    }
};

struct MiniCPMO45AudioEncoder {
    ::tyloi::Tensor conv1_weight, conv1_bias, conv2_weight, conv2_bias, position_embedding;
    std::vector<MiniCPMO45WhisperLayer> layers;
    ::tyloi::Tensor final_norm_weight, final_norm_bias;
    tyloi::weights::BiasWeight projector1;
    tyloi::weights::BiasWeight projector2;

    static MiniCPMO45AudioEncoder load(const tyloi::weights::Loader& weights) {
        MiniCPMO45AudioEncoder result;
        result.conv1_weight = weights.native("audio.patch_embedding.conv1.weight");
        result.conv1_bias = weights.native("audio.patch_embedding.conv1.bias");
        result.conv2_weight = weights.native("audio.patch_embedding.conv2.weight");
        result.conv2_bias = weights.native("audio.patch_embedding.conv2.bias");
        result.position_embedding = weights.native("audio.position_embedding.weight");
        result.final_norm_weight = weights.native("audio.output_norm.weight");
        result.final_norm_bias = weights.native("audio.output_norm.bias");
        result.projector1 = weights.biased("audio.projector.input.weight", "audio.projector.input.bias");
        result.projector2 = weights.biased("audio.projector.output.weight", "audio.projector.output.bias");
        result.layers.reserve(24);
        for (int index = 0; index < 24; ++index) {
            result.layers.push_back(MiniCPMO45WhisperLayer::load(weights, index));
        }
        if (result.conv1_weight.dim() != 3 || result.conv1_weight.size(0) != 1024 ||
            result.conv1_weight.size(1) != 80 || result.conv1_weight.size(2) != 3 ||
            result.conv2_weight.dim() != 3 || result.conv2_weight.size(0) != 1024 ||
            result.conv2_weight.size(1) != 1024 || result.conv2_weight.size(2) != 3 ||
            result.position_embedding.dim() != 2 || result.position_embedding.size(0) != 1500 ||
            result.position_embedding.size(1) != 1024) {
            throw std::runtime_error("MiniCPM-o Whisper tensor shapes disagree with version 4.5");
        }
        return result;
    }

    struct State {
        std::vector<MiniCPMO45WhisperLayer::State> layers;

        explicit State(const MiniCPMO45AudioEncoder& encoder) : layers(encoder.layers.size()) {}

        void reset() {
            for (auto& layer : layers) layer = {};
        }

        int64_t cache_length() const {
            if (layers.empty() || !layers.front().key_cache.defined()) return 0;
            return layers.front().key_cache.size(2);
        }
    };

private:
    ::tyloi::DenseTensor convolve(::tyloi::DenseTensor features) const {
        const auto& weight1 = conv1_weight.dense();
        const auto& bias1 = conv1_bias.dense();
        const auto& weight2 = conv2_weight.dense();
        const auto& bias2 = conv2_bias.dense();
        auto hidden = ::tyloi::cuda::gelu(::tyloi::cuda::conv1d(
            features.to(weight1.scalar_type()), weight1, bias1,
            std::vector<int64_t>{1}, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 1));
        return ::tyloi::cuda::gelu(::tyloi::cuda::conv1d(
            hidden, weight2, bias2,
            std::vector<int64_t>{2}, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 1));
    }

    ::tyloi::DenseTensor project_and_pool(
        ::tyloi::DenseTensor hidden, bool streaming) const {
        hidden = minicpmo45_layer_norm(hidden, final_norm_weight, final_norm_bias, 1e-5);
        hidden = tyloi::ops::linear(hidden, projector1.weight, projector1.bias).dense();
        hidden = ::tyloi::cuda::relu(hidden);
        hidden = tyloi::ops::linear(hidden, projector2.weight, projector2.bias).dense();
        if (streaming && hidden.size(1) < 5)
            throw std::runtime_error("MiniCPM-o streaming audio chunk is too short for stride-5 pooling");
        return ::tyloi::cuda::avg_pool1d(
                   hidden.transpose(1, 2), std::vector<int64_t>{5}, std::vector<int64_t>{5})
            .transpose(1, 2).contiguous();
    }

public:
    ::tyloi::Tensor forward_streaming(::tyloi::Tensor features_value,
                                      int64_t prefix_extra_frames, int64_t suffix_extra_frames, State& state) const {
        if (!features_value.defined())
            throw std::runtime_error("MiniCPM-o streaming audio expects [1,80,frames] and "
                                     "non-negative extra-frame counts");
        const auto& features = features_value.dense();
        if (features.dim() != 3 || features.size(0) != 1 || features.size(1) != 80 ||
            prefix_extra_frames < 0 || suffix_extra_frames < 0) {
            throw std::runtime_error("MiniCPM-o streaming audio expects [1,80,frames] and "
                                     "non-negative extra-frame counts");
        }
        const int64_t conv_tokens_before_crop = (features.size(2) - 1) / 2 + 1;
        if (state.cache_length() + conv_tokens_before_crop >= position_embedding.size(0)) state.reset();
        auto hidden = convolve(features);
        const int64_t prefix_to_remove = prefix_extra_frames > 0 ? (prefix_extra_frames + 1) / 2 : 0;
        const int64_t suffix_to_remove = suffix_extra_frames > 0 ? (suffix_extra_frames + 1) / 2 : 0;
        if (prefix_to_remove + suffix_to_remove >= hidden.size(2))
            throw std::runtime_error("MiniCPM-o streaming audio extra context removes every frame");
        if (prefix_to_remove > 0)
            hidden = hidden.narrow(2, prefix_to_remove, hidden.size(2) - prefix_to_remove);
        if (suffix_to_remove > 0)
            hidden = hidden.narrow(2, 0, hidden.size(2) - suffix_to_remove);
        hidden = hidden.transpose(1, 2).contiguous();
        const int64_t past = state.cache_length();
        const int64_t tokens = hidden.size(1);
        if (past + tokens > position_embedding.size(0))
            throw std::runtime_error("MiniCPM-o streaming Whisper position cache exceeds 1500 frames");
        const auto& position = position_embedding.dense();
        hidden = hidden + position.narrow(0, past, tokens);
        auto mask = ::tyloi::cuda::zeros({1, 1, tokens, past + tokens}, hidden.options());
        for (size_t index = 0; index < layers.size(); ++index) {
            hidden = layers[index].forward(hidden, mask, state.layers[index], true).dense();
        }
        return project_and_pool(
            hidden, true);
    }

    static std::vector<int64_t> pooled_lengths(::tyloi::Tensor raw_lengths) {
        auto lengths = raw_lengths.dense().to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        const auto *values = lengths.data_ptr<int64_t>();
        std::vector<int64_t> result(static_cast<size_t>(lengths.numel()));
        for (int64_t index = 0; index < lengths.numel(); ++index) {
            result[static_cast<size_t>(index)] =
                tyloi::models::minicpmo45::pooled_audio_length(values[index]);
        }
        return result;
    }

    tyloi::StepSequence<::tyloi::Tensor> forward_steps(
        ::tyloi::Tensor features_value, ::tyloi::Tensor raw_lengths_value, State& state, bool use_cache = false) const {
        if (!features_value.defined() || !raw_lengths_value.defined())
            throw std::runtime_error("MiniCPM-o audio input geometry is invalid");
        const auto& features = features_value.dense();
        const auto& raw_lengths = raw_lengths_value.dense();
        if (features.dim() != 3 || features.size(1) != 80 || raw_lengths.dim() != 1 ||
            raw_lengths.size(0) != features.size(0))
            throw std::runtime_error("MiniCPM-o audio input geometry is invalid");
        auto hidden = convolve(features).transpose(1, 2).contiguous();
        const int64_t tokens = hidden.size(1);
        const int64_t past = use_cache ? state.cache_length() : 0;
        if (past + tokens > position_embedding.size(0))
            throw std::runtime_error("MiniCPM-o Whisper position cache exceeds 1500 frames");
        const auto& position = position_embedding.dense();
        hidden = hidden + position.narrow(0, past, tokens);
        // The official graph compares raw feature lengths with post-convolution positions.
        auto length_tensor = raw_lengths.to(::tyloi::kCUDA, ::tyloi::kInt64).contiguous();
        auto key_positions = ::tyloi::cuda::arange(
            past + tokens, ::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64));
        auto valid_keys = key_positions.unsqueeze(0) < (length_tensor + past).unsqueeze(1);
        auto query_positions = ::tyloi::cuda::arange(
            past, past + tokens, ::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64));
        auto chunk_visible = key_positions.unsqueeze(0) <
            ((query_positions / tyloi::models::minicpmo45::audio_chunk_frames + 1) *
             tyloi::models::minicpmo45::audio_chunk_frames).unsqueeze(1);
        auto visible = valid_keys.unsqueeze(1) & chunk_visible.unsqueeze(0);
        auto mask = ::tyloi::cuda::zeros(
            {features.size(0), 1, tokens, past + tokens},
            ::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kFloat32))
            .masked_fill(visible.logical_not().unsqueeze(1), -std::numeric_limits<float>::infinity());
        co_yield tyloi::StepState::advanced;
        for (size_t index = 0; index < layers.size(); ++index) {
            hidden = layers[index].forward(hidden, mask, state.layers[index], use_cache).dense();
            co_yield tyloi::StepState::advanced;
        }
        co_yield ::tyloi::Tensor(project_and_pool(hidden, false));
    }

    ::tyloi::Tensor forward(::tyloi::Tensor features, ::tyloi::Tensor lengths, State& state, bool use_cache = false) const {
        return tyloi::finish_steps(forward_steps(std::move(features), std::move(lengths), state, use_cache));
    }
};
