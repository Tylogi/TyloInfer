#pragma once

#include "nlohmann/json.hpp"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace mfq::models::lfm3 {

struct Config {
    int hidden = 0, intermediate = 0, heads = 0, kv_heads = 0, head_dim = 0;
    int vocabulary = 0, context = 0, convolution = 0, window = 0, set_size = 0;
    float epsilon = 0;
    double full_theta = 0, sliding_theta = 0, full_factor = 1, sliding_factor = 1;
    bool sandwich = true;
    std::vector<std::string> layers;

    static Config parse(const nlohmann::json& root) {
        const auto require = [](bool ok, const char* text) {
            if (!ok) throw std::invalid_argument(text);
        };
        require(root.is_object() && root.value("model_type", std::string{}) == "lfm3_vl" &&
                root.at("architectures") == nlohmann::json::array({"D1Model"}),
                "native d1 requires a D1Model/LFM3-VL config");
        require(root.value("d1_attention", std::string{}) == "set", "native d1 requires set attention");
        require(root.value("tie_word_embeddings", false), "native d1 requires tied output weights");
        const auto& t = root.at("text_config");
        require(t.value("model_type", std::string{}) == "lfm3", "native d1 requires LFM3 text layers");
        Config c;
        const auto positive = [&](const char* key) {
            const auto& v = t.at(key);
            require(v.is_number_integer() && !v.is_boolean(), "LFM3 geometry must be integer");
            const auto n = v.get<std::int64_t>();
            require(n > 0 && n <= 0x7fffffff, "LFM3 geometry is out of range");
            return int(n);
        };
        c.hidden = positive("hidden_size"); c.intermediate = positive("intermediate_size");
        c.heads = positive("num_attention_heads"); c.kv_heads = positive("num_key_value_heads");
        c.head_dim = positive("head_dim"); c.vocabulary = positive("vocab_size");
        c.context = positive("max_position_embeddings"); c.convolution = positive("conv_L_cache");
        c.window = positive("sliding_window");
        c.epsilon = t.at("norm_eps").get<float>();
        require(std::isfinite(c.epsilon) && c.epsilon > 0, "invalid LFM3 norm epsilon");
        require(c.heads % c.kv_heads == 0 && c.head_dim % 2 == 0, "invalid LFM3 GQA geometry");
        require(!t.value("conv_bias", false), "native d1 currently requires unbiased convolution projections");
        require(t.value("tie_word_embeddings", false), "native d1 requires tied language embeddings");
        c.sandwich = t.value("use_sandwich_norm", true);
        c.layers = t.at("layer_types").get<std::vector<std::string>>();
        require(int(c.layers.size()) == positive("num_hidden_layers"), "LFM3 layer count mismatch");
        for (const auto& kind : c.layers)
            require(kind == "conv" || kind == "full_attention" || kind == "sliding_attention", "unknown LFM3 mixer");
        const auto rope = [&](const char* kind, double& theta, double& factor) {
            const auto& r = t.at("rope_parameters").at(kind);
            require(r.value("rope_type", std::string{}) == "default", "native d1 supports default RoPE only");
            theta = r.at("rope_theta").get<double>(); factor = r.value("partial_rotary_factor", 1.0);
            require(std::isfinite(theta) && theta > 0 && std::isfinite(factor) && factor > 0 && factor <= 1,
                    "invalid LFM3 RoPE parameters");
        };
        rope("full_attention", c.full_theta, c.full_factor);
        rope("sliding_attention", c.sliding_theta, c.sliding_factor);
        const auto& size = root.at("d1_set_size");
        require(size.is_number_integer() && size.get<std::int64_t>() >= 2 && size.get<std::int64_t>() <= 128,
                "invalid trained d1 set size");
        c.set_size = size.get<int>();
        return c;
    }
};

// The model owns the residual/norm order; the backend supplies numerical ops.
template<class Ops, class Tensor>
Tensor forward(const Config& config, Ops& ops, Tensor input) {
    auto x = std::move(input);
    for (std::size_t layer = 0; layer < config.layers.size(); ++layer) {
        auto lease = ops.enter_layer(layer);
        auto normalized = ops.normalize(x, layer, "operator_norm");
        auto mixed = config.layers[layer] == "conv"
            ? ops.convolve(normalized, layer) : ops.attend(normalized, layer);
        if (config.sandwich) mixed = ops.normalize(mixed, layer, "post_operator_norm");
        x = ops.add(x, mixed);
        auto ffn = ops.feed_forward(ops.normalize(x, layer, "ffn_norm"), layer);
        if (config.sandwich) ffn = ops.normalize(ffn, layer, "post_ffn_norm");
        x = ops.add(x, ffn);
    }
    return ops.finalize(x);
}

} // namespace mfq::models::lfm3
