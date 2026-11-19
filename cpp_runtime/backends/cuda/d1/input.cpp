#include "input.h"
#include "tyloinfer/hf_safetensors_source.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <stdexcept>

namespace mfq::cuda::d1 {
namespace {
using Clock = std::chrono::steady_clock;
std::uint16_t bf16(float x) {
    std::uint32_t bits; std::memcpy(&bits, &x, 4);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return std::uint16_t((bits >> 16) | 0x40U);
    return std::uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
std::string gguf_name(const std::string& name) {
    const std::string root = "model.language_model.";
    if (name == root + "embed_tokens.weight") return "token_embd.weight";
    if (name == root + "embedding_norm.weight") return "output_norm.weight";
    const auto prefix = root + "layers.";
    if (name.rfind(prefix, 0) != 0) throw std::invalid_argument("unknown LFM3 weight name");
    const auto split = name.find('.', prefix.size());
    if (split == std::string::npos) throw std::invalid_argument("invalid LFM3 layer name");
    const auto suffix = name.substr(split + 1);
    static const std::map<std::string, std::string> names{
        {"operator_norm.weight", "attn_norm.weight"}, {"ffn_norm.weight", "ffn_norm.weight"},
        {"post_operator_norm.weight", "post_attention_norm.weight"}, {"post_ffn_norm.weight", "post_ffw_norm.weight"},
        {"feed_forward.w1.weight", "ffn_gate.weight"}, {"feed_forward.w2.weight", "ffn_down.weight"},
        {"feed_forward.w3.weight", "ffn_up.weight"}, {"conv.conv.weight", "shortconv.conv.weight"},
        {"conv.in_proj.weight", "shortconv.in_proj.weight"}, {"conv.out_proj.weight", "shortconv.out_proj.weight"},
        {"self_attn.q_proj.weight", "attn_q.weight"}, {"self_attn.k_proj.weight", "attn_k.weight"},
        {"self_attn.v_proj.weight", "attn_v.weight"}, {"self_attn.o_proj.weight", "attn_output.weight"},
        {"self_attn.gate_proj.weight", "attn_gate.weight"}, {"self_attn.q_norm.weight", "attn_q_norm.weight"},
        {"self_attn.k_norm.weight", "attn_k_norm.weight"}};
    const auto found = names.find(suffix);
    if (found == names.end()) throw std::invalid_argument("unknown LFM3 layer weight");
    return "blk." + name.substr(prefix.size(), split - prefix.size()) + "." + found->second;
}
std::int64_t elements(const std::vector<std::int64_t>& shape) {
    std::int64_t n = 1;
    for (auto d : shape) {
        if (d <= 0 || n > INT64_MAX / d) throw std::invalid_argument("invalid LFM3 weight size");
        n *= d;
    }
    return n;
}
}
struct Input::Impl {
    std::unique_ptr<mfq::HfSafetensorsSource> hf;
    gguf_context* gguf = nullptr;
    ggml_context* tensors = nullptr;
    std::ifstream file;
    ~Impl() { if (gguf) gguf_free(gguf); if (tensors) ggml_free(tensors); }
};
Input::Input(const std::filesystem::path& assets, const std::filesystem::path& weights)
    : impl_(std::make_unique<Impl>()) {
    std::ifstream cfg(assets / "config.json");
    if (!cfg) throw std::runtime_error("cannot open d1 config");
    nlohmann::json document; cfg >> document; config = models::lfm3::Config::parse(document);
    if (weights.extension() != ".gguf") {
        if (!std::filesystem::is_directory(weights)) throw std::invalid_argument("HF weights must be a local directory");
        impl_->hf = std::make_unique<mfq::HfSafetensorsSource>(weights);
        format = "hf-bf16";
        return;
    }
    impl_->gguf = gguf_init_from_file(weights.string().c_str(), {true, &impl_->tensors});
    if (!impl_->gguf) throw std::runtime_error("cannot read GGUF metadata");
    const auto string = [&](const char* key) {
        const auto i = gguf_find_key(impl_->gguf, key);
        if (i < 0 || gguf_get_kv_type(impl_->gguf, i) != GGUF_TYPE_STRING)
            throw std::invalid_argument("GGUF is missing d1 identity metadata");
        return std::string(gguf_get_val_str(impl_->gguf, i));
    };
    if (string("general.architecture") != "lfm3" || string("lfm3.decision.type") != "lfm3-d1")
        throw std::invalid_argument("GGUF must declare the lfm3-d1 decision contract");
    const auto equal = [&](const char* key, int expected) {
        const auto i = gguf_find_key(impl_->gguf, key);
        if (i < 0 || gguf_get_kv_type(impl_->gguf, i) != GGUF_TYPE_UINT32 ||
                gguf_get_val_u32(impl_->gguf, i) != std::uint32_t(expected))
            throw std::invalid_argument("GGUF geometry differs from the supplied HF contract");
    };
    equal("lfm3.block_count", int(config.layers.size())); equal("lfm3.embedding_length", config.hidden);
    equal("lfm3.feed_forward_length", config.intermediate); equal("lfm3.attention.head_count", config.heads);
    equal("lfm3.attention.key_length", config.head_dim); equal("lfm3.shortconv.l_cache", config.convolution);
    equal("lfm3.attention.value_length", config.head_dim); equal("lfm3.context_length", config.context);
    equal("lfm3.attention.sliding_window", config.window); equal("lfm3.rope.dimension_count", config.head_dim);
    const auto real = [&](const char* key, double expected) {
        const auto i = gguf_find_key(impl_->gguf, key);
        if (i < 0 || gguf_get_kv_type(impl_->gguf, i) != GGUF_TYPE_FLOAT32)
            throw std::invalid_argument("GGUF is missing floating LFM3 metadata");
        const double value = gguf_get_val_f32(impl_->gguf, i);
        if (!std::isfinite(value) || std::abs(value - expected) > std::abs(expected) * 1e-6)
            throw std::invalid_argument("GGUF RoPE/norm differs from supplied HF assets");
    };
    real("lfm3.rope.freq_base", config.full_theta); real("lfm3.rope.freq_base_swa", config.sliding_theta);
    real("lfm3.attention.layer_norm_rms_epsilon", config.epsilon);
    const auto array = [&](const char* key, gguf_type type) {
        const auto i = gguf_find_key(impl_->gguf, key);
        if (i < 0 || gguf_get_kv_type(impl_->gguf, i) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(impl_->gguf, i) != type || gguf_get_arr_n(impl_->gguf, i) != config.layers.size())
            throw std::invalid_argument("GGUF is missing per-layer LFM3 metadata");
        return gguf_get_arr_data(impl_->gguf, i);
    };
    const auto* kv = static_cast<const std::int32_t*>(array("lfm3.attention.head_count_kv", GGUF_TYPE_INT32));
    const auto* sliding = static_cast<const bool*>(array("lfm3.attention.sliding_window_pattern", GGUF_TYPE_BOOL));
    for (std::size_t i = 0; i < config.layers.size(); ++i)
        if (kv[i] != (config.layers[i] == "conv" ? 0 : config.kv_heads) || sliding[i] != (config.layers[i] == "sliding_attention"))
            throw std::invalid_argument("GGUF layer order differs from supplied HF assets");
    const auto causal = gguf_find_key(impl_->gguf, "lfm3.attention.causal");
    if (causal < 0 || gguf_get_kv_type(impl_->gguf, causal) != GGUF_TYPE_BOOL || gguf_get_val_bool(impl_->gguf, causal))
        throw std::invalid_argument("GGUF d1 requires noncausal set attention");
    if (config.sliding_factor != 1.0) throw std::invalid_argument("GGUF d1 sliding partial RoPE is unsupported");
    impl_->file.open(weights, std::ios::binary);
    if (!impl_->file) throw std::runtime_error("cannot open GGUF weight payload");
    const auto frequency = gguf_find_tensor(impl_->gguf, "rope_freqs.weight");
    if (frequency < 0 && config.full_factor != 1.0) throw std::invalid_argument("GGUF is missing partial RoPE frequencies");
    if (frequency >= 0) {
        const auto* tensor = ggml_get_tensor(impl_->tensors, "rope_freqs.weight");
        if (!tensor || ggml_nelements(tensor) != config.head_dim / 2 || gguf_get_tensor_type(impl_->gguf, frequency) != GGML_TYPE_F32)
            throw std::invalid_argument("GGUF partial RoPE frequency shape/type mismatch");
        std::vector<float> factors(std::size_t(config.head_dim / 2));
        impl_->file.seekg(gguf_get_data_offset(impl_->gguf) + gguf_get_tensor_offset(impl_->gguf, frequency));
        impl_->file.read(reinterpret_cast<char*>(factors.data()), std::streamsize(factors.size() * sizeof(float)));
        if (!impl_->file) throw std::runtime_error("truncated GGUF partial RoPE frequencies");
        const int active = int(config.head_dim * config.full_factor) / 2;
        for (int i = 0; i < int(factors.size()); ++i)
            if (!std::isfinite(factors[i]) || (i < active ? factors[i] != 1.0f : factors[i] < 1e20f))
                throw std::invalid_argument("GGUF does not use the matching zero-tail partial RoPE");
    }
    format = "gguf-q8_0-packed-w8a16";
}
Input::~Input() = default;
HostWeight Input::load(const std::string& name, const std::vector<std::int64_t>& shape) {
    const auto n = elements(shape);
    HostWeight result;
    if (impl_->hf) {
        result.values.resize(std::size_t(n));
        const auto* t = impl_->hf->find_tensor(name);
        if (!t || !t->shape || *t->shape != shape || t->dtype != "BF16" || t->nbytes != std::uint64_t(n) * 2)
            throw std::invalid_argument("missing or incompatible HF BF16 weight: " + name);
        impl_->hf->read_range_into(name, 0, reinterpret_cast<std::byte*>(result.values.data()), std::size_t(n) * 2);
        return result;
    }
    const auto stored = gguf_name(name);
    const auto i = gguf_find_tensor(impl_->gguf, stored.c_str());
    const auto* tensor = ggml_get_tensor(impl_->tensors, stored.c_str());
    if (i < 0 || !tensor || ggml_nelements(tensor) != n) throw std::invalid_argument("missing GGUF d1 weight: " + stored);
    auto expected = shape;
    if (expected.size() == 3 && expected[1] == 1) expected.erase(expected.begin() + 1);
    for (std::size_t d = 0; d < expected.size(); ++d)
        if (tensor->ne[expected.size() - d - 1] != expected[d]) throw std::invalid_argument("GGUF weight shape differs from d1 contract");
    const auto type = gguf_get_tensor_type(impl_->gguf, i);
    if (type != GGML_TYPE_Q8_0 && type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16)
        throw std::invalid_argument("native d1 GGUF loader currently supports Q8_0 and floating auxiliary weights only");
    const auto count = gguf_get_tensor_size(impl_->gguf, i);
    std::vector<std::uint8_t> bytes(count);
    impl_->file.seekg(gguf_get_data_offset(impl_->gguf) + gguf_get_tensor_offset(impl_->gguf, i));
    impl_->file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(count));
    if (!impl_->file) throw std::runtime_error("truncated GGUF tensor");
    if (type == GGML_TYPE_Q8_0) {
        if (shape.size() != 2 || shape[1] % 32 || count != std::size_t(n / 32) * 34)
            throw std::invalid_argument("Q8_0 d1 weights require complete packed matrix rows");
        result.packed = std::move(bytes);
        return result;
    }
    if (shape.size() == 2) throw std::invalid_argument("GGUF d1 projections and embeddings must stay Q8_0 packed");
    result.values.resize(std::size_t(n));
    const auto start = Clock::now();
    if (type == GGML_TYPE_BF16) {
        if (count != std::size_t(n) * 2) throw std::invalid_argument("invalid BF16 payload size");
        std::memcpy(result.values.data(), bytes.data(), count);
    } else {
        std::vector<float> decoded(std::size_t(n), 0.0f);
        if (type == GGML_TYPE_F32) std::memcpy(decoded.data(), bytes.data(), std::size_t(n) * 4);
        else {
            const auto* traits = ggml_get_type_traits(type);
            if (!traits || !traits->to_float) throw std::invalid_argument("GGUF type has no decoder");
            traits->to_float(bytes.data(), decoded.data(), n);
        }
        std::transform(decoded.begin(), decoded.end(), result.values.begin(), bf16);
    }
    conversion_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return result;
}
}
