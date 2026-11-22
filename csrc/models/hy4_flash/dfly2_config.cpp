#include "models/hy4_flash/dfly2_config.h"

#include "nlohmann/json.hpp"
#include "tyloinfer/model_source.h"

#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace mfq::models::hy4_flash {
namespace {

std::vector<int64_t> weight_shape(const ModelSource& source, const std::string& name) {
    const auto* tensor = source.find_tensor(name);
    if (!tensor || !tensor->shape) throw std::runtime_error("missing DFly2 weight: " + name);
    return *tensor->shape;
}

void require_shape(const ModelSource& source, const std::string& name,
                   const std::vector<int64_t>& expected) {
    if (weight_shape(source, name) != expected)
        throw std::runtime_error("DFly2 weight shape differs from the model: " + name);
}

} // namespace

Dfly2Config Dfly2Config::from_sources(const ModelSource& target, const ModelSource& draft) {
    using json = nlohmann::json;
    const auto target_config = json::parse(target.model_config_json());
    const auto draft_config = json::parse(draft.model_config_json());
    if (target_config.value("model_type", "") != "hy_v4_flash" || draft_config.value("model_type", "") != "hy_v4_flash")
        throw std::runtime_error("DFly2 requires matching Hy4 Flash checkpoints");
    const auto& draft_text = draft_config.at("text_config");
    const auto& layer_types = target_config.at("layer_types");
    const auto layer_count = draft_config.at("dfly_num_hidden_layers").get<int64_t>();
    const auto target_layers = target_config.at("num_hidden_layers").get<int64_t>();
    const auto storage_layers = draft_config.at("dfly_target_kv_layers").get<std::vector<int64_t>>();
    const auto index_layers = draft_config.at("dfly_target_index_layer").get<std::vector<int64_t>>();
    const auto vocab = target_config.at("vocab_size").get<int64_t>();
    const auto q_rank = draft_text.at("q_lora_rank").get<int64_t>();
    const auto kv_rank = draft_text.at("kv_lora_rank").get<int64_t>();
    const auto q_dim = draft_text.at("qk_head_dim").get<int64_t>();
    const auto rope_dim = draft_text.at("qk_rope_head_dim").get<int64_t>();
    const auto v_dim = draft_text.at("v_head_dim").get<int64_t>();
    const auto& target_index = target_config.at("index_pool");

    Dfly2Config config;
    config.hidden_size = target_config.at("hidden_size").get<int64_t>();
    config.target_top_k = draft_config.at("dfly_target_index_topk").get<int64_t>();
    config.mask_token_id = draft_config.at("dfly_mask_token_id").get<int64_t>();
    config.hidden_correction = draft_config.value("dfly_enable_hidden_correction", false);
    config.fusion = draft_config.value("dfly_mtp_fusion", std::string("none"));
    if (const auto window = draft_config.find("dfly_context_local_window"); window != draft_config.end() && !window->is_null())
        config.recent_window = window->get<int64_t>();
    if (layer_count <= 0 || target_layers <= 0 || uint64_t(layer_count) != storage_layers.size() ||
        storage_layers.size() != index_layers.size() || !layer_types.is_array() ||
        uint64_t(target_layers) != layer_types.size() || draft_text.at("num_hidden_layers") != target_layers ||
        config.hidden_size <= 0 || config.hidden_size > std::numeric_limits<int64_t>::max() / 2 ||
        draft_text.at("hidden_size") != config.hidden_size || draft_text.at("vocab_size") != vocab ||
        config.target_top_k <= 0 || target_index.at("token_budget") != config.target_top_k ||
        config.mask_token_id < 0 || config.mask_token_id >= vocab || config.recent_window < 0 ||
        q_rank <= 0 || kv_rank <= 0 || rope_dim <= 0 || rope_dim >= q_dim || v_dim <= 0 ||
        kv_rank > std::numeric_limits<int64_t>::max() - rope_dim ||
        q_dim - rope_dim > std::numeric_limits<int64_t>::max() - v_dim)
        throw std::runtime_error("DFly2 draft geometry is incompatible with the target");

    const std::string prefix = "model.dfly_layers.";
    const auto hidden = config.hidden_size;
    require_shape(draft, prefix + "final_norm.weight", {hidden});
    if (config.fusion != "none") {
        require_shape(draft, prefix + "mtp_fusion.eh_proj.weight", {hidden, 2 * hidden});
        require_shape(draft, prefix + "mtp_fusion.enorm.weight", {hidden});
        require_shape(draft, prefix + "mtp_fusion.hnorm.weight", {hidden});
        require_shape(draft, prefix + "mtp_fusion.gate", {hidden});
    }
    if (config.hidden_correction) {
        const auto root = prefix + "hidden_correction.";
        const auto down = weight_shape(draft, root + "down_proj.weight");
        if (down.size() != 2 || down[0] != hidden || down[1] <= 0)
            throw std::runtime_error("invalid DFly2 correction geometry");
        require_shape(draft, root + "gate_proj.weight", {down[1], 2 * hidden});
        require_shape(draft, root + "up_proj.weight", {down[1], 2 * hidden});
        require_shape(draft, root + "embed_norm.weight", {hidden});
        require_shape(draft, root + "hidden_norm.weight", {hidden});
    }

    std::unordered_set<int64_t> kv_seen;
    for (std::size_t layer = 0; layer < storage_layers.size(); ++layer) {
        const auto kv_layer = storage_layers[layer], index_layer = index_layers[layer];
        const auto indexed = [&](int64_t id) {
            return id >= 0 && id < target_layers && layer_types[id] == "index_pool_attention";
        };
        if (!indexed(kv_layer) || !indexed(index_layer) || !kv_seen.insert(kv_layer).second)
            throw std::runtime_error("DFly2 requires a distinct target KV layer and an IndexPool selection for each draft layer");
        const auto root = prefix + std::to_string(layer) + ".";
        const auto q = weight_shape(draft, root + "self_attn.q_b_proj.weight");
        const auto ffn = weight_shape(draft, root + "mlp.gate_proj.weight");
        if (q.size() != 2 || q[0] <= 0 || q[0] % q_dim || q[1] != q_rank ||
            ffn.size() != 2 || ffn[0] <= 0 || ffn[1] != hidden ||
            (config.intermediate_size && ffn[0] != config.intermediate_size))
            throw std::runtime_error("invalid DFly2 attention or FFN geometry");
        config.intermediate_size = ffn[0];
        const auto heads = q[0] / q_dim;
        const auto kv_expanded = q_dim - rope_dim + v_dim;
        if (heads > std::numeric_limits<int64_t>::max() / kv_expanded)
            throw std::runtime_error("invalid DFly2 value projection width");
        const auto draft_values = heads * v_dim;
        const auto target_root = "model.block." + std::to_string(kv_layer) + ".attention.";
        require_shape(target, target_root + "key_value_a.weight", {kv_rank + rope_dim, hidden});
        require_shape(target, target_root + "query_b.weight", q);
        require_shape(draft, root + "self_attn.kv_a_proj_with_mqa.weight", {kv_rank + rope_dim, hidden});
        require_shape(draft, root + "self_attn.kv_b_proj.weight", {heads * kv_expanded, kv_rank});
        require_shape(draft, root + "self_attn.q_a_proj.weight", {q_rank, hidden});
        require_shape(draft, root + "self_attn.linear_gate.weight", {draft_values, hidden});
        require_shape(draft, root + "self_attn.o_proj.weight", {hidden, draft_values});
        require_shape(draft, root + "mlp.up_proj.weight", {config.intermediate_size, hidden});
        require_shape(draft, root + "mlp.down_proj.weight", {hidden, config.intermediate_size});
        config.layers.push_back({kv_layer, index_layer, heads});
    }
    return config;
}

} // namespace mfq::models::hy4_flash
