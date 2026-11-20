#include "model.h"
#include "decision.h"

#include "engine/weights/source/gguf.h"
#include "engine/weights/source/json.h"
#include "engine/weights/source/read.h"
#include "hf_source.h"
#include "gguf_source.h"
#include "mfq_source.h"
#include "residual.h"
#include "tq_fixture_source.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdlib>

namespace tyloi {

std::shared_ptr<const ModelSource> open_model_source(
        const std::filesystem::path& path) {
    for (const char* environment :
         {"TYLOINFER_TENSOR_OVERLAY", "TYLOINFER_EXPERT_OVERLAY"}) {
        const char* overlay = std::getenv(environment);
        if (overlay != nullptr && overlay[0] != '\0') {
            throw std::runtime_error(
                std::string(environment) +
                " is unsupported by the canonical ModelSource path");
        }
    }

    const auto filename=path.filename().u8string();
    const std::u8string fixture_suffix=u8".tq-fixture.json";
    if(filename.size()>=fixture_suffix.size() &&
       filename.compare(filename.size()-fixture_suffix.size(),fixture_suffix.size(),fixture_suffix)==0)
        return open_tq_fixture_source(path);
    std::error_code error;
    if (std::filesystem::is_directory(path, error) && !error) {
        return std::make_shared<models::HfSource>(path);
    }
    if (path.extension() == ".json") {
        const auto document = source::parse_json(source::read_text(path), path);
        if (document.is_object() &&
            document.value("format", std::string{}) ==
                "tyloinfer-row-residual-v1") {
            return models::open_residual_model_source(path);
        }
    }
    if (is_gguf_file(path)) return std::make_shared<models::GgufSource>(path);
    return std::make_shared<models::MfqSource>(path);
}

namespace models {
void validate_model_source(const tyloi::ModelSource& source) {
    const auto* embedding = source.find_tensor("model.token_embedding.weight");
    if (!embedding) {
        throw std::runtime_error(
            "canonical text tensor inventory has no token embedding");
    }
    const auto model_graph = tyloi::models::graph(source);
    for (const auto& component : model_graph.components) {
        if (component.tensor_root != "runtime" &&
                !has_tensor_prefix(source, component.tensor_root)) {
            throw std::runtime_error(
                "model graph declares " + component.kind +
                " but its canonical tensor inventory is missing");
        }
    }
    if (const auto decision = decision_spec(source)) {
        if (model_graph.backbone != "qwen3_5" && model_graph.backbone != "generic_qwen")
            throw std::invalid_argument("Jev decisions require a Qwen3.5/Qwen3.8 backbone");
        if (!decision->vocabulary_readout) {
            if (!embedding->shape || embedding->shape->size() != 2 ||
                *std::max_element(decision->token_ids.begin(), decision->token_ids.end()) >= (*embedding->shape)[0])
                throw std::invalid_argument("decision token IDs exceed the model vocabulary");
        }
    }
}
} // namespace models

} // namespace tyloi
