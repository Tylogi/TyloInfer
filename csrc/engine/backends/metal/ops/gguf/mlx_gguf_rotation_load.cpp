#include "mlx_gguf.h"
#include "mfq_container.h"
#include "nlohmann/json.hpp"

namespace tyloi::metal {
void MlxGgufWeight::configure_hadamard(const MfqContainer& model, const std::string& name) {
    const auto found = model.header().extra_json.find("gguf_hadamard");
    if (found == model.header().extra_json.end()) return;
    const auto rotations = nlohmann::json::parse(found->second);
    if (!rotations.contains(name)) return;
    const auto& entry = rotations.at(name);
    set_hadamard(entry.at("block_size").get<std::int64_t>(), entry.at("signs").get<std::vector<float>>());
}


} // namespace tyloi::metal
