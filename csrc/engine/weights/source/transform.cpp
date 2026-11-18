#include "transform.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>

namespace tyloi::source {

std::optional<HadamardTransform> hadamard(const Source& source, std::string_view key) {
    const std::string name(key);
    HadamardTransform result;
    if (const auto* cached = source.hadamard_transforms()) {
        const auto found = cached->find(name);
        if (found == cached->end()) return std::nullopt;
        result = found->second;
    } else {
        const auto& metadata = source.metadata();
        const auto found = metadata.find("gguf_hadamard");
        if (found == metadata.end()) {
            const auto required = metadata.find("gguf_requires_hadamard");
            if (required != metadata.end() && required->second == "true")
                throw std::invalid_argument("GGUF Hadamard descriptors are missing");
            return std::nullopt;
        }
        const auto transforms = nlohmann::json::parse(found->second);
        if (!transforms.contains(name)) return std::nullopt;
        result.block_size = transforms.at(name).at("block_size").get<std::int64_t>();
        for (const auto sign : transforms.at(name).at("signs").get<std::vector<int>>()) {
            if (sign != 1 && sign != -1)
                throw std::invalid_argument("GGUF Hadamard sign value differs");
            result.signs.push_back(static_cast<std::int8_t>(sign));
        }
    }
    if (result.block_size < 1 || result.block_size > INT_MAX ||
            (result.block_size & (result.block_size - 1)))
        throw std::invalid_argument("invalid GGUF Hadamard block");
    if (std::any_of(result.signs.begin(), result.signs.end(), [](int sign) {
            return sign != 1 && sign != -1;
        }))
        throw std::invalid_argument("GGUF Hadamard sign value differs");
    return result;
}

} // namespace tyloi::source
