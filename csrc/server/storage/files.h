#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>

namespace tyloi::server {

void write_json(const std::filesystem::path& path, const nlohmann::json& value);

} // namespace tyloi::server
