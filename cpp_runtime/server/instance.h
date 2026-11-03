#pragma once

#include <filesystem>
#include <cstdint>
#include <string>
#include <utility>
#include <nlohmann/json.hpp>

namespace mfq::server {
inline std::string path_utf8(const std::filesystem::path& path) {
    const auto value = path.u8string();
    return {value.begin(), value.end()};
}

std::filesystem::path resolved_path(const std::filesystem::path& path);
std::string normalize_uuid(std::string value);

/** Host physical and available memory in bytes; a zero half means the platform does not report it. */
std::pair<std::int64_t, std::int64_t> host_memory_bytes();

class InstanceLock {
public:
    InstanceLock(const std::filesystem::path& data_dir, std::string instance_id, bool studio_owned);
    ~InstanceLock();
    InstanceLock(const InstanceLock&) = delete;
    InstanceLock& operator=(const InstanceLock&) = delete;
    const nlohmann::json& identity() const { return identity_; }

private:
    nlohmann::json identity_;
    int descriptor_ = -1;
};

} // namespace mfq::server
