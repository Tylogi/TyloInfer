#pragma once
#include "models/lfm3/config.h"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace mfq::models::lfm3::d1 {
struct HostWeight {
    std::vector<std::uint16_t> values;
    std::vector<std::uint8_t> packed;
};
class Input {
public:
    Input(const std::filesystem::path& assets, const std::filesystem::path& weights);
    ~Input();
    Input(const Input&) = delete;
    Input& operator=(const Input&) = delete;
    models::lfm3::Config config;
    std::string format;
    double conversion_ms = 0;
    HostWeight load(const std::string& name, const std::vector<std::int64_t>& shape);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
