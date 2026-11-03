#pragma once

#include <nlohmann/json.hpp>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>

namespace mfq::server {

// The optional MiniCPM-o Token2Wav assets, downloaded and verified on demand.
// Files land under <data>/components/minicpmo45-token2wav; a manifest marks readiness.
class VoiceOutput {
public:
    explicit VoiceOutput(const std::filesystem::path& data_root);
    nlohmann::json status() const;
    // Throws on download or verification failure; progress reports completed bytes over total.
    nlohmann::json install(const std::function<void(double, const std::string&)>& progress,
        const std::function<bool()>& cancelled) const;

private:
    std::filesystem::path root_;
    mutable std::mutex mutex_;
};
} // namespace mfq::server
