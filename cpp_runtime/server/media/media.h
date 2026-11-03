#pragma once

#include "storage/storage.h"
#include <filesystem>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>

struct sqlite3;

namespace mfq::server {

/** Uploads addressed by their sha256; the SQLite row and the blob file share that one identity. */
class Media {
public:
    struct Error : std::runtime_error {
        enum Kind { invalid, not_found, unprocessable, too_large, unsupported, save_failed } kind;
        std::string code;
        Error(Kind kind, std::string code, std::string message)
            : std::runtime_error(std::move(message)), kind(kind), code(std::move(code)) {}
    };
    // Matches the Python service limit; the HTTP listener raises its payload cap to this size for uploads.
    static constexpr std::size_t max_bytes = 512 * 1024 * 1024;

    Media(Storage& storage, std::filesystem::path root);

    /** Stores one upload, returning the existing resource when the digest is already stored. */
    nlohmann::json put(const std::string& data, const std::string& mime_type, const std::string& expected_sha256) const;
    nlohmann::json get(const std::string& id) const;
    std::filesystem::path path(const std::string& id) const;
    // The sqlite3* overloads join a caller-owned transaction; Storage::access is not reentrant.
    nlohmann::json put(sqlite3* database, const std::string& data, const std::string& mime_type,
        const std::string& expected_sha256) const;
    nlohmann::json get(sqlite3* database, const std::string& id) const;
    std::filesystem::path path(sqlite3* database, const std::string& id) const;

private:
    Storage& storage_;
    std::filesystem::path root_;
};

std::string base64_encode(const std::string& data);
/** Rejects unpadded or non-alphabet input so a damaged archive fails validation instead of truncating. */
std::string base64_decode(const std::string& value);

} // namespace mfq::server
