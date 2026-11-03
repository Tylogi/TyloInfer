#include "media.h"
#include "instance.h"
#include "storage/sqlite.h"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>

namespace mfq::server {
namespace {

using Json = nlohmann::json;
using Error = Media::Error;
using sqlite::Statement;
namespace fs = std::filesystem;

std::string timestamp() {
    const auto value = std::time(nullptr);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &value);
#else
    gmtime_r(&value, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S+00:00");
    return out.str();
}

/** Mirrors the Python upload whitelist; document MIME types stay uploadable even without extraction. */
bool supported_mime(const std::string& mime) {
    for (const auto* prefix : {"image/", "audio/", "video/", "text/"})
        if (mime.rfind(prefix, 0) == 0) return true;
    constexpr std::array<const char*, 6> exact{"application/json", "application/pdf", "application/xml",
        "application/yaml", "application/octet-stream",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"};
    return std::any_of(exact.begin(), exact.end(), [&](const char* value) { return value == mime; });
}

std::string normalized_mime(std::string value) {
    value = value.substr(0, value.find(';'));
    const auto finish = value.find_last_not_of(" \t\r\n");
    value.erase(finish == std::string::npos ? 0 : finish + 1);
    value.erase(0, value.find_first_not_of(" \t\r\n"));
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string hex_sha256(const std::string& data) {
    std::array<unsigned char, EVP_MAX_MD_SIZE> bytes;
    unsigned length = 0;
    if (EVP_Digest(data.data(), data.size(), bytes.data(), &length, EVP_sha256(), nullptr) != 1 || length != 32)
        throw Error(Error::save_failed, "checksum_failed", "could not checksum the upload");
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (unsigned index = 0; index < length; ++index) {
        result += digits[bytes[index] >> 4];
        result += digits[bytes[index] & 15];
    }
    return result;
}

Json resource(sqlite3* database, const char* clause, const std::string& value) {
    const auto sql = std::string("SELECT id,sha256,mime_type,byte_size,created_at FROM media WHERE ") + clause;
    Statement query(database, sql.c_str());
    query.bind(1, value);
    if (!query.step()) throw Error(Error::not_found, "media_not_found", "media is not stored: " + value);
    return {{"media", {{"id", query.column(0)}, {"sha256", query.column(1)}, {"mime_type", query.column(2)},
                {"byte_size", std::stoll(query.column(3).get<std::string>())}}},
        {"created_at", query.column(4)}};
}

std::string storage_key(sqlite3* database, const std::string& id) {
    Statement query(database, "SELECT storage_key FROM media WHERE id=?");
    query.bind(1, id);
    if (!query.step()) throw Error(Error::not_found, "media_not_found", "media is not stored: " + id);
    return query.column(0).get<std::string>();
}

} // namespace

Media::Media(Storage& storage, fs::path root) : storage_(storage), root_(std::move(root)) {}

Json Media::put(sqlite3* database, const std::string& data, const std::string& mime_type,
    const std::string& expected_sha256) const {
    const auto mime = normalized_mime(mime_type);
    if (!supported_mime(mime))
        throw Error(Error::unsupported, "unsupported_media_type", "unsupported media or document MIME type");
    if (data.empty()) throw Error(Error::invalid, "empty_media", "media upload must not be empty");
    if (data.size() > max_bytes)
        throw Error(Error::too_large, "media_too_large", "media upload exceeds the " + std::to_string(max_bytes) + " byte limit");
    static const std::regex pattern(R"(^[0-9a-f]{64}$)");
    if (!std::regex_match(expected_sha256, pattern))
        throw Error(Error::invalid, "invalid_request", "X-Content-SHA256 must be 64 lowercase hexadecimal characters");
    const auto digest = hex_sha256(data);
    if (digest != expected_sha256)
        throw Error(Error::unprocessable, "media_digest_mismatch",
            "media digest mismatch: expected " + expected_sha256 + ", computed " + digest);
    try { return resource(database, "sha256=?", digest); } catch (const Error& error) {
        if (error.kind != Error::not_found) throw;
    }
    const auto key = digest.substr(0, 2) + "/" + digest.substr(2, 2) + "/" + digest;
    const auto destination = root_ / fs::u8path(key);
    fs::create_directories(destination.parent_path());
    if (!fs::exists(destination)) {
        const auto temporary = destination.parent_path() / (digest + "." + normalize_uuid({}) + ".upload");
        bool stored = false;
        std::error_code ignored;
        try {
            std::ofstream output(path_utf8(temporary), std::ios::binary);
            output.write(data.data(), static_cast<std::streamsize>(data.size()));
            output.flush();
            if (!output) throw std::runtime_error("cannot write media blob");
            fs::permissions(temporary, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ignored);
            fs::rename(temporary, destination, ignored);
            if (ignored) throw std::runtime_error("cannot replace media blob");
            stored = true;
        } catch (...) {
            if (!stored) fs::remove(temporary, ignored);
            throw Error(Error::save_failed, "media_save_failed", "could not store the media blob");
        }
    }
    Statement insert(database, "INSERT OR IGNORE INTO media(id,sha256,mime_type,byte_size,storage_key,created_at) VALUES(?,?,?,?,?,?)");
    insert.bind(1, normalize_uuid({})); insert.bind(2, digest); insert.bind(3, mime);
    insert.bind(4, std::to_string(data.size())); insert.bind(5, key); insert.bind(6, timestamp());
    insert.step();
    return resource(database, "sha256=?", digest);
}

Json Media::put(const std::string& data, const std::string& mime_type, const std::string& expected_sha256) const {
    auto access = storage_.access();
    return put(access.database, data, mime_type, expected_sha256);
}

Json Media::get(sqlite3* database, const std::string& id) const { return resource(database, "id=?", id); }

Json Media::get(const std::string& id) const {
    auto access = storage_.access();
    return get(access.database, id);
}

fs::path Media::path(sqlite3* database, const std::string& id) const {
    const auto result = root_ / fs::u8path(storage_key(database, id));
    if (!fs::is_regular_file(result)) throw Error(Error::not_found, "media_not_found", "media blob is missing: " + id);
    return result;
}

fs::path Media::path(const std::string& id) const {
    auto access = storage_.access();
    return path(access.database, id);
}

std::string base64_encode(const std::string& data) {
    constexpr char digits[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve((data.size() + 2) / 3 * 4);
    for (std::size_t offset = 0; offset < data.size(); offset += 3) {
        const auto first = static_cast<unsigned char>(data[offset]);
        const auto second = offset + 1 < data.size() ? static_cast<unsigned char>(data[offset + 1]) : 0;
        const auto third = offset + 2 < data.size() ? static_cast<unsigned char>(data[offset + 2]) : 0;
        const unsigned group = (first << 16) | (second << 8) | third;
        result += digits[(group >> 18) & 63];
        result += digits[(group >> 12) & 63];
        result += offset + 1 < data.size() ? digits[(group >> 6) & 63] : '=';
        result += offset + 2 < data.size() ? digits[group & 63] : '=';
    }
    return result;
}

std::string base64_decode(const std::string& value) {
    auto digit = [](char c) {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    if (value.empty() || value.size() % 4 != 0) throw Error(Error::invalid, "invalid_request", "data_base64 must be padded base64");
    std::string result;
    result.reserve(value.size() / 4 * 3);
    for (std::size_t offset = 0; offset < value.size(); offset += 4) {
        const bool final = offset + 4 == value.size();
        int sextets[4]{};
        for (int index = 0; index < 4; ++index) {
            const char c = value[offset + index];
            if (c == '=') {
                if (!final || index < 2) throw Error(Error::invalid, "invalid_request", "data_base64 has invalid base64 padding");
                continue;
            }
            sextets[index] = digit(c);
            if (sextets[index] < 0) throw Error(Error::invalid, "invalid_request", "data_base64 contains invalid base64 data");
        }
        const unsigned group = (unsigned(sextets[0]) << 18) | (unsigned(sextets[1]) << 12) |
            (unsigned(sextets[2]) << 6) | unsigned(sextets[3]);
        result += static_cast<char>((group >> 16) & 255);
        if (value[offset + 2] != '=') result += static_cast<char>((group >> 8) & 255);
        if (value[offset + 3] != '=') result += static_cast<char>(group & 255);
    }
    return result;
}

} // namespace mfq::server
