#include "components.h"
#include "api/errors.h"
#include "mfq_paged_prefix_cache.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>

#include <curl/curl.h>

namespace mfq::server {
namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr const char* component_id = "minicpmo45-token2wav";
constexpr const char* repository = "openbmb/MiniCPM-o-4_5";
constexpr const char* revision = "503e754207c94da6bb26850b4469f367c9ea3582";

struct FileSpec {
    const char* path;
    std::uintmax_t size;
    const char* sha256;
};
constexpr FileSpec files[] = {
    {"assets/system_ref_audio.wav", 539032, "2c4109b2d685e1923ed66433eb08c92047a1f67510629a27edf49af4e5c606dd"},
    {"assets/token2wav/campplus.onnx", 28303423, "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73"},
    {"assets/token2wav/flow.pt", 623466603, "15ccff24256ff61537c7f8b51e025116b83405f3fb017b54b008fc97da115446"},
    {"assets/token2wav/flow.yaml", 1099, "723295d37bf11f5f1b896ca4f2f4c81ebc2fbb3e51b753c2507ef8461c751486"},
    {"assets/token2wav/hift.pt", 83390254, "3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879"},
    {"assets/token2wav/speech_tokenizer_v2_25hz.onnx", 496082973, "d43342aa12163a80bf07bffb94c9de2e120a8df2f9917cd2f642e7f4219c6f71"},
};
constexpr std::uintmax_t total_bytes = 539032ull + 28303423 + 623466603 + 1099 + 83390254 + 496082973;

std::string digest_of(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::string whole((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return mfq::cache::block_hash_hex(mfq::cache::sha256(whole));
}

std::size_t sink(char* data, std::size_t size, std::size_t count, void* user) {
    static_cast<std::ofstream*>(user)->write(data, static_cast<std::streamsize>(size * count));
    return size * count;
}

void download(const std::string& url, const fs::path& target) {
    std::string error;
    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL* handle = curl_easy_init();
    if (!handle) { curl_global_cleanup(); throw std::runtime_error("curl initialization failed"); }
    std::ofstream file(target, std::ios::binary);
    curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, sink);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, &file);
    curl_easy_setopt(handle, CURLOPT_USERAGENT, "TyloInfer/voice-component");
    curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(handle, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(handle, CURLOPT_LOW_SPEED_TIME, 120L);
    curl_easy_setopt(handle, CURLOPT_ERRORBUFFER, error.data());
    const CURLcode result = curl_easy_perform(handle);
    curl_easy_cleanup(handle);
    curl_global_cleanup();
    file.close();
    if (result != CURLE_OK) {
        std::error_code ignored;
        fs::remove(target, ignored);
        throw std::runtime_error(error.empty() ? curl_easy_strerror(result) : error);
    }
}
} // namespace

VoiceOutput::VoiceOutput(const fs::path& data_root)
    : root_(data_root / "components" / component_id) {}

json VoiceOutput::status() const {
    std::lock_guard guard(mutex_);
    bool ready = false;
    try {
        std::ifstream manifest(root_ / "manifest.json");
        if (manifest) {
            const json saved = json::parse(manifest);
            ready = saved.value("component", "") == component_id && saved.value("revision", "") == revision;
        }
    } catch (const std::exception&) { ready = false; }
    std::uintmax_t installed = 0;
    if (ready)
        for (const auto& item : files) {
            std::error_code failure;
            const fs::path path = root_ / item.path;
            if (!fs::is_regular_file(path, failure) || fs::file_size(path, failure) != item.size) ready = false;
            else installed += item.size;
        }
    if (!ready) {
        installed = 0;
        for (const auto& item : files) {
            std::error_code failure;
            const fs::path path = root_ / item.path;
            const fs::path partial = fs::path(path) += ".part";
            const auto candidate = fs::is_regular_file(path, failure) ? path : partial;
            if (fs::is_regular_file(candidate, failure))
                installed += std::min<std::uintmax_t>(fs::file_size(candidate, failure), item.size);
        }
    }
    // shortcut: activation state arrives with the realtime duplex seam; a native server reports inert.
    return json{{"id", component_id}, {"state", ready ? "ready" : "missing"}, {"ready", ready},
        {"installed_bytes", installed}, {"total_bytes", total_bytes}, {"repository", repository},
        {"revision", revision}, {"error", nullptr}, {"active", false}, {"supported_model_loaded", false}};
}

json VoiceOutput::install(const std::function<void(double, const std::string&)>& progress,
        const std::function<bool()>& cancelled) const {
    std::lock_guard guard(mutex_);
    try {
        progress(0.001, "Preparing voice output component");
        fs::create_directories(root_);
        std::uintmax_t completed = 0;
        for (const auto& item : files) {
            if (cancelled && cancelled()) throw std::runtime_error("cancelled");
            const fs::path target = root_ / item.path;
            std::error_code failure;
            bool valid = fs::is_regular_file(target, failure) && fs::file_size(target, failure) == item.size;
            if (valid) valid = digest_of(target) == item.sha256;
            if (!valid) {
                fs::remove(target, failure);
                fs::create_directories(target.parent_path(), failure);
                const fs::path partial = fs::path(target) += ".part";
                download(std::string("https://huggingface.co/") + repository + "/resolve/" + revision + "/" + item.path, partial);
                if (!fs::is_regular_file(partial, failure) || fs::file_size(partial, failure) != item.size ||
                    digest_of(partial) != item.sha256) {
                    fs::remove(partial, failure);
                    throw std::runtime_error("verification failed for " + std::string(item.path));
                }
                fs::rename(partial, target, failure);
                if (failure) throw std::runtime_error("cannot finalize " + std::string(item.path));
            }
            completed += item.size;
            progress(static_cast<double>(completed) / static_cast<double>(total_bytes),
                "Verified " + fs::path(item.path).filename().string());
        }
        const json manifest{{"component", component_id}, {"repository", repository}, {"revision", revision},
            {"total_bytes", total_bytes}, {"files", json::array()}};
        std::ofstream(root_ / "manifest.json.tmp") << manifest.dump(2) << '\n';
        std::error_code manifest_failure;
        fs::rename(root_ / "manifest.json.tmp", root_ / "manifest.json", manifest_failure);
        if (manifest_failure) throw std::runtime_error("cannot write the component manifest");
        progress(1.0, "Voice output component ready");
    } catch (const std::exception& failure) {
        throw ApiError(502, "voice_component_download_failed",
            std::string("voice output component download failed: ") + failure.what());
    }
    return status();
}
} // namespace mfq::server
