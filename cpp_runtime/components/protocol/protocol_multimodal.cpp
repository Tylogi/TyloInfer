#include "protocol.h"

#include "nlohmann/json.hpp"
#include "chat.h"
#include "mtp_metrics.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <aclapi.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mfq::protocol {
static int base64_digit(unsigned char value) {
    if (value >= 'A' && value <= 'Z') return value - 'A';
    if (value >= 'a' && value <= 'z') return value - 'a' + 26;
    if (value >= '0' && value <= '9') return value - '0' + 52;
    if (value == '+') return 62;
    if (value == '/') return 63;
    return -1;
}
static std::vector<uint8_t> decode_base64(
        const std::string & encoded,
        const std::string & parameter = "audio_features") {
    std::string compact;
    compact.reserve(encoded.size());
    for (const unsigned char value : encoded) {
        if (value == ' ' || value == '\t' || value == '\r' ||
            value == '\n') {
            continue;
        }
        compact.push_back(static_cast<char>(value));
    }
    if (compact.empty() || compact.size() % 4 != 0) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " must be padded base64", parameter);
    }

    std::vector<uint8_t> output;
    output.reserve(compact.size() / 4 * 3);
    for (size_t offset = 0; offset < compact.size(); offset += 4) {
        const bool pad2 = compact[offset + 2] == '=';
        const bool pad3 = compact[offset + 3] == '=';
        if (pad2 && !pad3) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " has invalid base64 padding", parameter);
        }
        if ((pad2 || pad3) && offset + 4 != compact.size()) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " has interior base64 padding", parameter);
        }
        const int a = base64_digit(compact[offset]);
        const int b = base64_digit(compact[offset + 1]);
        const int c = pad2 ? 0 : base64_digit(compact[offset + 2]);
        const int d = pad3 ? 0 : base64_digit(compact[offset + 3]);
        if (a < 0 || b < 0 || c < 0 || d < 0) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " contains invalid base64 data", parameter);
        }
        const uint32_t merged =
            (static_cast<uint32_t>(a) << 18) |
            (static_cast<uint32_t>(b) << 12) |
            (static_cast<uint32_t>(c) << 6) |
            static_cast<uint32_t>(d);
        output.push_back(static_cast<uint8_t>(merged >> 16));
        if (!pad2) output.push_back(static_cast<uint8_t>(merged >> 8));
        if (!pad3) output.push_back(static_cast<uint8_t>(merged));
    }
    return output;
}
static size_t tensor_element_count(
        const std::vector<int64_t> & shape,
        const std::string & parameter) {
    if (shape.empty() || shape.size() > 4) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " shape must have between 1 and 4 dimensions",
            parameter);
    }
    size_t count = 1;
    for (const int64_t dimension : shape) {
        if (dimension <= 0 ||
            static_cast<uint64_t>(dimension) >
                static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            count > std::numeric_limits<size_t>::max() /
                static_cast<size_t>(dimension)) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " has an invalid tensor shape", parameter);
        }
        count *= static_cast<size_t>(dimension);
    }
    return count;
}
static uint8_t hex_digit_value(char value, const std::string & parameter) {
    if (value >= '0' && value <= '9') {
        return static_cast<uint8_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
        return static_cast<uint8_t>(value - 'a' + 10);
    }
    if (value >= 'A' && value <= 'F') {
        return static_cast<uint8_t>(value - 'A' + 10);
    }
    throw ApiError(
        400, "invalid_request_error",
        parameter + " contains an invalid hexadecimal token", parameter);
}
static std::array<uint8_t, 32> decode_file_token(
        const std::string & encoded,
        const std::string & parameter) {
    if (encoded.size() != 64) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " must contain a 32-byte token", parameter);
    }
    std::array<uint8_t, 32> result{};
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<uint8_t>(
            (hex_digit_value(encoded[2 * index], parameter) << 4) |
            hex_digit_value(encoded[2 * index + 1], parameter));
    }
    return result;
}
class TensorFileReader final {
public:
    explicit TensorFileReader(const json & tensors) {
        const std::string parameter = "mfq_multimodal.binary_file";
        if (!tensors.contains("binary_file") ||
            !tensors["binary_file"].is_object()) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " must be an object", parameter);
        }
        const auto & spec = tensors["binary_file"];
        if (!spec.contains("path") || !spec["path"].is_string() ||
            !spec.contains("token") || !spec["token"].is_string() ||
            !spec.contains("size") || !spec["size"].is_number_integer()) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " must include path, token, and size", parameter);
        }
        const auto declared_size = spec["size"].get<int64_t>();
        if (declared_size < 64 || declared_size > 1024LL * 1024LL * 1024LL) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " has an invalid size", parameter);
        }
        size_ = static_cast<size_t>(declared_size);
        // Preserve UTF-8 paths on Windows by building the path from UTF-8 bytes.
        const std::string utf8_path = spec["path"].get<std::string>();
        const auto path = std::filesystem::path(
            std::u8string(utf8_path.begin(), utf8_path.end()));
        const auto filename = path.filename().native();
        if (!path.is_absolute() ||
            filename.rfind(std::filesystem::path("mfq-multimodal-").native(), 0) != 0 ||
            path.extension() != ".bin") {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " path is not an MFQ temporary tensor file", parameter);
        }
        const auto token = decode_file_token(
            spec["token"].get<std::string>(), parameter + ".token");
#ifdef _WIN32
        descriptor_=CreateFileW(path.c_str(),GENERIC_READ|READ_CONTROL,FILE_SHARE_READ,
            nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        BY_HANDLE_FILE_INFORMATION metadata{};
        bool valid=descriptor_!=INVALID_HANDLE_VALUE && GetFileType(descriptor_)==FILE_TYPE_DISK &&
            GetFileInformationByHandle(descriptor_,&metadata) && metadata.nNumberOfLinks==1 &&
            !(metadata.dwFileAttributes&(FILE_ATTRIBUTE_DIRECTORY|FILE_ATTRIBUTE_REPARSE_POINT)) &&
            ((uint64_t(metadata.nFileSizeHigh)<<32)|metadata.nFileSizeLow)==size_;
        if(valid) {
            PSID owner=nullptr;PSECURITY_DESCRIPTOR security=nullptr;HANDLE process_token=nullptr;
            valid=GetSecurityInfo(descriptor_,SE_FILE_OBJECT,OWNER_SECURITY_INFORMATION,
                &owner,nullptr,nullptr,nullptr,&security)==ERROR_SUCCESS;
            if(valid)valid=OpenProcessToken(GetCurrentProcess(),TOKEN_QUERY,&process_token)!=0;
            if(valid) {
                DWORD bytes=0;
                GetTokenInformation(process_token,TokenOwner,nullptr,0,&bytes);
                std::vector<uint8_t> info(bytes);
                valid=bytes && GetTokenInformation(process_token,TokenOwner,info.data(),bytes,&bytes) &&
                    EqualSid(owner,reinterpret_cast<TOKEN_OWNER*>(info.data())->Owner);
            }
            if(process_token)CloseHandle(process_token);
            if(security)LocalFree(security);
        }
        if(!valid) {
            close_descriptor();
            throw ApiError(400,"invalid_request_error",
                parameter+" failed ownership, type, or size validation",parameter);
        }
#else
        int flags = O_RDONLY;
#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
        flags |= O_NOFOLLOW;
#endif
        descriptor_ = ::open(path.c_str(), flags);
        if (descriptor_ < 0) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " cannot be opened", parameter);
        }
        struct stat metadata {};
        if (::fstat(descriptor_, &metadata) != 0 ||
            !S_ISREG(metadata.st_mode) || metadata.st_nlink != 1 ||
            metadata.st_uid != ::geteuid() ||
            (metadata.st_mode & 0077) != 0 ||
            metadata.st_size < 0 ||
            static_cast<uint64_t>(metadata.st_size) != size_) {
            close_descriptor();
            throw ApiError(
                400, "invalid_request_error",
                parameter + " failed ownership, mode, or size validation", parameter);
        }
#endif
        std::array<uint8_t, 64> header{};
        try {read_exact(0, header.data(), header.size(), parameter);}
        catch(...) {close_descriptor();throw;}
        constexpr std::array<uint8_t, 8> magic{
            'M', 'F', 'Q', 'M', 'M', '0', '1', 0};
        if (!std::equal(magic.begin(), magic.end(), header.begin()) ||
            !std::equal(token.begin(), token.end(), header.begin() + 8)) {
            close_descriptor();
            throw ApiError(
                400, "invalid_request_error",
                parameter + " header or token does not match", parameter);
        }
    }

    ~TensorFileReader() {
        close_descriptor();
    }

    TensorFileReader(const TensorFileReader &) = delete;
    TensorFileReader & operator=(const TensorFileReader &) = delete;

    void read(
            size_t offset,
            void * destination,
            size_t length,
            const std::string & parameter) const {
        if (offset < 64 || offset % 64 != 0 ||
            length > size_ || offset > size_ - length) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " byte range is outside the tensor file", parameter);
        }
        read_exact(offset, destination, length, parameter);
    }

private:
    void close_descriptor() noexcept {
#ifdef _WIN32
        if(descriptor_!=INVALID_HANDLE_VALUE) {
            CloseHandle(descriptor_);descriptor_=INVALID_HANDLE_VALUE;
        }
#else
        if (descriptor_ >= 0) {
            ::close(descriptor_);
            descriptor_ = -1;
        }
#endif
    }

    void read_exact(
            size_t offset,
            void * destination,
            size_t length,
            const std::string & parameter) const {
#ifdef _WIN32
        LARGE_INTEGER position{};position.QuadPart=static_cast<LONGLONG>(offset);
        if(!SetFilePointerEx(descriptor_,position,nullptr,FILE_BEGIN))
            throw ApiError(400,"invalid_request_error",parameter+" could not be positioned",parameter);
        size_t completed=0;
        while(completed<length) {
            DWORD count=0;
            if(!ReadFile(descriptor_,static_cast<uint8_t*>(destination)+completed,
                    static_cast<DWORD>(length-completed),&count,nullptr) || !count)
                throw ApiError(400,"invalid_request_error",parameter+" could not be read completely",parameter);
            completed+=count;
        }
#else
        auto * output = static_cast<uint8_t *>(destination);
        size_t completed = 0;
        while (completed < length) {
            const ssize_t count = ::pread(
                descriptor_, output + completed, length - completed,
                static_cast<off_t>(offset + completed));
            if (count > 0) {
                completed += static_cast<size_t>(count);
                continue;
            }
            if (count < 0 && errno == EINTR) continue;
            throw ApiError(
                400, "invalid_request_error",
                parameter + " could not be read completely", parameter);
        }
#endif
    }

    size_t size_ = 0;
#ifdef _WIN32
    HANDLE descriptor_=INVALID_HANDLE_VALUE;
#else
    mutable int descriptor_ = -1;
#endif
};
template <typename Value>
static std::pair<std::vector<Value>, std::vector<int64_t>> decode_tensor(
        const json & tensors,
        const std::string & name,
        const std::string & expected_dtype,
        const TensorFileReader * file_reader) {
    const std::string parameter = "mfq_multimodal." + name;
    if (!tensors.contains(name) || !tensors[name].is_object()) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " must be a tensor object", parameter);
    }
    const auto & tensor = tensors[name];
    if (!tensor.contains("dtype") || !tensor["dtype"].is_string() ||
        tensor["dtype"].get<std::string>() != expected_dtype) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " must use dtype " + expected_dtype, parameter);
    }
    if (!tensor.contains("shape") || !tensor["shape"].is_array()) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " must include an integer shape", parameter);
    }
    std::vector<int64_t> shape;
    shape.reserve(tensor["shape"].size());
    for (const auto & dimension : tensor["shape"]) {
        if (!dimension.is_number_integer()) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " shape must contain integers", parameter);
        }
        shape.push_back(dimension.get<int64_t>());
    }
    const size_t count = tensor_element_count(shape, parameter);
    if (count > std::numeric_limits<size_t>::max() / sizeof(Value)) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " byte length does not match its shape", parameter);
    }
    const size_t expected_bytes = count * sizeof(Value);
    std::vector<Value> values(count);
    if (tensor.contains("data_base64") &&
        tensor["data_base64"].is_string()) {
        const auto bytes = decode_base64(
            tensor["data_base64"].get<std::string>(), parameter);
        if (bytes.size() != expected_bytes) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " byte length does not match its shape", parameter);
        }
        std::memcpy(values.data(), bytes.data(), bytes.size());
    } else {
        if (file_reader == nullptr ||
            !tensor.contains("data_offset") ||
            !tensor["data_offset"].is_number_integer() ||
            !tensor.contains("data_length") ||
            !tensor["data_length"].is_number_integer()) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " must include data_base64 or a binary file range",
                parameter);
        }
        const int64_t raw_offset = tensor["data_offset"].get<int64_t>();
        const int64_t raw_length = tensor["data_length"].get<int64_t>();
        if (raw_offset < 0 || raw_length < 0 ||
            static_cast<uint64_t>(raw_length) != expected_bytes) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " byte length does not match its shape", parameter);
        }
        file_reader->read(
            static_cast<size_t>(raw_offset), values.data(), expected_bytes,
            parameter);
    }
    return {std::move(values), std::move(shape)};
}
static MfqMultimodalInput parse_deepseek_multimodal(
        const json & value,
        TensorFileReader * file_reader,
        bool v41) {
    const std::string label = v41 ? "DeepSeek-V4.1" : "DeepSeek-V4";
    for (const char * name : {"pixel_values", "patch_mask", "vision_grid"}) {
        if (!value.contains(name)) {
            throw ApiError(
                400, "invalid_request_error",
                label + " multimodal payload is missing " + name,
                "mfq_multimodal");
        }
    }
    MfqMultimodalInput result;
    result.processor = v41
        ? MfqMultimodalProcessor::deepseek_v41
        : MfqMultimodalProcessor::deepseek_v4;
    result.processor_name = v41 ? "deepseek_v41" : "deepseek_v4";
    auto pixels = decode_tensor<float>(
        value, "pixel_values", "float32", file_reader);
    result.pixel_values = std::move(pixels.first);
    result.pixel_shape = std::move(pixels.second);
    auto mask = decode_tensor<uint8_t>(
        value, "patch_mask", "uint8", file_reader);
    result.patch_mask = std::move(mask.first);
    result.patch_mask_shape = std::move(mask.second);
    auto grid = decode_tensor<int32_t>(
        value, "vision_grid", "int32", file_reader);
    result.vision_grid = std::move(grid.first);
    result.vision_grid_shape = std::move(grid.second);

    if (result.pixel_shape.size() != 3 ||
        result.pixel_shape[0] <= 0 || result.pixel_shape[0] > 128 ||
        result.pixel_shape[2] != 3 * 14 * 14 ||
        result.patch_mask_shape.size() != 2 ||
        result.vision_grid_shape.size() != 2 ||
        result.vision_grid_shape[1] != 4 ||
        result.pixel_shape[0] != result.patch_mask_shape[0] ||
        result.pixel_shape[0] != result.vision_grid_shape[0] ||
        result.pixel_shape[1] != result.patch_mask_shape[1]) {
        throw ApiError(
            400, "invalid_request_error",
            label + " multimodal tensor geometry is invalid",
            "mfq_multimodal");
    }
    if (!std::all_of(
            result.pixel_values.begin(), result.pixel_values.end(),
            [](float item) { return std::isfinite(item); })) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal pixel_values contains a non-finite value",
            "mfq_multimodal.pixel_values");
    }
    const int64_t images = result.pixel_shape[0];
    const int64_t padded_patches = result.pixel_shape[1];
    for (int64_t image = 0; image < images; ++image) {
        const int64_t vit_h = result.vision_grid[4 * image];
        const int64_t vit_w = result.vision_grid[4 * image + 1];
        const int64_t llm_h = result.vision_grid[4 * image + 2];
        const int64_t llm_w = result.vision_grid[4 * image + 3];
        if (vit_h <= 0 || vit_w <= 0 ||
            vit_h > std::numeric_limits<int64_t>::max() / vit_w ||
            vit_h * vit_w > padded_patches ||
            llm_h != (vit_h + 2) / 3 ||
            llm_w != (vit_w + 2) / 3) {
            throw ApiError(
                400, "invalid_request_error",
                label + " vision grid disagrees with patch geometry",
                "mfq_multimodal.vision_grid");
        }
        const int64_t active = vit_h * vit_w;
        for (int64_t patch = 0; patch < padded_patches; ++patch) {
            const uint8_t actual = result.patch_mask[
                static_cast<size_t>(image * padded_patches + patch)];
            if (actual > 1 || static_cast<bool>(actual) != (patch < active)) {
                throw ApiError(
                    400, "invalid_request_error",
                    label + " patch mask is not a contiguous active prefix",
                    "mfq_multimodal.patch_mask");
            }
        }
    }

    return result;
}
static bool valid_processor_identity(const std::string & value) {
    if (value.empty() || value.size() > 64) return false;
    return std::all_of(
        value.begin(), value.end(), [](unsigned char item) {
            return std::isalnum(item) || item == '_' || item == '-' || item == '.';
        });
}
static MfqMultimodalInput parse_grid_vision_multimodal(
        const json & value,
        TensorFileReader * file_reader) {
    if (!value.contains("processor") || !value["processor"].is_string()) {
        throw ApiError(
            400, "invalid_request_error",
            "grid-vision payload requires a processor identity",
            "mfq_multimodal.processor");
    }
    const auto processor = value["processor"].get<std::string>();
    if (!valid_processor_identity(processor)) {
        throw ApiError(
            400, "invalid_request_error",
            "grid-vision processor identity is invalid",
            "mfq_multimodal.processor");
    }

    auto pixels = decode_tensor<float>(
        value, "pixel_values", "float32", file_reader);
    auto grid = decode_tensor<int32_t>(
        value, "vision_grid_thw", "int32", file_reader);
    auto types = decode_tensor<int32_t>(
        value, "vision_types", "int32", file_reader);
    if (pixels.second.size() != 2 || pixels.second[0] <= 0 ||
        pixels.second[1] <= 0 || grid.second.size() != 2 ||
        grid.second[0] <= 0 || grid.second[1] != 3 ||
        types.second != std::vector<int64_t>{grid.second[0]}) {
        throw ApiError(
            400, "invalid_request_error",
            "grid-vision tensor geometry is invalid",
            "mfq_multimodal");
    }
    if (!std::all_of(
            pixels.first.begin(), pixels.first.end(),
            [](float item) { return std::isfinite(item); })) {
        throw ApiError(
            400, "invalid_request_error",
            "grid-vision pixels contain a non-finite value",
            "mfq_multimodal.pixel_values");
    }

    std::uint64_t patches = 0;
    std::vector<int32_t> derived_images;
    std::vector<int32_t> derived_videos;
    for (int64_t item = 0; item < grid.second[0]; ++item) {
        const auto offset = static_cast<size_t>(3 * item);
        const int32_t temporal = grid.first[offset];
        const int32_t height = grid.first[offset + 1];
        const int32_t width = grid.first[offset + 2];
        const int32_t type = types.first[static_cast<size_t>(item)];
        if (temporal <= 0 || height <= 0 || width <= 0 ||
            (type != 1 && type != 2)) {
            throw ApiError(
                400, "invalid_request_error",
                "grid-vision dimensions or modality type are invalid",
                "mfq_multimodal.vision_grid_thw");
        }
        const auto frame = static_cast<std::uint64_t>(temporal);
        const auto rows = static_cast<std::uint64_t>(height);
        const auto columns = static_cast<std::uint64_t>(width);
        if (rows > std::numeric_limits<std::uint64_t>::max() / columns ||
            frame > std::numeric_limits<std::uint64_t>::max() / (rows * columns) ||
            patches > std::numeric_limits<std::uint64_t>::max() -
                frame * rows * columns) {
            throw ApiError(
                400, "invalid_request_error",
                "grid-vision patch count overflows",
                "mfq_multimodal.vision_grid_thw");
        }
        patches += frame * rows * columns;
        auto & destination = type == 1 ? derived_images : derived_videos;
        destination.insert(
            destination.end(),
            grid.first.begin() + static_cast<std::ptrdiff_t>(offset),
            grid.first.begin() + static_cast<std::ptrdiff_t>(offset + 3));
        if (type == 1 && temporal != 1) {
            throw ApiError(
                400, "invalid_request_error",
                "grid-vision images must have temporal size one",
                "mfq_multimodal.vision_grid_thw");
        }
    }
    if (patches != static_cast<std::uint64_t>(pixels.second[0])) {
        throw ApiError(
            400, "invalid_request_error",
            "grid-vision patch count disagrees with pixel_values",
            "mfq_multimodal.pixel_values");
    }

    const auto optional_grid = [&](const char * name) {
        if (!value.contains(name)) {
            return std::pair<std::vector<int32_t>, std::vector<int64_t>>{
                {}, {0, 3}};
        }
        auto result = decode_tensor<int32_t>(
            value, name, "int32", file_reader);
        if (result.second.size() != 2 || result.second[1] != 3) {
            throw ApiError(
                400, "invalid_request_error",
                std::string(name) + " must have [items,3] shape",
                std::string("mfq_multimodal.") + name);
        }
        return result;
    };
    auto images = optional_grid("image_grid_thw");
    auto videos = optional_grid("video_grid_thw");
    if (images.first != derived_images || videos.first != derived_videos) {
        throw ApiError(
            400, "invalid_request_error",
            "grid-vision split grids disagree with combined media order",
            "mfq_multimodal");
    }

    MfqMultimodalInput result;
    result.processor = MfqMultimodalProcessor::grid_vision;
    result.processor_name = processor;
    result.pixel_values = std::move(pixels.first);
    result.pixel_shape = std::move(pixels.second);
    result.vision_grid = std::move(grid.first);
    result.vision_grid_shape = std::move(grid.second);
    result.vision_types = std::move(types.first);
    result.image_grid = std::move(images.first);
    result.image_grid_shape = std::move(images.second);
    result.video_grid = std::move(videos.first);
    result.video_grid_shape = std::move(videos.second);
    return result;
}
MfqMultimodalInput parse_mfq_vision(const json & value) {
    if (!value.is_object()) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal must be an object", "mfq_multimodal");
    }
    if (!value.contains("version") || !value["version"].is_number_integer()) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal.version must be an integer", "mfq_multimodal.version");
    }

    std::unique_ptr<TensorFileReader> file_reader;
    if (value.contains("binary_file")) {
        file_reader = std::make_unique<TensorFileReader>(value);
    }
    const int version = value["version"].get<int>();
    if (version == 3) {
        return parse_grid_vision_multimodal(value, file_reader.get());
    }
    if (version == 2) {
        if (!value.contains("processor") ||
            !value["processor"].is_string()) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_multimodal version 2 requires a DeepSeek processor",
                "mfq_multimodal.processor");
        }
        const auto processor = value["processor"].get<std::string>();
        if (processor != "deepseek_v4" && processor != "deepseek_v41") {
            throw ApiError(
                400, "invalid_request_error",
                "unsupported DeepSeek multimodal processor",
                "mfq_multimodal.processor");
        }
        return parse_deepseek_multimodal(
            value, file_reader.get(), processor == "deepseek_v41");
    }
    if (version != 1) {
        throw ApiError(
            400, "invalid_request_error",
            "unsupported mfq_multimodal.version", "mfq_multimodal.version");
    }

    MfqMultimodalInput result;
    result.processor = MfqMultimodalProcessor::minicpmo;
    const bool has_any_image_tensor =
        value.contains("pixel_values") || value.contains("patch_mask") ||
        value.contains("target_sizes");
    const bool has_all_image_tensors =
        value.contains("pixel_values") && value.contains("patch_mask") &&
        value.contains("target_sizes");
    if (has_any_image_tensor != has_all_image_tensors) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal image tensors must be provided together",
            "mfq_multimodal");
    }
    const bool has_any_audio_tensor =
        value.contains("audio_features") || value.contains("audio_lengths");
    const bool has_all_audio_tensors =
        value.contains("audio_features") && value.contains("audio_lengths");
    if (has_any_audio_tensor != has_all_audio_tensors) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal audio tensors must be provided together",
            "mfq_multimodal");
    }
    if (!has_all_image_tensors && !has_all_audio_tensors) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal contains no media tensors", "mfq_multimodal");
    }

    if (has_all_image_tensors) {
        auto pixels = decode_tensor<float>(
            value, "pixel_values", "float32", file_reader.get());
        result.pixel_values = std::move(pixels.first);
        result.pixel_shape = std::move(pixels.second);
        auto mask = decode_tensor<uint8_t>(
            value, "patch_mask", "uint8", file_reader.get());
        result.patch_mask = std::move(mask.first);
        result.patch_mask_shape = std::move(mask.second);
        auto sizes = decode_tensor<int32_t>(
            value, "target_sizes", "int32", file_reader.get());
        result.target_sizes = std::move(sizes.first);
        result.target_sizes_shape = std::move(sizes.second);

        if (result.pixel_shape.size() != 4 || result.pixel_shape[1] != 3 ||
            result.pixel_shape[2] != 14 ||
            result.patch_mask_shape.size() != 2 ||
            result.target_sizes_shape.size() != 2 ||
            result.target_sizes_shape[1] != 2 ||
            result.pixel_shape[0] != result.patch_mask_shape[0] ||
            result.pixel_shape[0] != result.target_sizes_shape[0] ||
            result.pixel_shape[3] % 14 != 0 ||
            result.patch_mask_shape[1] != result.pixel_shape[3] / 14) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_multimodal image tensor geometry is invalid",
                "mfq_multimodal");
        }
        const int64_t source_count = result.pixel_shape[0];
        if (source_count <= 0 || source_count > 576) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_multimodal contains an invalid number of image slices",
                "mfq_multimodal.pixel_values");
        }
        for (int64_t source = 0; source < source_count; ++source) {
            const int64_t rows = result.target_sizes[2 * source];
            const int64_t columns = result.target_sizes[2 * source + 1];
            if (rows <= 0 || columns <= 0 ||
                rows * columns > result.patch_mask_shape[1]) {
                throw ApiError(
                    400, "invalid_request_error",
                    "mfq_multimodal target size disagrees with pixel geometry",
                    "mfq_multimodal.target_sizes");
            }
            for (int64_t patch = 0; patch < result.patch_mask_shape[1]; ++patch) {
                const bool expected = patch < rows * columns;
                const uint8_t actual = result.patch_mask[
                    static_cast<size_t>(source * result.patch_mask_shape[1] + patch)];
                if (actual > 1 || static_cast<bool>(actual) != expected) {
                    throw ApiError(
                        400, "invalid_request_error",
                        "mfq_multimodal patch mask is not a contiguous active prefix",
                        "mfq_multimodal.patch_mask");
                }
            }
        }
        if (!std::all_of(
                result.pixel_values.begin(), result.pixel_values.end(),
                [](float item) { return std::isfinite(item); })) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_multimodal pixel_values contains a non-finite value",
                "mfq_multimodal.pixel_values");
        }
    }

    if (has_all_audio_tensors) {
        auto features = decode_tensor<float>(
            value, "audio_features", "float32", file_reader.get());
        result.audio_features = std::move(features.first);
        result.audio_features_shape = std::move(features.second);
        auto lengths = decode_tensor<int64_t>(
            value, "audio_lengths", "int64", file_reader.get());
        result.audio_lengths = std::move(lengths.first);
        const auto & length_shape = lengths.second;
        if (result.audio_features_shape.size() != 3 ||
            result.audio_features_shape[0] <= 0 ||
            result.audio_features_shape[0] > 128 ||
            result.audio_features_shape[1] != 80 ||
            result.audio_features_shape[2] < 9 ||
            result.audio_features_shape[2] > 3000 ||
            length_shape.size() != 1 ||
            length_shape[0] != result.audio_features_shape[0]) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_multimodal audio tensor geometry is invalid",
                "mfq_multimodal");
        }
        if (!std::all_of(
                result.audio_features.begin(), result.audio_features.end(),
                [](float item) { return std::isfinite(item); })) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_multimodal audio_features contains a non-finite value",
                "mfq_multimodal.audio_features");
        }
        for (const int64_t length : result.audio_lengths) {
            if (length < 9 || length > result.audio_features_shape[2]) {
                throw ApiError(
                    400, "invalid_request_error",
                    "mfq_multimodal audio length is out of range",
                    "mfq_multimodal.audio_lengths");
            }
        }
    }
    return result;
}
std::vector<float> decode_audio_features(
        const std::string & encoded,
        int32_t frames) {
    if (frames < 3 || frames > 4096) {
        throw ApiError(
            400, "invalid_request_error",
            "audio_frames must be in [3, 4096]", "audio_frames");
    }
    const auto bytes = decode_base64(encoded);
    const size_t expected = static_cast<size_t>(frames) * 80 * sizeof(float);
    if (bytes.size() != expected) {
        throw ApiError(
            400, "invalid_request_error",
            "audio_features byte length does not match audio_frames",
            "audio_features");
    }
    std::vector<float> features(static_cast<size_t>(frames) * 80);
    std::memcpy(features.data(), bytes.data(), bytes.size());
    if (!std::all_of(features.begin(), features.end(), [](float value) {
            return std::isfinite(value);
        })) {
        throw ApiError(
            400, "invalid_request_error",
            "audio_features contains a non-finite value",
            "audio_features");
    }
    return features;
}

} // namespace mfq::protocol
