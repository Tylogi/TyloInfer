#pragma once

#include "engine/tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

class CudaSessionStateError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

using SessionBlockPayload =
    std::shared_ptr<const std::vector<uint8_t>>;

class SessionBlockWriter {
public:
    template <typename T>
    void scalar(T value) {
        using Unsigned = std::make_unsigned_t<T>;
        const auto converted = static_cast<Unsigned>(value);
        for (size_t index = 0; index < sizeof(T); ++index) {
            bytes_.push_back(static_cast<uint8_t>(
                converted >> (index * 8)));
        }
    }

    void raw(const void* data, size_t size);
    /** Append tensor metadata and host bytes, preserving a decoded tensor's original device. */
    void tensor(const ::tyloi::Tensor& source, int original_device = -1);
    void tensor(const ::tyloi::Tensor& source, ::tyloi::Device original_device);
    SessionBlockPayload finish() &&;

private:
    std::vector<uint8_t> bytes_;
};

class SessionBlockReader {
public:
    explicit SessionBlockReader(const std::vector<uint8_t>& bytes)
        : cursor_(bytes.data()), end_(bytes.data() + bytes.size()) {}

    template <typename T>
    T scalar(const char* name) {
        if (remaining() < sizeof(T)) {
            throw CudaSessionStateError(
                std::string("truncated CUDA paged cache ") + name);
        }
        using Unsigned = std::make_unsigned_t<T>;
        Unsigned value = 0;
        for (size_t index = 0; index < sizeof(T); ++index) {
            value |= static_cast<Unsigned>(cursor_[index]) << (index * 8);
        }
        cursor_ += sizeof(T);
        return static_cast<T>(value);
    }

    void raw(void* destination, size_t size, const char* name);
    /** Read a host tensor and its original accelerator index. */
    std::pair<::tyloi::Tensor, int> tensor() { return tensor(false); }
    void skip_tensor() { (void)tensor(true); }
    ::tyloi::Device tensor_device() const noexcept { return tensor_device_; }
    void expect_end() const;
    size_t remaining() const;

private:
    std::pair<::tyloi::Tensor, int> tensor(bool skip);

    const uint8_t* cursor_;
    const uint8_t* end_;
    ::tyloi::Device tensor_device_{::tyloi::DeviceType::cpu, 0};
};
