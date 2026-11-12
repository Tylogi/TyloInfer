#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "engine/tensor.h"

namespace tyloi::weights {

class Memory {
    struct Payload { std::weak_ptr<void> owner; std::size_t bytes; };
    struct Backing { std::weak_ptr<void> owner; const void* identity; Device device; std::size_t bytes; };
    std::vector<std::weak_ptr<tyloi::TensorStorage>> weights_;
    std::vector<Backing> backings_;
    std::vector<Payload> ple_;

public:
    void observe(const Tensor& tensor);
    void observe_ple(const std::shared_ptr<void>& owner, std::size_t bytes);
    std::size_t resident_bytes(int device = -1) const;
    std::size_t resident_bytes(Device device) const;
    std::size_t ple_bytes() const;

private:
    std::size_t resident_bytes(int index, const Device* device) const;
    friend class LoadScope;
};

class LoadScope {
    Memory* previous_;
    std::shared_ptr<void> allocation_;

public:
    explicit LoadScope(Memory& ledger);
    ~LoadScope();
    LoadScope(const LoadScope&) = delete;
    LoadScope& operator=(const LoadScope&) = delete;
};

void observe_streamed_ple(const std::shared_ptr<void>& owner, std::size_t bytes);
void observe_weight(const Tensor& tensor);
}
