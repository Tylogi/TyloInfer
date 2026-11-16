#pragma once

#include "engine/tensor/tensor.h"

#include <memory>
#include <string>
#include <vector>

namespace tyloi::metal { class MlxLinear; class MlxEmbedding; }
namespace tyloi::source { class Source; }

namespace tyloi::encoding {

// A packed Metal weight. Both operations share the underlying format buffers.
class MetalTensor final : public TensorImpl {
public:
    MetalTensor(std::shared_ptr<const metal::MlxLinear> linear,
        std::shared_ptr<const metal::MlxEmbedding> embedding,
        std::vector<std::int64_t> shape);

    const metal::MlxLinear& linear() const noexcept { return *linear_; }
    const metal::MlxEmbedding& embedding() const noexcept { return *embedding_; }
    bool defined() const noexcept override { return bool(linear_); }
    std::int64_t dim() const noexcept override { return shape_.size(); }
    std::int64_t numel() const override;
    std::size_t nbytes() const override { return std::size_t(numel()) * scalar_size(scalar_type()); }
    std::size_t backing_nbytes() const override;
    std::int64_t size(std::int64_t dimension) const override;
    ScalarType scalar_type() const noexcept override { return kFloat16; }
    const Device& device() const noexcept override { return device_; }
    bool is_contiguous() const noexcept override { return false; }
    bool is_encoded() const noexcept override { return true; }
    std::vector<const DenseTensor*> buffers() const override { return {}; }
    std::vector<std::shared_ptr<TensorStorage>> backings() const override;
    void record_stream(std::uintptr_t, std::shared_ptr<void> = {}) const override {
        throw std::runtime_error("Metal tensor stream recording is not supported");
    }

private:
    std::shared_ptr<const metal::MlxLinear> linear_;
    std::shared_ptr<const metal::MlxEmbedding> embedding_;
    std::vector<std::int64_t> shape_;
    Device device_{DeviceType::metal};
};

Tensor load_metal_encoded(const source::Source& source, const std::string& name);

} // namespace tyloi::encoding
