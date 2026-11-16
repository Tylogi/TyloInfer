#pragma once

#include "engine/tensor/tensor.h"

#include <array>
#include <memory>

namespace tyloi::metal { class MlxMfeWeight; }

namespace tyloi::encoding {

class MetalExperts final : public TensorImpl {
public:
    explicit MetalExperts(metal::MlxMfeWeight weight);

    const metal::MlxMfeWeight& weight() const noexcept { return *weight_; }
    bool defined() const noexcept override { return bool(weight_); }
    std::int64_t dim() const noexcept override { return 3; }
    std::int64_t numel() const override;
    std::size_t nbytes() const override;
    std::size_t backing_nbytes() const override;
    std::int64_t size(std::int64_t dimension) const override;
    ScalarType scalar_type() const noexcept override { return kFloat16; }
    const Device& device() const noexcept override { return device_; }
    bool is_contiguous() const noexcept override { return false; }
    bool is_encoded() const noexcept override { return true; }
    std::vector<const DenseTensor*> buffers() const override { return {}; }
    std::vector<std::shared_ptr<TensorStorage>> backings() const override;
    void record_stream(std::uintptr_t, std::shared_ptr<void> = {}) const override;

private:
    std::shared_ptr<const metal::MlxMfeWeight> weight_;
    std::array<std::int64_t, 3> shape_{};
    Device device_{DeviceType::metal, 0};
};

} // namespace tyloi::encoding
