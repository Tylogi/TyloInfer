#pragma once

#include "engine/tensor/tensor.h"

#include <array>
#include <memory>
#include <variant>

namespace tyloi::metal {

class MlxPackedExpertCache;
class MlxTqExpertCache;

enum class ExpertRole { gate, up, down };
using ExpertCache = std::variant<std::shared_ptr<MlxPackedExpertCache>,
                                 std::shared_ptr<MlxTqExpertCache>>;

// A logical expert bank backed by the shared, budgeted cache for its layer.
class ExpertBank final : public TensorImpl {
public:
    ExpertBank(ExpertCache cache, std::size_t layer, ExpertRole role,
               std::array<std::int64_t, 3> shape);

    const ExpertCache& cache() const noexcept { return cache_; }
    std::size_t layer() const noexcept { return layer_; }
    ExpertRole role() const noexcept { return role_; }
    bool defined() const noexcept override { return true; }
    std::int64_t dim() const noexcept override { return 3; }
    std::int64_t numel() const override;
    std::size_t nbytes() const override { return std::size_t(numel()) * scalar_size(scalar_type()); }
    // Cache memory is accounted for by its owner, not by each logical bank.
    std::size_t backing_nbytes() const override { return 0; }
    std::int64_t size(std::int64_t dimension) const override;
    ScalarType scalar_type() const noexcept override { return kFloat16; }
    const Device& device() const noexcept override { return device_; }
    bool is_contiguous() const noexcept override { return false; }
    bool is_encoded() const noexcept override { return true; }
    std::vector<const DenseTensor*> buffers() const override { return {}; }
    void record_stream(std::uintptr_t, std::shared_ptr<void> = {}) const override;

private:
    ExpertCache cache_;
    std::size_t layer_;
    ExpertRole role_;
    std::array<std::int64_t, 3> shape_;
    Device device_{DeviceType::metal};
};

} // namespace tyloi::metal
