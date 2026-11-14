#pragma once
#include "mlx_tq_expert_cache.h"
#include "mlx_tq_rows.h"
#include "engine/weights/source/source.h"
#include <array>
#include <optional>
#include <string>
#include <stdexcept>
#include <vector>

namespace tyloi::metal::tq_cache_detail {
inline std::size_t aligned(std::size_t size) {
    if (size > SIZE_MAX - 255) throw std::overflow_error("TQ cache allocation is too large");
    return (size + 255) / 256 * 256;
}
struct LegacyPlan {
    struct Bank {
        std::string name;
        source::Source::TensorBatchReader reader;
        std::vector<MlxMappedTqRows::PackedRows> ranges;
        std::optional<MlxTqWeight> weight;
    };
    struct Layer { std::array<Bank, 3> banks; int experts = 0; };
    std::vector<Layer> layers;
    std::array<std::size_t, 3> largest{}, capacity{}, chunk{};
    std::size_t backing = 0;
    int io_workers = 0;
};
}
