#pragma once

#include "engine/weights/encoding/tq/rows.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi { class TqTable; }

namespace tyloi::weights {

struct TqRow {
    TqRowImageLayout image;
    std::uint32_t width, group;
    bool column_scales;
};

struct TqSelection {
    std::vector<TqRow> rows;
    std::vector<std::uint32_t> inverse;
};

class TqRead final {
public:
    static constexpr std::size_t prefetch_limit = 64u << 20;
    struct Source;
    static std::shared_ptr<const Source> source(
        std::vector<std::shared_ptr<TqTable>> tables, std::int64_t rows_per_shard);
    explicit TqRead(std::shared_ptr<const Source> source);
    ~TqRead();
    bool issue(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape);
    bool matches(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) const;
    TqSelection collect(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape,
                        std::uint8_t* target, std::size_t capacity);
    std::size_t stride() const;
    void clear();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tyloi::weights
