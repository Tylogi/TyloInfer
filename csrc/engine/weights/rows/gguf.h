#pragma once

#include "engine/io/read_span.h"
#include "components/ggml/include/ggml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tyloi::source { class Source; }

namespace tyloi::weights {

class GgufRows {
public:
    GgufRows(const source::Source& source, const std::string& name);
    ~GgufRows();

    std::int64_t rows() const noexcept { return rows_; }
    std::int64_t width() const noexcept { return width_; }
    std::uint64_t row_bytes() const noexcept { return row_bytes_; }
    ggml_type type() const noexcept { return type_; }
    std::uint64_t staged_row_bytes() const noexcept { return staged_row_bytes_; }
    ggml_type staged_type() const noexcept { return staged_type_; }
    bool parallel_reads() const noexcept { return parallel_; }

    void read_rows(const std::vector<ReadSpan>& rows) const;
    void read_staged_rows(const std::vector<ReadSpan>& rows) const;
    void decode_row(std::int64_t row, float* output) const;

private:
    struct Cache;
    std::unique_ptr<Cache> cache_;
    std::int64_t rows_ = 0;
    std::int64_t width_ = 0;
    std::uint64_t row_bytes_ = 0;
    std::uint64_t staged_row_bytes_ = 0;
    ggml_type type_ = GGML_TYPE_F32;
    ggml_type staged_type_ = GGML_TYPE_F32;
    bool parallel_ = false;
};

} // namespace tyloi::weights
