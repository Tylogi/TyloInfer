#include "engine/weights/encoding/rows.h"
#include "engine/weights/encoding/tq_table.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/encoding/quant/nint_rows.h"
#endif
#include <limits>
#include <stdexcept>

namespace tyloi::weights {
Rows::Rows(std::shared_ptr<TqTable> table) : tq_(std::move(table)) {
    if (tq_ && tq_->rows() > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("row table size overflow");
}
Rows::Rows(std::shared_ptr<NintRows> table) : nint_(std::move(table)) {}
Rows::Rows(std::int64_t rows, std::int64_t width, ScalarType dtype, Read read)
    : shape_{rows, width}, dtype_(dtype), read_(std::move(read)) {
    if (rows <= 0 || width <= 0 || !read_ ||
        (dtype != kFloat16 && dtype != kBFloat16 && dtype != kFloat32))
        throw std::invalid_argument("invalid dense row table");
    (void)nbytes();
}
void Rows::read_rows(const std::vector<Row>& rows) const {
    if (!read_) throw std::invalid_argument("row table has no dense reader");
    const auto bytes = static_cast<std::size_t>(shape_[1]) * scalar_size(dtype_);
    std::vector<ReadSpan> ranges;
    ranges.reserve(rows.size());
    for (const auto& [row, output] : rows) {
        if (row < 0 || row >= shape_[0] || !output)
            throw std::out_of_range("invalid dense table row read");
        ranges.push_back({static_cast<std::uint64_t>(row) * bytes, output, bytes});
    }
    if (!ranges.empty()) read_(ranges);
}
bool Rows::defined() const noexcept { return bool(tq_) || bool(nint_) || bool(read_); }
std::int64_t Rows::size(std::int64_t dimension) const {
    if (dimension < 0) dimension += 2;
    if (dimension < 0 || dimension >= 2 || !defined()) throw std::out_of_range("invalid row table dimension");
    if (tq_) return dimension == 0 ? static_cast<std::int64_t>(tq_->rows()) : tq_->width();
    if (read_) return shape_[dimension];
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return dimension == 0 ? nint_->rows() : nint_->width();
#else
    throw std::logic_error("NINT rows require compatibility support");
#endif
}
std::int64_t Rows::numel() const {
    const auto rows = size(0), width = size(1);
    if (width && rows > std::numeric_limits<std::int64_t>::max() / width)
        throw std::overflow_error("row table size overflow");
    return rows * width;
}
std::size_t Rows::nbytes() const {
    const auto count = static_cast<std::size_t>(numel());
    const auto bytes = scalar_size(dtype_);
    if (count > std::numeric_limits<std::size_t>::max() / bytes) throw std::overflow_error("row table size overflow");
    return count * bytes;
}
std::size_t Rows::backing_nbytes() const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return nint_ ? nint_->index_nbytes() : 0;
#else
    return 0;
#endif
}
std::shared_ptr<TensorImpl> Rows::relocated(Device device) const {
    if (!device.is_cpu()) throw std::invalid_argument("streamed rows cannot be relocated");
    return std::make_shared<Rows>(*this);
}
}
