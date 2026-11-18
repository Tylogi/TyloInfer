#pragma once

#include "engine/tensor/tensor.h"
#include "engine/io/read_span.h"
#include <functional>
#include <memory>

namespace tyloi { class NintRows; class TqTable; }
namespace tyloi::weights {
class GgufRows;
/** Own immutable selected-row readers without materializing the table. */
class Rows final : public TensorImpl {
public:
    /** Retain a streamed TQ table and its segment readers. */
    explicit Rows(std::shared_ptr<TqTable> table);
    /** Retain a streamed NINT table and its row selectors. */
    explicit Rows(std::shared_ptr<NintRows> table);
    explicit Rows(std::shared_ptr<GgufRows> table);
    using Read = std::function<void(const std::vector<ReadSpan>&)>;
    using Row = std::pair<std::int64_t, std::byte*>;
    /** Retain a dense matrix's payload reader without loading the table. */
    Rows(std::int64_t rows, std::int64_t width, ScalarType dtype, Read read);
    bool defined() const noexcept override;
    std::int64_t dim() const noexcept override { return 2; }
    std::int64_t numel() const override;
    std::size_t nbytes() const override;
    std::size_t backing_nbytes() const override;
    std::int64_t size(std::int64_t dimension) const override;
    ScalarType scalar_type() const noexcept override { return dtype_; }
    const Device& device() const noexcept override { return device_; }
    bool is_contiguous() const noexcept override { return false; }
    bool is_encoded() const noexcept override { return true; }
    std::vector<const DenseTensor*> buffers() const override { return {}; }
    void record_stream(std::uintptr_t, std::shared_ptr<void> = {}) const override {}
    /** Retain the source on CPU; reject relocating an unread table to a device. */
    std::shared_ptr<TensorImpl> relocated(Device device) const override;
    /** Borrow the TQ reader, if this table uses TQ encoding. */
    const std::shared_ptr<TqTable>& tq() const noexcept { return tq_; }
    /** Borrow the NINT reader, if this table uses NINT encoding. */
    const std::shared_ptr<NintRows>& nint() const noexcept { return nint_; }
    const std::shared_ptr<GgufRows>& gguf() const noexcept { return gguf_; }
    bool dense_source() const noexcept { return bool(read_); }
    /** Read selected rows into caller-owned storage, preserving source precision. */
    void read_rows(const std::vector<Row>& rows) const;
private:
    std::shared_ptr<TqTable> tq_;
    std::shared_ptr<NintRows> nint_;
    std::shared_ptr<GgufRows> gguf_;
    std::vector<std::int64_t> shape_;
    ScalarType dtype_ = kFloat16;
    Read read_;
    Device device_{kCPU, 0};
};
}
