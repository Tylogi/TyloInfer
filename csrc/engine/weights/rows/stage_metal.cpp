#include "engine/weights/rows/stage.h"
#include "engine/weights/encoding/rows.h"
#include "engine/weights/encoding/tq/table.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "backends/metal/storage/mlx_tq_rows.h"
#include "engine/ops/ops.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/encoding/quant/nint_rows.h"
#include "engine/backends/metal/ops/mfq/mlx_nint_rows.h"
#endif

#include <mlx/mlx.h>
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::weights {
namespace {
std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("Metal row stage size overflow");
    return a * b;
}
std::size_t add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error("Metal row stage size overflow");
    return a + b;
}
std::size_t count_rows(const std::vector<std::int64_t>& shape) {
    std::size_t count = 1;
    for (auto size : shape) {
        if (size <= 0) throw std::invalid_argument("invalid Metal row stage shape");
        count = multiply(count, std::size_t(size));
    }
    if (count > INT_MAX) throw std::overflow_error("Metal row stage count exceeds int32");
    return count;
}
}

struct RowReader::Source {
    enum class Kind { dense, tq, nint } kind;
    std::vector<Tensor> tables;
    std::int64_t rows;
};
struct RowReader::Impl { std::shared_ptr<const Source> source; };

std::shared_ptr<const RowReader::Source> RowReader::source(std::vector<Tensor> tables, std::int64_t rows) {
    if (tables.empty()) return {};
    const auto* first = tables.front().representation<Rows>();
    if (!first) return {};
    Source::Kind kind;
    if (first->dense_source()) kind = Source::Kind::dense;
    else if (first->tq()) kind = Source::Kind::tq;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    else if (first->nint()) kind = Source::Kind::nint;
#endif
    else return {};
    if (rows <= 0) throw std::invalid_argument("invalid row reader geometry");
    for (std::size_t i = 0; i < tables.size(); ++i) {
        const auto* table = tables[i].representation<Rows>();
        if (!table || (kind == Source::Kind::dense && !table->dense_source()) ||
            (kind == Source::Kind::tq && !table->tq()) ||
            (kind == Source::Kind::nint && !table->nint()) ||
            tables[i].dim() != 2 || tables[i].size(1) != tables.front().size(1) ||
            tables[i].scalar_type() != tables.front().scalar_type() || tables[i].size(0) > rows ||
            (i + 1 < tables.size() && tables[i].size(0) != rows))
            throw std::invalid_argument("Metal row shard geometry differs");
    }
    return std::make_shared<Source>(Source{kind, std::move(tables), rows});
}

std::vector<std::pair<Device, std::size_t>> RowReader::stage_growth(
        const std::shared_ptr<const Source>& source, const std::vector<std::int64_t>& shape, const Device& device) {
    if (!source || device != Device(DeviceType::metal)) throw std::invalid_argument("invalid Metal row stage source");
    const auto count = count_rows(shape);
    const auto& first = source->tables.front();
    const auto output = multiply(multiply(count, std::size_t(first.size(1))), scalar_size(first.scalar_type()));
    std::size_t host = output, packed = 0;
    if (source->kind != Source::Kind::dense) {
        std::size_t bound = 0;
        for (const auto& table : source->tables) {
            const auto* rows = table.representation<Rows>();
            if (rows->tq()) bound = std::max(bound, multiply(rows->tq()->row_image_bytes_bound(), 6));
#if TYLOINFER_ENABLE_MFQ_COMPAT
            if (rows->nint()) bound = std::max(bound, add(rows->nint()->selected_row_nbytes_bound(), 24));
#endif
        }
        packed = multiply(count, bound);
        host = add(packed, bound);
    }
    // Stable destination, decoded temporary and selected packed inputs coexist.
    return {{Device(kCPU), host}, {device, add(multiply(output, 2), packed)}};
}

namespace {
class MetalRowStage final : public RowStage {
public:
    MetalRowStage(std::shared_ptr<const RowReader::Source> source,
                  std::vector<std::int64_t> shape, Device device)
        : source_(std::move(source)), shape_(std::move(shape)), count_(count_rows(shape_)) {
        if (!source_ || device != Device(DeviceType::metal))
            throw std::invalid_argument("invalid Metal row stage");
        auto dimensions = shape_;
        dimensions.push_back(source_->tables.front().size(1));
        output_ = ops::zeros(dimensions, TensorOptions().device(device).dtype(source_->tables.front().scalar_type()));
    }
    const Tensor& output() const override { return output_; }
    bool mapped_rows() const override { return false; }
    void upload(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) override {
        if (shape != shape_ || ids.size() != count_) throw std::invalid_argument("Metal row upload shape differs");
        const auto width = source_->tables.front().size(1);
        metal::MlxTqRowBatch tq;
#if TYLOINFER_ENABLE_MFQ_COMPAT
        metal::MlxNintRowBatch nint;
#endif
        std::vector<std::byte> dense;
        if (source_->kind == RowReader::Source::Kind::dense)
            dense.resize(multiply(multiply(count_, std::size_t(width)), output_.element_size()));
        std::vector<std::vector<Rows::Row>> reads(source_->tables.size());
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const auto id = ids[i];
            if (id < 0 || std::uint64_t(id / source_->rows) >= source_->tables.size())
                throw std::out_of_range("Metal row shard id is out of bounds");
            const auto shard = std::size_t(id / source_->rows);
            const auto row = id % source_->rows;
            const auto& table = source_->tables[shard];
            if (row >= table.size(0)) throw std::out_of_range("Metal row id is out of bounds");
            const auto* rows = table.representation<Rows>();
            if (source_->kind == RowReader::Source::Kind::dense) {
                reads[shard].emplace_back(row, dense.data() + i * std::size_t(width) * output_.element_size());
            } else if (source_->kind == RowReader::Source::Kind::tq) {
                auto image = rows->tq()->slice_rows(row, row + 1);
                metal::MlxMappedTqRows(image, int(width)).append_row(0, tq);
#if TYLOINFER_ENABLE_MFQ_COMPAT
            } else {
                rows->nint()->append_row(row, nint);
#endif
            }
        }
        for (std::size_t i = 0; i < reads.size(); ++i)
            if (!reads[i].empty()) source_->tables[i].representation<Rows>()->read_rows(reads[i]);
        dense_ = std::move(dense);
        tq_ = std::move(tq);
#if TYLOINFER_ENABLE_MFQ_COMPAT
        nint_ = std::move(nint);
#endif
        ready_ = true;
    }
    void decode() override {
        if (!ready_) throw std::logic_error("Metal row stage has no prepared rows");
        auto dimensions = shape_;
        dimensions.push_back(source_->tables.front().size(1));
        Tensor values;
        if (source_->kind == RowReader::Source::Kind::dense) {
            values = metal::copy_from_host(dense_.data(), dimensions, output_.scalar_type());
        } else {
            mlx::core::array decoded = [&] {
                if (source_->kind == RowReader::Source::Kind::tq) return tq_.decode();
#if TYLOINFER_ENABLE_MFQ_COMPAT
                return nint_.decode();
#else
                throw std::logic_error("NINT rows require compatibility support");
#endif
            }();
            mlx::core::Shape shape;
            for (auto extent : dimensions) {
                if (extent > INT_MAX) throw std::overflow_error("Metal row stage dimension exceeds int32");
                shape.push_back(int(extent));
            }
            values = metal::tensor(mlx::core::contiguous(mlx::core::reshape(decoded, shape)));
        }
        // Write through the original backing so every stage-output alias remains valid.
        ops::copy_into(output_, values);
        std::vector<std::byte>().swap(dense_);
        tq_ = {};
#if TYLOINFER_ENABLE_MFQ_COMPAT
        nint_ = {};
#endif
        ready_ = false;
    }
private:
    std::shared_ptr<const RowReader::Source> source_;
    std::vector<std::int64_t> shape_;
    std::size_t count_;
    Tensor output_;
    std::vector<std::byte> dense_;
    metal::MlxTqRowBatch tq_;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    metal::MlxNintRowBatch nint_;
#endif
    bool ready_ = false;
};
}

RowReader::RowReader(std::shared_ptr<const Source> source) : impl_(std::make_unique<Impl>()) {
    if (!source) throw std::invalid_argument("row reader requires a source");
    impl_->source = std::move(source);
}
RowReader::~RowReader() = default;
bool RowReader::prefetch(const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, bool) { return false; }
Tensor RowReader::collect(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape, const Device& device) {
    auto selected = stage(shape, device);
    selected->upload(ids, shape);
    selected->decode();
    return selected->output();
}
std::shared_ptr<RowStage> RowReader::stage(const std::vector<std::int64_t>& shape, const Device& device) {
    return std::make_shared<MetalRowStage>(impl_->source, shape, device);
}

} // namespace tyloi::weights
