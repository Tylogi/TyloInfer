#include "engine/weights/rows/stage.h"
#include "engine/backends/cuda/ops/row_stage.h"
#include "engine/backends/cuda/ops/gguf/rows.h"
#include "engine/weights/encoding/rows.h"
#include "engine/weights/rows/gguf.h"
#include "engine/weights/rows/gguf_read.h"
#include "engine/backends/cuda/ops/projection/tq/rows.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/rows/nint_pipeline.h"
#endif
#include <limits>
#include <stdexcept>

namespace tyloi::weights {
#if TYLOINFER_ENABLE_MFQ_COMPAT
namespace {
struct NintStage final : RowStage {
    std::shared_ptr<NintRowPipeline> pipeline;
    std::shared_ptr<NintRowStage> stage;
    NintStage(std::shared_ptr<NintRowPipeline> value, const std::vector<std::int64_t>& shape, const Device& device)
        : pipeline(std::move(value)), stage(pipeline->make_stage(shape, device)) {}
    const Tensor& output() const override { return stage->output(); }
    bool mapped_rows() const override { return stage->mapped_rows(); }
    void decode() override { stage->decode(); }
    void upload(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) override {
        pipeline->upload(*stage, ids, shape);
    }
};
}
#endif
struct RowReader::Source {
    std::vector<std::shared_ptr<TqTable>> tq;
    std::vector<std::shared_ptr<GgufRows>> gguf;
    std::vector<Tensor> dense;
    std::int64_t rows;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::shared_ptr<const NintRowSource> nint;
#endif
};
struct RowReader::Impl {
    std::shared_ptr<const Source> source;
    std::shared_ptr<GgufRead> gguf;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::shared_ptr<NintRowPipeline> nint;
#endif
};
std::shared_ptr<const RowReader::Source> RowReader::source(std::vector<Tensor> tables, std::int64_t rows_per_shard) {
    if (tables.empty()) return {};
    bool tq = true, dense = true, gguf = true;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    bool nint_supported = true;
#endif
    for (const auto& table : tables) {
        const auto* rows = table.representation<Rows>();
        tq = tq && rows && bool(rows->tq());
        dense = dense && rows && rows->dense_source();
        gguf = gguf && rows && bool(rows->gguf());
#if TYLOINFER_ENABLE_MFQ_COMPAT
        nint_supported = nint_supported && rows && bool(rows->nint());
#endif
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!tq && !nint_supported && !dense && !gguf) return {};
#else
    if (!tq && !dense && !gguf) return {};
#endif
    if (rows_per_shard <= 0) throw std::invalid_argument("invalid row reader geometry");
    auto result = std::make_shared<Source>();
    result->rows = rows_per_shard;
    if (gguf) {
        for (std::size_t i = 0; i < tables.size(); ++i) {
            const auto& table = tables[i].representation<Rows>()->gguf();
            if (table->width() != tables.front().representation<Rows>()->gguf()->width() ||
                table->rows() <= 0 || table->rows() > rows_per_shard ||
                (i + 1 < tables.size() && table->rows() != rows_per_shard))
                throw std::invalid_argument("GGUF row shard geometry differs");
            result->gguf.push_back(table);
        }
        return result;
    }
    if (dense) {
        for (std::size_t i = 0; i < tables.size(); ++i) {
            if (tables[i].size(1) != tables.front().size(1) ||
                tables[i].scalar_type() != tables.front().scalar_type() ||
                tables[i].size(0) > rows_per_shard ||
                (i + 1 < tables.size() && tables[i].size(0) != rows_per_shard))
                throw std::invalid_argument("dense row shard geometry differs");
        }
        result->dense = std::move(tables);
        return result;
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::vector<std::shared_ptr<NintRows>> nint;
#endif
    for (const auto& table : tables) {
        const auto& rows = *table.representation<Rows>();
        if (rows.tq()) result->tq.push_back(rows.tq());
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if (rows.nint()) nint.push_back(rows.nint());
#endif
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!nint.empty()) result->nint = std::make_shared<NintRowSource>(std::move(nint));
#endif
    return result;
}
std::vector<std::pair<Device, std::size_t>> RowReader::stage_growth(
        const std::shared_ptr<const Source>& source, const std::vector<std::int64_t>& shape, const Device& device) {
    if (!source) throw std::invalid_argument("row reader requires a source");
    std::pair<std::size_t, std::size_t> bytes;
    if (!source->gguf.empty()) {
        bytes = cuda::gguf::row_stage_growth(source->gguf, source->rows, shape, device);
        return {{Device(kCPU), bytes.first}, {device, bytes.second}};
    }
    if (!source->dense.empty()) {
        auto output_shape = shape;
        output_shape.push_back(source->dense.front().size(1));
        const auto size = make_contiguous_view(nullptr, output_shape,
            source->dense.front().scalar_type(), device).nbytes();
        return {{Device(kCPU), size}, {device, size}};
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (source->nint) bytes = NintRowPipeline::stage_growth(*source->nint, shape, device);
    else
#endif
    bytes = cuda::tq::row_stage_growth(source->tq, source->rows, shape, device);
    return {{Device(kCPU), bytes.first}, {device, bytes.second}};
}
RowReader::RowReader(std::shared_ptr<const Source> source) : impl_(std::make_unique<Impl>()) {
    if (!source) throw std::invalid_argument("row reader requires a source");
    impl_->source = std::move(source);
    if (!impl_->source->gguf.empty())
        impl_->gguf = std::make_shared<GgufRead>(impl_->source->gguf, impl_->source->rows);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (impl_->source->nint) impl_->nint = std::make_shared<NintRowPipeline>(impl_->source->nint);
#endif
}
RowReader::~RowReader() = default;
bool RowReader::prefetch(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape, bool early_prefix) {
    if (impl_->gguf) return impl_->gguf->issue(ids, shape);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (impl_->nint) { impl_->nint->issue(ids, shape, early_prefix); return true; }
#endif
    return false;
}
Tensor RowReader::collect(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape, const Device& device) {
    if (impl_->gguf)
        return cuda::gguf::lookup_rows(impl_->source->gguf, impl_->source->rows, ids, shape, device, impl_->gguf);
    if (!impl_->source->dense.empty()) {
        auto transfer = stage(shape, device);
        transfer->upload(ids, shape);
        transfer->decode();
        return transfer->output();
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (impl_->nint) return impl_->nint->collect(ids, shape, device);
#endif
    std::size_t count = 1;
    for (auto extent : shape) {
        if (extent < 0 || (extent && count > std::numeric_limits<std::size_t>::max() / std::size_t(extent)))
            throw std::invalid_argument("invalid row reader shape");
        count *= std::size_t(extent);
    }
    if (count != ids.size()) throw std::invalid_argument("row reader shape mismatch");
    auto output = cuda::tq::lookup_rows(impl_->source->tq, impl_->source->rows, ids, device);
    auto result_shape = shape;
    result_shape.push_back(impl_->source->tq.front()->width());
    return output.dense().reshape(result_shape);
}
std::shared_ptr<RowStage> RowReader::stage(const std::vector<std::int64_t>& shape, const Device& device) {
    if (impl_->gguf)
        return cuda::gguf::make_row_stage(impl_->source->gguf, impl_->source->rows, shape, device, impl_->gguf);
    if (!impl_->source->dense.empty())
        return cuda::make_dense_row_stage(impl_->source->dense, impl_->source->rows, shape, device);
    std::shared_ptr<RowStage> result;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (impl_->nint) result = std::make_shared<NintStage>(impl_->nint, shape, device);
    else
#endif
    result = cuda::tq::make_row_stage(impl_->source->tq, impl_->source->rows, shape, device);
    return result;
}
}
