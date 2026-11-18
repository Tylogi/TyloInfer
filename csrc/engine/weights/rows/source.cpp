#include "engine/weights/rows/source.h"
#include "engine/weights/source/read.h"
#include "engine/weights/memory.h"
#include "engine/weights/encoding/rows.h"
#include "engine/weights/encoding/dense/blob.h"
#if TYLOINFER_HAS_CUDA
#include "engine/weights/rows/gguf.h"
#include "engine/backends/cuda/native/env.h"
#else
#include <cstdlib>
#endif
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/encoding/quant/nint_rows.h"
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

#if TYLOINFER_HAS_CUDA
std::shared_ptr<tyloi::weights::Rows> tyloi::weights::load_gguf_rows(
        const source::Source& source, const std::string& name) {
    auto table = std::make_shared<GgufRows>(source, name);
    observe_streamed_ple(table, static_cast<std::size_t>(require_tensor(source, name).nbytes));
    return std::make_shared<Rows>(std::move(table));
}
#endif

std::shared_ptr<tyloi::weights::Rows> tyloi::weights::load_dense_rows(
        const source::Source& source, const std::string& name) {
    const auto& record = tyloi::require_tensor(source, name);
    ScalarType dtype;
    if (record.dtype == "BF16") dtype = kBFloat16;
    else if (record.dtype == "F16") dtype = kFloat16;
    else if (record.dtype == "F32") dtype = kFloat32;
    else throw std::invalid_argument("dense row table requires floating storage: " + name);
    if (record.nbytes > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("dense row table byte count overflows");
    const auto header = source.read(name, 0, sizeof(std::uint32_t) + 2 * sizeof(std::int64_t));
    std::uint32_t rank;
    std::memcpy(&rank, header.data(), sizeof(rank));
    if (rank != 2) throw std::invalid_argument("dense row table must be a matrix: " + name);
    const auto layout = dense::parse(
        std::vector<std::uint8_t>(reinterpret_cast<const std::uint8_t*>(header.data()),
            reinterpret_cast<const std::uint8_t*>(header.data()) + header.size()),
        scalar_size(dtype), record.shape, static_cast<std::size_t>(record.nbytes));
    auto batch = source.tensor_batch_reader(name);
    if (!batch) {
        auto read = source.tensor_reader(name);
        if (!read) throw std::invalid_argument("dense row table has no retained reader: " + name);
        batch = [read = std::move(read)](const std::vector<ReadSpan>& ranges) {
            for (const auto& range : ranges) read(range.offset, range.destination, range.size);
        };
    }
    auto table = std::make_shared<Rows>(layout.shape[0], layout.shape[1], dtype,
        [batch = std::move(batch), offset = layout.offset](const std::vector<ReadSpan>& ranges) {
            auto source_ranges = ranges;
            for (auto& range : source_ranges) range.offset += offset;
            batch(source_ranges);
        });
    observe_streamed_ple(table, table->nbytes());
    return table;
}

namespace {
const char* row_option(const char* name) {
#if TYLOINFER_HAS_CUDA
    return tyloi::cuda::env_lookup(name);
#else
    return std::getenv(name);
#endif
}
}

std::shared_ptr<tyloi::NintRows> tyloi::weights::load_nint_rows(
        const tyloi::source::Source& source, const std::string& name, bool prefer_mapping) {
#if TYLOINFER_ENABLE_MFQ_COMPAT

    const auto& metadata = tyloi::require_tensor(source, name);
    if (metadata.dtype != "NINT") throw std::runtime_error("NINT row table requires packed NINT");
    if (metadata.nbytes > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("NINT row table size overflow");
    auto read = source.tensor_reader(name);
    bool parallel=source.supports_parallel_tensor_reads();
    const auto* batched=row_option("TYLOINFER_PLE_BATCH_READ");
    auto batch=(!batched || batched[0]!='0') ? source.tensor_batch_reader(name) : tyloi::source::Source::TensorBatchReader{};
    bool mapped_file=false;
    const auto* mapped=row_option("TYLOINFER_PLE_FILE_MAP");
    if(mapped ? mapped[0]=='1' : prefer_mapping && source.file_read_stats().mode!=tyloi::FileReadMode::Direct) {
        if(auto mapping=source.mapped_tensor_batch_reader(name)) {
            batch=std::move(mapping);
            read=[batch](uint64_t offset,std::byte* out,std::size_t count){batch({{offset,out,count}});};
            parallel=true;mapped_file=true;
        } else if(auto mapping=source.mapped_tensor_reader(name)) {
            read=std::move(mapping);parallel=true;batch={};mapped_file=true;
        }
    }
    if(const auto* trace=row_option("TYLOINFER_TRACE_PLE_TIMINGS");trace && trace[0]=='1')
        std::fprintf(stderr,"nint_row_source mapped_file=%d nbytes=%llu name=%s\n",int(mapped_file),
            static_cast<unsigned long long>(metadata.nbytes),name.c_str());
    auto table=std::make_shared<tyloi::NintRows>(static_cast<std::size_t>(metadata.nbytes),
        [read = std::move(read)](std::size_t offset, std::uint8_t* out, std::size_t count) {
            read(offset, reinterpret_cast<std::byte*>(out), count);
        }, parallel,std::move(batch));
    tyloi::weights::observe_streamed_ple(table,static_cast<std::size_t>(metadata.nbytes));
    return table;

#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}
