#include "engine/weights/rows/gguf_read.h"
#include "engine/weights/rows/gguf.h"
#include "engine/weights/rows/prefetch.h"
#include "engine/backends/cpu/native/host_parallel.h"
#include "engine/backends/cuda/native/tensor/tensor.h"

#include <algorithm>
#include <cstring>
#include <future>
#include <limits>
#include <stdexcept>

namespace tyloi::weights {
namespace {
struct Job {
    std::vector<std::shared_ptr<GgufRows>> tables;
    std::int64_t rows;
    std::vector<std::int64_t> ids;
    DenseTensor packet;
    std::promise<DenseTensor> result;
};

void read(Job& job) {
    const auto stride = static_cast<std::size_t>(job.packet.size(1));
    auto* data = static_cast<std::byte*>(job.packet.data_ptr());
    if (job.packet.nbytes()) std::memset(data, 0, job.packet.nbytes());
    std::vector<std::vector<ReadSpan>> requests(job.tables.size());
    for (std::size_t i = 0; i < job.ids.size(); ++i) {
        const auto shard = static_cast<std::size_t>(job.ids[i] / job.rows);
        const auto row = static_cast<std::uint64_t>(job.ids[i] % job.rows);
        const auto& table = job.tables[shard];
        requests[shard].push_back({row * table->row_bytes(), data + i * stride,
            static_cast<std::size_t>(table->staged_row_bytes())});
    }
    std::vector<std::size_t> active;
    for (std::size_t i = 0; i < requests.size(); ++i)
        if (!requests[i].empty()) active.push_back(i);
    auto run = [&](std::int64_t begin, std::int64_t end) {
        for (auto i = begin; i < end; ++i) {
            const auto shard = active[static_cast<std::size_t>(i)];
            job.tables[shard]->read_staged_rows(requests[shard]);
        }
    };
    const bool parallel = std::all_of(job.tables.begin(), job.tables.end(),
        [](const auto& table) { return table->parallel_reads(); });
    if (parallel && active.size() > 1)
        host_parallel_for(0, static_cast<std::int64_t>(active.size()), 1,
            static_cast<int>(std::min<std::size_t>(active.size(), 8)), run);
    else run(0, static_cast<std::int64_t>(active.size()));
}

}

struct GgufRead::Impl {
    std::vector<std::shared_ptr<GgufRows>> tables;
    std::int64_t rows, stride;
    bool parallel;
    std::vector<std::int64_t> ids, shape;
    std::future<DenseTensor> pending;

    void validate(const std::vector<std::int64_t>& selected, const std::vector<std::int64_t>& dimensions) const {
        std::int64_t count = 1;
        for (const auto size : dimensions) {
            if (size < 0 || (size && count > std::numeric_limits<std::int64_t>::max() / size))
                throw std::invalid_argument("invalid GGUF row shape");
            count *= size;
        }
        if (static_cast<std::uint64_t>(count) != selected.size())
            throw std::invalid_argument("GGUF row shape mismatch");
        if (stride && count > std::numeric_limits<std::int64_t>::max() / stride)
            throw std::overflow_error("GGUF row packet size");
        for (const auto id : selected) {
            if (id < 0 || static_cast<std::uint64_t>(id / rows) >= tables.size() ||
                id % rows >= tables[static_cast<std::size_t>(id / rows)]->rows())
                throw std::out_of_range("GGUF row id out of range");
        }
    }
    std::unique_ptr<Job> make_job(const std::vector<std::int64_t>& selected) const {
        auto job = std::make_unique<Job>();
        job->tables = tables;
        job->rows = rows;
        job->ids = selected;
        job->packet = cuda::empty({static_cast<std::int64_t>(selected.size()), stride},
            TensorOptions().dtype(kUInt8).device(kCPU));
        return job;
    }
};

GgufRead::GgufRead(std::vector<std::shared_ptr<GgufRows>> tables, std::int64_t rows)
    : impl_(std::make_unique<Impl>()) {
    if (tables.empty() || rows <= 0 || !tables.front())
        throw std::invalid_argument("invalid GGUF row shards");
    std::uint64_t stride = 0;
    for (std::size_t i = 0; i < tables.size(); ++i) {
        const auto& table = tables[i];
        if (!table || table->width() != tables.front()->width() || table->rows() <= 0 ||
            table->rows() > rows || (i + 1 < tables.size() && table->rows() != rows) ||
            table->row_bytes() == 0 ||
            static_cast<std::uint64_t>(table->rows() - 1) > std::numeric_limits<std::uint64_t>::max() / table->row_bytes())
            throw std::invalid_argument("invalid GGUF row shard geometry");
        stride = std::max(stride, table->staged_row_bytes());
    }
    if (stride > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) - 7)
        throw std::overflow_error("GGUF row stride");
    impl_->stride = static_cast<std::int64_t>((stride + 7) & ~std::uint64_t(7));
    impl_->rows = rows;
    impl_->parallel = std::all_of(tables.begin(), tables.end(),
        [](const auto& table) { return table->parallel_reads(); });
    impl_->tables = std::move(tables);
}
GgufRead::~GgufRead() { clear(); }

bool GgufRead::issue(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) {
    impl_->validate(ids, shape);
    if (!impl_->parallel) return false;
    if (impl_->pending.valid() && impl_->ids == ids && impl_->shape == shape) return true;
    clear();
    auto job = impl_->make_job(ids);
    auto future = job->result.get_future();
    impl_->ids = ids;
    impl_->shape = shape;
    submit_row_read(std::packaged_task<void()>([job = std::move(job)] {
        try { read(*job); job->result.set_value(std::move(job->packet)); }
        catch (...) { job->result.set_exception(std::current_exception()); }
    }));
    impl_->pending = std::move(future);
    return true;
}
DenseTensor GgufRead::collect(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) {
    impl_->validate(ids, shape);
    if (impl_->pending.valid() && impl_->ids == ids && impl_->shape == shape) {
        auto future = std::move(impl_->pending);
        impl_->ids.clear();
        impl_->shape.clear();
        return future.get();
    }
    clear();
    auto job = impl_->make_job(ids);
    read(*job);
    return std::move(job->packet);
}
void GgufRead::clear() {
    if (impl_->pending.valid()) impl_->pending.wait();
    impl_->pending = {};
    impl_->ids.clear();
    impl_->shape.clear();
}
}
