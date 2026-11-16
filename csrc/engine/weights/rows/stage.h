#pragma once
#include "engine/tensor/tensor.h"
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

// A prepared table-row transfer. Encodings stay in their own loader; graph
// consumers use stable output storage and never allocate inside upload/decode.
class RowStage {
public:
    virtual ~RowStage()=default;
    virtual const ::tyloi::Tensor& output()const=0;
    virtual bool mapped_rows()const=0;
    virtual void upload(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape)=0;
    virtual void decode()=0;
    // The owning decode window has completed earlier consumers before this
    // upload; this window's consumer remains behind its publication flag.
    // Providers needing DMA retain their normal upload synchronization.
    virtual void upload_graph(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape) {
        upload(ids,shape);
    }
};

namespace tyloi::weights {
class RowReader {
public:
    struct Source;
    static std::shared_ptr<const Source> source(std::vector<Tensor> tables, std::int64_t rows_per_shard);
    static std::vector<std::pair<Device, std::size_t>> stage_growth(
        const std::shared_ptr<const Source>&, const std::vector<std::int64_t>& shape, const Device& device);
    explicit RowReader(std::shared_ptr<const Source> source);
    ~RowReader();
    bool prefetch(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape, bool early_prefix);
    Tensor collect(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape, const Device& device);
    std::shared_ptr<RowStage> stage(const std::vector<std::int64_t>& shape, const Device& device);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
