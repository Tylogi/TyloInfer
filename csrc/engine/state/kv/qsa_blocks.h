#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tyloi::cache {

struct QsaKvStoreConfig {
    std::size_t budget_bytes = 0;
    std::size_t target_index_bytes = 0;
    std::size_t buffer_bytes = 64 * 1024 * 1024;
    std::size_t microblock_rows = 4;
    std::string directory;
    std::function<void(std::size_t)> reserve;
};

struct QsaKvStoreStats {
    std::size_t index_bytes = 0;
    std::size_t hot_bytes = 0;
    std::size_t hot_limit = 0;
    std::size_t pending_bytes = 0;
    std::size_t disk_bytes = 0;
    std::uint64_t reads = 0;
    std::uint64_t hits = 0;
    std::uint64_t read_bytes = 0;
    std::uint64_t written_bytes = 0;
    std::size_t budget_bytes = 0;
};

class QsaKvStore {
public:
    struct Block;
    using BlockPtr = std::shared_ptr<const Block>;

    explicit QsaKvStore(QsaKvStoreConfig config);
    ~QsaKvStore();

    QsaKvStore(const QsaKvStore&) = delete;
    QsaKvStore& operator=(const QsaKvStore&) = delete;

    BlockPtr write(std::vector<std::uint8_t> bytes, int priority = 0);
    std::shared_ptr<const std::vector<std::uint8_t>> read(const BlockPtr& block);
    // Read into caller-owned storage without allocating or populating the
    // pageable read cache. Existing pending/hot images remain valid sources.
    void read_into(const BlockPtr& block,std::uint8_t* destination,std::size_t bytes);
    struct ReadRequest {BlockPtr block;std::uint8_t* destination;std::size_t bytes;};
    // Independent destinations, with all blocks belonging to this store.
    // Sequential contiguous file/destination spans read directly into caller
    // storage. Scatter reads retain bounded workspace and amplification;
    // ordinary random recall keeps its exact ranges.
    void read_into_batch(const std::vector<ReadRequest>& requests);
    static std::size_t block_bytes(const BlockPtr& block);
    void flush();
    QsaKvStoreStats stats() const;
    const QsaKvStoreConfig& config() const noexcept;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace tyloi::cache
