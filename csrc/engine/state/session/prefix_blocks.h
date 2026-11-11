#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include "components/io/read_span.h"

namespace tyloi {class FileRangeReader;}

namespace tyloi::cache {

using BlockHash = std::array<std::uint8_t, 32>;
using PagedPrefixPayload =
    std::shared_ptr<const std::vector<std::uint8_t>>;

// Retains the existing SSD file, without retaining its full payload in RAM.
// Byte ranges remain readable after cache eviction, replacement or clear().
struct PagedPrefixReadLease {
    std::size_t payload_bytes=0;
    std::function<std::vector<std::uint8_t>(std::size_t,std::size_t)> read_range;
    std::function<void(std::size_t,std::uint8_t*,std::size_t)> read_range_into;
    std::function<void(const std::vector<tyloi::ReadSpan>&)> read_ranges_into;
    std::shared_ptr<const tyloi::FileRangeReader> file_reader;
    std::uint64_t file_offset=0;
};

BlockHash sha256(const void* data, std::size_t size);
BlockHash sha256(std::string_view value);
std::string block_hash_hex(const BlockHash& hash);
BlockHash prefix_block_hash(
    const BlockHash& compatibility_hash,
    std::size_t block_size_tokens,
    const BlockHash& parent,
    const std::int64_t* token_ids,
    std::size_t token_count,
    std::string_view extra_key = {});

struct PrefixBlockStoreConfig {
    std::filesystem::path cache_dir;
    std::string compatibility_key;
    std::size_t block_size_tokens = 256;
    std::uint64_t max_disk_bytes = 100ULL * 1024ULL * 1024ULL * 1024ULL;
    std::uint64_t max_hot_bytes = 0;
    std::size_t max_pending_writes = 64;
    // Zero forces writes onto the caller instead of retaining raw KV payloads
    // in the asynchronous queue. The count limit remains a second guard.
    std::uint64_t max_pending_bytes = 512ULL * 1024ULL * 1024ULL;
    std::size_t max_parallel_reads = 4;
    std::size_t retain_tail_versions = 2;
    std::uint64_t min_disk_free_bytes = 1ULL << 30;
    std::filesystem::path model_path;
    std::function<void(std::size_t)> reserve_memory;
    // Disposable intermediate RAM tier: persist on eviction, never pin RAM.
    bool spill_hot_on_eviction = false;
    std::function<std::uint64_t()> available_ram_bytes;
    std::uint64_t min_free_ram_bytes = 512ULL << 20;
};

struct PrefixBlockStoreMetrics {
    std::uint64_t blocks = 0;
    std::uint64_t disk_hits = 0;
    std::uint64_t hot_hits = 0;
    std::uint64_t writes = 0;
    std::uint64_t deduplicated_writes = 0;
    std::uint64_t failed_writes = 0;
    std::uint64_t corrupt_blocks = 0;
    std::uint64_t evictions = 0;
    std::uint64_t disk_blocks = 0;
    std::uint64_t disk_bytes = 0;
    std::uint64_t hot_blocks = 0;
    std::uint64_t hot_bytes = 0;
    std::uint64_t pending_writes = 0;
    std::uint64_t pending_bytes = 0;
    std::uint64_t resident_bytes = 0;
    std::uint64_t pending_max_bytes = 0;
    std::uint64_t hot_pressure_bytes = 0;
    std::uint64_t superseded_blocks = 0;
    std::uint64_t low_disk_space_skips = 0;
};

class PrefixBlockStore {
public:
    explicit PrefixBlockStore(PrefixBlockStoreConfig config);
    ~PrefixBlockStore();

    PrefixBlockStore(const PrefixBlockStore&) = delete;
    PrefixBlockStore& operator=(const PrefixBlockStore&) = delete;

    std::size_t block_size_tokens() const noexcept;
    const BlockHash& compatibility_hash() const noexcept;

    BlockHash store(
        const BlockHash& hash,
        const BlockHash& parent,
        const std::int64_t* token_ids,
        std::size_t token_count,
        PagedPrefixPayload payload,
        std::string_view extra_key = {},
        bool replace_existing = false);
    bool contains(
        const BlockHash& hash,
        const BlockHash& parent,
        std::size_t token_count);
    std::vector<std::size_t> tail_lengths(
        const BlockHash& parent, std::size_t maximum) const;

    std::optional<std::vector<std::uint8_t>> load(const BlockHash& hash);
    std::vector<PagedPrefixPayload> load_prefix(
        const std::vector<BlockHash>& blocks);

    // No SSD lease is returned while a newer RAM version awaits persistence.
    // load/load_prefix still expose that current version without forcing a write.
    std::shared_ptr<const PagedPrefixReadLease> borrow(const BlockHash& block);
    // Verify file leases with the configured bounded read parallelism. Keeps
    // only file handles and small checksum buffers; missing entries are null.
    std::vector<std::shared_ptr<const PagedPrefixReadLease>> borrow_prefix(
        const std::vector<BlockHash>& blocks);

    // Remove a block that passed file-integrity checks but failed the
    // runtime codec's semantic validation. This rare path waits for an
    // in-flight write of the same content-addressed block to settle first.
    bool invalidate(const BlockHash& block);

    void pin(const std::vector<BlockHash>& blocks);
    void unpin(const std::vector<BlockHash>& blocks);
    std::uint64_t trim_hot(std::uint64_t target_bytes = 0);
    std::uint64_t set_hot_limit(std::uint64_t max_bytes);
    std::uint64_t reserve_ram(std::uint64_t allocation_bytes);
    std::uint64_t set_disk_limit(std::uint64_t max_bytes);
    std::size_t refresh_disk_index();
    void flush();
    std::size_t clear();
    PrefixBlockStoreMetrics metrics() const;

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;
};

// Disposable RAM payloads yield to active KV before its precision is reduced.
// The callback does not extend the cache's lifetime or capture its engine.
std::function<std::uint64_t(std::uint64_t)> make_ram_prefix_reclaimer(
    std::weak_ptr<PagedPrefixCache> cache);

} // namespace tyloi::cache
