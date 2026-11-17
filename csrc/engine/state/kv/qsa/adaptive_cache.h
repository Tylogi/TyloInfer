#pragma once
#include "engine/tensor/encoding/qsa/image.h"
#include "engine/state/kv/qsa_blocks.h"
#include "engine/state/kv/qsa/precision_policy.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tyloinfer::qsa {

struct BlockReplacement {
    std::size_t block = 0;
    std::uint8_t tier = 5;
    std::shared_ptr<const MicroblockImage> image;
};
struct AdaptiveCacheConfig {
    PrecisionPolicyConfig policy;
    std::size_t rows_per_kind = 0;
    std::uint8_t canonical_tag = affine8_tag;
    RowPrecisionAllocation row_allocation = RowPrecisionAllocation::Energy;
    std::uint8_t tq_state_bits = 16;
    std::size_t io_buffer_bytes = 8ULL<<20;
    // Without originals on SSD, retain only the current compressed image and
    // permit downward transitions. Raising a bit count cannot recover data.
    bool ssd_cache = false;
    std::string directory;
};
struct AdaptiveCacheStats {
    std::size_t blocks=0,resident_bytes=0;
    std::array<std::size_t,6> blocks_per_tier{};
    std::uint64_t promotions=0,demotions=0,updates=0;
    std::uint64_t canonical_read_bytes=0;
    double last_update_ms=0;
    tyloi::cache::QsaKvStoreStats canonical;
};

class AdaptiveCache {
public:
    using Reencode=std::function<MicroblockImage(const MicroblockImage&,Precision)>;
    using ReencodeBatch=std::function<std::vector<MicroblockImage>(
        const std::vector<MicroblockImage>&,const std::vector<Precision>&)>;
    // A publisher must retain source images through its asynchronous copies
    // and publish descriptors only after those copies and prior readers finish.
    // A throwing publisher must leave device descriptors unchanged.
    using Publish=std::function<void(const std::vector<BlockReplacement>&)>;
    AdaptiveCache(AdaptiveCacheConfig config,Reencode encode,Publish publish);
    AdaptiveCache(AdaptiveCacheConfig config,ReencodeBatch encode,Publish publish);
    ~AdaptiveCache();
    AdaptiveCache(const AdaptiveCache&)=delete;
    AdaptiveCache& operator=(const AdaptiveCache&)=delete;
    std::size_t append(MicroblockImage canonical,std::uint64_t token_begin);
    // Publish contiguous microblocks together. Returns their first block ID;
    // an empty batch leaves the cache unchanged.
    std::size_t append_batch(std::vector<MicroblockImage> canonical,std::uint64_t token_begin);
    // Observations are consumed once. A pending request coalesces observations
    // instead of discarding them when another re-quantization is in progress.
    void update(std::uint64_t tokens,std::size_t budget,
        std::vector<double> hits,std::vector<double> attention_mass);
    // Finish pending quantization and release retired host images.
    void wait();
    void truncate(std::size_t blocks);
    // Immutable original, including after the resident image was demoted.
    MicroblockImage canonical_image(std::size_t block) const;
    MicroblockImage reconstruction_image(std::size_t block) const;
    AdaptiveCacheStats stats() const;
    std::array<std::size_t,6> tier_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tyloinfer::qsa
