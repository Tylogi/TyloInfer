#pragma once

#include "engine/memory/host.h"
#include "engine/weights/cache/lru.h"
#include "engine/weights/moe/ggml_bank.h"
#include "engine/weights/source/source.h"

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::cuda { class Event; }

namespace tyloi::weights {

struct PreparedExperts;

class GgmlExpertCache final : public HostCache {
public:
    GgmlExpertCache(std::size_t device_bytes, std::size_t host_bytes, std::size_t host_limit);
    ~GgmlExpertCache() override;
    std::uint32_t register_source(const std::shared_ptr<const GgmlExpertBank>& bank,
        source::Source::TensorReader read, source::Source::TensorBatchReader read_batch);
    PreparedExperts prepare(const GgmlExpertBank& bank, std::span<const std::int64_t> ids,
        bool graph_upload = false);
    void observe_routes(std::span<const GgmlExpertBank* const> banks, const DenseTensor& ids,
        int row_offset, int row_total, bool prefill);
    struct Prefill;
    std::shared_ptr<Prefill> begin_prefill(std::span<const int> request_rows);
    void detach_prefill(const Prefill& journal) noexcept;
    void finish_prefill(Prefill& journal, std::span<const std::uint8_t> accepted);
    void begin_routes(int rows);
    void commit_routes(std::span<const std::uint8_t> keep, bool single_sequence);
    void discard_routes() noexcept;
    bool set_learning(bool enabled);
    void admit_routes(Device device, std::size_t workspace_reserve);
    void resize(std::size_t bytes);
    void reclaim_device(Device device, std::size_t available);
    void share_host(std::size_t available);
    std::size_t reclaim(std::size_t bytes) override;
    std::size_t reclaim_duplicates(std::size_t bytes);
    std::size_t budget() const;
    bool has_cached_sources() const;
    bool host_routes_required() const;
    std::uint64_t streamed_bytes() const;
    std::vector<std::pair<std::string, double>> metrics() const;

private:
    struct Routes;
    struct Reader {
        std::weak_ptr<const GgmlExpertBank> bank;
        source::Source::TensorReader read;
        source::Source::TensorBatchReader read_batch;
    };
    struct HostImage {
        DenseTensor tensor;
        std::atomic_size_t& physical;
        HostImage(DenseTensor tensor, std::atomic_size_t& physical);
        ~HostImage();
    };
    struct DeviceImage {
        DenseTensor tensor;
        std::shared_ptr<cuda::Event> ready;
    };
    struct Upload {
        DenseTensor packet;
        std::shared_ptr<cuda::Event> ready;
    };
    mutable std::mutex mutex_;
    std::mutex staging_mutex_;
    std::array<Upload, 2> staging_;
    std::atomic_size_t host_physical_{0};
    Lru<std::uint64_t, std::shared_ptr<const HostImage>> host_;
    Lru<std::uint64_t, DeviceImage> device_;
    std::size_t host_limit_, ordinary_host_budget_, source_bytes_ = 0, permanent_bytes_ = 0;
    std::vector<Reader> sources_;
    std::unique_ptr<Routes> routes_;
    Prefill* prefill_ = nullptr;
    bool learning_ = true;
    bool host_routes_required_ = false;
    std::uint64_t fetched_ = 0, reads_ = 0, transferred_ = 0, device_hits_ = 0, host_hits_ = 0;
};

} // namespace tyloi::weights
