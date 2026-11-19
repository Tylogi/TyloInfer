#pragma once

#include "engine/memory/host.h"
#include "engine/weights/encoding/tq/tensor.h"
#include "engine/weights/cache/lru.h"

#include <array>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tyloi { class TqTable; }
class TqExec;

// Selected weights own their backing after eviction; uploads retain host data
// until DMA completes. Cache mutation is serialized across request streams.
class TqCache final : public ::tyloi::HostCache {
public:
    TqCache(std::size_t budget, std::size_t host_budget, std::size_t process_limit,
            bool retain_all, std::shared_ptr<TqExec> exec);
    ~TqCache() override;

    std::uint64_t add(std::shared_ptr<const tyloi::TqTable>, std::int64_t rows);
    std::size_t prefill_peak(const std::array<std::uint64_t,3>& banks,
                             std::span<const std::int64_t> experts);
    std::vector<::tyloi::encoding::TqTensor> get(std::uint64_t source, const std::int64_t* ids,
                              std::size_t count, int device);
    ::tyloi::DenseTensor views(std::uint64_t source, const std::int64_t* ids,
                                      const std::vector<::tyloi::encoding::TqTensor>& weights, int device);
    std::size_t budget();
    std::size_t resident_bytes();
    void resize(std::size_t bytes);
    void preload(int device, std::size_t workspace_reserve);
    void reclaim(int device, std::size_t available_bytes);
    std::size_t reclaim(std::size_t bytes) override;
    std::size_t reclaim_duplicates(std::size_t bytes);
    std::size_t share_host(std::size_t free_bytes);
    void prepare_decode(int device, std::size_t workspace_reserve);
    // Decode boundary only. Replay can select experts but cannot admit a prepared group.
    void admit_routes(int device, std::size_t workspace_reserve);
    void observe_routes(const std::array<std::uint64_t,3>& banks, std::span<const std::int64_t> ids,
                        int rows, int row_offset, int row_total, int device, bool independent);
    void begin_routes(int rows);
    void commit_routes(std::span<const std::uint8_t> keep, bool single_sequence);
    void discard_routes() noexcept;
    bool set_learning(bool enabled);
    std::uint64_t streamed_bytes();
    std::size_t reserve(std::size_t bytes);
    std::vector<std::pair<std::string, double>> metrics();

private:
    struct HostAccounting;
    struct Upload;
    struct HostImage;
    struct Stage;
    struct HostTransfer {
        std::shared_ptr<Upload> ready;
        std::vector<std::shared_ptr<HostImage>> images;
        ~HostTransfer() noexcept;
    };
    struct DeviceKey {
        std::uint64_t image;
        int device;
        bool operator==(const DeviceKey&) const = default;
    };
    struct DeviceKeyHash {
        std::size_t operator()(const DeviceKey& key) const noexcept {
            return std::hash<std::uint64_t>{}(key.image) ^ (std::hash<int>{}(key.device) << 1);
        }
    };
    struct DeviceEntry {
        ::tyloi::encoding::TqTensor weight;
        std::shared_ptr<Upload> uploaded;
        std::size_t source_bytes = 0;
    };
    struct ViewTable;
    struct Routes;
    struct DeviceStages {
        std::array<std::unique_ptr<Stage>, 2> stages;
        unsigned index = 0;
    };
    struct Source {
        std::shared_ptr<const tyloi::TqTable> table;
        std::int64_t rows;
    };

    // Streams outlive every cached tensor and its final consumer fence.
    std::shared_ptr<TqExec> tq_;
    std::shared_ptr<HostAccounting> host_accounting_;
    tyloi::weights::Lru<std::uint64_t, std::shared_ptr<HostImage>> host_;
    tyloi::weights::Lru<DeviceKey, DeviceEntry, DeviceKeyHash> device_;
    std::optional<std::size_t> device_budget_;
    std::unique_ptr<Routes> routes_;
    std::unordered_set<DeviceKey, DeviceKeyHash> retained_;
    bool learning_ = true;
    std::unordered_map<std::uint64_t, std::vector<ViewTable>> view_tables_;
    std::size_t payload_bytes_ = 0, resident_bound_ = 0, process_host_limit_ = 0;
    std::size_t ordinary_host_limit_;
    std::uint64_t source_reads_ = 0, source_batches_ = 0;
    std::vector<Source> sources_;
    std::mutex mutex_;
    std::unordered_map<int, DeviceStages> stages_;
    std::list<HostTransfer> host_transfers_;
    std::uint64_t calls_ = 0, gpu_hits_ = 0, host_hits_ = 0, source_bytes_ = 0, transfer_bytes_ = 0;
    std::uint64_t direct_transfer_bytes_ = 0, pin_attempts_ = 0, pin_failures_ = 0;
    std::uint64_t route_admit_groups_ = 0, route_admit_images_ = 0, route_admit_bytes_ = 0;
    std::uint64_t route_admit_skips_ = 0, route_admit_failures_ = 0;
    double fetch_ms_ = 0, upload_submit_ms_ = 0;
    bool profile_;

    std::size_t reclaim_host_locked(std::size_t bytes);
    std::size_t reserve_host_headroom_locked(std::size_t bytes);
    std::size_t reserve_host_locked(std::size_t bytes);
    bool admit_host_locked(std::size_t bytes, std::size_t reserve) const;
    std::size_t trim_device(std::size_t bytes, bool protect = false, int only_device = -1);
    bool reserve_device(std::size_t bytes, int device);
    void adapt_routes(bool prefill);
    std::vector<::tyloi::encoding::TqTensor> get_locked(std::uint64_t source,const std::int64_t* ids,
                                                       std::size_t count,int device,bool demand=true);
    void report() const;
    void upload(const ::tyloi::DenseTensor&, const std::vector<std::uint8_t>&);
};
