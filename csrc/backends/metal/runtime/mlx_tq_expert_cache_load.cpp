#include "mlx_tq_expert_cache_internal.h"
#include "mfq_container.h"
#include "engine/backends/metal/native/tq_pool.h"
#include <mlx/allocator.h>
#include <algorithm>
#include <array>
#include <climits>
#include <exception>
#include <stdexcept>
#include <utility>
#include <nlohmann/json.hpp>

namespace tyloi::metal {
using namespace mlx::core;
namespace {
using tq_cache_detail::aligned;
struct LegacyBank {
    MfqMappedBytes mapping;
    std::unique_ptr<MlxMappedTqRows> rows;
};
}

MlxTqExpertCache::MlxTqExpertCache(const MfqContainer& source, const std::vector<std::string>& prefixes,
        std::size_t budget, int io_workers) {
    tq_cache_detail::LegacyPlan plan;
    try {
    plan.io_workers = io_workers;
    if (prefixes.empty() || prefixes.size() > std::size_t(UINT32_MAX) || io_workers < 1)
        throw std::invalid_argument("invalid TQ expert cache configuration");
    std::vector<std::array<std::shared_ptr<LegacyBank>, 3>> mappings(prefixes.size());
    auto retained = std::make_shared<MfqContainer>(source);
    const auto meta = source.header().extra_json.find("tq_shapes");
    const auto shapes = meta == source.header().extra_json.end() ? nlohmann::json::object()
        : nlohmann::json::parse(meta->second);
    std::array<std::size_t, 3> largest{};
    plan.layers.resize(prefixes.size());
    for (std::size_t index = 0; index < plan.layers.size(); ++index) {
        auto& layer = plan.layers[index];
        for (int projection = 0; projection < 3; ++projection) {
            auto& bank = layer.banks[projection];
            auto legacy = std::make_shared<LegacyBank>();
            mappings[index][projection] = legacy;
            bank.name = prefixes[index] + ".mlp.experts."
                + std::array<const char*, 3>{"gate", "up", "down"}[projection] + ".weight";
            std::vector<int> shape;
            if (source.record(bank.name).dtype != "TQ")
                throw std::invalid_argument("TQ expert cache requires three TQ banks per layer");
            legacy->mapping = source.map_record(bank.name, true);
            bank.reader = [retained, legacy, name = bank.name](const std::vector<ReadSpan>& spans) {
                (void)legacy; // Keep the mapped metadata and rows alive with the bank.
                if (spans.empty()) return;
                std::vector<std::span<std::byte>> destinations;
                destinations.reserve(spans.size());
                auto offset = spans.front().offset;
                for (const auto& span : spans) {
                    if (span.offset != offset) throw std::logic_error("noncontiguous TQ expert read");
                    destinations.emplace_back(span.destination, span.size);
                    offset += span.size;
                }
                retained->readv_range(name, spans.front().offset, destinations);
            };
            if (shapes.contains(bank.name)) shape = shapes.at(bank.name).get<std::vector<int>>();
            else {
                legacy->rows = std::make_unique<MlxMappedTqRows>(legacy->mapping.view());
                const auto& rows = *legacy->rows;
                if (!rows.expert_container()) throw std::invalid_argument("TQ expert bank lacks logical shape");
                shape = {int(rows.expert_count()), int(rows.rows() / rows.expert_count()), rows.width()};
            }
            plan.backing += source.record(bank.name).nbytes;
            if (shape.size() != 3 || std::any_of(shape.begin(), shape.end(), [](int x) { return x < 1; })
                || (projection && layer.experts != shape[0]))
                throw std::invalid_argument("invalid TQ expert bank geometry");
            layer.experts = shape[0];
            if (!legacy->rows) legacy->rows = std::make_unique<MlxMappedTqRows>(legacy->mapping.view(), shape[2]);
            if (legacy->rows->rows() != std::int64_t(shape[0]) * shape[1]
                || (legacy->rows->expert_container() && legacy->rows->expert_count() != std::size_t(shape[0])))
                throw std::invalid_argument("TQ expert bank row count differs");
            for (int expert = 0; expert < layer.experts; ++expert) {
                bank.ranges.push_back(legacy->rows->packed_rows(std::int64_t(expert) * shape[1], shape[1]));
                const auto span = bank.ranges.at(expert);
                plan.capacity[projection] += aligned(span.bytes);
                largest[projection] = std::max(largest[projection], aligned(span.bytes));
            }
        }
    }
    for (int projection = 0; projection < 3; ++projection) {
        const auto words = (largest[projection] + 3) / 4;
        auto temporary = array(allocator::malloc(words * 4), Shape{int(words)}, uint32);
        for (std::size_t index = 0; index < plan.layers.size(); ++index) {
            auto& layer = plan.layers[index];
            auto& bank = layer.banks[projection];
            const auto& legacy = mappings[index][projection];
            const std::vector<std::uint64_t> offsets(layer.experts, 0);
            bank.weight.emplace(legacy->rows->expert_container()
                ? MlxTqWeight::from_expert_rows(*legacy->rows, layer.experts, &temporary, offsets)
                : MlxTqWeight::from_pool(legacy->mapping.view(), temporary, offsets, legacy->rows->width()));
            bank.weight->words_ = bank.weight->descriptors_;
        }
    }
    plan.largest = largest;
    for(int projection=0;projection<3;++projection) {
        auto& chunk=plan.chunk[projection];
        chunk=pool_page();
        std::size_t layer_max=0;
        for(const auto& layer:plan.layers) {
            std::size_t bytes=0;
            for(const auto& range:layer.banks[projection].ranges)bytes+=aligned(range.bytes);
            layer_max=std::max(layer_max,bytes);
        }
        const auto device_limit=std::size_t(mlx::core::metal::device(Device::gpu).mtl_device()->maxBufferLength())/tyloi::metal::pool_page()*tyloi::metal::pool_page();
        chunk=std::min(device_limit,std::max(chunk,(layer_max+tyloi::metal::pool_page()-1)/tyloi::metal::pool_page()*tyloi::metal::pool_page()));
        std::size_t packed=0;
        for(const auto& layer:plan.layers)for(const auto& range:layer.banks[projection].ranges) {
            const auto size=aligned(range.bytes);
            if(size>chunk)throw std::invalid_argument("TQ expert exceeds the Metal chunk size");
            if(size>chunk-packed%chunk)packed=(packed/chunk+1)*chunk;
            packed+=size;
        }
        plan.capacity[projection]=packed;
    }
    for (auto& capacity : plan.capacity) capacity = (capacity + tyloi::metal::pool_page() - 1) / tyloi::metal::pool_page() * tyloi::metal::pool_page();
    load_legacy(std::move(plan), budget);
    } catch (...) {
        const auto failure = std::current_exception();
        try { mlx::core::synchronize(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}


} // namespace tyloi::metal
