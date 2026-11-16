#include "engine/weights/cache/metal_mfe.h"
#include "engine/backends/metal/ops/mfq/mlx_moe.h"
#include "engine/weights/source/source.h"
#include "mfq_container.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace tyloi::metal {
namespace {
// Own the legacy container for every retained record reader.
class MfeLegacySource final : public source::Source {
public:
    explicit MfeLegacySource(std::shared_ptr<MfqContainer> model)
        : model_(std::move(model)) {
        for (const auto& [name, record] : model_->records()) {
            if (name.starts_with("__mfq_asset__/")) continue;
            tensors_.push_back({name, record.dtype, record.stored_dtype, std::nullopt, record.nbytes});
        }
        if (model_->has_legacy_aliases()) {
            std::vector<std::string> names;
            for (const auto& tensor : tensors_) names.push_back(tensor.name);
            const auto aliases = models::legacy_names(model_->header().architecture,
                model_->read_text(std::string(kModelConfigAsset)), names).canonical_to_stored;
            for (const auto& [canonical, stored] : aliases) {
                if (!model_->contains(canonical)) continue;
                const auto& record = model_->record(canonical);
                tensors_.push_back({canonical, record.dtype, record.stored_dtype, std::nullopt, record.nbytes});
            }
        }
        for (std::size_t i = 0; i < tensors_.size(); ++i) index_.emplace(tensors_[i].name, i);
    }
    const std::vector<std::filesystem::path>& source_paths() const noexcept override { return model_->source_paths(); }
    const std::unordered_map<std::string, std::string>& metadata() const noexcept override { return metadata_; }
    const std::vector<TensorMetadata>& tensors() const noexcept override { return tensors_; }
    const TensorMetadata* find_tensor(std::string_view name) const noexcept override {
        const auto it = index_.find(std::string(name));
        return it == index_.end() ? nullptr : &tensors_[it->second];
    }
    void read_range_into(std::string_view name, std::uint64_t offset, std::byte* destination, std::size_t size) const override {
        model_->read_range_into(std::string(name), offset, {destination, size});
    }
    TensorReader tensor_reader(std::string_view name) const override {
        return [model = model_, name = std::string(name)](std::uint64_t offset, std::byte* out, std::size_t size) {
            model->read_range_into(name, offset, {out, size});
        };
    }
    TensorBatchReader tensor_batch_reader(std::string_view name) const override {
        return [model = model_, name = std::string(name)](const std::vector<ReadSpan>& ranges) {
            if (ranges.empty()) return;
            auto offset = ranges.front().offset;
            std::vector<std::span<std::byte>> destinations;
            destinations.reserve(ranges.size());
            for (const auto& range : ranges) {
                if (range.offset != offset) {
                    for (const auto& item : ranges)
                        model->read_range_into(name, item.offset, {item.destination, item.size});
                    return;
                }
                destinations.emplace_back(range.destination, range.size);
                offset += range.size;
            }
            model->readv_range(name, ranges.front().offset, destinations);
        };
    }
    bool supports_parallel_tensor_reads() const noexcept override { return true; }
    const std::vector<std::string>& assets() const noexcept override { return assets_; }
    bool has_asset(std::string_view) const noexcept override { return false; }
    std::vector<std::byte> read_asset(std::string_view) const override { throw std::runtime_error("MFE reader has no assets"); }
private:
    std::shared_ptr<MfqContainer> model_;
    std::vector<TensorMetadata> tensors_;
    std::unordered_map<std::string, std::size_t> index_;
    std::unordered_map<std::string, std::string> metadata_;
    std::vector<std::string> assets_;
};
} // namespace

MlxMfeOffloadCache::MlxMfeOffloadCache(const MfqContainer& model, std::size_t limit, int experts)
    : MlxMfeOffloadCache(std::make_shared<MfqContainer>(model), limit, experts) {}

MlxMfeOffloadCache::MlxMfeOffloadCache(std::shared_ptr<MfqContainer> model, std::size_t limit, int experts)
    : MlxMfeOffloadCache(MfeLegacySource(model), limit, experts,
        [model](const std::string& name) { model->record_prepared(name); }) {}

MlxMfeWeight load_routed_gate_up_weight(
    const MfqContainer& model,
    const std::string& mlp_prefix) {
    const auto base = mlp_prefix + ".experts";
    const auto gate_name = base + ".gate.weight";
    const auto up_name = base + ".up.weight";
    const bool has_gate = model.contains(gate_name);
    const bool has_up = model.contains(up_name);
    if (has_gate != has_up) {
        throw std::runtime_error(
            "incomplete routed Gate/Up pair under " + base);
    }
    const auto load = [&model](const std::string& name) {
        if (model.record(name).dtype != "MFE") {
            throw std::runtime_error(
                "routed expert tensor must use MFE: " + name);
        }
        const auto mapped = model.map_record(name);
        return MlxMfeWeight::from_blob(mapped.view());
    };
    if (has_gate) {
        if (model.record(gate_name).dtype != "MFE" || model.record(up_name).dtype != "MFE") {
            throw std::runtime_error("routed expert tensors must use MFE under " + base);
        }
        const auto gate = model.map_record(gate_name);
        const auto up = model.map_record(up_name);
        const std::array blobs{gate.view(), up.view()};
        return MlxMfeWeight::from_projection_blobs(blobs);
    }
    return load(base + ".gate_up.weight");
}

} // namespace tyloi::metal
