#include "residual.h"

#include "engine/models/source/gguf.h"
#include "engine/models/source/gguf_source.h"
#include "engine/weights/encoding/gguf_quant.h"
#include "engine/weights/source/json.h"
#include "engine/weights/source/row_residual.h"
#include "nlohmann/json.hpp"

#include <limits>
#include <stdexcept>

namespace tyloi::models {
namespace {

class ResidualModelSource final : public ModelSource {
    std::shared_ptr<const ModelSource> base_;
    std::unordered_map<std::string, std::shared_ptr<const RowResidualFile>> residuals_;
    std::vector<std::filesystem::path> paths_;

public:
    explicit ResidualModelSource(const std::filesystem::path& path) {
        const auto document = source::parse_json(source::read_text(path), path);
        if (document.value("format", std::string{}) != "tyloinfer-row-residual-v1")
            throw std::invalid_argument("unsupported residual model manifest");
        if (!document.contains("base") || !document.at("base").is_string())
            throw std::invalid_argument("row residual manifest has no base model");
        const auto base = path.parent_path() /
            std::filesystem::u8path(document.at("base").get<std::string>());
        if (base.extension() == ".json")
            throw std::invalid_argument("residual manifests cannot reference another manifest");
        base_ = open_model_source(base);
        paths_ = base_->source_paths();
        paths_.push_back(path);
        if (!document.contains("tensors") || !document.at("tensors").is_object())
            throw std::invalid_argument("residual tensors must be an object");
        for (const auto& item : document.at("tensors").items()) {
            auto name = item.key();
            if (!base_->find_tensor(name) &&
                    dynamic_cast<const GgufSource*>(base_.get()) != nullptr) {
                name = gguf_name(name);
            }
            const auto* tensor = base_->find_tensor(name);
            if (!tensor || !tensor->shape || tensor->shape->size() < 2 ||
                    !tyloi::gguf::is_quantized(tensor->dtype)) {
                throw std::invalid_argument(
                    "residual target must be an existing GGUF weight tensor: " + item.key());
            }
            if (!item.value().is_string())
                throw std::invalid_argument("residual target path must be a string: " + item.key());
            const auto residual_path = path.parent_path() /
                std::filesystem::u8path(item.value().get<std::string>());
            auto residual = std::make_shared<RowResidualFile>(residual_path);
            std::uint64_t rows = 1;
            for (std::size_t index = 0; index + 1 < tensor->shape->size(); ++index) {
                const auto extent = tensor->shape->at(index);
                if (extent <= 0 || static_cast<std::uint64_t>(extent) >
                        std::numeric_limits<std::uint64_t>::max() / rows) {
                    throw std::invalid_argument("invalid residual target shape");
                }
                rows *= static_cast<std::uint64_t>(extent);
            }
            if (rows != residual->row_count() ||
                    tensor->shape->back() != static_cast<std::int64_t>(residual->columns())) {
                throw std::invalid_argument("residual geometry differs from base tensor: " + item.key());
            }
            if (!residuals_.emplace(name, std::move(residual)).second)
                throw std::invalid_argument("duplicate residual target after name normalization");
            paths_.push_back(residual_path);
        }
    }

    const std::vector<std::filesystem::path>& source_paths() const noexcept override { return paths_; }
    std::string_view architecture() const noexcept override { return base_->architecture(); }
    TensorLayout tensor_layout() const noexcept override { return base_->tensor_layout(); }
    const std::unordered_map<std::string, std::string>& metadata() const noexcept override { return base_->metadata(); }
    const HadamardTransforms* hadamard_transforms() const noexcept override { return base_->hadamard_transforms(); }
    const std::vector<TensorMetadata>& tensors() const noexcept override { return base_->tensors(); }
    const TensorMetadata* find_tensor(std::string_view name) const noexcept override { return base_->find_tensor(name); }
    void read_range_into(std::string_view name, std::uint64_t offset,
            std::byte* destination, std::size_t size) const override {
        base_->read_range_into(name, offset, destination, size);
    }
    void drop_file_cache() const noexcept override { base_->drop_file_cache(); }
    FileReadStats file_read_stats() const noexcept override { return base_->file_read_stats(); }
    bool supports_parallel_tensor_reads() const noexcept override { return base_->supports_parallel_tensor_reads(); }
    std::optional<TensorFileRange> tensor_file_range(std::string_view name) const override { return base_->tensor_file_range(name); }
    std::optional<TensorFileRange> tensor_file_range_at(std::string_view name, std::uint64_t offset, std::size_t size) const override { return base_->tensor_file_range_at(name, offset, size); }
    TensorReader tensor_reader(std::string_view name) const override { return base_->tensor_reader(name); }
    TensorBatchReader tensor_batch_reader(std::string_view name) const override { return base_->tensor_batch_reader(name); }
    TensorReader mapped_tensor_reader(std::string_view name) const override { return base_->mapped_tensor_reader(name); }
    TensorBatchReader mapped_tensor_batch_reader(std::string_view name) const override { return base_->mapped_tensor_batch_reader(name); }
    const std::vector<std::string>& assets() const noexcept override { return base_->assets(); }
    bool has_asset(std::string_view name) const noexcept override { return base_->has_asset(name); }
    std::vector<std::byte> read_asset(std::string_view name) const override { return base_->read_asset(name); }
    std::optional<ModelGraph> model_graph() const override { return base_->model_graph(); }
    std::shared_ptr<const RowResidualFile> row_residual(std::string_view name) const override {
        const auto found = residuals_.find(std::string(name));
        return found == residuals_.end() ? nullptr : found->second;
    }
};

} // namespace

std::shared_ptr<const ModelSource> open_residual_model_source(
        const std::filesystem::path& path) {
    return std::make_shared<ResidualModelSource>(path);
}

} // namespace tyloi::models
