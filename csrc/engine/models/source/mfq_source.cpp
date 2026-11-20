#include "engine/models/source/model.h"
#include "decision.h"
#include "mfq_source.h"
#include "mfq.h"
#include "engine/weights/source/files.h"
#include "engine/weights/source/mapped.h"

#include "engine/models/source/legacy_names.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tyloi::models {

namespace {
using json = nlohmann::json;

std::vector<std::int64_t> metadata_shape(const json& value, const std::string& name) {
    if (!value.is_array()) throw std::runtime_error("MFQ tensor shape is not an array: " + name);
    std::vector<std::int64_t> shape;
    shape.reserve(value.size());
    for (const auto& dimension : value) {
        if (!dimension.is_number_integer() ||
            (dimension.is_number_unsigned() &&
             dimension.get<std::uint64_t>() > std::uint64_t(std::numeric_limits<std::int64_t>::max())))
            throw std::runtime_error("invalid MFQ tensor dimension: " + name);
        const auto size = dimension.get<std::int64_t>();
        if (size < 0) throw std::runtime_error("negative MFQ tensor dimension: " + name);
        shape.push_back(size);
    }
    return shape;
}
} // namespace

struct MfqSource::Impl {
    std::unique_ptr<source::Files> files;
    archive::Index index;
    std::vector<TensorMetadata> tensors;
    std::vector<std::string> assets;
    TensorLayout layout;
    std::unordered_map<std::string, std::string> metadata;
    std::unordered_map<std::string, std::size_t> tensors_by_name;
};

MfqSource::MfqSource(std::filesystem::path requested, FileReadMode mode)
    : impl_(std::make_unique<Impl>()) {
    const auto initial_path = archive::resolve(requested);
    std::vector<std::filesystem::path> source_paths;
    auto initial = archive::read(initial_path);
    const auto split_no = archive::integer(initial.header, "split.no", 0);
    const auto split_count = archive::integer(initial.header, "split.count", 1);
    if (split_count == 0 || split_count > 99999 || split_no >= split_count) {
        throw std::runtime_error("invalid MFQ split metadata: " +
                                 initial_path.string());
    }

    archive::Merge merge(initial.header, split_count);

    if (split_count == 1) {
        source_paths.push_back(initial_path);
        merge.add(std::move(initial), 0);
    } else {
        // A numbered entry may be a symlink to an unnumbered downloaded file.
        // Discover siblings from the requested name before resolving each file.
        auto parts = archive::shards(
            std::filesystem::absolute(requested).lexically_normal(),
            split_no, split_count);
        engram_shards(parts);
        const auto paths = archive::resolve(std::move(parts));
        for (std::uint64_t index = 0; index < split_count; ++index) {
            auto current = archive::read(paths[static_cast<std::size_t>(index)]);
            check_engram(paths[index], current.records);
            merge.add(std::move(current), index);
        }
        source_paths = paths;
    }

    impl_->index = merge.finish();
    impl_->files = std::make_unique<source::Files>(std::move(source_paths), mode);
    std::vector<std::string> stored_names;
    for (const auto& record : impl_->index.records) {
        if (record.asset) {
            impl_->assets.push_back(record.tensor.name);
        } else {
            stored_names.push_back(record.tensor.name);
        }
    }
    std::sort(impl_->assets.begin(), impl_->assets.end());

    impl_->metadata = impl_->index.header.metadata;
    std::unordered_map<std::string, json> tensor_metadata;
    std::unordered_set<std::string> logical_names;
    for (const auto* key : {"tensor_shapes", "tq_shapes", "gguf_mixed_experts", "gguf_hadamard",
                            "tq_row_shards", "tq.segments"}) {
        const auto found = impl_->metadata.find(key);
        if (found == impl_->metadata.end()) continue;
        auto values = json::parse(found->second);
        if (!values.is_object()) throw std::runtime_error(std::string("invalid MFQ tensor metadata: ") + key);
        if (std::string_view(key) == "tq_row_shards" || std::string_view(key) == "tq.segments") {
            for (const auto& [name, parts] : values.items()) {
                if (!parts.is_array() || parts.empty() ||
                    std::any_of(parts.begin(), parts.end(), [](const auto& part) { return !part.is_string(); }))
                    throw std::runtime_error("invalid MFQ TQ row-shard manifest: " + name);
                if (!impl_->index.names.count(name)) {
                    logical_names.insert(name);
                    stored_names.push_back(name);
                }
            }
        }
        tensor_metadata.emplace(key, std::move(values));
    }

    std::unordered_map<std::string, std::string> stored_to_canonical;
    Names aliases;
    if (!has_asset(kModelGraphAsset) && has_asset(kModelConfigAsset)) {
        const auto config = read_asset(kModelConfigAsset);
        aliases = legacy_names(
            architecture(),
            std::string_view(
                reinterpret_cast<const char*>(config.data()), config.size()),
            stored_names);
        for (const auto& [canonical, stored] :
             aliases.canonical_to_stored) {
            const auto record = impl_->index.names.find(stored);
            if ((record == impl_->index.names.end() && !logical_names.count(stored)) ||
                    (record != impl_->index.names.end() && impl_->index.records[record->second].asset) ||
                    impl_->index.names.find(canonical) !=
                        impl_->index.names.end()) {
                throw std::runtime_error(
                    "invalid legacy MFQ tensor alias: " + canonical);
            }
            const auto [found, inserted] =
                stored_to_canonical.emplace(stored, canonical);
            if (!inserted && found->second != canonical) {
                throw std::runtime_error(
                    "legacy MFQ tensor has multiple canonical names: " +
                    stored);
            }
        }
    }

    for (auto& [key, values] : tensor_metadata) {
        auto normalized = json::object();
        for (const auto& [stored, entry] : values.items()) {
            const auto alias = stored_to_canonical.find(stored);
            const auto& name = alias == stored_to_canonical.end() ? stored : alias->second;
            auto value = entry;
            if (key == "tensor_shapes" || key == "tq_shapes") (void)metadata_shape(value, name);
            if (key == "tq_row_shards" || key == "tq.segments") {
                for (auto& part : value) {
                    const auto stored_part = part.get<std::string>();
                    if (const auto found = stored_to_canonical.find(stored_part); found != stored_to_canonical.end())
                        part = found->second;
                }
            }
            if (normalized.contains(name) && normalized.at(name) != value)
                throw std::runtime_error("conflicting canonical MFQ tensor metadata: " + name);
            normalized[name] = std::move(value);
        }
        values = std::move(normalized);
        impl_->metadata[key] = values.dump();
    }

    for (const auto& record : impl_->index.records) {
        if (record.asset) continue;
        auto tensor = record.tensor;
        if (const auto found = stored_to_canonical.find(tensor.name);
            found != stored_to_canonical.end()) {
            tensor.name = found->second;
        }
        for (const auto* key : {"tq_shapes", "tensor_shapes"}) {
            if (std::string_view(key) == "tq_shapes" && tensor.dtype != "TQ") continue;
            const auto found = tensor_metadata.find(key);
            if (found != tensor_metadata.end() && found->second.contains(tensor.name)) {
                tensor.shape = metadata_shape(found->second.at(tensor.name), tensor.name);
                break;
            }
        }
        if (!impl_->tensors_by_name.emplace(
                tensor.name, impl_->tensors.size()).second) {
            throw std::runtime_error(
                "legacy MFQ tensor aliases collide at canonical name: " +
                tensor.name);
        }
        impl_->tensors.push_back(std::move(tensor));
    }
    for (const auto& [stored, canonical] : stored_to_canonical) {
        if (const auto found = impl_->index.names.find(stored); found != impl_->index.names.end())
            impl_->index.names.emplace(canonical, found->second);
    }
    impl_->layout = aliases.layout;
}

MfqSource::~MfqSource() = default;
MfqSource::MfqSource(MfqSource&&) noexcept = default;
MfqSource& MfqSource::operator=(MfqSource&&) noexcept = default;

const archive::Header& MfqSource::header() const noexcept { return impl_->index.header; }
const std::vector<std::filesystem::path>& MfqSource::source_paths() const noexcept {
    return impl_->files->paths();
}
const std::vector<archive::Record>& MfqSource::records() const noexcept {
    return impl_->index.records;
}
std::string_view MfqSource::architecture() const noexcept {
    return impl_->index.header.architecture;
}
const std::unordered_map<std::string, std::string>&
MfqSource::metadata() const noexcept {
    return impl_->metadata;
}
const std::vector<TensorMetadata>& MfqSource::tensors() const noexcept {
    return impl_->tensors;
}

const TensorMetadata* MfqSource::find_tensor(std::string_view name) const noexcept {
    const auto found = impl_->tensors_by_name.find(std::string(name));
    return found == impl_->tensors_by_name.end()
        ? nullptr
        : &impl_->tensors[found->second];
}

TensorLayout MfqSource::tensor_layout() const noexcept {
    return impl_->layout;
}

void MfqSource::read_range_into(
    std::string_view name,
    std::uint64_t relative_offset,
    std::byte* destination,
    std::size_t size) const {
    const auto found = impl_->index.names.find(std::string(name));
    if (found == impl_->index.names.end() ||
            impl_->index.records[found->second].asset) {
        throw std::runtime_error("model tensor not found: " + std::string(name));
    }
    const auto& record = impl_->index.records[found->second];
    if (relative_offset > record.tensor.nbytes ||
        size > record.tensor.nbytes - relative_offset) {
        throw std::out_of_range("model tensor byte range is out of bounds: " +
                                std::string(name));
    }
    if (relative_offset > std::numeric_limits<std::uint64_t>::max() - record.offset) {
        throw std::overflow_error("model tensor byte offset overflow");
    }
    impl_->files->read(record.source_path, record.offset + relative_offset,
                       destination, size);
}

ModelSource::TensorReader MfqSource::tensor_reader(std::string_view name) const {
    const auto found = impl_->index.names.find(std::string(name));
    if (found == impl_->index.names.end() || impl_->index.records[found->second].asset)
        throw std::runtime_error("model tensor not found: " + std::string(name));
    const auto& record = impl_->index.records[found->second];
    return impl_->files->reader(record.source_path, record.offset,
                                record.tensor.nbytes);
}

ModelSource::TensorReader MfqSource::mapped_tensor_reader(std::string_view name) const {
    const auto found=impl_->index.names.find(std::string(name));
    if(found==impl_->index.names.end() || impl_->index.records[found->second].asset)
        throw std::runtime_error("mapped model tensor not found: "+std::string(name));
    const auto& record=impl_->index.records[found->second];
    return tyloi::mapped_reader(record.source_path,record.offset,record.tensor.nbytes);
}

bool MfqSource::supports_parallel_tensor_reads() const noexcept {
    return impl_->files->mode() == FileReadMode::Direct;
}
ModelSource::TensorBatchReader MfqSource::mapped_tensor_batch_reader(std::string_view name) const {
    const auto found=impl_->index.names.find(std::string(name));
    if(found==impl_->index.names.end() || impl_->index.records[found->second].asset)
        throw std::runtime_error("mapped model tensor not found: "+std::string(name));
    const auto& record=impl_->index.records[found->second];
    return tyloi::mapped_batch_reader(record.source_path,record.offset,record.tensor.nbytes);
}
ModelSource::TensorBatchReader MfqSource::tensor_batch_reader(std::string_view name) const {
    const auto found=impl_->index.names.find(std::string(name));
    if(found==impl_->index.names.end() || impl_->index.records[found->second].asset)throw std::runtime_error("model tensor not found: "+std::string(name));
    const auto& record=impl_->index.records[found->second];
    return impl_->files->batch_reader(record.source_path, record.offset,
                                      record.tensor.nbytes);
}

FileReadStats MfqSource::file_read_stats() const noexcept {
    return impl_->files->stats();
}

void MfqSource::drop_file_cache() const noexcept {
    impl_->files->drop_cache();
}

const std::vector<std::string>& MfqSource::assets() const noexcept {
    return impl_->assets;
}

bool MfqSource::has_asset(std::string_view name) const noexcept {
    const auto found = impl_->index.names.find(std::string(name));
    return found != impl_->index.names.end() &&
           impl_->index.records[found->second].asset;
}

std::vector<std::byte> MfqSource::read_asset(std::string_view name) const {
    const auto found = impl_->index.names.find(std::string(name));
    if (found == impl_->index.names.end() ||
        !impl_->index.records[found->second].asset) {
        throw std::runtime_error("model asset not found: " + std::string(name));
    }
    const auto& record = impl_->index.records[found->second];
    if (record.tensor.nbytes > decision_asset_limit(name)) {
        throw std::overflow_error("model asset is too large to read");
    }
    std::vector<std::byte> result(static_cast<std::size_t>(record.tensor.nbytes));
    impl_->files->read(record.source_path, record.offset, result.data(),
                       result.size());
    return result;
}

std::optional<ModelGraph> MfqSource::model_graph() const {
    if (!has_asset(kModelGraphAsset)) return std::nullopt;
    const auto bytes = read_asset(kModelGraphAsset);
    return ModelGraph::from_json(std::string_view(
        reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

} // namespace tyloi::models
