#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::weights {

struct Segment {
    std::string tensor;
    std::uint64_t offset = 0;
    std::size_t expert_bytes = 0;
    std::vector<std::uint64_t> expert_offsets;
    std::vector<std::size_t> expert_sizes;
    std::uint64_t begin(int expert) const noexcept {
        return expert_offsets.empty() ? offset + std::uint64_t(expert) * expert_bytes : expert_offsets[expert];
    }
    std::size_t size(int expert) const noexcept {
        return expert_sizes.empty() ? expert_bytes : expert_sizes[expert];
    }
};

struct Segments {
    int experts = 0;
    std::vector<Segment> fields;
};

class SegmentLayout {
public:
    SegmentLayout(Segments& source, std::span<const std::size_t> extents, std::size_t page) {
        if (source.experts < 1 || source.fields.empty())
            throw std::invalid_argument("expert segment layer is empty");
        if (extents.size() != source.fields.size() || !page)
            throw std::invalid_argument("invalid expert segment layout geometry");
        const auto limit = std::numeric_limits<std::size_t>::max();
        const bool variable = std::any_of(source.fields.begin(), source.fields.end(),
            [](const Segment& field) { return !field.expert_offsets.empty() || !field.expert_sizes.empty(); });
        if (variable) {
            entry_sizes_.resize(source.experts);
            payload_sizes_.resize(source.experts);
        }
        for (std::size_t i = 0; i < source.fields.size(); ++i) {
            auto& field = source.fields[i];
            const auto extent = extents[i];
            if (variable) {
                const bool mapped = !field.expert_offsets.empty() || !field.expert_sizes.empty();
                if (mapped && (field.expert_offsets.size() != std::size_t(source.experts) ||
                    field.expert_sizes.size() != std::size_t(source.experts)))
                    throw std::invalid_argument("expert segment map length differs from expert count");
                if (!mapped && (!field.expert_bytes || field.offset > extent ||
                    field.expert_bytes > (extent - field.offset) / source.experts))
                    throw std::invalid_argument("expert segment lies outside its tensor");
                std::vector<std::size_t> offsets(source.experts);
                std::size_t maximum = 0;
                for (int e = 0; e < source.experts; ++e) {
                    const auto bytes = field.size(e);
                    const auto begin = field.begin(e);
                    if (!bytes || begin > extent || bytes > extent - begin || bytes > limit - page + 1)
                        throw std::invalid_argument("expert segment lies outside its tensor");
                    const auto rounded = (bytes + page - 1) / page * page;
                    if (rounded > limit - entry_sizes_[e] || bytes > limit - backing_)
                        throw std::overflow_error("expert segment cache extent is too large");
                    offsets[e] = entry_sizes_[e]; entry_sizes_[e] += rounded;
                    payload_sizes_[e] += bytes; backing_ += bytes;
                    maximum = std::max(maximum, bytes);
                }
                field_offsets_.push_back(std::move(offsets));
                if (!field.expert_sizes.empty()) field.expert_bytes = maximum;
            } else {
                if (!field.expert_bytes || field.offset > extent ||
                    field.expert_bytes > (extent - field.offset) / source.experts ||
                    field.expert_bytes > limit - page + 1)
                    throw std::invalid_argument("expert segment lies outside its tensor");
                const auto rounded = (field.expert_bytes + page - 1) / page * page;
                if (rounded > limit - entry_bytes_ ||
                    field.expert_bytes > (limit - backing_) / source.experts)
                    throw std::overflow_error("expert segment cache extent is too large");
                offsets_.push_back(entry_bytes_); entry_bytes_ += rounded;
                payload_bytes_ += field.expert_bytes;
                backing_ += field.expert_bytes * source.experts;
            }
        }
        layer_bytes_ = static_cast<long double>(entry_bytes_) * source.experts;
        if (variable) {
            entry_bytes_ = *std::max_element(entry_sizes_.begin(), entry_sizes_.end());
            payload_bytes_ = *std::max_element(payload_sizes_.begin(), payload_sizes_.end());
            layer_bytes_ = 0;
            for (const auto bytes : entry_sizes_) layer_bytes_ += bytes;
        }
    }

    std::size_t entry_size(int expert) const noexcept { return entry_sizes_.empty() ? entry_bytes_ : entry_sizes_[expert]; }
    std::size_t payload_size(int expert) const noexcept { return payload_sizes_.empty() ? payload_bytes_ : payload_sizes_[expert]; }
    std::size_t field_offset(std::size_t field, int expert) const noexcept {
        return field_offsets_.empty() ? offsets_[field] : field_offsets_[field][expert];
    }
    std::size_t max_entry() const noexcept { return entry_bytes_; }
    std::size_t backing() const noexcept { return backing_; }
    long double layer_bytes() const noexcept { return layer_bytes_; }

private:
    std::vector<std::size_t> offsets_, entry_sizes_, payload_sizes_;
    std::vector<std::vector<std::size_t>> field_offsets_;
    std::size_t entry_bytes_ = 0, payload_bytes_ = 0, backing_ = 0;
    long double layer_bytes_ = 0;
};

} // namespace tyloi::weights
