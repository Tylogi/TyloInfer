#include "engine/state/kv_offload.h"
#include "engine/ops.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda {
namespace {
Tensor indices(const std::vector<int64_t>& values, Device device) {
    auto host = ::tyloi::ops::empty({static_cast<int64_t>(values.size())},
        TensorOptions().device(Device{DeviceType::cpu, 0}).dtype(kInt64));
    if (!values.empty()) std::memcpy(host.dense().data_ptr(), values.data(), values.size() * sizeof(int64_t));
    return host.relocated(device);
}
}

KvOffloadSequence::KvOffloadSequence(std::shared_ptr<KvOffloadStore> store, int64_t width, int64_t block_rows, ScalarType dtype)
    : store_(std::move(store)), width_(width), block_rows_(block_rows), dtype_(dtype) {
    if (!store_ || width <= 0 || block_rows <= 0)
        throw std::invalid_argument("invalid CUDA KV sequence geometry");
    const auto device = store_->config().device;
    const bool metal = device.is_metal() && device.index == 0;
    if ((!device.is_cuda() || device.index < 0) && !metal)
        throw std::invalid_argument("invalid CUDA KV sequence geometry");
    if (dtype != kFloat16 && dtype != kInt32 &&
        !(metal && (dtype == ScalarType::bfloat16 || dtype == ScalarType::float32 || dtype == ScalarType::uint32)))
        throw std::invalid_argument("invalid CUDA KV sequence geometry");
    const auto width_bytes = scalar_size(dtype);
    const auto limit = store_->config().buffer_bytes / 2;
    if (static_cast<std::uint64_t>(width) > limit / width_bytes ||
        static_cast<std::uint64_t>(block_rows) > limit / (static_cast<std::size_t>(width) * width_bytes))
        throw std::invalid_argument("invalid CUDA KV sequence geometry");
}

void KvOffloadSequence::reset() { blocks_.clear(); position_ = 0; options_ = {}; }

void KvOffloadSequence::append(const Tensor& source) {
    const auto device = store_->config().device;
    if (!(device.is_metal() ? source.device() == device : source.is_cuda()) || source.is_encoded() ||
        source.dim() != 2 || source.size(1) != width_ || source.size(0) <= 0)
        throw std::invalid_argument("invalid CUDA KV rows");
    const auto offset = position_;
    const auto previous_count = blocks_.size();
    const auto previous_tail = position_ % block_rows_ ? blocks_.back() : KvOffloadStore::BlockPtr{};
    try {
        auto rows = ::tyloi::ops::cast(source, dtype_);
        const auto options = rows.options();
        if (position_ % block_rows_) {
            auto previous = store_->read(blocks_.back());
            const std::array<Tensor, 2> parts{previous, rows};
            rows = ::tyloi::ops::cat(parts, 0);
            blocks_.pop_back();
        }
        for (int64_t begin = 0; begin < rows.size(0); begin += block_rows_)
            blocks_.push_back(store_->write(::tyloi::ops::slice(rows, 0, begin, begin + std::min(block_rows_, rows.size(0) - begin))));
        position_ += source.size(0);
        if (offset == 0) options_ = options;
    } catch (...) {
        blocks_.resize(previous_count - (previous_tail ? 1 : 0));
        if (previous_tail) blocks_.push_back(previous_tail);
        position_ = offset; throw;
    }
}

void KvOffloadSequence::truncate(int64_t position) {
    if (position < 0 || position > position_) throw std::out_of_range("CUDA KV trim exceeds its sequence");
    if (position == position_) return;
    const auto full = position / block_rows_, tail = position % block_rows_;
    KvOffloadStore::BlockPtr remainder;
    if (tail) {
        auto source = store_->read(blocks_.at(full));
        remainder = store_->write(::tyloi::ops::slice(source, 0, 0, tail));
    }
    blocks_.resize(full);
    if (remainder) blocks_.push_back(std::move(remainder));
    position_ = position;
}

Tensor KvOffloadSequence::gather(const std::vector<int64_t>& rows) const {
    if (rows.empty() || blocks_.empty()) throw std::invalid_argument("empty CUDA KV selection");
    std::map<int64_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> groups;
    for (std::size_t output = 0; output < rows.size(); ++output) {
        if (rows[output] < 0 || rows[output] >= position_) throw std::out_of_range("CUDA KV selected row exceeds its sequence");
        auto& group = groups[rows[output] / block_rows_];
        group.first.push_back(rows[output] % block_rows_); group.second.push_back(output);
    }
    auto result = ::tyloi::ops::empty({static_cast<int64_t>(rows.size()), width_}, options_);
    for (const auto& [block, group] : groups) {
        auto payload = store_->read(blocks_.at(block));
        auto selected = ::tyloi::ops::index_select(payload, 0, indices(group.first, options_.target_device()));
        ::tyloi::ops::index_copy(result, 0, indices(group.second, options_.target_device()), selected);
    }
    return result;
}

Tensor KvOffloadSequence::range(int64_t start, int64_t count) const {
    if (start < 0 || count <= 0 || start > position_ || count > position_ - start)
        throw std::out_of_range("CUDA KV range exceeds its sequence");
    auto result = ::tyloi::ops::empty({count, width_}, options_);
    for (int64_t written = 0; written < count;) {
        const auto row = start + written, block = row / block_rows_, offset = row % block_rows_;
        auto source = store_->read(blocks_.at(block));
        const auto n = std::min(count - written, source.size(0) - offset);
        result = ::tyloi::ops::update_slice(result, ::tyloi::ops::slice(source, 0, offset, offset + n), 0, written);
        written += n;
    }
    return result;
}
} // namespace tyloi::cuda
