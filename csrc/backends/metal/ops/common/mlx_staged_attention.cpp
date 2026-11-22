#include "mlx_staged_attention.h"
#include <mlx/primitives.h>
#include "mlx_cached_kernel.h"
#include "mlx_platform.h"
#include "ops/gguf/native/gguf_sources.h"

#include <mlx/backend/metal/device.h>
#include <algorithm>
#include <cstdint>
#include <limits>

namespace mfq::metal {
namespace {
using namespace mlx::core;

class MlxAttentionKvRows final : public Primitive {
public:
    explicit MlxAttentionKvRows(const array& value)
        : Primitive(default_stream(Device::gpu)), copy_(contiguous(value).primitive_ptr()) {}
    const char* name() const override { return "MlxAttentionKvRows"; }
    void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        if (compatible(inputs.front())) outputs.front().copy_shared_buffer(inputs.front());
        else copy_->eval_cpu(inputs, outputs);
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        if (compatible(inputs.front())) outputs.front().copy_shared_buffer(inputs.front());
        else copy_->eval_gpu(inputs, outputs);
    }
private:
    bool compatible(const array& value) const {
        return value.strides()[3] == 1 && value.strides()[2] == 256 &&
            value.strides()[1] >= 0 && value.strides()[1] <= std::numeric_limits<int>::max();
    }
    std::shared_ptr<Primitive> copy_;
};

array attention_kv_rows(const array& value) {
    return array(value.shape(), value.dtype(), std::make_shared<MlxAttentionKvRows>(value), {value});
}

const fast::CustomKernelFunction& attention_pack_kv() {
    static const auto kernel = cached_metal_kernel("tyloinfer_gqa_pack_tail_kv_v1",
        {"keys", "values", "params"}, {"packed_keys", "packed_values"}, R"METAL(
const uint index = thread_position_in_grid.x, head = index / (32 * 256);
const uint flat = index % (32 * 256), dim = flat % 256;
const uint token = (params[0] - 1) * 32 + flat / 256;
Input k = Input(0), v = Input(0);
if (token < params[1]) {
    k = keys[ulong(head) * keys_strides[1] + token * keys_strides[2] + dim * keys_strides[3]];
    v = values[ulong(head) * values_strides[1] + token * values_strides[2] + dim * values_strides[3]];
}
packed_keys[index] = k; packed_values[index] = v;
)METAL", "", false, false, CompileOptions(static_cast<int>(MathMode::Fast)));
    return kernel;
}

const fast::CustomKernelFunction& attention_partial() {
    static const auto kernel = cached_metal_kernel("tyloinfer_gqa_staged_partial_v2",
        {"queries", "keys", "values", "tail_keys", "tail_values", "params"}, {"partials", "statistics"}, R"METAL(
constexpr uint M = 8 * GROUP, N = 32;
threadgroup float scores[M * N], row_max[M], row_sum[M], previous_scale[M];
threadgroup bfloat probabilities[M * N];
threadgroup atomic_uint rescale;
const uint head = threadgroup_position_in_grid.x, split = threadgroup_position_in_grid.y;
splash_paged_attention_tile<KH, GROUP, 8, 256, 0, Input>(
    const_cast<device Input*>(queries) + head * M * 256,
    const_cast<device Input*>(keys), const_cast<device Input*>(values),
    const_cast<device Input*>(tail_keys), const_cast<device Input*>(tail_values),
    keys_strides[1], values_strides[1],
    head, params[0], params[1], params[2], split, partials, statistics, head * params[2] + split,
    scores, probabilities, row_max, row_sum, previous_scale, &rescale, thread_index_in_threadgroup);
)METAL", native_gguf::staged_attention_source, false, false, CompileOptions(static_cast<int>(MathMode::Fast)));
    return kernel;
}

const fast::CustomKernelFunction& attention_reduce() {
    static const auto kernel = cached_metal_kernel("tyloinfer_gqa_staged_reduce_v1",
        {"partials", "statistics", "params"}, {"output"}, R"METAL(
threadgroup float weights[128], group_values[8];
constexpr uint M = 8 * GROUP;
const uint task = threadgroup_position_in_grid.x, head = task / M, row = task % M;
const uint pages = (params[0] + params[1] + 31) / 32;
const uint per_split = (pages + params[2] - 1) / params[2];
const uint written = (pages + per_split - 1) / per_split;
splash_attention_combine_splits<GROUP, 8, 256>(partials, statistics, output + head * M * 256,
    params[1], written, head * params[2], row, thread_index_in_threadgroup, weights, group_values);
)METAL", native_gguf::staged_attention_source, true, false, CompileOptions(static_cast<int>(MathMode::Fast)));
    return kernel;
}
}

std::optional<mlx::core::array> mlx_staged_attention(
    const mlx::core::array& query, const mlx::core::array& key,
    const mlx::core::array& value, bool causal, float scale,
    const std::optional<mlx::core::array>& mask) {
    using namespace mlx::core;
    if (!causal || mask || query.ndim() != 4 || key.ndim() != 4 || value.shape() != key.shape() ||
        query.shape(0) != 1 || key.shape(0) != 1 || query.shape(3) != 256 || key.shape(3) != 256 || key.shape(1) < 1 ||
        query.shape(1) % key.shape(1) || query.shape(2) < 1 || query.shape(2) > 8 || scale != 1.f / 16.f ||
        (query.dtype() != float16 && query.dtype() != bfloat16) || key.dtype() != query.dtype() || value.dtype() != query.dtype())
        return std::nullopt;
    const int rows = query.shape(2), heads = key.shape(1), group = query.shape(1) / heads, live = key.shape(2);
    if ((group != 6 && group != 8) || rows * group <= 32 || live < rows || mlx_metal_nax_disabled() ||
        !mlx::core::metal::device(Device::gpu).mtl_device()->supportsFamily(MTL::GPUFamilyApple10)) return std::nullopt;
    if (live > std::numeric_limits<int>::max() - 31) return std::nullopt;
    const int capacity = (live + 31) / 32 * 32, pages = capacity / 32;
    if (capacity > std::numeric_limits<int>::max() / 256 ||
        std::uint64_t(heads) * pages * 8 * 256 > std::uint64_t(std::numeric_limits<int>::max())) return std::nullopt;
    auto row_keys = attention_kv_rows(key), row_values = attention_kv_rows(value);
    auto staged_query = transpose(reshape(query, {heads, group, rows, 256}), {0, 2, 1, 3});
    if (rows < 8) staged_query = pad(staged_query, {{0, 0}, {0, 8 - rows}, {0, 0}, {0, 0}});
    else staged_query = contiguous(staged_query);
    auto packed = attention_pack_kv()({row_keys, row_values, array({std::uint32_t(pages), std::uint32_t(live)})},
        {{heads, 32, 256}, {heads, 32, 256}}, {query.dtype(), query.dtype()},
        {heads * 32 * 256, 1, 1}, {256, 1, 1}, {{"Input", query.dtype()}}, std::nullopt, false, {});
    const int splits = std::min(128, std::max(32, (pages + 15) / 16));
    auto params = array({std::uint32_t(live - rows), std::uint32_t(rows), std::uint32_t(splits)});
    auto partial = attention_partial()({staged_query, row_keys, row_values, packed[0], packed[1], params},
        {{heads * splits, 8 * group, 256}, {heads * splits, 8 * group, 2}}, {float32, float32},
        {heads * 256, splits, 1}, {256, 1, 1}, {{"Input", query.dtype()}, {"KH", heads}, {"GROUP", group}}, std::nullopt, false, {});
    auto result = attention_reduce()({partial[0], partial[1], params}, {{heads, 8, group, 256}}, {query.dtype()},
        {heads * 8 * group * 256, 1, 1}, {256, 1, 1}, {{"GROUP", group}}, std::nullopt, false, {})[0];
    return reshape(transpose(slice(result, {0, 0, 0, 0}, {heads, rows, group, 256}), {0, 2, 1, 3}), query.shape());
}

mlx::core::array mlx_attention_relative_bias(const mlx::core::array& table, const mlx::core::array& sink_logits,
    const mlx::core::array& key_positions, int first_row, int count,
    int first_position, int window, mlx::core::Dtype dtype) {
    using namespace mlx::core;
    if (table.ndim() != 4 || table.shape(-1) <= 0 || key_positions.ndim() != 1 ||
        sink_logits.shape() != Shape{table.shape(0), table.shape(1), table.shape(2)} ||
        key_positions.dtype() != int32 || !key_positions.size() || first_row < 0 || count <= 0 ||
        count > table.shape(1) - first_row || first_position < 0 || window < 0)
        throw std::invalid_argument("invalid relative attention bias geometry");
    const auto rows = std::size_t(table.shape(0)) * table.shape(2) * count;
    if (!rows || rows > std::numeric_limits<int>::max() / key_positions.size())
        throw std::invalid_argument("relative attention bias grid is too large");
    static const auto kernel = cached_metal_kernel("tyloinfer_relative_attention_bias_with_sink",
        {"table", "sinks", "positions", "params"}, {"bias"}, R"METAL(
        const uint i = thread_position_in_grid.x;
        const uint keys = uint(positions_shape[0]);
        const uint rows = uint(params[1]);
        const uint row = i / keys % rows;
        const uint head = i / (keys * rows);
        const int distance = params[2] + int(row) - positions[i % keys];
        if (distance < 0 || (params[3] != 0 && distance >= params[3])) {
            bias[i] = T(-INFINITY);
            return;
        }
        const uint heads = uint(table_shape[2]);
        const uint extent = uint(table_shape[3]);
        const ulong index = (ulong(head / heads) * uint(table_shape[1]) + uint(params[0]) + row) * heads + head % heads;
        const float relative = uint(distance) < extent ? float(table[index * extent + uint(distance)]) : 0.f;
        bias[i] = T(relative - float(sinks[index]));
        )METAL", "", true, false, CompileOptions(static_cast<int>(MathMode::Safe)));
    const int grid = static_cast<int>(rows * key_positions.size());
    const Shape shape{table.shape(0), table.shape(2), count, key_positions.shape(0)};
    return kernel({table, sink_logits, key_positions, array({first_row, count, first_position, window}, int32)},
        {shape}, {dtype}, {grid, 1, 1}, {std::min(256, grid), 1, 1}, {{"T", dtype}},
        std::nullopt, false, {}).front();
}
}
