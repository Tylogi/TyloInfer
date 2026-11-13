#include "engine/backends/metal/ops/attention/kv.h"
#include "engine/backends/metal/ops/attention/attention.h"
#include "engine/backends/metal/native/tensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace mfq::metal {
namespace {
using mlx::core::array;
using mlx::core::Shape;
using Kernel = mlx::core::fast::CustomKernelFunction;

array floating(const array& input, mlx::core::Dtype dtype) {
    auto result = input;
    if (result.dtype() != dtype) result = mlx::core::astype(result, dtype);
    return mlx::core::contiguous(result);
}

int rotation_width(int dimension) {
    if (dimension <= 0 || dimension > 1024) throw std::invalid_argument("unsupported TurboQuant KV head dimension");
    int width = 1;
    while (width < dimension) width *= 2;
    return width;
}

struct Tables { array centers; array signs; };
Tables tables(int width, int bits, bool value) {
    static std::mutex mutex;
    static std::map<std::tuple<int, int, bool>, Tables> cached;
    std::lock_guard<std::mutex> guard(mutex);
    const auto key = std::make_tuple(width, bits, value);
    if (auto found = cached.find(key); found != cached.end()) return found->second;
    constexpr int grid = 32768;
    const int levels = 1 << bits;
    std::vector<double> points(grid), weights(grid), cumulative(grid);
    double total = 0;
    for (int i = 0; i < grid; ++i) {
        const double x = -1.0 + 1e-6 + (2.0 - 2e-6) * i / (grid - 1);
        points[i] = x;
        weights[i] = std::pow(1.0 - x * x, (width - 3) / 2.0);
        cumulative[i] = total += weights[i];
    }
    std::vector<float> centers(levels);
    for (int i = 0; i < levels; ++i) {
        const auto index = std::lower_bound(cumulative.begin(), cumulative.end(), total * (i + 0.5) / levels);
        centers[i] = static_cast<float>(points[std::min(grid - 1, static_cast<int>(index - cumulative.begin()))]);
    }
    for (int iteration = 0; iteration < 100; ++iteration) {
        std::vector<double> sums(levels, 0), masses(levels, 0);
        int cell = 0;
        for (int i = 0; i < grid; ++i) {
            while (cell + 1 < levels && points[i] > (centers[cell] + centers[cell + 1]) * 0.5) ++cell;
            sums[cell] += points[i] * weights[i]; masses[cell] += weights[i];
        }
        double change = 0;
        for (int i = 0; i < levels; ++i) if (masses[i] > 0) {
            const float next = static_cast<float>(sums[i] / masses[i]);
            change = std::max(change, std::abs(static_cast<double>(next - centers[i])));
            centers[i] = next;
        }
        if (change < 1e-6) break;
    }
    std::vector<float> signs(width);
    std::uint32_t state = 0x6d2b79f5u ^ static_cast<std::uint32_t>(width * 7919 + (value ? 1 : 0));
    for (auto& sign : signs) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        sign = state & 1 ? 1.0f : -1.0f;
    }
    Tables result{array(centers.begin(), Shape{levels}), array(signs.begin(), Shape{width})};
    mlx::core::eval(result.centers, result.signs);
    cached.emplace(key, result);
    return result;
}

const Kernel& encoder() {
    static const auto kernel = mlx::core::fast::metal_kernel("mfq_turboquant_kv_encode_v1",
        {"input", "centers", "signs"}, {"out"}, R"(
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        threadgroup float work[W];
        threadgroup float norms[W];
        threadgroup uint codes[W];
        float x = lane < D ? float(input[row * D + lane]) : 0.0f;
        work[lane] = x * signs[lane]; norms[lane] = x * x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = W / 2; stride; stride /= 2) {
            if (lane < stride) norms[lane] += norms[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float norm = sqrt(norms[0]);
        for (uint stride = 1; stride < W; stride *= 2) {
            float a = work[lane & ~stride]; float b = work[lane | stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            work[lane] = (lane & stride) ? a - b : a + b;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float rotated = norm > 0 ? work[lane] / (norm * sqrt(float(W))) : 0.0f;
        uint lo = 0, hi = (1u << B) - 1;
        while (lo < hi) {
            uint mid = (lo + hi) / 2;
            if (rotated > (centers[mid] + centers[mid + 1]) * 0.5f) lo = mid + 1;
            else hi = mid;
        }
        codes[lane] = lo;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) out[row * P] = as_type<uint>(norm);
        for (uint word = lane; word < P - 1; word += W) {
            uint packed = 0;
            for (uint bit = 0; bit < 32; ++bit) {
                uint offset = word * 32 + bit;
                if (offset < W * B) packed |= ((codes[offset / B] >> (offset % B)) & 1u) << bit;
            }
            out[row * P + 1 + word] = packed;
        }
        )", "", true);
    return kernel;
}

const Kernel& decoder() {
    static const auto kernel = mlx::core::fast::metal_kernel("mfq_turboquant_kv_decode_v1",
        {"input", "centers", "signs"}, {"out"}, R"(
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        threadgroup float work[W];
        uint offset = lane * B;
        uint code = input[row * P + 1 + offset / 32] >> (offset % 32);
        if (offset % 32 + B > 32) code |= input[row * P + 2 + offset / 32] << (32 - offset % 32);
        work[lane] = centers[code & ((1u << B) - 1)];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1; stride < W; stride *= 2) {
            float a = work[lane & ~stride]; float b = work[lane | stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            work[lane] = (lane & stride) ? a - b : a + b;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane < D) out[row * D + lane] = O(work[lane] * signs[lane] *
            as_type<float>(input[row * P]) / sqrt(float(W)));
        )", "", true);
    return kernel;
}

const Kernel& compressed_attention() {
    static const auto kernel = mlx::core::fast::metal_kernel("mfq_turboquant_sparse_qk_av_v1",
        {"query", "keys", "values", "blocks", "key_centers", "value_centers", "signs", "params", "mask"},
        {"partial", "stats"}, R"(
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
        uint part = threadgroup_position_in_grid.x, group = threadgroup_position_in_grid.y;
        uint batch_token = threadgroup_position_in_grid.z;
        int tokens = params[0], heads = params[1], kv_heads = params[2];
        int position = params[3], offset = params[4], ratio = params[5], budget = params[6];
        int count = params[7], parts = params[8], span = params[9];
        int slots = params[10];
        int batch = batch_token / tokens, token = batch_token % tokens;
        int visible = offset + token + 1, gqa = heads / kv_heads;
        int head_groups = (gqa + SG - 1) / SG, kv_head = group / head_groups;
        int local_head = (group % head_groups) * SG + sg, head = kv_head * gqa + local_head;
        bool active = local_head < gqa;
        ulong qrow = (ulong(batch) * heads + head) * tokens + token;
        constexpr uint TILE = W <= 256 ? 8 : (W <= 512 ? 4 : 2);
        threadgroup float kc[1 << BK], vc[1 << BV], ktile[TILE * W], vtile[TILE * W];
        threadgroup int ids[TILE];
        for (uint i = tid; i < (1u << BK); i += SG * 32) kc[i] = key_centers[i];
        for (uint i = tid; i < (1u << BV); i += SG * 32) vc[i] = value_centers[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q[(W + 31) / 32], acc[(W + 31) / 32];
        float maximum = -INFINITY, mass = 0.0f;
        for (uint i = 0; i < (W + 31) / 32; ++i)
            q[i] = active && lane + i * 32 < D ? float(query[qrow * D + lane + i * 32]) * signs[lane + i * 32] : 0.0f;
        for (uint stride = 1; stride < min(uint(W), 32u); stride *= 2)
            for (uint i = 0; i < (W + 31) / 32; ++i) {
                float other = simd_shuffle_xor(q[i], ushort(stride));
                q[i] = lane & stride ? other - q[i] : q[i] + other;
            }
        for (uint stride = 1; stride < W / 32; stride *= 2) {
            float next[(W + 31) / 32];
            for (uint i = 0; i < (W + 31) / 32; ++i) {
                float a = q[i & ~stride], b = q[i | stride];
                next[i] = i & stride ? a - b : a + b;
            }
            for (uint i = 0; i < (W + 31) / 32; ++i) q[i] = next[i];
        }
        for (uint i = 0; i < (W + 31) / 32; ++i) q[i] /= sqrt(float(W));
        for (uint i = 0; i < (W + 31) / 32; ++i) acc[i] = 0.0f;
        for (int start = part * span; start < min((int(part) + 1) * span, slots); start += TILE) {
            for (uint at = tid; at < TILE * W; at += SG * 32) {
                int slot = start + at / W, id = -1;
                uint d = at % W;
                if (slot < slots && slot < (int(part) + 1) * span) {
                    if (visible <= budget) id = slot < visible ? slot : -1;
                    else if (slot < count * ratio) {
                        int block = blocks[(batch * tokens + token) * count + slot / ratio];
                        if (block >= 0 && block < visible / ratio) id = block * ratio + slot % ratio;
                    } else id = visible / ratio * ratio + slot - count * ratio;
                }
                bool valid = id >= 0 && id < visible && id < position;
                if (valid && params[11]) valid = mask[(ulong(batch) * tokens + token) * position + id];
                if (d == 0) ids[at / W] = valid ? id : -1;
                float k = 0.0f, v = 0.0f;
                if (valid) {
                    device const uint* kr = keys + ulong(batch) * keys_strides[0] + ulong(kv_head) * keys_strides[1] + ulong(id) * keys_strides[2];
                    device const uint* vr = values + ulong(batch) * values_strides[0] + ulong(kv_head) * values_strides[1] + ulong(id) * values_strides[2];
                    k = as_type<float>(kr[0]) * qsa_tq_code<BK>(kr, d, kc, keys_strides[3]);
                    v = as_type<float>(vr[0]) * qsa_tq_code<BV>(vr, d, vc, values_strides[3]);
                }
                ktile[at] = k; vtile[at] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float scores[TILE], weights[TILE];
            for (uint j = 0; j < TILE; ++j) {
                float score = 0.0f;
                for (uint i = 0; i < (W + 31) / 32; ++i)
                    if (lane + i * 32 < W) score += q[i] * ktile[j * W + lane + i * 32];
                scores[j] = ids[j] >= 0 ? simd_sum(score) / sqrt(float(D)) : -INFINITY;
                weights[j] = 0.0f;
            }
            float factor = 0.0f;
            if (lane == 0) {
                float next = maximum;
                for (uint j = 0; j < TILE; ++j) next = max(next, scores[j]);
                factor = isfinite(maximum) ? exp(maximum - next) : 0.0f;
                mass *= factor;
                for (uint j = 0; j < TILE; ++j) {
                    weights[j] = isfinite(scores[j]) ? exp(scores[j] - next) : 0.0f;
                    mass += weights[j];
                }
                maximum = next;
            }
            factor = simd_broadcast(factor, 0);
            for (uint j = 0; j < TILE; ++j) weights[j] = simd_broadcast(weights[j], 0);
            for (uint i = 0; i < (W + 31) / 32; ++i) {
                uint d = lane + i * 32;
                acc[i] *= factor;
                if (d < W) for (uint j = 0; j < TILE; ++j) acc[i] += weights[j] * vtile[j * W + d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        ulong result = qrow * parts + part;
        if (active) {
            for (uint i = 0; i < (W + 31) / 32; ++i)
                if (lane + i * 32 < W) partial[result * W + lane + i * 32] = acc[i];
            if (lane == 0) { stats[result * 2] = maximum; stats[result * 2 + 1] = mass; }
        }
        )", R"(
        template <uint Bits>
        inline float qsa_tq_code(device const uint* row, uint dimension, threadgroup const float* centers, ulong stride) {
            uint offset = dimension * Bits;
            uint code = row[(1 + offset / 32) * stride] >> (offset % 32);
            if constexpr (32 % Bits != 0)
                if (offset % 32 + Bits > 32) code |= row[(2 + offset / 32) * stride] << (32 - offset % 32);
            return centers[code & ((1u << Bits) - 1)];
        }
        )", false);
    return kernel;
}

const Kernel& attention_output() {
    static const auto kernel = mlx::core::fast::metal_kernel("mfq_turboquant_sparse_reduce_inverse_v1",
        {"partial", "stats", "signs", "params"}, {"out"}, R"(
        uint lane = thread_position_in_threadgroup.x, row = threadgroup_position_in_grid.x;
        uint parts = params[8];
        threadgroup float scales[32], norm, work[W];
        if (lane < 32) {
            float maximum = -INFINITY;
            for (uint i = lane; i < parts; i += min(uint(W), 32u))
                maximum = max(maximum, stats[(ulong(row) * parts + i) * 2]);
            float top = simd_max(maximum);
            float mass = 0.0f;
            for (uint i = lane; i < parts; i += min(uint(W), 32u)) {
                ulong at = (ulong(row) * parts + i) * 2;
                float scale = isfinite(stats[at]) ? metal::fast::exp(stats[at] - top) : 0.0f;
                scales[i] = scale; mass += scale * stats[at + 1];
            }
            float total = simd_sum(mass);
            if (lane == 0) norm = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float result = 0.0f;
        for (uint i = 0; i < parts; ++i) result += scales[i] * partial[(ulong(row) * parts + i) * W + lane];
        float x = norm > 0 ? result / norm : 0.0f;
        for (uint stride = 1; stride < min(uint(W), 32u); stride *= 2) {
            float other = simd_shuffle_xor(x, ushort(stride));
            x = lane & stride ? other - x : x + other;
        }
        work[lane] = x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 32; stride < W; stride *= 2) {
            float a = work[lane & ~stride], b = work[lane | stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            work[lane] = (lane & stride) ? a - b : a + b;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane < D) out[row * D + lane] = O(work[lane] * signs[lane] / sqrt(float(W)));
        )", "", true);
    return kernel;
}
}

std::pair<array, array> mlx_kv_append(const tyloi::state::KvView& kv,
    const array& key, const array& value, int position) {
    const auto& key_storage = tyloi::metal::array(kv.physical_key);
    const auto& value_storage = tyloi::metal::array(kv.physical_value);
    const Shape start{0, 0, position, 0};
    const Shape stop{static_cast<int>(kv.batch), static_cast<int>(kv.heads),
        position + key.shape(2), key_storage.shape(3)};
    auto next_key = mlx::core::slice_update(key_storage,
        kv.quantization.enabled() ? mlx_kv_encode(floating(key, tyloi::metal::dtype(kv.dtype)), kv.quantization.key_bits(), false)
                                  : floating(key, tyloi::metal::dtype(kv.dtype)), start, stop);
    auto next_value = mlx::core::slice_update(value_storage,
        kv.quantization.enabled() ? mlx_kv_encode(floating(value, tyloi::metal::dtype(kv.dtype)), kv.quantization.value_bits(), true)
                                  : floating(value, tyloi::metal::dtype(kv.dtype)), start,
        Shape{static_cast<int>(kv.batch), static_cast<int>(kv.heads), position + key.shape(2), value_storage.shape(3)});
    return {std::move(next_key), std::move(next_value)};
}

std::pair<array, array> mlx_kv_view(const tyloi::state::KvView& kv, int position) {
    const auto& key = tyloi::metal::array(kv.physical_key);
    const auto& value = tyloi::metal::array(kv.physical_value);
    const Shape start{0, 0, 0, 0};
    auto result = std::pair{
        mlx::core::slice(key, start, Shape{static_cast<int>(kv.batch), static_cast<int>(kv.heads), position, key.shape(3)}),
        mlx::core::slice(value, start, Shape{static_cast<int>(kv.batch), static_cast<int>(kv.heads), position, value.shape(3)})};
    if (kv.quantization.enabled())
        return {mlx_kv_decode(result.first, static_cast<int>(kv.head_dim), kv.quantization.key_bits(), false, tyloi::metal::dtype(kv.dtype)),
                mlx_kv_decode(result.second, static_cast<int>(kv.head_dim), kv.quantization.value_bits(), true, tyloi::metal::dtype(kv.dtype))};
    return result;
}

array mlx_kv_attention(const array& query, const tyloi::state::KvView& kv, int position, bool causal) {
    const auto& key_storage = tyloi::metal::array(kv.physical_key);
    const auto& value_storage = tyloi::metal::array(kv.physical_value);
    if (!kv.quantization.enabled()) {
        const auto cached = mlx_kv_view(kv, position);
        return scaled_dot_product_attention(query, cached.first, cached.second, causal);
    }
    if (query.ndim() != 4 || query.shape(0) != static_cast<int>(kv.batch) || query.shape(1) % static_cast<int>(kv.heads) ||
        query.shape(3) != static_cast<int>(kv.head_dim) || query.shape(2) > position)
        throw std::invalid_argument("quantized dense KV attention topology mismatch");
    const int tokens = query.shape(2), groups = query.shape(1) / static_cast<int>(kv.heads), offset = position - tokens;
    std::vector<array> outputs;
    for (int begin = 0; begin < tokens; begin += 32) {
        const int count = std::min(32, tokens - begin);
        auto q = mlx::core::reshape(mlx::core::astype(mlx::core::slice(query, Shape{0, 0, begin, 0},
            Shape{static_cast<int>(kv.batch), query.shape(1), begin + count, static_cast<int>(kv.head_dim)}), mlx::core::float32),
            Shape{static_cast<int>(kv.batch), static_cast<int>(kv.heads), groups, count, static_cast<int>(kv.head_dim)});
        std::optional<array> maximum, norm, weighted;
        const int end = causal ? offset + begin + count : position;
        for (int start = 0; start < end; start += 1024) {
            const int rows = std::min(1024, end - start);
            auto decode = [&](const array& source, bool value) {
                auto packed = mlx::core::slice(source, Shape{0, 0, start, 0}, Shape{static_cast<int>(kv.batch), static_cast<int>(kv.heads), start + rows, source.shape(3)});
                return mlx::core::expand_dims(mlx_kv_decode(packed, static_cast<int>(kv.head_dim),
                    value ? kv.quantization.value_bits() : kv.quantization.key_bits(), value, mlx::core::float32), 2);
            };
            auto scores = mlx::core::matmul(q, mlx::core::swapaxes(decode(key_storage, false), -1, -2)) /
                std::sqrt(static_cast<float>(static_cast<int>(kv.head_dim)));
            if (causal) {
                auto visible = mlx::core::reshape(mlx::core::arange(offset + begin + 1, offset + begin + count + 1, mlx::core::int32),
                    Shape{1, 1, 1, count, 1});
                scores = mlx::core::where(mlx::core::less(mlx::core::arange(start, start + rows, mlx::core::int32), visible),
                    scores, array(-std::numeric_limits<float>::infinity()));
            }
            auto local_max = mlx::core::max(scores, -1, true);
            auto safe_max = mlx::core::where(mlx::core::isfinite(local_max), local_max, array(0.0f));
            auto probabilities = mlx::core::exp(scores - safe_max);
            auto local_norm = mlx::core::sum(probabilities, -1, true);
            auto local_weighted = mlx::core::matmul(probabilities, decode(value_storage, true));
            if (maximum) {
                auto next_max = mlx::core::maximum(*maximum, local_max);
                auto current_scale = mlx::core::exp(safe_max - next_max);
                auto previous_scale = mlx::core::exp(*maximum - next_max);
                norm = *norm * previous_scale + local_norm * current_scale;
                weighted = *weighted * previous_scale + local_weighted * current_scale;
                maximum = std::move(next_max);
            } else {
                maximum = std::move(local_max); norm = std::move(local_norm); weighted = std::move(local_weighted);
            }
            mlx::core::eval(*maximum, *norm, *weighted);
        }
        auto output = mlx::core::reshape(mlx::core::astype(*weighted / *norm, query.dtype()),
            Shape{static_cast<int>(kv.batch), query.shape(1), count, static_cast<int>(kv.head_dim)});
        output.eval(); outputs.push_back(std::move(output));
    }
    return outputs.size() == 1 ? outputs.front() : mlx::core::concatenate(outputs, 2);
}


int mlx_kv_packed_width(int dimension, int bits) {
    if (bits < 2 || bits > 8) throw std::invalid_argument("invalid TurboQuant component bits");
    return 1 + (rotation_width(dimension) * bits + 31) / 32;
}
array mlx_kv_encode(const array& input, int bits, bool value) {
    const int dimension = input.shape(-1), width = rotation_width(dimension);
    const int packed = mlx_kv_packed_width(dimension, bits);
    const auto rows = input.size() / dimension;
    if (rows > std::numeric_limits<int>::max() / width) throw std::invalid_argument("TurboQuant KV grid overflow");
    auto shape = input.shape(); shape.back() = packed;
    if (!rows) return mlx::core::zeros(shape, mlx::core::uint32);
    const auto table = tables(width, bits, value);
    return encoder()({mlx::core::astype(input, mlx::core::float32), table.centers, table.signs},
        {shape}, {mlx::core::uint32}, {static_cast<int>(rows) * width, 1, 1}, {width, 1, 1},
        {{"D", dimension}, {"W", width}, {"B", bits}, {"P", packed}}, std::nullopt, false, {}).at(0);
}
array mlx_kv_decode(const array& input, int dimension, int bits, bool value, mlx::core::Dtype dtype) {
    const int width = rotation_width(dimension), packed = mlx_kv_packed_width(dimension, bits);
    if (input.dtype() != mlx::core::uint32 || input.shape(-1) != packed)
        throw std::invalid_argument("TurboQuant KV packed topology mismatch");
    const auto rows = input.size() / packed;
    if (rows > std::numeric_limits<int>::max() / width) throw std::invalid_argument("TurboQuant KV grid overflow");
    auto shape = input.shape(); shape.back() = dimension;
    if (!rows) return mlx::core::zeros(shape, dtype);
    const auto table = tables(width, bits, value);
    return decoder()({input, table.centers, table.signs}, {shape}, {dtype},
        {static_cast<int>(rows) * width, 1, 1}, {width, 1, 1},
        {{"D", dimension}, {"W", width}, {"B", bits}, {"P", packed}, {"O", dtype}}, std::nullopt, false, {}).at(0);
}

array mlx_kv_sparse_attention(const array& query, const array& key, const array& value,
    const std::optional<array>& blocks, tyloi::state::KvQuantization quantization,
    int position, int query_offset, int ratio, int budget, const std::optional<array>& key_mask) {
    if (!quantization.enabled() || !tyloi::state::kv_bits_valid(quantization.bits) || query.ndim() != 4 ||
        key.ndim() != 4 || value.ndim() != 4 || key.dtype() != mlx::core::uint32 || value.dtype() != mlx::core::uint32 ||
        query.shape(0) != key.shape(0) || key.shape(0) != value.shape(0) || key.shape(1) != value.shape(1) ||
        query.shape(0) <= 0 || query.shape(1) <= 0 || key.shape(1) <= 0 || query.shape(1) % key.shape(1) ||
        (query.dtype() != mlx::core::float16 && query.dtype() != mlx::core::bfloat16 && query.dtype() != mlx::core::float32) ||
        position <= 0 || position > key.shape(2) || position > value.shape(2) ||
        query_offset < 0 || query_offset > position - query.shape(2) || ratio <= 0 || budget <= 0 ||
        (position > budget && budget % ratio))
        throw std::invalid_argument("compressed QSA topology mismatch");
    const int batch = query.shape(0), heads = query.shape(1), tokens = query.shape(2);
    const int dimension = query.shape(3), width = rotation_width(dimension);
    const int kb = quantization.key_bits(), vb = quantization.value_bits();
    const int pk = mlx_kv_packed_width(dimension, kb), pv = mlx_kv_packed_width(dimension, vb);
    if (key.shape(3) != pk || value.shape(3) != pv || tokens <= 0 ||
        (blocks && blocks->shape() != Shape{batch, tokens, budget / ratio}) ||
        (position > budget && !blocks) ||
        (key_mask && key_mask->shape() != Shape{batch, tokens, position}))
        throw std::invalid_argument("compressed QSA selection mismatch");
    const std::int64_t rows = std::int64_t(batch) * heads * tokens;
    if (rows > std::numeric_limits<int>::max() / width)
        throw std::invalid_argument("compressed QSA grid overflow");
    const int slots = position > budget ? budget + ratio - 1 : position;
    const int parts = tokens > 8 ? 1 : std::min(32, (slots + 63) / 64);
    const int span = ((slots + parts - 1) / parts + 7) / 8 * 8;
    const int gqa = heads / key.shape(1), simdgroups = std::min(16, gqa), threads = simdgroups * 32;
    const array params({tokens, heads, key.shape(1), position, query_offset, ratio, budget,
        blocks ? blocks->shape(2) : 0, parts, span, slots, key_mask ? 1 : 0});
    const auto kt = tables(width, kb, false), vt = tables(width, vb, true);
    auto results = compressed_attention()({mlx::core::contiguous(query), key, value,
        blocks ? mlx::core::contiguous(mlx::core::astype(*blocks, mlx::core::int32)) : array({0}, Shape{1}),
        kt.centers, vt.centers, kt.signs, params,
        key_mask ? mlx::core::contiguous(mlx::core::astype(*key_mask, mlx::core::bool_)) : array({true}, Shape{1})},
        {Shape{batch, heads, tokens, parts, width}, Shape{batch, heads, tokens, parts, 2}},
        {mlx::core::float32, mlx::core::float32},
        {parts * threads, key.shape(1) * ((gqa + simdgroups - 1) / simdgroups), batch * tokens}, {threads, 1, 1},
        {{"D", dimension}, {"W", width}, {"BK", kb}, {"BV", vb}, {"SG", simdgroups}}, std::nullopt, false, {});
    auto output = attention_output()({results[0], results[1], vt.signs, params}, {query.shape()}, {query.dtype()},
        {static_cast<int>(rows) * width, 1, 1}, {width, 1, 1},
        {{"D", dimension}, {"W", width}, {"O", query.dtype()}}, std::nullopt, false, {}).at(0);
    return mlx::core::transpose(output, {0, 2, 1, 3});
}
}
