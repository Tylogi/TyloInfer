#include "models/lfm3/engine.h"
#include "models/lfm3/layer_weights.h"
#include "models/lfm3/request.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "backends/cuda/kernels/gguf.h"
#include "ggml-cuda/common.cuh"
#include "backends/cuda/kernels/gguf_decode.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace mfq::models::lfm3::d1 {
namespace {
using B = __nv_bfloat16;
using Clock = std::chrono::steady_clock;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void blas(cublasStatus_t e) { if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("d1 cuBLAS failure: " + std::to_string(int(e))); }
struct Tensor {
    std::shared_ptr<void> owner;
    B* data = nullptr;
    std::size_t count = 0, storage_bytes = 0;
    bool packed = false, column_planes = false, host = false;
    std::string label;
    int weight_rows = 0, weight_columns = 0;
    Tensor() = default;
    static Tensor host_weight(std::size_t elements, std::size_t bytes, bool quantized) {
        Tensor result;
        auto* p = new std::uint8_t[bytes];
        result.owner = std::shared_ptr<void>(p, [](void* x) { delete[] static_cast<std::uint8_t*>(x); });
        result.data = reinterpret_cast<B*>(p); result.count = elements;
        result.storage_bytes = bytes; result.packed = quantized; result.host = true;
        return result;
    }
    static Tensor packed_weight(std::size_t elements, std::size_t bytes) {
        Tensor result;
        void* p = nullptr; check(cudaMalloc(&p, bytes));
        result.owner = std::shared_ptr<void>(p, [](void* x) { cudaFree(x); });
        result.data = static_cast<B*>(p); result.count = elements;
        result.storage_bytes = bytes; result.packed = true;
        return result;
    }
    explicit Tensor(std::size_t n, bool asynchronous = true) : count(n), storage_bytes(n * sizeof(B)) {
        if (!n || n > SIZE_MAX / sizeof(B)) throw std::invalid_argument("invalid d1 allocation size");
        void* p = nullptr;
        bool pooled = asynchronous;
        auto error = asynchronous ? cudaMallocAsync(&p, n * sizeof(B), nullptr) : cudaMalloc(&p, n * sizeof(B));
        if (error == cudaErrorNotSupported) { cudaGetLastError(); pooled = false; error = cudaMalloc(&p, n * sizeof(B)); }
        check(error);
        owner = std::shared_ptr<void>(p, [pooled](void* x) { if (pooled) cudaFreeAsync(x, nullptr); else cudaFree(x); });
        data = static_cast<B*>(p);
    }
};
struct Raw {
    void* p = nullptr;
    bool pooled = true;
    explicit Raw(std::size_t bytes) {
        auto error = cudaMallocAsync(&p, bytes, nullptr);
        if (error == cudaErrorNotSupported) { cudaGetLastError(); pooled = false; error = cudaMalloc(&p, bytes); }
        check(error);
    }
    ~Raw() { if (p) { if (pooled) cudaFreeAsync(p, nullptr); else cudaFree(p); } }
    Raw(const Raw&) = delete;
    Raw& operator=(const Raw&) = delete;
};
struct Pinned {
    std::shared_ptr<void> owner;
    std::size_t bytes = 0;
    void allocate(std::size_t size) {
        if (size <= bytes) return;
        void* p = nullptr; check(cudaMallocHost(&p, size));
        owner = std::shared_ptr<void>(p, [](void* x) { cudaFreeHost(x); }); bytes = size;
    }
    std::uint8_t* data() const { return static_cast<std::uint8_t*>(owner.get()); }
};
struct Event {
    cudaEvent_t value{};
    Event() { check(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
struct CopyStream {
    cudaStream_t value{};
    CopyStream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~CopyStream() { cudaStreamDestroy(value); }
};
__device__ float value(B x) { return __bfloat162float(x); }
__device__ B rounded(float x) { return __float2bfloat16_rn(x); }
__device__ float reduction(float x, float* scratch, bool maximum = false) {
    scratch[threadIdx.x] = x; __syncthreads();
    for (int step = blockDim.x / 2; step; step /= 2) {
        if (threadIdx.x < step) scratch[threadIdx.x] = maximum ? fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + step]) : scratch[threadIdx.x] + scratch[threadIdx.x + step];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}
__global__ void embedding_kernel(const B* table, const int* ids, B* y, int width, int tokens) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < std::size_t(tokens) * width) y[i] = table[std::size_t(ids[i / width]) * width + i % width];
}
__global__ void packed_embedding_kernel(const std::uint8_t* table, const int* ids, B* y, int width, int tokens) {
    const auto i = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (i >= std::size_t(tokens) * width) return;
    const auto row = ids ? std::size_t(ids[i / width]) : i / width, column = i % width;
    float values[8];
    gguf_values8<GGML_TYPE_Q8_0>(table + row * (width / 32) * 34, column, values);
    for (int k = 0; k < 8; ++k) y[i + k] = rounded(values[k]);
}
__global__ void slot_indices_kernel(std::int64_t* indices, int count) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < std::size_t(count)) indices[i] = i;
}
__global__ void norm_kernel(const B* x, const B* weight, B* y, int width, float eps) {
    __shared__ float sums[256];
    const auto row = std::size_t(blockIdx.x) * width;
    float sum = 0;
    for (int d = threadIdx.x; d < width; d += blockDim.x) sum += value(x[row + d]) * value(x[row + d]);
    const float scale = rsqrtf(reduction(sum, sums) / width + eps);
    for (int d = threadIdx.x; d < width; d += blockDim.x)
        y[row + d] = rounded(value(rounded(value(x[row + d]) * scale)) * value(weight[d]));
}
__global__ void add_kernel(const B* x, const B* y, B* out, std::size_t n) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = rounded(value(x[i]) + value(y[i]));
}
__global__ void signal_kernel(const B* projected, B* out, int h, int tokens) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < std::size_t(tokens) * h) { const auto t = i / h, d = i % h; out[i] = rounded(value(projected[t * 3 * h + d]) * value(projected[t * 3 * h + 2 * h + d])); }
}
__global__ void convolution_kernel(const B* signal, const B* projected, const B* weight, B* out, int h, int tokens, int kernel) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(tokens) * h) return;
    const int t = int(i / h), d = int(i % h);
    float sum = 0;
    for (int k = 0; k < kernel; ++k) {
        const int at = t + k - kernel + 1;
        if (at >= 0) sum = fmaf(value(signal[std::size_t(at) * h + d]), value(weight[std::size_t(d) * kernel + k]), sum);
    }
    out[i] = rounded(value(rounded(sum)) * value(projected[std::size_t(t) * 3 * h + h + d]));
}
__global__ void swiglu_kernel(const B* gate, const B* up, B* out, int width, int stride, int tokens) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < std::size_t(tokens) * width) {
        const auto at = i / width * stride + i % width;
        const float g = value(gate[at]);
        out[i] = rounded(value(rounded(g / (1.0f + expf(-g)))) * value(up[at]));
    }
}
__global__ void prepare_qk_kernel(const B* x, const B* weight, B* out, int tokens, int stride,
        int heads, int source_heads, int dim, float eps, double theta, double factor) {
    __shared__ float sums[256];
    const int head = blockIdx.x / tokens, token = blockIdx.x % tokens;
    const int source_head = head / (heads / source_heads);
    const auto input = std::size_t(token) * stride + source_head * dim;
    const auto output = (std::size_t(head) * tokens + token) * dim;
    float sum = 0;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) sum += value(x[input + d]) * value(x[input + d]);
    const float scale = rsqrtf(reduction(sum, sums) / dim + eps);
    const int angles = int(dim * factor) / 2;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const int pair = d < dim / 2 ? d + dim / 2 : d - dim / 2;
        const int frequency = d % (dim / 2);
        const float inv = frequency < angles ? 1.0f / powf(float(theta), float(2 * frequency) / dim) : 0.0f;
        const float phase = token * inv;
        const float cosine = value(rounded(cosf(phase))), sine = value(rounded(sinf(phase)));
        const float a = value(rounded(value(rounded(value(x[input + d]) * scale)) * value(weight[d])));
        const float b = value(rounded(value(rounded(value(x[input + pair]) * scale)) * value(weight[pair])));
        out[output + d] = rounded(a * cosine + (d < dim / 2 ? -b : b) * sine);
    }
}
__global__ void prepare_v_kernel(const B* x, float* out, int tokens, int stride, int heads, int source_heads, int dim) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < std::size_t(heads) * tokens * dim) {
        const int d = i % dim, token = (i / dim) % tokens, head = int(i / dim / tokens);
        out[i] = value(x[std::size_t(token) * stride + (head / (heads / source_heads)) * dim + d]);
    }
}
__global__ void softmax_kernel(const float* scores, float* probability, const int* groups,
        int tokens, int chunk, int start, int state, int window, float scale) {
    __shared__ float sums[256];
    const auto row = std::size_t(blockIdx.x) * tokens;
    const int query = start + int(blockIdx.x % chunk);
    float maximum = -INFINITY;
    for (int key = threadIdx.x; key < tokens; key += blockDim.x) {
        const bool visible = window ? key <= query && key > query - window : query < state || key < state || groups[key] == groups[query];
        if (visible) maximum = fmaxf(maximum, scores[row + key] * scale);
    }
    maximum = reduction(maximum, sums, true);
    float total = 0;
    for (int key = threadIdx.x; key < tokens; key += blockDim.x) {
        const bool visible = window ? key <= query && key > query - window : query < state || key < state || groups[key] == groups[query];
        if (visible) total += expf(scores[row + key] * scale - maximum);
    }
    total = reduction(total, sums);
    for (int key = threadIdx.x; key < tokens; key += blockDim.x) {
        const bool visible = window ? key <= query && key > query - window : query < state || key < state || groups[key] == groups[query];
        probability[row + key] = visible ? expf(scores[row + key] * scale - maximum) / total : 0.0f;
    }
}
__global__ void gate_output_kernel(const float* attention, const B* gate, B* out, int tokens, int heads, int dim, int stride) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < std::size_t(tokens) * heads * dim) {
        const int d = i % dim, head = (i / dim) % heads, token = int(i / dim / heads);
        const float g = value(rounded(1.0f / (1.0f + expf(-value(gate[std::size_t(token) * stride + head])))));
        out[i] = rounded(value(rounded(attention[(std::size_t(head) * tokens + token) * dim + d])) * g);
    }
}
__global__ void gather_slots_kernel(const B* hidden, const int* slots, B* out, int count, int width) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < std::size_t(count) * width) out[i] = hidden[std::size_t(slots[i / width]) * width + i % width];
}
int blocks(std::size_t n) {
    const auto count = (n + 255) / 256;
    if (count > std::size_t(INT_MAX)) throw std::invalid_argument("native d1 launch size overflow");
    return int(count);
}
}
struct Engine::Impl {
    models::lfm3::Config c;
    cublasHandle_t handle = nullptr;
    bool fused, streaming;
    std::string format;
    double load_ms = 0, conversion_ms = 0, packed_repack_ms = 0;
    std::map<std::string, Tensor> weights;
    int tokens = 0, state = 0;
    const int* groups = nullptr;
    const std::int64_t* projection_slots = nullptr;
    std::map<std::string, std::uint64_t> packed_dispatch, packed_kernel_dispatch;
    std::uint64_t bf16_gemm_calls = 0, packed_embedding_rows = 0;
    std::size_t resident_packed_bytes = 0, resident_bf16_bytes = 0;
    struct Layer {
        Pinned image;
        std::vector<std::pair<Tensor*, std::size_t>> fields;
    };
    struct Slot {
        Tensor image;
        Event ready, done;
        std::size_t layer = SIZE_MAX;
        bool used = false;
    };
    std::vector<Layer> layers;
    std::array<Slot, 2> slots;
    CopyStream copy;
    Pinned embedding_stage;
    Event embedding_copied;
    bool embedding_copy_pending = false;
    std::size_t active_layer = SIZE_MAX, host_bytes = 0, buffer_bytes = 0;
    std::size_t layer_h2d_bytes = 0, embedding_h2d_bytes = 0;
    nlohmann::json tensor_manifest;
    std::size_t ready_device_used = 0;
    Impl(Input& input, bool fuse, bool stream) : c(input.config), fused(fuse), streaming(stream), format(input.format) {
        cudaDeviceProp device{}; check(cudaGetDeviceProperties(&device, 0));
        if (device.major < 8) throw std::invalid_argument("native BF16 cuBLAS execution requires BF16-capable CUDA hardware");
        blas(cublasCreate(&handle));
        blas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        const auto start = Clock::now();
        const auto load = [&](const std::string& key, const std::vector<std::int64_t>& shape) {
            auto host = input.load("model.language_model." + key, shape);
            std::size_t n = 1; for (auto d : shape) n *= std::size_t(d);
            const bool quantized = !host.packed.empty();
            const auto bytes = quantized ? host.packed.size() : n * sizeof(B);
            Tensor tensor = streaming && key != "embedding_norm.weight" ? Tensor::host_weight(n, bytes, quantized)
                : quantized ? Tensor::packed_weight(n, bytes) : Tensor(n, false);
            const void* data = tensor.packed ? static_cast<const void*>(host.packed.data()) : static_cast<const void*>(host.values.data());
            if (tensor.host) std::memcpy(tensor.data, data, tensor.storage_bytes);
            else check(cudaMemcpy(tensor.data, data, tensor.storage_bytes, cudaMemcpyHostToDevice));
            tensor.label = key;
            if (shape.size() == 2) { tensor.weight_rows = int(shape[0]); tensor.weight_columns = int(shape[1]); }
            weights.emplace(key, std::move(tensor));
        };
        load("embed_tokens.weight", {c.vocabulary, c.hidden});
        load("embedding_norm.weight", {c.hidden});
        for (std::size_t i = 0; i < c.layers.size(); ++i) {
            const auto p = "layers." + std::to_string(i) + ".";
            load_layer_weights(c, i, [&](const std::string& key, const std::vector<std::int64_t>& shape) {
                load(p + key, shape);
            });
            if (fused) {
                pack(p + "feed_forward.fused", {p + "feed_forward.w1.weight", p + "feed_forward.w3.weight"});
                if (c.layers[i] != "conv") {
                    const auto a = p + "self_attn.";
                    pack(a + "fused", {a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight", a + "gate_proj.weight"});
                }
            }
        }
        check(cudaDeviceSynchronize());
        const auto repack_start = Clock::now();
        for (auto& item : weights) {
            auto& source = item.second;
            if (!source.packed || item.first == "embed_tokens.weight") continue;
            Tensor reordered = streaming ? Tensor::host_weight(source.count, source.storage_bytes, true)
                : Tensor::packed_weight(source.count, source.storage_bytes);
            reordered.label = source.label; reordered.weight_rows = source.weight_rows;
            reordered.weight_columns = source.weight_columns; reordered.column_planes = true;
            if (streaming) {
                const auto rows = std::size_t(source.weight_rows), columns = std::size_t(source.weight_columns);
                const auto* original = reinterpret_cast<const std::uint8_t*>(source.data);
                auto* target = reinterpret_cast<std::uint8_t*>(reordered.data);
                for (std::size_t row = 0; row < rows; ++row) for (std::size_t block = 0; block < columns / 32; ++block) {
                    const auto* raw = original + (row * (columns / 32) + block) * 34;
                    std::memcpy(target + rows * columns + (block * rows + row) * 2, raw, 2);
                    for (std::size_t part = 0; part < 4; ++part)
                        std::memcpy(target + ((block * 4 + part) * rows + row) * 8, raw + 2 + part * 8, 8);
                }
            } else mfq::cuda::gguf::repack_q8_bf16_columns(source.data, std::uint64_t(source.weight_columns / 32) * 34,
                source.weight_rows, source.weight_columns, reordered.data, nullptr);
            // Release each original quantized allocation before repacking the
            // next matrix; only one extra packed matrix is live during loading.
            if (!streaming) check(cudaStreamSynchronize(nullptr));
            source = std::move(reordered);
        }
        packed_repack_ms = std::chrono::duration<double, std::milli>(Clock::now() - repack_start).count();
        if (streaming) prepare_stream();
        tensor_manifest = nlohmann::json::array();
        for (const auto& item : weights) {
            const auto& w = item.second;
            if (!w.host) (w.packed ? resident_packed_bytes : resident_bf16_bytes) += w.storage_bytes;
            tensor_manifest.push_back({{"name", item.first}, {"storage", w.packed ? "Q8_0" : "BF16"},
                {"bytes", w.storage_bytes}, {"elements", w.count}, {"rows", w.weight_rows}, {"columns", w.weight_columns},
                {"placement", w.host ? "ram" : "vram"},
                {"layout", w.column_planes ? "q8-codes-K8-N-8; fp16-scales-K32-N" : (w.packed ? "ggml-Q8_0-blocks" : "dense-BF16")}});
        }
        std::size_t free = 0, total = 0; check(cudaMemGetInfo(&free, &total)); ready_device_used = total - free;
        conversion_ms = input.conversion_ms;
        load_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    ~Impl() {
        cudaStreamSynchronize(copy.value); cudaStreamSynchronize(nullptr);
        if (handle) cublasDestroy(handle);
    }
    void prepare_stream() {
        layers.resize(c.layers.size());
        std::size_t maximum = 0;
        for (std::size_t index = 0; index < layers.size(); ++index) {
            auto& layer = layers[index];
            const auto name = prefix(index); std::size_t bytes = 0;
            for (auto& item : weights) if (item.first.rfind(name, 0) == 0) {
                if (bytes > SIZE_MAX - 255 || item.second.storage_bytes > SIZE_MAX - ((bytes + 255) & ~std::size_t(255)))
                    throw std::overflow_error("d1 streamed layer size overflow");
                bytes = (bytes + 255) & ~std::size_t(255);
                layer.fields.emplace_back(&item.second, bytes); bytes += item.second.storage_bytes;
            }
            if (!bytes) throw std::invalid_argument("d1 streamed layer is empty");
            layer.image.allocate(bytes);
            for (auto [weight, offset] : layer.fields) {
                std::memcpy(layer.image.data() + offset, weight->data, weight->storage_bytes);
                weight->owner = layer.image.owner;
                weight->data = reinterpret_cast<B*>(layer.image.data() + offset);
            }
            host_bytes += bytes; maximum = std::max(maximum, bytes);
        }
        host_bytes += weights.at("embed_tokens.weight").storage_bytes;
        for (auto& slot : slots) slot.image = Tensor((maximum + 1) / 2, false);
        buffer_bytes = slots.size() * ((maximum + 1) / 2) * 2;
    }
    void queue_layer(std::size_t index) {
        if (!streaming || index >= layers.size()) return;
        auto& slot = slots[index % slots.size()];
        if (slot.layer == index) return;
        if (slot.used) check(cudaStreamWaitEvent(copy.value, slot.done.value, 0));
        const auto& source = layers[index].image;
        constexpr std::size_t quantum = 4u << 20;
        for (std::size_t offset = 0; offset < source.bytes; offset += quantum)
            check(cudaMemcpyAsync(reinterpret_cast<std::uint8_t*>(slot.image.data) + offset, source.data() + offset,
                std::min(quantum, source.bytes - offset), cudaMemcpyHostToDevice, copy.value));
        check(cudaEventRecord(slot.ready.value, copy.value));
        slot.layer = index; layer_h2d_bytes += source.bytes;
    }
    struct LayerLease {
        Impl* owner;
        ~LayerLease() { if (owner) owner->leave_layer(); }
        LayerLease(const LayerLease&) = delete;
        explicit LayerLease(Impl* value) : owner(value) {}
    };
    LayerLease enter_layer(std::size_t index) {
        if (!streaming) return LayerLease(nullptr);
        queue_layer(index);
        auto& slot = slots[index % slots.size()];
        check(cudaStreamWaitEvent(nullptr, slot.ready.value, 0));
        for (auto [weight, offset] : layers[index].fields)
            weight->data = reinterpret_cast<B*>(reinterpret_cast<std::uint8_t*>(slot.image.data) + offset);
        active_layer = index;
        return LayerLease(this);
    }
    void leave_layer() noexcept {
        if (active_layer == SIZE_MAX) return;
        auto& slot = slots[active_layer % slots.size()];
        const auto result = cudaEventRecord(slot.done.value, nullptr);
        if (result != cudaSuccess) std::fprintf(stderr, "d1 layer fence failed: %s\n", cudaGetErrorString(result));
        slot.used = true; active_layer = SIZE_MAX;
    }
    void pack(const std::string& key, const std::vector<std::string>& names) {
        std::size_t count = 0, bytes = 0;
        const auto& first = weights.at(names.at(0));
        const bool packed = first.packed; const int columns = first.weight_columns;
        for (const auto& name : names) {
            const auto& source = weights.at(name);
            if (source.packed != packed || source.weight_columns != columns) throw std::invalid_argument("incompatible d1 fused weight storage");
            count += source.count; bytes += source.storage_bytes;
        }
        Tensor merged = streaming ? Tensor::host_weight(count, bytes, packed)
            : packed ? Tensor::packed_weight(count, bytes) : Tensor(count, false);
        merged.label = key; merged.weight_columns = columns; merged.weight_rows = int(count / columns);
        std::size_t offset = 0;
        for (const auto& name : names) {
            const auto& source = weights.at(name);
            if (streaming) std::memcpy(reinterpret_cast<std::uint8_t*>(merged.data) + offset, source.data, source.storage_bytes);
            else check(cudaMemcpy(reinterpret_cast<std::uint8_t*>(merged.data) + offset, source.data, source.storage_bytes, cudaMemcpyDeviceToDevice));
            offset += source.storage_bytes;
        }
        for (const auto& name : names) weights.erase(name);
        weights.emplace(key, std::move(merged));
    }
    std::string prefix(std::size_t layer) { return "layers." + std::to_string(layer) + "."; }
    Tensor linear(const Tensor& x, const Tensor& w, int out, int in, int rows = -1) {
        if (rows < 0) rows = tokens;
        Tensor y(std::size_t(rows) * out);
        if (w.packed) {
            if (in % 32 || w.weight_columns != in || w.weight_rows != out || !projection_slots)
                throw std::invalid_argument("unsupported packed d1 projection geometry");
            if (w.column_planes) {
                const auto kernel = mfq::cuda::gguf::matmul_q8_bf16_columns(w.data, out, in, rows, x.data, y.data, nullptr);
                const char* name = kernel == mfq::cuda::gguf::Q8Bf16Kernel::Pipeline32Plain ? "q8-column-pipeline32-fp32" :
                    kernel == mfq::cuda::gguf::Q8Bf16Kernel::Pipeline32 ? "q8-column-pipeline32" :
                    kernel == mfq::cuda::gguf::Q8Bf16Kernel::Native32 ? "q8-column-native32" : "q8-column-native64";
                ++packed_kernel_dispatch[name];
            } else {
                mfq::cuda::gguf::matmul_q8_bf16(w.data, std::uint64_t(in / 32) * 34, out, in, rows, x.data, y.data, nullptr);
                ++packed_kernel_dispatch["q8-raw-native"];
            }
            check(cudaGetLastError()); ++packed_dispatch[w.label];
            if (active_layer != SIZE_MAX) queue_layer(active_layer + 1);
            return y;
        }
        ++bf16_gemm_calls;
        const float one = 1, zero = 0;
        blas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &one, w.data, CUDA_R_16BF, in,
            x.data, CUDA_R_16BF, in, &zero, y.data, CUDA_R_16BF, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        if (active_layer != SIZE_MAX) queue_layer(active_layer + 1);
        return y;
    }
    Tensor norm(const Tensor& x, const Tensor& weight) {
        Tensor y(x.count);
        norm_kernel<<<tokens, 256>>>(x.data, weight.data, y.data, c.hidden, c.epsilon);
        check(cudaGetLastError()); return y;
    }
    Tensor normalize(const Tensor& x, std::size_t layer, const char* name) { return norm(x, weights.at(prefix(layer) + name + ".weight")); }
    Tensor finalize(const Tensor& x) { return norm(x, weights.at("embedding_norm.weight")); }
    void finish_layer(const Tensor&) {}
    Tensor add(const Tensor& x, const Tensor& y) {
        Tensor out(x.count); add_kernel<<<blocks(x.count), 256>>>(x.data, y.data, out.data, x.count); check(cudaGetLastError()); return out;
    }
    Tensor convolve(const Tensor& x, std::size_t layer) {
        const auto p = prefix(layer) + "conv.";
        auto projected = linear(x, weights.at(p + "in_proj.weight"), 3 * c.hidden, c.hidden);
        Tensor signal(std::size_t(tokens) * c.hidden), mixed(signal.count);
        signal_kernel<<<blocks(signal.count), 256>>>(projected.data, signal.data, c.hidden, tokens);
        convolution_kernel<<<blocks(mixed.count), 256>>>(signal.data, projected.data, weights.at(p + "conv.weight").data, mixed.data, c.hidden, tokens, c.convolution);
        check(cudaGetLastError());
        return linear(mixed, weights.at(p + "out_proj.weight"), c.hidden, c.hidden);
    }
    Tensor feed_forward(const Tensor& x, std::size_t layer) {
        const auto p = prefix(layer) + "feed_forward.";
        Tensor projected, gate, up;
        const B* g; const B* u; int stride;
        if (fused) { projected = linear(x, weights.at(p + "fused"), 2 * c.intermediate, c.hidden); g = projected.data; u = g + c.intermediate; stride = 2 * c.intermediate; }
        else { gate = linear(x, weights.at(p + "w1.weight"), c.intermediate, c.hidden); up = linear(x, weights.at(p + "w3.weight"), c.intermediate, c.hidden); g = gate.data; u = up.data; stride = c.intermediate; }
        Tensor activated(std::size_t(tokens) * c.intermediate);
        swiglu_kernel<<<blocks(activated.count), 256>>>(g, u, activated.data, c.intermediate, stride, tokens);
        check(cudaGetLastError());
        return linear(activated, weights.at(p + "w2.weight"), c.hidden, c.intermediate);
    }
    Tensor attend(const Tensor& x, std::size_t layer) {
        const auto p = prefix(layer) + "self_attn.";
        const int qn = c.heads * c.head_dim, kn = c.kv_heads * c.head_dim;
        Tensor combined, query, key, val, gate;
        const B *qp, *kp, *vp, *gp; int stride, gate_stride;
        if (fused) {
            stride = qn + 2 * kn + c.heads;
            combined = linear(x, weights.at(p + "fused"), stride, c.hidden);
            qp = combined.data; kp = qp + qn; vp = kp + kn; gp = vp + kn; gate_stride = stride;
        } else {
            query = linear(x, weights.at(p + "q_proj.weight"), qn, c.hidden);
            key = linear(x, weights.at(p + "k_proj.weight"), kn, c.hidden);
            val = linear(x, weights.at(p + "v_proj.weight"), kn, c.hidden);
            gate = linear(x, weights.at(p + "gate_proj.weight"), c.heads, c.hidden);
            qp = query.data; kp = key.data; vp = val.data; gp = gate.data; stride = 0; gate_stride = c.heads;
        }
        const auto count = std::size_t(c.heads) * tokens * c.head_dim;
        Tensor q(count), k(count), flattened(count);
        Raw v(count * sizeof(float)), attended(count * sizeof(float));
        const bool sliding = c.layers[layer] == "sliding_attention";
        const double theta = sliding ? c.sliding_theta : c.full_theta, factor = sliding ? c.sliding_factor : c.full_factor;
        prepare_qk_kernel<<<c.heads * tokens, 256>>>(qp, weights.at(p + "q_norm.weight").data, q.data, tokens, fused ? stride : qn, c.heads, c.heads, c.head_dim, c.epsilon, theta, factor);
        prepare_qk_kernel<<<c.heads * tokens, 256>>>(kp, weights.at(p + "k_norm.weight").data, k.data, tokens, fused ? stride : kn, c.heads, c.kv_heads, c.head_dim, c.epsilon, theta, factor);
        prepare_v_kernel<<<blocks(count), 256>>>(vp, static_cast<float*>(v.p), tokens, fused ? stride : kn, c.heads, c.kv_heads, c.head_dim);
        const int tile = std::min(256, tokens);
        Raw scores(std::size_t(c.heads) * tile * tokens * sizeof(float));
        // Do not round probabilities to BF16: SDPA accumulates PV before its
        // BF16 output cast, and an extra rounding can change median Score bins.
        Raw probability(std::size_t(c.heads) * tile * tokens * sizeof(float));
        const float one = 1, zero = 0;
        for (int start = 0; start < tokens; start += tile) {
            const int chunk = std::min(tile, tokens - start);
            const long long head_stride = std::int64_t(tokens) * c.head_dim;
            const long long score_stride = std::int64_t(chunk) * tokens;
            blas(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, tokens, chunk, c.head_dim, &one,
                k.data, CUDA_R_16BF, c.head_dim, head_stride, q.data + std::size_t(start) * c.head_dim, CUDA_R_16BF, c.head_dim, head_stride,
                &zero, scores.p, CUDA_R_32F, tokens, score_stride, c.heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            softmax_kernel<<<c.heads * chunk, 256>>>(static_cast<float*>(scores.p), static_cast<float*>(probability.p), groups, tokens, chunk, start, state, sliding ? c.window : 0, 1.0f / std::sqrt(float(c.head_dim)));
            blas(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, c.head_dim, chunk, tokens, &one,
                v.p, CUDA_R_32F, c.head_dim, head_stride, probability.p, CUDA_R_32F, tokens, score_stride,
                &zero, static_cast<float*>(attended.p) + std::size_t(start) * c.head_dim, CUDA_R_32F, c.head_dim, head_stride, c.heads, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
        }
        gate_output_kernel<<<blocks(count), 256>>>(static_cast<float*>(attended.p), gp, flattened.data, tokens, c.heads, c.head_dim, gate_stride);
        check(cudaGetLastError());
        return linear(flattened, weights.at(p + "o_proj.weight"), c.hidden, qn);
    }
    Tensor embedding(const int* ids, int rows, const std::vector<int>& host_ids) {
        Tensor output(std::size_t(rows) * c.hidden);
        const auto& table = weights.at("embed_tokens.weight");
        if (streaming) {
            if (embedding_copy_pending) check(cudaEventSynchronize(embedding_copied.value));
            const auto stride = table.storage_bytes / std::size_t(c.vocabulary);
            const auto bytes = std::size_t(rows) * stride;
            embedding_stage.allocate(bytes);
            for (int row = 0; row < rows; ++row)
                std::memcpy(embedding_stage.data() + std::size_t(row) * stride,
                    reinterpret_cast<const std::uint8_t*>(table.data) + std::size_t(host_ids[row]) * stride, stride);
            if (table.packed) {
                Raw gathered(bytes);
                check(cudaMemcpyAsync(gathered.p, embedding_stage.data(), bytes, cudaMemcpyHostToDevice, nullptr));
                check(cudaEventRecord(embedding_copied.value, nullptr));
                packed_embedding_kernel<<<blocks((output.count + 7) / 8), 256>>>(static_cast<const std::uint8_t*>(gathered.p), nullptr,
                    output.data, c.hidden, rows);
                packed_embedding_rows += rows;
            } else {
                check(cudaMemcpyAsync(output.data, embedding_stage.data(), bytes, cudaMemcpyHostToDevice, nullptr));
                check(cudaEventRecord(embedding_copied.value, nullptr));
            }
            embedding_copy_pending = true; embedding_h2d_bytes += bytes;
            check(cudaGetLastError()); return output;
        }
        if (table.packed) {
            packed_embedding_kernel<<<blocks((output.count + 7) / 8), 256>>>(reinterpret_cast<const std::uint8_t*>(table.data), ids,
                output.data, c.hidden, rows);
            packed_embedding_rows += rows;
        } else embedding_kernel<<<blocks(output.count), 256>>>(table.data, ids, output.data, c.hidden, rows);
        check(cudaGetLastError()); return output;
    }
    Tensor run(const int* ids, const std::vector<int>& host_ids, const std::vector<int>& slots, const std::vector<int>& candidates, const int* dslots, const int* dcandidates) {
        packed_dispatch.clear(); packed_kernel_dispatch.clear(); bf16_gemm_calls = 0; packed_embedding_rows = 0;
        layer_h2d_bytes = 0; embedding_h2d_bytes = 0;
        auto x = embedding(ids, tokens, host_ids);
        auto hidden = models::lfm3::forward(c, *this, std::move(x));
        Tensor chosen(std::size_t(slots.size()) * c.hidden);
        auto output_weights = embedding(dcandidates, int(candidates.size()), candidates);
        gather_slots_kernel<<<blocks(chosen.count), 256>>>(hidden.data, dslots, chosen.data, int(slots.size()), c.hidden);
        check(cudaGetLastError());
        return linear(chosen, output_weights, int(candidates.size()), c.hidden, int(slots.size()));
    }
};
Engine::Engine(Input& input, bool fused, bool streaming) : impl_(std::make_unique<Impl>(input, fused, streaming)) {}
Engine::~Engine() = default;
nlohmann::json Engine::ready() const {
    return {{"ready", true}, {"backend", "native-cuda-cublas-lfm3"}, {"native_runtime", true},
        {"weight_format", impl_->format}, {"fused_projections", impl_->fused}, {"load_ms", impl_->load_ms},
        {"weight_conversion_ms", impl_->conversion_ms}, {"packed_weight_repack_ms", impl_->packed_repack_ms},
        {"resident_packed_weight_bytes", impl_->resident_packed_bytes},
        {"resident_bf16_weight_bytes", impl_->resident_bf16_bytes}, {"full_matrix_dequant_cache_bytes", 0},
        {"weight_placement", impl_->streaming ? "stream" : "vram"}, {"streamed_weight_bytes", impl_->host_bytes},
        {"layer_buffer_bytes", impl_->buffer_bytes}, {"layer_buffer_count", impl_->streaming ? 2 : 0},
        {"device_used_after_load_bytes", impl_->ready_device_used}, {"tensors", impl_->tensor_manifest}};
}
nlohmann::json Engine::evaluate(const nlohmann::json& request) {
    auto& m = *impl_;
    const auto input = Request::parse(request, m.c);
    const auto& ids = input.tokens;
    const auto& groups = input.groups;
    const auto& slots = input.slots;
    const auto& candidates = input.candidates;
    const int state = input.state_tokens;
    const int iterations = input.iterations, warmup = input.warmup;
    Raw dids(ids.size() * 4), dgroups(groups.size() * 4), dslots(slots.size() * 4), dcandidates(candidates.size() * 4);
    check(cudaMemcpy(dids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dgroups.p, groups.data(), groups.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dslots.p, slots.data(), slots.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dcandidates.p, candidates.data(), candidates.size() * 4, cudaMemcpyHostToDevice));
    Raw projection_slots(ids.size() * sizeof(std::int64_t));
    slot_indices_kernel<<<blocks(ids.size()), 256>>>(static_cast<std::int64_t*>(projection_slots.p), int(ids.size()));
    check(cudaGetLastError()); m.projection_slots = static_cast<const std::int64_t*>(projection_slots.p);
    m.tokens = int(ids.size()); m.state = state; m.groups = static_cast<const int*>(dgroups.p);
    cudaEvent_t begin{}, end{}; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    struct Events { cudaEvent_t a, b; ~Events() { cudaEventDestroy(a); cudaEventDestroy(b); } } events{begin, end};
    Tensor result; std::vector<float> milliseconds;
    const auto wall = Clock::now();
    for (int i = -warmup; i < iterations; ++i) {
        check(cudaEventRecord(begin));
        result = m.run(static_cast<const int*>(dids.p), ids, slots, candidates, static_cast<const int*>(dslots.p), static_cast<const int*>(dcandidates.p));
        check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
        float elapsed = 0; check(cudaEventElapsedTime(&elapsed, begin, end));
        if (i >= 0) milliseconds.push_back(elapsed);
    }
    std::vector<std::uint16_t> output(result.count);
    check(cudaMemcpy(output.data(), result.data, result.count * 2, cudaMemcpyDeviceToHost));
    std::size_t free = 0, total = 0; check(cudaMemGetInfo(&free, &total));
    std::uint64_t pool_reserved_high = 0, pool_used_high = 0;
    cudaMemPool_t pool{};
    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess) {
        check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &pool_reserved_high));
        check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &pool_used_high));
    } else cudaGetLastError();
    std::uint64_t packed_calls = 0; for (const auto& item : m.packed_dispatch) packed_calls += item.second;
    nlohmann::json logits = nlohmann::json::array();
    for (std::size_t row = 0; row < slots.size(); ++row) {
        auto values = nlohmann::json::array();
        for (std::size_t col = 0; col < candidates.size(); ++col) {
            const std::uint32_t bits = std::uint32_t(output[row * candidates.size() + col]) << 16;
            float x; std::memcpy(&x, &bits, 4);
            if (!std::isfinite(x)) throw std::runtime_error("non-finite native d1 logit");
            values.push_back(x);
        }
        logits.push_back(std::move(values));
    }
    auto sorted = milliseconds; std::sort(sorted.begin(), sorted.end());
    return {{"backend", "native-cuda-cublas-lfm3"}, {"native_runtime", true}, {"logits", std::move(logits)},
        {"candidate_ids", candidates}, {"gpu_ms", milliseconds}, {"gpu_median_ms", sorted[sorted.size() / 2]},
        {"benchmark_wall_ms", std::chrono::duration<double, std::milli>(Clock::now() - wall).count()},
        {"fused_projections", m.fused}, {"weight_format", m.format}, {"output_tokens", 0},
        {"packed_kernel", packed_calls ? "gguf::matmul_q8_bf16_columns; BF16 activation, FP32 m16n8k16; policy in packed_kernel_dispatch" : "none"},
        {"packed_gemm_calls_per_forward", packed_calls}, {"packed_projection_dispatch", m.packed_dispatch},
        {"packed_kernel_dispatch", m.packed_kernel_dispatch},
        {"bf16_gemm_calls_per_forward", m.bf16_gemm_calls}, {"packed_embedding_rows_per_forward", m.packed_embedding_rows},
        {"layer_h2d_bytes", m.layer_h2d_bytes}, {"embedding_h2d_bytes", m.embedding_h2d_bytes},
        {"device_used_after_forward_bytes", total - free}, {"temporary_pool_reserved_high_bytes", pool_reserved_high},
        {"temporary_pool_used_high_bytes", pool_used_high}};
}
}
