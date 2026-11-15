#include "engine/backends/cuda/executor/kv_memory.h"

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "engine/backends/cuda/native/vmm.h"
#include "engine/weights/cache.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>

namespace tyloi::cuda {
namespace {
constexpr const char* insufficient = "kv_memory_insufficient: available VRAM and RAM cannot hold the configured KV cache; choose automatic with SSD KV or reduce context length";
std::size_t width_bytes(const TensorOptions& options) { return scalar_size(options.scalar_type()); }
::tyloi::DenseTensor external(void* data, std::shared_ptr<void> owner, const std::vector<int64_t>& shape,
                    const ::tyloi::TensorOptions& options) {
    auto view = make_contiguous_view(data, shape, options.scalar_type(), options.target_device());
    auto storage = std::make_shared<TensorStorage>();
    storage->base = data; storage->bytes = view.nbytes(); storage->device = options.target_device();
    storage->owner = std::move(owner);
    return DenseTensor(std::move(storage), view);
}
::tyloi::DenseTensor host_tensor(const std::vector<int64_t>& shape, const ::tyloi::TensorOptions& options) {
    std::size_t bytes = width_bytes(options);
    for (const auto dimension : shape) {
        if (dimension <= 0 || std::size_t(dimension) > std::numeric_limits<std::size_t>::max()/bytes)
            throw std::invalid_argument("KV tensor size overflow");
        bytes *= dimension;
    }
    void* host = nullptr;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
    charge_tensor_host_bytes(bytes);
    auto owner = std::shared_ptr<void>(host, [bytes](void* p) {
        (void)cudaDeviceSynchronize(); (void)cudaFreeHost(p); tensor_host_bytes.fetch_sub(bytes);
    });
    std::memset(host, 0, bytes);
    void* gpu = nullptr; TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&gpu, host, 0));
    return external(gpu, std::move(owner), shape, options);
}
}

namespace {
std::set<int64_t> virtual_pages(int64_t maximum, int64_t heads, int64_t width,
        std::size_t element, int64_t tokens) {
    std::set<int64_t> result;
    tokens=std::min(maximum,((tokens+8191)/8192)*8192);
    const auto g=vmm::vmm_granularity();
    for(int64_t head=0;head<heads;++head) {
        const auto begin=std::size_t(head)*maximum*width*element;
        const auto end=begin+std::size_t(tokens)*width*element;
        for(auto page=begin/g;page<(end+g-1)/g;++page)result.insert(int64_t(page));
    }
    return result;
}
}

struct KvMemoryAllocation::Virtual {
    vmm::VmmRange range;
    int64_t maximum, heads, width, committed = 0;
    std::size_t element;
    Virtual(int64_t maximum, int64_t heads, int64_t width, std::size_t element)
        : maximum(maximum), heads(heads), width(width), element(element) {
        if (!range.reserve(std::size_t(maximum)*heads*width*element))
            throw std::runtime_error("kv_dynamic_unavailable: CUDA virtual memory reservation failed");
    }
    ~Virtual() { (void)cudaDeviceSynchronize(); }
    std::set<int64_t> missing(int64_t tokens) const {
        tokens=std::min(maximum,((tokens+8191)/8192)*8192);
        if (tokens <= committed) return {};
        auto result=virtual_pages(maximum,heads,width,element,tokens);
        for(auto it=result.begin();it!=result.end();)
            if(range.mapped(*it))it=result.erase(it);else ++it;
        return result;
    }
    void ensure(int64_t tokens) {
        tokens=std::min(maximum,((tokens+8191)/8192)*8192);
        const auto fresh = missing(tokens);
        if (!fresh.empty()) TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
        for (const auto page : fresh) {
            if (!range.map_range(page,page+1,[]{return vmm::VmmChunk(0);}))
                throw std::runtime_error(insufficient);
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemsetAsync(range.base()+page*vmm::vmm_granularity(),0,
                vmm::vmm_granularity(),current_stream().stream()));
        }
        committed = std::max(committed,tokens);
    }
};

std::vector<Tensor> KvMemoryAllocation::storage() const {
    std::vector<Tensor> result;
    const auto add = [&](const Tensor& value) { if (value.defined()) result.push_back(value); };
    add(keys); add(values); add(index); add(pooled);
    add(hot.keys); add(hot.values); add(hot.mapping); add(hot.slot_ids); add(hot.requested);
    add(hot.epochs); add(hot.queue); add(hot.control); add(hot.cold_keys); add(hot.cold_values);
    add(hot.cold_mapping); add(hot.free_cold); add(hot.free_gpu); add(hot.cold_control);
    return result;
}

std::size_t KvMemoryAllocation::initial_bytes() const {
    const auto tokens = std::min<int64_t>(layout.maximum, 16384);
    const auto pages = virtual_pages(layout.maximum, layout.heads, layout.key_width, width_bytes(options), tokens).size()
        + virtual_pages(layout.maximum, layout.heads, layout.value_width, width_bytes(options), tokens).size();
    return pages * vmm::vmm_granularity();
}

std::size_t KvMemoryAllocation::gpu_bytes() const {
    if (!ready) return 0;
    if (virtual_keys_) return layout.index_bytes() +
        std::size_t(virtual_keys_->range.mapped_count() + virtual_values_->range.mapped_count()) * vmm::vmm_granularity();
    return layout.index_bytes() + hot.bytes() + (host || offload ? 0 : layout.raw_bytes());
}

std::size_t KvMemoryAllocation::ram_bytes() const {
    if (!ready) return 0;
    if (hot.exclusive()) return std::size_t(hot.cold_keys.numel() + hot.cold_values.numel()) * width_bytes(options);
    return host ? layout.raw_bytes() : 0;
}

std::size_t KvMemoryAllocation::growth_bytes(int64_t tokens) const {
    tokens = std::min(layout.maximum, tokens);
    return virtual_keys_ ? (virtual_keys_->missing(tokens).size() + virtual_values_->missing(tokens).size())
        * vmm::vmm_granularity() : 0;
}

void KvMemoryAllocation::ensure(int64_t tokens) {
    if (tokens < 0 || tokens > layout.maximum) throw std::invalid_argument("KV growth exceeds configured context");
    if (!expert_regions_.empty()) expert_regions_.front().advance(tokens + layout.pool - 1);
    if (virtual_keys_) { virtual_keys_->ensure(tokens); virtual_values_->ensure(tokens); }
}

void KvMemoryAllocation::materialize(const engine::KvOptions& policy, const engine::KvPlacement& placement,
        std::shared_ptr<KvOffloadStore> store, weights::Cache* lending) {
    const auto cells = placement.cells;
    const auto shape=[&](int64_t w){return std::vector<int64_t>{1,layout.heads,layout.maximum,w};};
    auto index_options=options;index_options.dtype(::tyloi::kFloat16);
    index=::tyloi::cuda::zeros({1,layout.maximum,layout.index_width},index_options);
    pooled=::tyloi::cuda::zeros({1,layout.pooled_rows(),layout.index_width},index_options);
    host=policy.placement=="ram" || placement.automatic_ram || placement.exclusive || placement.raw_host_only;offload=std::move(store);
    if(policy.dynamic) {
        virtual_keys_=std::make_shared<Virtual>(layout.maximum,layout.heads,layout.key_width,width_bytes(options));
        virtual_values_=std::make_shared<Virtual>(layout.maximum,layout.heads,layout.value_width,width_bytes(options));
        keys=external(virtual_keys_->range.base(),virtual_keys_,shape(layout.key_width),options);
        values=external(virtual_values_->range.base(),virtual_values_,shape(layout.value_width),options);
        ensure(std::min<int64_t>(layout.maximum,16384));
    } else if(placement.exclusive) {
        auto cold_shape=[&](int64_t width){return std::vector<int64_t>{1,layout.heads,layout.maximum-cells,width};};
        hot=HotKvCache::allocate(layout.heads,layout.maximum,cells,layout.key_width,layout.value_width,options);
        auto cold_keys=host_tensor(cold_shape(layout.key_width),options),cold_values=host_tensor(cold_shape(layout.value_width),options);
        hot.set_cold_storage(cold_keys,cold_values);
        // Logical geometry only: tier-aware kernels resolve every row via
        // the maps. The owner retains the smaller physical RAM tensors.
        keys=external(cold_keys.data_ptr(),std::make_shared<::tyloi::DenseTensor>(cold_keys),shape(layout.key_width),options);
        values=external(cold_values.data_ptr(),std::make_shared<::tyloi::DenseTensor>(cold_values),shape(layout.value_width),options);
    } else if(host) {
        keys=host_tensor(shape(layout.key_width),options);values=host_tensor(shape(layout.value_width),options);
        if(placement.automatic_ram)hot=HotKvCache::allocate(layout.heads,layout.maximum,cells,layout.key_width,layout.value_width,options);
    } else if(!offload) {
        keys=::tyloi::cuda::zeros(shape(layout.key_width),options);values=::tyloi::cuda::zeros(shape(layout.value_width),options);
    }
    ready=true;
    if(lending) {
        expert_regions_.push_back(lending->lend_kv(index,layout.maximum,std::size_t(layout.index_width)*2,false));
        expert_regions_.push_back(lending->lend_kv(pooled,layout.pooled_rows(),std::size_t(layout.index_width)*2,false,layout.pool));
        if(hot.exclusive() || (!host && !offload)) {
            const auto& k=hot.exclusive()?hot.keys:keys.dense();
            const auto& v=hot.exclusive()?hot.values:values.dense();
            for(int64_t h=0;h<layout.heads;++h) {
                expert_regions_.push_back(lending->lend_kv(k.narrow(1,h,1),k.size(2),std::size_t(layout.key_width)*width_bytes(options),hot.exclusive()));
                expert_regions_.push_back(lending->lend_kv(v.narrow(1,h,1),v.size(2),std::size_t(layout.value_width)*width_bytes(options),hot.exclusive()));
            }
        }
    }
}

} // namespace tyloi::cuda
