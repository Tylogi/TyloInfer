#include "engine/backends/cuda/executor/kv/memory.h"

#include "engine/memory/budget.h"
#include "engine/generation/prefill/budget.h"
#include "engine/weights/cache/cache.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "engine/backends/cuda/native/vmm.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <tuple>

namespace tyloi::cuda {
namespace {
constexpr const char* insufficient = "kv_memory_insufficient: available VRAM and RAM cannot hold the configured KV cache; choose automatic with SSD KV or reduce context length";
std::size_t width_bytes(const TensorOptions& options) { return scalar_size(options.scalar_type()); }
std::size_t setting(const char* name, std::size_t fallback) {
    const char* value = tyloi::cuda::env_lookup(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const auto result = std::strtoull(value, &end, 10);
    if (*value == '-' || !end || *end) throw std::invalid_argument(std::string("invalid ") + name);
    return std::size_t(result);
}
} // namespace

engine::KvOptions kv_memory_options() {
    engine::KvOptions o;
    if (const auto* p=tyloi::cuda::env_lookup("TYLOINFER_KV_PLACEMENT")) o.placement=p;
    if (o.placement!="auto" && o.placement!="auto_ssd" && o.placement!="vram" && o.placement!="ram")
        throw std::invalid_argument("invalid KV placement");
    if (const auto* p=tyloi::cuda::env_lookup("TYLOINFER_KV_ALLOCATION")) {
        const std::string allocation=p;
        if (allocation!="preallocated" && allocation!="dynamic") throw std::invalid_argument("invalid KV allocation");
        o.dynamic=allocation=="dynamic";
    }
    if (o.dynamic && (o.placement=="ram" || o.placement=="auto_ssd"))
        throw std::invalid_argument("kv_dynamic_unavailable: experimental dynamic KV requires VRAM or automatic without SSD");
    o.gpu_limit=setting("TYLOINFER_KV_GPU_LIMIT_BYTES",0);
    o.gpu_budget=setting("TYLOINFER_KV_GPU_BUDGET_BYTES",0);
    o.budget_policy=tyloi::cuda::env_lookup("TYLOINFER_KV_RAM_PREALLOCATED") || o.gpu_budget;
    o.ram_preallocated=setting("TYLOINFER_KV_RAM_PREALLOCATED",1)!=0;
    if(!o.ram_preallocated && o.placement!="auto_ssd")throw std::invalid_argument("kv_ssd_required: disabling RAM preallocation requires SSD KV");
    o.ram_limit_set=tyloi::cuda::env_lookup("TYLOINFER_KV_RAM_LIMIT_BYTES")!=nullptr;
    o.ram_limit=setting("TYLOINFER_KV_RAM_LIMIT_BYTES",0);
    o.auto_stream_threshold=int64_t(setting("TYLOINFER_KV_AUTO_STREAM_MIN_CONTEXT",65536));
    o.resident_cells=int64_t(setting("TYLOINFER_KV_RESIDENT_CELLS",32768));
    if (const auto* p=tyloi::cuda::env_lookup("TYLOINFER_RUNTIME_PREFIX_CACHE_DIR")) o.directory=p;
    return o;
}
KvMemoryManager::KvMemoryManager(engine::KvOptions options):options_(std::move(options)) {}
std::shared_ptr<KvMemoryAllocation> KvMemoryManager::add(int64_t maximum,int64_t heads,int64_t kw,int64_t vw,
        int64_t iw,int64_t pool,const ::tyloi::TensorOptions& options,int64_t selected_columns) {
    if (ready_) {
        auto policy=options_;policy.additional_request=true;
        if(policy.budget_policy && policy.gpu_budget)for(const auto& weak:allocations_)if(auto existing=weak.lock()) {
            const auto& layout = existing->layout;
            if (layout.maximum!=maximum || layout.heads!=heads || layout.key_width!=kw || layout.value_width!=vw ||
                layout.index_width!=iw || layout.pool!=pool || layout.dtype!=options.scalar_type() ||
                existing->selected_columns!=selected_columns || existing->options.target_device()!=options.target_device())
                continue;
            policy.gpu_budget=layout.index_bytes()+(existing->hot.exclusive()?
                std::size_t(existing->hot.keys.numel()+existing->hot.values.numel())*width_bytes(existing->options):
                existing->host || existing->offload?0:layout.raw_bytes());
            break;
        }
        // Independent graph sessions require directly addressable raw caches.
        // Admit extra GPU sessions against currently free memory.
        if(!automatic_ram_ && !policy.budget_policy && (policy.placement=="auto" || policy.placement=="auto_ssd"))policy.placement="vram";
        KvMemoryManager child(policy);
        child.bind_budget(budget_);
        auto a=child.add(maximum,heads,kw,vw,iw,pool,options,selected_columns);
        child.finalize(*cache_);allocations_.push_back(a);return a;
    }
    auto a=std::make_shared<KvMemoryAllocation>();
    a->layout={maximum,heads,kw,vw,iw,pool,options.scalar_type()};a->options=options;
    a->selected_columns=selected_columns;
    allocations_.push_back(a);return a;
}
void KvMemoryManager::finalize(tyloi::weights::Cache& cache, std::size_t minimum_experts) {
    if (ready_) return;
    cache_=&cache;
    if(!options_.ram_preallocated && options_.placement!="auto_ssd")throw std::invalid_argument("kv_ssd_required: disabling RAM preallocation requires SSD KV");
    std::size_t raw=0,index=0;
    for (auto& weak:allocations_) if(auto a=weak.lock()){raw+=a->layout.raw_bytes();index+=a->layout.index_bytes();}
    if (!raw) {ready_=true;return;}
    const bool admission=budget_ && options_.additional_request;
    auto device=tyloi::Device(tyloi::kCUDA);
    for(const auto& weak:allocations_)if(auto a=weak.lock()) {device=a->options.target_device();break;}
    auto context=default_context(device.index);context->reclaim_idle_memory();
    const auto reserve=(options_.additional_request?std::size_t(256)<<20 : ::tyloi::cuda::options::workspace_reserve_bytes())+minimum_experts;
    // Long discrete-memory sessions keep canonical raw KV in RAM. The GPU
    // cache is bounded by resident cells; selected rows are resolved on-device
    // and the same fixed addresses remain valid in decode/MTP CUDA Graphs.
    int64_t maximum=0,maximum_cells=options_.resident_cells;
    std::size_t per_cell=0,fixed_metadata=0;
    for(auto& weak:allocations_)if(auto a=weak.lock()) {
        maximum=std::max(maximum,a->layout.maximum);maximum_cells=std::min(maximum_cells,a->layout.maximum);
        per_cell+=a->layout.row_bytes();
        fixed_metadata+=std::size_t(a->layout.maximum)*16+56;
    }
    const bool discrete=separate_memory(device.index);
    const bool tiered_budget=options_.budget_policy && discrete;
    const auto initial_free=context->available_memory();
    if(!admission && (initial_free<reserve || index>initial_free-reserve))throw std::runtime_error(insufficient);
    std::size_t initial_mapped=0;
    if(admission && options_.dynamic) {
        if(tiered_budget && options_.gpu_budget && options_.gpu_budget<index)
            throw std::runtime_error("kv_budget_below_indexer: VRAM KV budget must cover the complete target indexer cache");
        if(!vmm::vmm_available())throw std::runtime_error("kv_dynamic_unavailable: CUDA virtual memory management is unavailable");
        for(const auto& weak:allocations_)if(auto a=weak.lock())
            initial_mapped=engine::prefill_budget_add(initial_mapped,a->initial_bytes());
    }
    // A small target context must not shrink staging below the selected KV
    // workspace. That would turn an 8K prefill into thousands of per-row
    // gathers even though the fixed decode workspace can hold a whole layer.
    std::size_t stream_buffer=std::min<std::size_t>(64ULL<<20,std::max<std::size_t>(4096,raw/16));
    if(tiered_budget)for(const auto& weak:allocations_)if(auto a=weak.lock()) {
        stream_buffer=std::max(stream_buffer,std::size_t(16*a->selected_columns*a->layout.heads)*
            (a->layout.key_width+a->layout.value_width)*width_bytes(a->options));
    }
    const engine::KvDemand demand{
        .raw=raw, .index=index, .reserve=reserve, .per_cell=per_cell, .metadata=fixed_metadata,
        .initial_mapped=initial_mapped, .stream_buffer=stream_buffer,
        .allocations=std::size_t(std::count_if(allocations_.begin(),allocations_.end(),
            [](const auto& weak){return !weak.expired();})),
        .maximum=maximum, .cells=maximum_cells, .discrete=discrete,
        .dynamic_available=options_.dynamic && vmm::vmm_available(), .admission=admission};
    const auto* stream_graphs=tyloi::cuda::env_lookup("TYLOINFER_KV_STREAM_GRAPHS");
    const auto layout_bytes=[&](const engine::KvPlacement& placement) {
        const auto& offload_config=placement.offload;
        std::vector<std::pair<tyloi::Device,std::size_t>> required;
        const auto add=[&](tyloi::Device device,std::size_t bytes) {
            auto entry=std::find_if(required.begin(),required.end(),[&](const auto& item){return item.first==device;});
            if(entry==required.end())required.emplace_back(device,bytes);
            else entry->second=engine::prefill_budget_add(entry->second,bytes);
        };
        if(offload_config)add(device,offload_config->gpu_hot_bytes());
        std::set<std::tuple<int64_t,int64_t,int,int64_t,int64_t>> gather_layouts;
        for(auto& weak:allocations_)if(auto a=weak.lock()) {
            auto bytes=a->layout.index_bytes();
            if(options_.dynamic) {
                bytes=engine::prefill_budget_add(bytes,a->initial_bytes());
            } else if(placement.exclusive || placement.automatic_ram) {
                bytes=engine::prefill_budget_add(bytes,HotKvCache::layout(a->layout.heads,a->layout.maximum,placement.cells,
                    a->layout.key_width,a->layout.value_width,a->options,placement.exclusive).gpu_bytes());
            } else if(options_.placement!="ram" && !placement.raw_host_only && !offload_config) {
                bytes=engine::prefill_budget_add(bytes,a->layout.raw_bytes());
            }
            add(a->options.target_device(),bytes);
            if(offload_config && (!stream_graphs || stream_graphs[0]!='0') && a->selected_columns) {
                const auto rows=8*a->selected_columns,width=a->layout.heads*(a->layout.key_width+a->layout.value_width);
                const auto key=std::make_tuple(rows,width,int(a->options.scalar_type()),a->layout.heads,a->layout.key_width);
                if(gather_layouts.insert(key).second)
                    add(device,state::KvOffloadStore::gather_layout(rows,width,
                        a->options.scalar_type(),a->layout.heads,a->layout.key_width).gpu_bytes());
            }
        }
        return required;
    };
    if(admission) {
        const auto count=std::count_if(allocations_.begin(),allocations_.end(),[](const auto& weak){return !weak.expired();});
        auto capacity=engine::prefill_budget_add(std::max(raw,initial_mapped),fixed_metadata);
        capacity=engine::prefill_budget_add(capacity,std::size_t(std::max<int64_t>(0,maximum_cells))*std::size_t(count)*12);
        const auto candidate=engine::Budget::place_kv(options_,demand,engine::prefill_budget_add(engine::prefill_budget_add(reserve,index),capacity),cache);
        for(const auto& [device,bytes]:layout_bytes(candidate))
            budget_->reclaim_kv(device,engine::prefill_budget_add(bytes,reserve));
    }
    auto available=admission?context->available_memory():initial_free;
    auto selected=engine::Budget::place_kv(options_,demand,available,cache);
    while(admission && (selected.exclusive || selected.automatic_ram || selected.offload)) {
        std::size_t shortage=0;
        for(const auto& [device,bytes]:layout_bytes(selected)) {
            const auto needed=engine::prefill_budget_add(bytes,reserve);
            const auto free=default_context(device.index)->available_memory();
            if(needed>free)shortage=std::max(shortage,needed-free);
        }
        if(!shortage || available<=reserve || index>=available-reserve)break;
        const auto headroom=available-reserve-index;
        available-=std::min(shortage,headroom);
        selected=engine::Budget::place_kv(options_,demand,available,cache);
    }
    if(budget_)for(const auto& [device,bytes]:layout_bytes(selected))
        budget_->reserve_kv(device,engine::prefill_budget_add(bytes,reserve));
    automatic_ram_=selected.automatic_ram;
    const auto cells=selected.cells;
    const auto gpu=selected.gpu;
    auto& offload_config=selected.offload;
    if(selected.disk_needed)std::cerr<<"runtime_kv_warning code=kv_ssd_overflow target_spill_bytes="<<*selected.disk_needed
        <<" message=KV may spill to SSD; long-context latency and disk usage will increase\n";
    if(offload_config) {
        offload_config->device=device;
        offload_=std::make_shared<state::KvOffloadStore>(std::move(*offload_config));
    }
    ResidentAllocationScope resident;
    if(tiered_budget && !options_.additional_request && !options_.dynamic && setting("TYLOINFER_KV_EXPERT_LOAN",1)) {
        cache_->prepare_kv();
        lending_=true;
    }
    for(auto& weak:allocations_)if(auto allocation=weak.lock())
        allocation->materialize(options_,selected,offload_,lending_?cache_:nullptr);
    if(offload_ && lending_)offload_->lend(*cache_);
    // Reserve shared decode I/O buffers before sizing the expert arenas.
    // Graph capture must not discover a large workspace after admission.
    if(offload_ && (!stream_graphs || stream_graphs[0]!='0'))for(auto& weak:allocations_)if(auto a=weak.lock()) {
        if(a->selected_columns)offload_->gather_workspace(8*a->selected_columns,
            a->layout.heads*(a->layout.key_width+a->layout.value_width),a->options.scalar_type(),a->layout.heads,a->layout.key_width);
    }
    ready_=true;
    gpu_budget_=index+std::min(raw,gpu);ram_required_=raw-std::min(raw,gpu);
    std::cerr<<"runtime_kv_memory placement="<<options_.placement<<" allocation="<<(options_.dynamic?"dynamic":"preallocated")
        <<" raw_bytes="<<raw<<" index_bytes="<<index<<" streaming="<<bool(offload_)<<'\n';
    if(automatic_ram_)std::cerr<<"runtime_kv_ram_resident cells="<<cells<<" maximum="<<maximum<<" raw_ram_bytes="<<raw<<" graph=1\n";
    if(tiered_budget)std::cerr<<"runtime_kv_tier_budget gpu_bytes="<<index+gpu<<" raw_ram_required="<<raw-std::min(raw,gpu)
        <<" ram_preallocated="<<options_.ram_preallocated<<" exclusive="<<selected.exclusive<<"\n";
}
std::vector<std::pair<tyloi::Device, std::size_t>> KvMemoryManager::growth(int64_t tokens) const {
    if (tokens < 0) throw std::invalid_argument("KV growth position must be nonnegative");
    std::vector<std::pair<tyloi::Device, std::size_t>> result;
    for (const auto& weak : allocations_) if (auto allocation = weak.lock()) {
        const auto bytes = allocation->growth_bytes(std::min(allocation->layout.maximum, tokens));
        if (!bytes) continue;
        const auto device = allocation->options.target_device();
        auto entry = std::find_if(result.begin(), result.end(), [&](const auto& item) { return item.first == device; });
        if (entry == result.end()) result.emplace_back(device, bytes);
        else entry->second = engine::prefill_budget_add(entry->second, bytes);
    }
    return result;
}
void KvMemoryManager::ensure(int64_t tokens) {
    if (lending_) cache_->protect_kv(tokens);
    const auto required = growth(tokens);
    if (budget_) for (const auto& [device, bytes] : required)
        budget_->reserve_kv(device, engine::prefill_budget_add(bytes, 64ULL << 20));
    for (auto& weak : allocations_) if (auto allocation = weak.lock())
        allocation->ensure(std::min(allocation->layout.maximum, tokens));
}
std::size_t KvMemoryManager::growth_reserve() const {
    std::size_t result=0;
    for(auto& weak:allocations_)if(auto a=weak.lock())result+=a->growth_bytes(a->layout.maximum);
    return result;
}
std::size_t KvMemoryManager::expert_budget_charge() const {
    std::size_t bytes=growth_reserve();
    for(auto& weak:allocations_)if(auto a=weak.lock())bytes+=a->gpu_bytes();
    if(offload_)bytes+=offload_->config().gpu_budget_bytes+
        (offload_->graph_workspace_bytes()>offload_->config().buffer_bytes?offload_->graph_workspace_bytes()-offload_->config().buffer_bytes:0);
    return bytes;
}
std::vector<std::pair<std::string,double>> KvMemoryManager::metrics() const {
    std::size_t gpu=0,ram=0,capacity=0,vmm=0;
    int64_t resident_cells=0;
    const auto loan=lending_?cache_->kv_metrics():tyloi::weights::Cache::KvMetrics{};
    for(auto& weak:allocations_)if(auto a=weak.lock()){
        gpu+=a->gpu_bytes();ram+=a->ram_bytes();capacity+=a->layout.raw_bytes()+a->layout.index_bytes();
        if(a->virtual_keys_)vmm+=a->gpu_bytes()-a->layout.index_bytes();
        if(a->hot.defined())resident_cells=resident_cells?std::min(resident_cells,a->hot.keys.size(2)):a->hot.keys.size(2);
    }
    return {{"kv_memory_supported",capacity?1.0:0.0},{"kv_memory_dynamic",options_.dynamic?1.0:0.0},
        {"kv_memory_preallocated",options_.dynamic?0.0:1.0},{"kv_memory_capacity_bytes",double(capacity)},
        {"kv_memory_gpu_bytes",double(gpu+(offload_?offload_->lent_bytes():0)-loan.bytes)},{"kv_memory_ram_bytes",double(ram)},
        {"kv_memory_expert_loan_bytes",double(loan.bytes)},
        {"kv_memory_expert_loan_hit_bytes",double(loan.hit_bytes)},
        {"kv_memory_expert_loan_lookup_us",loan.lookup_us},
        {"kv_memory_gpu_budget_bytes",double(gpu_budget_)},{"kv_memory_cold_required_bytes",double(ram_required_)},
        {"kv_memory_ram_preallocated",options_.ram_preallocated?1.0:0.0},
        {"kv_memory_vmm_bytes",double(vmm)}, {"kv_memory_auto_ram",automatic_ram_?1.0:0.0},
        {"kv_memory_resident_cells",double(resident_cells)},
        {"kv_memory_ssd_enabled",options_.placement=="auto_ssd"?1.0:0.0}};
}
}
