#include "engine/state/kv/offload.h"
#include "engine/state/kv/qsa_blocks.h"
#include "engine/memory/budget.h"
#include "engine/executor.h"
#include "engine/ops/ops.h"
#include "engine/weights/cache/cache.h"
#include "engine/weights/cache/workspace.h"

#include <algorithm>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <limits>
#include <tuple>
namespace tyloi::state {
namespace {
std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("CUDA KV gather size overflow");
    return a * b;
}
std::size_t checked_add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error("CUDA KV gather size overflow");
    return a + b;
}
constexpr auto no_offset = std::numeric_limits<std::size_t>::max();
struct Slots {
    std::map<std::size_t,std::size_t> free;
    explicit Slots(std::size_t bytes=0) {if(bytes)free[0]=bytes;}
    std::size_t take(std::size_t bytes) {
        for(auto it=free.begin();it!=free.end();++it)if(it->second>=bytes) {
            const auto offset=it->first,remaining=it->second-bytes;
            free.erase(it);if(remaining)free[offset+bytes]=remaining;return offset;
        }
        return no_offset;
    }
    void give(std::size_t offset,std::size_t bytes) {
        if(offset==no_offset)return;
        auto next=free.lower_bound(offset);
        if(next!=free.begin()) {auto prior=std::prev(next);if(prior->first+prior->second==offset){offset=prior->first;bytes+=prior->second;free.erase(prior);}}
        next=free.lower_bound(offset);
        if(next!=free.end() && offset+bytes==next->first){bytes+=next->second;free.erase(next);}
        free[offset]=bytes;
    }
};

}

struct KvOffloadStore::Block {
    std::uint64_t id = 0;
    std::size_t bytes = 0;
    ::tyloi::Tensor gpu;
    ::tyloi::TensorOptions options;
    int64_t rows = 0, width = 0;
    cache::QsaKvStore::BlockPtr cold;
    std::size_t gpu_offset = no_offset, ram_offset = no_offset;
};

struct KvOffloadStore::Impl {
    tyloi::engine::Budget* budget = nullptr;
    explicit Impl(state::KvOffloadConfig value, ::tyloi::Executor* owner) : config(std::move(value)), executor(owner) {
        if ((!config.device.is_cuda() || config.device.index < 0) &&
            (!config.device.is_metal() || config.device.index != 0))
            throw std::invalid_argument("KV offload requires CUDA or Metal 0");
        if (config.device.is_metal() && (!executor || config.preallocate))
            throw std::invalid_argument("Metal KV offload requires an executor and non-preallocated storage");
#if !TYLOINFER_HAS_CUDA
        if (config.device.is_cuda()) throw std::invalid_argument("CUDA KV offload is unavailable");
#endif
#if !TYLOINFER_HAS_METAL
        if (config.device.is_metal()) throw std::invalid_argument("Metal KV offload is unavailable");
#endif
        if (config.buffer_bytes < 4096 || config.gpu_budget_bytes < config.buffer_bytes)
            throw std::invalid_argument("CUDA KV budget cannot hold its I/O staging buffer");
        if(config.preallocate) {
            weights::Resident resident;
            gpu_arena=ops::empty({int64_t(hot_limit())},TensorOptions().device(config.device).dtype(kUInt8));
            gpu_slots=Slots(hot_limit());
            if(config.ram_preallocated && config.ram_budget_bytes)
                ram_arena=ops::empty({int64_t(config.ram_budget_bytes)},TensorOptions().device(kCPU).dtype(kUInt8));
            ram_slots=Slots(config.ram_preallocated?config.ram_budget_bytes:0);
            if(!config.ram_preallocated)initialize_disk();
        } else initialize_disk();
    }
    void initialize_disk() {
        if(ram)return;
        if(!config.ssd_enabled)throw std::runtime_error("kv_memory_insufficient: available VRAM and RAM cannot hold the configured KV cache; choose automatic with SSD KV or reduce context length");
        cache::QsaKvStoreConfig cold;
        cold.buffer_bytes = config.buffer_bytes;
        cold.budget_bytes = (config.preallocate && config.ram_preallocated?0:config.ram_budget_bytes) + config.buffer_bytes;
        cold.directory = config.directory;
        ram = std::make_shared<cache::QsaKvStore>(std::move(cold));
    }
    std::size_t hot_limit() const { return config.gpu_hot_bytes(); }
    void spill_ram() {
        while(!ram_lru.empty()) {
            auto weak=ram_lru.back();auto block=weak.lock();
            if(!block || block->ram_offset==no_offset){ram_lru.pop_back();continue;}
            initialize_disk();
            if(!block->cold) {
                std::vector<std::uint8_t> payload(block->bytes);
                std::memcpy(payload.data(),ram_arena.dense().data_ptr<uint8_t>()+block->ram_offset,block->bytes);
                block->cold=ram->write(std::move(payload));
            }
            ram_lru.pop_back();ram_slots.give(block->ram_offset,block->bytes);block->ram_offset=no_offset;
            ram_bytes-=block->bytes;return;
        }
        throw std::runtime_error("kv_memory_insufficient: no free RAM KV blocks");
    }
    void store_cold(const BlockPtr& block) {
        if(block->ram_offset!=no_offset || block->cold)return;
        if(config.preallocate && config.ram_preallocated && block->bytes<=config.ram_budget_bytes) {
            auto offset=ram_slots.take(block->bytes);
            while(offset==no_offset){spill_ram();offset=ram_slots.take(block->bytes);}
            auto target=ops::storage_view(ram_arena,offset,{block->rows,block->width},block->options.scalar_type());
            ops::copy_into(target,block->gpu);
            block->ram_offset=offset;ram_bytes+=block->bytes;ram_lru.push_front(block);
        } else {
            initialize_disk();
            if(budget)budget->reserve_ram(block->bytes*2);
            auto cpu=ops::contiguous(block->gpu.relocated(Device{DeviceType::cpu, 0}));
            std::vector<std::uint8_t> payload(block->bytes);
            std::memcpy(payload.data(),cpu.dense().data_ptr(),payload.size());
            block->cold=ram->write(std::move(payload));
        }
        d2h_bytes+=block->bytes;
    }
    bool evict_one() {
        while(!lru.empty()) {
            auto found=hot.find(lru.back());auto block=found->second.block.lock();
            if(block) {
                store_cold(block);
                gpu_slots.give(block->gpu_offset,block->bytes);block->gpu_offset=no_offset;
                block->gpu={};gpu_bytes-=block->bytes;++evictions;
            }
            lru.pop_back();hot.erase(found);if(block)return true;
        }
        return false;
    }
    Tensor allocate_gpu(const BlockPtr& block) {
        auto offset=gpu_slots.take(block->bytes);
        while(offset==no_offset && evict_one())offset=gpu_slots.take(block->bytes);
        if(offset==no_offset)throw std::runtime_error("CUDA KV arena cannot hold one block");
        loan_region.protect(int64_t(offset+block->bytes));
        block->gpu_offset=offset;
        return ops::storage_view(gpu_arena,offset,{block->rows,block->width},block->options.scalar_type());
    }
    void touch(const BlockPtr& block) {
        auto found = hot.find(block->id);
        if (found != hot.end()) lru.splice(lru.begin(), lru, found->second.order);
        else {
            lru.push_front(block->id);
            hot.emplace(block->id, Hot{block, lru.begin()});
            gpu_bytes += block->bytes;
        }
    }
    void evict() {
        if(config.preallocate){while(gpu_bytes>hot_limit())if(!evict_one())break;return;}
        while (gpu_bytes > hot_limit() && !lru.empty()) {
            const auto id = lru.back();
            auto found = hot.find(id);
            auto block = found->second.block.lock();
            if (block) {
                if (!block->cold) {
                    if (budget) budget->reserve_ram(block->bytes * 2);
                    Tensor cpu=ops::contiguous(block->gpu.relocated(Device{DeviceType::cpu, 0}));
                    std::vector<std::uint8_t> payload(block->bytes);
                    std::memcpy(payload.data(), cpu.dense().data_ptr(), payload.size());
                    block->cold = ram->write(std::move(payload));
                    d2h_bytes += block->bytes;
                }
                block->gpu = {};
                gpu_bytes -= block->bytes;
                ++evictions;
            }
            lru.pop_back(); hot.erase(found);
        }
    }
    void release(Block* block) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        auto found = hot.find(block->id);
        if (found != hot.end()) {
            gpu_bytes -= block->bytes;
            lru.erase(found->second.order); hot.erase(found);
        }
        if(config.preallocate) {
            gpu_slots.give(block->gpu_offset,block->bytes);
            if(block->ram_offset!=no_offset){ram_slots.give(block->ram_offset,block->bytes);ram_bytes-=block->bytes;}
        }
    }
    struct Hot { std::weak_ptr<Block> block; std::list<std::uint64_t>::iterator order; };
    state::KvOffloadConfig config;
    ::tyloi::Executor* executor = nullptr;
    std::shared_ptr<cache::QsaKvStore> ram;
    Tensor gpu_arena;
    ::tyloi::weights::KvRegion loan_region;
    Tensor ram_arena;
    Slots gpu_slots,ram_slots;
    std::list<std::weak_ptr<Block>> ram_lru;
    std::size_t ram_bytes=0;
    std::uint64_t ram_hits=0;
    mutable std::recursive_mutex mutex;
    std::list<std::uint64_t> lru;
    std::unordered_map<std::uint64_t, Hot> hot;
    std::map<std::tuple<int64_t,int64_t,int,int64_t,int64_t>,std::shared_ptr<ops::KvGatherWorkspace>> gather_workspaces;
    std::size_t gpu_bytes = 0;
    std::uint64_t next_id = 0, hits = 0, h2d_bytes = 0, d2h_bytes = 0, evictions = 0;
};

KvOffloadStore::KvOffloadStore(state::KvOffloadConfig config, ::tyloi::Executor* executor) : impl_(std::make_shared<Impl>(std::move(config), executor)) {}
KvOffloadStore::~KvOffloadStore() = default;
KvOffloadStore::GatherLayout KvOffloadStore::gather_layout(int64_t rows,int64_t width,::tyloi::ScalarType dtype,int64_t heads,int64_t key_width) {
    if(rows<=0 || width<=0 || (dtype!=kFloat16 && dtype!=kInt16 && dtype!=kInt32) ||
       heads<=0 || width%heads || key_width<0 || (key_width && key_width>=width/heads))
        throw std::invalid_argument("invalid CUDA KV gather geometry");
    constexpr auto max_descriptors=(std::numeric_limits<int64_t>::max()-4)/4;
    if(rows>std::numeric_limits<int64_t>::max()/width || rows>max_descriptors ||
       (key_width && heads>max_descriptors/rows/2))
        throw std::overflow_error("CUDA KV gather size overflow");
    GatherLayout layout{rows,width,4+rows*(key_width?2*heads:1)*4,dtype};
    (void)layout.gpu_bytes();
    return layout;
}
std::size_t KvOffloadStore::GatherLayout::gpu_bytes() const {
    const auto payload=checked_multiply(checked_multiply(std::size_t(rows),std::size_t(width)),::tyloi::scalar_size(dtype));
    return checked_add(checked_multiply(payload,2),checked_multiply(std::size_t(descriptors),sizeof(int64_t)));
}
std::shared_ptr<ops::KvGatherWorkspace> KvOffloadStore::gather_workspace(int64_t rows,int64_t width,ScalarType dtype,int64_t heads,int64_t key_width) {
    if (impl_->config.device.is_metal()) throw std::runtime_error("Metal KV fixed gather is unsupported");
    const auto plan=gather_layout(rows,width,dtype,heads,key_width);
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    auto& workspace=impl_->gather_workspaces[{rows,width,int(dtype),heads,key_width}];
    if(!workspace) {
        weights::Resident resident;
        auto fresh=std::make_shared<ops::KvGatherWorkspace>();
        fresh->heads=heads;fresh->key_width=key_width;
        const auto gpu=TensorOptions().device(impl_->config.device).dtype(dtype);
        const auto host=TensorOptions().device(kCPU).dtype(dtype).pinned_memory(true);
        fresh->rows=ops::zeros({plan.rows,plan.width},gpu);
        fresh->upload=ops::empty({plan.rows,plan.width},gpu);
        fresh->host=ops::empty({plan.rows,plan.width},host);
        fresh->descriptors=ops::zeros({plan.descriptors},TensorOptions().device(kCPU).dtype(kInt64).pinned_memory(true));
        fresh->device_descriptors=ops::zeros({plan.descriptors},TensorOptions().device(impl_->config.device).dtype(kInt64));
        workspace=std::move(fresh);
    }
    return workspace;
}
std::size_t KvOffloadStore::graph_workspace_bytes() const {
    if (impl_->config.device.is_metal()) return 0;
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    std::size_t bytes=0;
    for(const auto& [key,work]:impl_->gather_workspaces)bytes+=work->rows.nbytes()+work->upload.nbytes()+work->device_descriptors.nbytes();
    return bytes;
}
const state::KvOffloadConfig& KvOffloadStore::config() const { return impl_->config; }

KvOffloadStore::BlockPtr KvOffloadStore::write(const ::tyloi::Tensor& input) {
    const bool metal = impl_->config.device.is_metal();
    const auto type = input.scalar_type();
    const bool valid_type = type == kFloat16 ||
        (metal ? (type == ScalarType::bfloat16 || type == ScalarType::float32 || type == ScalarType::uint32)
               : (type == kInt16 || type == kInt32));
    if (input.device() != impl_->config.device || input.is_encoded() ||
        input.dim() != 2 || !input.numel() || !valid_type)
        throw std::invalid_argument("KV microblock must contain nonempty supported GPU rows");
    const auto bytes = input.nbytes();
    if (bytes > impl_->config.buffer_bytes / 2)
        throw std::invalid_argument("CUDA KV microblock exceeds its write staging buffer");
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    std::weak_ptr<Impl> owner = impl_;
    auto block = BlockPtr(new Block, [owner](Block* value) {
        if (auto impl = owner.lock()) impl->release(value);
        delete value;
    });
    block->id = ++impl_->next_id;
    block->bytes = bytes; block->rows = input.size(0); block->width = input.size(1);
    block->options = input.options();
    if (metal) {
        if (impl_->budget) impl_->budget->reserve_kv(impl_->config.device, bytes);
        block->gpu = ::tyloi::ops::clone(input);
        impl_->executor->submit(std::span<const Tensor>(&block->gpu, 1));
    } else {
        auto rows=ops::contiguous(input);
        if(impl_->config.preallocate && bytes>impl_->hot_limit()) {
            block->gpu=rows;impl_->store_cold(block);block->gpu={};++impl_->evictions;
            return block;
        }
        if(impl_->config.preallocate) {
            block->gpu=impl_->allocate_gpu(block);ops::copy_into(block->gpu,rows);
        } else block->gpu=ops::clone(rows);
    }
    impl_->touch(block); impl_->evict();
    return block;
}

Tensor KvOffloadStore::read(const BlockPtr& block) {
    if (!block) throw std::invalid_argument("missing CUDA KV block");
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    if (block->gpu.defined()) {
        ++impl_->hits; impl_->touch(block);
        if (!impl_->config.device.is_metal()) return Tensor{block->gpu.dense()};
        return block->gpu;
    }
    if(impl_->config.preallocate) {
        auto result=ops::empty({block->rows,block->width},block->options);
        if(block->ram_offset!=no_offset) {
            auto source=ops::storage_view(impl_->ram_arena,block->ram_offset,{block->rows,block->width},block->options.scalar_type());
            ops::copy_into(result,source);
            impl_->ram_lru.remove_if([&](const auto& weak){auto value=weak.lock();return !value || value==block;});
            impl_->ram_lru.push_front(block);++impl_->ram_hits;
        } else {
            if(!impl_->ram || !block->cold)throw std::runtime_error("missing spilled KV block");
            auto payload=impl_->ram->read(block->cold);
            auto cpu_options=block->options;
            cpu_options.device(kCPU).pinned_memory(false);
            auto source=ops::from_blob(const_cast<uint8_t*>(payload->data()),{block->rows,block->width},cpu_options);
            ops::copy_into(result,source);
        }
        impl_->h2d_bytes+=block->bytes;return result;
    }
    if (impl_->budget) impl_->budget->reserve_ram(block->bytes * 2);
    auto payload = impl_->ram->read(block->cold);
    if (impl_->config.device.is_metal() && impl_->budget) impl_->budget->reserve_kv(impl_->config.device, block->bytes);
    auto cpu_options=block->options;
    cpu_options.device(kCPU);
    auto cpu=ops::empty({block->rows,block->width},cpu_options);
    std::memcpy(cpu.dense().data_ptr(),payload->data(),block->bytes);
    auto gpu=cpu.relocated(impl_->config.device);
    impl_->h2d_bytes += block->bytes;
    if (block->bytes <= impl_->hot_limit()) {
        block->gpu=impl_->config.device.is_metal() ? gpu : Tensor{gpu.dense()};
        impl_->touch(block); impl_->evict();
    }
    return gpu;
}

void KvOffloadStore::flush() { if(impl_->ram)impl_->ram->flush(); }
void KvOffloadStore::bind_budget(tyloi::engine::Budget* budget) {
    impl_->budget = budget;
}
void KvOffloadStore::lend(tyloi::weights::Cache& cache) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    impl_->loan_region.reset();
    if(!impl_->gpu_arena.defined() || !impl_->gpu_arena.nbytes())return;
    impl_->loan_region=cache.lend_kv(impl_->gpu_arena);
}
std::size_t KvOffloadStore::lent_bytes() const {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    return impl_->loan_region.bytes();
}
std::vector<std::pair<std::string, double>> KvOffloadStore::metrics() const {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    const auto cold = impl_->ram ? impl_->ram->stats() : cache::QsaKvStoreStats{};
    auto ram_bytes = cold.hot_bytes;
    if (impl_->config.preallocate && impl_->config.ram_preallocated) ram_bytes = impl_->ram_arena.nbytes();
    return {{"qsa_kv_offload_enabled", 1.0}, {"qsa_kv_ssd_enabled",impl_->config.ssd_enabled?1.0:0.0},
        {"qsa_kv_graph_workspace_bytes",double(graph_workspace_bytes())},
        {"qsa_kv_resident_bytes", double((impl_->config.preallocate?impl_->hot_limit():impl_->gpu_bytes)-lent_bytes())},
        {"qsa_kv_budget_bytes", double(impl_->config.gpu_budget_bytes)},
        {"qsa_kv_ram_enabled", impl_->config.ram_budget_bytes ? 1.0 : 0.0},
        {"qsa_kv_ram_bytes", double(ram_bytes)}, {"qsa_kv_ram_limit_bytes", double(impl_->config.ram_budget_bytes)},
        {"qsa_kv_ssd_bytes", double(cold.disk_bytes)}, {"qsa_kv_pending_bytes", double(cold.pending_bytes)},
        {"qsa_kv_ssd_reads", double(cold.reads)}, {"qsa_kv_ram_hits", double(cold.hits+impl_->ram_hits)},
        {"qsa_kv_gpu_hits", double(impl_->hits)}, {"qsa_kv_ssd_read_bytes", double(cold.read_bytes)},
        {"qsa_kv_ssd_written_bytes", double(cold.written_bytes)},
        {"qsa_kv_h2d_bytes", double(impl_->h2d_bytes)}, {"qsa_kv_d2h_bytes", double(impl_->d2h_bytes)},
        {"qsa_kv_gpu_evictions", double(impl_->evictions)}};
}

void KvOffloadSequence::gather_into(const std::vector<int64_t>& rows,const ops::KvGatherWorkspace& work) const {
    if (store_->config().device.is_metal()) throw std::runtime_error("Metal KV fixed gather is unsupported");
    if(rows.empty() || int64_t(rows.size())>work.rows.size(0) || work.rows.size(1)!=width_ || work.rows.scalar_type()!=dtype_)
        throw std::invalid_argument("invalid fixed CUDA KV gather workspace");
    auto& impl=*store_->impl_;
    std::lock_guard<std::recursive_mutex> lock(impl.mutex);
    std::map<int64_t,std::vector<std::pair<int64_t,std::size_t>>> groups;
    for(std::size_t i=0;i<rows.size();++i) {
        if(rows[i]<0 || rows[i]>=position_)throw std::out_of_range("CUDA KV selected row exceeds its sequence");
        groups[rows[i]/block_rows_].emplace_back(rows[i]%block_rows_,i);
    }
    std::vector<ops::KvGatherRow> gather_rows;
    gather_rows.reserve(rows.size());
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> payloads;
    for(const auto& [index,selected]:groups) {
        const auto& block=blocks_.at(index);
        Tensor source;
        if(block->gpu.defined()) {
            source=block->gpu;impl.touch(block);++impl.hits;
        } else if(block->ram_offset!=no_offset) {
            source=ops::storage_view(impl.ram_arena,block->ram_offset,{block->rows,block->width},dtype_);
            ++impl.ram_hits;
            impl.ram_lru.remove_if([&](const auto& weak){auto value=weak.lock();return !value || value==block;});
            impl.ram_lru.push_front(block);
        } else {
            if(!impl.ram || !block->cold)throw std::runtime_error("missing spilled KV block");
            auto payload=impl.ram->read(block->cold);
            auto cpu_options=block->options;
            cpu_options.device(kCPU).pinned_memory(false);
            source=ops::from_blob(const_cast<uint8_t*>(payload->data()),{block->rows,block->width},cpu_options);
            payloads.push_back(std::move(payload));
        }
        for(const auto& [row,output]:selected)
            gather_rows.push_back({source,row,int64_t(output)});
    }
    impl.h2d_bytes+=ops::gather_kv(gather_rows,work);
}

} // namespace tyloi::state
