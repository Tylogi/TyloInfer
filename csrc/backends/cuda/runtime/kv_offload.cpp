#include "kv_offload.h"
#include "kv_expert_loan.h"
#include "moe_pipeline.h"

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
#include "mfq_cuda_context.h"
#include "mfq_cuda_resident_allocations.h"

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;
namespace {
std::size_t kv_element_size(tb::ScalarType dtype) {
    if(dtype==tb::kFloat16 || dtype==tb::kInt16)return 2;
    return dtype==tb::kInt32?4:0;
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
tb::Tensor indices(const std::vector<int64_t>& values, const tb::TensorOptions& options) {
    auto cpu = tb::empty({static_cast<int64_t>(values.size())}, options.device(tb::kCPU).dtype(tb::kInt64));
    if (!values.empty()) std::memcpy(cpu.data_ptr(), values.data(), values.size() * sizeof(int64_t));
    return cpu.to(options.dtype(tb::kInt64));
}
}

struct KvOffloadStore::Block {
    std::uint64_t id = 0;
    std::size_t bytes = 0;
    tb::Tensor gpu;
    tb::TensorOptions options;
    int64_t rows = 0, width = 0;
    cache::QsaKvStore::BlockPtr cold;
    std::size_t gpu_offset = no_offset, ram_offset = no_offset;
    std::list<std::weak_ptr<Block>>::iterator ram_order;
};

struct KvOffloadStore::Impl {
    std::function<void(std::size_t)> reserve_ram;
    explicit Impl(KvOffloadConfig value) : config(std::move(value)) {
        if (config.buffer_bytes < 4096 || config.gpu_budget_bytes < config.buffer_bytes)
            throw std::invalid_argument("CUDA KV budget cannot hold its I/O staging buffer");
        if(config.preallocate) {
            ResidentAllocationScope resident;
            gpu_arena=tb::empty({int64_t(hot_limit())},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
            gpu_slots=Slots(hot_limit());
            if(config.ram_preallocated && config.ram_budget_bytes) {
                ram_arena.resize(config.ram_budget_bytes);
                charge_tensor_host_bytes(config.ram_budget_bytes);
            }
            ram_slots=Slots(config.ram_preallocated?config.ram_budget_bytes:0);
            if(!config.ram_preallocated)initialize_disk();
        } else initialize_disk();
    }
    ~Impl() {if(config.preallocate)tensor_host_bytes.fetch_sub(ram_arena.size());}
    void initialize_disk() {
        if(ram)return;
        if(!config.ssd_enabled)throw std::runtime_error("kv_memory_insufficient: available VRAM and RAM cannot hold the configured KV cache; choose automatic with SSD KV or reduce context length");
        cache::QsaKvStoreConfig cold;
        cold.buffer_bytes = config.buffer_bytes;
        cold.budget_bytes = (config.preallocate && config.ram_preallocated?0:config.ram_budget_bytes) + config.buffer_bytes;
        cold.directory = config.directory;
        ram = std::make_shared<cache::QsaKvStore>(std::move(cold));
    }
    std::size_t hot_limit() const { return config.gpu_budget_bytes - config.buffer_bytes; }
    BlockPtr make_block(const tb::Tensor& rows,tb::TensorOptions options,const std::shared_ptr<Impl>& self) {
        std::weak_ptr<Impl> owner=self;
        auto block=BlockPtr(new Block,[owner](Block* value) {
            if (auto state=owner.lock()) state->release(value);
            delete value;
        });
        block->id=++next_id;
        block->bytes=rows.nbytes();
        block->rows=rows.size(0);
        block->width=rows.size(1);
        block->options=std::move(options);
        return block;
    }
    void spill_ram() {
        while(!ram_lru.empty()) {
            auto weak=ram_lru.back();auto block=weak.lock();
            if(!block || block->ram_offset==no_offset){ram_lru.pop_back();continue;}
            initialize_disk();
            if(!block->cold) {
                std::vector<std::uint8_t> payload(block->bytes);
                std::memcpy(payload.data(),ram_arena.data()+block->ram_offset,block->bytes);
                block->cold=ram->write(std::move(payload));
            }
            ram_lru.pop_back();ram_slots.give(block->ram_offset,block->bytes);block->ram_offset=no_offset;
            ram_bytes-=block->bytes;return;
        }
        throw std::runtime_error("kv_memory_insufficient: no free RAM KV blocks");
    }
    void store_cold(const BlockPtr& block,const void* host_data=nullptr) {
        if(block->ram_offset!=no_offset || block->cold)return;
        if(config.preallocate && config.ram_preallocated && block->bytes<=config.ram_budget_bytes) {
            auto offset=ram_slots.take(block->bytes);
            while(offset==no_offset){spill_ram();offset=ram_slots.take(block->bytes);}
            try {
                if (host_data) std::memcpy(ram_arena.data()+offset,host_data,block->bytes);
                else {
                    TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(ram_arena.data()+offset,block->gpu.data_ptr(),block->bytes,cudaMemcpyDeviceToHost,current_stream().stream()));
                    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
                }
                ram_lru.push_front(block);
            } catch (...) {
                ram_slots.give(offset,block->bytes);
                throw;
            }
            block->ram_offset=offset;ram_bytes+=block->bytes;block->ram_order=ram_lru.begin();
        } else {
            initialize_disk();
            if(reserve_ram)reserve_ram(block->bytes*2);
            std::vector<std::uint8_t> payload(block->bytes);
            if (host_data) std::memcpy(payload.data(),host_data,payload.size());
            else {
                const auto cpu=block->gpu.to(tb::kCPU).contiguous();
                std::memcpy(payload.data(),cpu.data_ptr(),payload.size());
            }
            block->cold=ram->write(std::move(payload));
        }
        if (!host_data) d2h_bytes+=block->bytes;
    }
    void touch_ram(const BlockPtr& block) {
        ram_lru.splice(ram_lru.begin(),ram_lru,block->ram_order);
        ++ram_hits;
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
    tb::Tensor allocate_gpu(const BlockPtr& block) {
        auto offset=gpu_slots.take(block->bytes);
        while(offset==no_offset && evict_one())offset=gpu_slots.take(block->bytes);
        if(offset==no_offset)throw std::runtime_error("CUDA KV arena cannot hold one block");
        if(expert_loan)expert_loan->protect(loan_region,int64_t(offset+block->bytes));
        block->gpu_offset=offset;
        auto* data=static_cast<std::uint8_t*>(gpu_arena.data_ptr())+offset;
        auto storage=std::make_shared<TensorStorage>();
        storage->base=data;storage->bytes=block->bytes;storage->device=block->options.target_device();
        storage->owner=std::make_shared<tb::Tensor>(gpu_arena);
        const std::vector<int64_t> shape{block->rows,block->width};
        return Tensor(storage,make_contiguous_view(data,shape,block->options.scalar_type(),block->options.target_device()));
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
                    if (reserve_ram) reserve_ram(block->bytes * 2);
                    auto cpu = block->gpu.to(tb::kCPU).contiguous();
                    std::vector<std::uint8_t> payload(block->bytes);
                    std::memcpy(payload.data(), cpu.data_ptr(), payload.size());
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
            if(block->ram_offset!=no_offset){
                ram_lru.erase(block->ram_order);ram_slots.give(block->ram_offset,block->bytes);ram_bytes-=block->bytes;
            }
        }
    }
    struct Hot { std::weak_ptr<Block> block; std::list<std::uint64_t>::iterator order; };
    KvOffloadConfig config;
    std::shared_ptr<cache::QsaKvStore> ram;
    tb::Tensor gpu_arena;
    std::shared_ptr<KvExpertLoan> expert_loan;
    std::size_t loan_region=0;
    std::vector<std::uint8_t> ram_arena;
    Slots gpu_slots,ram_slots;
    std::list<std::weak_ptr<Block>> ram_lru;
    std::size_t ram_bytes=0;
    std::uint64_t ram_hits=0;
    mutable std::recursive_mutex mutex;
    std::list<std::uint64_t> lru;
    std::unordered_map<std::uint64_t, Hot> hot;
    std::map<std::tuple<int64_t,int64_t,int,int64_t,int64_t>,std::shared_ptr<GatherWorkspace>> gather_workspaces;
    std::size_t gpu_bytes = 0;
    std::uint64_t next_id = 0, hits = 0, h2d_bytes = 0, d2h_bytes = 0, evictions = 0;
};

KvOffloadStore::KvOffloadStore(KvOffloadConfig config) : impl_(std::make_shared<Impl>(std::move(config))) {}
KvOffloadStore::~KvOffloadStore() = default;
std::shared_ptr<KvOffloadStore::GatherWorkspace> KvOffloadStore::gather_workspace(int64_t rows,int64_t width,tb::ScalarType dtype,int64_t heads,int64_t key_width) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    if(rows<=0 || width<=0 || !kv_element_size(dtype) || heads<=0 || width%heads || key_width<0 || (key_width && key_width>=width/heads))
        throw std::invalid_argument("invalid CUDA KV gather geometry");
    auto& workspace=impl_->gather_workspaces[{rows,width,int(dtype),heads,key_width}];
    if(!workspace) {
        ResidentAllocationScope resident;
        auto fresh=std::make_shared<GatherWorkspace>();
        fresh->heads=heads;fresh->key_width=key_width;
        const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(dtype);
        fresh->rows=tb::zeros({rows,width},options);
        fresh->host=tb::empty({rows,width},options.device(tb::kCPU).pinned_memory(true));
        if(key_width || impl_->hot_limit()) {
            fresh->upload=tb::empty({rows,width},options);
            const auto fields=key_width?2*heads:1;
            fresh->descriptors=tb::zeros({4+rows*fields*4},options.device(tb::kCPU).dtype(tb::kInt64).pinned_memory(true));
            fresh->device_descriptors=tb::zeros({4+rows*fields*4},options.dtype(tb::kInt64));
        }
        workspace=std::move(fresh);
    }
    return workspace;
}
std::size_t KvOffloadStore::graph_workspace_bytes() const {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    std::size_t bytes=0;
    for(const auto& [key,work]:impl_->gather_workspaces) {
        bytes+=work->rows.nbytes();
        if(work->upload.defined())bytes+=work->upload.nbytes()+work->device_descriptors.nbytes();
    }
    return bytes;
}
const KvOffloadConfig& KvOffloadStore::config() const { return impl_->config; }

KvOffloadStore::BlockPtr KvOffloadStore::write(const tb::Tensor& rows) {
    if (!rows.is_cuda() || rows.dim() != 2 || !rows.numel() || !kv_element_size(rows.scalar_type()))
        throw std::invalid_argument("CUDA KV microblock must contain nonempty FP16, INT16 or packed INT32 rows");
    const auto bytes = static_cast<std::size_t>(rows.numel()) * rows.element_size();
    if (bytes > impl_->config.buffer_bytes / 2)
        throw std::invalid_argument("CUDA KV microblock exceeds its write staging buffer");
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    auto block=impl_->make_block(rows,rows.options(),impl_);
    if(impl_->config.preallocate && bytes>impl_->hot_limit()) {
        block->gpu=rows.contiguous();impl_->store_cold(block);block->gpu={};++impl_->evictions;
        return block;
    }
    if(impl_->config.preallocate) {
        block->gpu=impl_->allocate_gpu(block);block->gpu.copy_(rows);
        // A slot can be reused by another write on this same stream.
    } else block->gpu = rows.contiguous().clone();
    impl_->touch(block); impl_->evict();
    return block;
}

tb::Tensor KvOffloadStore::read(const BlockPtr& block) {
    if (!block) throw std::invalid_argument("missing CUDA KV block");
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    if (block->gpu.defined()) { ++impl_->hits; impl_->touch(block); return block->gpu; }
    if(impl_->config.preallocate) {
        // Keep read results outside the eviction arena: a gather may retain
        // this tensor while subsequent reads recycle other arena slots.
        auto result=tb::empty({block->rows,block->width},block->options);
        if(block->ram_offset!=no_offset) {
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(result.data_ptr(),impl_->ram_arena.data()+block->ram_offset,block->bytes,cudaMemcpyHostToDevice,current_stream().stream()));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
            impl_->touch_ram(block);
        } else {
            if(!impl_->ram || !block->cold)throw std::runtime_error("missing spilled KV block");
            auto payload=impl_->ram->read(block->cold);
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(result.data_ptr(),payload->data(),block->bytes,cudaMemcpyHostToDevice,current_stream().stream()));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
        }
        impl_->h2d_bytes+=block->bytes;return result;
    }
    if (impl_->reserve_ram) impl_->reserve_ram(block->bytes * 2);
    auto payload = impl_->ram->read(block->cold);
    const auto options = block->options;
    auto cpu = tb::empty({block->rows, block->width}, options.device(tb::kCPU));
    std::memcpy(cpu.data_ptr(), payload->data(), block->bytes);
    auto gpu = cpu.to(block->options);
    impl_->h2d_bytes += block->bytes;
    if (block->bytes <= impl_->hot_limit()) {
        block->gpu = gpu; impl_->touch(block); impl_->evict();
    }
    return gpu;
}

void KvOffloadStore::flush() { if(impl_->ram)impl_->ram->flush(); }
void KvOffloadStore::set_ram_reservation(std::function<void(std::size_t)> reserve) {
    impl_->reserve_ram = std::move(reserve);
}
void KvOffloadStore::set_expert_loan(std::shared_ptr<KvExpertLoan> loan) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    if(!loan || !impl_->gpu_arena.defined() || !impl_->gpu_arena.nbytes())return;
    impl_->loan_region=loan->add(impl_->gpu_arena,int64_t(impl_->gpu_arena.nbytes()),1,false,1,true);
    impl_->expert_loan=std::move(loan);
}
std::size_t KvOffloadStore::expert_loan_bytes() const {
    return impl_->expert_loan?impl_->expert_loan->region_bytes(impl_->loan_region):0;
}
std::vector<std::pair<std::string, double>> KvOffloadStore::metrics() const {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    const auto cold = impl_->ram ? impl_->ram->stats() : cache::QsaKvStoreStats{};
    return {{"qsa_kv_offload_enabled", 1.0}, {"qsa_kv_ssd_enabled",impl_->config.ssd_enabled?1.0:0.0},
        {"qsa_kv_graph_workspace_bytes",double(graph_workspace_bytes())},
        {"qsa_kv_resident_bytes", double((impl_->config.preallocate?impl_->hot_limit():impl_->gpu_bytes)-expert_loan_bytes())},
        {"qsa_kv_budget_bytes", double(impl_->config.gpu_budget_bytes)},
        {"qsa_kv_ram_enabled", impl_->config.ram_budget_bytes ? 1.0 : 0.0},
        {"qsa_kv_ram_bytes", double(impl_->config.preallocate && impl_->config.ram_preallocated?impl_->ram_arena.size():cold.hot_bytes)}, {"qsa_kv_ram_limit_bytes", double(impl_->config.ram_budget_bytes)},
        {"qsa_kv_ssd_bytes", double(cold.disk_bytes)}, {"qsa_kv_pending_bytes", double(cold.pending_bytes)},
        {"qsa_kv_ssd_reads", double(cold.reads)}, {"qsa_kv_ram_hits", double(cold.hits+impl_->ram_hits)},
        {"qsa_kv_gpu_hits", double(impl_->hits)}, {"qsa_kv_ssd_read_bytes", double(cold.read_bytes)},
        {"qsa_kv_ssd_written_bytes", double(cold.written_bytes)},
        {"qsa_kv_h2d_bytes", double(impl_->h2d_bytes)}, {"qsa_kv_d2h_bytes", double(impl_->d2h_bytes)},
        {"qsa_kv_gpu_evictions", double(impl_->evictions)}};
}

KvOffloadSequence::KvOffloadSequence(std::shared_ptr<KvOffloadStore> store, int64_t width, int64_t block_rows, tb::ScalarType dtype)
    : store_(std::move(store)), width_(width), block_rows_(block_rows), dtype_(dtype) {
    const auto bytes=kv_element_size(dtype);
    if (!store_ || width <= 0 || block_rows <= 0 || !bytes ||
        std::uint64_t(width) > std::uint64_t(store_->config().buffer_bytes/2)/bytes/std::uint64_t(block_rows))
        throw std::invalid_argument("invalid CUDA KV sequence geometry");
}
void KvOffloadSequence::reset() { blocks_.clear(); position_ = 0; }
void KvOffloadSequence::append(const tb::Tensor& source) {
    if (!source.is_cuda() || source.dim() != 2 || source.size(1) != width_ || source.size(0) <= 0)
        throw std::invalid_argument("invalid CUDA KV rows");
    if (!store_->impl_->hot_limit()) {
        auto host=source.to(dtype_).contiguous().to(tb::kCPU);
        const std::lock_guard<std::recursive_mutex> lock(store_->impl_->mutex);
        store_->impl_->d2h_bytes+=host.nbytes();
        append_host(host,source.device());
        return;
    }
    const auto offset = position_;
    const auto previous_count = blocks_.size();
    const auto previous_tail = position_ % block_rows_ ? blocks_.back() : KvOffloadStore::BlockPtr{};
    try {
        auto rows = source.to(dtype_);
        if (position_ % block_rows_) {
            rows = tb::cat({store_->read(blocks_.back()), rows}, 0);
            blocks_.pop_back();
        }
        for (int64_t begin = 0; begin < rows.size(0); begin += block_rows_)
            blocks_.push_back(store_->write(rows.narrow(0, begin, std::min(block_rows_, rows.size(0) - begin))));
        position_ += source.size(0);
    } catch (...) {
        blocks_.resize(previous_count - (previous_tail ? 1 : 0));
        if (previous_tail) blocks_.push_back(previous_tail);
        position_ = offset; throw;
    }
}
void KvOffloadSequence::append_host(const tb::Tensor& source,tb::Device device) {
    if (!source.is_cpu() || source.dim()!=2 || source.size(1)!=width_ ||
        source.size(0)<=0 || source.scalar_type()!=dtype_)
        throw std::invalid_argument("invalid host KV rows");
    auto& impl=*store_->impl_;
    if (impl.hot_limit()) {
        // Limit transient uploads when restoring long host checkpoints.
        const auto chunk=std::max(block_rows_,int64_t(impl.config.buffer_bytes/(width_*source.element_size())));
        for (int64_t start=0;start<source.size(0);start+=chunk)
            append(source.narrow(0,start,std::min(chunk,source.size(0)-start)).to(device));
        return;
    }
    auto rows=source.contiguous();
    if (const auto tail=position_%block_rows_;tail)
        rows=tb::cat({range_host(position_-tail,tail),rows},0);
    std::vector<KvOffloadStore::BlockPtr> added;
    added.reserve((rows.size(0)+block_rows_-1)/block_rows_);
    const std::lock_guard<std::recursive_mutex> lock(impl.mutex);
    for (int64_t start=0;start<rows.size(0);start+=block_rows_) {
        const auto part=rows.narrow(0,start,std::min(block_rows_,rows.size(0)-start));
        auto block=impl.make_block(part,part.options().device(device),store_->impl_);
        impl.store_cold(block,part.data_ptr());
        added.push_back(std::move(block));
    }
    blocks_.reserve(position_/block_rows_+added.size());
    blocks_.resize(position_/block_rows_);
    blocks_.insert(blocks_.end(),std::make_move_iterator(added.begin()),std::make_move_iterator(added.end()));
    position_+=source.size(0);
}
void KvOffloadSequence::truncate(int64_t position) {
    if (position < 0 || position > position_) throw std::out_of_range("CUDA KV trim exceeds its sequence");
    if (position == position_) return;
    const auto full = position / block_rows_, tail = position % block_rows_;
    KvOffloadStore::BlockPtr remainder;
    if (tail) {
        auto& impl=*store_->impl_;
        if (impl.hot_limit()) {
            remainder=store_->write(store_->read(blocks_.at(full)).narrow(0,0,tail));
        } else {
            const std::lock_guard<std::recursive_mutex> lock(impl.mutex);
            auto rows=range_host(position-tail,tail);
            remainder=impl.make_block(rows,blocks_.at(full)->options,store_->impl_);
            impl.store_cold(remainder,rows.data_ptr());
        }
    }
    blocks_.resize(full);
    if (remainder) blocks_.push_back(std::move(remainder));
    position_ = position;
}
tb::Tensor KvOffloadSequence::gather(const std::vector<int64_t>& rows) const {
    if (rows.empty() || blocks_.empty()) throw std::invalid_argument("empty CUDA KV selection");
    std::map<int64_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> groups;
    for (std::size_t output = 0; output < rows.size(); ++output) {
        if (rows[output] < 0 || rows[output] >= position_) throw std::out_of_range("CUDA KV selected row exceeds its sequence");
        auto& group = groups[rows[output] / block_rows_];
        group.first.push_back(rows[output] % block_rows_); group.second.push_back(output);
    }
    const auto options = blocks_.front()->options;
    auto result = tb::empty({static_cast<int64_t>(rows.size()), width_}, options);
    for (const auto& [block, group] : groups) {
        auto payload = store_->read(blocks_.at(block));
        auto selected = payload.index_select(0, indices(group.first, options));
        result.index_copy_(0, indices(group.second, options), selected);
    }
    return result;
}
void KvOffloadSequence::gather_into(const std::vector<int64_t>& rows,const KvOffloadStore::GatherWorkspace& work) const {
    if (rows.empty()) throw std::invalid_argument("empty fixed CUDA KV gather");
    const GatherSlice slice{this,rows,0};
    gather_into(std::span(&slice,1),work);
}

void KvOffloadSequence::gather_into(std::span<const GatherSlice> slices,const KvOffloadStore::GatherWorkspace& work) const {
    if(work.rows.size(1)!=width_ || work.rows.scalar_type()!=dtype_)
        throw std::invalid_argument("invalid fixed CUDA KV gather workspace");
    auto& impl=*store_->impl_;
    std::lock_guard<std::recursive_mutex> lock(impl.mutex);
    const bool host_only=!work.upload.defined();
    const auto row_bytes=std::size_t(width_)*work.rows.element_size();
    auto* host=static_cast<uint8_t*>(work.host.data_ptr());
    auto* upload=host_only?nullptr:static_cast<uint8_t*>(work.upload.data_ptr());
    auto* destination=static_cast<uint8_t*>(work.rows.data_ptr());
    auto* header=host_only?nullptr:work.descriptors.data_ptr<int64_t>();
    auto* copies=host_only?nullptr:reinterpret_cast<MoeMappedCopyDescriptor*>(header+4);
    std::size_t cold_bytes=0,count=0,total=0;
    std::vector<cache::QsaKvStore::ReadRequest> disk_reads;
    const auto capacity=work.rows.size(0);
    int64_t previous_end=0;
    for (const auto& [sequence,rows,output_offset]:slices) {
        if (!sequence || sequence->store_!=store_ || sequence->width_!=width_ || sequence->dtype_!=dtype_ ||
            output_offset<previous_end || output_offset>capacity || int64_t(rows.size())>capacity-output_offset)
            throw std::invalid_argument("invalid CUDA KV gather slice");
        previous_end=output_offset+int64_t(rows.size());
        std::map<int64_t,std::vector<std::pair<int64_t,std::size_t>>> groups;
        for (std::size_t i=0;i<rows.size();++i) {
            if (rows[i]<0 || rows[i]>=sequence->position_)
                throw std::out_of_range("CUDA KV selected row exceeds its sequence");
            const auto n=sequence->block_rows_;
            groups[rows[i]/n].emplace_back(rows[i]%n,size_t(output_offset)+i);
        }
        for(const auto& [index,selected]:groups) {
            const auto& block=sequence->blocks_.at(index);
            const uint8_t* source=nullptr;
            std::shared_ptr<const std::vector<uint8_t>> payload;
            const bool hot=block->gpu.defined();
            bool direct_read=false;
            if(hot) {source=static_cast<const uint8_t*>(block->gpu.data_ptr());impl.touch(block);++impl.hits;}
            else if(block->ram_offset!=no_offset) {
                source=impl.ram_arena.data()+block->ram_offset;impl.touch_ram(block);
            } else {
                if(!impl.ram || !block->cold)throw std::runtime_error("missing spilled KV block");
                if (int64_t(selected.size())==block->rows) {
                    int64_t next=0;
                    const auto first=selected.front().second;
                    direct_read=std::all_of(selected.begin(),selected.end(),[&](const auto& item){
                        const auto row=next++;
                        return item.first==row && (!host_only || item.second==first+std::size_t(row));
                    });
                }
                if (direct_read) {
                    const auto offset=host_only?selected.front().second*row_bytes:cold_bytes;
                    disk_reads.push_back({block->cold,host+offset,block->bytes});
                }
                else {payload=impl.ram->read(block->cold);source=payload->data();}
            }
            for(const auto& [row,output]:selected) {
                const auto* input=direct_read?nullptr:source+std::size_t(row)*row_bytes;
                if(host_only) {
                    if(!direct_read)std::memcpy(host+output*row_bytes,input,row_bytes);
                    cold_bytes+=row_bytes;
                    continue;
                }
                if(!hot) {
                    if (!direct_read) std::memcpy(host+cold_bytes,input,row_bytes);
                    input=upload+cold_bytes;
                    cold_bytes+=row_bytes;
                }
                const auto add=[&](uint8_t* target,const uint8_t* source,std::size_t bytes) {
                    copies[count++]={reinterpret_cast<uint64_t>(target),reinterpret_cast<uint64_t>(source),total,bytes};
                    total+=(bytes+15)&~std::size_t(15);
                };
                if(!work.key_width)add(destination+output*row_bytes,input,row_bytes);
                else {
                    const auto keys=std::size_t(work.key_width)*work.rows.element_size();
                    const auto values=row_bytes/work.heads-keys;
                    for(int64_t head=0;head<work.heads;++head) {
                        add(destination+(head*capacity+output)*keys,input+head*keys,keys);
                        add(destination+capacity*work.heads*keys+(head*capacity+output)*values,
                            input+work.heads*keys+head*values,values);
                    }
                }
            }
        }
    }
    if (!disk_reads.empty()) impl.ram->read_into_batch(disk_reads);
    if (!count && !cold_bytes) return;
    impl.h2d_bytes+=cold_bytes;
    const auto stream=current_stream().stream();
    if (host_only) {
        for (const auto& slice:slices) if (!slice.rows.empty()) {
            const auto offset=std::size_t(slice.output_offset)*row_bytes;
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(destination+offset,host+offset,slice.rows.size()*row_bytes,
                cudaMemcpyHostToDevice,stream));
        }
        return;
    }
    header[0]=int64_t(count);header[1]=int64_t(total);header[2]=header[3]=0;
    if(cold_bytes)TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(upload,host,cold_bytes,cudaMemcpyHostToDevice,stream));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(work.device_descriptors.data_ptr(),header,(4+count*4)*sizeof(int64_t),cudaMemcpyHostToDevice,stream));
    moe_copy_mapped(work.device_descriptors.data_ptr(),reinterpret_cast<const uint32_t*>(work.device_descriptors.data_ptr<int64_t>()+2),
        int(std::min<std::size_t>(256,(total+4095)/4096)),256,stream);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());
}
tb::Tensor KvOffloadSequence::range(int64_t start, int64_t count) const {
    if (start < 0 || count <= 0 || start > position_ || count > position_ - start)
        throw std::out_of_range("CUDA KV range exceeds its sequence");
    auto result = tb::empty({count, width_}, blocks_.front()->options);
    for (int64_t written = 0; written < count;) {
        const auto row = start + written, block = row / block_rows_, offset = row % block_rows_;
        auto source = store_->read(blocks_.at(block));
        const auto n = std::min(count-written,source.size(0)-offset);
        result.narrow(0,written,n).copy_(source.narrow(0,offset,n));
        written += n;
    }
    return result;
}

tb::Tensor KvOffloadSequence::range_host(int64_t start,int64_t count) const {
    if (start<0 || count<=0 || start>position_ || count>position_-start)
        throw std::out_of_range("CUDA KV host range exceeds its sequence");
    auto& impl=*store_->impl_;
    const std::lock_guard<std::recursive_mutex> lock(impl.mutex);
    auto out=tb::empty({count,width_},blocks_.front()->options.device(tb::kCPU));
    auto* destination=static_cast<std::uint8_t*>(out.data_ptr());
    const auto row_bytes=size_t(width_)*out.element_size();
    const auto stream=current_stream().stream();
    std::vector<cache::QsaKvStore::ReadRequest> disk_reads;
    bool queued=false;
    for (int64_t finished=0;finished<count;) {
        const auto index=start+finished,row=index%block_rows_;
        const auto& block=blocks_.at(index/block_rows_);
        const auto copied=std::min(count-finished,block->rows-row);
        auto* output=destination+finished*row_bytes;
        const auto offset=row*row_bytes,bytes=copied*row_bytes;
        if (block->gpu.defined()) {
            auto* data=static_cast<const std::uint8_t*>(block->gpu.data_ptr());
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(output,data+offset,bytes,cudaMemcpyDeviceToHost,stream));
            impl.d2h_bytes+=bytes;
            queued=true;
        } else if (block->ram_offset!=no_offset) {
            std::memcpy(output,impl.ram_arena.data()+block->ram_offset+offset,bytes);
        } else {
            if (!impl.ram || !block->cold) throw std::runtime_error("missing spilled KV block");
            if (!offset && bytes==block->bytes) disk_reads.push_back({block->cold,output,bytes});
            else {
                const auto payload=impl.ram->read(block->cold);
                std::memcpy(output,payload->data()+offset,bytes);
            }
        }
        finished+=copied;
    }
    if (queued) TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!disk_reads.empty()) impl.ram->read_into_batch(disk_reads);
    return out;
}
}
