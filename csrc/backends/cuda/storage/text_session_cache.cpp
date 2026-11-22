#include "storage/text_session_cache.h"
#include "runtime/device_memory_topology.h"
#include "storage/qsa_ram_session.h"
#include "runtime/qsa_tq_cache.h"
#include "core/full_block.h"
#include "mfq_cuda_paged_kv.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_runtime_config.h"
#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/hy4_flash/ops.h"
#include "models/inkling2/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/qwen35/ops.h"
#include "models/nemotron_h/ops.h"
#include "models/qwen4_exp/ops.h"
#include "storage/session_state.h"
#include "core/mtp.h"
#include "mfq_paged_prefix_cache.h"
#include "tyloinfer/host_memory.h"
#include "paged_session_bindings.h"
#include "session_cache.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <fstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

namespace mfq::cuda::internal {

namespace {
constexpr std::array<uint8_t, 8> hybrid_magic{'M', 'F', 'Q', 'C', 'U', 'D', '4', 0};

struct HybridPageHeader {
    uint32_t start, count, layers, flags;
};

HybridPageHeader read_hybrid_header(CudaPagedReader& reader) {
    std::array<uint8_t, 8> magic{};
    reader.raw(magic.data(), magic.size(), "magic");
    if (magic != hybrid_magic || reader.scalar<uint32_t>("version") != 1)
        throw CudaSessionStateError("unsupported CUDA hybrid prefix payload");
    const HybridPageHeader result{reader.scalar<uint32_t>("start"), reader.scalar<uint32_t>("count"),
        reader.scalar<uint32_t>("layers"), reader.scalar<uint32_t>("flags")};
    if (!result.count || !result.layers || result.layers > 4096 ||
        (result.flags!=0 && result.flags!=1 && result.flags!=3 && result.flags!=7))
        throw CudaSessionStateError("invalid CUDA hybrid prefix geometry");
    return result;
}
}

CudaPagedPayload encode_cuda_hybrid_paged_block(const TextSessionState&, size_t, size_t);
TextSessionState decode_cuda_hybrid_paged_session(const std::vector<CudaPagedPayload>&,
    const std::vector<int64_t>&, size_t);

bool cuda_has_separate_memory() {
    return device_has_separate_memory(mfq_current_cuda_device());
}

static std::string cuda_prefix_cache_compatibility_key(
    const mfq::ModelSource &source, int64_t max_position_embeddings) {
    std::ostringstream key;
    key << "mfq-cuda-prefix-v1\n"
        << "codec=" << (source.architecture()=="qwen4_exp" ? "cuda-qwen4-session-v1" :
            source.architecture()=="deepseek_v41" ? "cuda-deepseek-v41-session-v1" :
            (source.architecture()=="sensenova_loop" || source.architecture()=="inkling2_loop" ||
                source.architecture()=="hy_v4_flash") ?
                "cuda-hybrid-session-v1" : "cuda-full-attention-kv-v1") << '\n'
        << "context=" << max_position_embeddings << '\n'
        << "architecture=" << source.architecture() << '\n';
    const auto quantization=mfq::cuda::cuda_kv_quantization_tag();
    if(!quantization.empty())key<<"kv="<<quantization<<'\n';
    if(source.architecture()=="qwen4_exp" || source.architecture()=="qwen4exp")
        key<<"qsa-ram-session-v3="<<mfq::cuda::qsa_cache_mode()<<'\n';
    for (std::size_t index = 0; index < source.source_paths().size(); ++index) {
        const auto &path = source.source_paths()[index];
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify model source for prefix cache: " + error.message());
        }
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error) {
            throw std::runtime_error("cannot identify model source timestamp: " + error.message());
        }
        const auto modified_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(modified.time_since_epoch())
                .count();
        key << "source=" << index << ':' << path.filename().string() << ':' << size << ':'
            << modified_ns << '\n';
    }
    for (const auto &tensor : source.tensors()) {
        key << "tensor=" << tensor.name << ':' << tensor.dtype << ':' << tensor.nbytes << '\n';
    }
    return key.str();
}

std::shared_ptr<mfq::cache::PagedPrefixCache> make_cuda_paged_prefix_cache(
    const mfq::ModelSource &source, int64_t max_position_embeddings,
    bool supports_paged_text_session_state, const mfq::engine::PrefixCacheConfig &config) {
    const auto format = source.metadata().find("source.format");
    // ponytail: HF source fingerprints exclude config sidecars for now.
    if (!config.enabled ||
        (format != source.metadata().end() && format->second == "hf-safetensors") ||
        !supports_paged_text_session_state) {
        return {};
    }
    mfq::cache::PagedPrefixCacheConfig cache;
    cache.cache_dir = config.directory;
    if (!source.source_paths().empty()) cache.model_path = source.source_paths().front();
    cache.compatibility_key = cuda_prefix_cache_compatibility_key(source, max_position_embeddings);
    cache.block_size_tokens = static_cast<size_t>(config.block_tokens);
    // A longer block may lack the local/rolling state of a shorter exact
    // checkpoint. Retain those boundaries under the existing byte-budget LRU.
    cache.retain_tail_versions = 0;
    cache.max_disk_bytes = config.disk_bytes;
    const bool separate = cuda_has_separate_memory();
    cache.max_hot_bytes = separate ? config.ram_bytes : config.hot_bytes;
    cache.spill_hot_on_eviction = separate;
    if (separate) cache.available_ram_bytes = []() -> std::uint64_t {
        return mfq::sample_host_memory().available_host_bytes();
    };
    cache.max_pending_writes = config.pending_writes;
    cache.max_pending_bytes = config.pending_bytes;
    return std::make_shared<mfq::cache::PagedPrefixCache>(std::move(cache));
}

struct CudaSessionOps {
    using Snapshot = TextSessionState;
    using Restore = TextSessionRestore;
    using Predictor = MtpModule;
    using StateError = CudaSessionStateError;
    template <class Payloads>
    static auto decode(
        const Payloads &payloads, const std::vector<int64_t> &tokens, size_t block_size) {
        if(payloads.front() && payloads.front()->size()>6 && (*payloads.front())[6]=='2')
            return decode_cuda_qwen4_paged_session(payloads,tokens,block_size);
        if(payloads.front() && payloads.front()->size()>6 && (*payloads.front())[6]=='3')
            return decode_cuda_deepseek_v41_paged_session(payloads,tokens,block_size);
        if(payloads.front() && payloads.front()->size()>6 && (*payloads.front())[6]=='4')
            return decode_cuda_hybrid_paged_session(payloads,tokens,block_size);
        return decode_cuda_paged_session(payloads, tokens, block_size);
    }
    static auto encode(const Snapshot &state, size_t block_size, size_t index) {
        if(state.kind()==TextSessionStateKind::Qwen4)return encode_cuda_qwen4_paged_block(state,block_size,index);
        if(state.kind()==TextSessionStateKind::DeepseekV41)return encode_cuda_deepseek_v41_paged_block(state,block_size,index);
        if(state.kind()==TextSessionStateKind::HybridAttention)return encode_cuda_hybrid_paged_block(state,block_size,index);
        return encode_cuda_paged_block(state, block_size, index);
    }
    static Snapshot decode_paged(mfq::cache::PagedPrefixCache& cache,mfq::cache::PrefixMatch& match,
        const std::vector<int64_t>& prompt,size_t limit) {
        auto first=cache.load_prefix({match.blocks.front()});
        if(first.empty()){match.blocks.clear();match.matched_tokens=0;return {};}
        if(first[0]->size()>=12 && (*first[0])[6]=='2' && (*first[0])[8]==2) {
            auto state=decode_cuda_qwen4_paged_prefix(cache,match.blocks,prompt,cache.block_size_tokens());
            match.matched_tokens=state.tokens.size();
            if(match.matched_tokens>limit)throw StateError("QSA prefix exceeds requested tokens");
            return state;
        }
        auto payloads=cache.load_prefix(match.blocks);
        payloads.resize(decodable_blocks(payloads));match.blocks.resize(payloads.size());match.matched_tokens=0;
        for(const auto& payload:payloads) {
            const auto n=token_count(payload);
            if(!n || n>cache.block_size_tokens() || n>limit-match.matched_tokens)
                throw StateError("invalid paged prefix token count");
            match.matched_tokens+=n;
        }
        if(payloads.empty())return {};
        return decode(payloads,std::vector<int64_t>(prompt.begin(),prompt.begin()+match.matched_tokens),cache.block_size_tokens());
    }
    static void bind_paged(const Snapshot& state,mfq::cache::PagedPrefixCache& cache,
        const std::vector<mfq::cache::BlockHash>& blocks) {bind_cuda_qwen4_paged_prefix(state,cache,blocks);}
    static void refresh_bytes(Snapshot& state) {refresh_cuda_qwen4_session_bytes(state);}
    static void prepare_restore(const Snapshot& state,bool disk) {prepare_cuda_qwen4_session_restore(state,disk);}
    static bool paged_mtp(const Snapshot& state) {
        return state.kind()==TextSessionStateKind::Qwen4 || state.kind()==TextSessionStateKind::DeepseekV41 ||
            state.kind()==TextSessionStateKind::HybridAttention;
    }
    static bool partial_blocks(const Snapshot& state) { return paged_mtp(state); }
    static bool paged_media(const Snapshot& state) { return state.kind()==TextSessionStateKind::DeepseekV41; }
    static bool checkpoint_needs_update(const CudaPagedPayload& payload,const Snapshot& state) {
        if (state.kind()==TextSessionStateKind::HybridAttention) {
            if (!payload) return true;
            CudaPagedReader reader(*payload);
            const auto flags=read_hybrid_header(reader).flags;
            if (state.mtp && !state.mtp->blocks.empty() && state.mtp->cache_pos==state.cache_pos && !(flags&4))
                return true;
            return !(flags&1) || (state.mtp && !(flags&2));
        }
        return cuda_paged_checkpoint_needs_update(payload,state);
    }
    static size_t decodable_blocks(const std::vector<CudaPagedPayload>& payloads) { return cuda_paged_decodable_blocks(payloads); }
    static size_t token_count(const CudaPagedPayload& payload) { return cuda_paged_payload_tokens(payload); }
    static void release_host_cache() { mfq_release_host_allocator_cache(); }
};

struct TextSessionCache::Impl : mfq::engine::SessionCache<CudaSessionOps> {
    using SessionCache::SessionCache;
};

TextSessionCache::TextSessionCache(const mfq::engine::SessionCacheConfig &session_config,
    const mfq::engine::PrefixCacheConfig &prefix_config,
    std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache, bool supported, int disabled_reason,
    bool native_hot_snapshots)
    : impl_(std::make_unique<Impl>(
          session_config, prefix_config, std::move(paged_cache), supported, disabled_reason,
          native_hot_snapshots)), prefix_block_tokens_(std::max<std::uint64_t>(1,prefix_config.block_tokens)) {}

TextSessionCache::~TextSessionCache() = default;

bool TextSessionCache::persistent_prefix_enabled() const noexcept {
    return impl_->persistent_prefix_enabled();
}

template <typename Model>
TextSessionRestore TextSessionCache::restore_best(Model &model, MtpModule *mtp,
    const std::string &requested_session, const std::vector<int64_t> &prompt,
    size_t maximum_prefix_tokens, const std::string &input_key) {
    return impl_->restore_best(
        model, mtp, requested_session, prompt, maximum_prefix_tokens, input_key);
}

void TextSessionCache::store(const std::string &session_id, TextSessionState state) {
    impl_->reserve_ram(state.bytes);
    impl_->store(session_id, std::move(state));
}

size_t TextSessionCache::fork_session(
    const std::string &source_session, const std::string &target_session) {
    return impl_->fork_session(source_session, target_session);
}

size_t TextSessionCache::close_session(const std::string &session_id) {
    return impl_->close_session(session_id);
}

std::vector<std::pair<std::string, double>> TextSessionCache::metrics() const {
    return impl_->metrics();
}

std::map<int, std::size_t> TextSessionCache::device_hot_bytes() const {
    std::map<std::pair<int, std::uintptr_t>, std::size_t> allocations;
    const auto tensor = [&](const mfq_tensor_backend::Tensor& value) {
        if (!value.defined() || !value.is_cuda()) return;
        auto& bytes = allocations[{value.get_device(), reinterpret_cast<std::uintptr_t>(value.data_ptr())}];
        bytes = std::max(bytes, std::size_t(value.numel()) * value.element_size());
    };
    const auto full = [&](const FullBlockSessionState& block) { tensor(block.k); tensor(block.v); };
    const auto qwen = [&](const Qwen4BlockSessionState& block) {
        tensor(block.keys); tensor(block.values); tensor(block.index); tensor(block.convolution);
        tensor(block.recurrent); tensor(block.ple_convolution);
    };
    const auto pool = [&](const Dsv4PoolSessionState& value) {
        tensor(value.state_kv); tensor(value.state_gate); tensor(value.previous_kv);
        tensor(value.previous_gate); tensor(value.pool);
    };
    impl_->visit_snapshots([&](const TextSessionState& state) {
        tensor(state.qwen4_positions);
        std::visit([&](const auto& blocks) {
            if constexpr (!std::is_same_v<std::decay_t<decltype(blocks)>, std::monostate>)
                for (const auto& block : blocks) {
                    using Block = std::decay_t<decltype(block)>;
                    if constexpr (std::is_same_v<Block, FullBlockSessionState>) full(block);
                    else if constexpr (std::is_same_v<Block, Qwen4BlockSessionState>) qwen(block);
                    else if constexpr (std::is_same_v<Block, HybridBlockSessionState>) {
                        full(block.full_attention); tensor(block.convolution_state); tensor(block.recurrent_state);
                    } else if constexpr (std::is_same_v<Block, Dsv4BlockSessionState>) {
                        tensor(block.local_cache); pool(block.compressor); pool(block.indexer_compressor);
                    } else if constexpr (std::is_same_v<Block, DeepseekV41BlockSessionState>) {
                        tensor(block.local_kv); tensor(block.compressed_kv); tensor(block.index_k);
                        tensor(block.partial_kv); tensor(block.partial_score);
                    } else { tensor(block.kv_cache); tensor(block.index_cache); }
                }
        }, state.payload);
        if (state.mtp) {
            tensor(state.mtp->last_target_hidden);
            for (const auto& block : state.mtp->blocks) full(block);
            for (const auto& block : state.mtp->qwen4_blocks) qwen(block);
            for (const auto& ring : state.mtp->deepseek_v41_rings) tensor(ring);
            for (const auto& value : state.mtp->qwen4_positions) tensor(value);
        }
    });
    std::map<int, std::size_t> result;
    for (const auto& [address, bytes] : allocations) result[address.first] += bytes;
    return result;
}

size_t TextSessionCache::clear_live_sessions() noexcept { return impl_->clear_live_sessions(); }

size_t TextSessionCache::clear() { return impl_->clear(); }

uint64_t TextSessionCache::trim_hot(uint64_t target_bytes) { return impl_->trim_hot(target_bytes); }
uint64_t TextSessionCache::set_hot_limit(uint64_t max_bytes) { return impl_->set_hot_limit(max_bytes); }
uint64_t TextSessionCache::set_ram_limit(uint64_t max_bytes) { return impl_->set_ram_limit(max_bytes); }
uint64_t TextSessionCache::trim_ram(uint64_t target_bytes) { return impl_->trim_ram(target_bytes); }
void TextSessionCache::reserve_ram(uint64_t bytes) { impl_->reserve_ram(bytes); }
std::function<uint64_t(uint64_t)> TextSessionCache::ram_reclaimer() const { return impl_->ram_reclaimer(); }
uint64_t TextSessionCache::set_disk_limit(uint64_t max_bytes) { return impl_->set_disk_limit(max_bytes); }
uint64_t TextSessionCache::refresh_disk_index() { return impl_->refresh_disk_index(); }

#define TYLOINFER_INSTANTIATE_SESSION_CACHE(MODEL)                                                       \
    template TextSessionRestore TextSessionCache::restore_best(MODEL &,                            \
        MtpModule *,                                                                               \
        const std::string &,                                                                       \
        const std::vector<int64_t> &,                                                              \
        size_t,                                                                                    \
        const std::string &);

TYLOINFER_INSTANTIATE_SESSION_CACHE(mfq::cuda::Qwen35CausalLm)
TYLOINFER_INSTANTIATE_SESSION_CACHE(mfq::cuda::MiniCPMO45CausalLm)
TYLOINFER_INSTANTIATE_SESSION_CACHE(mfq::cuda::MiniCPMOTtsCausalLm)
TYLOINFER_INSTANTIATE_SESSION_CACHE(mfq::cuda::Gemma4CausalLm)
TYLOINFER_INSTANTIATE_SESSION_CACHE(mfq::cuda::GlmDsaCausalLm)
TYLOINFER_INSTANTIATE_SESSION_CACHE(mfq::cuda::Glm5CausalLm)
TYLOINFER_INSTANTIATE_SESSION_CACHE(mfq::cuda::Qwen4CausalLm)
TYLOINFER_INSTANTIATE_SESSION_CACHE(mfq::cuda::DeepseekV4CausalLm)
TYLOINFER_INSTANTIATE_SESSION_CACHE(mfq::cuda::DeepseekV41CausalLm)
TYLOINFER_INSTANTIATE_SESSION_CACHE(mfq::cuda::NemotronHCausalLm)
TYLOINFER_INSTANTIATE_SESSION_CACHE(mfq::cuda::InklingCausalLm)
TYLOINFER_INSTANTIATE_SESSION_CACHE(mfq::cuda::Hy4CausalLm)

#undef TYLOINFER_INSTANTIATE_SESSION_CACHE

} // namespace mfq::cuda::internal

using mfq::cuda::internal::read_ram_qsa_page;
using mfq::cuda::internal::write_ram_qsa_page;

namespace {
struct PagedTensorType {
    int32_t code;
    mfq_tensor_backend::ScalarType dtype;
    std::size_t bytes;
};
// Persistent IDs are independent of the backend's tensor enum.
constexpr std::array<PagedTensorType,8> paged_tensor_types{{
    {1,mfq_tensor_backend::kUInt8,1}, {2,mfq_tensor_backend::kInt8,1},
    {3,mfq_tensor_backend::kInt16,2}, {4,mfq_tensor_backend::kInt32,4},
    {5,mfq_tensor_backend::kInt64,8}, {6,mfq_tensor_backend::kFloat16,2},
    {7,mfq_tensor_backend::kBFloat16,2}, {8,mfq_tensor_backend::kFloat32,4}}};
template<class Predicate>
const PagedTensorType* paged_tensor_type(Predicate matches) {
    const auto found=std::find_if(paged_tensor_types.begin(),paged_tensor_types.end(),matches);
    return found==paged_tensor_types.end()?nullptr:&*found;
}
}

void CudaPagedWriter::raw(const void *data, size_t size) {
    const auto *begin = static_cast<const uint8_t *>(data);
    bytes_.insert(bytes_.end(), begin, begin + size);
}

void CudaPagedWriter::tensor(const mfq_tensor_backend::Tensor &source) {
    if (!source.defined() || !source.is_cuda()) throw CudaSessionStateError("invalid CUDA paged cache tensor");
    tensor(source,source.get_device());
}

void CudaPagedWriter::tensor(const mfq_tensor_backend::Tensor &source,int device) {
    if (!source.defined() || device<0 || source.dim()<=0 || source.dim()>8 ||
        (source.is_cuda() && source.get_device()!=device)) {
        throw std::runtime_error("invalid CUDA paged cache tensor");
    }
    const auto dtype = source.scalar_type();
    const auto* type=paged_tensor_type([&](const auto& type){return type.dtype==dtype;});
    if (!type) {
        throw std::runtime_error("unsupported CUDA paged cache tensor dtype");
    }
    for (const auto dimension : source.sizes())
        if (dimension <= 0) throw std::runtime_error("invalid CUDA paged cache tensor shape");
    const auto cpu = source.contiguous().to(mfq_tensor_backend::kCPU);
    scalar<int32_t>(type->code);
    scalar<int32_t>(device);
    scalar<uint32_t>(static_cast<uint32_t>(cpu.dim()));
    scalar<uint32_t>(0);
    for (const auto dimension : cpu.sizes())
        scalar<int64_t>(dimension);
    const auto bytes = static_cast<uint64_t>(cpu.numel() * cpu.element_size());
    scalar<uint64_t>(bytes);
    raw(cpu.data_ptr(), static_cast<size_t>(bytes));
}

CudaPagedPayload CudaPagedWriter::finish() && {
    return std::make_shared<const std::vector<uint8_t>>(std::move(bytes_));
}

CudaPagedReader::CudaPagedReader(const mfq::cache::PagedPrefixReadLease& lease)
    :lease_(&lease),size_(lease.payload_bytes) {}

void CudaPagedReader::raw(void *destination, size_t size, const char *name) {
    if (remaining() < size) {
        throw CudaSessionStateError(std::string("truncated CUDA paged cache ") + name);
    }
    if(!size)return;
    if(data_)std::memcpy(destination,data_+offset_,size);
    else {
        constexpr size_t window_bytes=4096;
        if(size>=window_bytes)lease_->read_range_into(offset_,static_cast<uint8_t*>(destination),size);
        else {
            if(offset_<window_offset_ || offset_-window_offset_>window_.size() ||
                    size>window_.size()-(offset_-window_offset_)) {
                window_offset_=offset_;
                window_.resize(std::min(window_bytes,remaining()));
                lease_->read_range_into(offset_,window_.data(),window_.size());
            }
            std::memcpy(destination,window_.data()+(offset_-window_offset_),size);
        }
    }
    offset_+=size;
}

std::pair<mfq_tensor_backend::Tensor, int> CudaPagedReader::tensor() {
    const auto dtype_value = scalar<int32_t>("dtype");
    const auto device = scalar<int32_t>("device");
    const auto rank = scalar<uint32_t>("rank");
    (void)scalar<uint32_t>("tensor flags");
    if (rank == 0 || rank > 8 || device < 0) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor header");
    }
    const auto* type=paged_tensor_type([&](const auto& type){return type.code==dtype_value;});
    if (!type) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor dtype");
    }
    const auto dtype=type->dtype;
    const uint64_t element_size=type->bytes;
    std::vector<int64_t> shape;
    shape.reserve(rank);
    uint64_t elements = 1;
    for (uint32_t index = 0; index < rank; ++index) {
        const auto dimension = scalar<int64_t>("shape");
        if (dimension <= 0 ||
            elements > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dimension)) {
            throw CudaSessionStateError("invalid CUDA paged cache tensor shape");
        }
        shape.push_back(dimension);
        elements *= static_cast<uint64_t>(dimension);
    }
    const auto bytes = scalar<uint64_t>("tensor size");
    if (elements > std::numeric_limits<uint64_t>::max() / element_size ||
        bytes != elements * element_size || bytes > remaining()) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor size");
    }
    auto cpu = mfq_tensor_backend::empty(
        shape, mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(dtype));
    raw(cpu.data_ptr(), static_cast<size_t>(bytes), "tensor");
    return {std::move(cpu), device};
}

void CudaPagedReader::expect_end() const {
    if (remaining()) {
        throw CudaSessionStateError("trailing CUDA paged cache payload bytes");
    }
}

namespace {
constexpr std::array<uint8_t,8> qwen4_magic{'M','F','Q','C','U','D','2',0};
struct Qwen4PagedHeader { uint32_t start,count,layers;bool exact;uint32_t version; };
Qwen4PagedHeader read_qwen4_header(CudaPagedReader& reader) {
    std::array<uint8_t,8> magic{};reader.raw(magic.data(),magic.size(),"magic");
    const auto version=reader.scalar<uint32_t>("version");
    if(magic!=qwen4_magic || (version!=1 && version!=2))
        throw CudaSessionStateError("unsupported Qwen4 prefix cache payload");
    Qwen4PagedHeader header{reader.scalar<uint32_t>("start"),reader.scalar<uint32_t>("count"),
        reader.scalar<uint32_t>("layers"),reader.scalar<uint32_t>("exact boundary")!=0,version};
    if(!header.count || !header.layers || header.layers>4096)throw CudaSessionStateError("invalid Qwen4 prefix geometry");
    return header;
}
void write_optional(CudaPagedWriter& writer,const mfq_tensor_backend::Tensor& tensor) {
    writer.scalar<uint32_t>(tensor.defined()?1:0);if(tensor.defined())writer.tensor(tensor);
}
mfq_tensor_backend::Tensor read_optional(CudaPagedReader& reader,bool host=false) {
    if(!reader.scalar<uint32_t>("tensor presence"))return {};
    auto [tensor,device]=reader.tensor();
    return host?std::move(tensor):tensor.to(mfq_tensor_backend::Device{mfq_tensor_backend::DeviceType::cuda,device});
}
void write_recurrent(CudaPagedWriter& writer,const Qwen4BlockSessionState& layer) {
    write_optional(writer,layer.convolution);write_optional(writer,layer.recurrent);write_optional(writer,layer.ple_convolution);
    writer.scalar<int64_t>(layer.ple_batch);writer.scalar<uint32_t>(uint32_t(layer.ple_tokens.size()));
    for(auto token:layer.ple_tokens)writer.scalar<int64_t>(token);
}
void read_recurrent(CudaPagedReader& reader,Qwen4BlockSessionState& layer,bool stage_gdn=false) {
    layer.convolution=read_optional(reader,stage_gdn);layer.recurrent=read_optional(reader,stage_gdn);
    layer.ple_convolution=read_optional(reader);
    layer.ple_batch=reader.scalar<int64_t>("PLE batch");
    const auto count=reader.scalar<uint32_t>("PLE history length");
    if(count>65536)throw CudaSessionStateError("invalid Qwen4 PLE history");
    for(uint32_t i=0;i<count;++i)layer.ple_tokens.push_back(reader.scalar<int64_t>("PLE token"));
}
void write_qwen4_mtp(CudaPagedWriter& writer,const MtpSessionState& state,bool packed=false) {
    if(state.qwen4_blocks.size()!=state.qwen4_positions.size() || state.qwen4_blocks.size()!=state.qwen4_lengths.size())
        throw CudaSessionStateError("incomplete Qwen4 MTP prefix state");
    writer.scalar<int64_t>(state.cache_pos);write_optional(writer,state.last_target_hidden);
    writer.scalar<uint32_t>(uint32_t(state.qwen4_blocks.size()));
    for(std::size_t i=0;i<state.qwen4_blocks.size();++i) {
        const auto& layer=state.qwen4_blocks[i];
        if(packed)writer.scalar<uint32_t>(layer.adaptive_qsa?1:0);
        if(packed && layer.adaptive_qsa) {
            const auto count=layer.adaptive_qsa->tokens;
            if(count<=0 || count!=state.qwen4_lengths[i])throw CudaSessionStateError("incompatible packed MTP position");
            writer.scalar<int64_t>(count);write_ram_qsa_page(writer,*layer.adaptive_qsa,0,size_t(count),true);
        }else {write_optional(writer,layer.keys);write_optional(writer,layer.values);write_optional(writer,layer.index);}
        write_recurrent(writer,layer);write_optional(writer,state.qwen4_positions[i]);
        writer.scalar<int64_t>(state.qwen4_lengths[i]);
    }
}
std::size_t qwen4_layer_bytes(const Qwen4BlockSessionState& layer) {
    std::size_t bytes=layer.ple_tokens.size()*sizeof(int64_t);
    if(layer.adaptive_qsa)bytes+=layer.adaptive_qsa->resident_bytes();
    for(const auto* tensor:{&layer.keys,&layer.values,&layer.index,&layer.convolution,&layer.recurrent,&layer.ple_convolution})
        bytes+=session_tensor_bytes(*tensor);
    return bytes;
}
MtpSessionState read_qwen4_mtp(CudaPagedReader& reader,
    const std::shared_ptr<const mfq::cache::PagedPrefixReadLease>& lease,bool packed) {
    MtpSessionState state;state.cache_pos=reader.scalar<int64_t>("MTP position");state.last_target_hidden=read_optional(reader);
    const auto count=reader.scalar<uint32_t>("MTP layers");
    if(!count || count>64)throw CudaSessionStateError("invalid Qwen4 MTP layers");
    state.bytes=session_tensor_bytes(state.last_target_hidden);
    for(uint32_t i=0;i<count;++i) {
        Qwen4BlockSessionState layer;
        const auto kind=packed?reader.scalar<uint32_t>("MTP QSA page kind"):0;
        if(kind>1)throw CudaSessionStateError("invalid MTP QSA page kind");
        if(kind) {
            const auto count=reader.scalar<int64_t>("MTP compressed position");
            if(count<=0 || count>state.cache_pos)throw CudaSessionStateError("invalid MTP compressed position");
            read_ram_qsa_page(reader,lease,0,size_t(count),true,layer.adaptive_qsa);
        }else {layer.keys=read_optional(reader);layer.values=read_optional(reader);layer.index=read_optional(reader);}
        read_recurrent(reader,layer);state.bytes+=qwen4_layer_bytes(layer);state.qwen4_blocks.push_back(std::move(layer));
        auto positions=read_optional(reader);state.bytes+=session_tensor_bytes(positions);state.qwen4_positions.push_back(std::move(positions));
        state.qwen4_lengths.push_back(reader.scalar<int64_t>("MTP layer position"));
        if(state.qwen4_blocks.back().adaptive_qsa && state.qwen4_blocks.back().adaptive_qsa->tokens!=state.qwen4_lengths.back())
            throw CudaSessionStateError("inconsistent MTP compressed position");
    }
    return state;
}
}

CudaPagedPayload encode_cuda_qwen4_paged_block(const TextSessionState& state,size_t block_size,size_t index) {
    const auto* layers=std::get_if<std::vector<Qwen4BlockSessionState>>(&state.payload);
    if(!block_size || !layers || layers->empty() || state.cache_pos<=0 || state.tokens.size()!=std::size_t(state.cache_pos) ||
        index>state.tokens.size()/block_size || index*block_size>=state.tokens.size() || state.cache_pos>UINT32_MAX)
        throw std::runtime_error("invalid Qwen4 prefix block boundary");
    const auto start=index*block_size,count=std::min(block_size,state.tokens.size()-start);
    const bool exact=start+count==state.tokens.size();
    const bool packed=std::any_of(layers->begin(),layers->end(),[](const auto& l){return bool(l.adaptive_qsa);});
    CudaPagedWriter writer;writer.raw(qwen4_magic.data(),qwen4_magic.size());writer.scalar<uint32_t>(packed?2:1);
    writer.scalar<uint32_t>(uint32_t(start));writer.scalar<uint32_t>(uint32_t(count));
    writer.scalar<uint32_t>(uint32_t(layers->size()));writer.scalar<uint32_t>(exact?1:0);
    for(const auto& layer:*layers) {
        if(packed)writer.scalar<uint32_t>(layer.adaptive_qsa?1:0);
        if(layer.adaptive_qsa)write_ram_qsa_page(writer,*layer.adaptive_qsa,start,count,exact);
        else for(const auto* tensor:{&layer.keys,&layer.values,&layer.index})
            write_optional(writer,tensor->defined()?tensor->narrow(1,int64_t(start),int64_t(count)):mfq_tensor_backend::Tensor{});
        if(exact)write_recurrent(writer,layer);
    }
    if(exact) {
        write_optional(writer,state.qwen4_positions);writer.scalar<uint32_t>(state.mtp?1:0);
        if(state.mtp)write_qwen4_mtp(writer,*state.mtp,packed);
    }
    return std::move(writer).finish();
}

namespace mfq::cuda::internal {
namespace {
namespace tb = mfq_tensor_backend;

void write_hybrid_kv(CudaPagedWriter& writer, const FullBlockSessionState& kv,
                     int64_t start, int64_t count) {
    if (!session_kv_geometry_matches(kv.k, kv.v) || kv.k.size(0)!=1 ||
        start+count>kv.k.size(2) || kv.capacity<kv.k.size(2))
        throw CudaSessionStateError("invalid CUDA hybrid prefix KV geometry");
    writer.tensor(kv.k.narrow(2,start,count));
    writer.tensor(kv.v.narrow(2,start,count));
}

std::pair<FullBlockSessionState, int> read_hybrid_kv(CudaPagedReader& reader,
                                                    int64_t capacity, int64_t count, bool ring) {
    auto [key,device]=reader.tensor();
    auto [value,value_device]=reader.tensor();
    if (device!=value_device || count<=0 || capacity<count ||
        !session_kv_geometry_matches(key,value) || key.size(0)!=1 || key.size(2)!=count)
        throw CudaSessionStateError("invalid CUDA hybrid prefix KV page");
    return {{std::move(key),std::move(value),capacity,ring},device};
}

void upload_hybrid_kv(FullBlockSessionState& kv, int device, size_t& bytes) {
    const tb::Device target(tb::kCUDA,device);
    kv.k=kv.k.to(target);
    kv.v=kv.v.to(target);
    bytes+=session_tensor_bytes(kv.k)+session_tensor_bytes(kv.v);
}

void write_latent_page(CudaPagedWriter& writer,const HybridBlockSessionState& layer,
                       int64_t start,int64_t count,bool exact) {
    const auto& kv=layer.full_attention;
    const auto group=layer.group_size;
    if (!kv.k.defined() || kv.k.dim()!=4 || kv.k.size(0)!=1 || kv.k.size(1)!=1 ||
        kv.capacity<kv.k.size(2) || (!kv.ring && kv.k.size(2)<start+count) ||
        (kv.ring && kv.capacity!=kv.k.size(2)) || group<0 || group>64)
        throw CudaSessionStateError("invalid latent KV checkpoint geometry");
    writer.scalar<int64_t>(kv.capacity);
    writer.scalar<uint32_t>(kv.ring);
    writer.scalar<int64_t>(group);
    if (!kv.ring) writer.tensor(kv.k.narrow(2,start,count),layer.latent_device);
    else if (exact) writer.tensor(kv.k);
    if (!group) return;
    const auto first=start/group,last=(start+count)/group;
    if (first<last) writer.tensor(layer.index_pool.narrow(1,first,last-first),layer.latent_device);
    if (exact) {writer.tensor(layer.pending_values);writer.tensor(layer.pending_logits);}
}

tb::Tensor join_hybrid_pages(const std::vector<tb::Tensor>& pages, int64_t tokens) {
    if (pages.size()==1) return pages.front();
    const auto heads=pages.front().size(1);
    const auto width=pages.front().size(3);
    const auto row_bytes=size_t(width)*pages.front().element_size();
    auto joined=tb::empty({1,heads,tokens,width},pages.front().options());
    auto* destination=static_cast<uint8_t*>(joined.data_ptr());
    size_t offset=0;
    for (const auto& page:pages) {
        const auto count=size_t(page.size(2));
        const auto* source=static_cast<const uint8_t*>(page.data_ptr());
        for (size_t head=0;head<size_t(heads);++head)
            std::memcpy(destination+(head*tokens+offset)*row_bytes,
                source+head*count*row_bytes,count*row_bytes);
        offset+=count;
    }
    return joined;
}
}

CudaPagedPayload encode_cuda_hybrid_paged_block(const TextSessionState& state, size_t block_size, size_t index) {
    using Kind=HybridBlockSessionStateKind;
    const auto* layers=std::get_if<std::vector<HybridBlockSessionState>>(&state.payload);
    if (!layers || layers->empty() || layers->size()>4096 || !block_size || state.cache_pos<=0 ||
        state.cache_pos>UINT32_MAX || state.tokens.size()!=size_t(state.cache_pos) ||
        index>(state.tokens.size()-1)/block_size || state.decode_position_delta)
        throw CudaSessionStateError("invalid CUDA hybrid prefix boundary");
    const auto start=index*block_size,count=std::min(block_size,state.tokens.size()-start);
    const bool exact=start+count==state.tokens.size();
    const bool full_predictor=state.mtp && !state.mtp->blocks.empty() && state.mtp->cache_pos==state.cache_pos;
    const size_t flags=exact?1+(state.mtp?2:0)+(full_predictor?4:0):0;
    CudaPagedWriter writer;
    writer.raw(hybrid_magic.data(),hybrid_magic.size());
    writer.scalar<uint32_t>(1);
    for (const auto field:{start,count,layers->size(),flags})
        writer.scalar<uint32_t>(uint32_t(field));
    for (const auto& layer:*layers) {
        writer.scalar<uint32_t>(static_cast<uint32_t>(layer.kind));
        if (layer.kind==Kind::LatentAttention) write_latent_page(writer,layer,start,count,exact);
        if (layer.kind==Kind::FullAttention || layer.kind==Kind::AttentionConvolution) {
            const auto& kv=layer.full_attention;
            writer.scalar<int64_t>(kv.capacity);
            writer.scalar<uint32_t>(kv.ring);
            if (!kv.ring) write_hybrid_kv(writer,kv,start,count);
            else if (exact) write_hybrid_kv(writer,kv,0,kv.capacity);
        }
        if (layer.kind==Kind::AttentionConvolution && exact) writer.tensor(layer.convolution_state);
        if (layer.kind==Kind::Recurrent && exact) {
            writer.tensor(layer.convolution_state);
            writer.tensor(layer.recurrent_state);
        }
    }
    if (exact && state.mtp) {
        const auto& mtp=*state.mtp;
        if (mtp.cache_pos<=0 || mtp.cache_pos!=state.cache_pos-int64_t(!mtp.blocks.empty() && !full_predictor) ||
            mtp.blocks.size()>4096)
            throw CudaSessionStateError("invalid CUDA hybrid predictor checkpoint");
        writer.scalar<uint32_t>(uint32_t(mtp.blocks.size()));
        writer.tensor(mtp.last_target_hidden);
        for (const auto& kv:mtp.blocks) {
            writer.scalar<int64_t>(kv.capacity);
            writer.scalar<uint32_t>(kv.ring);
            write_hybrid_kv(writer,kv,0,kv.ring?kv.capacity:mtp.cache_pos);
        }
    }
    return std::move(writer).finish();
}

TextSessionState decode_cuda_hybrid_paged_session(const std::vector<CudaPagedPayload>& payloads,
    const std::vector<int64_t>& tokens, size_t block_size) {
    using Kind=HybridBlockSessionStateKind;
    if (!block_size || tokens.empty() || payloads.size()!=(tokens.size()-1)/block_size+1)
        throw CudaSessionStateError("invalid CUDA hybrid prefix chain length");
    struct KvFragments {
        std::vector<tb::Tensor> keys, values, index;
        int device=-1;
        bool ring=false;
    };
    TextSessionState state;
    state.tokens=tokens;
    state.cache_pos=int64_t(tokens.size());
    auto& layers=state.payload.emplace<std::vector<HybridBlockSessionState>>();
    std::vector<KvFragments> fragments;
    for (size_t page=0;page<payloads.size();++page) {
        CudaPagedReader reader(*payloads[page]);
        const auto header=read_hybrid_header(reader);
        const bool final=page+1==payloads.size();
        if (header.start!=page*block_size || header.count!=std::min(block_size,tokens.size()-header.start) ||
            (page && header.layers!=layers.size()) || (final && !(header.flags&1)))
            throw CudaSessionStateError("invalid CUDA hybrid prefix chain boundary");
        if (!page) { layers.resize(header.layers);fragments.resize(header.layers); }
        for (size_t i=0;i<layers.size();++i) {
            auto& layer=layers[i];
            auto& parts=fragments[i];
            const auto kind=reader.scalar<uint32_t>("layer kind");
            if (kind>uint32_t(Kind::LatentAttention) || (page && layer.kind!=static_cast<Kind>(kind)))
                throw CudaSessionStateError("inconsistent CUDA hybrid prefix layer kind");
            layer.kind=static_cast<Kind>(kind);
            if (layer.kind==Kind::LatentAttention) {
                const auto capacity=reader.scalar<int64_t>("latent capacity");
                const auto ring=reader.scalar<uint32_t>("latent ring");
                const auto group=reader.scalar<int64_t>("index group size");
                const bool exact=(header.flags&1)!=0;
                if (ring>1 || capacity<=0 || (!ring && capacity<int64_t(header.start)+header.count) ||
                    group<0 || group>64 || (page && (parts.ring!=bool(ring) || layer.group_size!=group)))
                    throw CudaSessionStateError("inconsistent latent cache topology");
                parts.ring=ring;
                layer.group_size=group;
                if (!ring || exact) {
                    auto [rows,device]=reader.tensor();
                    if (rows.dim()!=4 || rows.size(0)!=1 || rows.size(1)!=1 || rows.size(2)!=(ring?capacity:header.count) ||
                        (rows.scalar_type()!=tb::kFloat16 && rows.scalar_type()!=tb::kInt32) ||
                        (parts.device>=0 && device!=parts.device) ||
                        (!parts.keys.empty() && (parts.keys.front().size(3)!=rows.size(3) ||
                            parts.keys.front().scalar_type()!=rows.scalar_type())))
                        throw CudaSessionStateError("invalid latent prefix page");
                    parts.device=device;
                    if (!ring) parts.keys.push_back(std::move(rows));
                    if (final) {
                        auto key=ring?std::move(rows):join_hybrid_pages(parts.keys,state.cache_pos);
                        if (ring) key=key.to(tb::Device(tb::kCUDA,device));
                        layer.latent_device=device;
                        state.bytes+=session_tensor_bytes(key);
                        layer.full_attention={std::move(key),{},capacity,bool(ring)};
                        parts.keys.clear();
                    }
                }
                if (!group) continue;
                const auto end=int64_t(header.start)+header.count;
                const auto count=end/group-header.start/group;
                if (count) {
                    auto [index,device]=reader.tensor();
                    if (index.dim()!=3 || index.size(0)!=1 || index.size(1)!=count ||
                        index.scalar_type()!=tb::kInt16 || device!=parts.device ||
                        (!parts.index.empty() && parts.index.front().size(3)!=index.size(2)))
                        throw CudaSessionStateError("invalid latent index prefix page");
                    parts.index.push_back(index.unsqueeze(1));
                }
                if (exact) {
                    auto [values,device]=reader.tensor();
                    auto [logits,logits_device]=reader.tensor();
                    if (values.dim()!=3 || values.size(0)!=1 || values.size(1)!=group ||
                        logits.sizes()!=values.sizes() || values.scalar_type()!=tb::kFloat32 ||
                        logits.scalar_type()!=tb::kFloat32 || device!=parts.device || device!=logits_device)
                        throw CudaSessionStateError("invalid latent pending group");
                    if (final) {
                        const tb::Device target(tb::kCUDA,device);
                        layer.pending_values=values.to(target);
                        layer.pending_logits=logits.to(target);
                        state.bytes+=session_tensor_bytes(values)+session_tensor_bytes(logits);
                        if (!parts.index.empty()) {
                            layer.index_pool=join_hybrid_pages(parts.index,end/group).squeeze(1);
                            state.bytes+=session_tensor_bytes(layer.index_pool);
                        }
                    }
                }
                continue;
            }
            if (layer.kind==Kind::FullAttention || layer.kind==Kind::AttentionConvolution) {
                const auto capacity=reader.scalar<int64_t>("KV capacity");
                const auto ring=reader.scalar<uint32_t>("KV ring");
                if (ring>1 || capacity<0 || (!ring && capacity<int64_t(header.start)+header.count) ||
                    (page && parts.ring!=bool(ring)))
                    throw CudaSessionStateError("inconsistent CUDA hybrid prefix KV mode");
                parts.ring=ring;
                if (ring && !(header.flags&1)) continue;
                auto [kv,device]=read_hybrid_kv(reader,capacity,ring?capacity:header.count,ring);
                if (layer.kind==Kind::AttentionConvolution && (header.flags&1)) {
                    auto [conv,conv_device]=reader.tensor();
                    if (device!=conv_device) throw CudaSessionStateError("inconsistent CUDA convolution device");
                    if (final) {
                        layer.convolution_state=conv.to(tb::Device(tb::kCUDA,device));
                        state.bytes+=session_tensor_bytes(conv);
                    }
                }
                if (ring) {
                    if (final) { upload_hybrid_kv(kv,device,state.bytes);layer.full_attention=std::move(kv); }
                    continue;
                }
                if ((parts.device>=0 && parts.device!=device) || (!parts.keys.empty() &&
                    (parts.keys.front().scalar_type()!=kv.k.scalar_type() || parts.keys.front().size(1)!=kv.k.size(1) ||
                    parts.keys.front().size(3)!=kv.k.size(3) || parts.values.front().size(3)!=kv.v.size(3))))
                    throw CudaSessionStateError("inconsistent CUDA hybrid prefix KV topology");
                parts.device=device;
                parts.keys.push_back(std::move(kv.k));
                parts.values.push_back(std::move(kv.v));
                if (final) {
                    layer.full_attention={join_hybrid_pages(parts.keys,state.cache_pos),
                        join_hybrid_pages(parts.values,state.cache_pos),capacity,false};
                    upload_hybrid_kv(layer.full_attention,device,state.bytes);
                    parts={};
                }
            } else if (layer.kind==Kind::Recurrent && (header.flags&1)) {
                auto [conv,device]=reader.tensor();
                auto [recurrent,recurrent_device]=reader.tensor();
                if (device!=recurrent_device)
                    throw CudaSessionStateError("inconsistent CUDA hybrid recurrence device");
                if (final) {
                    const tb::Device target(tb::kCUDA,device);
                    layer.convolution_state=conv.to(target);
                    layer.recurrent_state=recurrent.to(target);
                    state.bytes+=session_tensor_bytes(conv)+session_tensor_bytes(recurrent);
                }
            }
        }
        if (header.flags&2) {
            const auto count=reader.scalar<uint32_t>("predictor blocks");
            const bool full_predictor=(header.flags&4)!=0;
            const auto anchor_lag=int64_t(count!=0 && !full_predictor);
            if (count>4096 || (full_predictor && !count) || header.start+header.count<=anchor_lag)
                throw CudaSessionStateError("invalid CUDA hybrid prefix predictor geometry");
            MtpSessionState mtp;
            mtp.cache_pos=int64_t(header.start)+header.count-anchor_lag;
            auto [hidden,device]=reader.tensor();
            if (final) {
                mtp.last_target_hidden=hidden.to(tb::Device(tb::kCUDA,device));
                mtp.bytes=session_tensor_bytes(hidden);
            }
            for (uint32_t i=0;i<count;++i) {
                const auto capacity=reader.scalar<int64_t>("predictor KV capacity");
                const auto ring=reader.scalar<uint32_t>("predictor KV ring");
                if (ring>1) throw CudaSessionStateError("invalid CUDA hybrid prefix predictor KV mode");
                auto [kv,kv_device]=read_hybrid_kv(reader,capacity,ring?capacity:mtp.cache_pos,ring);
                if (final) { upload_hybrid_kv(kv,kv_device,mtp.bytes);mtp.blocks.push_back(std::move(kv)); }
            }
            if (final) { state.bytes+=mtp.bytes;state.mtp=std::move(mtp); }
        }
        reader.expect_end();
    }
    return state;
}
}

size_t cuda_paged_payload_tokens(const CudaPagedPayload& payload) {
    if(!payload)throw CudaSessionStateError("null CUDA prefix block");
    CudaPagedReader reader(*payload);std::array<uint8_t,8> magic{};reader.raw(magic.data(),magic.size(),"magic");
    constexpr std::array<uint8_t,8> full_magic{'M','F','Q','C','U','D','1',0};
    constexpr std::array<uint8_t,8> deepseek_magic{'M','F','Q','C','U','D','3',0};
    const auto version=reader.scalar<uint32_t>("version");
    if((magic!=qwen4_magic && magic!=full_magic && magic!=deepseek_magic && magic!=mfq::cuda::internal::hybrid_magic) ||
        (version!=1 && !(magic==qwen4_magic && version==2)))
        throw CudaSessionStateError("invalid CUDA prefix block header");
    (void)reader.scalar<uint32_t>("start");return reader.scalar<uint32_t>("count");
}
size_t cuda_paged_decodable_blocks(const std::vector<CudaPagedPayload>& payloads) {
    for(size_t count=payloads.size();count>0;--count) {
        const auto& payload=payloads[count-1];
        (void)cuda_paged_payload_tokens(payload);
        if((*payload)[6]=='1')return count;
        if((*payload)[6]=='3') {
            if(cuda_deepseek_v41_paged_exact(payload))return count;
            continue;
        }
        if((*payload)[6]=='4') {
            CudaPagedReader reader(*payload);
            if (mfq::cuda::internal::read_hybrid_header(reader).flags&1) return count;
            continue;
        }
        CudaPagedReader reader(*payload);if(read_qwen4_header(reader).exact)return count;
    }
    return 0;
}

namespace {
template<class Load>
TextSessionState decode_qwen4_pages(size_t pages,Load load,
    const std::vector<int64_t>& tokens,size_t block_size,
    const std::vector<std::shared_ptr<const mfq::cache::PagedPrefixReadLease>>& leases={}) {
    namespace tb=mfq_tensor_backend;
    TextSessionState state;state.tokens=tokens;state.cache_pos=int64_t(tokens.size());
    std::vector<Qwen4BlockSessionState> layers;
    std::vector<std::array<std::vector<tb::Tensor>,3>> fragments;
    size_t position=0;
    for(size_t i=0;i<pages;++i) {
        const auto payload=load(i);
        const auto lease=leases.empty()?nullptr:leases.at(i);
        if(!payload && (!lease || !lease->read_range_into))throw CudaSessionStateError("null Qwen4 prefix payload");
        auto reader=payload?CudaPagedReader(*payload):CudaPagedReader(*lease);
        const auto header=read_qwen4_header(reader);
        if(header.start!=position || header.count>block_size || (i+1<pages && header.count!=block_size) ||
            (i && header.layers!=layers.size()))throw CudaSessionStateError("incompatible Qwen4 prefix chain");
        if(!i){layers.resize(header.layers);fragments.resize(header.layers);}
        position+=header.count;
        for(size_t layer=0;layer<layers.size();++layer) {
            const auto packed=header.version==2?reader.template scalar<uint32_t>("QSA page kind"):0;
            if(packed>1)throw CudaSessionStateError("invalid QSA page kind");
            if(packed)read_ram_qsa_page(reader,lease,
                header.start,header.count,header.exact,layers[layer].adaptive_qsa);
            else for(size_t field=0;field<3;++field) {
                auto tensor=read_optional(reader);
                if(tensor.defined()) {
                    if(tensor.dim()!=3 || tensor.size(0)!=1 || tensor.size(1)!=header.count)
                        throw CudaSessionStateError("invalid Qwen4 QSA prefix geometry");
                    fragments[layer][field].push_back(std::move(tensor));
                }
            }
            if(header.exact) {
                layers[layer].convolution={};layers[layer].recurrent={};layers[layer].ple_convolution={};layers[layer].ple_tokens.clear();
                // The live request owns its own mutable recurrence. Keep the
                // archive on the host until copying each layer into that state.
                read_recurrent(reader,layers[layer],!leases.empty());
            }
        }
        if(header.exact) {
            state.qwen4_positions=read_optional(reader);
            state.mtp.reset();if(reader.template scalar<uint32_t>("MTP presence"))
                state.mtp=read_qwen4_mtp(reader,lease,header.version==2);
        }
        if(i+1==pages && !header.exact)throw CudaSessionStateError("Qwen4 prefix has no exact recurrent checkpoint");
        reader.expect_end();
    }
    if(!pages || position!=tokens.size() || !state.qwen4_positions.defined() || state.qwen4_positions.size(-1)!=int64_t(position))
        throw CudaSessionStateError("Qwen4 prefix boundary mismatch");
    for(size_t i=0;i<layers.size();++i) {
        auto& layer=layers[i];std::array<tb::Tensor*,3> fields{&layer.keys,&layer.values,&layer.index};
        for(size_t j=0;j<3;++j)if(!fragments[i][j].empty()) {
            if(fragments[i][j].size()!=pages)throw CudaSessionStateError("incomplete Qwen4 QSA prefix chain");
            *fields[j]=fragments[i][j].size()==1?fragments[i][j].front():tb::cat(fragments[i][j],1);
        }
        state.bytes+=qwen4_layer_bytes(layer);
    }
    state.bytes+=session_tensor_bytes(state.qwen4_positions)+(state.mtp?state.mtp->bytes:0);
    state.payload=std::move(layers);return state;
}
}

TextSessionState decode_cuda_qwen4_paged_session(const std::vector<CudaPagedPayload>& payloads,
    const std::vector<int64_t>& tokens,size_t block_size) {
    return decode_qwen4_pages(payloads.size(),[&](size_t i){return payloads[i];},tokens,block_size);
}

TextSessionState decode_cuda_qwen4_paged_prefix(mfq::cache::PagedPrefixCache& cache,
    std::vector<mfq::cache::BlockHash>& blocks,const std::vector<int64_t>& prompt,size_t block_size) {
    const bool trace=std::getenv("TYLOINFER_TRACE_QSA_TQ")!=nullptr;
    const auto begin=std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<const mfq::cache::PagedPrefixReadLease>> leases;
    std::vector<CudaPagedPayload> hot_only;
    std::vector<bool> sparse;
    size_t position=0,exact_pages=0,exact_tokens=0;
    const auto incoming=cache.borrow_prefix(blocks);
    // Check small headers first. The decoder then holds only one full page
    // at a time; retained file ranges never expand the context to FP16.
    for(const auto& block:blocks) {
        auto lease=incoming[leases.size()];
        CudaPagedPayload header;
        if(lease)header=std::make_shared<const std::vector<uint8_t>>(lease->read_range(0,std::min(size_t(28),lease->payload_bytes)));
        else {
            auto loaded=cache.load_prefix({block});if(loaded.empty())break;
            header=loaded.front();
        }
        CudaPagedReader reader(*header);const auto h=read_qwen4_header(reader);
        if(h.version!=2 || h.start!=position || h.count>block_size || h.count>prompt.size()-position)
            throw CudaSessionStateError("invalid compressed QSA prefix chain");
        position+=h.count;leases.push_back(lease);hot_only.push_back(lease?nullptr:header);
        // A retained, validated file version already has immutable offsets
        // for each packed QSA layer. Only read the framing between those layers.
        sparse.push_back(!h.exact && lease && lease->read_range_into &&
            mfq::cuda::internal::has_cached_ram_qsa_pages(lease));
        if(h.exact){exact_pages=leases.size();exact_tokens=position;}
        if(h.count!=block_size)break;
    }
    blocks.resize(exact_pages);leases.resize(exact_pages);hot_only.resize(exact_pages);
    if(!exact_pages)return {};
    std::vector<int64_t> tokens(prompt.begin(),prompt.begin()+exact_tokens);
    const auto validated=std::chrono::steady_clock::now();
    auto result=decode_qwen4_pages(exact_pages,[&](size_t i) {
        if(sparse[i])return CudaPagedPayload{};
        return leases[i]?std::make_shared<const std::vector<uint8_t>>(leases[i]->read_range(0,leases[i]->payload_bytes)):hot_only[i];
    },tokens,block_size,leases);
    if(trace)std::fprintf(stderr,"native_qsa_prefix_decode pages=%zu sparse_pages=%zu validate_ms=%.3f parse_ms=%.3f\n",
        exact_pages,size_t(std::count(sparse.begin(),sparse.begin()+exact_pages,true)),
        std::chrono::duration<double,std::milli>(validated-begin).count(),
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-validated).count());
    return result;
}

void refresh_cuda_qwen4_session_bytes(TextSessionState& state) {
    const auto* layers=std::get_if<std::vector<Qwen4BlockSessionState>>(&state.payload);if(!layers)return;
    if(state.mtp && !state.mtp->qwen4_blocks.empty()) {
        auto& head=*state.mtp;head.bytes=session_tensor_bytes(head.last_target_hidden);
        for(const auto& layer:head.qwen4_blocks)head.bytes+=qwen4_layer_bytes(layer);
        for(const auto& positions:head.qwen4_positions)head.bytes+=session_tensor_bytes(positions);
    }
    state.bytes=session_tensor_bytes(state.qwen4_positions)+(state.mtp?state.mtp->bytes:0);
    for(const auto& layer:*layers)state.bytes+=qwen4_layer_bytes(layer);
}
void prepare_cuda_qwen4_session_restore(const TextSessionState& state,bool enabled) {
    const auto* layers=std::get_if<std::vector<Qwen4BlockSessionState>>(&state.payload);if(!layers)return;
    for(const auto& layer:*layers)if(layer.adaptive_qsa)
        layer.adaptive_qsa->borrow_prefix=enabled && layer.adaptive_qsa->durable_prefix;
    if(state.mtp)for(const auto& layer:state.mtp->qwen4_blocks)if(layer.adaptive_qsa)
        layer.adaptive_qsa->borrow_prefix=enabled && layer.adaptive_qsa->durable_prefix;
}

void bind_cuda_qwen4_paged_prefix(const TextSessionState& state,mfq::cache::PagedPrefixCache& cache,
    const std::vector<mfq::cache::BlockHash>& blocks) {
    const auto* layers=std::get_if<std::vector<Qwen4BlockSessionState>>(&state.payload);
    if(!layers || std::none_of(layers->begin(),layers->end(),[](const auto& b){return bool(b.adaptive_qsa);}))return;
    const auto skip_optional=[](CudaPagedReader& r){if(r.scalar<uint32_t>("tensor presence"))(void)r.tensor();};
    size_t reused_pages=0,scanned_pages=0,page=0;
    for(const auto& hash:blocks) {
        const auto start=page++*cache.block_size_tokens();
        auto lease=cache.borrow(hash);if(!lease)continue;
        // Older complete pages already have immutable KV/index readers after
        // restore or a previous snapshot. Do not reread/reparse the full chain
        // merely to recreate those same readers. The final recurrent/MTP page
        // still follows the normal codec validation and binding path.
        if(page<blocks.size() && start<state.tokens.size()) {
            const auto count=std::min(cache.block_size_tokens(),state.tokens.size()-start);
            const auto ready=std::all_of(layers->begin(),layers->end(),[&](const auto& layer) {
                return !layer.adaptive_qsa || mfq::cuda::internal::reuse_ram_qsa_page(*layer.adaptive_qsa,start,count,lease,false);
            });
            if(ready) {
                for(const auto& layer:*layers)if(layer.adaptive_qsa)
                    (void)mfq::cuda::internal::reuse_ram_qsa_page(*layer.adaptive_qsa,start,count,lease,true);
                ++reused_pages;continue;
            }
        }
        ++scanned_pages;
        auto payload=std::make_shared<const std::vector<uint8_t>>(lease->read_range(0,lease->payload_bytes));
        CudaPagedReader reader(*payload);const auto h=read_qwen4_header(reader);
        if(h.version!=2 || h.layers!=layers->size())continue;
        for(const auto& layer:*layers) {
            const auto kind=reader.scalar<uint32_t>("QSA page kind");
            if(kind>1)throw CudaSessionStateError("invalid QSA page kind");
            if(kind) {
                auto snapshot=layer.adaptive_qsa;
                read_ram_qsa_page(reader,lease,h.start,h.count,h.exact,snapshot,true);
            }else for(int field=0;field<3;++field)skip_optional(reader);
            if(h.exact) {
                for(int field=0;field<3;++field)skip_optional(reader);
                (void)reader.scalar<int64_t>("PLE batch");const auto n=reader.scalar<uint32_t>("PLE history length");
                if(n>65536)throw CudaSessionStateError("invalid Qwen4 PLE history");reader.skip(size_t(n)*8);
            }
        }
        if(h.exact && state.mtp) {
            skip_optional(reader);
            if(!reader.scalar<uint32_t>("MTP presence"))continue;
            (void)reader.scalar<int64_t>("MTP position");skip_optional(reader);
            const auto n=reader.scalar<uint32_t>("MTP layers");
            if(n!=state.mtp->qwen4_blocks.size())throw CudaSessionStateError("incompatible MTP prefix layers");
            for(const auto& layer:state.mtp->qwen4_blocks) {
                const auto kind=reader.scalar<uint32_t>("MTP QSA page kind");
                if(kind>1)throw CudaSessionStateError("invalid MTP QSA page kind");
                if(kind) {
                    const auto count=reader.scalar<int64_t>("MTP compressed position");
                    if(count<=0 || count>state.cache_pos)throw CudaSessionStateError("invalid MTP compressed position");
                    auto snapshot=layer.adaptive_qsa;
                    read_ram_qsa_page(reader,lease,0,size_t(count),true,snapshot,true);
                }else for(int field=0;field<3;++field)skip_optional(reader);
                for(int field=0;field<3;++field)skip_optional(reader);
                (void)reader.scalar<int64_t>("PLE batch");const auto history=reader.scalar<uint32_t>("PLE history length");
                if(history>65536)throw CudaSessionStateError("invalid Qwen4 PLE history");reader.skip(size_t(history)*8);
                skip_optional(reader);(void)reader.scalar<int64_t>("MTP layer position");
            }
        }
    }
    if(std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
        "native_qsa_prefix_rebind actual_match=1 reused_pages=%zu scanned_pages=%zu\n",reused_pages,scanned_pages);
}



std::vector<CudaPagedPayload> encode_cuda_paged_session(const TextSessionState &state,
                                                        size_t block_size, size_t first_block,
                                                        size_t maximum_blocks) {
    static constexpr std::array<uint8_t, 8> magic{'M', 'F', 'Q', 'C', 'U', 'D', '1', 0};
    const auto *layers = std::get_if<std::vector<FullBlockSessionState>>(&state.payload);
    // Paged prefixes are text-only; media positions are not block-sliceable.
    if (state.kind() != TextSessionStateKind::FullAttention || state.decode_position_delta != 0 ||
        !state.input_key.empty() || state.cache_pos <= 0 || layers == nullptr || layers->empty() ||
        state.tokens.size() != static_cast<size_t>(state.cache_pos)) {
        throw std::runtime_error("CUDA session state is not block-sliceable");
    }
    const size_t full_blocks = state.tokens.size() / block_size;
    if (first_block > full_blocks) {
        throw std::runtime_error("invalid CUDA cache first block");
    }
    const size_t end_block =
        maximum_blocks > full_blocks - first_block ? full_blocks : first_block + maximum_blocks;
    std::vector<CudaPagedPayload> result;
    result.reserve(end_block - first_block);
    for (size_t block_index = first_block; block_index < end_block; ++block_index) {
        const int64_t start = static_cast<int64_t>(block_index * block_size);
        CudaPagedWriter writer;
        writer.raw(magic.data(), magic.size());
        writer.scalar<uint32_t>(1);
        writer.scalar<uint32_t>(static_cast<uint32_t>(start));
        writer.scalar<uint32_t>(static_cast<uint32_t>(block_size));
        writer.scalar<uint32_t>(static_cast<uint32_t>(layers->size()));
        for (const auto &layer : *layers) {
            if (layer.ring || layer.capacity <= 0 || !layer.k.defined() || !layer.v.defined() ||
                !session_kv_geometry_matches(layer.k,layer.v) ||
                layer.k.size(2) < start + static_cast<int64_t>(block_size)) {
                throw std::runtime_error("CUDA session layer cannot be block-sliced");
            }
            writer.scalar<int64_t>(layer.capacity);
            writer.tensor(layer.k.narrow(2, start, static_cast<int64_t>(block_size)));
            writer.tensor(layer.v.narrow(2, start, static_cast<int64_t>(block_size)));
        }
        result.push_back(std::move(writer).finish());
    }
    return result;
}

CudaPagedPayload encode_cuda_paged_block(const TextSessionState &state, size_t block_size,
                                         size_t block_index) {
    auto result = encode_cuda_paged_session(state, block_size, block_index, 1);
    if (result.size() != 1) {
        throw std::runtime_error("invalid CUDA cache block index");
    }
    return std::move(result.front());
}

CudaDecodedPagedBlock decode_cuda_paged_block(const std::vector<uint8_t> &payload) {
    static constexpr std::array<uint8_t, 8> magic{'M', 'F', 'Q', 'C', 'U', 'D', '1', 0};
    CudaPagedReader reader(payload);
    std::array<uint8_t, 8> found_magic{};
    reader.raw(found_magic.data(), found_magic.size(), "magic");
    if (found_magic != magic || reader.scalar<uint32_t>("version") != 1) {
        throw CudaSessionStateError("unsupported CUDA paged cache payload");
    }
    CudaDecodedPagedBlock block;
    block.start = reader.scalar<uint32_t>("start");
    block.count = reader.scalar<uint32_t>("count");
    const auto layer_count = reader.scalar<uint32_t>("layer count");
    if (block.count == 0 || layer_count == 0 || layer_count > 4096) {
        throw CudaSessionStateError("invalid CUDA paged cache geometry");
    }
    block.layers.reserve(layer_count);
    for (uint32_t index = 0; index < layer_count; ++index) {
        CudaDecodedPagedLayer layer;
        layer.capacity = reader.scalar<int64_t>("capacity");
        auto key = reader.tensor();
        auto value = reader.tensor();
        if (key.second != value.second || !session_kv_geometry_matches(key.first,value.first) ||
            key.first.size(2) != static_cast<int64_t>(block.count)) {
            throw CudaSessionStateError("invalid CUDA paged cache layer");
        }
        layer.device = key.second;
        layer.k = std::move(key.first);
        layer.v = std::move(value.first);
        block.layers.push_back(std::move(layer));
    }
    reader.expect_end();
    return block;
}

TextSessionState decode_cuda_paged_session(const std::vector<CudaPagedPayload> &payloads,
                                           const std::vector<int64_t> &tokens, size_t block_size) {
    if (payloads.empty() || tokens.size() != payloads.size() * block_size) {
        throw CudaSessionStateError("CUDA paged cache chain length mismatch");
    }
    std::vector<CudaDecodedPagedBlock> blocks;
    blocks.reserve(payloads.size());
    size_t expected_start = 0;
    size_t layer_count = 0;
    for (const auto &payload : payloads) {
        if (!payload) {
            throw CudaSessionStateError("CUDA paged cache block payload is null");
        }
        auto block = decode_cuda_paged_block(*payload);
        if (block.start != expected_start || block.count != block_size ||
            (layer_count != 0 && block.layers.size() != layer_count)) {
            throw CudaSessionStateError("incompatible CUDA paged cache chain");
        }
        expected_start += block.count;
        layer_count = block.layers.size();
        blocks.push_back(std::move(block));
    }
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = static_cast<int64_t>(tokens.size());
    state.payload = std::vector<FullBlockSessionState>{};
    auto &layers = std::get<std::vector<FullBlockSessionState>>(state.payload);
    layers.reserve(layer_count);
    for (size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        const auto &final = blocks.back().layers[layer_index];
        std::vector<mfq_tensor_backend::Tensor> keys;
        std::vector<mfq_tensor_backend::Tensor> values;
        keys.reserve(blocks.size());
        values.reserve(blocks.size());
        for (const auto &block : blocks) {
            const auto &layer = block.layers[layer_index];
            if (layer.capacity != final.capacity || layer.device != final.device ||
                layer.k.scalar_type() != final.k.scalar_type() ||
                layer.k.size(0) != final.k.size(0) || layer.k.size(1) != final.k.size(1) ||
                layer.k.size(3) != final.k.size(3) || layer.v.size(3)!=final.v.size(3)) {
                throw CudaSessionStateError("inconsistent CUDA paged cache topology");
            }
            keys.push_back(layer.k);
            values.push_back(layer.v);
        }
        auto key = mfq_tensor_backend::cat(keys, 2).contiguous();
        auto value = mfq_tensor_backend::cat(values, 2).contiguous();
        MfqCudaGuard guard(final.device);
        const auto options = key.options().device(
            mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, final.device));
        key = key.to(options, true, false).contiguous();
        value = value.to(options, true, false).contiguous();
        state.bytes += static_cast<size_t>(key.numel() * key.element_size() +
                                           value.numel() * value.element_size());
        layers.push_back(FullBlockSessionState{
            std::move(key),
            std::move(value),
            final.capacity,
            false,
        });
    }
    return state;
}

size_t session_tensor_bytes(const mfq_tensor_backend::Tensor &tensor) {
    if (!tensor.defined())
        return 0;
    return static_cast<size_t>(tensor.numel()) * tensor.element_size();
}

bool session_tensor_layout_matches(const mfq_tensor_backend::Tensor &target,
                                   const mfq_tensor_backend::Tensor &source) {
    return target.defined() && source.defined() && target.sizes() == source.sizes() &&
           target.scalar_type() == source.scalar_type() && target.device() == source.device();
}

void restore_session_tensor(mfq_tensor_backend::Tensor &target,
                            const mfq_tensor_backend::Tensor &source) {
    if (!source.defined()) {
        target = mfq_tensor_backend::Tensor();
        return;
    }
    if (!session_tensor_layout_matches(target, source)) {
        target = mfq_tensor_backend::empty(source.sizes(), source.options());
    }
    target.copy_(source);
}

void restore_session_prefix_tensor(mfq_tensor_backend::Tensor &target,
                                   const mfq_tensor_backend::Tensor &source, int64_t dimension,
                                   int64_t capacity) {
    if (!source.defined() || dimension < 0 || dimension >= source.dim() || capacity <= 0 ||
        source.size(dimension) > capacity) {
        throw CudaSessionStateError("session prefix tensor layout is invalid");
    }
    auto target_shape = source.sizes().vec();
    target_shape.at(static_cast<size_t>(dimension)) = capacity;
    const bool compatible =
        target.defined() && target.sizes() == mfq_tensor_backend::IntArrayRef(target_shape) &&
        target.scalar_type() == source.scalar_type() && target.device() == source.device();
    if (!compatible) {
        target = mfq_tensor_backend::empty(target_shape, source.options());
    }
    if (source.size(dimension) > 0) {
        target.narrow(dimension, 0, source.size(dimension)).copy_(source);
    }
}

FullBlockSessionState capture_full_attention_session_state(const FullBlock &block,
                                                           int64_t cache_pos, size_t &bytes) {
    const auto& cache=block.cache;
    if (cache.is_paged() && cache.defined() && cache.paged_batch==1 && cache_pos>0 &&
        cache_pos<=cache.page_table.size(1)*cache.page_size) {
        namespace tb=mfq_tensor_backend;
        const auto quantization=cache.quantization;
        const auto dimension=cache.paged_head_dim;
        const auto key_width=quantization.enabled()?mfq::cuda::cuda_kv_packed_width(int(dimension),quantization.key_bits()):dimension;
        const auto value_width=quantization.enabled()?mfq::cuda::cuda_kv_packed_width(int(dimension),quantization.value_bits()):dimension;
        const auto options=cache.page_table.options().dtype(quantization.enabled()?tb::kInt32:cache.paged_dtype);
        auto keys=tb::empty({1,cache.paged_heads,cache_pos,key_width},options);
        auto values=tb::empty({1,cache.paged_heads,cache_pos,value_width},options);
        paged_kv_cache_gather_cuda(cache.k_chunk_ptrs,cache.v_chunk_ptrs,cache.page_table,keys,values,
            cache.page_size,cache.pages_per_chunk);
        bytes+=session_tensor_bytes(keys)+session_tensor_bytes(values);
        return {std::move(keys),std::move(values),cache_pos,false};
    }
    if (!block.cache.k.defined() || !block.cache.v.defined() || block.cache.k.dim() != 4 ||
        !session_kv_geometry_matches(block.cache.k,block.cache.v) || block.cache.k.size(0) != 1) {
        throw std::runtime_error("full-attention KV cache is unavailable");
    }
    FullBlockSessionState state;
    state.capacity = block.cache.k.size(2);
    state.ring = block.cache.ring;
    const int64_t saved_tokens =
        state.ring ? state.capacity : std::min<int64_t>(cache_pos, state.capacity);
    state.k = block.cache.k.narrow(2, 0, saved_tokens).clone();
    state.v = block.cache.v.narrow(2, 0, saved_tokens).clone();
    bytes += session_tensor_bytes(state.k);
    bytes += session_tensor_bytes(state.v);
    return state;
}

void restore_full_attention_session_state(FullBlock &block, const FullBlockSessionState &state) {
    if (!state.k.defined() || !state.v.defined() || state.capacity <= 0 || state.k.dim() != 4 ||
        !session_kv_geometry_matches(state.k,state.v) || state.k.size(0) != 1 ||
        state.k.size(2) > state.capacity) {
        throw CudaSessionStateError("full-attention session KV layout is invalid");
    }
    restore_session_prefix_tensor(block.cache.k, state.k, 2, state.capacity);
    restore_session_prefix_tensor(block.cache.v, state.v, 2, state.capacity);
    block.cache.ring = state.ring;
    block.cache.quantization = block.kv_quantization;
}
