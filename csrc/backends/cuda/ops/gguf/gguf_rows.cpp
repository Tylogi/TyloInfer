#include "gguf_weight.h"
#include "gguf_rotation.h"
#include "tyloinfer/row_residual.h"
#include "tyloinfer/encoded_row_cache.h"
#include "tyloinfer/mapped_prefetch_scope.h"
#include "strata/ngram/ple_reader.hpp"
#include "runtime/row_stage_fence.h"
#include "runtime/row_packet_prefetch.h"
#include "../../kernels/gguf.h"
#include "mfq_weight_memory.h"
#include "mfq_cuda_resident_allocations.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <list>
#include <mutex>
#include <unordered_map>
#include <future>

namespace tb = mfq_tensor_backend;

struct GgufRowTable::RowCache {
    std::mutex mutex;
    using Cache=mfq::EncodedRowCache<std::vector<std::byte>>;
    std::shared_ptr<Cache> rows;
    bool parallel=false;
    std::unique_ptr<strata::ngram::PleReader> direct_prefill;
    explicit RowCache(std::size_t bytes)
        :rows(std::make_shared<Cache>(bytes,[](const auto& row){return row.size();},
            [](std::size_t size,bool retained) {
                if(retained)mfq::cuda::charge_tensor_host_bytes(size);
                else mfq::cuda::tensor_host_bytes.fetch_sub(size);
            })) {mfq::register_host_cache(rows);}
};

GgufRowTable::GgufRowTable(const mfq::ModelSource& source, const std::string& name) {
    const auto* record = source.find_tensor(name);
    if (!record)throw std::invalid_argument("missing row table: " + name);
    auto shape=record->shape;
    std::uint64_t payload_offset=0;
    const bool floating=record->dtype=="F16" || record->dtype=="BF16" || record->dtype=="F32";
    if(!shape && floating) {
        // MFQ dense records embed rank and dimensions before their row data.
        // Read only this header; PLE tables must never be materialized here.
        std::array<std::byte,20> header{};
        if(record->nbytes<header.size())throw std::invalid_argument("truncated floating row table: "+name);
        source.read_range_into(name,0,header.data(),header.size());
        const auto integer=[&](std::size_t start,std::size_t bytes) {
            std::uint64_t value=0;
            for(std::size_t i=0;i<bytes;++i)value|=std::uint64_t(std::to_integer<unsigned char>(header[start+i]))<<(8*i);
            return value;
        };
        if(integer(0,4)!=2 || integer(4,8)>INT64_MAX || integer(12,8)>INT64_MAX)
            throw std::invalid_argument("floating row table requires a matrix: "+name);
        shape=std::vector<int64_t>{int64_t(integer(4,8)),int64_t(integer(12,8))};payload_offset=header.size();
    }
    if(!shape || shape->size()!=2)throw std::invalid_argument("row table requires a matrix: "+name);
    type_ = record->dtype=="F16"?GGML_TYPE_F16:record->dtype=="BF16"?GGML_TYPE_BF16:
        record->dtype=="F32"?GGML_TYPE_F32:mfq::gguf::quant_type(record->dtype);
    rows_ = shape->at(0); columns_ = shape->at(1);
    if(load_gguf_input_rotation(source,name,columns_,0,columns_,0,false))
        throw std::invalid_argument("GGUF streaming row tables do not yet support an inverse Hadamard transform");
    row_bytes_ = mfq::gguf::row_bytes(type_, columns_);
    if (rows_ <= 0 || std::uint64_t(rows_) > UINT64_MAX / row_bytes_ ||
            record->nbytes-payload_offset != std::uint64_t(rows_) * row_bytes_)
        throw std::invalid_argument("invalid GGUF row table payload: " + name);
    // A source-provided batch reader owns its I/O scheduling and staging.
    // Use mapped sparse reads when that capability is absent.
    auto batch=source.tensor_batch_reader(name);
    auto reader=batch?mfq::ModelSource::TensorReader{}:source.mapped_tensor_reader(name);
    const bool mapped=bool(reader);
    if(!reader)reader=source.tensor_reader(name);
    read_ = std::make_shared<mfq::ModelSource::TensorReader>(
        [reader=std::move(reader),payload_offset](std::uint64_t offset,std::byte* destination,std::size_t bytes) {
            reader(payload_offset+offset,destination,bytes);
        });
    if(mapped)batch=source.mapped_tensor_batch_reader(name);
    if(batch)batch_=[batch=std::move(batch),payload_offset](const auto& requests) {
        auto adjusted=requests;for(auto& request:adjusted)request.offset+=payload_offset;batch(adjusted);
    };
    correction_=source.row_residual(name);
    if(correction_ && !correction_->payload_bytes())correction_.reset();
    std::unique_ptr<strata::ngram::PleReader> direct_prefill;
    if(!correction_ && std::uint64_t(rows_)<=UINT32_MAX && row_bytes_<=strata::ngram::PAGE) {
        if(const auto range=source.tensor_file_range(name);range && range->read_mode==mfq::FileReadMode::Direct) {
            auto direct=std::make_unique<strata::ngram::PleReader>();
            const auto path=range->path.u8string();
            std::string error;
            if(direct->open(std::string(path.begin(),path.end()),range->offset+payload_offset,rows_,256,0,error,true,row_bytes_)) {
                direct->set_keepalive(100,60);
                direct_prefill=std::move(direct);
            }
        }
    }
    // Direct I/O has no OS page cache. Retain recently used encoded rows
    // in the reclaimable cache and charge them only when first read.
    const auto cache_bytes=direct_prefill?std::clamp<std::uint64_t>(record->nbytes/64,8ULL<<20,64ULL<<20):8ULL<<20;
    cache_ = std::make_unique<RowCache>(cache_bytes);
    cache_->parallel=source.supports_parallel_tensor_reads() || (mapped && !correction_);
    cache_->direct_prefill=std::move(direct_prefill);
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<unsigned> reported{0};
        const auto bit=mapped?1u:batch_?2u:4u;
        if(!(reported.fetch_or(bit)&bit))std::fprintf(stderr,
            "native_gguf_sparse_rows actual_match=1 mapped=%d batch=%d parallel=%d\n",
            int(mapped),int(bool(batch_)),int(cache_->parallel));
    }
    mfq::cuda::observe_streamed_ple(read_, record->nbytes);
}
GgufRowTable::~GgufRowTable()=default;
bool GgufRowTable::parallel_reads()const noexcept{return cache_->parallel;}

void GgufRowTable::read_row(std::int64_t id, void* destination) const {
    if (id < 0 || id >= rows_) throw std::out_of_range("GGUF table row id");
    read_rows({{std::uint64_t(id)*row_bytes_,static_cast<std::byte*>(destination),std::size_t(row_bytes_)}});
}
void GgufRowTable::decode_row(std::int64_t id,float* output) const {
    std::vector<std::uint8_t> base(row_bytes_);read_row(id,base.data());
    decode_gguf_cpu(type_,base.data(),output,columns_);
    if(correction_) {
        const auto& row=correction_->rows().at(std::size_t(id));
        if(row.type) {
            auto packed=correction_->read_row(id);
            std::vector<float> values((columns_+127)/128*128);
            decode_gguf_cpu(ggml_type(row.type),packed.data(),values.data(),std::int64_t(values.size()));
            for(std::int64_t column=0;column<columns_;++column)output[column]+=values[column];
        }
    }
}
void GgufRowTable::read_staged_rows(const std::vector<mfq::ReadSpan>& rows) const {
    if(!correction_) {read_rows(rows);return;}
    std::vector<float> values(columns_);
    for(const auto& row:rows) {
        if(row.offset%row_bytes_ || row.offset/row_bytes_>=std::uint64_t(rows_) ||
            row.size!=staged_row_bytes() || !row.destination)
            throw std::invalid_argument("invalid corrected GGUF row request");
        decode_row(std::int64_t(row.offset/row_bytes_),values.data());
        auto* output=reinterpret_cast<ggml_fp16_t*>(row.destination);
        for(std::int64_t column=0;column<columns_;++column)output[column]=ggml_fp32_to_fp16(values[column]);
    }
}
void GgufRowTable::read_rows(const std::vector<mfq::ReadSpan>& requests) const {
    std::lock_guard lock(cache_->mutex);
    std::vector<mfq::ReadSpan> missing;
    std::unordered_map<std::uint64_t,std::byte*> unique;
    std::vector<std::pair<std::byte*,const std::byte*>> repeated;
    for(const auto& row:requests) {
        if(row.size!=row_bytes_ || row.offset%row_bytes_ || row.offset/row_bytes_>=std::uint64_t(rows_) || !row.destination)
            throw std::out_of_range("invalid GGUF table row range");
        if(const auto found=cache_->rows->find(row.offset)) {
            std::memcpy(row.destination,found->data(),row_bytes_);
        }else if(auto [first,inserted]=unique.emplace(row.offset,row.destination);inserted) missing.push_back(row);
        else repeated.emplace_back(row.destination,first->second);
    }
    std::sort(missing.begin(),missing.end(),[](const auto& a,const auto& b){return a.offset<b.offset;});
    const auto count=std::max<std::size_t>(1,std::min<std::size_t>(256,(8ULL<<20)/std::max<std::uint64_t>(4096,row_bytes_)));
    const mfq::MappedPrefetchScope copied(missing.size()>128,true);
    bool read_direct=false;
    if(cache_->direct_prefill && missing.size()>=2*count) {
        auto& direct=*cache_->direct_prefill;
        const auto* dispatch=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");
        const bool trace=dispatch && dispatch[0]=='1';
        const auto before=trace?direct.snapshot().reads:0;
        const auto start=std::chrono::steady_clock::now();
        std::vector<std::uint32_t> ids;
        ids.reserve(missing.size());
        for(const auto& row:missing)ids.push_back(static_cast<std::uint32_t>(row.offset/row_bytes_));
        std::vector<std::uint8_t> packed(ids.size()*row_bytes_);
        std::string error;
        read_direct=direct.collect(direct.issue(ids.data(),ids.size(),packed.data()),error);
        if(read_direct) {
            for(std::size_t i=0;i<missing.size();++i)
                std::memcpy(missing[i].destination,packed.data()+i*row_bytes_,row_bytes_);
            if(trace)std::fprintf(stderr,"native_ple_reference_io rows=%zu reads=%llu elapsed_ms=%.3f\n",
                ids.size(),static_cast<unsigned long long>(direct.snapshot().reads-before),
                std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
        } else cache_->direct_prefill.reset();
    }
    for(std::size_t begin=0;!read_direct && begin<missing.size();begin+=count) {
        const auto end=std::min(missing.size(),begin+count);
        if(batch_)batch_({missing.begin()+begin,missing.begin()+end});
        else for(auto i=begin;i<end;++i)(*read_)(missing[i].offset,missing[i].destination,missing[i].size);
    }
    for(const auto& [destination,source]:repeated)std::memcpy(destination,source,row_bytes_);
    // Physical reads are sorted, but retention follows the caller's token
    // order so a long prompt leaves its most recent rows hot for decode.
    for(const auto& row:requests) {
        if(cache_->rows->find(row.offset))continue;
        cache_->rows->insert(row.offset,std::vector<std::byte>(row.destination,row.destination+row_bytes_));
    }
}

RowStageResources gguf_row_stage_resources(const std::vector<std::shared_ptr<GgufRowTable>>& tables,
        const std::vector<std::int64_t>& shape,const tb::Device& device) {
    if(tables.empty() || !tables[0] || !device.is_cuda())throw std::invalid_argument("Invalid GGUF row workspace");
    std::uint64_t count=1;std::size_t stride=0;
    for(const auto size:shape) {
        if(size<=0 || count>std::uint64_t(INT64_MAX)/std::uint64_t(size))throw std::overflow_error("GGUF row workspace shape");
        count*=size;
    }
    for(const auto& table:tables) {
        if(!table || table->width()!=tables[0]->width())throw std::invalid_argument("GGUF row workspace widths differ");
        stride=std::max(stride,std::size_t(table->staged_row_bytes()));
    }
    if(stride>SIZE_MAX-7)throw std::overflow_error("GGUF row workspace stride");
    stride=(stride+7)&~std::size_t(7);
    if(stride>SIZE_MAX-sizeof(mfq::cuda::gguf::StagedRow) ||
            count>std::uint64_t(INT64_MAX)/(stride+sizeof(mfq::cuda::gguf::StagedRow)))
        throw std::overflow_error("GGUF row workspace packet");
    const auto packet=std::size_t(count)*(stride+sizeof(mfq::cuda::gguf::StagedRow));
    const auto width=std::uint64_t(tables[0]->width());
    if(width>SIZE_MAX/2 || count>SIZE_MAX/(2*width))throw std::overflow_error("GGUF row workspace output");
    const auto output=std::size_t(count)*std::size_t(width)*2;
    MfqCudaGuard guard(device);int can_map=0;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(&can_map,cudaDevAttrCanMapHostMemory,device.index));
    const bool mapped=can_map && packet<=16*1024;
    if(packet>SIZE_MAX/2 || (!mapped && packet>SIZE_MAX-output))throw std::overflow_error("GGUF row workspace total");
    return {"gguf_ple_rows",count,width,2*packet,packet,output+(mapped?0:packet)};
}

namespace {
class GgufRowStage final : public RowStage {
    std::vector<std::shared_ptr<GgufRowTable>> tables_;
    std::vector<std::int64_t> shape_;
    std::int64_t rows_per_shard_, count_ = 1, width_;
    std::size_t stride_ = 0, offset_, bytes_;
    tb::Device device_;
    // Tensor release may record fences on every observed stream. Keep these
    // owners alive until both the tensor allocations and host packet are gone.
    std::vector<MfqCudaStream> streams_;
    std::unique_ptr<mfq::cuda::HostBuffer> host_;
    tb::Tensor packet_, output_;
    void* packet_address_ = nullptr;
    bool mapped_ = false;
    mfq::cuda::RowStageFence fence_;
    mfq::cuda::RowPacketPrefetch prefetch_;
    void retain_stream() {
        const auto stream = mfq_get_current_cuda_stream(device_.index);
        if (std::none_of(streams_.begin(), streams_.end(), [&](const auto& prior) { return prior.stream() == stream.stream(); }))
            streams_.push_back(stream);
    }
public:
    GgufRowStage(std::vector<std::shared_ptr<GgufRowTable>> tables, std::int64_t rows,
        std::vector<std::int64_t> shape, tb::Device device)
        : tables_(std::move(tables)), shape_(std::move(shape)), rows_per_shard_(rows), device_(device) {
        if (tables_.empty() || !tables_[0] || rows <= 0 || !device.is_cuda())
            throw std::invalid_argument("invalid GGUF row stage");
        width_ = tables_[0]->width();
        for (const auto size : shape_) {
            if (size <= 0 || count_ > INT64_MAX / size) throw std::overflow_error("GGUF row stage shape");
            count_ *= size;
        }
        for (const auto& table : tables_) {
            if (!table || table->width() != width_) throw std::invalid_argument("GGUF table widths differ");
            stride_ = std::max(stride_, static_cast<std::size_t>(table->staged_row_bytes()));
        }
        stride_ = (stride_ + 7) & ~std::size_t(7);
        if (std::uint64_t(count_) > std::uint64_t(INT64_MAX) /
                (sizeof(mfq::cuda::gguf::StagedRow) + stride_))
            throw std::overflow_error("GGUF row stage packet size");
        offset_ = std::size_t(count_) * sizeof(mfq::cuda::gguf::StagedRow);
        bytes_ = offset_ + std::size_t(count_) * stride_;
        MfqCudaGuard guard(device_);
        retain_stream();
        fence_.initialize();
        int can_map = 0;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, device_.index));
        // As in the TQ row stage, a small packet is cheaper to read directly
        // than to submit a separate host-to-device transfer for every layer.
        mapped_ = can_map && bytes_ <= 16 * 1024;
        host_ = std::make_unique<mfq::cuda::HostBuffer>(bytes_, mapped_);
        if (mapped_) TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&packet_address_, host_->data(), 0));
        else {
            packet_ = tb::empty({static_cast<std::int64_t>(bytes_)},
                tb::TensorOptions().device(device_).dtype(tb::kUInt8));
            packet_address_ = packet_.data_ptr();
        }
        auto output_shape = shape_; output_shape.push_back(width_);
        output_ = tb::zeros(output_shape, tb::TensorOptions().device(device_).dtype(tb::kFloat16));
        std::memset(host_->data(), 0, bytes_);
        auto* descriptors = static_cast<mfq::cuda::gguf::StagedRow*>(host_->data());
        for (std::int64_t i = 0; i < count_; ++i)
            descriptors[i] = {offset_ + i * stride_, std::int32_t(tables_[0]->staged_type()), 0};
        if (!mapped_) TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(packet_address_, host_->data(), bytes_,
            cudaMemcpyHostToDevice, mfq_current_cuda_stream()));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        mfq::cuda::charge_tensor_host_bytes(bytes_);
        if(!mapped_)if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true))std::fprintf(stderr,
                "native_gguf_row_stage actual_match=1 rows=%lld width=%lld host_bytes=%zu output_bytes=%zu mapped=0\n",
                static_cast<long long>(count_),static_cast<long long>(width_),bytes_,output_.nbytes());
        }
    }
    ~GgufRowStage() override {
        try {
            MfqCudaGuard guard(device_);
            for (const auto& stream : streams_) (void)cudaStreamSynchronize(stream.stream());
        } catch (...) {}
        mfq::cuda::tensor_host_bytes.fetch_sub(bytes_);
    }
    tb::Tensor output() const override { return output_; }
    bool mapped_rows() const override { return mapped_; }
    bool prefetch(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape)override {
        if(std::any_of(tables_.begin(),tables_.end(),[](const auto& table){return !table->parallel_reads();}))return false;
        prefetch_.start(ids,shape,bytes_,[this,ids,shape](void* packet) {
            prepare_rows_to(ids,shape,packet);return bytes_;
        });return true;
    }
    void upload(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) override {
        MfqCudaGuard guard(device_);
        retain_stream();
        fence_.before_write();
        prepare_rows(ids,shape);
        upload_prepared();
    }
    void upload_graph(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape) override {
        if(!mapped_){upload(ids,shape);return;}
        // The graph owner already fences the preceding read. A mapped packet
        // needs only CPU writes and the caller's publication flag, with no
        // event recorded on an otherwise idle CUDA stream during replay.
        fence_.owner_fenced();prepare_rows(ids,shape);
    }
    // Runs on a CPU I/O worker. All storage/events were prepared by the
    // model thread; this method never enters CUDA or changes GPU ownership.
    void prepare_rows(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape) {
        if(!prefetch_.consume(ids,shape,host_->data()))prepare_rows_to(ids,shape,host_->data());
    }
    void prepare_rows_to(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape,void* packet) {
        if (shape != shape_ || ids.size() != static_cast<std::size_t>(count_))
            throw std::invalid_argument("GGUF row stage upload shape differs");
        auto* descriptors = static_cast<mfq::cuda::gguf::StagedRow*>(packet);
        auto* bytes = static_cast<std::uint8_t*>(packet);
        std::vector<std::vector<mfq::ReadSpan>> requests(tables_.size());
        for (std::int64_t i = 0; i < count_; ++i) {
            const auto id = ids[i];
            if (id < 0 || std::uint64_t(id / rows_per_shard_) >= tables_.size())
                throw std::out_of_range("GGUF row stage shard id");
            const auto& table = tables_[id / rows_per_shard_];
            descriptors[i].offset=offset_+i*stride_;
            descriptors[i].reserved=0;
            descriptors[i].type = table->staged_type();
            requests[id/rows_per_shard_].push_back({std::uint64_t(id%rows_per_shard_)*table->row_bytes(),
                reinterpret_cast<std::byte*>(bytes+descriptors[i].offset),std::size_t(table->staged_row_bytes())});
        }
        std::vector<std::size_t> active;
        bool parallel=true;
        for(std::size_t shard=0;shard<tables_.size();++shard)if(!requests[shard].empty()) {
            active.push_back(shard);parallel=parallel && tables_[shard]->parallel_reads();
        }
        const auto read=[&](std::int64_t begin,std::int64_t end) {
            for(auto index=begin;index<end;++index) {
                const auto shard=active[index];tables_[shard]->read_staged_rows(requests[shard]);
            }
        };
        if(parallel)mfq_parallel_for(0,static_cast<std::int64_t>(active.size()),1,read);
        else read(0,static_cast<std::int64_t>(active.size()));
    }
    void upload_prepared() {
        MfqCudaGuard guard(device_);
        retain_stream();
        if (!mapped_) TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(packet_address_, host_->data(), bytes_,
            cudaMemcpyHostToDevice, mfq_current_cuda_stream()));
        fence_.record_transfer(mfq_current_cuda_stream());
        if (packet_.defined()) packet_.record_stream(reinterpret_cast<std::uintptr_t>(mfq_current_cuda_stream()));
    }
    void decode() override {
        MfqCudaGuard guard(device_);
        retain_stream();
        const auto stream = mfq_current_cuda_stream();
        const auto capture=fence_.join(stream);
        if (packet_.defined()) packet_.record_stream(reinterpret_cast<std::uintptr_t>(stream));
        output_.record_stream(reinterpret_cast<std::uintptr_t>(stream));
        mfq::cuda::gguf::staged_rows(packet_address_, count_, width_, output_.data_ptr(), true,
            mfq_current_cuda_stream());
        fence_.record_consumer(stream,capture);
    }
};
}

std::shared_ptr<RowStage> make_gguf_row_stage(const std::vector<std::shared_ptr<GgufRowTable>>& tables,
    std::int64_t rows, const std::vector<std::int64_t>& shape, const tb::Device& device) {
    return std::make_shared<GgufRowStage>(tables, rows, shape, device);
}

std::function<tb::Tensor()> prefetch_gguf_shards(
    const std::vector<std::shared_ptr<GgufRowTable>>& tables,std::int64_t rows,
    const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape,const tb::Device& device) {
    // Providers with a shared seek cursor must retain synchronous reads.
    if(std::any_of(tables.begin(),tables.end(),[](const auto& table){return !table || !table->parallel_reads();}))return {};
    return mfq::cuda::prefetch_table_rows([&]{return make_gguf_row_stage(tables,rows,shape,device);},ids,shape);
}

tb::Tensor lookup_gguf_shards(const std::vector<std::shared_ptr<GgufRowTable>>& tables,
    std::int64_t rows, const std::vector<std::int64_t>& ids, const tb::Device& device) {
    if (tables.empty() || rows <= 0) throw std::invalid_argument("invalid GGUF table shards");
    if (ids.empty()) return tb::empty({0, tables[0]->width()},
        tb::TensorOptions().device(device).dtype(device.is_cuda() ? tb::kFloat16 : tb::kFloat32));
    if (!device.is_cuda()) {
        if (tables.empty()) throw std::invalid_argument("empty GGUF table shards");
        const auto width = tables[0]->width();
        auto result = tb::empty({static_cast<std::int64_t>(ids.size()), width},
            tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] < 0 || std::uint64_t(ids[i] / rows) >= tables.size()) throw std::out_of_range("GGUF table id");
            const auto& table = tables[ids[i] / rows];
            table->decode_row(ids[i]%rows,result.data_ptr<float>()+i*width);
        }
        return result;
    }
    auto stage = make_gguf_row_stage(tables, rows, {static_cast<std::int64_t>(ids.size())}, device);
    stage->upload(ids, {static_cast<std::int64_t>(ids.size())}); stage->decode();
    return stage->output();
}

tb::Tensor GgufRowTable::lookup(tb::Tensor ids) const {
    const auto device = ids.device();
    auto shape = ids.sizes().vec(); shape.push_back(columns_);
    auto host = ids.to(tb::kInt64).cpu().contiguous().reshape({-1});
    auto table = std::const_pointer_cast<GgufRowTable>(shared_from_this());
    std::vector<std::int64_t> values(host.data_ptr<std::int64_t>(), host.data_ptr<std::int64_t>() + host.numel());
    return lookup_gguf_shards({table}, rows_, values, device).reshape(shape);
}
