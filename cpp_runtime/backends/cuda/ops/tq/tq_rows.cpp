#include "tq_detail.h"
#include "tyloinfer/encoded_row_cache.h"
#include "tyloinfer/mapped_prefetch_scope.h"
#include "runtime/row_stage_fence.h"
#include "runtime/row_packet_prefetch.h"
#include "mfq_weight_memory.h"
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstring>
#include <unordered_set>

namespace tb=mfq_tensor_backend;
using tq_detail::logical_shape;
using tq_detail::row_parts;
using tq_detail::join_rows;
using tq_detail::tq_image;
using tq_detail::status;

bool is_tq_row_table(const mfq::ModelSource& source,const std::string& name) {
    return mfq::tq_source::is_row_table(source,name);
}

struct TqRowTable::RowCache {
    struct Row {std::vector<std::uint8_t> image;mfq::TqRowImageLayout layout;};
    static constexpr std::size_t capacity=8u<<20;
    using Cache=mfq::EncodedRowCache<Row>;
    std::shared_ptr<Cache> rows=std::make_shared<Cache>(capacity,[](const Row& row){return row.image.size();},
        [](std::size_t bytes,bool retained) {
            if(retained)mfq::cuda::charge_tensor_host_bytes(bytes);
            else mfq::cuda::tensor_host_bytes.fetch_sub(bytes);
        });
    RowCache(){mfq::register_host_cache(rows);}
};

TqRowTable::TqRowTable(const mfq::ModelSource& source,const std::string& name)
        :row_cache_(std::make_shared<RowCache>()) {
    const auto shape=logical_shape(source,name);
    columns_=shape.back();
    const auto names=row_parts(source,name);
    std::unordered_set<std::string> unique;
    std::uint64_t end=0;
    for(const auto& part:names) {
        const auto* record=source.find_tensor(part);
        if(!unique.insert(part).second || !record || record->dtype!="TQ")
            throw std::invalid_argument("invalid TQ row segment: "+part);
        // Sparse table reads touch a descriptor, selectors and a short payload.
        // Mapped ranges avoid cursor changes and buffered read-ahead per row.
        // Routed banks retain their source's bulk buffered/direct I/O policy.
        auto reader=shape.size()==2?source.mapped_tensor_reader(part):mfq::ModelSource::TensorReader{};
        parallel_reads_=parallel_reads_ && (bool(reader) || source.supports_parallel_tensor_reads());
        if(reader)if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true))std::fprintf(stderr,
                "native_tq_sparse_rows actual_match=1 mapped=1 metadata_and_payload=1\n");
        }
        auto batch=reader?source.mapped_tensor_batch_reader(part):source.tensor_batch_reader(part);
        if(!reader)reader=source.tensor_reader(part);
        auto segments=mfq::tq_row_segments(record->nbytes,reader,columns_);
        const auto origin=readers_.size();readers_.push_back({std::move(reader),std::move(batch)});
        if(storage_bytes_>UINT64_MAX-record->nbytes)throw std::overflow_error("TQ row table size exceeds bounds");
        storage_bytes_+=record->nbytes;
        for(auto& rows:segments) {
            if(end>UINT64_MAX-rows->layout().rows)throw std::overflow_error("TQ row count exceeds bounds");
            end+=rows->layout().rows;ends_.push_back(end);segments_.push_back(std::move(rows));origins_.push_back(origin);
        }
    }
    std::uint64_t expected=1;
    for(std::size_t i=0;i+1<shape.size();++i)expected*=shape[i];
    if(end!=expected)throw std::invalid_argument("TQ table row count differs");
}
void TqRowTable::cache_row_metadata() {
    for(const auto& segment:segments_)segment->cache_metadata();
}
TrellisProjectionView TqRowTable::stage_row(std::uint64_t row,std::uint8_t* host,
        std::uint8_t* device,std::size_t capacity)const {
    if(row>=rows() || !host || !device)throw std::out_of_range("TQ staged row is out of bounds");
    const auto segment=std::upper_bound(ends_.begin(),ends_.end(),row)-ends_.begin();
    const auto start=segment?ends_[segment-1]:0;
    const auto& metadata=segments_[segment]->layout();
    mfq::TqRowImageLayout image;
    if(const auto cached=row_cache_->rows->find(row)) {
        if(cached->image.size()>capacity)throw std::overflow_error("TQ staged row image exceeds capacity");
        std::memcpy(host,cached->image.data(),cached->image.size());image=cached->layout;
    }else {
        const mfq::MappedPrefetchScope copied(false,true);
        image=segments_[segment]->stage_row_image(row-start,host,capacity);
        if(image.bytes<=RowCache::capacity-128)
            row_cache_->rows->insert(row,{std::vector<std::uint8_t>(host,host+image.bytes),image});
    }
    int gslog=0;for(auto gs=metadata.gs;gs>1;gs>>=1)++gslog;
    return {reinterpret_cast<const std::uint32_t*>(device+image.packed_offset),
        reinterpret_cast<const std::uint64_t*>(device+64),device+72,device+image.relative_offset,nullptr,
        1,int(metadata.width),int(columns_),gslog,int(metadata.columns)};
}
std::size_t TqRowTable::routed_image_storage_bound(std::uint64_t experts)const {
    if(!experts || experts>rows())throw std::invalid_argument("invalid TQ image count");
    std::uint64_t column_scales=0;
    for(const auto& segment:segments_)
        if(segment->layout().columns)column_scales=std::max(column_scales,std::uint64_t(segment->layout().scale_bytes));
    // Splitting a bank repeats its header and column scales. Each image also
    // carries a decoder table, selector rounding and aligned allocation tails.
    const auto extra=64+column_scales+1+3+32+15;
    if(storage_bytes_>SIZE_MAX || experts>(SIZE_MAX-storage_bytes_)/extra)
        throw std::overflow_error("TQ resident images exceed addressable storage");
    return static_cast<std::size_t>(storage_bytes_+experts*extra);
}
std::vector<std::uint8_t> TqRowTable::slice_rows(std::uint64_t begin,std::uint64_t end)const {
    if(begin>=end || end>rows())throw std::out_of_range("invalid TQ table row range");
    std::uint64_t start=0;
    std::vector<std::vector<std::uint8_t>> pieces;
    for(std::size_t i=0;i<segments_.size();++i) {
        const auto a=std::max(begin,start),b=std::min(end,ends_[i]);
        if(a<b)pieces.push_back(segments_[i]->slice_rows(a-start,b-start));
        start=ends_[i];if(start>=end)break;
    }
    if(pieces.size()==1)return std::move(pieces[0]);
    return join_rows(pieces);
}

std::size_t TqRowTable::expert_image_bytes(std::uint64_t expert,std::uint64_t count)const {
    if(!count || rows()%count || expert>=rows()/count)throw std::out_of_range("invalid TQ expert image range");
    const auto begin=expert*count,end=begin+count;
    auto segment=std::upper_bound(ends_.begin(),ends_.end(),begin)-ends_.begin();
    const auto& layout=segments_.at(segment)->layout();
    std::size_t packed=0;
    while(segment<segments_.size()) {
        const auto start=segment?ends_[segment-1]:0;
        const auto a=std::max(begin,start),b=std::min(end,ends_[segment]);
        if(a<b) {
            const auto& part=segments_[segment]->layout();
            if(part.width!=layout.width || part.columns!=layout.columns || part.gs!=layout.gs)
                throw std::invalid_argument("incompatible TQ expert row segments");
            const auto bytes=segments_[segment]->packed_row_bytes(a-start,b-start);
            if(bytes>SIZE_MAX-packed)throw std::overflow_error("TQ expert packed bytes");
            packed+=bytes;
        }
        if(ends_[segment]>=end)break;
        ++segment;
    }
    const auto scales=layout.columns?layout.scale_bytes:(count*layout.groups+1)/2;
    const auto metadata=(64+count*8+scales+3)&~std::uint64_t(3);
    if(metadata>SIZE_MAX-32 || packed>SIZE_MAX-32-metadata)throw std::overflow_error("TQ expert image bytes");
    return std::size_t(metadata)+packed+32;
}

std::vector<std::vector<std::uint8_t>> TqRowTable::slice_experts(
        const std::vector<std::int64_t>& experts,std::uint64_t rows_per_expert)const {
    if(!rows_per_expert || rows()%rows_per_expert)throw std::invalid_argument("invalid TQ expert row count");
    std::vector<std::vector<std::vector<std::uint8_t>>> pieces(experts.size());
    std::vector<std::vector<mfq::ReadSpan>> spans(readers_.size());
    for(std::size_t index=0;index<experts.size();++index) {
        const auto expert=experts[index];
        if(expert<0 || std::uint64_t(expert)>=rows()/rows_per_expert)throw std::out_of_range("TQ expert ID");
        const auto begin=std::uint64_t(expert)*rows_per_expert,end=begin+rows_per_expert;
        auto segment=std::upper_bound(ends_.begin(),ends_.end(),begin)-ends_.begin();
        while(segment<segments_.size()) {
            const auto start=segment?ends_[segment-1]:0;
            const auto a=std::max(begin,start),b=std::min(end,ends_[segment]);
            if(a<b)pieces[index].push_back(segments_[segment]->slice_rows(a-start,b-start,&spans[origins_[segment]]));
            if(ends_[segment]>=end)break;
            ++segment;
        }
    }
    for(std::size_t origin=0;origin<readers_.size();++origin)if(!spans[origin].empty()) {
        const auto& reader=readers_[origin];
        if(reader.batch)reader.batch(spans[origin]);
        else for(const auto& span:spans[origin])reader.read(span.offset,span.destination,span.size);
    }
    std::vector<std::vector<std::uint8_t>> images;images.reserve(experts.size());
    for(auto& parts:pieces)images.push_back(parts.size()==1?std::move(parts.front()):join_rows(parts));
    return images;
}

std::size_t TqRowTable::row_image_bytes_bound(std::uint64_t count)const {
    if(!count || count>INT_MAX)throw std::overflow_error("TQ image row count exceeds CUDA limits");
    std::size_t bytes=0;
    for(const auto& segment:segments_) {
        const auto& m=segment->layout();
        if(m.width>INT_MAX)throw std::overflow_error("TQ row stage width exceeds CUDA limits");
        const auto scales=m.columns?m.scale_bytes:(count*m.groups+1)/2;
        // Stored padding and scale layout come from the encoded segments.
        const auto bound=((64+count*8+scales+3)&~std::uint64_t(3))+count*m.width+32;
        if(bound>SIZE_MAX-15)throw std::overflow_error("TQ image exceeds addressable storage");
        bytes=std::max(bytes,static_cast<std::size_t>(bound));
    }
    return (bytes+7)&~std::size_t(7);
}
tb::Tensor TqRowTable::lookup(const std::int64_t* ids,std::size_t count,const tb::Device& device)const {
    if(!device.is_cuda() || !count || !ids)throw std::invalid_argument("TQ table lookup requires CUDA and row IDs");
    std::vector<std::int64_t> selected(ids,ids+count);
    std::sort(selected.begin(),selected.end());selected.erase(std::unique(selected.begin(),selected.end()),selected.end());
    if(selected.front()<0 || std::uint64_t(selected.back())>=rows())throw std::out_of_range("TQ table lookup row is out of bounds");
    // Shared table IDs occur across tokens and heads. Read each compressed
    // row once, coalesce adjacent rows, then restore the caller's order on GPU.
    // Sorting also groups metadata reads without fetching intervening weights.
    const MfqCudaGuard guard(device);
    const auto gather=[&](std::size_t segment,std::size_t begin,std::size_t end) {
        const auto start=segment?ends_[segment-1]:0;
        std::vector<std::vector<std::uint8_t>> pieces;
        for(auto run=begin;run<end;) {
            auto next=run+1;
            while(next<end && selected[next]==selected[next-1]+1)++next;
            pieces.push_back(segments_[segment]->slice_rows(selected[run]-start,selected[next-1]+1-start));
            run=next;
        }
        return to_device_tq(join_rows(pieces),columns_,device.index);
    };
    const auto first=std::upper_bound(ends_.begin(),ends_.end(),std::uint64_t(selected.front()))-ends_.begin();
    const auto last=std::upper_bound(ends_.begin(),ends_.end(),std::uint64_t(selected.back()))-ends_.begin();
    if(first==last) {
        // Preserve the single-segment path: decode duplicates directly in
        // caller order, without an intermediate table or gather kernel.
        const auto weight=gather(first,0,selected.size());
        if(selected.size()!=count || !std::equal(selected.begin(),selected.end(),ids)) {
            std::vector<std::int64_t> inverse;inverse.reserve(count);
            for(std::size_t i=0;i<count;++i)
                inverse.push_back(std::lower_bound(selected.begin(),selected.end(),ids[i])-selected.begin());
            const auto index=tb::from_blob(inverse.data(),{static_cast<std::int64_t>(count)},tb::TensorOptions().dtype(tb::kInt64))
                .clone().to(device,false,false);
            return weight.embedding(index);
        }
        if(count>65535)return weight.embedding(tb::arange(count,weight.image.options().dtype(tb::kInt64)));
        auto result=tb::empty({static_cast<std::int64_t>(count),columns_},weight.image.options().dtype(tb::kFloat16));
        status(mfq_tq_embedding(weight.view,nullptr,static_cast<int>(count),result.data_ptr(),0,mfq_current_cuda_stream()));
        return result;
    }
    auto result=tb::empty({static_cast<std::int64_t>(selected.size()),columns_},
        tb::TensorOptions().device(device).dtype(tb::kFloat16));
    std::size_t begin=0;
    // Decode each physical segment separately: its padding, group size and
    // column scales need not match the other segments of this logical table.
    for(std::size_t segment=0;segment<segments_.size() && begin<selected.size();++segment) {
        const auto end=std::lower_bound(selected.begin()+begin,selected.end(),ends_[segment],
            [](std::int64_t row,std::uint64_t end){return std::uint64_t(row)<end;})-selected.begin();
        if(std::size_t(end)>begin) {
            const auto weight=gather(segment,begin,end);
            const auto count=end-begin;
            if(count>65535) {
                const auto index=tb::arange(count,weight.image.options().dtype(tb::kInt64));
                result.slice(0,begin,end).copy_(weight.embedding(index));
            } else status(mfq_tq_embedding(weight.view,nullptr,static_cast<int>(count),
                static_cast<std::uint8_t*>(result.data_ptr())+begin*columns_*sizeof(std::uint16_t),
                0,mfq_current_cuda_stream()));
        }
        begin=end;
    }
    if(selected.size()!=count || !std::equal(selected.begin(),selected.end(),ids)) {
        std::vector<std::int64_t> inverse;inverse.reserve(count);
        for(std::size_t i=0;i<count;++i)inverse.push_back(std::lower_bound(selected.begin(),selected.end(),ids[i])-selected.begin());
        const auto index=tb::from_blob(inverse.data(),{static_cast<std::int64_t>(count)},tb::TensorOptions().dtype(tb::kInt64))
            .clone().to(device,false,false);
        return result.index_select(0,index);
    }
    return result;
}
tb::Tensor TqRowTable::lookup(tb::Tensor ids)const {
    auto host=ids.contiguous().cpu().to(tb::kInt64);
    auto shape=ids.sizes().vec();shape.push_back(columns_);
    if(!ids.numel())return tb::empty(shape,ids.options().dtype(tb::kFloat16));
    return lookup(host.data_ptr<std::int64_t>(),host.numel(),ids.device()).reshape(shape);
}

tb::Tensor lookup_tq_shards(const std::vector<std::shared_ptr<TqRowTable>>& shards,
        std::int64_t rows_per_shard,const std::vector<std::int64_t>& ids,const tb::Device& device) {
    if(shards.empty() || !shards.front() || rows_per_shard<1 || !device.is_cuda())
        throw std::invalid_argument("invalid TQ table shard geometry");
    const auto columns=shards.front()->width();
    for(const auto& table:shards)if(!table || table->width()!=columns)
        throw std::invalid_argument("TQ table shard widths differ");
    static std::atomic<bool> reported{false};
    if(std::getenv("TYLOINFER_TQ_PROFILE_CACHE") && !reported.exchange(true))
        std::fprintf(stderr,"tq_table_shards_dispatch enabled=1 shards=%zu lookups=%zu width=%lld CPU_ids=1\n",
            shards.size(),ids.size(),static_cast<long long>(columns));
    const MfqCudaGuard guard(device);
    auto result=tb::empty({static_cast<std::int64_t>(ids.size()),columns},
        tb::TensorOptions().device(device).dtype(tb::kFloat16));
    std::vector<std::vector<std::int64_t>> local(shards.size()),positions(shards.size());
    for(std::size_t i=0;i<ids.size();++i) {
        if(ids[i]<0 || std::uint64_t(ids[i]/rows_per_shard)>=shards.size())
            throw std::out_of_range("TQ embedding shard ID is out of bounds");
        const auto shard=ids[i]/rows_per_shard;
        local[shard].push_back(ids[i]%rows_per_shard);positions[shard].push_back(static_cast<std::int64_t>(i));
    }
    // Hashing already produced CPU IDs. Keep shard selection on CPU, read
    // only that shard's requested rows, then scatter once on GPU. There is no
    // masked whole-batch lookup or GPU->CPU round trip for each shard.
    for(std::size_t shard=0;shard<shards.size();++shard)if(!local[shard].empty()) {
        const auto values=shards[shard]->lookup(local[shard].data(),local[shard].size(),device);
        const auto index=tb::from_blob(positions[shard].data(),{static_cast<std::int64_t>(positions[shard].size())},
            tb::TensorOptions().dtype(tb::kInt64)).clone().to(device,false,false);
        result.index_copy_(0,index,values);
    }
    return result;
}

namespace {
class TqRowStage final:public RowStage {
    mfq::cuda::RowStageFence fence_;
    std::vector<std::shared_ptr<TqRowTable>> tables_;
    std::vector<std::int64_t> shape_;
    std::int64_t rows_per_shard_,count_=1,columns_=0;
    std::size_t image_stride_=0,images_offset_=0,bytes_=0;
    tb::Device device_;
    std::unique_ptr<mfq::cuda::HostBuffer> host_;
    tb::Tensor packet_,output_;
    std::uint8_t* packet_address_=nullptr;
    bool mapped_=false,charged_=false;
    std::vector<MfqCudaStream> streams_;
    mfq::cuda::RowPacketPrefetch prefetch_;
    void retain_stream() {
        const auto stream=mfq_get_current_cuda_stream(device_.index);
        if(std::none_of(streams_.begin(),streams_.end(),[&](const auto& s){return s.stream()==stream.stream();}))
            streams_.push_back(stream);
    }
    TrellisProjectionView pack_row(const std::vector<std::uint8_t>& wire,std::size_t slot) {
        const mfq::TqLayout layout(wire.data(),wire.size());
        const auto image=tq_image(wire,layout);
        if(layout.rows!=1 || columns_>layout.width || image.size()>image_stride_)
            throw std::invalid_argument("TQ stage row exceeds its prepared layout");
        const auto offset=images_offset_+slot*image_stride_;
        std::memcpy(static_cast<std::uint8_t*>(host_->data())+offset,image.data(),image.size());
        auto* address=packet_address_+offset;
        const auto words=(layout.packed+3)&~std::size_t(3);
        int gslog=0;for(auto gs=layout.gs;gs>1;gs>>=1)++gslog;
        return {reinterpret_cast<const std::uint32_t*>(address+words),
            reinterpret_cast<const std::uint64_t*>(address+64),address+layout.scales,
            address+words+layout.packed_bytes,nullptr,1,int(layout.width),int(columns_),gslog,int(layout.columns)};
    }
public:
    TqRowStage(std::vector<std::shared_ptr<TqRowTable>> tables,std::int64_t rows_per_shard,
            std::vector<std::int64_t> shape,const tb::Device& device)
        :tables_(std::move(tables)),shape_(std::move(shape)),rows_per_shard_(rows_per_shard),device_(device) {
        if(tables_.empty() || !tables_[0] || rows_per_shard_<1 || !device.is_cuda())
            throw std::invalid_argument("invalid TQ row stage geometry");
        for(const auto size:shape_) {
            if(size<1 || count_>INT_MAX/size)throw std::overflow_error("TQ row stage shape exceeds bounds");
            count_*=size;
        }
        columns_=tables_[0]->width();
        for(const auto& table:tables_) {
            if(!table || table->width()!=columns_)throw std::invalid_argument("TQ row stage table widths differ");
            image_stride_=std::max(image_stride_,table->row_image_bytes_bound());
        }
        if(columns_>INT_MAX || std::uint64_t(count_)>(SIZE_MAX-7)/sizeof(TrellisProjectionView))
            throw std::overflow_error("TQ row stage metadata exceeds bounds");
        images_offset_=(std::size_t(count_)*sizeof(TrellisProjectionView)+7)&~std::size_t(7);
        if(std::uint64_t(count_)>(SIZE_MAX-images_offset_)/image_stride_)
            throw std::overflow_error("TQ row stage packet exceeds bounds");
        bytes_=images_offset_+std::size_t(count_)*image_stride_;
        if(bytes_>INT64_MAX)throw std::overflow_error("TQ row stage packet exceeds tensor limits");
        const MfqCudaGuard guard(device);
        retain_stream();
        int can_map=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&can_map,cudaDevAttrCanMapHostMemory,device.index));
        const auto* setting=std::getenv("TYLOINFER_PLE_MAPPED_ROWS");
        mapped_=can_map && (setting?setting[0]=='1':bytes_<=16u*1024);
        host_=std::make_unique<mfq::cuda::HostBuffer>(bytes_,mapped_);
        if(mapped_) {
            void* address=nullptr;
            const auto error=cudaHostGetDevicePointer(&address,host_->data(),0);
            if(error==cudaErrorNotSupported || error==cudaErrorInvalidValue) {
                (void)cudaGetLastError();mapped_=false;
            }else {TYLOINFER_CUDA_CHECK(error);packet_address_=static_cast<std::uint8_t*>(address);}
        }
        const auto options=tb::TensorOptions().device(device).dtype(tb::kFloat16);
        if(!mapped_) {
            packet_=tb::empty({static_cast<std::int64_t>(bytes_)},options.dtype(tb::kUInt8));
            packet_address_=packet_.data_ptr<std::uint8_t>();
        }
        auto output_shape=shape_;output_shape.push_back(columns_);
        output_=tb::zeros(output_shape,options);
        std::memset(host_->data(),0,bytes_);
        // Valid zero-anchor rows allow graph preparation without source reads
        // or model-state changes. Actual upload publishes the requested rows.
        const auto zero_width=(std::uint64_t(columns_)+255)&~std::uint64_t(255);
        if(zero_width>INT_MAX)throw std::overflow_error("TQ row stage padding exceeds CUDA limits");
        std::vector<std::uint8_t> zero(72+zero_width/32+zero_width/8);
        std::memcpy(zero.data(),"RTRELL05",8);
        mfq::TqLayout::integer(zero.data()+8,5,4);
        mfq::TqLayout::integer(zero.data()+12,1,4);
        mfq::TqLayout::integer(zero.data()+16,zero_width,4);
        mfq::TqLayout::integer(zero.data()+24,16,4);
        mfq::TqLayout::integer(zero.data()+28,zero_width/8,4);
        const auto view=pack_row(zero,0);
        auto* views=static_cast<TrellisProjectionView*>(host_->data());
        std::fill_n(views,count_,view);
        if(!mapped_)TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(packet_address_,host_->data(),bytes_,cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        mfq::cuda::charge_tensor_host_bytes(bytes_);charged_=true;
    }
    ~TqRowStage()override {
        // The pinned packet is not tensor storage. Fence its I/O and target
        // consumers explicitly before freeing mapped memory or DMA sources.
        int previous=0;const bool restore=cudaGetDevice(&previous)==cudaSuccess;
        (void)cudaSetDevice(device_.index);
        for(const auto& stream:streams_)(void)cudaStreamSynchronize(stream.stream());
        output_={};packet_={};
        if(charged_)mfq::cuda::tensor_host_bytes.fetch_sub(bytes_);
        if(restore)(void)cudaSetDevice(previous);
    }
    tb::Tensor output()const override{return output_;}
    bool mapped_rows()const override{return mapped_;}
    bool prefetch(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape)override {
        if(std::any_of(tables_.begin(),tables_.end(),[](const auto& table){return !table->parallel_reads();}))return false;
        prefetch_.start(ids,shape,bytes_,[this,ids,shape](void* packet) {
            return prepare_rows_to(ids,shape,packet);
        });return true;
    }
    void upload(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape)override {
        const MfqCudaGuard guard(device_);fence_.before_write();
        prepare_upload(ids,shape);
    }
    void upload_graph(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape)override {
        const MfqCudaGuard guard(device_);fence_.owner_fenced();
        prepare_upload(ids,shape);
    }
private:
    void prepare_upload(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape) {
        retain_stream();
        const auto cached=prefetch_.consume(ids,shape,host_->data());
        const auto used=cached?*cached:prepare_rows_to(ids,shape,host_->data());
        if(!mapped_) {
            mfq_cuda_record_stream(packet_,mfq_get_current_cuda_stream(device_.index));
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(packet_address_,host_->data(),used,cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            fence_.record_transfer(mfq_current_cuda_stream());
        }
    }
    std::size_t prepare_rows_to(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape,void* packet) {
        if(shape!=shape_ || ids.size()!=std::size_t(count_))throw std::invalid_argument("TQ stage upload shape changed");
        const auto selection=mfq::select_tq_rows(ids,rows_per_shard_,tables_.size());
        std::vector<TrellisProjectionView> prepared(selection.selected.size());
        const auto prepare=[&](std::int64_t begin,std::int64_t end) {
            for(auto index=begin;index<end;++index) {
                const auto id=selection.selected[index],shard=id/rows_per_shard_,row=id%rows_per_shard_;
                const auto offset=images_offset_+std::size_t(index)*image_stride_;
                prepared[index]=tables_[shard]->stage_row(row,
                    static_cast<std::uint8_t*>(packet)+offset,packet_address_+offset,image_stride_);
            }
        };
        const bool parallel=std::all_of(selection.selected.begin(),selection.selected.end(),[&](auto id) {
            return tables_[id/rows_per_shard_]->parallel_reads();
        });
        if(parallel)mfq_parallel_for(0,static_cast<std::int64_t>(prepared.size()),1,prepare);
        else prepare(0,static_cast<std::int64_t>(prepared.size()));
        if(parallel && prepared.size()>1)if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true))std::fprintf(stderr,
                "native_tq_row_stage actual_match=1 parallel=1 selected=%zu worker_limit=%d\n",prepared.size(),mfq_get_num_threads());
        }
        auto* views=static_cast<TrellisProjectionView*>(packet);
        for(std::size_t index=0;index<ids.size();++index)views[index]=prepared[selection.inverse[index]];
        return images_offset_+prepared.size()*image_stride_;
    }
public:
    void decode()override {
        const MfqCudaGuard guard(device_);retain_stream();
        const auto capture=fence_.join(mfq_current_cuda_stream());
        mfq_cuda_record_stream(output_,mfq_get_current_cuda_stream(device_.index));
        if(packet_.defined())mfq_cuda_record_stream(packet_,mfq_get_current_cuda_stream(device_.index));
        status(mfq_tq_embedding_rows(reinterpret_cast<const TrellisProjectionView*>(packet_address_),
            int(count_),int(columns_),output_.data_ptr(),0,mfq_current_cuda_stream()));
        fence_.record_consumer(mfq_current_cuda_stream(),capture);
    }
};
}
std::shared_ptr<RowStage> make_tq_row_stage(const std::vector<std::shared_ptr<TqRowTable>>& tables,
        std::int64_t rows_per_shard,const std::vector<std::int64_t>& shape,const tb::Device& device) {
    return std::make_shared<TqRowStage>(tables,rows_per_shard,shape,device);
}
std::function<tb::Tensor()> prefetch_tq_shards(const std::vector<std::shared_ptr<TqRowTable>>& tables,
        std::int64_t rows,const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape,const tb::Device& device) {
    if(std::any_of(tables.begin(),tables.end(),[](const auto& table){return !table || !table->parallel_reads();}))return {};
    return mfq::cuda::prefetch_table_rows([&]{return make_tq_row_stage(tables,rows,shape,device);},ids,shape);
}
