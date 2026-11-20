#include "tq.h"
#include "tyloinfer/tq_rows.h"
#include "gguf_weight.h"
#include "gguf_prefill_fusion.h"
#include "kernels/gguf.h"
#include "runtime/routed_cache_resources.h"
#include "runtime/decode_window.h"
#include "runtime/moe_pipeline.h"
#include "runtime/routed_transfer.h"
#include "runtime/independent_branch.h"
#include "runtime/expert_route_usage.h"
#include "runtime/kv_memory.h"
#include "core/ffn.h"
#include "core/vision_ops.h"
#include "storage/mapped_embedding.h"
#include "storage/weight_loader.h"
#include "storage/routed_prefill_stager.h"
#include "runtime/routed_prefill_pipeline.h"
#include "runtime/routed_graph_pools.h"
#include "runtime/routed_event_retirement.h"
#include "runtime/routed_dma_submission.h"
#include "runtime/routed_stream_fence.h"
#include "runtime/cuda_event_status.h"
#include "runtime/routed_host_cache.h"
#include "runtime/routed_image_layout.h"
#include "storage/routed_host_image.h"
#include "qwen4_exp.h"
#include "models/qwen4_exp/ops.h"
#include "float_projection.h"
#include "common_prefill_projection_test.h"
#include "mfq_cuda_activation_ops.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>
#include <chrono>
#include <numeric>
#include <condition_variable>
#include <mutex>

namespace tb=mfq_tensor_backend;
namespace {
class Source final:public mfq::ModelSource {
    std::vector<mfq::TensorMetadata> tensors_;
    std::unordered_map<std::string,std::shared_ptr<std::vector<std::byte>>> data_;
    std::vector<std::filesystem::path> paths_;
    std::unordered_map<std::string,std::string> metadata_;
    std::vector<std::string> assets_;
    std::shared_ptr<std::uint64_t> fail_from_=std::make_shared<std::uint64_t>(UINT64_MAX);
    std::shared_ptr<std::string> fail_tensor_=std::make_shared<std::string>();
public:
    mutable std::size_t mapped_requests=0;
    bool batch_rows=false;
    std::shared_ptr<std::size_t> batch_requests=std::make_shared<std::size_t>(0);
    void fail_reads_from(std::uint64_t offset){*fail_from_=offset;}
    void fail_tensor_reads(const std::string& name){*fail_tensor_=name;*fail_from_=0;}
    explicit Source(std::uint32_t stored_width=256,bool mixed=false,
            std::array<unsigned,5> precisions={2,7,3,6,4}) {
        for(const auto* name:{"gate","up","down"}) {
            const bool down=std::strcmp(name,"down")==0;
            const std::uint32_t rows=5*(down?33:19),width=stored_width;
            const auto bit_sum=mixed?std::accumulate(precisions.begin(),precisions.end(),0u):15u;
            const std::uint32_t packed=rows*width*bit_sum/40;
            const auto selectors=rows*(width/256);
            auto bytes=std::make_shared<std::vector<std::byte>>(64+rows*8+(selectors+1)/2+packed);
            const auto put=[&](std::size_t offset,std::uint64_t value,int count) {
                for(int i=0;i<count;++i)(*bytes)[offset+i]=std::byte(value>>(i*8));
            };
            std::memcpy(bytes->data(),"RTRELL05",8);
            for(const auto [offset,value]:{std::pair{8u,5u},{12u,rows},{16u,width},{20u,0u},{24u,256u},{28u,packed}})
                put(offset,value,4);
            std::uint64_t offset=0;
            for(std::uint32_t i=0;i<rows;++i) {
                const auto bits=mixed?precisions[i/(rows/5)]:3u;
                put(64+i*8,offset|(std::uint64_t(bits-1)<<32)|(0x2800ULL<<40),8);
                offset+=width*bits/32;
            }
            std::uint32_t state=37;
            for(std::size_t i=64+rows*8;i<bytes->size();++i) {
                state=state*1664525u+1013904223u;(*bytes)[i]=std::byte(state>>24);
            }
            if(selectors%2)(*bytes)[64+rows*8+selectors/2]&=std::byte{15};
            mfq::TensorMetadata t;t.name=name;t.dtype=t.stored_dtype="TQ";
            t.shape=std::vector<std::int64_t>{5,down?33:19,down?19:33};t.nbytes=bytes->size();
            tensors_.push_back(t);data_.emplace(name,std::move(bytes));
        }
        {
            auto tensor=tensors_.front();tensor.name="linear";
            tensor.shape=std::vector<std::int64_t>{95,33};
            tensors_.push_back(tensor);data_.emplace(tensor.name,data_.at("gate"));
            tensor=tensors_[2];tensor.name="linear_down";tensor.shape=std::vector<std::int64_t>{165,95};
            tensors_.push_back(tensor);data_.emplace(tensor.name,data_.at("down"));
        }
        for(const auto* name:{"gate","up","down"}) {
            auto tensor=tensors_[std::strcmp(name,"gate")==0?0:std::strcmp(name,"up")==0?1:2];
            tensor.name=std::string("predictor.")+name;
            tensors_.push_back(tensor);data_.emplace(tensor.name,data_.at(name));
        }
        {
            auto bytes=std::make_shared<std::vector<std::byte>>(20+7*33*sizeof(float));
            const std::uint32_t rank=2;const int64_t dimensions[]{7,33};
            std::memcpy(bytes->data(),&rank,4);std::memcpy(bytes->data()+4,dimensions,16);
            for(int i=0;i<7*33;++i) {
                const float value=(i%23-11)*.0025f;std::memcpy(bytes->data()+20+i*sizeof(float),&value,sizeof(float));
            }
            mfq::TensorMetadata tensor;tensor.name="auxiliary";tensor.dtype=tensor.stored_dtype="F32";
            tensor.shape=std::vector<std::int64_t>{7,33};tensor.nbytes=bytes->size();
            tensors_.push_back(tensor);data_.emplace(tensor.name,std::move(bytes));
        }
        {
            auto bytes=std::make_shared<std::vector<std::byte>>(5*19*18,std::byte{0x97});
            const auto scale=ggml_fp32_to_fp16(.03125f);
            for(std::size_t at=0;at<bytes->size();at+=18)std::memcpy(bytes->data()+at,&scale,2);
            mfq::TensorMetadata tensor;tensor.name="gguf.weight";tensor.dtype=tensor.stored_dtype="GGUF_Q4_0";
            tensor.shape=std::vector<std::int64_t>{5,19,32};tensor.nbytes=bytes->size();
            tensors_.push_back(tensor);data_.emplace(tensor.name,std::move(bytes));
            tensor.name="gguf.linear";tensor.shape=std::vector<std::int64_t>{95,32};
            tensors_.push_back(tensor);data_.emplace(tensor.name,data_.at("gguf.weight"));
        }
        for(const auto* name:{"gate","up","down"}) {
            auto bytes=std::make_shared<std::vector<std::byte>>(5*32*18,std::byte{0x97});
            const auto scale=ggml_fp32_to_fp16(.03125f);
            for(std::size_t at=0;at<bytes->size();at+=18)std::memcpy(bytes->data()+at,&scale,2);
            mfq::TensorMetadata tensor;tensor.name=std::string("gguf.")+name;
            tensor.dtype=tensor.stored_dtype="GGUF_Q4_0";
            tensor.shape=std::vector<std::int64_t>{5,32,32};tensor.nbytes=bytes->size();
            tensors_.push_back(tensor);data_.emplace(tensor.name,std::move(bytes));
        }
        for(int bits:{4,8}) {
            constexpr int rows=33,columns=128;
            const int stored_columns=columns*bits/8,scale_rows=bits==4?rows:(rows+31)/32;
            std::vector<std::uint8_t> payload{'M','X','T','1',1,std::uint8_t(bits),0,0};
            for(std::uint64_t dimension:{std::uint64_t(rows),std::uint64_t(columns),std::uint64_t(rows),
                    std::uint64_t(stored_columns),std::uint64_t(scale_rows),std::uint64_t(columns/32)})
                for(int byte=0;byte<8;++byte)payload.push_back(std::uint8_t(dimension>>(byte*8)));
            for(int i=0;i<rows*stored_columns;++i)
                payload.push_back(bits==4?std::uint8_t(i*31+7):std::uint8_t((i*37)%120|((i&1)<<7)));
            payload.insert(payload.end(),scale_rows*(columns/32),std::uint8_t(123));
            auto bytes=std::make_shared<std::vector<std::byte>>(payload.size());
            std::memcpy(bytes->data(),payload.data(),payload.size());
            mfq::TensorMetadata tensor;tensor.name="mx"+std::to_string(bits)+".linear";
            tensor.dtype=tensor.stored_dtype="MXFP"+std::to_string(bits);
            tensor.shape=std::vector<std::int64_t>{rows,columns};tensor.nbytes=bytes->size();
            tensors_.push_back(tensor);data_.emplace(tensor.name,std::move(bytes));
        }
        for(const auto* dtype:{"F16","BF16","F32"}) {
            const std::string name=std::string("rows.")+dtype;
            const auto element=std::strcmp(dtype,"F32")==0?4u:2u;
            auto bytes=std::make_shared<std::vector<std::byte>>(7*13*element);
            for(int row=0;row<7;++row)for(int col=0;col<13;++col) {
                const float value=row*.25f+col*.03125f;const auto i=row*13+col;
                if(element==4)std::memcpy(bytes->data()+i*4,&value,4);
                else {
                    std::uint32_t bits;std::memcpy(&bits,&value,4);
                    const auto encoded=std::strcmp(dtype,"BF16")==0?std::uint16_t(bits>>16):ggml_fp32_to_fp16(value);
                    std::memcpy(bytes->data()+i*2,&encoded,2);
                }
            }
            mfq::TensorMetadata t;t.name=name;t.dtype=t.stored_dtype=dtype;
            t.shape=std::vector<int64_t>{7,13};t.nbytes=bytes->size();
            tensors_.push_back(t);data_.emplace(name,bytes);
            auto encoded=std::make_shared<std::vector<std::byte>>(bytes->size()+20);
            const std::uint32_t rank=2;const int64_t dims[]{7,13};
            std::memcpy(encoded->data(),&rank,4);std::memcpy(encoded->data()+4,dims,16);
            std::memcpy(encoded->data()+20,bytes->data(),bytes->size());
            t.name=std::string("payload.")+name;t.shape.reset();t.nbytes=encoded->size();
            tensors_.push_back(t);data_.emplace(t.name,std::move(encoded));
        }
    }
    const std::vector<std::filesystem::path>& source_paths()const noexcept override{return paths_;}
    std::string_view architecture()const noexcept override{return "tq-test";}
    const std::unordered_map<std::string,std::string>& metadata()const noexcept override{return metadata_;}
    const std::vector<mfq::TensorMetadata>& tensors()const noexcept override{return tensors_;}
    const mfq::TensorMetadata* find_tensor(std::string_view name)const noexcept override {
        for(const auto& t:tensors_)if(t.name==name)return &t;return nullptr;
    }
    TensorReader tensor_reader(std::string_view name)const override {
        const auto found=data_.find(std::string(name));
        if(found==data_.end())throw std::out_of_range("missing TQ fixture tensor: "+std::string(name));
        auto bytes=found->second;
        return [bytes,fail=fail_from_,filter=fail_tensor_,selected=std::string(name)](std::uint64_t offset,std::byte* out,std::size_t count) {
            if(offset>=*fail && (filter->empty() || *filter==selected))throw std::runtime_error("injected TQ source read failure");
            if(offset>bytes->size() || count>bytes->size()-offset)throw std::out_of_range("TQ test read");
            std::memcpy(out,bytes->data()+offset,count);
        };
    }
    TensorReader mapped_tensor_reader(std::string_view name)const override {
        if(name!="linear" && !batch_rows)return {};
        ++mapped_requests;return tensor_reader(name);
    }
    TensorBatchReader tensor_batch_reader(std::string_view name)const override {
        if(!batch_rows)return {};
        return [read=tensor_reader(name),count=batch_requests](const auto& spans) {
            ++*count;
            for(const auto& span:spans)read(span.offset,span.destination,span.size);
        };
    }
    void read_range_into(std::string_view name,std::uint64_t offset,std::byte* out,std::size_t count)const override {
        tensor_reader(name)(offset,out,count);
    }
    const std::vector<std::string>& assets()const noexcept override{return assets_;}
    bool has_asset(std::string_view)const noexcept override{return false;}
    std::vector<std::byte> read_asset(std::string_view)const override{throw std::invalid_argument("no assets");}
    std::optional<mfq::ModelGraph> model_graph()const override{return {};}
};
double metric(CudaExecutionContext& execution,const char* key) {
    for(const auto& [name,value]:mfq::cuda::routed_expert_memory_metrics(execution))if(name==key)return value;
    throw std::runtime_error("missing TQ resource metric");
}
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}

void check_host_image_range_trim() {
    using Image=mfq::cuda::RoutedHostImage<>;
    struct Cache:mfq::cuda::RoutedHostCache<Image> {
        void add(std::uint64_t key,const std::shared_ptr<MoePinnedExpertArena::Block>& block,std::size_t offset) {
            auto image=std::make_shared<Image>();image->bytes=80;image->block=block;image->block_offset=offset;
            std::memset(image->data(),int(key),image->size());host_index_insert(key,image,false);
        }
        const std::shared_ptr<Image>& image(std::uint64_t key)const{return host_.at(key).image;}
        void erase(std::uint64_t key){host_index_erase(key);}
        std::size_t trim(const void* allocation) {
            return trim_host_image_ranges(allocation,std::array<std::uint64_t,3>{1,2,3});
        }
    } cache;
    auto arena=std::make_shared<MoePinnedExpertArena>(256);
    auto paired=arena->allocate(192);cache.add(1,paired,16);cache.add(2,paired,96);paired.reset();
    require(cache.trim(arena->allocation_id())==0 && arena->used_bytes()==192,
        "host range trim changed a lease shared by live expert images");
    cache.erase(2);
    auto consumer=cache.image(1);
    require(cache.trim(arena->allocation_id())==0 && arena->used_bytes()==192,
        "host range trim changed an image held by an asynchronous consumer");
    consumer.reset();
    auto lease=cache.image(1)->block;
    require(cache.trim(arena->allocation_id())==0,"host range trim changed an externally leased block");
    lease.reset();
    const auto pointer=cache.image(1)->data(),mapped=cache.image(1)->mapped_data(0);
    require(cache.trim(nullptr)==0 && cache.trim(arena->allocation_id())==112 && arena->used_bytes()==80 &&
        cache.image(1)->block_offset==0 && cache.image(1)->data()==pointer && cache.image(1)->mapped_data(0)==mapped,
        "host range trim did not preserve survivor addresses while releasing unused sibling bytes");
    for(std::size_t i=0;i<80;++i)require(pointer[i]==1,"host range trim changed encoded expert bytes");
    require(cache.trim(arena->allocation_id())==0 && arena->largest_free_bytes()==160,
        "host range trim double-counted or failed to expose reusable intervals");
    std::cout<<"Common pinned range trim: consumer leases, shared siblings and stable addresses passed\n";
}

void check_host_image_relocation() {
    using Image=mfq::cuda::RoutedHostImage<>;
    struct Cache:mfq::cuda::RoutedHostCache<Image> {
        void add(std::uint64_t key,const std::shared_ptr<MoePinnedExpertArena::Block>& block,std::size_t offset,
                std::size_t bytes=80) {
            auto image=std::make_shared<Image>();image->bytes=bytes;image->block=block;image->block_offset=offset;
            std::memset(image->data(),int(key),image->size());host_index_insert(key,image,false);
        }
        const std::shared_ptr<Image>& image(std::uint64_t key)const{return host_.at(key).image;}
        std::size_t relocate(const std::vector<std::shared_ptr<MoePinnedExpertArena>>& arenas,const void* allocation,
                std::unordered_set<const void*>& retained) {
            return relocate_host_images(arenas,allocation,std::array<std::uint64_t,2>{1,2},retained,
                [](std::uint64_t,const void*,const void*){});
        }
        bool compact(const std::vector<std::shared_ptr<MoePinnedExpertArena>>& arenas,std::size_t bytes,
                std::size_t& copied) {
            std::unordered_map<const void*,std::vector<std::uint64_t>> index;
            for(const auto& [key,entry]:host_)index[entry.image->block->allocation_id()].push_back(key);
            return compact_host_interval(arenas,index,bytes,copied);
        }
    } cache;
    auto source=std::make_shared<MoePinnedExpertArena>(256),target=std::make_shared<MoePinnedExpertArena>(256);
    const std::vector<std::shared_ptr<MoePinnedExpertArena>> arenas{source,target};
    auto paired=source->allocate(160);cache.add(1,paired,0);cache.add(2,paired,80);paired.reset();
    auto hole=target->allocate(96),guard=target->allocate(32),tail=target->allocate(64),anchor=target->allocate(64);
    std::memset(guard->data(),0xa7,guard->bytes());const auto fixed=guard->data();hole.reset();tail.reset();
    const auto first=cache.image(1)->data(),second=cache.image(2)->data();
    std::unordered_set<const void*> retained;
    require(cache.relocate(arenas,source->allocation_id(),retained)==0 && source->used_bytes()==160 &&
        target->used_bytes()==96 && cache.image(1)->data()==first && cache.image(2)->data()==second && retained.empty(),
        "failed host relocation changed pointers or retained temporary pinned leases");
    anchor.reset();
    auto active=cache.image(2);
    require(cache.relocate(arenas,source->allocation_id(),retained)==0 && target->used_bytes()==32,
        "host relocation moved an image held by an asynchronous consumer");
    active.reset();
    require(cache.relocate(arenas,source->allocation_id(),retained)==160 && source->used_bytes()==0 &&
        target->used_bytes()==192 && retained.count(target->allocation_id()) && guard->data()==fixed,
        "host relocation did not consolidate paired survivors into existing holes");
    for(std::uint64_t key:{1,2}) {
        const auto& image=cache.image(key);
        require(image->block->allocation_id()==target->allocation_id() && image->block_offset==0,
            "host relocation retained a stale paired-image offset");
        for(std::size_t i=0;i<image->size();++i)require(image->data()[i]==key,
            "host relocation changed an encoded expert byte");
    }
    for(std::size_t i=0;i<guard->bytes();++i)require(guard->data()[i]==0xa7,
        "host relocation overwrote an unrelated pinned interval");
    {
        Cache fragmented;auto arena=std::make_shared<MoePinnedExpertArena>(256);
        const std::vector<std::shared_ptr<MoePinnedExpertArena>> one{arena};
        auto a=arena->allocate(64),hole=arena->allocate(64),c=arena->allocate(64);
        fragmented.add(1,a,0,64);fragmented.add(2,c,0,64);a.reset();c.reset();hole.reset();
        std::size_t copied=0;
        auto consumer=fragmented.image(2);
        require(!fragmented.compact(one,96,copied) && copied==0,
            "host interval compaction moved a live asynchronous source");
        consumer.reset();
        require(fragmented.compact(one,96,copied) && copied==64 && arena->largest_free_bytes()==128,
            "host interval compaction did not join reusable pinned holes");
        for(std::uint64_t key:{1,2})for(std::size_t i=0;i<64;++i)
            require(fragmented.image(key)->data()[i]==key,"host interval compaction changed an expert byte");
    }
    std::cout<<"Common pinned host relocation: transactional failure, consumer leases and paired byte parity passed\n";
}

void check_routed_storage_fence(int device) {
    auto consumer=std::make_shared<mfq::cuda::Stream>(device);
    auto producer=std::make_shared<mfq::cuda::Stream>(device);
    auto unrelated=std::make_shared<mfq::cuda::Stream>(device);
    mfq::cuda::Event pending(cudaEventDisableTiming);
    const MfqCudaStream lane(device,consumer->get(),consumer);
    auto values=tb::empty({2},tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
    auto host=tb::empty({2},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
    struct Gate {
        std::mutex mutex;std::condition_variable changed;
        bool entered=false,finished=false,released=false,timed_out=false;
        cudaStream_t stream=nullptr;
        std::thread watchdog;
        ~Gate() {
            {std::lock_guard lock(mutex);released=true;}
            changed.notify_all();if(watchdog.joinable())watchdog.join();
            if(stream)(void)cudaStreamSynchronize(stream);
        }
    } gate;
    gate.stream=unrelated->get();
    TYLOINFER_CUDA_CHECK(cudaLaunchHostFunc(unrelated->get(),[](void* opaque) {
        auto& gate=*static_cast<Gate*>(opaque);std::unique_lock lock(gate.mutex);
        gate.entered=true;gate.changed.notify_all();
        gate.changed.wait(lock,[&]{return gate.released;});
    },&gate));
    pending.record(unrelated->get());
    {
        std::unique_lock lock(gate.mutex);
        require(gate.changed.wait_for(lock,std::chrono::seconds(2),[&]{return gate.entered;}),
            "unrelated stream did not enter the storage-fence test");
    }
    gate.watchdog=std::thread([&] {
        std::unique_lock lock(gate.mutex);
        gate.timed_out=!gate.changed.wait_for(lock,std::chrono::seconds(2),[&]{return gate.finished;});
        gate.released=true;gate.changed.notify_all();
    });
    require(!mfq::cuda::cuda_event_ready(pending.get()) && !pending.ready(),
        "common raw/owned event check published an unfinished producer");
    TYLOINFER_CUDA_CHECK(cudaMemsetAsync(values.data_ptr(),0x31,1,consumer->get()));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(host.data_ptr(),values.data_ptr(),1,cudaMemcpyDeviceToHost,consumer->get()));
    TYLOINFER_CUDA_CHECK(cudaMemsetAsync(values.data_ptr<uint8_t>()+1,0x72,1,producer->get()));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(host.data_ptr<uint8_t>()+1,values.data_ptr<uint8_t>()+1,1,
        cudaMemcpyDeviceToHost,producer->get()));
    mfq::cuda::synchronize_routed_streams(lane,{producer->get(),nullptr,producer->get(),consumer->get()});
    {std::lock_guard lock(gate.mutex);gate.finished=true;}gate.changed.notify_all();
    gate.watchdog.join();
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(unrelated->get()));
    require(mfq::cuda::cuda_event_ready(pending.get()) && pending.ready(),
        "common raw/owned event check did not retire a completed producer");
    bool invalid_rejected=false;
    try {(void)mfq::cuda::cuda_event_ready(nullptr);}
    catch(const mfq::cuda::Error&) {invalid_rejected=true;}
    (void)cudaGetLastError();
    require(invalid_rejected,"common event check silently treated an invalid handle as pending");
    require(!gate.timed_out,"routed storage fence waited for an unrelated stream");
    require(host.data_ptr<uint8_t>()[0]==0x31 && host.data_ptr<uint8_t>()[1]==0x72,
        "routed storage fence returned before its consumer or producer completed");
    for(const auto captured:{consumer->get(),producer->get()}) {
        TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(captured,cudaStreamCaptureModeThreadLocal));
        bool rejected=false;
        try{mfq::cuda::synchronize_routed_streams(lane,{producer->get()});}
        catch(const std::logic_error&){rejected=true;}
        cudaGraph_t graph=nullptr;
        TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(captured,&graph));
        if(graph)TYLOINFER_CUDA_CHECK(cudaGraphDestroy(graph));
        require(rejected,"routed storage fence accepted a captured consumer or producer");
    }
    std::cout<<"Common storage fence: producer completion, unrelated-stream isolation and capture rejection passed\n";
}

void check_routed_event_retirement(int device) {
    require(mfq::cuda::routed_native_copy_flags(2)==1,"ordinary uploads lost their overlap hint");
    {
        const mfq::cuda::RoutedHostSubmissionScope outer;
        try {const mfq::cuda::RoutedHostSubmissionScope inner;throw std::runtime_error("interrupted submission");}
        catch(const std::runtime_error&) {}
        require(mfq::cuda::routed_native_copy_flags(2)==0,"nested submission restored an unsafe copy hint");
    }
    require(mfq::cuda::routed_native_copy_flags(2)==1,"failed submission retained its dependency scope");
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kInt32);
    mfq::cuda::RoutedGraphPools pools;pools.prepare(mfq_get_current_cuda_stream());
    {
        mfq::cuda::RoutedCompletionEvents events;
        const auto source=mfq_get_current_cuda_stream(),consumer=mfq_get_stream_from_pool(false,device);
        events.prepare(device,source.stream());events.prepare(device,consumer.stream());
        const auto first=events.at(device,source.stream()),second=events.at(device,consumer.stream());
        require(first!=second,"independent DMA producers shared one completion event");
        events.prepare(device,source.stream());
        require(events.at(device,source.stream())==first,"preparing an existing lane replaced its live event");
        auto value=tb::zeros({1},options);
        for(int expected:{3,7,19,41}) {
            value.fill_(expected);first->record(source.stream());
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(consumer.stream(),first->get(),0));
            int actual=0;
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(&actual,value.data_ptr(),sizeof(actual),cudaMemcpyDeviceToHost,consumer.stream()));
            second->record(consumer.stream());second->synchronize();
            require(actual==expected,"re-recorded completion lost cross-stream producer ordering");
        }
    }
    for(const bool fail:{false,true}) {
        mfq::cuda::HostBuffer publication(64,true);
        auto* flag=static_cast<std::uint32_t*>(publication.data());
        void* mapped=nullptr;TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,flag,0));
        mfq::cuda::publish_mapped_flag(flag);
        auto output=tb::zeros({1},options);
        mfq::cuda::RoutedEventRetirement retirement;
        std::mutex mutex;bool locked=false,safe=false,released_safely=true;int releases=0;
        std::shared_ptr<mfq::cuda::Event> owner;std::weak_ptr<mfq::cuda::Event> weak;
        const auto release=[&] {
            require(!mfq::cuda::RoutedHostSubmissionScope::active(),"completion cleanup retained a waiting-consumer scope");
            std::vector<std::shared_ptr<mfq::cuda::Event>> retired;
            {std::lock_guard lock(mutex);locked=true;retired=retirement.take();locked=false;}
            safe=true;retired.clear();safe=false;
        };
        auto window=std::make_unique<mfq::cuda::DecodeWindow>(mfq_current_cuda_stream());
        window->capture([&] {
            window->enroll({&retirement,[&] {
                *flag=0;output.fill_(0);
                owner=std::shared_ptr<mfq::cuda::Event>(new mfq::cuda::Event(cudaEventDisableTiming),[&](auto* event) {
                    released_safely=released_safely && safe && !locked;++releases;delete event;
                });
                owner->record(mfq_current_cuda_stream());weak=owner;
            },[&] {
                require(mfq::cuda::routed_native_copy_flags(2)==0,"window host work requested compute overlap with its waiting consumer");
                {std::lock_guard lock(mutex);locked=true;
                 retirement.retain(owner);retirement.retain(owner);owner.reset();locked=false;}
                require(!weak.expired(),"completed event was destroyed inside a waiting graph");
                if(fail)throw std::runtime_error("interrupted event retirement");
                mfq::cuda::publish_mapped_flag(flag);
            },release,[&]{mfq::cuda::publish_mapped_flag(flag);},release,{}});
            mfq::cuda::wait_mapped_flag(static_cast<std::uint32_t*>(mapped),mfq_current_cuda_stream());
            output.fill_(17);
        },[]{});
        bool rejected=false;try{window->run();}catch(const std::runtime_error&){rejected=true;}
        require(rejected==fail,"event-retirement graph changed exception propagation");
        if(fail)require(!weak.expired(),"failed graph discarded retirement before synchronized cleanup");
        window.reset();
        require(weak.expired() && releases==1 && released_safely && retirement.size()==0,
            "event retirement leaked or ran under its owner lock");
        require(output.cpu().contiguous().data_ptr<std::int32_t>()[0]==17,
            "event retirement returned before the cancelled or completed GPU consumer");
    }
    std::cout<<"Common event retirement: waiting graphs, unlocked destruction and cancellation passed\n";
}

void check_routed_graph_pool_ownership(int device) {
    const auto context=mfq::cuda::default_context(device);
    auto first=std::make_shared<mfq::cuda::Stream>(device);
    auto second=std::make_shared<mfq::cuda::Stream>(device);
    const auto first_raw=first->get(),second_raw=second->get();
    std::weak_ptr<mfq::cuda::Stream> lifetime=first;
    mfq::cuda::RoutedGraphPools shared;
    const MfqCudaStream a(device,first_raw,first),b(device,second_raw,second);
    require(shared.prepare_shared_buffers(a)==context && shared.prepare_shared_buffers(a)==context,
        "shared graph buffers failed repeated enrollment on their owning stream");
    bool rejected=false;
    try{shared.prepare_shared_buffers(b);}catch(const std::logic_error&){rejected=true;}
    require(rejected && !context->has_graph_pool(second_raw),
        "shared graph buffers enrolled a second execution stream");
    mfq::cuda::RoutedGraphPools other_cache;
    other_cache.prepare(a);shared.synchronize();shared.clear();
    require(context->has_graph_pool(first_raw),"one cache released another cache's shared graph pool");
    other_cache.clear();
    require(!context->has_graph_pool(first_raw),"repeated graph enrollment retained an extra pool reference");
    {
        const MfqCudaStreamGuard lane(a);
        CudaExecutionContext dense_execution;
        std::unique_ptr<mfq::cuda::RoutedGraphPools> owned,second_owner;
        mfq::cuda::prepare_execution_graph_pool(dense_execution,owned);
        const auto* original=owned.get();
        require(original && context->has_graph_pool(first_raw),
            "cache-free execution did not acquire a graph pool");
        mfq::cuda::prepare_execution_graph_pool(dense_execution,owned);
        require(owned.get()==original,"cache-free graph preparation replaced its retained pool");
        bool rejected=false;
        {
            const MfqCudaStreamGuard other(b);
            try{mfq::cuda::prepare_execution_graph_pool(dense_execution,owned);}
            catch(const std::logic_error&){rejected=true;}
        }
        require(rejected && !context->has_graph_pool(second_raw),
            "cache-free graph preparation reused shared buffers on another stream");
        mfq::cuda::prepare_execution_graph_pool(dense_execution,second_owner);
        owned.reset();
        require(context->has_graph_pool(first_raw),"one decoder released another decoder's graph pool");
        second_owner.reset();
        require(!context->has_graph_pool(first_raw),"cache-free graph pool outlived its final owner");
    }
    {
        mfq::cuda::RoutedGraphPools independent;
        independent.prepare(a);independent.prepare(b);independent.prepare(a);
        require(context->has_graph_pool(first_raw) && context->has_graph_pool(second_raw),
            "independent graph buffers did not enroll both execution streams");
        independent.synchronize();
    }
    require(!context->has_graph_pool(first_raw) && !context->has_graph_pool(second_raw),
        "graph pool owner did not release its stream registrations");
    {
        mfq::cuda::RoutedGraphPools retained;
        auto owned=std::make_shared<mfq::cuda::Stream>(device);
        lifetime=owned;retained.prepare(MfqCudaStream(device,owned->get(),owned));owned.reset();
        require(!lifetime.expired(),"graph pool lost its owning stream before release");
        retained.synchronize();retained.clear();
        require(lifetime.expired(),"released graph pool retained its execution stream");
    }
    std::cout<<"Common graph pools: duplicate enrollment, shared-buffer exclusion, independent streams and ownership passed\n";
}

void check_routed_workspace_backing(int device) {
    const auto context=mfq::cuda::default_context(device);
    const auto stream=mfq_current_cuda_stream();
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8);
    auto owner=std::make_shared<std::array<tb::Tensor,2>>(
        std::array{tb::empty({4096},options),tb::empty({4096},options)});
    std::weak_ptr<std::array<tb::Tensor,2>> lifetime=owner;
    std::vector<std::pair<void*,std::size_t>> regions;
    for(const auto& tensor:*owner)regions.emplace_back(tensor.data_ptr(),tensor.nbytes());
    context->begin_workspace_loan(regions,stream,owner);owner.reset();
    const auto largest=context->workspace_largest_available(stream);
    require(largest && largest<context->workspace_available(),
        "workspace capacity merged independent backing allocations");
    const auto other=mfq::cuda::stream_from_pool(false,device);
    require(context->workspace_largest_available(other.stream())==0,
        "workspace capacity crossed an unordered stream");
    const auto available=context->workspace_available();
    require(context->workspace_allocations_fit({3072,3072},0,stream) &&
        !context->workspace_allocations_fit({6000},2000,stream),
        "workspace planning combined independent backing allocations");
    require(!context->workspace_allocations_fit({3072},0,other.stream()) &&
        context->workspace_allocations_fit({3072},3072,other.stream()),
        "workspace planning priced another stream's borrowed interval");
    bool foreign_thread_fit=true;
    std::thread foreign([&]{foreign_thread_fit=context->workspace_allocations_fit({256},0,stream);});
    foreign.join();
    require(!foreign_thread_fit && context->workspace_available()==available,
        "workspace planning changed intervals or admitted another thread");
    auto* borrowed=context->allocate(512,stream);
    require(context->workspace_live_bytes()>=512,"workspace allocation bypassed an eligible loan");
    TYLOINFER_CUDA_CHECK(cudaMemsetAsync(borrowed,7,512,stream));
    bool rejected=false;
    try{context->end_workspace_loan();}catch(const std::exception&){rejected=true;}
    require(rejected && !lifetime.expired(),"workspace end released backing of a live tensor");
    context->retire_workspace_loan();
    require(context->workspace_loan_active() && !lifetime.expired() && !context->workspace_largest_available(stream),
        "retired workspace lost live backing or accepted new allocations");
    std::array<std::uint8_t,512> actual;
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(actual.data(),borrowed,actual.size(),cudaMemcpyDeviceToHost,stream));
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
    require(std::all_of(actual.begin(),actual.end(),[](auto value){return value==7;}),
        "retired workspace changed a live output");
    context->release(borrowed,512,stream);
    require(!context->workspace_loan_active() && lifetime.expired(),"retired workspace retained its completed backing");
    {
        auto backing=tb::empty({1<<20},options),storage=tb::empty({256},options);
        context->begin_workspace_loan({{backing.data_ptr(),backing.nbytes()}},stream);
        int preparations=0;
        auto exponents=prefill_weight_exponents(storage,64,128,{7},[&](const auto& output) {
            ++preparations;auto writable=output;writable.fill_(7);
        });
        auto cached=prefill_weight_exponents(storage,64,128,{7},[&](const auto&){++preparations;});
        require(preparations==1 && cached.data_ptr()==exponents.data_ptr(),
            "immutable prefill row metadata did not retain its cached allocation");
        require(context->workspace_live_bytes()==0,
            "persistent prefill row metadata was allocated from a temporary workspace loan");
        context->end_workspace_loan();backing.fill_(0);
        auto host=exponents.cpu();
        for(int i=0;i<64;++i)require(host.data_ptr<int32_t>()[i]==7,
            "persistent prefill metadata changed after its workspace was returned");
    }
    std::cout<<"Routed workspace: separate regions, stream eligibility, live-output ownership and deferred retirement passed\n";
}

void check_causal_position_workspace(int device) {
    CudaExecutionContext execution;
    mfq::cuda::Qwen4CausalLm model;
    model.execution=&execution;
    auto ops=model.forward_ops(model,{});
    const auto context=mfq::cuda::default_context(device);
    const auto stream=mfq_current_cuda_stream();
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
    auto backing=tb::empty({1<<20},options.dtype(tb::kUInt8));
    context->begin_workspace_loan({{backing.data_ptr(),backing.nbytes()}},stream);
    auto current=ops.position_range(7,4096);
    auto shifted=ops.offset_positions(current,23);
    auto full=model.concat_positions(current,shifted);
    {
        auto scratch=tb::empty({4096},options.dtype(tb::kInt64));
        require(context->workspace_live_bytes()>0,
            "position preparation disabled borrowing for subsequent scratch");
    }
    require(context->workspace_live_bytes()==0,
        "persistent causal positions retained a temporary workspace allocation");
    context->end_workspace_loan();backing.fill_(0);
    const auto actual=full.cpu().contiguous();
    for(int i=0;i<8192;++i)require(actual.data_ptr<int64_t>()[i]==7+i%4096+(i>=4096?23:0),
        "causal positions changed after returning their forward workspace");
    std::cout<<"Causal positions: persistent range, offset and concatenation preserve transient borrowing\n";
}

void check_tq_host_evacuation(int device) {
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
    for(const bool enough:{false,true}) {
        Source source(256,true,{1,8,2,7,4}),oracle_source(256,true,{1,8,2,7,4});
        CudaExecutionContext execution,oracle_execution;
        execution.config.moe_device_cache_bytes=0;execution.config.moe_host_cache_bytes=128u<<10;
        oracle_execution.config.moe_device_cache_bytes=0;oracle_execution.config.moe_host_cache_bytes=128u<<10;
        const auto make=[](auto& owner,const auto& file) {
            return TqMoeBanks{load_tq_routed_weight(owner,file,"gate"),load_tq_routed_weight(owner,file,"up"),
                load_tq_routed_weight(owner,file,"down")};
        };
        const auto banks=make(execution,source),oracle=make(oracle_execution,oracle_source);
        const std::array sources{banks.gate->source_id,banks.up->source_id,banks.down->source_id};
        const std::array roles{banks.gate,banks.up,banks.down};
        std::size_t total=0,smallest=SIZE_MAX;
        for(unsigned expert=0;expert<5;++expert) {
            std::array<std::size_t,3> bytes{};
            for(unsigned role=0;role<3;++role)bytes[role]=roles[role]->table->expert_image_bytes(expert,roles[role]->out);
            const mfq::cuda::RoutedExpertImageLayout layout(sources,bytes);
            total+=layout.bytes;smallest=std::min(smallest,layout.bytes);
        }
        // Register all three banks and retain every encoded image in RAM,
        // without admitting any GPU expert or requiring a real model.
        auto seed=tb::ones({33,33},options.dtype(tb::kFloat16))*.1f;
        auto ids=tb::arange(33,options.dtype(tb::kInt32)).remainder(5).reshape({33,1});
        banks.forward(seed,ids,tb::ones({33,1},options.dtype(tb::kFloat32))).cpu();
        mfq::cuda::finish_routed_expert_exchanges(execution);
        require(metric(execution,"resident_expert_cache_bytes")==0 &&
            metric(execution,"ram_expert_pinned_reserved_bytes")>0,
            "TQ evacuation fixture did not start with unique pinned RAM images");
        const auto budget=enough?total:smallest-16;
        resize_tq_expert_cache(execution.tq_expert_cache,budget);
        finalize_tq_expert_cache(execution.tq_expert_cache,0);
        const auto evictions=metric(execution,"expert_cache_evictions");
        require(reclaim_tq_expert_host(execution.tq_expert_cache,SIZE_MAX)>0,
            "TQ unique-image RAM reclaim did not release a physical allocation");
        require(metric(execution,"expert_cache_evictions")==evictions &&
            metric(execution,"expert_cache_reserved_bytes")<=double(budget) &&
            metric(execution,"resident_expert_cache_bytes")<=double(budget),
            "TQ evacuation evicted a resident or manufactured cache capacity");
        if(!enough) {
            require(metric(execution,"ram_expert_evacuated_to_vram_images")==0 &&
                metric(execution,"resident_expert_cache_bytes")==0,
                "TQ evacuation priced a new group slot as one cheap projection");
            continue;
        }
        require(metric(execution,"ram_expert_evacuated_to_vram_images")==15 &&
            metric(execution,"resident_expert_cache_bytes")==double(total) &&
            metric(execution,"expert_cache_reserved_bytes")==double(total),
            "TQ evacuation lost a mixed-bit group role or its exact complete-slot accounting");
        source.fail_reads_from(0);
        mfq::cuda::RoutedCacheLearningScope frozen(execution,false);
        prepare_tq_graph_pool(execution.tq_expert_cache);
        const auto streamed=tq_expert_streamed_bytes(execution.tq_expert_cache);
        const auto metadata=metric(execution,"expert_decode_metadata_bytes");
        for(const int rows:{1,4}) {
            auto input=tb::ones({rows,33},options.dtype(tb::kFloat16))*.1f;
            auto routes=tb::arange(rows*2,options.dtype(tb::kInt32)).remainder(5).reshape({rows,2});
            auto scores=tb::ones({rows,2},options.dtype(tb::kFloat32))*.5f;
            mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());tb::Tensor output;
            window.capture([&]{output=banks.forward(input,routes,scores);},[]{});
            for(int pass=0;pass<2;++pass) {
                routes.copy_((routes+1).remainder(5));input.copy_(input*.7f);
                const auto expected=oracle.forward(input,routes,scores).to(tb::kFloat32).cpu().contiguous();
                window.run();const auto actual=output.to(tb::kFloat32).cpu().contiguous();
                double error=0,energy=0;
                for(int64_t i=0;i<actual.numel();++i) {
                    const auto delta=actual.data_ptr<float>()[i]-expected.data_ptr<float>()[i];
                    error+=double(delta)*delta;energy+=double(expected.data_ptr<float>()[i])*expected.data_ptr<float>()[i];
                }
                require(std::isfinite(error) && std::sqrt(error/std::max(energy,1e-30))<.003,
                    "TQ RAM evacuation changed M1/M4 graph output or left stale resident addresses");
            }
        }
        require(double(tq_expert_streamed_bytes(execution.tq_expert_cache)-streamed)==
                metric(execution,"expert_decode_metadata_bytes")-metadata,
            "TQ evacuated group required another encoded upload during graph replay");
    }
    {
        Source source(256,true,{1,8,2,7,4});CudaExecutionContext execution;
        execution.config.moe_device_cache_bytes=0;execution.config.moe_host_cache_bytes=128u<<10;
        const auto gate=load_tq_routed_weight(execution,source,"gate");
        auto input=tb::ones({1,33},options.dtype(tb::kFloat16))*.1f;
        auto ids=tb::arange(5,options.dtype(tb::kInt64)).reshape({1,5});
        const auto expected=gate->forward(input,ids).cpu().contiguous();
        std::size_t bytes=0;
        for(unsigned expert=0;expert<5;++expert)
            bytes+=mfq::cuda::routed_aligned_bytes(gate->table->expert_image_bytes(expert,gate->out));
        resize_tq_expert_cache(execution.tq_expert_cache,bytes);
        gate->forward(input,tb::zeros({1,1},options.dtype(tb::kInt64))).cpu();
        finalize_tq_expert_cache(execution.tq_expert_cache,0);
        const auto evictions=metric(execution,"expert_cache_evictions");
        reclaim_tq_expert_host(execution.tq_expert_cache,SIZE_MAX);source.fail_reads_from(0);
        const auto actual=gate->forward(input,ids).cpu().contiguous();
        require(metric(execution,"ram_expert_evacuated_to_vram_images")==4 &&
            metric(execution,"expert_cache_evictions")==evictions &&
            actual.nbytes()==expected.nbytes() && !std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes()),
            "TQ raw-image evacuation displaced a hot resident or lost its existing arena holes");
    }
    std::cout<<"TQ host evacuation: complete mixed-bit slots, exact capacity, hot residents and changing M1/M4 graphs passed\n";
}

void check_pending_tq_dma_retirement(int device) {
    Source source(65536,true,{1,8,2,7,4}),oracle_source(65536,true,{1,8,2,7,4});
    CudaExecutionContext execution,oracle_execution;
    execution.config.moe_device_cache_bytes=2u<<20;
    execution.config.moe_host_cache_bytes=4u<<20;
    oracle_execution.config.moe_device_cache_bytes=0;
    const auto gate=load_tq_routed_weight(execution,source,"gate");
    const auto up=load_tq_routed_weight(execution,source,"up");
    const auto oracle=load_tq_routed_weight(oracle_execution,oracle_source,"up");
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
    const auto input=tb::ones({1,33},options.dtype(tb::kFloat16))*.1f;
    const auto ids=[&](std::vector<int64_t> values) {
        const auto count=values.size();return tb::tensor(values).reshape({1,int64_t(count)}).to(input.device());
    };
    gate->forward(input,ids({0,1,2,3,4})).cpu();
    for(const int expert:{0,2,4})gate->forward(input,ids({expert})).cpu();
    const auto selected=ids({0,1,2});
    const auto actual=up->forward(input,selected).cpu();
    const auto expected=oracle->forward(input,selected).cpu();
    require(actual.sizes()==expected.sizes() && !std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes()),
        "pinned reclamation retired an expert DMA before its completion event was recorded");
    require(metric(execution,"ram_expert_pinned_fragment_retries")>0,
        "pending DMA regression did not exercise pinned fragmentation recovery");
    std::cout<<"TQ pending DMA survives pinned reclamation before event submission with exact outputs passed\n";
}

void check_compact_tq_moe(int device) {
    int major=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
    if(major<7){std::cout<<"SKIP compact TQ product: FP16 WMMA requires SM70\n";return;}
    Source source(256,true,{1,8,2,7,4});
    TqRowTable gate_table(source,"gate"),up_table(source,"up"),down_table(source,"down"),wide_down(source,"linear");
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
    for(int width:{17,33,95})for(bool biased:{false,true}) {
        std::vector<TqWeight> images;std::vector<tb::Tensor> biases;
        std::vector<TrellisPrefillMoeJob> host;
        for(int expert:{0,3}) {
            auto gate=to_device_tq(gate_table.slice_rows(expert*19,(expert+1)*19),33,device);
            auto up=to_device_tq(up_table.slice_rows(expert*19,(expert+1)*19),33,device);
            auto down=to_device_tq(width>33?wide_down.slice_rows(0,width):
                down_table.slice_rows(expert*33,(expert+1)*33),19,device);
            down.out=down.view.rows=width;
            for(auto* image:{&gate,&up,&down})if(biased) {
                biases.push_back((tb::arange(image->out,options.dtype(tb::kFloat32))*.0013f-.01f).contiguous());
                image->view.bias=biases.back().data_ptr<float>();
            }
            host.push_back({gate.view,up.view,down.view,expert==0?5:42,expert==0?37:13});
            images.push_back(std::move(gate));images.push_back(std::move(up));images.push_back(std::move(down));
        }
        std::vector<int64_t> slots(55,-1);
        for(int i=0;i<50;++i)slots[5+i]=(i*17)%50;
        const auto jobs=tb::from_blob(host.data(),{int64_t(host.size()*sizeof(host[0]))},
            tb::TensorOptions().dtype(tb::kUInt8)).clone().to(tb::Device(tb::kCUDA,device));
        const auto device_slots=tb::tensor(slots).to(tb::Device(tb::kCUDA,device));
        const auto bounds=tb::tensor(std::vector<int64_t>{5,42,55}).to(tb::Device(tb::kCUDA,device));
        auto input=(tb::arange(50*33,options.dtype(tb::kFloat32))*.00019f-.1f).reshape({50,33}).to(tb::kFloat16);
        auto output=tb::empty({50,width},options.dtype(tb::kFloat16));
        auto scratch=tb::empty({50*(2*33+4*std::max(width,38))+32},options.dtype(tb::kUInt8));
        const auto check=[&] {
            auto expected=tb::empty({50,width},options.dtype(tb::kFloat16));
            for(int b=0;b<2;++b) {
                const auto ids=device_slots.narrow(0,host[b].first,host[b].count);
                const auto x=input.index_select(0,ids);
                auto g=images[b*3].forward(x).to(tb::kFloat32),u=images[b*3+1].forward(x).to(tb::kFloat32);
                if(biased){g=tb::clamp_max(g,.1);u=tb::clamp(u,-.1,.1);}
                expected.index_copy_(0,ids,images[b*3+2].forward((tb::silu(g)*u).to(tb::kFloat16)));
            }
            const auto a=output.to(tb::kFloat32).cpu(),b=expected.to(tb::kFloat32).cpu();
            double error=0,energy=0;
            for(int64_t i=0;i<a.numel();++i) {
                const auto x=a.data_ptr<float>()[i],y=b.data_ptr<float>()[i];
                require(std::isfinite(x),"compact TQ produced nonfinite values");
                error+=(x-y)*(x-y);energy+=y*y;
            }
            require(std::sqrt(error/std::max(energy,1e-30))<.004,
                "compact TQ differs from independent biased routed projections");
        };
        for(bool tiled:{false,true}) {
            auto decoded=tb::empty({tiled?int64_t(2*33*2*17):int64_t(2*33*38*2)},options.dtype(tb::kUInt8));
            mfq::cuda::Event down_ready(cudaEventDisableTiming);
            const auto run=[&] {
                down_ready.record(mfq_current_cuda_stream());
                const int status=mfq_tq_prefill_moe_compact(reinterpret_cast<const TrellisPrefillMoeJob*>(jobs.data_ptr()),
                    device_slots.data_ptr<int64_t>(),bounds.data_ptr<int64_t>(),input.data_ptr(),output.data_ptr(),2,37,5,50,
                    33,19,width,1,biased?.1f:0.f,decoded.data_ptr(),decoded.nbytes(),scratch.data_ptr(),scratch.nbytes(),
                    down_ready.get(),mfq_current_cuda_stream());
                if(status)throw std::runtime_error("compact TQ API status="+std::to_string(status)+
                    " width="+std::to_string(width)+" tiled="+std::to_string(tiled));
            };
            run();check();
            mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());graph.capture(run,[]{});
            for(int pass=0;pass<2;++pass){input.copy_(input*.7f);graph.run();check();}
        }
    }
    std::cout<<"Compact TQ PP: ragged routes, prefix offsets, biased tails, independent output width and changing graphs passed\n";
}

void check_tq_predictor_residency(int device) {
    Source source(256,true,{1,8,2,7,4}),reference_source(256,true,{1,8,2,7,4});
    CudaExecutionContext execution,reference_execution;
    execution.config.moe_device_cache_bytes=32u<<10;
    execution.config.moe_host_cache_bytes=256u<<10;
    reference_execution.config.moe_device_cache_bytes=0;
    const auto make=[&](CudaExecutionContext& owner,const Source& file,const std::string& prefix) {
        return TqMoeBanks{load_tq_routed_weight(owner,file,prefix+"gate"),
            load_tq_routed_weight(owner,file,prefix+"up"),load_tq_routed_weight(owner,file,prefix+"down")};
    };
    auto predictor=make(execution,source,"predictor.");
    const std::array banks{predictor.gate,predictor.up,predictor.down};
    std::array<const void*,3> packed{},tables{};
    std::size_t permanent_bytes=0;
    for(std::size_t role=0;role<banks.size();++role) {
        require(bool(banks[role]->resident),"TQ predictor did not retain an immutable expert bank");
        packed[role]=banks[role]->resident->packed.data_ptr();
        tables[role]=banks[role]->resident->views.data_ptr();
        permanent_bytes+=banks[role]->resident->packed.nbytes();
        require(banks[role]->resident->experts.size()==5,"TQ predictor lost a resident expert");
    }
    require(permanent_bytes>0 && metric(execution,"tq_predictor_resident_expert_bytes")==double(permanent_bytes),
        "TQ predictor resident accounting differs from its encoded allocation");
    require(tq_expert_source_bytes(execution.tq_expert_cache)==0 &&
        mfq::cuda::routed_expert_cache_budget_bytes(execution)==0,
        "predictor-only TQ cache charged an evictable source or budget");

    auto main=make(execution,source,"");
    auto reference=make(reference_execution,reference_source,"");
    const auto evictable=source.find_tensor("gate")->nbytes+source.find_tensor("up")->nbytes+
        source.find_tensor("down")->nbytes;
    require(tq_expert_source_bytes(execution.tq_expert_cache)==evictable,
        "TQ predictor payload entered the main expert source budget");
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
    auto seed_input=tb::ones({1,33},options.dtype(tb::kFloat16))*.1f;
    auto seed_ids=tb::zeros({1,1},options.dtype(tb::kInt32));
    auto seed_scores=tb::ones({1,1},options.dtype(tb::kFloat32));
    require(mfq::cuda::default_context(device)->has_graph_pool(mfq_current_cuda_stream()),
        "TQ predictor test has no graph pool");
    {
        mfq::cuda::DecodeWindow main_window(mfq_current_cuda_stream());tb::Tensor output;
        main_window.capture([&]{output=main.forward(seed_input,seed_ids,seed_scores);},[]{});
        main_window.run();output.cpu();
    }
    mfq::cuda::finish_routed_expert_exchanges(execution);
    require(metric(execution,"expert_admission_observations")>0,
        "TQ predictor isolation test did not first train the main expert cache");
    // Every subsequent predictor warmup/replay must use its original immutable
    // storage, including after all ordinary expert capacity has been removed.
    source.fail_reads_from(0);
    resize_tq_expert_cache(execution.tq_expert_cache,0);
    require(mfq::cuda::routed_expert_cache_budget_bytes(execution)==0 &&
        metric(execution,"resident_expert_cache_bytes")==0,
        "zero TQ cache budget retained evictable main-model experts");
    const auto observations=metric(execution,"expert_admission_observations");
    const auto route_mass=metric(execution,"expert_admission_total_route_mass");
    const auto transfers=tq_expert_streamed_bytes(execution.tq_expert_cache);
    for(const auto dtype:{tb::kFloat16,tb::kFloat32})for(const int rows:{1,4}) {
        auto input=tb::empty({rows,33},options.dtype(dtype));
        auto ids=tb::zeros({rows,3},options.dtype(tb::kInt32));
        auto scores=tb::ones({rows,3},options.dtype(tb::kFloat32))/3.f;
        input.fill_(.1f);
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());tb::Tensor output;
        window.capture([&]{output=predictor.forward(input,ids,scores,.25f);},[]{});
        const auto nodes=window.nodes();
        for(int pass=0;pass<6;++pass) {
            auto host_input=tb::empty({rows,33},tb::TensorOptions().dtype(tb::kFloat32));
            auto host_ids=tb::empty({rows,3},tb::TensorOptions().dtype(tb::kInt32));
            for(int token=0;token<rows;++token) {
                for(int column=0;column<33;++column)
                    host_input.data_ptr<float>()[token*33+column]=std::sin(float((pass+1)*17+token*33+column)*.071f)*.2f;
                for(int route=0;route<3;++route)
                    host_ids.data_ptr<std::int32_t>()[token*3+route]=(pass+token+route*2)%5;
            }
            input.copy_(host_input.to(input.device()).to(dtype));ids.copy_(host_ids.to(ids.device()));
            const auto expected=reference.forward(input,ids,scores,.25f).to(tb::kFloat32).cpu().contiguous();
            window.run();const auto actual=output.to(tb::kFloat32).cpu().contiguous();
            double error=0,norm=0;
            require(actual.sizes()==expected.sizes(),"TQ predictor graph changed its output shape");
            for(std::int64_t i=0;i<actual.numel();++i) {
                const double a=actual.data_ptr<float>()[i],b=expected.data_ptr<float>()[i];
                require(std::isfinite(a) && std::isfinite(b),"TQ predictor graph produced a nonfinite result");
                error+=(a-b)*(a-b);norm+=b*b;
            }
            require(std::sqrt(error/std::max(norm,1e-20))<.003,
                "TQ predictor changing graph differs from independent streamed expert projections");
            require(window.nodes()==nodes,"TQ predictor route changes rebuilt the graph");
            for(std::size_t role=0;role<banks.size();++role)
                require(banks[role]->resident->packed.data_ptr()==packed[role] &&
                    banks[role]->resident->views.data_ptr()==tables[role],
                    "TQ predictor graph replaced its permanent encoded image or descriptor table");
        }
    }
    require(metric(execution,"tq_predictor_resident_expert_bytes")==double(permanent_bytes) &&
        tq_expert_source_bytes(execution.tq_expert_cache)==evictable &&
        mfq::cuda::routed_expert_cache_budget_bytes(execution)==0 &&
        metric(execution,"resident_expert_cache_bytes")==0,
        "TQ predictor replay changed permanent or evictable resource accounting");
    require(metric(execution,"expert_admission_observations")==observations &&
        metric(execution,"expert_admission_total_route_mass")==route_mass &&
        metric(execution,"expert_admission_pending_routes")==0,
        "TQ predictor capture/replay polluted the main expert cache observations or decay");
    require(tq_expert_streamed_bytes(execution.tq_expert_cache)==transfers,
        "TQ predictor replay transferred permanent expert weights or descriptor metadata");
    std::cout<<"TQ predictor mixed-bit immutable residency, zero cache, M1/M4 changing graphs and main-cache isolation passed\n";
}

void check_tq_prefill_scales(int device) {
    for(const auto precisions:{std::array<unsigned,5>{2,7,3,6,4},std::array<unsigned,5>{1,5,8,2,3}}) {
    Source source(2048,true,precisions);const auto original=load_tq_weight(source,"linear",device);
    const auto options=original.image.options();
    const auto scales=(tb::arange(2048,options.dtype(tb::kFloat32))*.0001f+.5f).to(tb::kFloat16);
    const auto exact=[](const tb::Tensor& a,const tb::Tensor& b) {
        const auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
        require(x.sizes().vec()==y.sizes().vec() && x.scalar_type()==y.scalar_type() &&
            !std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes()),"direct TQ scaled decoding changed values or exponents");
    };
    for(bool columns:{false,true})for(int width:{33,256,1025}) {
        auto weight=original;weight.neuron_len=weight.view.input_columns=width;
        if(columns){weight.view.columns=1;weight.view.scales=scales.data_ptr();}
        auto decoded=tb::empty({weight.out,width},options.dtype(tb::kFloat32));
        auto half=tb::empty({weight.out,width},options.dtype(tb::kFloat16));
        auto exponents=tb::empty({weight.out},options.dtype(tb::kInt32));
        require(!mfq_tq_embedding(weight.view,nullptr,int(weight.out),decoded.data_ptr(),2,mfq_current_cuda_stream()),
            "TQ exponent reference decoding failed");
        mfq::cuda::gguf::pack_scaled_f16_unit(decoded.data_ptr<float>(),half.data_ptr(),exponents.data_ptr<int32_t>(),
            weight.out,width,mfq_current_cuda_stream());
        const auto adapter=prefill_projection_weight(weight);
        require(adapter.decode_row_bytes==std::uint64_t(width)*2,"TQ decoder still budgets an FP32 image");
        const auto branch=mfq::cuda::stream_from_pool(false,device);
        PrefillWeightImage first;
        {const MfqCudaStreamGuard guard(branch);first=adapter.decode();}
        const auto second=adapter.decode();
        require(first.column_exponents.data_ptr()==second.column_exponents.data_ptr(),"TQ row exponents were recomputed");
        exact(second.values,half);exact(second.column_exponents,exponents);
    }
    }
    std::cout<<"TQ row scales: 1..8 bits, direct FP16 images, logical tails, column scales and cross-stream reuse passed\n";
}

void check_loaded_tq_groups(int device) {
    Source source;CudaExecutionContext execution;
    for(const auto& names:{std::array<const char*,3>{"linear","auxiliary","linear"},
            {"mx4.linear","mx8.linear","mx4.linear"},
            {"gguf.linear","gguf.linear","gguf.linear"}}) {
    std::vector<mfq::cuda::weight_loader::Linear> functions{
        mfq::cuda::weight_loader::linear(execution,source,names[0]),
        mfq::cuda::weight_loader::linear(execution,source,names[1]),
        mfq::cuda::weight_loader::linear(execution,source,names[2])};
    const auto width=mfq::cuda::weight_loader::linear_weight(functions.front())->neuron_len();
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kFloat32);
    for(bool combine:{false,true})for(std::size_t count:{2u,3u})for(std::size_t prefix:{1u,2u}) {
        auto selected=std::vector<mfq::cuda::weight_loader::Linear>(functions.begin(),functions.begin()+count);
        auto grouped=mfq::cuda::weight_loader::grouped_linear(execution,selected,prefix,combine);
        require(bool(grouped),"loaded projection sharing was rejected without row concatenation");
        for(bool parallel:{false,true})for(int tokens:{1,3,513}) {
            execution.config.decode_branch_parallel=parallel;
            auto input=(tb::arange(tokens*width,options)*.00013f-.1f).reshape({tokens,width});
            const auto check=[&](const std::vector<tb::Tensor>& actual) {
                std::vector<tb::Tensor> expected;
                for(auto& function:selected)expected.push_back(function(execution,input));
                if(prefix>1) {
                    auto joined=tb::cat(std::vector<tb::Tensor>(expected.begin(),expected.begin()+prefix),-1);
                    expected.erase(expected.begin(),expected.begin()+prefix);expected.insert(expected.begin(),joined);
                }
                require(actual.size()==expected.size(),"TQ grouped projection output count changed");
                for(std::size_t i=0;i<actual.size();++i) {
                    const auto a=actual[i].to(tb::kFloat32).contiguous().cpu(),
                        b=expected[i].to(tb::kFloat32).contiguous().cpu();
                    require(a.sizes().vec()==b.sizes().vec(),"TQ grouped projection output order changed");
                    for(int64_t j=0;j<a.numel();++j)require(std::abs(a.data_ptr<float>()[j]-b.data_ptr<float>()[j])<
                        1e-6f*(1.f+std::abs(b.data_ptr<float>()[j])),"TQ grouped projection changed its result");
                }
            };
            check(grouped(execution,input));
            if(tokens==3 && parallel && !combine) {
                std::vector<tb::Tensor> output;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
                window.capture([&]{output=grouped(execution,input);},[]{});
                input.copy_(input*.75f);window.run();check(output);
            }
        }
    }
    }
    std::cout<<"TQ/GGUF/MXFP loaded groups: immutable images, mixed formats, joined outputs and changing graphs passed\n";
}

void check_indexed_column_tails(int device) {
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
    int cases=0;
    for(const auto bits:{std::array<unsigned,5>{1,2,4,6,8},std::array<unsigned,5>{2,3,5,7,4}}) {
        Source source(256,true,bits);
        const auto base=load_tq_weight(source,"linear",device);
        for(const int width:{1,17,33,63}) {
            auto scales=tb::full({width+16},std::numeric_limits<float>::quiet_NaN(),
                options.device(tb::kCPU).dtype(tb::kFloat32));
            scales.narrow(0,0,width).fill_(.25);
            scales=scales.to(tb::Device(tb::kCUDA,device)).to(tb::kFloat16);
            auto weight=base;weight.neuron_len=weight.view.input_columns=width;
            weight.view.columns=1;weight.view.scales=scales.data_ptr();
            auto rows=tb::arange(weight.out,options.dtype(tb::kInt64));
            auto decoded=tb::empty({weight.out,width},options.dtype(tb::kFloat32));
            require(mfq_tq_embedding(weight.view,rows.data_ptr<int64_t>(),int(weight.out),
                decoded.data_ptr(),2,mfq_current_cuda_stream())==0,"column-tail reference decode failed");
            decoded=decoded.cpu().contiguous();
            for(const int tokens:{1,2,3,4}) {
                auto input=((tb::arange(tokens*width,options.dtype(tb::kFloat32)).remainder(13)-6)
                    .reshape({tokens,width})/32.).to(tb::kFloat16);
                auto output=tb::empty({tokens,weight.out},options.dtype(tb::kFloat16));
                TrellisIndexedProjectionJob descriptor{};descriptor.weight=weight.view;descriptor.count=tokens;
                for(int t=0;t<tokens;++t) {
                    descriptor.input_offsets[t]=int64_t(t)*width;
                    descriptor.output_offsets[t]=int64_t(t)*weight.out;
                }
                auto jobs=tb::from_blob(&descriptor,{int64_t(sizeof(descriptor))},
                    tb::TensorOptions().dtype(tb::kUInt8)).clone().to(tb::Device(tb::kCUDA,device));
                const auto launch=[&] {
                    require(mfq_tq_indexed(reinterpret_cast<const TrellisIndexedProjectionJob*>(jobs.data_ptr()),
                        input.data_ptr(),output.data_ptr(),1,int(weight.out),0,tokens,mfq_current_cuda_stream())==0,
                        "indexed column-tail projection failed");
                };
                mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
                if(tokens>2)window.capture(launch,[]{});
                for(int pass=0;pass<2;++pass) {
                    if(pass)input.copy_(input*.5);
                    if(tokens>2)window.run();else launch();
                    const auto x=input.to(tb::kFloat32).cpu().contiguous();
                    const auto y=output.to(tb::kFloat32).cpu().contiguous();
                    for(int t=0;t<tokens;++t)for(int64_t row=0;row<weight.out;++row) {
                        double expected=0;
                        for(int col=0;col<width;++col)expected+=double(x.data_ptr<float>()[t*width+col])*
                            decoded.data_ptr<float>()[row*width+col];
                        const auto actual=y.data_ptr<float>()[t*weight.out+row];
                        require(std::isfinite(actual) && std::isfinite(expected) &&
                            std::abs(actual-expected)<=.00004+.001*std::abs(expected),
                            "indexed projection consumed poisoned column padding or changed its dot product");
                    }
                    ++cases;
                }
            }
        }
    }
    std::cout<<"Indexed TQ columns: "<<cases<<" FP64 references with poisoned tails and changing graphs PASS\n";
}

void check_tq_ffn_prefill_residual(int device) {
    Source source;CudaExecutionContext execution;
    FFN ffn;auto linear=load_quant_linear(execution,source,"linear");
    ffn.gate_up=make_quant_group(execution,{linear,linear});ffn.down=load_quant_linear(execution,source,"linear_down");
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kFloat16);
    auto input=(tb::arange(513*33,options.dtype(tb::kFloat32))*.00013f-.1f).reshape({513,33}).to(tb::kFloat16);
    auto residual=(tb::arange(513*165,options.dtype(tb::kFloat32))*.000017f-.2f).reshape({513,165}).to(tb::kFloat16);
    const auto unchanged=residual.cpu();
    const auto actual=ffn.try_forward_prefill_residual(execution,input,residual);
    require(actual && actual->defined(),"dense TQ FFN did not select shared prefill residual GEMM");
    auto parts=ffn.gate_up.forward(execution,input);auto hidden=tb::empty_like(parts[0]);
    mfq::cuda::materialize_gated_input(parts[1].data_ptr(),parts[0].data_ptr(),true,hidden.numel(),true,
        hidden.data_ptr(),mfq_current_cuda_stream());
    const auto expected=(ffn.down.forward(execution,hidden).to(tb::kFloat32)+residual.to(tb::kFloat32)).to(tb::kFloat16).to(tb::kFloat32).cpu();
    const auto values=actual->to(tb::kFloat32).cpu();double error=0,energy=0;
    for(int64_t i=0;i<expected.numel();++i) {
        const auto a=values.data_ptr<float>()[i],b=expected.data_ptr<float>()[i];
        require(std::isfinite(a),"TQ residual prefill produced a nonfinite value");error+=(a-b)*(a-b);energy+=b*b;
    }
    require(std::sqrt(error/std::max(energy,1e-30))<.004,"TQ residual GEMM exceeded its FP16 rounding bound");
    const auto after=residual.cpu();require(!std::memcmp(after.data_ptr(),unchanged.data_ptr(),after.nbytes()),
        "TQ residual GEMM modified its immutable residual");
    require(!ffn.try_forward_prefill_residual(execution,input.narrow(0,0,4),residual.narrow(0,0,4)),
        "small-M TQ verification entered the PP residual path");
    tb::Tensor result;mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    graph.capture([&]{result=*ffn.try_forward_prefill_residual(execution,input,residual);},[]{});
    input.copy_(input*.75f);graph.run();
    const auto reference=ffn.try_forward_prefill_residual(execution,input,residual)->cpu();
    const auto replayed=result.cpu();require(!std::memcmp(reference.data_ptr(),replayed.data_ptr(),reference.nbytes()),
        "TQ residual FFN graph failed to replay changing inputs");
    std::cout<<"Dense TQ FFN prefill residual: independent epilogue, immutable addend, tails and changing graph passed\n";
}

void check_tq_residuals(int device) {
    Source source(65536,true);CudaExecutionContext execution;
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kFloat32);
    const auto original=load_tq_weight(source,"linear",device);
    const auto scales=(tb::arange(original.view.stored_columns,options)*.0001f+.5f).to(tb::kFloat16);
    const auto bias=(tb::arange(original.out,options)*.0002f-.01f);
    const auto exact=[](const tb::Tensor& actual,const tb::Tensor& expected) {
        const auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
        require(a.sizes().vec()==b.sizes().vec() && a.scalar_type()==b.scalar_type() &&
            !std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()),"TQ residual epilogue differs from separate projection");
    };
    int cases=0;
    for(const bool columns:{false,true})for(const int width:{33,320,1025,8193,32769})for(const int tokens:{1,2,3,4}) {
        auto base_weight=original;base_weight.neuron_len=base_weight.view.input_columns=width;base_weight.view.bias=bias.data_ptr<float>();
        if(columns){base_weight.view.columns=1;base_weight.view.scales=scales.data_ptr();}
        auto input=(tb::arange(tokens*(width+7),options)*.00013f-.1f).reshape({tokens,width+7}).narrow(1,3,width);
        for(const int streams:{1,3,4,5,7}) {
            auto weight=base_weight;weight.out=weight.view.rows=int(weight.out-weight.out%streams);
            TqProjectionGroup reference(std::vector<TqWeight>{weight});
            auto normalized=(tb::arange(tokens*weight.out,options)*.0007f-.2f).reshape({tokens,weight.out});
            std::vector<float> gate_values(size_t(streams)*width);
            for(size_t i=0;i<gate_values.size();++i)gate_values[i]=float(int(i%43)-21)*.0002f;
            const auto gate_weight=tb::tensor(gate_values).reshape({streams,width}).to(tb::Device(tb::kCUDA,device)).to(tb::kBFloat16);
            tb::Tensor actual[3];
            std::vector<tb::Tensor> combined;
            mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
            window.capture([&] {
                actual[0]=weight.try_residual(input,{},streams);
                actual[1]=weight.try_residual(input,{},streams,true);
                actual[2]=weight.try_residual(input,normalized,streams);
                combined=weight.try_residual_down(input,gate_weight,streams);
            },[]{});
            for(int turn=0;turn<2;++turn) {
                input.copy_(input+.03125f);normalized.copy_(normalized-.015625f);window.run();
                const auto projected=weight.forward(input);
                exact(projected,reference.forward(input).front());
                exact(actual[0],mfq_qwen4_exp::gated_residual_bottleneck(projected,streams));
                exact(actual[1],mfq_qwen4_exp::gated_residual_injection(projected,streams));
                exact(actual[2],mfq_qwen4_exp::gated_residual_mix(projected,normalized,streams));cases+=3;
                require(combined.size()==2,"TQ/BF16 combined residual projection was not selected");
                exact(combined[0],actual[0]);
                exact(combined[1],try_gr_bf16_injection_cuda(input.contiguous(),gate_weight,streams));cases+=2;
            }
            require(!weight.try_residual(input.to(tb::kFloat16),{},streams).defined(),
                "TQ residual changed the FP16 intermediate rounding path");
            require(!weight.try_residual(input,normalized,weight.out+1).defined(),
                "TQ residual accepted a fractional stream width");
            require(weight.try_residual_down(input.to(tb::kFloat16),gate_weight,streams).empty(),
                "TQ combined residual changed FP16 rounding");
            require(weight.try_residual_down(input,gate_weight.to(tb::kFloat32),streams).empty(),
                "TQ combined residual accepted a non-BF16 injection");
        }
    }
    auto loaded=mfq::cuda::weight_loader::residual_linear(execution,source,"linear");
    require(bool(loaded.mixed) && bool(loaded.activated),"TQ residual loader omitted its fused callbacks");
    auto input=tb::ones({1,33},options)*.125f,normalized=tb::ones({1,95},options)*.25f;
    exact(loaded.activated(execution,input,5,false),
        mfq_qwen4_exp::gated_residual_bottleneck(loaded.forward(execution,input),5));
    exact(loaded.mixed(execution,input,normalized,5),
        mfq_qwen4_exp::gated_residual_mix(loaded.forward(execution,input),normalized,5));
    auto gate=std::make_shared<QuantLinear>();gate->kind=QuantLinearKind::Dense;
    gate->dense=(tb::ones({3,original.neuron_len},options)*.002f).to(tb::kBFloat16);
    mfq::cuda::weight_loader::ResidualLinear injection;
    injection.weight=gate;
    injection.forward=[gate](CudaExecutionContext&,const tb::Tensor& x) {
        return tb::matmul(x,gate->dense.to(tb::kFloat32).transpose(0,1));
    };
    injection.activated=[gate](CudaExecutionContext&,const tb::Tensor& x,int64_t streams,bool) {
        return try_gr_bf16_injection_cuda(x,gate->dense,streams);
    };
    auto combined=mfq::cuda::weight_loader::residual_down_injection(loaded,injection);
    require(bool(combined),"TQ/BF16 residual loader omitted its combined projection");
    const auto x=tb::ones({2,original.neuron_len},options)*.125f;
    const auto norm=tb::ones({original.neuron_len},options);
    const auto branch=tb::ones({2,original.neuron_len/3},options)*.03f;
    const auto prior=tb::ones({2,3},options)*.5f;
    GatedResidualProjection up=[width=original.neuron_len](CudaExecutionContext&,const tb::Tensor& low) {
        return tb::ones({low.size(0),width},low.options())*.125f;
    };
    int combined_calls=0;
    GatedResidualDownInjectionProjection observed=[&](CudaExecutionContext& context,const tb::Tensor& values,int64_t streams) {
        ++combined_calls;return combined(context,values,streams);
    };
    for(const bool after:{false,true}) {
        const auto run=[&](const GatedResidualDownInjectionProjection& fused) {
            if(after)return gated_residual_pre_after_projected(execution,branch,x,prior,norm,loaded.forward,up,
                injection.forward,original.neuron_len/3,3,1e-6,{}, {},loaded.activated,injection.activated,fused);
            return gated_residual_pre_projected(execution,x,norm,loaded.forward,up,injection.forward,
                original.neuron_len/3,3,1e-6,{}, {},loaded.activated,injection.activated,fused);
        };
        const auto separate=run({}),fused=run(observed);
        for(size_t i=0;i<separate.size();++i)exact(fused[i],separate[i]);
    }
    require(combined_calls==2,"TQ/BF16 combined residual callback was not used by chained GR");
    std::cout<<"TQ residual Q8 activation/mix: "<<cases<<" exact changing-graph cases passed\n";
}

void check_mapped_embeddings(Source& source,CudaExecutionContext& execution,int device) {
    const auto before=mfq::cuda::mapped_embedding_host_bytes.load();
    int cases=0;
    for(const auto* name:{"linear","gguf.linear","payload.rows.F16","payload.rows.BF16","payload.rows.F32"}) {
        auto reference=load_quant_linear(execution,source,name);
        execution.loading_cpu_layer=true;
        auto host=load_quant_linear(execution,source,name);
        execution.loading_cpu_layer=false;
        for(const bool control:{false,true})for(const auto id_type:{tb::kInt32,tb::kInt64})
        for(const int count:{1,4,17}) {
            auto mapped=host.kind==QuantLinearKind::Dense?map_dense_embedding(host.dense,device,control):
                map_quant_embedding(host,device,control);
            auto ids=tb::zeros({1,count},tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(id_type));
            tb::Tensor output;
            mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
            window.capture([&]{output=quant_embedding_lookup(mapped,ids);},[]{},[&]{output={};});
            for(int step=0;step<3;++step) {
                std::vector<int64_t> next(count);
                for(int i=0;i<count;++i)next[i]=i%3==0?reference.out()-1:i%3==1?0:(i*11+step*7)%reference.out();
                ids.copy_(tb::tensor(next).reshape({1,count}).to(id_type).to(tb::Device(tb::kCUDA,device)));
                window.run();
                const auto actual=output.cpu().contiguous();
                const auto expected=quant_embedding_lookup(reference,ids).cpu().contiguous();
                require(actual.sizes().vec()==expected.sizes().vec() && actual.scalar_type()==expected.scalar_type() &&
                    !std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes()),
                    "mapped embedding differs from its original device bytes");
                ++cases;
            }
        }
        if(host.kind==QuantLinearKind::Tq) {
            bool rejected=false;
            try {(void)host.tq.with_image(tb::empty({1},host.tq.image.options()));}
            catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"TQ image rebind accepted a truncated allocation");
        }
    }
    require(mfq::cuda::mapped_embedding_host_bytes.load()==before,"mapped embedding owner leaked host bytes");
    std::cout<<"Mapped TQ/GGUF/F16/BF16/F32 embeddings: "<<cases<<" exact changing-graph cases passed\n";
}

void check_selected_projection_rows(Source& source,CudaExecutionContext& execution,int device) {
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
    int cases=0;
    for(const auto* name:{"linear","gguf.linear","payload.rows.F16","payload.rows.BF16","payload.rows.F32"}) {
        const auto weight=load_quant_linear(execution,source,name);
        const auto all=tb::arange(weight.out(),options.dtype(tb::kInt64));
        const auto decoded=(weight.is_gguf()?weight.gguf.embedding(all):quant_embedding_lookup(weight,all))
            .to(tb::kFloat32).cpu().contiguous();
        const std::vector<int64_t> rows{weight.out()-1,0,3,3,weight.out()/2};
        for(const auto type:{tb::kFloat16,tb::kBFloat16,tb::kFloat32})for(const int count:{1,3}) {
            auto input=(tb::arange(count*weight.neuron_len(),options.dtype(tb::kFloat32)).remainder(17)-8)/16;
            input=input.reshape({1,count,weight.neuron_len()}).to(type);
            const auto actual=weight.forward_rows(execution,input,rows).cpu().contiguous();
            const auto x=input.to(tb::kFloat32).cpu().contiguous();
            require(actual.sizes().vec()==std::vector<int64_t>({1,count,int64_t(rows.size())}) &&
                actual.scalar_type()==tb::kFloat32,"selected projection changed output geometry or precision");
            for(int token=0;token<count;++token)for(std::size_t i=0;i<rows.size();++i) {
                double expected=0;
                for(int64_t k=0;k<weight.neuron_len();++k)
                    expected+=double(x.data_ptr<float>()[token*weight.neuron_len()+k])*
                        decoded.data_ptr<float>()[rows[i]*weight.neuron_len()+k];
                require(std::abs(actual.data_ptr<float>()[token*rows.size()+i]-expected)<=
                    2e-5*std::max(1.,std::abs(expected)),"selected projection differs from decoded FP64 dot");
            }
            require(weight.forward_rows(execution,input,{}).size(-1)==0,
                "empty selected projection produced output channels");
            ++cases;
        }
        const auto input=tb::zeros({1,weight.neuron_len()},options.dtype(tb::kFloat16));
        for(const int64_t invalid:{int64_t(-1),weight.out()}) {
            bool rejected=false;
            try{(void)weight.forward_rows(execution,input,{invalid});}catch(const std::out_of_range&){rejected=true;}
            require(rejected,"selected projection accepted an out-of-range output channel");
        }
    }
    std::cout<<"Selected TQ/GGUF/F16/BF16/F32 projections: "<<cases<<" FP64 oracles, repeated rows and bounds passed\n";
}
}
int main()try {
    {
        Source source;source.batch_rows=true;
        GgufRowTable table(source,"rows.F32");
        std::vector<float> row(13);table.read_row(3,row.data());
        require(source.mapped_requests==0 && *source.batch_requests==1,
            "sparse rows bypassed the source's batch I/O scheduler");
        for(int column=0;column<13;++column)require(row[column]==.75f+column*.03125f,
            "source batch row changed floating payload bytes");
    }
    {
        Source source;source.batch_rows=true;
        TqRowTable table(source,"gate");table.cache_row_metadata();
        const auto images=table.slice_experts({4,0,3},19);
        require(*source.batch_requests==1,"TQ expert payloads bypassed batch source I/O");
        for(std::size_t index=0;index<images.size();++index) {
            const int expert=std::array{4,0,3}[index];
            require(images[index]==table.slice_rows(expert*19,(expert+1)*19),"batched TQ expert changed encoded bytes");
        }
    }
    {
        mfq::cuda::ExpertRouteUsage usage;
        usage.image(1,32);usage.image(2,64);usage.image(3,16);
        usage.prefill(1,20,64);usage.prefill(2,10,64);usage.prefill(3,1,64);
        usage.select(48);
        require(usage.retained_images()==2 && usage.retained(1) && usage.retained(3),"route admission exceeded byte budget");
        usage.select(0);require(usage.retained_images()==0,"zero route budget retained an image");
        const auto score=usage.score(1);usage.decay(4);
        require(std::abs(usage.score(1)-.7f*score)<1e-5f,"route decay changed");
        usage.reset_prefill(0);require(usage.score(1)==0,"new prefill retained stale route demand");
    }
    {
        mfq::cuda::ExpertRouteUsage usage;
        std::vector<std::uint64_t> ranked;
        for(std::uint64_t key=0;key<97;++key) {
            usage.image(key,17*(1+key%11));usage.prefill(key,(key*13)%19,64);ranked.push_back(key);
        }
        std::sort(ranked.begin(),ranked.end(),[&](auto a,auto b) {
            return usage.score(a)!=usage.score(b)?usage.score(a)>usage.score(b):a<b;
        });
        for(const auto budget:{std::size_t(0),std::size_t(1),std::size_t(33),std::size_t(255),
                std::size_t(1024),std::numeric_limits<std::size_t>::max()}) {
            std::unordered_set<std::uint64_t> expected;auto remaining=budget;
            for(auto key:ranked)if(usage.score(key)>0 && 17*(1+key%11)<=remaining) {
                expected.insert(key);remaining-=17*(1+key%11);
            }
            usage.select(budget);require(usage.retained_images()==expected.size(),"admission selection count changed");
            for(auto key:ranked)require(usage.retained(key)==bool(expected.count(key)),
                "admission selection differs from score/key ordered byte fitting");
        }
    }

    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    for(int device=0;device<devices;++device) {
        MfqCudaGuard guard(device);Source source;
        TqRowTable table(source,"gate");const auto wire=table.slice_rows(0,19);
        const mfq::TqLayout layout(wire.data(),wire.size());
        const auto bytes=((layout.packed+3)&~std::size_t(3))+layout.packed_bytes+32;
        CudaExecutionContext bounded,reference;
        bounded.config.moe_device_cache_bytes=reference.config.moe_device_cache_bytes=0;
        bounded.config.moe_host_cache_bytes=bytes;reference.config.moe_host_cache_bytes=4*bytes;
        auto gate=load_tq_routed_weight(bounded,source,"gate");
        auto reference_gate=load_tq_routed_weight(reference,source,"gate");
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
        auto input=tb::ones({1,33},options.dtype(tb::kFloat16))*.1;
        gate->forward(input,tb::zeros({1,1},options.dtype(tb::kInt64))).cpu();
        require(metric(bounded,"resident_expert_cache_bytes")==0 &&
            metric(bounded,"ram_expert_payload_bytes")==bytes,"TQ late-miss test did not prime its RAM cache");
        // Loading experts 1 and 2 evicts cached expert 0 after the source batch was planned.
        std::int64_t selected[]{1,2,0,0};
        auto ids=tb::from_blob(selected,{1,4},tb::TensorOptions().dtype(tb::kInt64)).to(input.device());
        const auto expected=reference_gate->forward(input,ids).cpu().contiguous();
        const auto reads=metric(bounded,"ssd_expert_read_count");
        const auto actual=gate->forward(input,ids).cpu().contiguous();
        require(metric(bounded,"ssd_expert_read_count")==reads+3,
            "TQ source batch did not reload a late cache miss exactly once");
        require(actual.nbytes()==expected.nbytes() && !std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes()),
            "TQ source batch changed output after evicting a later requested expert");
        std::cout<<"TQ batched reads recover experts evicted after miss planning on device "<<device<<'\n';
    }
    for(int device=0;device<devices;++device) {
        MfqCudaGuard guard(device);Source source(1024,true);
        auto base=load_tq_weight(source,"linear",device);
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
        mfq::cuda::Graph memory;memory.prepare_memory();
        mfq::cuda::default_context(device)->end_graph_warmup(mfq_current_cuda_stream());
        mfq::cuda::BlasHandle blas(device);blas.set_stream(mfq_current_cuda_stream());
        auto scratch=tb::empty({8<<20},options.dtype(tb::kUInt8));
        mfq::cuda::test::check_common_prefill_projection(device);
        check_tq_prefill_scales(device);check_loaded_tq_groups(device);check_tq_ffn_prefill_residual(device);
        check_indexed_column_tails(device);
        int cases=0;
        for(int width:{33,513})for(bool columns:{false,true})for(int groups:{1,5})
        for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
            auto weight=base;weight.neuron_len=weight.view.input_columns=width;
            auto scales=tb::full({width},.25,options.dtype(tb::kFloat16));
            if(columns){weight.view.columns=1;weight.view.scales=scales.data_ptr();}
            constexpr int tokens=513;
            auto cpu=tb::empty({tokens,groups,width},tb::TensorOptions().dtype(tb::kFloat32));
            for(int64_t i=0;i<cpu.numel();++i)cpu.data_ptr<float>()[i]=
                std::sin(float(i%773)*.071f)*(dtype==tb::kFloat32 && i<groups*width?100000.f:.125f);
            auto input=cpu.to(tb::Device(tb::kCUDA,device)).to(dtype);
            if(groups==1)input=input.reshape({tokens,width});
            auto expected=tb::empty({tokens,weight.out},options.dtype(dtype));
            const auto reference=[&] {
                const int result=dtype==tb::kFloat16?mfq_tq_grouped_gemm(weight.view,input.data_ptr(),
                    expected.data_ptr(),tokens,groups,scratch.data_ptr(),scratch.nbytes(),blas.get(),mfq_current_cuda_stream()):
                    mfq_tq_float_gemm(weight.view,input.data_ptr(),expected.data_ptr(),tokens,groups,2,
                        scratch.data_ptr(),scratch.nbytes(),blas.get(),mfq_current_cuda_stream());
                require(result==0,"TQ prefill reference GEMM failed");return expected.to(tb::kFloat32).cpu().contiguous();
            };
            const auto check=[&](const tb::Tensor& result) {
                const auto target=reference(),actual=result.to(tb::kFloat32).cpu().contiguous();
                double error=0,norm=0;
                for(int64_t i=0;i<actual.numel();++i) {
                    const double a=actual.data_ptr<float>()[i],b=target.data_ptr<float>()[i];
                    require(std::isfinite(a),"scaled TQ prefill overflowed");error+=(a-b)*(a-b);norm+=b*b;
                }
                require(std::sqrt(error/std::max(norm,1e-30))<.004,
                    "TQ shared prefill GEMM exceeded the FP16 accumulation error bound");++cases;
            };
            check(groups==1?weight.forward(input):weight.forward_groupwise(input,groups));
            tb::Tensor output;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
            window.capture([&]{output=groups==1?weight.forward(input):weight.forward_groupwise(input,groups);},[]{});
            input.copy_(input*.75f);window.run();check(output);
        }
        std::cout<<"TQ shared prefill GEMM: "<<cases<<" scaled/grouped/changing-graph accuracy cases passed\n";
        CudaExecutionContext execution;
        auto linear=load_quant_linear(execution,source,"linear");
        linear.tq.neuron_len=linear.tq.view.input_columns=1024;
        linear.logical_neuron_len=1024;
        QuantLinearGroup group;group.layers={linear,linear};group.outs={95,95};
        if(mfq::cuda::gguf::rms_prefill_available(513,1024,device))for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
            auto input=(tb::arange(513*1024,options.dtype(tb::kFloat32)).remainder(8191)*.00013-.2)
                .reshape({513,1024}).to(dtype);
            auto norm=tb::full({1024},1.01,options.dtype(tb::kFloat32));tb::Tensor normalized;
            auto scope=group.prepare_prefill_rms_norm_input(execution,input,norm,1e-6,0,normalized);
            require(bool(scope) && normalized.defined(),"TQ group excluded the shared prefill RMS preparation");
            tb::Tensor packed,exponents;
            require(GgufPrefillInputScope::lookup(normalized,packed,exponents),"TQ group lost its shared scaled input");
            const auto before=packed.data_ptr();const auto projected=group.forward(execution,normalized);
            require(projected.size()==2 && projected[0].size(0)==513 && projected[0].size(1)==95,
                "TQ shared RMS projection changed output dimensions");
            require(GgufPrefillInputScope::lookup(normalized,packed,exponents) && packed.data_ptr()==before,
                "TQ group replaced the prepared RMS activation packet");
            const auto source_cpu=input.to(tb::kFloat32).cpu().contiguous(),result=normalized.cpu().contiguous();
            for(int row:{0,137,512}) {
                double squared=0;for(int column=0;column<1024;++column) {
                    const double value=source_cpu.data_ptr<float>()[row*1024+column];squared+=value*value;
                }
                for(int column:{0,513,1023})require(std::abs(result.data_ptr<float>()[row*1024+column]-
                    source_cpu.data_ptr<float>()[row*1024+column]*1.01/std::sqrt(squared/1024+1e-6))<1e-5,
                    "TQ shared RMS normalization changed the reference values");
            }
        }
    }
    for(int device=0;device<devices;++device) {
        MfqCudaGuard guard(device);
        Source source(512,true);
        auto weight=load_tq_weight(source,"linear",device);
        weight.neuron_len=weight.view.input_columns=320;
        const auto options=weight.image.options().dtype(tb::kFloat32);
        const auto input=(tb::arange(513*320,options)*.00013f-.2f).reshape({513,320});
        const auto plain=weight.forward(input);
        const auto product=prefill_product(prefill_projection_weight(weight),input);
        require(product.defined(),"TQ excluded the shared scaled prefill product");
        const auto actual=prefill_materialize(product).cpu(),expected=plain.cpu();
        require(actual.sizes()==expected.sizes() && !std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes()),
            "deferred TQ restoration changed the ordinary projection");
        const auto attended=input.reshape({1,513,5,64}).permute({0,2,1,3}).contiguous();
        const auto gate=(input*.01f).to(tb::kFloat16);
        const auto normal=tb::ones({64},options)*.75f;
        const PrefillProduct scaled_gate{gate,tb::zeros({513},options.dtype(tb::kInt32)),
            tb::zeros({320},options.dtype(tb::kInt32))};
        for(bool silu:{false,true}) {
            const auto direct=prefill_rms_gated_product(prefill_projection_weight(weight),attended,
                scaled_gate,normal,1e-5,silu);
            const auto restored=prefill_rms_gated_product(prefill_projection_weight(weight),attended,
                gate.to(tb::kFloat32).reshape({1,513,320}),normal,1e-5,silu);
            if(!mfq::cuda::gguf::rms_gated_prefill_available(320,device)) {
                require(!direct.defined() && !restored.defined(),"unsupported shared RMS projection did not decline");continue;
            }
            require(direct.defined() && restored.defined(),"TQ excluded shared RMS/gate/deferred restoration");
            for(auto pair:{std::pair{direct.values,restored.values},
                    std::pair{direct.row_exponents,restored.row_exponents},
                    std::pair{direct.column_exponents,restored.column_exponents}}) {
                const auto a=pair.first.cpu(),b=pair.second.cpu();
                require(!std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()),"TQ scaled gate changed its projection or exponents");
            }
        }
    }
    for(int device=0;device<devices;++device)for(int tokens:{33,129,257}) {
        MfqCudaGuard guard(device);Source source(512,true);CudaExecutionContext execution;
        execution.config.moe_device_cache_bytes=1u<<20;
        TqMoeBanks banks{load_tq_routed_weight(execution,source,"gate"),load_tq_routed_weight(execution,source,"up"),
            load_tq_routed_weight(execution,source,"down")};
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kFloat16);
        auto input=(tb::arange(tokens*33,options.dtype(tb::kFloat32))*.00013f-.1f).reshape({tokens,33}).to(tb::kFloat16);
        std::vector<int32_t> routes(tokens*2);
        for(int token=0;token<tokens;++token) {
            routes[2*token]=token<tokens*2/3?0:1+token%4;
            routes[2*token+1]=1+(token+1)%4;
        }
        auto ids=tb::tensor(routes).reshape({tokens,2}).to(input.device());
        auto scores=tb::ones({tokens,2},options.dtype(tb::kFloat32))*.5f;
        auto backing=tb::empty({4<<20},options.dtype(tb::kUInt8));
        const auto context=mfq::cuda::default_context(device);
        context->begin_workspace_loan({{backing.data_ptr(),backing.nbytes()}},mfq_current_cuda_stream());
        const auto actual=banks.forward(input,ids,scores).to(tb::kFloat32).cpu();
        require(context->workspace_live_bytes()==0,
            "routed prefill retained persistent plans in its temporary workspace");
        context->end_workspace_loan();backing.fill_(0);
        const auto gate=banks.gate->forward(input,ids),up=banks.up->forward(input,ids);
        const auto hidden=(tb::silu(gate.to(tb::kFloat32))*up.to(tb::kFloat32)).to(tb::kFloat16);
        const auto pairs=banks.down->forward(hidden,ids);
        const auto expected=(pairs.to(tb::kFloat32).sum(1)*.5f).to(tb::kFloat16).to(tb::kFloat32).cpu();
        double error=0,energy=0;
        for(int64_t i=0;i<actual.numel();++i) {
            const double a=actual.data_ptr<float>()[i],b=expected.data_ptr<float>()[i];
            require(std::isfinite(a),"staged TQ prefill produced a nonfinite output");error+=(a-b)*(a-b);energy+=b*b;
        }
        require(std::sqrt(error/std::max(energy,1e-30))<.004,"staged TQ prefill disagrees with independent routed projections");
    }
    for(int device=0;device<devices;++device) {
        MfqCudaGuard guard(device);
        mfq::cuda::KvMemoryOptions config;config.placement="vram";
        mfq::cuda::KvMemoryManager manager(config);
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kFloat16);
        auto initial=manager.add(64,2,32,48,16,4,options,32);manager.finalize();
        std::size_t reservation=0,calls=0;
        manager.set_reservation([&](std::size_t bytes){reservation=bytes;++calls;});
        auto extra=manager.add(64,2,32,48,16,4,options,32);
        require(calls==1 && reservation==initial->gpu_bytes()+(std::size_t(256)<<20),
            "additional KV session did not reserve its GPU footprint before placement");
        require(extra->ready && extra->gpu_bytes()==initial->gpu_bytes(),"additional KV session changed its placement");
        manager.set_reservation([](std::size_t){throw std::runtime_error("reject session reservation");});
        bool rejected=false;
        try{manager.add(64,2,32,48,16,4,options,32);}catch(const std::runtime_error&){rejected=true;}
        require(rejected && initial->ready && extra->ready,"failed KV admission damaged live sessions");
    }
    for(int device=0;device<devices;++device) {
        MfqCudaGuard guard(device);
        RoutedPrefillStager stager;stager.kRing=4;
        require(stager.init(128,2),"routed staging ring allocation failed");
        const std::array<std::size_t,7> counts{129,128,63,62,31,7,1};
        const auto ends=mfq::cuda::routed_prefill_group_ends(counts.size(),4,[&](auto i){return counts[i];});
        require(ends==std::vector<std::size_t>({2,4,5,6,7}),"routed staging group boundaries changed");
        require(mfq::cuda::routed_prefill_group_capacity(100,8ULL<<20,32ULL<<20)==8 &&
            mfq::cuda::routed_prefill_group_capacity(3,128ULL<<20,512ULL<<20)==1,
            "routed formats disagree on their bounded PP image capacity");
        const auto stream=mfq_get_current_cuda_stream(device);
        auto output=tb::zeros({21,128},tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
        for(int generation=0;generation<2;++generation) {
            std::vector<RoutedPrefillStager::Job> jobs;
            std::size_t begin=0;
            for(auto end:ends) {
                for(int role=0;role<3;++role)for(auto expert=begin;expert<end;++expert) {
                    auto bytes=std::make_shared<std::vector<std::uint8_t>>(37+expert*3+role,
                        std::uint8_t(1+generation*51+expert*3+role));
                    jobs.push_back({bytes->data(),bytes->size(),bytes});
                }
                begin=end;
            }
            stager.start(std::move(jobs));
            for(int j=0;j<21;++j) {
                if(j%5==0){stager.skip(j);stager.pass(j+1);continue;}
                stager.pass(j);
                const auto* source=stager.wait(j);
                TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr<std::uint8_t>()+j*128,source,
                    stager.jobs[j].bytes,cudaMemcpyHostToDevice,stream.stream()));
                stager.issued_one(j,stream.stream());
            }
            bool rejected=false;
            try{stager.wait(0);}catch(const std::logic_error&){rejected=true;}
            require(rejected,"routed staging accepted an overwritten earlier job");
            const auto actual=output.cpu();stager.finish();
            for(int j=0;j<21;++j)if(j%5)require(std::memcmp(actual.data_ptr<std::uint8_t>()+j*128,
                stager.jobs[j].src,stager.jobs[j].bytes)==0,"routed staging wraparound changed expert bytes");
        }
        std::cout<<"Routed host staging: uneven groups, ring wraparound, skipped jobs and generation reuse passed\n";
    }
    for(int device=0;device<devices;++device)for(const bool phased:{false,true})for(const int fail:{-1,3}) {
        MfqCudaGuard guard(device);
        const auto compute=mfq_get_current_cuda_stream(device),upload=mfq::cuda::stream_from_pool(false,device);
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8);
        auto stages=tb::empty({2,128},options),output=tb::zeros({7,128},options);
        auto host=tb::empty({7,128},options.device(tb::kCPU).pinned_memory(true));
        for(int group=0;group<7;++group)std::memset(host.data_ptr<uint8_t>()+group*128,group+1,128);
        std::vector<std::size_t> submitted;
        mfq::cuda::RoutedPrefillPipeline pipeline(compute.stream(),upload.stream());
        const auto acquire=[&](std::size_t group) {
            require(submitted.size()==group,"expert acquisition blocked an earlier ready compute submission");
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(stages.data_ptr<uint8_t>()+(group%2)*128,
                host.data_ptr<uint8_t>()+group*128,phased?64:128,cudaMemcpyHostToDevice,upload.stream()));
            pipeline.stage(group).record_gate_up(upload.stream());
        };
        const auto consume=[&](std::size_t group) {
            pipeline.join(group,true);
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr<uint8_t>()+group*128,
                stages.data_ptr<uint8_t>()+(group%2)*128,64,cudaMemcpyDeviceToDevice,compute.stream()));
            if(int(group)==fail)throw std::runtime_error("injected phased prefill failure");
            if(phased) {
                TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(stages.data_ptr<uint8_t>()+(group%2)*128+64,
                    host.data_ptr<uint8_t>()+group*128+64,64,cudaMemcpyHostToDevice,upload.stream()));
                pipeline.stage(group).record_ready(upload.stream());
            }
            pipeline.join(group);
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr<uint8_t>()+group*128+64,
                stages.data_ptr<uint8_t>()+(group%2)*128+64,64,cudaMemcpyDeviceToDevice,compute.stream()));
            submitted.push_back(group);
        };
        const auto release=[&](std::size_t group) {
            require(submitted.size()==group+1,"expert stage released before its compute submission");
        };
        bool failed=false;
        try {
            if(phased)pipeline.run_phased(7,acquire,consume,release);
            else pipeline.run(7,acquire,consume,release);
        }catch(const std::runtime_error& error) {
            if(std::string(error.what())!="injected phased prefill failure")throw;
            failed=true;
        }
        require(failed==(fail>=0) && submitted.size()==std::size_t(fail<0?7:fail),
            "expert pipeline continued after an incomplete compute submission");
        const auto actual=output.cpu();
        const auto valid_bytes=fail<0?host.nbytes():std::size_t(fail)*128+64;
        require(!std::memcmp(actual.data_ptr(),host.data_ptr(),valid_bytes),"expert pipeline overwrote an in-flight image");
        std::cout<<"Routed prefill pipeline: ready compute precedes host lookahead and fenced arena reuse passed\n";
    }
    for(int device=0;device<devices;++device) {
        MfqCudaGuard guard(device);
        CudaIndependentBranchExecutor branches;
        std::vector<tb::Tensor> outputs;
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kFloat32);
        auto input=tb::ones({17},options);
        bool rejected=false;
        try {
            branches.run(3,[&](std::size_t index) {
                auto result=input+double(index);
                if(index==2)throw std::runtime_error("cancel independent branch");
                return result;
            },outputs);
        }catch(const std::runtime_error&){rejected=true;}
        require(rejected,"branch exception was swallowed");
        require(branches.run(2,[&](std::size_t index){return input+double(index);},outputs),
            "branch executor could not resume after cancellation");
        for(std::size_t i=0;i<outputs.size();++i) {
            const auto actual=outputs[i].cpu().contiguous();
            require(actual.data_ptr<float>()[0]==1.f+float(i),"branch cancellation broke parent dependencies");
        }
        tb::Tensor shared_value;
        rejected=false;
        try {
            try_independent_branch(&branches,[&]{shared_value=input+4.;},[]() -> tb::Tensor {throw 7;},1);
        }catch(int value){rejected=value==7;}
        require(rejected && shared_value.cpu().data_ptr<float>()[0]==5.f,
            "nonstandard branch cancellation lost its parent dependency");
        const auto parent=mfq_get_current_cuda_stream(device);
        if(devices>1) {
            MfqCudaGuard other((device+1)%devices);
            auto separate=std::make_unique<CudaIndependentBranchExecutor>();
            require(separate->ensure(2,parent),"branch events could not be created on their owning device");
            separate.reset();
            require(mfq_current_cuda_device()==(device+1)%devices,"branch destruction changed current device");
        }
    }
    for(int device=0;device<devices;++device) {
        MfqCudaGuard guard(device);Source source;CudaExecutionContext execution;
        check_host_image_relocation();
        check_host_image_range_trim();
        check_routed_graph_pool_ownership(device);
        check_routed_event_retirement(device);
        check_routed_storage_fence(device);
        QuantLinear projection;projection.kind=QuantLinearKind::Tq;
        projection.tq=load_tq_weight(source,"linear",device,true);
        projection.logical_out=95;projection.logical_neuron_len=33;
        const auto context=mfq::cuda::default_context(device);
        const auto stream=mfq_current_cuda_stream();
        check_routed_workspace_backing(device);
        check_causal_position_workspace(device);
        const bool own_pool=!context->has_graph_pool(stream);
        if(own_pool)context->begin_graph_pool(stream);
        struct PoolLease {
            std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;bool owned;
            ~PoolLease(){if(owned)context->end_graph_pool(stream);}
        } pool{context,stream,own_pool};
        check_pending_tq_dma_retirement(device);
        check_tq_host_evacuation(device);
        check_compact_tq_moe(device);
        check_tq_predictor_residency(device);
        check_mapped_embeddings(source,execution,device);
        check_selected_projection_rows(source,execution,device);
        check_tq_residuals(device);
        {
            Source large(65536,true);
            CudaExecutionContext pageable;
            pageable.config.moe_device_cache_bytes=0;
            pageable.config.moe_host_cache_bytes=64u<<20;
            auto bank=load_tq_routed_weight(pageable,large,"gate");
            const mfq::cuda::HostRegistrationBlock no_new_pins;
            auto input=tb::ones({5,33},tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kFloat16))*.1f;
            auto ids=tb::tensor(std::vector<int64_t>{0,1,2,3,4}).reshape({5,1}).to(tb::Device(tb::kCUDA,device));
            auto expected=bank->forward(input,ids).cpu().contiguous();
            for(int pass=0;pass<24;++pass) {
                const auto actual=bank->forward(input,ids).cpu().contiguous();
                require(!std::memcmp(expected.data_ptr(),actual.data_ptr(),expected.nbytes()),
                    "pageable DMA ring reuse changed expert bytes");
            }
            require(metric(pageable,"expert_batched_upload_packets")>=25,
                "pageable experts bypassed grouped DMA submission");
            require(metric(pageable,"ram_expert_pinned_reserved_bytes")==0,
                "blocked expert registration allocated another pinned arena");
            require(metric(pageable,"ram_expert_pcie_read_bytes")>(64u<<20),
                "pageable DMA fixture did not wrap both ring buffers");
            std::cout<<"Pageable grouped DMA wraps both staging buffers with exact outputs passed\n";
        }
        {
            CudaExecutionContext pinned;
            pinned.config.moe_device_cache_bytes=0;
            pinned.config.moe_host_cache_bytes=1u<<20;
            const auto make_banks=[&] {return TqMoeBanks{load_tq_routed_weight(pinned,source,"gate"),
                load_tq_routed_weight(pinned,source,"up"),load_tq_routed_weight(pinned,source,"down")};};
            auto first=make_banks();
            const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kFloat16);
            auto input=tb::ones({1,33},options)*.1f;
            auto ids=tb::tensor(std::vector<int64_t>{0,1,2,3,4}).reshape({1,5}).to(tb::Device(tb::kCUDA,device));
            auto scores=tb::ones({1,5},options)/5.f;
            (void)first.forward(input,ids,scores);
            TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
            auto second=make_banks();
            const mfq::cuda::HostRegistrationBlock no_new_pins;
            auto expected=second.forward(input,ids,scores).cpu().contiguous();
            TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
            auto last=ids.narrow(1,4,1).contiguous();
            auto hidden=tb::ones({1,19},options)*.1f;
            const auto pinned_before=metric(pinned,"ram_expert_pinned_reserved_bytes");
            second.down->forward(hidden,last).cpu();
            require(metric(pinned,"ram_expert_pinned_reserved_bytes")==pinned_before,
                "cached expert reuse registered another pinned arena");
            resize_tq_expert_cache(pinned.tq_expert_cache,256u<<10);
            (void)first.forward(input,ids,scores);
            TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
            const auto reserved=metric(pinned,"ram_expert_pinned_reserved_bytes");
            tb::Tensor output;
            mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
            window.capture([&]{output=second.forward(input,ids,scores);},[]{});
            resize_tq_expert_cache(pinned.tq_expert_cache,0);
            resize_tq_expert_cache(pinned.tq_expert_cache,256u<<10);
            first.forward(input,ids,scores).cpu();
            const auto recycled_before=metric(pinned,"ram_expert_unpinned_copy_bytes");
            window.run();
            auto actual=output.cpu().contiguous();
            require(!std::memcmp(expected.data_ptr(),actual.data_ptr(),expected.nbytes()),
                "reusing pinned GPU backups changed expert output");
            require(metric(pinned,"ram_expert_unpinned_copy_bytes")==recycled_before,
                    "cached expert reuse unnecessarily moved a pinned GPU backup");
            require(metric(pinned,"ram_expert_pinned_reserved_bytes")==reserved,
                    "cached expert reuse allocated another pinned arena");
            std::cout<<"Cached experts reuse pinned intervals and preserve exact graph outputs passed\n";
        }
        for(const auto dtype:{tb::kFloat16,tb::kFloat32})for(const int rows:{0,1,4,33})for(const int mode:{1,2}) {
            auto host=tb::empty({rows,33},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
            for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=std::sin(float(i)*.19f)*4.f;
            auto input=host.to(tb::Device(tb::kCUDA,device)).to(dtype);
            auto gate=(input*.7f).contiguous();
            const auto check=[&](const tb::Tensor& actual) {
                auto activation=mode==1?tb::sigmoid(gate):tb::silu(gate);
                auto expected=projection.forward(execution,input*activation).to(tb::kFloat32).cpu();
                auto result=actual.to(tb::kFloat32).cpu();
                require(expected.sizes()==result.sizes(),"common gated TQ projection changed shape");
                for(int64_t i=0;i<result.numel();++i)require(
                    std::abs(expected.data_ptr<float>()[i]-result.data_ptr<float>()[i])<=
                        2e-6f*std::max(1.f,std::abs(expected.data_ptr<float>()[i])),
                    "common gated TQ projection changed activation rounding");
            };
            check(projection.forward_input_mul(execution,input,gate,mode));
            if(rows==4) {
                tb::Tensor output;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
                window.capture([&]{output=projection.forward_input_mul(execution,input,gate,mode);},[]{});
                for(int replay=0;replay<3;++replay) {
                    gate.copy_(gate*.5f);window.run();check(output);
                }
            }
        }
        std::cout<<"Common gated TQ projections, FP16/FP32, empty/tail rows and changing graphs passed\n";
    }
    for(int device=0;device<devices;++device) {
        MfqCudaGuard guard(device);
        const auto stream=mfq_get_current_cuda_stream(device);
        mfq::cuda::HostBuffer host(512,true);void* alias=nullptr;
        TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&alias,host.data(),0));
        auto target=tb::empty({256},tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
        for(int mode:{0,2})for(int count:{0,3,129}) {
            std::memset(host.data(),0,512);std::vector<mfq::cuda::DmaCopy> copies;
            for(int i=0;i<count;++i) {
                static_cast<std::uint8_t*>(host.data())[i]=std::uint8_t(i+7);
                copies.push_back({target.data_ptr<std::uint8_t>()+i,static_cast<std::uint8_t*>(host.data())+i,1});
            }
            mfq::cuda::submit_routed_transfers(copies,stream.stream(),mode,[&](const void* source,std::size_t) {
                return reinterpret_cast<CUdeviceptr>(alias)+
                    (static_cast<const std::uint8_t*>(source)-static_cast<const std::uint8_t*>(host.data()));
            });
            auto* ready=reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host.data())+256);
            auto* mapped=reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(alias)+256);
            mfq::cuda::enqueue_routed_completion(ready,mapped,stream.stream(),mode!=0);
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            require(*ready==1,"routed transfer completion was not published");
            const auto value=target.cpu();
            for(int i=0;i<count;++i)require(value.data_ptr<std::uint8_t>()[i]==std::uint8_t(i+7),
                "routed transfer batch changed bytes");
        }
        {
            mfq::cuda::DmaCopyBatch fallback(stream.stream());
            fallback.prepare(3,target.data_ptr(),host.data());
            int calls=0;
            for(int count:{1,3}) {
                std::vector<mfq::cuda::DmaCopy> copies;
                for(int i=0;i<count;++i) {
                    static_cast<std::uint8_t*>(host.data())[i]=std::uint8_t(137+i);
                    copies.push_back({target.data_ptr<std::uint8_t>()+i,static_cast<std::uint8_t*>(host.data())+i,1});
                }
                const bool native=mfq::cuda::submit_routed_transfers(copies,stream.stream(),0,
                    [](const void*,std::size_t){return CUdeviceptr(0);},[&] {
                        ++calls;require(fallback.submit(copies),"shared transfer lost the prepared graph fallback");
                    });
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
                require(!native,"disabled native transfer skipped its configured graph fallback");
                const auto result=target.cpu();
                for(int i=0;i<count;++i)require(result.data_ptr<std::uint8_t>()[i]==std::uint8_t(137+i),
                    "shared prepared-graph fallback changed packet bytes");
            }
            require(calls==2,"shared transfer submitted its graph fallback more than once");
            for(int mode:{0,2}) {
                target.zero_();int aliases=0,fallbacks=0;bool rejected=false;
                const std::vector<mfq::cuda::DmaCopy> invalid{{target.data_ptr(),host.data(),1},
                    {target.data_ptr<std::uint8_t>()+1,nullptr,1}};
                try {
                    mfq::cuda::submit_routed_transfers(invalid,stream.stream(),mode,[&](const void*,std::size_t) {
                        ++aliases;return reinterpret_cast<CUdeviceptr>(alias);
                    },[&]{++fallbacks;});
                }catch(const std::invalid_argument&){rejected=true;}
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
                require(rejected && !aliases && !fallbacks && target.cpu().data_ptr<std::uint8_t>()[0]==0,
                    "invalid routed packet partially submitted before complete validation");
            }
        }
        for(int mode:{0,1,2})for(bool missing_alias:{false,true})for(int count:{0,1,3,129}) {
            std::memset(host.data(),0,512);
            std::vector<mfq::cuda::DmaCopy> upload,readback;
            for(int i=0;i<count;++i) {
                static_cast<std::uint8_t*>(host.data())[i]=std::uint8_t(91+i);
                upload.push_back({target.data_ptr<std::uint8_t>()+i,static_cast<std::uint8_t*>(host.data())+i,1});
                readback.push_back({static_cast<std::uint8_t*>(host.data())+256+i,target.data_ptr<std::uint8_t>()+i,1});
            }
            const auto mapped_host=[&](const void* pointer,std::size_t) -> const void* {
                const auto offset=static_cast<const std::uint8_t*>(pointer)-static_cast<const std::uint8_t*>(host.data());
                if(missing_alias && offset%256==count-1)return nullptr;
                return static_cast<const std::uint8_t*>(alias)+offset;
            };
            const bool sent=mfq::cuda::submit_routed_host_transfers(upload,cudaMemcpyHostToDevice,
                stream.stream(),mode,mapped_host);
            const bool received=mfq::cuda::submit_routed_host_transfers(readback,cudaMemcpyDeviceToHost,
                stream.stream(),mode,mapped_host);
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            require((!missing_alias || (!sent && !received)) && (mode || (!sent && !received)),
                "routed host transfer used an incomplete or disabled native alias packet");
            require(std::memcmp(host.data(),static_cast<const std::uint8_t*>(host.data())+256,count)==0,
                "routed upload/readback changed bytes through a native or ordinary address");
        }
        {
            int aliases=0;bool rejected=false;
            try {
                const std::vector<mfq::cuda::DmaCopy> invalid{{target.data_ptr(),host.data(),1},
                    {target.data_ptr<std::uint8_t>()+1,nullptr,1}};
                mfq::cuda::submit_routed_host_transfers(invalid,cudaMemcpyHostToDevice,stream.stream(),2,
                    [&](const void*,std::size_t) -> const void* {++aliases;return alias;});
            }catch(const std::invalid_argument&){rejected=true;}
            require(rejected && !aliases,"routed host transfer mapped an invalid packet before complete validation");
        }
        std::exception_ptr failure;
        std::thread worker([&] {
            try {
                const MfqCudaGuard device_guard(device);
                static_cast<std::uint8_t*>(host.data())[0]=193;
                const std::vector<mfq::cuda::DmaCopy> copies{{target.data_ptr(),host.data(),1}};
                mfq::cuda::submit_routed_transfers(copies,stream.stream(),2,[&](const void*,std::size_t) {
                    return reinterpret_cast<CUdeviceptr>(alias);
                });
                auto* ready=reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host.data())+256);
                auto* mapped=reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(alias)+256);
                *ready=0;mfq::cuda::enqueue_routed_completion(ready,mapped,stream.stream());
            }catch(...){failure=std::current_exception();}
        });
        worker.join();if(failure)std::rethrow_exception(failure);
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        require(target.cpu().data_ptr<std::uint8_t>()[0]==193 &&
                *reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host.data())+256)==1,
            "routed driver operations failed on a fresh device worker");
        const auto driver_entry=[](const char* name) {
            void* function=nullptr;cudaDriverEntryPointQueryResult query=cudaDriverEntryPointSymbolNotFound;
            TYLOINFER_CUDA_CHECK(cudaGetDriverEntryPoint(name,&function,cudaEnableDefault,&query));
            require(function && query==cudaDriverEntryPointSuccess,"missing context test entry");
            return function;
        };
        const auto current=reinterpret_cast<decltype(&cuCtxGetCurrent)>(driver_entry("cuCtxGetCurrent"));
        const auto bind=reinterpret_cast<decltype(&cuCtxSetCurrent)>(driver_entry("cuCtxSetCurrent"));
        bool native_signal=false;
        std::thread detached([&] {
            try {
                const MfqCudaGuard device_guard(device);
                CUcontext previous=nullptr,after=nullptr;
                require(current(&previous)==CUDA_SUCCESS && bind(nullptr)==CUDA_SUCCESS,
                    "could not detach the test worker context");
                auto* ready=reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host.data())+256);
                auto* mapped=reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(alias)+256);
                *ready=0;
                native_signal=mfq::cuda::NativeStreamSignal::submit(mapped,stream.stream());
                require(current(&after)==CUDA_SUCCESS && after==nullptr,
                    "native stream completion did not restore a detached worker context");
                require(bind(previous)==CUDA_SUCCESS,"could not restore the test worker context");
                if(!native_signal)mfq::cuda::enqueue_routed_completion(ready,mapped,stream.stream(),false);
            }catch(...){failure=std::current_exception();}
        });
        detached.join();if(failure)std::rethrow_exception(failure);
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        require(*reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host.data())+256)==1,
            "detached worker completion was not published");
        std::cout<<"Routed completion preserves detached worker context; native="<<native_signal<<'\n';
    }
    for(int device=0;device<devices;++device)for(const bool gguf:{false,true}) {
        MfqCudaGuard guard(device);Source source;CudaExecutionContext execution;
        execution.config.moe_device_cache_bytes=1u<<20;
        const int width=gguf?32:33;
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kFloat16);
        FFN ffn;ffn.is_moe=true;ffn.moe_top_k=2;ffn.moe_shared_ungated=true;
        ffn.moe_router=tb::zeros({5,width},options.dtype(tb::kFloat32));
        if(gguf)ffn.moe_gguf=std::make_shared<GgufMoeBanks>(GgufMoeBanks{
            std::make_shared<GgufRoutedWeight>(execution,source,"gguf.gate"),
            std::make_shared<GgufRoutedWeight>(execution,source,"gguf.up"),
            std::make_shared<GgufRoutedWeight>(execution,source,"gguf.down")});
        else ffn.moe_tq=std::make_shared<TqMoeBanks>(TqMoeBanks{
            load_tq_routed_weight(execution,source,"gate"),load_tq_routed_weight(execution,source,"up"),
            load_tq_routed_weight(execution,source,"down")});
        auto dense=[&](int out,int in) {
            QuantLinear value;value.kind=QuantLinearKind::Dense;value.logical_out=out;value.logical_neuron_len=in;
            value.dense=tb::full({out,in},.003,options);return value;
        };
        ffn.shared=std::make_unique<FFN>();
        ffn.shared->gate_up.layers={dense(16,width),dense(16,width)};
        ffn.shared->gate_up.outs={16,16};ffn.shared->down=dense(width,16);
        execution.config.decode_branch_parallel=true;prepare_ffn_workspaces(execution,ffn);
        for(const int rows:{1,4,9,33}) {
            auto input=tb::ones({1,rows,width},options)*.1;
            execution.config.decode_branch_parallel=false;
            const auto expected=ffn.forward(execution,input).to(tb::kFloat32).cpu().contiguous();
            execution.config.decode_branch_parallel=true;
            const auto actual=ffn.forward(execution,input).to(tb::kFloat32).cpu().contiguous();
            require(actual.sizes()==expected.sizes(),"generic routed FFN changed output dimensions");
            for(int64_t i=0;i<actual.numel();++i)
                require(std::isfinite(actual.data_ptr<float>()[i]) &&
                    std::abs(actual.data_ptr<float>()[i]-expected.data_ptr<float>()[i])<=.002f*(1+std::abs(expected.data_ptr<float>()[i])),
                    "generic routed FFN shared overlap changed values");
        }
        const auto budget=mfq::cuda::routed_expert_cache_budget_bytes(execution);
        bool cancelled=false;
        try {
            mfq::cuda::vision_ops::vision_workspace(execution,SIZE_MAX/2,[&]() -> tb::Tensor {
                require(gguf || mfq::cuda::routed_expert_cache_budget_bytes(execution)==0,
                    "vision workspace omitted a native expert cache");
                throw std::runtime_error("cancel vision workspace");
            });
        }catch(const std::runtime_error& error){cancelled=std::string(error.what())=="cancel vision workspace";}
        require(cancelled && mfq::cuda::routed_expert_cache_budget_bytes(execution)==budget,
            "vision cancellation failed to restore the expert cache");
        std::cout<<"Generic "<<(gguf?"GGUF":"TQ")<<" routed FFN and vision workspace cancellation passed on device "<<device<<'\n';
    }
    for(int device=0;device<devices;++device)for(const auto stored_width:{256u,1024u})for(const int count:{4,64}) {
        MfqCudaGuard guard(device);Source source(stored_width,true);
        auto table=std::make_shared<TqRowTable>(source,"linear");
        require(source.mapped_requests==1,"sparse TQ rows did not select the mapped source reader");
        const std::vector<std::int64_t> shape{count};
        auto stage=make_tq_row_stage({table},95,shape,tb::Device(tb::kCUDA,device));
        std::vector<std::int64_t> ids(count);
        constexpr std::int64_t pattern[]{0,2,0,3};
        for(int i=0;i<count;++i)ids[i]=pattern[i%4];
        auto changed=ids;changed[0]=1;
        const auto reference=table->lookup(ids.data(),ids.size(),tb::Device(tb::kCUDA,device)).cpu().contiguous();
        const auto retained=std::make_shared<TqRowTable>(source,"gate")->lookup(
            ids.data(),ids.size(),tb::Device(tb::kCUDA,device)).cpu().contiguous();
        require(reference.nbytes()==retained.nbytes() && std::memcmp(reference.data_ptr(),retained.data_ptr(),reference.nbytes())==0,
            "mapped and retained TQ row readers decoded different values");
        require(stage->prefetch(ids,shape),"TQ PLE stage rejected asynchronous row preparation");
        stage->upload(ids,shape);stage->decode();
        auto actual=stage->output().cpu().contiguous();
        require(actual.nbytes()==reference.nbytes() &&
            std::memcmp(actual.data_ptr(),reference.data_ptr(),actual.nbytes())==0,"TQ direct row stage changed decoded values");
        source.fail_tensor_reads("linear");
        stage->upload(ids,shape);stage->decode();actual=stage->output().cpu().contiguous();
        require(std::memcmp(actual.data_ptr(),reference.data_ptr(),actual.nbytes())==0,
            "TQ row cache failed to reuse encoded rows without source reads");
        bool rejected=false;
        try {stage->prefetch(changed,shape);stage->upload(changed,shape);}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"TQ row cache masked a failed uncached read");
        source.fail_reads_from(UINT64_MAX);
        stage->prefetch(changed,shape);stage->upload(changed,shape);stage->decode();
        actual=stage->output().cpu().contiguous();
        const auto expected=table->lookup(changed.data(),changed.size(),tb::Device(tb::kCUDA,device)).cpu().contiguous();
        require(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0,
            "TQ row staging retry retained a partial image");
        stage->prefetch(changed,shape);stage->prefetch(ids,shape);
        stage->upload(ids,shape);stage->decode();
        (void)stage->output().cpu();
        mfq::cuda::HostBuffer publication(128,true);std::memset(publication.data(),0,128);
        void* mapped=nullptr;TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,publication.data(),0));
        const auto side=mfq_get_stream_from_pool(false,device);
        {
            MfqCudaStreamGuard side_guard(side);
            mfq::cuda::wait_mapped_flag(static_cast<std::uint32_t*>(mapped),side.stream());
            stage->decode();mfq::cuda::check_stream_progress(side.stream());
        }
        stage->prefetch(changed,shape);
        std::jthread release([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            mfq::cuda::publish_mapped_flag(static_cast<std::uint32_t*>(publication.data()));
        });
        stage->upload(changed,shape);
        {
            MfqCudaStreamGuard side_guard(side);
            const auto previous=stage->output().cpu().contiguous();
            require(previous.nbytes()==reference.nbytes() &&
                std::memcmp(previous.data_ptr(),reference.data_ptr(),reference.nbytes())==0,
                "TQ row upload overwrote a packet still consumed by another stream");
        }
        stage->decode();actual=stage->output().cpu().contiguous();
        require(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0,
            "TQ row upload did not resume after its previous consumer");
        mfq_release_host_allocator_cache();source.fail_tensor_reads("linear");
        rejected=false;
        try {stage->upload(ids,shape);}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"RAM pressure did not release the optional TQ row cache");
        source.fail_reads_from(UINT64_MAX);
        stage->upload(ids,shape);stage->decode();actual=stage->output().cpu().contiguous();
        require(std::memcmp(actual.data_ptr(),reference.data_ptr(),actual.nbytes())==0,
            "TQ row reload changed values after optional cache reclamation");
    }
    {
        Source source;GgufRowTable table(source,"rows.F16");std::vector<std::byte> row(table.row_bytes());
        table.read_row(0,row.data());source.fail_tensor_reads("rows.F16");
        table.read_row(0,row.data());mfq_release_host_allocator_cache();
        bool rejected=false;
        try {table.read_row(0,row.data());}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"RAM pressure did not release the optional GGUF row cache");
    }
    for(int device=0;device<devices;++device) {
        MfqCudaGuard guard(device);Source source;CudaExecutionContext execution;
        execution.config.moe_device_cache_bytes=1u<<20;
        auto gate=load_tq_routed_weight(execution,source,"gate");
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
        auto input=tb::ones({1,33},options.dtype(tb::kFloat16))*.1;
        auto ids=tb::zeros({1,2},options.dtype(tb::kInt64));ids.select(1,1).fill_(1);
        source.fail_reads_from(64+95*8+48+19*256*3/8);
        bool rejected=false;try{gate->forward(input,ids);}catch(const std::runtime_error&){rejected=true;}
        require(rejected && metric(execution,"resident_expert_cache_bytes")==0,
            "partial source read left an unsubmitted GPU cache hit");
        source.fail_reads_from(UINT64_MAX);
        const auto actual=gate->forward(input,ids).to(tb::kFloat32).cpu().contiguous();
        const auto reference=gate->forward(input,tb::zeros({1,1},options.dtype(tb::kInt64))).to(tb::kFloat32).cpu().contiguous();
        for(int i=0;i<19;++i)require(actual.data_ptr<float>()[i]==reference.data_ptr<float>()[i],
            "retry consumed an incomplete expert transfer");
        mfq::cuda::finalize_routed_weight_images(execution,SIZE_MAX/2);
        require(tq_expert_cache_budget_bytes(execution.tq_expert_cache)==0,
            "TQ load finalization ignored required KV headroom");
        gate->forward(input,ids).cpu();
        require(metric(execution,"resident_expert_cache_bytes")==0,
            "TQ admitted experts beyond its finalized physical headroom");
        execution.kv_memory=std::make_shared<mfq::cuda::KvMemoryManager>(mfq::cuda::KvMemoryOptions{});
        std::weak_ptr<mfq::cuda::KvMemoryManager> manager=execution.kv_memory;
        execution.reset();require(manager.expired(),"execution reset retained the previous KV manager");
    }
    for(int device=0;device<devices;++device)for(const auto dtype:{tb::kFloat16,tb::kBFloat16,tb::kFloat32})
    for(const int rows:{1,2,3,4,5,6,7,8})for(int pattern=0;pattern<3;++pattern) {
        MfqCudaGuard guard(device);Source source(256,true);
        CudaExecutionContext execution,reference_execution;
        execution.config.moe_device_cache_bytes=32u<<10;
        reference_execution.config.moe_device_cache_bytes=0;
        const auto make=[&](CudaExecutionContext& owner) {
            return TqMoeBanks{load_tq_routed_weight(owner,source,"gate"),load_tq_routed_weight(owner,source,"up"),
                load_tq_routed_weight(owner,source,"down")};
        };
        auto banks=make(execution),reference_banks=make(reference_execution);
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
        auto input=tb::ones({rows,33},options.dtype(dtype))*.1;
        auto routes=tb::zeros({rows,3},options.dtype(tb::kInt32));
        auto scores=tb::ones({rows,3},options.dtype(tb::kFloat32))/3;
        require(prepare_tq_graph_pool(execution.tq_expert_cache),"TQ partition test has no graph pool");
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());tb::Tensor output;
        window.capture([&]{output=banks.forward(input,routes,scores);},[]{});
        std::vector<std::int32_t> selected=pattern==0?std::vector<std::int32_t>{0,0,0}:
            pattern==1?std::vector<std::int32_t>{0,2,0}:std::vector<std::int32_t>{2,3,4};
        const auto row=selected;
        for(int token=1;token<rows;++token)for(int route=0;route<3;++route)
            selected.push_back(row[(route+token)%3]);
        routes.copy_(tb::from_blob(selected.data(),{rows,3},tb::TensorOptions().dtype(tb::kInt32)).to(routes.device()));
        const auto reference=reference_banks.forward(input,routes,scores).to(tb::kFloat32).cpu().contiguous();
        if(pattern==1) {
            auto partial=tb::full({rows,1},2,options.dtype(tb::kInt32));
            banks.gate->forward(input,partial).cpu();source.fail_tensor_reads("gate");
        }else if(pattern==0)source.fail_reads_from(0);
        window.run();const auto actual=output.to(tb::kFloat32).cpu().contiguous();
        double error=0,norm=0;
        for(int i=0;i<actual.numel();++i) {
            const auto a=actual.data_ptr<float>()[i],b=reference.data_ptr<float>()[i];
            require(std::isfinite(a),"nonfinite resident/streamed TQ result");
            error+=double(a-b)*(a-b);norm+=double(b)*b;
        }
        require(std::sqrt(error/std::max(norm,1e-20))<.003,"TQ cache transition changed the projection");
        window.run();const auto cached=output.to(tb::kFloat32).cpu().contiguous();
        require(cached.nbytes()==actual.nbytes() && std::memcmp(cached.data_ptr(),actual.data_ptr(),actual.nbytes())==0,
            "TQ output bits changed when streamed operands became resident");
    }
    std::cout<<"TQ cache transitions preserve FP16/BF16/FP32 bits across full, partial and missing operands\n";
    for(int device=0;device<devices;++device)for(bool with_prefill:{true,false}) {
        MfqCudaGuard guard(device);Source source;CudaExecutionContext execution;
        execution.config.moe_device_cache_bytes=16u<<10;
        TqMoeBanks banks{load_tq_routed_weight(execution,source,"gate"),load_tq_routed_weight(execution,source,"up"),
            load_tq_routed_weight(execution,source,"down")};
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
        auto input=tb::ones({1,33},options.dtype(tb::kFloat16))*.1;
        auto ids=tb::zeros({1,1},options.dtype(tb::kInt32));
        auto scores=tb::ones({1,1},options.dtype(tb::kFloat32));
        if(with_prefill) {
            auto prompt_ids=tb::arange(33,options.dtype(tb::kInt32)).remainder(3).reshape({33,1});
            banks.forward(tb::ones({33,33},options.dtype(tb::kFloat16))*.1,prompt_ids,
                tb::ones({33,1},options.dtype(tb::kFloat32))).cpu();
        }
        require(prepare_tq_graph_pool(execution.tq_expert_cache),"route retention graph preparation failed");
        banks.forward(input,ids,scores).cpu();ids.fill_(2);banks.forward(input,ids,scores).cpu();ids.fill_(0);
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());tb::Tensor output;
        window.capture([&]{output=banks.forward(input,ids,scores);},[]{});
        for(int expert:{0,0,1,2}){ids.fill_(expert);window.run();output.cpu();}
        require(metric(execution,"expert_admission_retained_images")>0,"route retention test did not activate admission");
        mfq::cuda::RoutedCacheLearningScope fixed_selection(execution,false);
        for(int expert:{0,1,2,0,1,2,0,1,2}){ids.fill_(expert);window.run();output.cpu();}
        if(metric(execution,"expert_admission_resident_images")!=metric(execution,"expert_admission_retained_images")) {
            std::cerr<<"retention prefill="<<with_prefill;
            for(const auto* key:{"expert_admission_resident_images","expert_admission_retained_images",
                    "expert_cache_reserved_bytes","resident_expert_cache_bytes","expert_cache_evictions",
                    "expert_cache_budget_bytes","expert_cache_allocation_reuses","expert_decode_arena_recycles"})
                std::cerr<<' '<<key<<'='<<metric(execution,key);
            std::cerr<<'\n';
        }
        require(metric(execution,"expert_admission_resident_images")==
                metric(execution,"expert_admission_retained_images"),
            "TQ eviction discarded an admitted hot image while a cold image occupied its budget");
    }
    for(int device=0;device<devices;++device)for(const char* failed_bank:{"up","down"})for(int rows:{1,4,8}) {
        MfqCudaGuard guard(device);Source source;CudaExecutionContext execution;
        execution.config.moe_device_cache_bytes=1u<<20;
        TqMoeBanks banks{load_tq_routed_weight(execution,source,"gate"),load_tq_routed_weight(execution,source,"up"),
            load_tq_routed_weight(execution,source,"down")};
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
        auto input=tb::ones({rows,33},options.dtype(tb::kFloat16))*.1;
        auto ids=tb::zeros({rows,1},options.dtype(tb::kInt32));
        auto scores=tb::ones({rows,1},options.dtype(tb::kFloat32));tb::Tensor output;
        require(mfq::cuda::prepare_routed_graph_pool(execution),"batch failure test has no graph pool");
        auto window=std::make_unique<mfq::cuda::DecodeWindow>(mfq_current_cuda_stream());
        window->capture([&]{output=banks.forward(input,ids,scores);},[]{});
        const auto before=metric(execution,"resident_expert_cache_bytes");
        source.fail_tensor_reads(failed_bank);ids.fill_(4);
        bool rejected=false;try{window->run();}catch(const std::runtime_error&){rejected=true;}
        require(rejected && metric(execution,"resident_expert_cache_bytes")==before,
            "failed grouped upload retained an unsubmitted Gate projection");
        window.reset();TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());source.fail_reads_from(UINT64_MAX);
        const auto hits=metric(execution,"expert_cache_hits");
        banks.gate->forward(input,ids).cpu();
        require(metric(execution,"expert_cache_hits")==hits+rows-1,
            "retry's first route hit an unsubmitted grouped upload");
        const auto reference=banks.forward(input,ids,scores).to(tb::kFloat32).cpu().contiguous();
        window=std::make_unique<mfq::cuda::DecodeWindow>(mfq_current_cuda_stream());
        window->capture([&]{output=banks.forward(input,ids,scores);},[]{});window->run();
        const auto actual=output.to(tb::kFloat32).cpu().contiguous();
        for(int i=0;i<33;++i)require(actual.data_ptr<float>()[i]==reference.data_ptr<float>()[i],
            "grouped upload failure broke a subsequent graph");
    }
    if(devices>1) {
        Source source;CudaExecutionContext execution;execution.config.moe_device_cache_bytes=16u<<10;
        std::vector<std::shared_ptr<TqRoutedWeight>> banks;
        std::vector<tb::Tensor> inputs,ids;
        for(int device=0;device<devices;++device) {
            MfqCudaGuard guard(device);banks.push_back(load_tq_routed_weight(execution,source,"gate"));
            const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
            inputs.push_back(tb::ones({1,33},options.dtype(tb::kFloat16))*.1);
            ids.push_back(tb::zeros({1,1},options.dtype(tb::kInt64)));
            banks.back()->forward(inputs.back(),ids.back()).cpu();
        }
        MfqCudaGuard primary(0);
        mfq::cuda::RoutedPrefillWorkspaceLoan loan(execution);loan.begin(SIZE_MAX/2);
        {MfqCudaGuard secondary(1);
         require(tq_expert_cache_budget_bytes(execution.tq_expert_cache)==(16u<<10),
             "workspace loan changed a different device budget");
         const auto hits=metric(execution,"expert_cache_hits");banks[1]->forward(inputs[1],ids[1]).cpu();
         require(metric(execution,"expert_cache_hits")==hits+1,"workspace loan evicted another device's expert");
         loan.finish();require(mfq_current_cuda_device()==1,"loan restoration changed the caller's current device");}
    }
    for(int device=0;device<devices;++device) {
        MfqCudaGuard guard(device);Source source;CudaExecutionContext execution;
        execution.config.moe_device_cache_bytes=16u<<10;
        execution.config.moe_host_cache_bytes=128u<<10;
        auto tq=load_tq_routed_weight(execution,source,"gate");
        GgufRoutedWeight gguf(execution,source,"gguf.weight");
        mfq::cuda::finalize_routed_weight_images(execution,0);
        require(tq_expert_cache_budget_bytes(execution.tq_expert_cache)>0 &&
            gguf_expert_cache_budget_bytes(execution.gguf_expert_cache)>0 &&
            mfq::cuda::routed_expert_cache_budget_bytes(execution)==(16u<<10),
            "mixed encodings did not share the configured cache budget");
        const auto host_cap=[](const auto& values) {
            for(const auto& [key,value]:values)if(key=="ram_expert_cache_budget_bytes")return value;
            return 0.;
        };
        const auto tq_host_cap=host_cap(tq_expert_memory_metrics(execution.tq_expert_cache));
        const auto gguf_host_cap=host_cap(gguf_expert_memory_metrics(execution.gguf_expert_cache));
        require(tq_host_cap>0 && gguf_host_cap>0 && tq_host_cap+gguf_host_cap==double(execution.config.moe_host_cache_bytes) &&
            metric(execution,"ram_expert_cache_budget_bytes")==tq_host_cap+gguf_host_cap,
            "mixed encodings duplicated the explicit total host admission budget");
        mfq::cuda::share_routed_expert_host(execution,0);
        require(host_cap(tq_expert_memory_metrics(execution.tq_expert_cache))==tq_host_cap &&
            host_cap(gguf_expert_memory_metrics(execution.gguf_expert_cache))==gguf_host_cap,
            "host-share restoration assigned the complete mixed-format budget to each adapter");
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
        auto ids=tb::zeros({1,1},options.dtype(tb::kInt64));
        auto tq_input=tb::ones({1,33},options.dtype(tb::kFloat16))*.1;
        auto gguf_input=tb::ones({1,32},options.dtype(tb::kFloat16))*.1;
        tq->forward(tq_input,ids).cpu();gguf.forward(gguf_input,ids).cpu();
        const auto resident=[](const auto& values) {
            for(const auto& [key,value]:values)if(key=="resident_expert_cache_bytes")return value;
            return 0.;
        };
        const auto tq_used=resident(tq_expert_memory_metrics(execution.tq_expert_cache));
        const auto gguf_used=resident(gguf_expert_memory_metrics(execution.gguf_expert_cache));
        require(tq_used>0 && gguf_used>0 && metric(execution,"resident_expert_cache_bytes")==tq_used+gguf_used,
            "mixed cache metrics omitted an encoding");
        mfq::cuda::reserve_routed_capture_workspace(execution,SIZE_MAX/2);
        require(tq_expert_cache_budget_bytes(execution.tq_expert_cache)==0 &&
            gguf_expert_cache_budget_bytes(execution.gguf_expert_cache)==0,
            "mixed capture reservation left one cache admitted");
        {mfq::cuda::RoutedPrefillWorkspaceLoan loan(execution);loan.begin(SIZE_MAX/2);loan.finish();}
        mfq::cuda::restore_routed_decode_workspace(execution,0,0);
        require(mfq::cuda::routed_expert_cache_budget_bytes(execution)==(16u<<10),
            "mixed decode restoration lost a cache budget");
        tq->forward(tq_input,ids).cpu();gguf.forward(gguf_input,ids).cpu();
        const auto timings=mfq::cuda::routed_expert_phase_timings(execution);
        require(timings[1]==metric(execution,"expert_host_fetch_ms") &&
            timings[2]==metric(execution,"expert_host_copy_ms"),
            "lightweight mixed-cache timing counters differ from resource metrics");
        std::cout<<"Mixed TQ/GGUF cache budgets, accounting, capture loans and decode restoration passed on device "<<device<<'\n';
    }
    for(int device=0;device<devices;++device) {
        MfqCudaGuard guard(device);Source source(16384);
        CudaExecutionContext execution;execution.config.moe_device_cache_bytes=1u<<20;
        execution.config.moe_host_cache_bytes=4u<<20;
        TqMoeBanks banks{load_tq_routed_weight(execution,source,"gate"),load_tq_routed_weight(execution,source,"up"),
            load_tq_routed_weight(execution,source,"down")};
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
        auto input=tb::ones({1,33},options.dtype(tb::kFloat16))*.1;
        auto ids=tb::zeros({1,3},options.dtype(tb::kInt32));
        auto weights=tb::ones({1,3},options.dtype(tb::kFloat32))/3;
        prepare_tq_graph_pool(execution.tq_expert_cache);
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());tb::Tensor output;
        window.capture([&]{output=banks.forward(input,ids,weights,.25f);},[]{});
        for(int round=0;round<16;++round) {
            for(int route=0;route<3;++route)ids.select(1,route).fill_((round+route)%5);
            const auto expected=banks.forward(input,ids,weights,.25f).to(tb::kFloat32).cpu().contiguous();
            reclaim_tq_expert_duplicates(execution.tq_expert_cache,std::numeric_limits<std::size_t>::max());
            window.run();const auto actual=output.to(tb::kFloat32).cpu().contiguous();
            for(int64_t i=0;i<actual.numel();++i)require(actual.data_ptr<float>()[i]==expected.data_ptr<float>()[i],
                "TQ arena recycling raced its asynchronous writeback");
        }
        require(metric(execution,"ram_expert_writeback_bytes")>0,"large TQ images did not exercise GPU writeback");
        require(metric(execution,"expert_upload_groups")>0,"TQ graph did not batch its projection uploads");
    }
    for(int device=0;device<devices;++device) {
        MfqCudaGuard guard(device);Source source;
        CudaExecutionContext execution;execution.config.moe_device_cache_bytes=1u<<20;
        TqMoeBanks banks{load_tq_routed_weight(execution,source,"gate"),load_tq_routed_weight(execution,source,"up"),
            load_tq_routed_weight(execution,source,"down")};
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
        banks.forward(tb::ones({64,33},options.dtype(tb::kFloat16))*.1,
            tb::zeros({64,1},options.dtype(tb::kInt32)),tb::ones({64,1},options.dtype(tb::kFloat32)));
        TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
        require(metric(execution,"resident_expert_cache_bytes")==0,
            "temporary prefill images polluted the decode expert cache");
        restore_tq_decode_workspace(execution.tq_expert_cache,0,0);
        require(metric(execution,"expert_admission_retained_images")==3,"prefill did not seed all routed projections");
        require(metric(execution,"expert_admission_independent_lanes")>0,
            "background admission shares the foreground decode upload lane");
        auto input=tb::ones({1,33},options.dtype(tb::kFloat16))*.1;
        auto ids=tb::zeros({1,1},options.dtype(tb::kInt32));
        auto scores=tb::ones({1,1},options.dtype(tb::kFloat32));tb::Tensor output;
        require(prepare_tq_graph_pool(execution.tq_expert_cache),"TQ graph pool unavailable for admission test");
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
        {
            mfq::cuda::RoutedCacheLearningScope learning(execution,false);
            window.capture([&]{output=banks.forward(input,ids,scores);},[]{});
            window.run();
            require(metric(execution,"expert_admission_observations")==0,"graph warmup trained route demand");
        }
        {
            mfq::cuda::RoutedCacheLearningScope outer(execution,false);
            try {
                mfq::cuda::RoutedCacheLearningScope inner(execution,false);
                window.run();
                throw std::runtime_error("interrupted warmup");
            }catch(const std::runtime_error&) {}
            window.run();
            require(metric(execution,"expert_admission_observations")==0,
                "nested warmup restored learning before its outer scope ended");
        }
        ids.fill_(1);window.run();ids.fill_(0);
        require(metric(execution,"expert_admission_observations")==3,
            "warmup exception did not restore route learning");
        const auto hits=metric(execution,"expert_decode_hits");
        const auto bypasses=metric(execution,"expert_decode_resident_bypasses");
        const auto metadata=metric(execution,"expert_decode_metadata_updates");
        const auto demand=metric(execution,"expert_decode_pcie_read_bytes");window.run();
        require(metric(execution,"expert_decode_hits")==hits+3,"cold miss displaced protected hot projections");
        if(bypasses>0)require(metric(execution,"expert_decode_resident_bypasses")==bypasses+3,
            "resident decode re-entered generic bank preparation");
        require(metric(execution,"expert_decode_metadata_updates")==metadata &&
            metric(execution,"expert_decode_pcie_read_bytes")==demand,
            "fully resident replay submitted expert weights or metadata");
        auto wide=tb::ones({4,33},options.dtype(tb::kFloat16))*.1;
        auto wide_ids=tb::zeros({4,1},options.dtype(tb::kInt32));
        auto wide_scores=tb::ones({4,1},options.dtype(tb::kFloat32));tb::Tensor wide_output;
        mfq::cuda::DecodeWindow wide_window(mfq_current_cuda_stream());
        {mfq::cuda::RoutedCacheLearningScope learning(execution,false);
         wide_window.capture([&]{wide_output=banks.forward(wide,wide_ids,wide_scores);},[]{});}
        auto observations=metric(execution,"expert_admission_observations");
        {
            mfq::cuda::RoutedSpeculativeRoutes transaction;transaction.begin(execution,4);wide_window.run();
            require(metric(execution,"expert_admission_pending_routes")==12,"speculative routes not deferred");
            require(metric(execution,"expert_admission_observations")==observations,"unverified draft trained cache");
            transaction.resolve(1,false);
        }
        require(metric(execution,"expert_admission_observations")==observations+12,"executed verifier rows did not train expert reuse");
        observations+=12;
        {mfq::cuda::RoutedSpeculativeRoutes transaction;transaction.begin(execution,4);wide_window.run();}
        require(metric(execution,"expert_admission_pending_routes")==0 &&
            metric(execution,"expert_admission_observations")==observations,"route transaction cancellation leaked demand");
        {
            mfq::cuda::RoutedSpeculativeRoutes transaction;transaction.begin(execution,4);
            bool rejected=false;try{transaction.resolve(5,false);}catch(const std::invalid_argument&){rejected=true;}
            require(rejected && metric(execution,"expert_admission_pending_routes")==0,"invalid acceptance left active route transaction");
        }
        const auto learned_mass=metric(execution,"expert_admission_total_route_mass");
        banks.forward(tb::ones({64,33},options.dtype(tb::kFloat16))*.1,
            tb::zeros({64,1},options.dtype(tb::kInt32)),tb::ones({64,1},options.dtype(tb::kFloat32)));
        TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
        require(metric(execution,"expert_admission_total_route_mass")==learned_mass,
            "later prefill erased learned decode heat");
        std::cout<<"TQ byte admission, warmup isolation, preserved decode heat and accepted-prefix route transactions passed on device "<<device<<'\n';
    }
    for(int device=0;device<devices;++device)for(const auto stored_width:{256u,1024u}) {
        MfqCudaGuard guard(device);Source source(stored_width);CudaExecutionContext execution;
        execution.config.moe_device_cache_bytes=1u<<20;
        TqMoeBanks banks{load_tq_routed_weight(execution,source,"gate"),load_tq_routed_weight(execution,source,"up"),
            load_tq_routed_weight(execution,source,"down")};
        const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
        for(const auto* dtype:{"F16","BF16","F32"})for(const auto* prefix:{"rows.","payload.rows."}) {
            auto table=std::make_shared<GgufRowTable>(source,std::string(prefix)+dtype);
            auto stage=make_gguf_row_stage({table},7,{3},tb::Device(tb::kCUDA,device));
            stage->upload({6,1,6},{3});stage->decode();
            auto values=stage->output().to(tb::kFloat32).cpu().contiguous();
            for(int row=0;row<3;++row)for(int col=0;col<13;++col)
                require(values.data_ptr<float>()[row*13+col]==(row==1?1:6)*.25f+col*.03125f,
                    "floating PLE row staging changed values");
            float decoded[13];table->decode_row(1,decoded);
            for(int col=0;col<13;++col)require(decoded[col]==.25f+col*.03125f,"floating PLE CPU row decode differs");
        }
        const auto x=tb::ones({1,33},options.dtype(tb::kFloat16));
        auto ids=tb::zeros({1,3},options.dtype(tb::kInt32));
        {
            Source ragged(stored_width,true);TqRowTable table(ragged,"gate");
            const auto size=[&](int expert) {
                const auto wire=table.slice_rows(expert*19,(expert+1)*19);
                const mfq::TqLayout layout(wire.data(),wire.size());
                return ((layout.packed+3)&~std::size_t(3))+layout.packed_bytes+32;
            };
            CudaExecutionContext bounded;bounded.config.moe_device_cache_bytes=(size(1)+15)&~std::size_t(15);
            auto gate=load_tq_routed_weight(bounded,ragged,"gate");
            auto route=tb::ones({1,1},options.dtype(tb::kInt32));gate->forward(x,route).cpu();
            route.fill_(0);gate->forward(x,route).cpu();
            require(metric(bounded,"resident_expert_cache_bytes")<=((size(0)+15)&~std::size_t(15)),
                "TQ slot reuse retained the prior expert's larger allocation");
            resize_tq_expert_cache(bounded.tq_expert_cache,(size(0)+15)&~std::size_t(15));
            route.fill_(0);gate->forward(x,route).cpu();
            route.fill_(1);gate->forward(x,route).cpu();
            const auto hits=metric(bounded,"expert_cache_hits");
            route.fill_(0);gate->forward(x,route).cpu();
            require(metric(bounded,"expert_cache_hits")==hits+1,
                "oversized TQ miss displaced an expert it could not replace");
        }
        {
            TqRowTable table(source,"gate");const auto wire=table.slice_rows(0,19);
            const mfq::TqLayout layout(wire.data(),wire.size());
            const auto bytes=((layout.packed+3)&~std::size_t(3))+layout.packed_bytes+32;
            CudaExecutionContext bounded;bounded.config.moe_device_cache_bytes=1u<<20;
            bounded.config.moe_host_cache_bytes=2*bytes;
            auto gate=load_tq_routed_weight(bounded,source,"gate");
            auto route=tb::zeros({1,1},options.dtype(tb::kInt32));
            const auto run=[&](int expert) {route.fill_(expert);gate->forward(x,route).cpu();};
            run(0);resize_tq_expert_cache(bounded.tq_expert_cache,0);
            resize_tq_expert_cache(bounded.tq_expert_cache,1u<<20);run(1);run(2);
            require(metric(bounded,"ram_expert_payload_bytes")<=2*bytes,"TQ RAM admission exceeded its explicit budget");
            const auto reads=metric(bounded,"ssd_expert_read_count");run(0);
            require(metric(bounded,"ssd_expert_read_count")==reads,
                "TQ RAM admission evicted the only host copy before a GPU duplicate");
        }
        {
            TqRowTable table(source,"gate");const auto wire=table.slice_rows(0,19);
            const mfq::TqLayout layout(wire.data(),wire.size());
            const auto bytes=((layout.packed+3)&~std::size_t(3))+layout.packed_bytes+32;
            CudaExecutionContext bounded;bounded.config.moe_device_cache_bytes=(bytes+15)&~std::size_t(15);
            bounded.config.moe_host_cache_bytes=2*((bytes+15)&~std::size_t(15));
            auto gate=load_tq_routed_weight(bounded,source,"gate");
            auto route=tb::zeros({1,1},options.dtype(tb::kInt32));
            const auto reference=gate->forward(x,route).to(tb::kFloat32).cpu().contiguous();
            reclaim_tq_expert_duplicates(bounded.tq_expert_cache,std::numeric_limits<std::size_t>::max());
            require(metric(bounded,"ram_expert_payload_bytes")==0,"TQ GPU duplicate remained in RAM");
            route.fill_(1);gate->forward(x,route).cpu();
            require(metric(bounded,"ram_expert_writeback_bytes")>0,"TQ GPU eviction did not restore its RAM copy");
            const auto reads=metric(bounded,"ssd_expert_read_count");
            route.fill_(0);const auto restored=gate->forward(x,route).to(tb::kFloat32).cpu().contiguous();
            require(metric(bounded,"ssd_expert_read_count")==reads,"TQ demoted GPU image was reread from its source");
            for(int64_t i=0;i<reference.numel();++i)require(reference.data_ptr<float>()[i]==restored.data_ptr<float>()[i],
                "TQ GPU-to-RAM restoration changed its encoded image");
        }
        const auto weights=tb::ones({1,3},options.dtype(tb::kFloat32))/3;
        const auto evaluate=[&]{return banks.forward(x,ids,weights,.25f).to(tb::kFloat32).cpu().contiguous();};
        const auto expected=evaluate();
        require(metric(execution,"resident_expert_cache_bytes")>0,"TQ cache was not populated");
        const auto budget=mfq::cuda::routed_expert_cache_budget_bytes(execution);
        for(const bool alternate:{false,true}) {
            const auto owner=alternate?mfq::cuda::stream_from_pool(false,device):mfq_get_current_cuda_stream(device);
            const MfqCudaStreamGuard use(owner);
            mfq::cuda::RoutedPrefillWorkspaceLoan outer(execution),inner(execution);
            const auto context=mfq::cuda::default_context(device);
            require(outer.begin(4096)>=4096 && context->workspace_loan_active(),
                "TQ did not lend an existing contiguous cache hole");
            require(inner.begin(4096)>=4096,"nested TQ reservation lost its shared backing");
            evaluate();inner.finish();
            require(context->workspace_loan_active(),"inner TQ reservation ended its outer Context loan");
            outer.suspend();
            require(context->workspace_largest_available(owner.stream())==0,"TQ suspend allowed new borrowed allocations");
            outer.finish();
            require(!context->workspace_loan_active(),"last TQ reservation retained its Context loan");
        }
        {
            void* live=nullptr;
            const auto context=mfq::cuda::default_context(device);
            {
                TqPrefillWorkspaceLoan abandoned(execution.tq_expert_cache);
                require(abandoned.begin(4096)>=4096,"TQ abandoned-loan regression had no borrowed backing");
                live=context->allocate(256,mfq_current_cuda_stream());
            }
            require(context->workspace_loan_active() && context->workspace_live_bytes()>=256 &&
                mfq::cuda::routed_expert_cache_budget_bytes(execution)==budget,
                "abandoned TQ workspace lost live backing or retained its cache reservation");
            context->release(live,256,mfq_current_cuda_stream());
            require(!context->workspace_loan_active(),"abandoned TQ workspace did not retire its final allocation");
        }
        {
            mfq::cuda::RoutedPrefillWorkspaceLoan outer(execution),inner(execution);
            require(outer.begin(std::numeric_limits<std::size_t>::max()/2)==0,"TQ reported impossible borrowed capacity");
            require(inner.begin(std::numeric_limits<std::size_t>::max()/2)==0,"nested TQ reported impossible borrowed capacity");
            evaluate();require(metric(execution,"resident_expert_cache_bytes")==0,"TQ loan allowed cache growth");
            inner.finish();evaluate();
            require(metric(execution,"resident_expert_cache_bytes")==0,"inner loan released outer reservation");
        }
        const auto actual=evaluate();
        for(std::int64_t i=0;i<actual.numel();++i)
            require(actual.data_ptr<float>()[i]==expected.data_ptr<float>()[i],"TQ output changed after workspace loan");
        require(metric(execution,"resident_expert_cache_bytes")>0,"TQ cache did not resume after loan");
        restore_tq_decode_workspace(execution.tq_expert_cache,0,0);
        mfq::cuda::reserve_routed_capture_workspace(execution,SIZE_MAX/2);
        require(tq_expert_cache_budget_bytes(execution.tq_expert_cache)==0,"capture reservation allowed expert regrowth");
        mfq::cuda::restore_routed_decode_workspace_after_capture(execution);
        require(tq_expert_cache_budget_bytes(execution.tq_expert_cache)>0,"capture replaced the normal decode workspace plan");
        evaluate();
        require(mfq::cuda::routed_expert_cache_budget_bytes(execution)==budget,"TQ loan changed configured budget");
        const auto before=metric(execution,"ram_expert_payload_bytes");
        const auto reclaimed=reclaim_tq_expert_duplicates(execution.tq_expert_cache,1);
        require(reclaimed>0 && metric(execution,"ram_expert_payload_bytes")<before,"TQ duplicate reclaim failed");
        evaluate();restore_tq_decode_workspace(execution.tq_expert_cache,0,4096);evaluate();
        std::cout<<"TQ cache loans, overlapping reservations, output and RAM reclaim passed on device "<<device<<'\n';
        resize_tq_expert_cache(execution.tq_expert_cache,16u<<10);
        for(const auto dtype:{tb::kFloat16,tb::kBFloat16})for(const int rows:{1,2,3,4,8,32}) {
            if(!tq_graph_supports_tokens(rows))continue;
            auto input=tb::ones({rows,33},options.dtype(dtype))*.1;
            auto routes=tb::zeros({rows,3},options.dtype(tb::kInt32));
            const auto scores=tb::ones({rows,3},options.dtype(tb::kFloat32))/3;
            require(prepare_tq_graph_pool(execution.tq_expert_cache),"TQ graph pool unavailable");
            CudaIndependentBranchExecutor shared;shared.ensure(1,mfq_get_current_cuda_stream(device));
            mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());tb::Tensor output,side;
            bool forked=false;
            window.capture([&]{
                output=banks.forward(input,routes,scores,.25f,nullptr,0,-1,[&] {
                    forked=mfq_current_cuda_stream()==shared.streams[0].stream();
                    side=input+input;
                },&shared);
                mfq_cuda_record_stream(side,mfq_get_current_cuda_stream(device));
            },[]{});
            require(forked,"TQ shared branch did not use its independent stream");
            int mapped=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,device));
            if(mapped && mfq::cuda::default_context(device)->supports_async_allocations())
                require(window.segments()==1,"TQ grouped graph retained a host barrier");
            for(int turn=0;turn<(rows==1?256:3);++turn) {
                mfq::cuda::RoutedPrefillWorkspaceLoan loan(execution);
                if(turn==0 || (rows==1 && turn%2==0))loan.begin(std::numeric_limits<std::size_t>::max()/2);
                routes.fill_(turn%3+1);
                if(rows>1 && turn==2)routes.copy_(tb::arange(rows*3,options.dtype(tb::kInt32)).remainder(5).reshape({rows,3}));
                const auto reference=banks.forward(input,routes,scores,.25f).to(tb::kFloat32).cpu().contiguous();
                window.run();const auto result=output.to(tb::kFloat32).cpu().contiguous();
                const auto shared_value=side.to(tb::kFloat32).cpu().contiguous();
                const auto shared_reference=(input+input).to(tb::kFloat32).cpu().contiguous();
                require(shared_value.data_ptr<float>()[0]==shared_reference.data_ptr<float>()[0],
                    "TQ shared branch output was consumed before its join");
                double error=0,norm=0;
                for(std::int64_t i=0;i<result.numel();++i) {
                    const auto a=result.data_ptr<float>()[i],b=reference.data_ptr<float>()[i];
                    require(std::isfinite(a),"nonfinite TQ graph output after eviction");
                    error+=double(a-b)*(a-b);norm+=double(b)*b;
                }
                require(std::sqrt(error/std::max(norm,1e-20))<(dtype==tb::kBFloat16?.015:.003),
                    "TQ graph differs after expert eviction");
            }
            {
                routes.fill_(99);bool rejected=false;
                try{window.run();}catch(const std::out_of_range&){rejected=true;}
                require(rejected,"TQ graph accepted an invalid expert route");
                TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
        std::vector<tb::Tensor> references;
        {
            CudaExecutionContext streamed;streamed.config.moe_device_cache_bytes=16u<<10;
            streamed.config.moe_host_cache_bytes=1;
            TqMoeBanks alternating{load_tq_routed_weight(streamed,source,"gate"),
                load_tq_routed_weight(streamed,source,"up"),load_tq_routed_weight(streamed,source,"down")};
            std::vector<std::unique_ptr<mfq::cuda::DecodeWindow>> windows;
            std::vector<tb::Tensor> inputs,routes,scores,outputs;
            for(const int rows:{1,3}) {
                inputs.push_back(tb::ones({rows,33},options.dtype(tb::kFloat16))*.1);
                routes.push_back(tb::zeros({rows,3},options.dtype(tb::kInt32)));
                scores.push_back(tb::ones({rows,3},options.dtype(tb::kFloat32))/3);
                outputs.emplace_back();
                windows.push_back(std::make_unique<mfq::cuda::DecodeWindow>(mfq_current_cuda_stream()));
                const auto i=windows.size()-1;
                windows[i]->capture([&]{outputs[i]=alternating.forward(inputs[i],routes[i],scores[i],.25f);},[]{});
            }
            for(int pass=0;pass<32;++pass)for(int i=1;i>=0;--i) {
                routes[i].fill_(pass%5);
                const auto reference=alternating.forward(inputs[i],routes[i],scores[i],.25f).to(tb::kFloat32).cpu().contiguous();
                windows[i]->run();const auto result=outputs[i].to(tb::kFloat32).cpu().contiguous();
                double error=0,norm=0;
                for(int64_t j=0;j<result.numel();++j) {
                    const auto a=result.data_ptr<float>()[j],b=reference.data_ptr<float>()[j];
                    require(std::isfinite(a),"nonfinite TQ output after switching graph widths");
                    error+=double(a-b)*(a-b);norm+=double(b)*b;
                }
                require(std::sqrt(error/std::max(norm,1e-20))<.003,"TQ staged upload changed after switching graph widths");
            }
        }
        for(int pass=0;pass<2;++pass)for(int expert=0;expert<5;++expert) {
            ids.fill_(expert);auto value=evaluate();
            if(!pass)references.push_back(std::move(value));
            else for(int64_t i=0;i<value.numel();++i)
                require(value.data_ptr<float>()[i]==references[expert].data_ptr<float>()[i],
                    "TQ expert allocation reuse changed eager output");
        }
        require(metric(execution,"expert_cache_allocation_reuses")>0,"TQ expert storage reuse was not exercised");
        require(metric(execution,"expert_cache_reserved_bytes")<=(16u<<10),"TQ cache backing exceeded its budget");
        std::cout<<"TQ graph replay after cache eviction passed for FP16/BF16 and M=1/2/3/4/8/32 on device "<<device<<'\n';
        TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
        resize_tq_expert_cache(execution.tq_expert_cache,0);
        require(metric(execution,"expert_cache_reserved_bytes")==0,"TQ cache retained empty device chunks");
        const auto allocated=metric(execution,"ram_expert_cache_allocated_bytes");
        const auto released=reclaim_tq_expert_host(execution.tq_expert_cache,std::numeric_limits<std::size_t>::max());
        require(double(released)==allocated-metric(execution,"ram_expert_cache_allocated_bytes"),
            "TQ host reclaim reported payload deletion instead of released allocation bytes");
        require(metric(execution,"ram_expert_payload_bytes")==0,"TQ host cache did not release images");
        require(metric(execution,"ram_expert_pinned_reserved_bytes")==0,"TQ host cache retained empty pinned arenas");
        const auto ordinary=metric(execution,"ram_expert_cache_budget_bytes");
        share_tq_expert_host(execution.tq_expert_cache,SIZE_MAX);
        require(metric(execution,"ram_expert_cache_budget_bytes")==0,"TQ experts ignored shared KV headroom");
        evaluate();require(metric(execution,"ram_expert_payload_bytes")==0,"TQ expert admission consumed shared KV headroom");
        share_tq_expert_host(execution.tq_expert_cache,0);
        require(metric(execution,"ram_expert_cache_budget_bytes")==ordinary,"TQ shared RAM reservation did not restore its limit");
    }
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
