#include "layer_weight_stream.h"
#include "mfq_cuda_context.h"
#include "mfq_cuda_resident_allocations.h"
#include "mfq_weight_memory.h"
#include "core/full_block.h"
#include "models/qwen35/linear_attention.h"
#include <array>
#include <cstring>
#include <cstdio>
#include <unordered_map>
#include <limits>
#include <chrono>
#include <cstdlib>
#include <exception>
namespace mfq::cuda {
namespace tb=mfq_tensor_backend;
namespace {
thread_local LayerWeightStream* active_weight_stream=nullptr;
tb::Tensor view(const tb::Tensor& owner,std::size_t offset,const tb::Tensor& shape) {
    auto* pointer=static_cast<std::byte*>(owner.data_ptr())+offset;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto storage=std::make_shared<tb::TensorStorage>();
    storage->owner=std::make_shared<tb::Tensor>(owner);storage->base=pointer;
    storage->bytes=shape.nbytes();storage->device=owner.device();storage->pinned=owner.is_cpu();
    return tb::Tensor(storage,tb::make_contiguous_view(pointer,shape.sizes(),shape.scalar_type(),owner.device()));
#else
    return tb::from_blob(pointer,shape.sizes(),[owner](void*){},shape.options().device(owner.device()));
#endif
}
void append(std::vector<LayerWeightBinding>& result,tb::Tensor& value){
    if(value.defined() && value.numel())result.push_back({&value,{}});
}
void append(std::vector<LayerWeightBinding>& result,QuantLinear& weight){
    if(weight.tensor_parallel())throw std::invalid_argument("streamed projections require single-device ownership");
    if(weight.is_dense())append(result,weight.dense);
    else if(weight.is_channel_fp8()){append(result,weight.channel_fp8.values);append(result,weight.channel_fp8.scales);}
    else if(weight.is_gguf()){
        if(weight.gguf.rotation || weight.gguf.correction || weight.gguf.projection_correction)
            throw std::invalid_argument("streamed GGUF projections require self-contained storage");
        append(result,weight.gguf.packed);
    }else if(weight.is_tq()){
        result.push_back({&weight.tq.image,[target=&weight.tq](tb::Tensor image){*target=target->with_image(std::move(image));}});
    }else if(weight.is_mxfp4()){append(result,weight.mxfp4.weight.values);append(result,weight.mxfp4.weight.scales);}
    else if(weight.is_mxfp8()){append(result,weight.mxfp8.weight.values);append(result,weight.mxfp8.weight.scales);}
    else if(weight.logical_out)throw std::invalid_argument("projection has no streamed storage adapter");
}
void append(std::vector<LayerWeightBinding>& result,QuantLinearGroup& group){
    if(group.nint_grouped)throw std::invalid_argument("joined legacy projections have no streamed storage adapter");
    group.tq={};group.nvq_prefix2=false;
    for(auto& weight:group.layers)append(result,weight);
}
void append(std::vector<LayerWeightBinding>& result,FFN& ffn){
    if(ffn.is_moe)throw std::invalid_argument("layer weight streaming requires a dense feed-forward block");
    append(result,ffn.gate_up);append(result,ffn.down);
    if(ffn.important_neurons)append(result,*ffn.important_neurons);
}
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
std::shared_ptr<tb::TensorStorage> whole_prefill_weight(const tb::Tensor& weight) {
    auto storage=std::static_pointer_cast<tb::TensorStorage>(weight.weak_storage_owner().lock());
    // Borrowing changes device addresses. Only move a complete, unshared image.
    if(!storage || storage.use_count()!=2 || weight.byte_offset() || storage->bytes!=weight.nbytes())return {};
    return storage;
}
void rebind_prefill_weight(tb::Tensor& weight,const std::shared_ptr<tb::TensorStorage>& storage,
        void* address,std::shared_ptr<void> owner,
        std::function<void(std::uintptr_t,std::shared_ptr<void>)> record_stream,bool mapped) {
    storage->base=address;
    storage->mapped_host=mapped;
    storage->record_stream=std::move(record_stream);
    storage->owner=std::move(owner);
    weight=tb::Tensor(storage,weight.view_descriptor());
}
#endif
}

tb::Tensor* dense_prefill_gguf_image(QuantLinear& projection,int device) {
    if(!projection.is_gguf() || projection.tensor_parallel())return nullptr;
    auto& weight=projection.gguf;
    if(weight.rotation || weight.correction || weight.projection_correction || !weight.packed.is_cuda() ||
            weight.packed.get_device()!=device || weight.packed.is_mapped_host() || !weight.packed.is_contiguous())return nullptr;
    return &weight.packed;
}

std::size_t whole_prefill_weight_bytes(const tb::Tensor& weight) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const auto storage=whole_prefill_weight(weight);
    return storage?storage->bytes:0;
#else
    (void)weight;return 0;
#endif
}

std::size_t borrow_prefill_weight(tb::Tensor& weight) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const auto storage=whole_prefill_weight(weight);
    if(!storage)return 0;
    auto host=std::make_shared<tb::Tensor>(tb::empty({int64_t(storage->bytes)},
        tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true)));
    void* alias=nullptr;
    const auto status=cudaHostGetDevicePointer(&alias,host->data_ptr(),0);
    if(status==cudaErrorNotSupported || status==cudaErrorInvalidValue) {
        (void)cudaGetLastError();
        return 0;
    }
    TYLOINFER_CUDA_CHECK(status);
    const auto stream=mfq_current_cuda_stream();
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(host->data_ptr(),storage->base,storage->bytes,cudaMemcpyDeviceToHost,stream));
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
    rebind_prefill_weight(weight,storage,alias,std::move(host),{},true);
    return storage->bytes;
#else
    (void)weight;return 0;
#endif
}

std::size_t restore_prefill_weight(tb::Tensor& weight,Context& context,std::size_t floor) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const auto storage=whole_prefill_weight(weight);
    const auto available=context.available_memory();
    const auto headroom=available>floor?available-floor:0;
    if(!storage || (storage->bytes>headroom && storage->bytes-headroom>context.reusable_dense_weight_bytes()))return 0;
    const DenseWeightAllocationScope residency;
    auto image=tb::empty({int64_t(storage->bytes)},weight.options().dtype(tb::kUInt8));
    const auto host=std::static_pointer_cast<tb::Tensor>(storage->owner);
    const auto stream=mfq_current_cuda_stream();
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(image.data_ptr(),host->data_ptr(),storage->bytes,cudaMemcpyHostToDevice,stream));
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
    context.check_memory_limit();
    if(context.available_memory()<std::min(available,floor)) {
        image={};
        context.reclaim_idle_memory();
        return 0;
    }
    const auto device_storage=std::static_pointer_cast<tb::TensorStorage>(image.weak_storage_owner().lock());
    rebind_prefill_weight(weight,storage,device_storage->base,device_storage->owner,device_storage->record_stream,false);
    return storage->bytes;
#else
    (void)weight;(void)context;(void)floor;return 0;
#endif
}
struct LayerWeightStream::Impl {
    struct Field{LayerWeightBinding binding;tb::Tensor host;std::size_t offset;};
    struct Layer{tb::Tensor host;std::vector<Field> fields;};
    struct Slot{tb::Tensor image;Event ready,done;std::size_t layer=SIZE_MAX;bool used=false;};
    struct Trace{Event copy_start{cudaEventDefault},copy_end{cudaEventDefault},wait_start{cudaEventDefault},compute_start{cudaEventDefault},done{cudaEventDefault};bool valid=false;double submit_ms=0;};
    std::vector<Trace> traces;bool timings=false,prefetch=true;std::exception_ptr prefetch_error;
    LayerWeightStream* previous_active=nullptr;
    int device;Stream copy;std::array<Slot,2> slots;
    std::vector<Layer> layers;std::unordered_map<const void*,std::size_t> indices;
    std::size_t bytes=0,maximum=0,active=SIZE_MAX;
    explicit Impl(int target):device(target),copy(target){}
    void queue(std::size_t index){
        auto& slot=slots[prefetch?index%2:0];if(slot.layer==index)return;
        if(slot.used)copy.wait(slot.done);
        const auto submitted=std::chrono::steady_clock::now();
        if(timings)traces[index].copy_start.record(copy.get());
        const auto& source=layers[index].host;
        // Bound each DMA submission so a whole-layer copy can interleave with
        // tensor-core work without increasing the two resident layer buffers.
        constexpr std::size_t quantum=4u<<20;
        for(std::size_t offset=0;offset<source.nbytes();offset+=quantum)
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(slot.image.data_ptr())+offset,
                static_cast<const std::byte*>(source.data_ptr())+offset,std::min(quantum,source.nbytes()-offset),
                cudaMemcpyHostToDevice,copy.get()));
        if(timings)traces[index].copy_end.record(copy.get());
        slot.ready.record(copy.get());slot.layer=index;
        if(timings)traces[index].submit_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-submitted).count();
    }
};
LayerWeightStream::LayerWeightStream(int device,bool timings){
    const MfqCudaGuard guard(device);impl_=std::make_unique<Impl>(device);impl_->timings=timings;
}
LayerWeightStream::~LayerWeightStream(){
    if(!impl_)return;const MfqCudaGuard guard(impl_->device);
    try{impl_->copy.synchronize();for(auto& slot:impl_->slots)if(slot.used)slot.done.synchronize();}
    catch(const std::exception& error){std::fprintf(stderr,"layer_weight_stream cleanup_error=%s\n",error.what());}
}
void LayerWeightStream::add(Block& block){
    std::vector<LayerWeightBinding> bindings;
    if(auto* full=dynamic_cast<FullBlock*>(&block)){
        for(auto* value:{&full->attn_norm,&full->ffn_norm,&full->q_norm,&full->k_norm,&full->v_norm})append(bindings,*value);
        append(bindings,full->qkv);append(bindings,full->q_projection);append(bindings,full->k_projection);
        append(bindings,full->v_projection);append(bindings,full->o);append(bindings,full->ffn);
    }else if(auto* linear=dynamic_cast<qwen35::LinearAttentionBlock*>(&block)){
        for(auto* value:{&linear->attn_norm,&linear->ffn_norm,&linear->conv_weight,&linear->conv_bias,
            &linear->dt_bias,&linear->a_log,&linear->linear_norm,&linear->ab_proj.w,&linear->zab_proj.w,
            &linear->out_proj_dense})append(bindings,*value);
        for(auto* group:{&linear->in_proj,&linear->qkv_proj,&linear->qkvz_proj,&linear->ab_nint_proj})append(bindings,*group);
        append(bindings,linear->z_proj);append(bindings,linear->out_proj);append(bindings,linear->ffn);
    }else throw std::invalid_argument("block has no immutable layer storage adapter");
    add(&block,std::move(bindings));
}
void LayerWeightStream::add(const void* identity,std::vector<LayerWeightBinding> bindings){
    if(!identity || impl_->indices.count(identity) || bindings.empty() || impl_->slots[0].image.defined())
        throw std::invalid_argument("invalid streamed layer registration");
    Impl::Layer layer;std::size_t bytes=0;
    for(auto& binding:bindings){
        if(!binding.tensor || !binding.tensor->is_cpu() || !binding.tensor->is_contiguous())
            throw std::invalid_argument("streamed layer parameters must be contiguous CPU tensors");
        if(bytes>SIZE_MAX-255 || binding.tensor->nbytes()>SIZE_MAX-((bytes+255)&~std::size_t(255)))
            throw std::overflow_error("streamed layer storage size overflow");
        bytes=(bytes+255)&~std::size_t(255);const auto source=*binding.tensor;
        layer.fields.push_back({std::move(binding),source,bytes});bytes+=source.nbytes();
    }
    if(bytes>SIZE_MAX-impl_->bytes || bytes>std::size_t(INT64_MAX) || bytes>SIZE_MAX/2)
        throw std::overflow_error("streamed model storage size overflow");
    layer.host=tb::empty({int64_t(bytes)},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
    for(auto& field:layer.fields){
        std::memcpy(static_cast<std::byte*>(layer.host.data_ptr())+field.offset,field.host.data_ptr(),field.host.nbytes());
        auto host=view(layer.host,field.offset,field.host);
        if(field.binding.assign)field.binding.assign(host);else *field.binding.tensor=host;
        field.host=std::move(host);
    }
    impl_->indices.emplace(identity,impl_->layers.size());impl_->layers.push_back(std::move(layer));
    impl_->bytes+=bytes;impl_->maximum=std::max(impl_->maximum,bytes);
}
void LayerWeightStream::prepare(){
    if(impl_->layers.empty() || impl_->slots[0].image.defined())throw std::invalid_argument("invalid layer stream preparation");
    const MfqCudaGuard guard(impl_->device);const ResidentAllocationScope resident;
    if(const auto* value=std::getenv("TYLOINFER_LAYER_WEIGHT_PREFETCH")){
        if(std::string_view(value)!="0" && std::string_view(value)!="1")
            throw std::invalid_argument("TYLOINFER_LAYER_WEIGHT_PREFETCH must be zero or one");
        impl_->prefetch=std::string_view(value)=="1";
    }
    if(impl_->timings)impl_->traces.resize(impl_->layers.size());
    for(int index=0;index<(impl_->prefetch?2:1);++index)impl_->slots[index].image=tb::empty({int64_t(impl_->maximum)},
        tb::TensorOptions().device(tb::Device(tb::kCUDA,impl_->device)).dtype(tb::kUInt8));
    std::fprintf(stderr,"runtime_layer_weight_stream layers=%zu host_bytes=%zu device_bytes=%zu prefetch_layers=%d double_buffer=%d\n",
        impl_->layers.size(),impl_->bytes,device_bytes(),int(impl_->prefetch),int(impl_->prefetch));
}
LayerWeightStream::Lease LayerWeightStream::enter(const void* identity){
    if(impl_->active!=SIZE_MAX || !impl_->slots[0].image.defined())throw std::logic_error("layer weight stream is not ready or already in use");
    if(impl_->prefetch_error)std::rethrow_exception(std::exchange(impl_->prefetch_error,{}));
    const auto found=impl_->indices.find(identity);if(found==impl_->indices.end())throw std::invalid_argument("layer is absent from its weight stream");
    const MfqCudaGuard guard(impl_->device);const auto stream=mfq_get_current_cuda_stream(impl_->device);
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(),&capture));
    if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("streamed layers require eager execution");
    const auto index=found->second;auto& slot=impl_->slots[impl_->prefetch?index%2:0];
    if(impl_->timings)impl_->traces[index].wait_start.record(stream.stream());
    impl_->queue(index);
    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),slot.ready.get(),0));
    mfq_cuda_record_stream(slot.image,stream);
    if(impl_->timings)impl_->traces[index].compute_start.record(stream.stream());
    impl_->active=index;Lease lease(this);
    impl_->previous_active=active_weight_stream;active_weight_stream=this;
    for(auto& field:impl_->layers[index].fields){
        auto device=view(slot.image,field.offset,field.host);
        if(field.binding.assign)field.binding.assign(device);else *field.binding.tensor=std::move(device);
    }
    return lease;
}
void LayerWeightStream::prefetch_next()noexcept{
    if(!impl_->prefetch || impl_->active==SIZE_MAX || impl_->prefetch_error)return;
    const auto next=impl_->active+1;
    if(next<impl_->layers.size())try{impl_->queue(next);}
    catch(...){impl_->prefetch_error=std::current_exception();}
}
void prefetch_active_layer_weights()noexcept{
    if(active_weight_stream)active_weight_stream->prefetch_next();
}
void LayerWeightStream::leave()noexcept{
    if(impl_->active==SIZE_MAX)return;auto& slot=impl_->slots[impl_->prefetch?impl_->active%2:0];
    const auto status=cudaEventRecord(slot.done.get(),mfq_get_current_cuda_stream(impl_->device).stream());
    if(status!=cudaSuccess)std::fprintf(stderr,"layer_weight_stream fence_error=%s\n",cudaGetErrorString(status));
    if(impl_->timings){auto& trace=impl_->traces[impl_->active];
        (void)cudaEventRecord(trace.done.get(),mfq_get_current_cuda_stream(impl_->device).stream());trace.valid=true;}
    slot.used=true;
    // A projection normally kicks this copy while the rest of its layer is
    // still being submitted. Retain the fallback for layers without one.
    prefetch_next();impl_->active=SIZE_MAX;
    active_weight_stream=impl_->previous_active;impl_->previous_active=nullptr;
}
void LayerWeightStream::report_timings(){
    if(!impl_->timings)return;
    const MfqCudaGuard guard(impl_->device);double copy=0,wait=0,compute=0,overlap=0;std::size_t layers=0;
    auto elapsed=[](Event& first,Event& last){float ms=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,first.get(),last.get()));return double(ms);};
    for(std::size_t index=0;index<impl_->traces.size();++index){auto& trace=impl_->traces[index];if(!trace.valid)continue;
        trace.done.synchronize();const auto c=elapsed(trace.copy_start,trace.copy_end),w=elapsed(trace.wait_start,trace.compute_start),g=elapsed(trace.compute_start,trace.done);
        std::fprintf(stderr,"runtime_layer_transfer layer=%zu copy_ms=%.6f wait_ms=%.6f compute_ms=%.6f submit_ms=%.6f bytes=%zu\n",index,c,w,g,trace.submit_ms,impl_->layers[index].host.nbytes());
        if(index && impl_->traces[index-1].valid) {
            auto& previous=impl_->traces[index-1];
            const auto begin=elapsed(previous.compute_start,trace.copy_start),end=elapsed(previous.compute_start,trace.copy_end);
            overlap+=std::max(0.,std::min(elapsed(previous.compute_start,previous.done),end)-std::max(0.,begin));
        }
        copy+=c;wait+=w;compute+=g;++layers;
    }
    for(auto& trace:impl_->traces)trace.valid=false;
    std::fprintf(stderr,"runtime_layer_transfer_summary layers=%zu copy_ms=%.6f wait_ms=%.6f compute_ms=%.6f copy_compute_interval_overlap_ms=%.6f copy_outside_compute_ms=%.6f device_bytes=%zu\n",
        layers,copy,wait,compute,overlap,std::max(0.,copy-overlap),device_bytes());
}
std::size_t LayerWeightStream::host_bytes()const{return impl_->bytes;}
std::size_t LayerWeightStream::device_bytes()const{return impl_->maximum*(impl_->prefetch?2:1);}
} // namespace mfq::cuda
