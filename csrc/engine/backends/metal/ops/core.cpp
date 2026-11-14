#include "engine/ops.h"
#include "engine/backends/cpu/ops/tensor.h"
#include "engine/backends/metal/executor/executor.h"
#include "engine/backends/metal/native/stream.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/backends/metal/ops/tensor.h"
#include "backends/metal/runtime/mlx_sampling.h"

#include <mlx/mlx.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <map>

namespace tyloi::ops {
namespace {
namespace mx = mlx::core;
const Device cpu{DeviceType::cpu, 0}, gpu{DeviceType::metal, 0};
mx::Stream stream() { return mx::default_stream(mx::Device::gpu); }

const mx::array& metal_value(const Tensor& value) {
    if (!value.defined() || value.is_encoded() || value.device() != gpu)
        throw std::invalid_argument("operation requires a dense Metal 0 tensor");
    return metal::array(value);
}
void compatible(const Tensor& a, const Tensor& b) {
    (void)metal_value(a); (void)metal_value(b);
    if (a.device() != b.device()) throw std::invalid_argument("tensor devices do not match");
}
int axis(const Tensor& value, std::int64_t dimension) {
    if (dimension < 0) dimension += value.dim();
    if (dimension < 0 || dimension >= value.dim()) throw std::out_of_range("tensor dimension is out of range");
    return static_cast<int>(dimension);
}
int checked_int(std::int64_t value) {
    if (value < 0 || value > std::numeric_limits<int>::max()) throw std::overflow_error("Metal extent exceeds int32");
    return static_cast<int>(value);
}
void write(Tensor& target, const mx::array& value) {
    metal::ops::copy_into(target, metal::tensor(mx::contiguous(value,stream())));
}
template<class T> Tensor values(std::span<const T> data, ScalarType type, Device target) {
    if (data.size() > std::size_t(std::numeric_limits<std::int64_t>::max())) throw std::overflow_error("tensor size overflow");
    const std::int64_t shape[] = {static_cast<std::int64_t>(data.size())};
    if (target == gpu) return metal::copy_from_host(data.data(), shape, type);
    if (target != cpu) throw std::invalid_argument("unsupported tensor device");
    auto result = tyloi::cpu::ops::empty(shape, TensorOptions().device(cpu).dtype(type));
    if (!data.empty()) std::memcpy(result.dense().data_ptr(), data.data(), data.size_bytes());
    return result;
}
float half_value(std::uint16_t bits) {
    const auto sign = (bits & 0x8000) ? -1.f : 1.f;
    const auto exponent = (bits >> 10) & 31, mantissa = bits & 1023;
    if (exponent == 31) return mantissa ? std::numeric_limits<float>::quiet_NaN() : sign * std::numeric_limits<float>::infinity();
    return sign * std::ldexp(exponent ? 1.f + float(mantissa) / 1024.f : float(mantissa) / 1024.f,
                            exponent ? int(exponent) - 15 : -14);
}
template<class Out> std::vector<Out> read_host(const Tensor& input) {
    auto host = input.device() == gpu ? metal::copy(input, cpu) : ops::contiguous(input);
    if (!host.defined() || host.device() != cpu || host.is_encoded() || !host.representation<DenseTensor>())
        throw std::invalid_argument("host read requires a dense CPU 0 or Metal 0 tensor");
    const auto& dense = host.dense();
    std::vector<Out> result(static_cast<std::size_t>(dense.numel()));
    const auto convert = [&]<class T>() { const auto* p = dense.data_ptr<T>(); for (std::size_t i=0;i<result.size();++i) result[i]=static_cast<Out>(p[i]); };
    switch (dense.scalar_type()) {
        case ScalarType::boolean: convert.template operator()<bool>(); break;
        case ScalarType::uint8: convert.template operator()<std::uint8_t>(); break;
        case ScalarType::uint16: convert.template operator()<std::uint16_t>(); break;
        case ScalarType::uint32: convert.template operator()<std::uint32_t>(); break;
        case ScalarType::uint64: convert.template operator()<std::uint64_t>(); break;
        case ScalarType::int8: convert.template operator()<std::int8_t>(); break;
        case ScalarType::int16: convert.template operator()<std::int16_t>(); break;
        case ScalarType::int32: convert.template operator()<std::int32_t>(); break;
        case ScalarType::int64: convert.template operator()<std::int64_t>(); break;
        case ScalarType::float32: convert.template operator()<float>(); break;
        case ScalarType::float64: convert.template operator()<double>(); break;
        case ScalarType::float16: { const auto* p=dense.data_ptr<std::uint16_t>(); for(std::size_t i=0;i<result.size();++i)result[i]=static_cast<Out>(half_value(p[i])); break; }
        case ScalarType::bfloat16: { const auto* p=dense.data_ptr<std::uint16_t>(); for(std::size_t i=0;i<result.size();++i)result[i]=static_cast<Out>(std::bit_cast<float>(std::uint32_t(p[i])<<16)); break; }
        default: throw std::invalid_argument("unsupported host read dtype");
    }
    return result;
}
struct Capture { bool selected=false, active=false; std::map<std::string, std::pair<std::size_t,double>> timings; };
thread_local Capture* capture = nullptr;
void report_capture(const Capture& value, const std::string& name) {
    for (const auto& [label, data] : value.timings)
        std::cerr << "profile_item report=" << name << " name=" << label << " calls=" << data.first << " elapsed_ms=" << data.second << '\n';
}
}

Tensor tensor(std::span<const float> v, Device d) { return values(v,kFloat32,d); }
Tensor tensor(std::span<const std::int32_t> v, Device d) { return values(v,kInt32,d); }
Tensor tensor(std::span<const std::int64_t> v, Device d) { return values(v,kInt64,d); }
Tensor tensor(std::span<const std::uint8_t> v, Device d) { return values(v,kUInt8,d); }
Tensor from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options) {
    if (options.target_device() != cpu) throw std::invalid_argument("borrowed storage requires CPU 0");
    auto view = make_contiguous_view(data, shape, options.scalar_type(), cpu);
    if (!data && view.nbytes()) throw std::invalid_argument("from_blob requires non-null storage");
    auto storage = std::make_shared<TensorStorage>();
    storage->base=data; storage->bytes=view.nbytes(); storage->device=cpu; storage->pinned=options.pinned();
    return DenseTensor(std::move(storage), view);
}
Tensor cast(const Tensor& input, ScalarType type, Device target) {
    if (target != cpu && target != gpu) throw std::invalid_argument("unsupported tensor device");
    if (!input.defined() || input.is_encoded() || (input.device()!=cpu && input.device()!=gpu))
        throw std::invalid_argument("cast requires a dense CPU 0 or Metal 0 tensor");
    if (input.device()==cpu && target==cpu) return ops::cast(input,type);
    auto value = input.device()==gpu ? input : metal::copy(input,gpu);
    value = metal::ops::cast(value,type);
    return target==gpu ? value : metal::copy(value,cpu);
}
Tensor arange(std::int64_t first, std::int64_t last, Device device) {
    if (last > first && first < 0 && last > std::numeric_limits<std::int64_t>::max()+first)
        throw std::overflow_error("arange extent overflow");
    auto count=last>first ? last-first : 0;
    if (device==gpu) return metal::tensor(mx::add(mx::arange(checked_int(count),mx::int64,stream()),mx::array(first,mx::int64),stream()));
    if (device!=cpu) throw std::invalid_argument("unsupported tensor device");
    std::vector<std::int64_t> result(static_cast<std::size_t>(count));
    for(std::int64_t i=0;i<count;++i)result[static_cast<std::size_t>(i)]=first+i;
    return tensor(std::span<const std::int64_t>(result),cpu);
}
Tensor full(std::span<const std::int64_t> shape,std::int64_t value,Device device) {
    if(device==gpu) {
        mx::Shape dims; for(auto size:shape)dims.push_back(checked_int(size));
        return metal::tensor(mx::full(dims,mx::array(value,mx::int64),mx::int64,stream()));
    }
    if(device!=cpu)throw std::invalid_argument("unsupported tensor device");
    auto output=tyloi::cpu::ops::empty(shape,TensorOptions().device(cpu).dtype(kInt64));
    std::fill_n(output.dense().data_ptr<std::int64_t>(),output.numel(),value);return output;
}
std::vector<Tensor> chunk(const Tensor& input,std::int64_t count,std::int64_t dim) {
    if(count<=0)throw std::invalid_argument("chunk count must be positive");
    const auto d=axis(input,dim);
    const auto size=input.size(d);
    const auto width=size ? 1+(size-1)/count : 0;
    std::vector<Tensor> out;
    if(!size) { for(std::int64_t i=0;i<count;++i)out.push_back(ops::slice(input,d,0,0)); }
    else for(std::int64_t first=0;first<size;first+=width)out.push_back(ops::slice(input,d,first,std::min(size,first+width)));
    return out;
}
Tensor clamp_min(const Tensor& x,double v) { const auto& a=metal_value(x);return metal::tensor(mx::maximum(a,mx::array(v,a.dtype()),stream())); }
Tensor clamp_max(const Tensor& x,double v) { const auto& a=metal_value(x);return metal::tensor(mx::minimum(a,mx::array(v,a.dtype()),stream())); }
Tensor remainder(const Tensor& x,double v) { const auto& a=metal_value(x);return metal::tensor(mx::remainder(a,mx::array(v,a.dtype()),stream())); }
Tensor eq(const Tensor& a,const Tensor& b) { compatible(a,b);return metal::tensor(mx::equal(metal_value(a),metal_value(b),stream())); }
Tensor eq(const Tensor& a,double b) { const auto& x=metal_value(a);return metal::tensor(mx::equal(x,mx::array(b,x.dtype()),stream())); }
Tensor ne(const Tensor& a,const Tensor& b) { compatible(a,b);return metal::tensor(mx::not_equal(metal_value(a),metal_value(b),stream())); }
Tensor ne(const Tensor& a,double b) { const auto& x=metal_value(a);return metal::tensor(mx::not_equal(x,mx::array(b,x.dtype()),stream())); }
bool equal(const Tensor& a,const Tensor& b) {
    compatible(a,b);auto result=mx::array_equal(metal_value(a),metal_value(b),stream());metal::eval(result);return result.item<bool>();
}
Tensor gather(const Tensor& x,std::int64_t d,const Tensor& index) {
    compatible(x,index);if(x.dim()!=index.dim() || (index.scalar_type()!=kInt64 && index.scalar_type()!=kInt32))
        throw std::invalid_argument("gather requires matching ranks and integer indices");
    return metal::tensor(mx::take_along_axis(metal_value(x),metal_value(index),axis(x,d),stream()));
}
Tensor repeat(const Tensor& x,std::span<const std::int64_t> repeats) {
    if(repeats.size()<static_cast<std::size_t>(x.dim()))throw std::invalid_argument("repeat geometry is invalid");
    std::vector<int> shape;for(auto n:repeats)shape.push_back(checked_int(n));
    return metal::tensor(mx::tile(metal_value(x),std::move(shape),stream()));
}
Tensor repeat_interleave(const Tensor& x,std::int64_t n,std::int64_t dim) {
    return metal::tensor(mx::repeat(metal_value(x),checked_int(n),axis(x,dim),stream()));
}
Tensor tril(const Tensor& x,std::int64_t diagonal) {
    if(diagonal<INT_MIN || diagonal>INT_MAX)throw std::overflow_error("diagonal exceeds int32");
    return metal::tensor(mx::tril(metal_value(x),static_cast<int>(diagonal),stream()));
}
Tensor masked_fill(const Tensor& x,const Tensor& mask,double value) {
    compatible(x,mask);
    const auto& a=metal_value(x);
    auto condition=mx::broadcast_to(mx::astype(metal_value(mask),mx::bool_,stream()),a.shape(),stream());
    return metal::tensor(mx::where(condition,mx::array(value,a.dtype()),a,stream()));
}
void masked_fill_inplace(Tensor& x,const Tensor& mask,double value) { write(x,metal_value(masked_fill(x,mask,value))); }
void scatter(Tensor& x,std::int64_t dim,const Tensor& indices,const Tensor& source) {
    compatible(x,indices);compatible(x,source);
    if ((indices.scalar_type()!=kInt32 && indices.scalar_type()!=kInt64) ||
        x.scalar_type()!=source.scalar_type() || metal_value(indices).shape()!=metal_value(source).shape() ||
        x.dim()!=indices.dim()) throw std::invalid_argument("scatter tensor geometry or dtype mismatch");
    write(x,mx::put_along_axis(metal_value(x),metal_value(indices),metal_value(source),axis(x,dim),stream()));
}
void scatter_add(Tensor& x,std::int64_t dim,const Tensor& indices,const Tensor& source) {
    compatible(x,indices);compatible(x,source);
    if ((indices.scalar_type()!=kInt32 && indices.scalar_type()!=kInt64) ||
        x.scalar_type()!=source.scalar_type() || metal_value(indices).shape()!=metal_value(source).shape() ||
        x.dim()!=indices.dim()) throw std::invalid_argument("scatter tensor geometry or dtype mismatch");
    write(x,mx::scatter_add_axis(metal_value(x),metal_value(indices),metal_value(source),axis(x,dim),stream()));
}
void index_fill(Tensor& x,std::int64_t dim,const Tensor& indices,double value) {
    compatible(x,indices);const auto d=axis(x,dim);
    if(indices.scalar_type()!=kInt32 && indices.scalar_type()!=kInt64)
        throw std::invalid_argument("index_fill indices must be integers");
    auto shape=metal_value(x).shape();shape[d]=checked_int(indices.numel());
    mx::Shape index_shape(x.dim(),1);index_shape[d]=shape[d];
    auto expanded=mx::broadcast_to(mx::reshape(metal_value(indices),index_shape,stream()),shape,stream());
    auto filled=mx::full(shape,mx::array(value,metal_value(x).dtype()),metal_value(x).dtype(),stream());
    write(x,mx::put_along_axis(metal_value(x),expanded,filled,d,stream()));
}
Tensor masked_select(const Tensor& x,const Tensor& mask) {
    compatible(x,mask);
    const auto& source=metal_value(x);
    auto condition=mx::broadcast_to(mx::astype(metal_value(mask),mx::bool_,stream()),source.shape(),stream());
    auto flat=mx::reshape(source,{-1},stream()), selected=mx::reshape(condition,{-1},stream());
    const auto n=checked_int(flat.size());
    auto count=mx::sum(mx::astype(selected,mx::int64,stream()),stream());metal::eval(count);
    const auto k=checked_int(count.item<std::int64_t>());
    auto positions=mx::arange(n,mx::int64,stream());
    auto keys=mx::where(selected,positions,mx::add(positions,mx::array(std::int64_t(n),mx::int64),stream()),stream());
    auto ordered=mx::slice(mx::sort(keys,stream()),{0},{k},stream());
    return metal::tensor(mx::take(flat,ordered,0,stream()));
}

std::vector<float> read_floats(const Tensor& input) { return read_host<float>(input); }
std::vector<std::int64_t> read_integers(const Tensor& input) { return read_host<std::int64_t>(input); }
std::int64_t item(const Tensor& input) {
    if (input.numel()!=1) throw std::invalid_argument("item requires one element");
    return read_integers(input).front();
}
Tensor sample_greedy(const Tensor& logits) {
    const auto& value=metal_value(logits);
    if(logits.dim()!=2 || !logits.is_contiguous())throw std::invalid_argument("greedy logits must be contiguous [rows,vocab]");
    return metal::tensor(mx::astype(metal::sample_greedy(value),mx::int64,stream()));
}
Tensor sample_stochastic(const Tensor& logits,const Tensor& random,double temperature,std::int64_t top_k,double top_p) {
    compatible(logits,random);
    if(logits.dim()!=2 || !logits.is_contiguous() || random.scalar_type()!=kFloat32 || !random.is_contiguous())
        throw std::invalid_argument("invalid sampling tensors");
    if(top_k>INT_MAX)throw std::invalid_argument("top-k exceeds bounds");
    auto output=top_k>0 ? metal::sample_top_k_top_p(metal_value(logits),metal_value(random),temperature,int(top_k),top_p)
                        : metal::sample_softmax(metal_value(logits),metal_value(random),temperature);
    return metal::tensor(mx::astype(output,mx::int64,stream()));
}
void add_token_counts(Tensor& counts,const Tensor& ids) {
    compatible(counts,ids);
    if(counts.scalar_type()!=kInt32 || counts.dim()!=1 || ids.scalar_type()!=kInt64 || !counts.is_contiguous() || !ids.is_contiguous())
        throw std::invalid_argument("invalid sampling token counts");
    write(counts,metal::sample_token_counts_add(metal_value(counts),metal_value(ids)));
}
void apply_penalties(Tensor& logits,const Tensor& counts,double presence,double frequency,double repetition) {
    compatible(logits,counts);
    if(logits.dim()!=2 || logits.size(0)!=1 || counts.scalar_type()!=kInt32 || counts.dim()!=1 ||
       counts.numel()!=logits.size(1) || !logits.is_contiguous() || !counts.is_contiguous())
        throw std::invalid_argument("invalid sampling penalty tensors");
    write(logits,metal::sample_apply_penalties(metal_value(logits),metal_value(counts),presence,frequency,repetition));
}
Tensor constrain(const Tensor& logits,const MfqTokenConstraintPtr& constraint) {
    if(!constraint)throw std::invalid_argument("constraint is unavailable");
    auto values=read_floats(logits);constraint->apply(values.data(),values.size());
    auto output=tensor(std::span<const float>(values),cpu);
    std::vector<std::int64_t> shape;for(std::int64_t i=0;i<logits.dim();++i)shape.push_back(logits.size(i));
    return ops::reshape(output,shape);
}
std::pair<std::vector<float>,std::vector<std::int64_t>> unpack_topk(const Tensor& packets,std::int64_t count) {
    if(!packets.defined() || packets.dim()!=2 || packets.size(0)<1 || packets.scalar_type()!=kInt64 ||
       count<1 || count>64 || packets.size(1)<count)
        throw std::invalid_argument("invalid top-k packets");
    const auto host=read_integers(packets);
    std::vector<float> values(static_cast<std::size_t>(packets.size(0)*count));
    std::vector<std::int64_t> ids(values.size());
    for(std::int64_t row=0;row<packets.size(0);++row)for(std::int64_t col=0;col<count;++col) {
        auto word=static_cast<std::uint64_t>(host[std::size_t(row*packets.size(1)+col)]);
        const auto index=std::size_t(row*count+col);
        values[index]=std::bit_cast<float>(static_cast<std::uint32_t>(word>>32));
        ids[index]=static_cast<std::uint32_t>(word);
    }
    return {std::move(values),std::move(ids)};
}
std::pair<std::vector<float>,std::vector<std::int64_t>> read_topk(const Tensor& logits,std::int64_t count,bool) {
    const auto& x=metal_value(logits);
    if(logits.dim()!=2 || logits.size(0)<1 || count<1 || count>64 || count>logits.size(1))
        throw std::invalid_argument("invalid top-k logits");
    auto indices=mx::slice(mx::argsort(mx::negative(x,stream()),-1,stream()),{0,0},{checked_int(logits.size(0)),checked_int(count)},stream());
    auto values=mx::take_along_axis(x,indices,-1,stream());
    return {read_floats(metal::tensor(std::move(values))),read_integers(metal::tensor(mx::astype(indices,mx::int64,stream())))};
}
void decode_commit(const Tensor& next,Tensor& generated,Tensor& step,Tensor& input,Tensor& position,Tensor& length) {
    const std::array<const Tensor*,6> operands{&next,&generated,&step,&input,&position,&length};
    for(const auto* value : operands) {
        compatible(next,*value);
        if(value->scalar_type()!=kInt64 || !value->is_contiguous())throw std::invalid_argument("decode commit requires contiguous int64 tensors");
    }
    if(next.numel()!=1 || generated.numel()<1 || step.numel()!=1 || input.numel()!=1 || position.numel()!=1 || length.numel()!=1)
        throw std::invalid_argument("invalid decode commit geometry");
    auto flat=mx::reshape(metal_value(generated),{-1},stream());
    auto updated=mx::put_along_axis(flat,mx::reshape(metal_value(step),{1},stream()),mx::reshape(metal_value(next),{1},stream()),0,stream());
    updated=mx::reshape(updated,metal_value(generated).shape(),stream());
    metal::eval(updated);
    write(generated,updated);write(input,metal_value(next));
    write(position,mx::add(metal_value(position),mx::array(std::int64_t(1),mx::int64),stream()));
    write(length,mx::add(metal_value(length),mx::array(std::int64_t(1),mx::int64),stream()));
    write(step,mx::add(metal_value(step),mx::array(std::int64_t(1),mx::int64),stream()));
}

std::unique_ptr<::tyloi::Executor::Scope> dense_scope(::tyloi::Executor* executor,Device target) {
    if(!executor) return metal::Executor::current().enter();
    if(target!=gpu)throw std::invalid_argument("dense scope requires Metal 0");
    auto* backend=dynamic_cast<metal::Executor*>(executor);
    if(!backend)throw std::invalid_argument("dense scope requires a Metal executor");
    return backend->enter();
}
struct ProfileRange::Impl {
    Capture* owner;
    std::string name;
    std::unique_ptr<::tyloi::Executor::Timer> timer;
    bool finished=false;
    Impl(Capture* capture,const char* label) : owner(capture), name(label), timer(metal::Executor::current().timer()) {}
    void finish() {
        if(finished)return;
        timer->stop();auto& entry=owner->timings[name];++entry.first;entry.second+=timer->elapsed_ms();finished=true;
    }
};
ProfileRange::ProfileRange(const char* name,std::optional<Device> device) {
    if(device && *device!=gpu && *device!=cpu)throw std::invalid_argument("unsupported profile device");
    if(capture && capture->active)impl_=std::make_unique<Impl>(capture,name);
}
ProfileRange::~ProfileRange() { if(impl_ && !impl_->finished)try { impl_->finish(); } catch(...) {} }
void ProfileRange::finish() { if(impl_)impl_->finish(); }
struct ProfileCapture::Impl {
    Capture state;
    Capture* previous;
    std::string report_name;
    Impl(bool selected,std::string name) : previous(capture),report_name(std::move(name)) {
        state.selected=selected;state.active=selected;
        if(selected)capture=&state;
    }
    ~Impl() {
        if(state.selected && !report_name.empty() && std::uncaught_exceptions()==0)report_capture(state,report_name);
        capture=previous;
    }
};
ProfileCapture::ProfileCapture(bool selected) : impl_(std::make_unique<Impl>(selected,std::string{})) {}
ProfileCapture::ProfileCapture(bool selected,const std::string& name) : impl_(std::make_unique<Impl>(selected,name)) {}
ProfileCapture::~ProfileCapture() = default;
void ProfileCapture::report(const std::string& name) { if(impl_ && impl_->state.selected)report_capture(impl_->state,name); }
void ProfileCapture::resume() { if(impl_)impl_->state.active=impl_->state.selected; }
void ProfileCapture::suspend() { if(impl_)impl_->state.active=false; }
ProfileCapture ProfileCapture::prefill_once(std::int64_t tokens) {
    const auto* option=std::getenv("TYLOINFER_PROFILE_PREFILL_ONCE");
    const bool enabled=option && option[0]=='1';
    const auto* skip=std::getenv("TYLOINFER_PROFILE_PREFILL_SKIP");
    const auto skipped=skip ? std::max<long long>(0,std::atoll(skip)) : 0;
    static std::atomic<std::uint64_t> requests{0};
    const auto index=enabled ? requests.fetch_add(1) : 0;
    const bool selected=enabled && index==std::uint64_t(skipped);
    if(enabled)std::cerr << "prefill_profile_request index=" << index << " tokens=" << tokens << " collect=" << int(selected) << '\n';
    return ProfileCapture(selected,"prefill_regression_once");
}
} // namespace tyloi::ops
