#include "engram.h"

#include "mfq_cuda_norm_ops.h"

#include <array>
#include <cmath>
#include <limits>

namespace mfq::cuda::hy4 {
namespace tb=mfq_tensor_backend;
using Tensor=tb::Tensor;

static const std::array<float,256>& e4m3_values() {
    static const auto table=[] {
        std::array<float,256> result{};
        for (int byte=0;byte<256;++byte) {
            const int exp=(byte>>3)&15, mantissa=byte&7;
            const float value=(byte&127)==127?std::numeric_limits<float>::quiet_NaN():
                exp?std::ldexp(1+mantissa/8.f,exp-7):std::ldexp(float(mantissa),-9);
            result[byte]=std::copysign(value,byte&128?-1.f:1.f);
        }
        return result;
    }();
    return table;
}

static size_t fill_rows(const mfq::ModelSource::TensorBatchReader& read,const std::vector<int64_t>& ids,
                        size_t width,void* destination) {
    auto* bytes=static_cast<std::byte*>(destination);
    const size_t size=ids.size()*width;
    std::memset(bytes,0,size);
    std::vector<mfq::ReadSpan> spans;
    spans.reserve(ids.size());
    for (size_t i=0;i<ids.size();++i) if (ids[i]>=0)
        spans.push_back({uint64_t(ids[i])*width,bytes+i*width,width});
    if (!spans.empty()) read(spans);
    return size;
}

void EngramState::load(const mfq::ModelSource& source,const mfq::models::hy4_flash::EngramConfig& config) {
    constexpr auto table_name="model.associative_memory.embedding.weight";
    const auto* table=source.find_tensor(table_name);
    if (!table || !table->shape || table->shape->size()!=2 || table->shape->at(0)<=0 ||
        table->shape->at(1)!=config.row_width || table->nbytes/uint64_t(config.row_width)!=uint64_t(table->shape->at(0)) ||
        table->nbytes%uint64_t(config.row_width))
        throw std::invalid_argument("Hy4 Engram FP8 table has incompatible row geometry");
    read_rows=source.tensor_batch_reader(table_name);
    if (!read_rows) throw std::invalid_argument("Hy4 Engram requires retained range reads");
    parallel_reads_=source.supports_parallel_tensor_reads();
    row_width=config.row_width;
    table_rows=uint64_t(table->shape->at(0));
    std::vector<int32_t> map;
    map.reserve(config.remap.size());
    for (auto id:config.remap) {
        if (id<-1 || id>=int64_t(config.remap.size())) throw std::invalid_argument("invalid Hy4 Engram token mapping");
        map.push_back(int32_t(id));
    }
    for (size_t i=0;i<config.offsets.size();++i)
        if (config.offsets[i]+config.vocab_sizes[i]>table_rows)
            throw std::invalid_argument("Hy4 Engram bucket exceeds the table");
    hash=std::make_unique<Hash>(int(map.size()),int(config.ngram),int(config.heads),map[config.pad],config.layers,
        std::vector<int64_t>(config.layers.size(),int64_t(table_rows)),
        std::vector<int64_t>(config.vocab_sizes.begin(),config.vocab_sizes.end()),
        std::vector<int64_t>(config.offsets.begin(),config.offsets.end()),
        std::vector<int64_t>(config.multipliers.begin(),config.multipliers.end()),std::move(map),Hash::HashPolicy::unsigned_modulo);
}

void EngramState::prepare(const Tensor& ids,int64_t position) {
    const auto host=ids.to(tb::kCPU).contiguous();
    prepare(hash->forward(host.data_ptr<int64_t>(),int(ids.size(0)),int(ids.size(1)),position));
}

void EngramState::prepare(HashBatch hashes) {
    batch=std::move(hashes);
    packed_valid_=false;
    if (parallel_reads_) enqueue(batch);
}

void EngramState::enqueue(const HashBatch& hashes) {
    prefetch_.start(hashes.values,{hashes.batch,hashes.tokens,hashes.layers,hashes.columns},
        hashes.values.size()*row_width,[read=read_rows,ids=hashes.values,width=size_t(row_width)](void* destination) {
            return fill_rows(read,ids,width,destination);
        });
}

bool EngramState::prefetch(const Tensor& ids,int64_t position) {
    if (!parallel_reads_) return false;
    if (!ids.is_cpu() || !ids.is_contiguous() || ids.dim()!=2 || ids.scalar_type()!=tb::kInt64 ||
        ids.size(0)<=0 || ids.size(1)<=0)
        throw std::invalid_argument("Hy4 Engram prefetch requires contiguous host token IDs");
    enqueue(hash->preview(ids.data_ptr<int64_t>(),int(ids.size(0)),int(ids.size(1)),position));
    return true;
}

std::pair<Tensor,Tensor> EngramState::lookup(int layer,tb::Device device) {
    const auto* ids=batch.layer_data(layer);
    const int64_t rows=int64_t(batch.batch)*batch.tokens;
    const int64_t columns=batch.columns;
    const size_t bytes=size_t(rows*columns*row_width);
    if (!packed_valid_) {
        const auto required=int64_t(batch.values.size()*row_width);
        if (!packed_.defined() || packed_.numel()<required)
            packed_=tb::empty({required},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8));
        if (!prefetch_.consume(batch.values,{batch.batch,batch.tokens,batch.layers,batch.columns},packed_.data_ptr()))
            fill_rows(read_rows,batch.values,row_width,packed_.data_ptr());
        packed_valid_=true;
    }
    auto opts=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32).pinned_memory(true);
    auto valid=tb::empty({batch.batch,batch.tokens,1,1},opts);
    for (int64_t row=0;row<rows;++row) valid.data_ptr<float>()[row]=ids[row*columns]>=0?1.f:0.f;
    auto values=tb::empty({batch.batch,batch.tokens,columns,row_width},opts);
    const auto& codebook=e4m3_values();
    const auto* packed=packed_.data_ptr<uint8_t>()+size_t(layer)*bytes;
    auto* data=values.data_ptr<float>();
    for (size_t i=0;i<bytes;++i) data[i]=codebook[packed[i]];
    return {values.to(device),valid.to(device)};
}

Tensor EngramLayer::forward(CudaExecutionContext& execution,EngramState& table,Tensor streams) const {
    auto [rows,valid]=table.lookup(slot,streams.device());
    return forward_rows(execution,std::move(streams),std::move(rows),valid);
}

Tensor EngramLayer::forward_rows(CudaExecutionContext& execution,Tensor streams,Tensor rows,const Tensor& valid) const {
    constexpr double epsilon=1e-6;
    auto normalized=rows*tb::rsqrt(tb::sum(rows*rows,-1,true)/rows.size(-1)+epsilon)*lookup_norm;
    const auto b=streams.size(0),t=streams.size(1),hc=streams.size(2),hidden=streams.size(3);
    auto key_value=projection.forward(execution,normalized.reshape({b,t,-1})).to(tb::kFloat32)
        .reshape({b,t,hc+1,hidden})+projection_bias;
    auto key=key_value.narrow(2,0,hc).contiguous();
    auto value=key_value.narrow(2,hc,1).contiguous();
    auto x=rms_norm_cuda(streams.contiguous(),unit,epsilon);
    auto k=rms_norm_cuda(key,unit,epsilon);
    auto dot=tb::sum(x*k*gate,-1,true)/std::sqrt(double(hidden));
    auto magnitude=tb::sqrt(tb::clamp_min(dot.abs(),epsilon));
    auto sign=(dot>=0.0).to(tb::kFloat32)*2.0-1.0;
    auto scale=tb::sigmoid(magnitude*sign)+hc_scale;
    auto signal=rms_norm_cuda(value,gamma,epsilon);
    return streams+(signal*scale*valid).to(streams.scalar_type());
}

}  // namespace mfq::cuda::hy4
