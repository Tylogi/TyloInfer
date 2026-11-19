#include "ram_session.h"

#include "engine/models/session/paged.h"
#include "engine/ops/ops.h"
#include "engine/state/kv/qsa/session.h"

#include <array>
#include <climits>
#include <limits>

namespace tyloi::models::qwen4_exp {
namespace qsa = tyloinfer::qsa;
namespace {
void require(bool condition,const char* message) {
    if(!condition)throw CudaSessionStateError(message);
}
size_t length(SessionBlockReader& r,const char* name) {
    const auto n=r.scalar<uint64_t>(name);
    require(n<=std::numeric_limits<size_t>::max(),"QSA prefix size overflow");return size_t(n);
}
void write_tail(SessionBlockWriter& w,const ::tyloi::DenseTensor& t) {
    if(!t.defined()){w.scalar<uint64_t>(0);return;}
    const auto cpu=::tyloi::ops::contiguous(::tyloi::ops::cast(t,::tyloi::kFloat32,::tyloi::Device(::tyloi::kCPU)));
    const auto bytes=cpu.nbytes();w.scalar<uint64_t>(bytes);w.raw(cpu.dense().data_ptr(),bytes);
}
::tyloi::DenseTensor read_tail(SessionBlockReader& r,const std::vector<int64_t>& shape) {
    size_t expected=sizeof(float);
    for(auto dim:shape) {
        require(size_t(dim)<=std::numeric_limits<size_t>::max()/expected,"QSA raw tail size overflow");
        expected*=size_t(dim);
    }
    const auto bytes=length(r,"QSA raw tail size");
    require(bytes==expected && bytes<=r.remaining(),"incompatible QSA raw tail size");
    auto t=::tyloi::ops::empty(shape,::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kFloat32));
    r.raw(t.dense().data_ptr(),bytes,"QSA raw tail");return t.dense();
}
}
void write_ram_page(SessionBlockWriter& w,const state::qsa::Session& s,size_t start,size_t count,bool exact) {
    require(s.tokens>0 && s.kv && s.index.size()==s.kv->blocks.size() &&
        s.kv->blocks.size()==size_t(s.tokens/4) && start<=size_t(s.tokens) && count<=size_t(s.tokens)-start,
        "invalid compressed QSA prefix boundary");
    for(int n:{s.heads,s.kv_heads,s.width,s.padded,s.index_width,s.ram_bits})w.scalar<uint32_t>(uint32_t(n));
    const auto first=start/4,last=(start+count)/4;
    require(last-first<=std::numeric_limits<uint32_t>::max(),"QSA prefix block count overflow");
    w.scalar<uint32_t>(uint32_t(last-first));
    for(auto id=first;id<last;++id) {
        const auto& b=s.kv->blocks[id];const auto image=b.read();
        w.scalar<uint32_t>(uint32_t(b.tier));
        for(auto n:b.bytes)w.scalar<uint64_t>(n);
        w.scalar<uint64_t>(b.access);
        for(unsigned n:{image.key_bits,image.value_bits,image.key_state_bits,image.value_state_bits})w.scalar<uint32_t>(n);
        require(bool(s.index[id]),"missing QSA index archive");const auto index=s.index[id]->read();
        w.scalar<uint64_t>(image.key.size());w.scalar<uint64_t>(image.value.size());w.scalar<uint64_t>(index->size());
        w.raw(image.key.data(),image.key.size());w.raw(image.value.data(),image.value.size());w.raw(index->data(),index->size());
    }
    if(exact && s.tokens%4) {write_tail(w,s.tail_keys);write_tail(w,s.tail_values);write_tail(w,s.tail_index);}
}

void read_ram_page(SessionBlockReader& r,state::qsa::Session& s,size_t start,size_t count,bool exact) {
    require(start<=size_t(std::numeric_limits<int64_t>::max()) &&
        count<=size_t(std::numeric_limits<int64_t>::max())-start,"invalid compressed QSA prefix boundary");
    std::array<uint32_t,6> geometry{};for(auto& n:geometry)n=r.scalar<uint32_t>("QSA geometry");
    const auto [heads,kv_heads,width,padded,iw,bits]=geometry;
    require(heads && heads<=INT_MAX && kv_heads && kv_heads<=INT_MAX && heads%kv_heads==0 &&
        width && width<=INT_MAX && padded>=width && padded<=INT_MAX && padded%256==0 &&
        iw && iw<=INT_MAX && (bits==16 || bits==8 || bits==4) && uint64_t(kv_heads)*padded<=INT_MAX,
        "invalid compressed QSA prefix geometry");
    if(!s.kv) {
        require(start==0,"incomplete compressed QSA prefix chain");
        s.heads=heads;s.kv_heads=kv_heads;s.width=width;s.padded=padded;
        s.index_width=iw;s.ram_bits=bits;s.kv=std::make_shared<qsa::RamKvSnapshot>();
    }
    require(s.heads==int(heads) && s.kv_heads==int(kv_heads) && s.width==int(width) && s.padded==int(padded) &&
        s.index_width==int(iw) && s.ram_bits==int(bits),"inconsistent compressed QSA prefix layers");
    const size_t first=start/4,last=(start+count)/4,n=r.scalar<uint32_t>("QSA block count");
    require(n==last-first && n<=r.remaining()/76,"invalid compressed QSA prefix block count");
    require(s.kv->blocks.size()==first && s.index.size()==first && s.tokens==int64_t(start),
        "incomplete compressed QSA prefix chain");
    const auto rows=uint64_t(kv_heads)*padded*4/256;
    const auto fp16=rows*512,affine=((rows*3+3)&~uint64_t(3))+rows*256;
    const auto tq4=((rows*2+(rows+1)/2+(rows*3+7)/8+3)&~uint64_t(3))+rows*128;
    require(2*fp16<=std::numeric_limits<size_t>::max() && uint64_t(iw)*10+4<=std::numeric_limits<size_t>::max(),
        "QSA prefix size overflow");
    const size_t index_bytes=uint64_t(iw)*10+4;
    for(auto id=first;id<last;++id) {
        qsa::RamKvSnapshotBlock b;const auto tier=r.scalar<uint32_t>("QSA RAM tier");
        require(tier<=3,"invalid QSA RAM tier");b.tier=qsa::RamKvTier(tier);
        for(auto& x:b.bytes)x=length(r,"QSA tier bytes");
        require(b.bytes==std::array<size_t,3>{size_t(2*fp16),size_t(2*affine),size_t(2*tq4)},
            "invalid QSA RAM pressure sizes");
        b.access=r.scalar<uint64_t>("QSA block access");
        const auto kb=r.scalar<uint32_t>("QSA key bits"),vb=r.scalar<uint32_t>("QSA value bits");
        const auto ks=r.scalar<uint32_t>("QSA key states"),vs=r.scalar<uint32_t>("QSA value states");
        const auto tq_bits=[](unsigned bits){return (bits>=1 && bits<=8) || bits==255;};
        require(((kb==0 && vb==0) || (kb==qsa::affine8_tag && vb==qsa::affine8_tag) || (tq_bits(kb) && tq_bits(vb))) &&
            (ks==14 || ks==16) && (vs==14 || vs==16),"invalid QSA encoding flags");
        const auto key_bytes=length(r,"QSA key size"),value_bytes=length(r,"QSA value size");
        const auto ilen=length(r,"QSA index size");
        require(key_bytes && value_bytes && key_bytes<=fp16 && value_bytes<=fp16 && ilen==index_bytes &&
            key_bytes<=r.remaining() && value_bytes<=r.remaining()-key_bytes && ilen<=r.remaining()-key_bytes-value_bytes,
            "invalid compressed QSA prefix extent");
        const auto actual=kb==0 && vb==0?qsa::RamKvTier::Fp16:
            kb==qsa::affine8_tag && vb==qsa::affine8_tag?qsa::RamKvTier::Affine8:qsa::RamKvTier::Tq4;
        require(tier==3 || unsigned(actual)==tier,"QSA payload precision does not match its RAM tier");
        auto data=std::make_shared<std::vector<uint8_t>>(key_bytes+value_bytes);
        r.raw(data->data(),data->size(),"QSA key/value");
        auto image=std::make_shared<qsa::RamKvImage>();image->owner=data;
        image->key=data->data();image->value=data->data()+key_bytes;image->key_bytes=key_bytes;image->value_bytes=value_bytes;
        image->key_bits=uint8_t(kb);image->value_bits=uint8_t(vb);
        image->key_state_bits=uint8_t(ks);image->value_state_bits=uint8_t(vs);
        b.image=std::move(image);
        auto index=std::make_shared<std::vector<uint8_t>>(ilen);r.raw(index->data(),ilen,"QSA index");
        auto archive=std::make_shared<state::qsa::IndexArchive>();archive->resident=std::move(index);
        s.kv->blocks.push_back(std::move(b));s.index.push_back(std::move(archive));
    }
    s.tokens=int64_t(start+count);
    if(exact && (start+count)%4) {
        const auto rem=int64_t((start+count)%4);
        s.tail_keys=read_tail(r,{1,rem,s.kv_heads,s.width});
        s.tail_values=read_tail(r,{1,rem,s.kv_heads,s.width});
        s.tail_index=read_tail(r,{1,rem,s.index_width});
    } else s.tail_keys=s.tail_values=s.tail_index={};
}
} // namespace tyloi::models::qwen4_exp
