#include "engine/models/qwen4_exp/session_codec.h"

#include "engine/models/session/snapshot.h"
#include "engine/models/qwen4_exp/ram_session.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include "engine/ops/ops.h"
#include "engine/state/kv/qsa/session.h"

#include <algorithm>
#include <array>

namespace tyloi::models::qwen4_exp {
namespace {
constexpr std::array<uint8_t,8> qwen4_magic{'M','F','Q','C','U','D','2',0};
struct Qwen4PagedHeader { uint32_t start,count,layers;bool exact,mtp; uint32_t version; };
Qwen4PagedHeader read_qwen4_header(SessionBlockReader& reader) {
    std::array<uint8_t,8> magic{};reader.raw(magic.data(),magic.size(),"magic");
    const auto version=reader.scalar<uint32_t>("version");
    if(magic!=qwen4_magic || (version!=1 && version!=2 && version!=3))
        throw CudaSessionStateError("unsupported Qwen4 prefix cache payload");
    const auto start=reader.scalar<uint32_t>("start"), count=reader.scalar<uint32_t>("count"),
        layers=reader.scalar<uint32_t>("layers"), flags=reader.scalar<uint32_t>("exact boundary");
    if(version>=3 && (flags>3 || flags==2))throw CudaSessionStateError("invalid Qwen4 checkpoint flags");
    Qwen4PagedHeader header{start,count,layers,flags!=0,version>=3 && (flags&2)!=0,version};
    if(!header.count || !header.layers || header.layers>4096)throw CudaSessionStateError("invalid Qwen4 prefix geometry");
    return header;
}
void write_optional(SessionBlockWriter& writer,const ::tyloi::Tensor& tensor,std::optional<::tyloi::Device> device = {}) {
    writer.scalar<uint32_t>(tensor.defined()?1:0);if(tensor.defined())writer.tensor(tensor,device.value_or(tensor.device()));
}
::tyloi::Tensor read_optional(SessionBlockReader& reader,std::optional<::tyloi::Device>& device) {
    if(!reader.scalar<uint32_t>("tensor presence"))return {};
    auto value=reader.tensor();
    device=reader.tensor_device();
    return std::move(value.first);
}
void write_recurrent(SessionBlockWriter& writer,const tyloi::models::qwen4_exp::LayerSnapshot& layer) {
    write_optional(writer,layer.convolution,layer.devices[3]);write_optional(writer,layer.recurrent,layer.devices[4]);write_optional(writer,layer.ple_convolution,layer.devices[5]);
    writer.scalar<int64_t>(layer.ple_batch);writer.scalar<uint32_t>(uint32_t(layer.ple_tokens.size()));
    for(auto token:layer.ple_tokens)writer.scalar<int64_t>(token);
}
void read_recurrent(SessionBlockReader& reader,tyloi::models::qwen4_exp::LayerSnapshot& layer) {
    layer.convolution=read_optional(reader,layer.devices[3]);layer.recurrent=read_optional(reader,layer.devices[4]);layer.ple_convolution=read_optional(reader,layer.devices[5]);
    layer.ple_batch=reader.scalar<int64_t>("PLE batch");
    const auto count=reader.scalar<uint32_t>("PLE history length");
    if(count>65536)throw CudaSessionStateError("invalid Qwen4 PLE history");
    for(uint32_t i=0;i<count;++i)layer.ple_tokens.push_back(reader.scalar<int64_t>("PLE token"));
}
void write_qwen4_mtp(SessionBlockWriter& writer,const MtpSessionState& state,uint32_t version) {
    const auto* snapshot=std::get_if<tyloi::models::qwen4_exp::MtpSnapshot>(&state.payload);
    if(!snapshot || snapshot->positions.size()!=snapshot->layers.size() ||
        snapshot->lengths.size()!=snapshot->layers.size() ||
        (!snapshot->positions_devices.empty() && snapshot->positions_devices.size()!=snapshot->layers.size()))
        throw CudaSessionStateError("incompatible Qwen4 MTP session state");
    writer.scalar<int64_t>(state.cache_pos);
    write_optional(writer,state.last_target_hidden,snapshot->hidden_device);
    writer.scalar<uint32_t>(uint32_t(snapshot->layers.size()));
    for(std::size_t i=0;i<snapshot->layers.size();++i) {
        const auto& layer=snapshot->layers[i];
        if(version>=3) {
            writer.scalar<int64_t>(snapshot->lengths[i]);writer.scalar<uint32_t>(layer.ram?1:0);
        }
        if(layer.ram) {
            if(version<3 || snapshot->lengths[i]<=0 || layer.ram->tokens!=snapshot->lengths[i] || layer.keys || layer.values || layer.index)
                throw CudaSessionStateError("invalid Qwen4 MTP RAM QSA state");
            write_ram_page(writer,*layer.ram,0,size_t(snapshot->lengths[i]),true);
        } else {
            write_optional(writer,layer.keys,layer.devices[0]);write_optional(writer,layer.values,layer.devices[1]);write_optional(writer,layer.index,layer.devices[2]);
        }
        write_recurrent(writer,layer);write_optional(writer,snapshot->positions[i],snapshot->positions_devices.empty() ? std::nullopt : snapshot->positions_devices[i]);
        if(version<3)writer.scalar<int64_t>(snapshot->lengths[i]);
    }
}
std::size_t qwen4_layer_bytes(const tyloi::models::qwen4_exp::LayerSnapshot& layer) {
    std::size_t bytes=layer.ple_tokens.size()*sizeof(int64_t);
    for(const auto* tensor:{&layer.keys,&layer.values,&layer.index,&layer.convolution,&layer.recurrent,&layer.ple_convolution})
        if(*tensor)bytes+=session_tensor_bytes(*tensor);
    if(layer.ram)bytes+=layer.ram->resident_bytes();
    return bytes;
}
MtpSessionState read_qwen4_mtp(SessionBlockReader& reader,uint32_t version) {
    MtpSessionState state;state.cache_pos=reader.scalar<int64_t>("MTP position");
    std::optional<::tyloi::Device> hidden_device; auto hidden=read_optional(reader,hidden_device);
    if(hidden)state.bytes=session_tensor_bytes(hidden);
    state.last_target_hidden=std::move(hidden);
    const auto count=reader.scalar<uint32_t>("MTP layers");
    if(!count || count>64)throw CudaSessionStateError("invalid Qwen4 MTP layers");
    auto& snapshot=state.payload.emplace<tyloi::models::qwen4_exp::MtpSnapshot>();snapshot.hidden_device=hidden_device;
    for(uint32_t i=0;i<count;++i) {
        tyloi::models::qwen4_exp::LayerSnapshot layer;
        const auto length=version>=3 ? reader.scalar<int64_t>("MTP layer position") : int64_t(0);
        const auto kind=version>=3 ? reader.scalar<uint32_t>("MTP layer kind") : 0;
        if(kind>1)throw CudaSessionStateError("invalid Qwen4 MTP layer kind");
        if(kind) {
            if(length<=0 || uint64_t(length)>SIZE_MAX)throw CudaSessionStateError("invalid Qwen4 MTP RAM QSA length");
            auto ram=std::make_shared<::tyloi::state::qsa::Session>();
            read_ram_page(reader,*ram,0,size_t(length),true);layer.ram=std::move(ram);
        } else {
            layer.keys=read_optional(reader,layer.devices[0]);layer.values=read_optional(reader,layer.devices[1]);layer.index=read_optional(reader,layer.devices[2]);
        }
        read_recurrent(reader,layer);state.bytes+=qwen4_layer_bytes(layer);snapshot.layers.push_back(std::move(layer));
        std::optional<::tyloi::Device> device; auto positions=read_optional(reader,device);if(positions)state.bytes+=session_tensor_bytes(positions);snapshot.positions.push_back(std::move(positions));snapshot.positions_devices.push_back(device);
        snapshot.lengths.push_back(version>=3 ? length : reader.scalar<int64_t>("MTP layer position"));
    }
    // Older DFlash2 snapshots used the Qwen4 envelope for [window, heads, width] rings.
    if(std::all_of(snapshot.layers.begin(),snapshot.layers.end(),[&](const auto& layer) {
        const auto i=std::size_t(&layer-snapshot.layers.data());
        return layer.keys && layer.values && layer.keys.dim()==3 && layer.values.dim()==3 &&
            !layer.ram && !layer.index && !layer.convolution && !layer.recurrent && !layer.ple_convolution &&
            !layer.ple_batch && layer.ple_tokens.empty() && !snapshot.positions[i] &&
            snapshot.lengths[i]==state.cache_pos && layer.devices[0]==layer.devices[1];
    })) {
        std::vector<KvSnapshot> rings;
        for(const auto& layer:snapshot.layers) {
            const auto convert=[](const ::tyloi::Tensor& tensor) {
                return ops::clone(ops::transpose(ops::reshape(tensor,
                    {1,tensor.size(0),tensor.size(1),tensor.size(2)}),1,2));
            };
            KvSnapshot ring{convert(layer.keys),convert(layer.values),layer.keys.size(0),true,layer.devices[0]};
            validate_kv(ring);
            rings.push_back(std::move(ring));
        }
        state.payload=std::move(rings);
    }
    return state;
}

void write_dflash_mtp(SessionBlockWriter& writer,const MtpSessionState& state,uint32_t version) {
    const auto& rings=std::get<std::vector<KvSnapshot>>(state.payload);
    if(rings.empty() || rings.size()>64)throw CudaSessionStateError("invalid DFlash2 session layers");
    MtpSessionState legacy;
    legacy.cache_pos=state.cache_pos;
    legacy.last_target_hidden=state.last_target_hidden;
    auto& snapshot=legacy.payload.emplace<MtpSnapshot>();
    snapshot.hidden_device=rings.front().target_device();
    for(const auto& ring:rings) {
        validate_kv(ring);
        if(!ring.ring || ring.k.size(2)!=ring.capacity)throw CudaSessionStateError("invalid DFlash2 session ring");
        const auto transpose=[](const Tensor& tensor) {
            return ops::reshape(ops::transpose(tensor,1,2),{tensor.size(2),tensor.size(1),tensor.size(3)});
        };
        auto& layer=snapshot.layers.emplace_back();
        layer.keys=transpose(ring.k);
        layer.values=transpose(ring.v);
        layer.devices[0]=layer.devices[1]=ring.destination;
        snapshot.positions.emplace_back();
        snapshot.lengths.push_back(state.cache_pos);
    }
    write_qwen4_mtp(writer,legacy,version);
}
}

SessionBlockPayload encode_block(const TextSessionState& state,size_t block_size,size_t index) {
    const auto* snapshot=std::get_if<tyloi::models::qwen4_exp::Snapshot>(&state.payload);
    if(!block_size || !snapshot || snapshot->layers.empty() || state.cache_pos<=0 || state.tokens.size()!=std::size_t(state.cache_pos) ||
        index>state.tokens.size()/block_size || index*block_size>=state.tokens.size() || state.cache_pos>UINT32_MAX)
        throw std::runtime_error("invalid Qwen4 prefix block boundary");
    const auto start=index*block_size,count=std::min(block_size,state.tokens.size()-start);
    const bool exact=start+count==state.tokens.size();
    const bool dflash=state.mtp && std::holds_alternative<std::vector<KvSnapshot>>(state.mtp->payload);
    const auto* mtp=state.mtp?std::get_if<MtpSnapshot>(&state.mtp->payload):nullptr;
    const auto has_ram=[](const auto& layers) { return std::any_of(layers.begin(),layers.end(),[](const auto& layer){return bool(layer.ram);}); };
    const uint32_t version=has_ram(snapshot->layers) || (mtp && has_ram(mtp->layers)) ? 3 : 2;
    SessionBlockWriter writer;writer.raw(qwen4_magic.data(),qwen4_magic.size());writer.scalar<uint32_t>(version);
    writer.scalar<uint32_t>(uint32_t(start));writer.scalar<uint32_t>(uint32_t(count));
    writer.scalar<uint32_t>(uint32_t(snapshot->layers.size()));
    writer.scalar<uint32_t>(exact ? (version>=3 && state.mtp ? 3 : 1) : 0);
    for(const auto& layer:snapshot->layers) {
        if(version>=3)writer.scalar<uint32_t>(layer.ram?1:0);
        if(layer.ram) {
            if(layer.ram->tokens!=state.cache_pos || layer.keys || layer.values || layer.index)
                throw CudaSessionStateError("invalid Qwen4 RAM QSA state");
            write_ram_page(writer,*layer.ram,start,count,exact);
        } else {
            const std::array<::tyloi::Tensor,3> tensors{layer.keys,layer.values,layer.index};
            for(size_t field=0;field<tensors.size();++field) {
                const auto& tensor=tensors[field];
                if(tensor.defined()) {
                    auto slice=ops::slice(tensor,1,int64_t(start),int64_t(start + count));
                    write_optional(writer,slice,layer.devices[field]);
                } else write_optional(writer,::tyloi::Tensor{});
            }
        }
        if(exact)write_recurrent(writer,layer);
    }
    if(exact) {
        write_optional(writer,snapshot->positions,snapshot->positions_device);writer.scalar<int64_t>(state.decode_position_delta);
        if(version<3)writer.scalar<uint32_t>(state.mtp?1:0);
        if(state.mtp) {
            if(dflash)write_dflash_mtp(writer,*state.mtp,version);
            else write_qwen4_mtp(writer,*state.mtp,version);
        }
    }
    return std::move(writer).finish();
}

bool exact(const SessionBlockPayload& payload) {
    SessionBlockReader reader(*payload);return read_qwen4_header(reader).exact;
}

bool checkpoint_needs_update(const SessionBlockPayload& payload,const TextSessionState& state) {
    if(!payload)return true;
    SessionBlockReader reader(*payload);
    const auto header=read_qwen4_header(reader);
    if(!header.exact)return true;
    if(!state.mtp)return false;
    if(header.version>=3)return !header.mtp;
    const auto skip_optional=[&] {
        const auto present=reader.scalar<uint32_t>("tensor presence");
        if(present>1)throw CudaSessionStateError("invalid Qwen4 tensor presence");
        if(present)reader.skip_tensor();
    };
    for(uint32_t layer=0;layer<header.layers;++layer) {
        for(int field=0;field<6;++field)skip_optional();
        (void)reader.scalar<int64_t>("PLE batch");
        const auto count=reader.scalar<uint32_t>("PLE history length");
        if(count>65536)throw CudaSessionStateError("invalid Qwen4 PLE history");
        for(uint32_t i=0;i<count;++i)(void)reader.scalar<int64_t>("PLE token");
    }
    skip_optional();
    if(header.version>=2)(void)reader.scalar<int64_t>("position delta");
    const auto mtp=reader.scalar<uint32_t>("MTP presence");
    if(mtp>1)throw CudaSessionStateError("invalid Qwen4 predictor presence");
    return !mtp;
}

TextSessionState decode_session(const std::vector<SessionBlockPayload>& payloads,
    const std::vector<int64_t>& tokens,size_t block_size) {
    TextSessionState state;state.tokens=tokens;state.cache_pos=int64_t(tokens.size());
    tyloi::models::qwen4_exp::Snapshot snapshot;
    auto& layers=snapshot.layers;
    std::vector<std::array<std::vector<::tyloi::Tensor>,3>> fragments;
    std::vector<std::shared_ptr<::tyloi::state::qsa::Session>> ram;
    std::vector<uint32_t> kinds;
    size_t position=0;
    for(size_t i=0;i<payloads.size();++i) {
        if(!payloads[i])throw CudaSessionStateError("null Qwen4 prefix payload");
        SessionBlockReader reader(*payloads[i]);const auto header=read_qwen4_header(reader);
        if(header.start!=position || header.count>tokens.size()-position ||
            header.count>block_size || (i+1<payloads.size() && header.count!=block_size) ||
            (i && header.layers!=layers.size()))throw CudaSessionStateError("incompatible Qwen4 prefix chain");
        if(!i){layers.resize(header.layers);fragments.resize(header.layers);ram.resize(header.layers);kinds.resize(header.layers);}
        position+=header.count;
        for(size_t layer=0;layer<layers.size();++layer) {
            const auto kind=header.version>=3 ? reader.scalar<uint32_t>("QSA layer kind") : 0;
            if(kind>1 || (i && kinds[layer]!=kind))throw CudaSessionStateError("inconsistent Qwen4 QSA layer kind");
            kinds[layer]=kind;
            if(kind) {
                if(!ram[layer])ram[layer]=std::make_shared<::tyloi::state::qsa::Session>();
                read_ram_page(reader,*ram[layer],header.start,header.count,header.exact);
            } else {
                for(size_t field=0;field<3;++field) {
                    std::optional<::tyloi::Device> device; auto tensor=read_optional(reader,device);
                    if(tensor && tensor.defined()) {
                        if(tensor.dim()!=3 || tensor.size(0)!=1 || tensor.size(1)!=header.count)
                            throw CudaSessionStateError("invalid Qwen4 QSA prefix geometry");
                        if(!fragments[layer][field].empty() &&
                            (layers[layer].devices[field]!=device || fragments[layer][field].front().scalar_type()!=tensor.scalar_type()))
                            throw CudaSessionStateError("inconsistent Qwen4 QSA prefix tensor");
                        layers[layer].devices[field]=device;
                        fragments[layer][field].push_back(std::move(tensor));
                    }
                }
            }
            if(header.exact) {
                layers[layer].convolution={};layers[layer].recurrent={};layers[layer].ple_convolution={};layers[layer].ple_tokens.clear();
                layers[layer].devices[3].reset();layers[layer].devices[4].reset();layers[layer].devices[5].reset();
                read_recurrent(reader,layers[layer]);
            }
        }
        if(header.exact) {
            snapshot.positions_device.reset(); snapshot.positions=read_optional(reader,snapshot.positions_device);
            state.decode_position_delta=header.version>=2 ? reader.scalar<int64_t>("position delta") : 0;
            state.mtp.reset();
            const auto mtp=header.version>=3 ? uint32_t(header.mtp) : reader.scalar<uint32_t>("MTP presence");
            if(mtp)state.mtp=read_qwen4_mtp(reader,header.version);
        }
        if(i+1==payloads.size() && !header.exact)throw CudaSessionStateError("Qwen4 prefix has no exact recurrent checkpoint");
        reader.expect_end();
    }
    if(payloads.empty() || position!=tokens.size() || !snapshot.positions || !snapshot.positions.defined() || snapshot.positions.size(-1)!=int64_t(position))
        throw CudaSessionStateError("Qwen4 prefix boundary mismatch");
    for(size_t i=0;i<layers.size();++i) {
        auto& layer=layers[i];std::array<::tyloi::Tensor*,3> fields{&layer.keys,&layer.values,&layer.index};
        if(ram[i]) {
            if(ram[i]->tokens!=int64_t(position))throw CudaSessionStateError("incomplete Qwen4 RAM QSA prefix chain");
            layer.ram=std::move(ram[i]);
        }
        for(size_t j=0;j<3;++j)if(!fragments[i][j].empty()) {
            if(fragments[i][j].size()!=payloads.size())throw CudaSessionStateError("incomplete Qwen4 QSA prefix chain");
            *fields[j]=fragments[i][j].size()==1?fragments[i][j].front(): concatenate_session_tensors(fragments[i][j],1);
        }
        state.bytes+=qwen4_layer_bytes(layer);
    }
    state.bytes+=session_tensor_bytes(snapshot.positions)+(state.mtp?state.mtp->bytes:0);
    state.payload=std::move(snapshot);return state;
}

} // namespace tyloi::models::qwen4_exp
