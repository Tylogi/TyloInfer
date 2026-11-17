#include "hf_packed_quant.h"

#include "ggml.h"
#include "tyloinfer/channel_fp8.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <regex>

namespace mfq {
namespace {
constexpr std::string_view suffix=".weight_packed";
bool packed_name(std::string_view name) {
    return name.size()>suffix.size() && name.substr(name.size()-suffix.size())==suffix;
}
bool packed_config(const nlohmann::json& config) {
    return config.is_object() && config.value("quant_method",std::string{})=="compressed-tensors" &&
        config.value("format",std::string{})=="pack-quantized";
}
std::uint32_t u32(const std::byte* bytes) {
    std::uint32_t value=0;
    for(unsigned i=0;i<4;++i)value|=std::uint32_t(std::to_integer<unsigned char>(bytes[i]))<<(8*i);
    return value;
}
std::uint16_t u16(const std::byte* bytes) {
    return std::uint16_t(std::to_integer<unsigned char>(bytes[0])) |
        (std::uint16_t(std::to_integer<unsigned char>(bytes[1]))<<8);
}
float scale_value(const std::byte* bytes,std::string_view dtype) {
    if(dtype=="F16")return ggml_fp16_to_fp32(u16(bytes));
    const auto bits=dtype=="BF16"?std::uint32_t(u16(bytes))<<16:u32(bytes);
    float result;std::memcpy(&result,&bits,sizeof(result));return result;
}
void half(std::byte* bytes,float value) {
    const auto bits=ggml_fp32_to_fp16(value);
    if(!std::isfinite(value) || !std::isfinite(ggml_fp16_to_fp32(bits)))
        throw std::runtime_error("packed INT4 scale exceeds its execution representation");
    bytes[0]=std::byte(bits);bytes[1]=std::byte(bits>>8);
}
std::vector<std::int64_t> shape_of(const HfSafetensorsSource& source,const std::string& name) {
    const auto* record=source.find_tensor(name);
    if(!record || record->dtype!="I64" || record->shape!=std::optional<std::vector<std::int64_t>>{{2}} || record->nbytes!=16)
        throw std::invalid_argument("packed INT4 weight_shape must contain two int64 dimensions");
    const auto bytes=source.read(name);
    std::vector<std::int64_t> result;
    for(unsigned d=0;d<2;++d) {
        std::uint64_t value=0;
        for(unsigned i=0;i<8;++i)value|=std::uint64_t(std::to_integer<unsigned char>(bytes[d*8+i]))<<(8*i);
        if(!value || value>std::uint64_t(INT64_MAX))throw std::invalid_argument("invalid packed INT4 shape");
        result.push_back(std::int64_t(value));
    }
    return result;
}
}

std::string hf_packed_tensor_name(std::string name,const nlohmann::json& config) {
    if(packed_config(config) && packed_name(name))name.replace(name.size()-suffix.size(),suffix.size(),".weight");
    return name;
}

std::optional<HfPackedQuantBinding> hf_packed_quant_binding(const HfSafetensorsSource& source,
        const nlohmann::json& config,const TensorMetadata& weight) {
    if(!packed_name(weight.name))return {};
    if(!packed_config(config) || weight.dtype!="I32")
        throw std::invalid_argument("unsupported packed HF quantization contract");
    const auto module=weight.name.substr(0,weight.name.size()-suffix.size());
    const nlohmann::json* scheme=nullptr;
    for(const auto& group:config.at("config_groups")) {
        bool match=false;
        for(const auto& value:group.at("targets")) {
            const auto target=value.get<std::string>();
            match|=target=="Linear" || target==module ||
                (target.rfind("re:",0)==0 && std::regex_match(module,std::regex(target.substr(3))));
        }
        if(match) {
            if(scheme)throw std::invalid_argument("packed HF weight matches multiple quantization groups");
            scheme=&group.at("weights");
        }
    }
    if(!scheme || scheme->value("num_bits",0)!=4 || scheme->value("type",std::string{})!="int" ||
        scheme->value("strategy",std::string{})!="group" ||
        (scheme->contains("actorder") && !(*scheme)["actorder"].is_null()))
        throw std::invalid_argument("packed HF execution requires grouped INT4 without activation reordering");
    const auto group=scheme->at("group_size").get<std::int64_t>();
    const auto shape_name=module+".weight_shape",scale_name=module+".weight_scale",zero_name=module+".weight_zero_point";
    const auto shape=shape_of(source,shape_name);
    const auto rows=std::uint64_t(shape[0]),width=std::uint64_t(shape[1]);
    if(group<32 || group%32 || width%std::uint64_t(group) || !weight.shape ||
        *weight.shape!=std::vector<std::int64_t>{shape[0],std::int64_t(width/8)} ||
        rows>UINT64_MAX/width || weight.nbytes!=rows*width/2)
        throw std::invalid_argument("invalid grouped packed INT4 matrix geometry");
    const auto groups=width/std::uint64_t(group),blocks=width/32;
    const auto* scale=source.find_tensor(scale_name);
    if(!scale || !scale->shape || *scale->shape!=std::vector<std::int64_t>{shape[0],std::int64_t(groups)} ||
        (scale->dtype!="F32" && scale->dtype!="F16" && scale->dtype!="BF16"))
        throw std::invalid_argument("packed INT4 weight_scale has incompatible geometry or dtype");
    const auto scale_bytes=scale->dtype=="F32"?4u:2u;
    if(scale->nbytes!=rows*groups*scale_bytes)throw std::invalid_argument("packed INT4 scale byte count differs");
    const auto* zero=source.find_tensor(zero_name);
    const bool symmetric=scheme->value("symmetric",true);
    if((!zero && !symmetric) || (zero && (zero->dtype!="I32" || !zero->shape ||
        *zero->shape!=std::vector<std::int64_t>{std::int64_t((rows+7)/8),std::int64_t(groups)} ||
        zero->nbytes!=(rows+7)/8*groups*4)))
        throw std::invalid_argument("packed INT4 zero-point has incompatible geometry");
    if(rows>UINT64_MAX/blocks/20)throw std::overflow_error("packed INT4 execution image is too large");
    HfPackedQuantBinding result;
    result.tensor={hf_packed_tensor_name(weight.name,config),"GGUF_Q4_1","GGUF_Q4_1",shape,rows*blocks*20};
    result.auxiliaries={shape_name,scale_name};if(zero)result.auxiliaries.push_back(zero_name);
    const auto dtype=scale->dtype,weight_name=weight.name;
    const auto bytes=result.tensor.nbytes;
    result.read=[&source,rows,width,groups,blocks,group,scale_bytes,dtype,weight_name,scale_name,zero_name,
                 have_zero=bool(zero),bytes](std::uint64_t offset,std::byte* destination,std::size_t count) {
        if(offset>bytes || count>bytes-offset || (count && !destination))
            throw std::out_of_range("packed INT4 execution read is out of bounds");
        if(!count)return;
        const auto first=offset/20,last=(offset+count+19)/20;
        const auto tile_rows=std::max<std::uint64_t>(1,(8u<<20)/(width/2+groups*scale_bytes));
        for(auto row=first/blocks;row<=(last-1)/blocks;) {
            const auto end=std::min((last-1)/blocks+1,row+tile_rows),n=end-row;
            const auto packed=source.read(weight_name,row*width/2,std::size_t(n*width/2));
            const auto scales=source.read(scale_name,row*groups*scale_bytes,std::size_t(n*groups*scale_bytes));
            const auto zero_row=row/8,zero_end=(end+7)/8;
            const auto zeros=have_zero?source.read(zero_name,zero_row*groups*4,std::size_t((zero_end-zero_row)*groups*4)):
                std::vector<std::byte>{};
            for(auto r=row;r<end;++r)for(auto b=std::max(first,r*blocks);b<std::min(last,(r+1)*blocks);++b) {
                const auto column=(b-r*blocks)*32,g=column/std::uint64_t(group);
                const auto d=scale_value(scales.data()+((r-row)*groups+g)*scale_bytes,dtype);
                if(!(d>=0))throw std::invalid_argument("packed INT4 scale must be finite and nonnegative");
                const auto z=have_zero?(u32(zeros.data()+((r/8-zero_row)*groups+g)*4)>>(4*(r%8)))&15u:8u;
                std::array<std::byte,20> block{};half(block.data(),d);half(block.data()+2,-d*float(z));
                const auto* values=packed.data()+(r-row)*width/2+column/2;
                for(unsigned i=0;i<16;++i) {
                    const auto q0=(std::to_integer<unsigned char>(values[i/2])>>(4*(i%2)))&15u;
                    const auto q1=(std::to_integer<unsigned char>(values[8+i/2])>>(4*(i%2)))&15u;
                    block[4+i]=std::byte(q0|(q1<<4));
                }
                const auto begin=std::max(offset,b*20),stop=std::min(offset+count,(b+1)*20);
                std::copy_n(block.data()+begin-b*20,std::size_t(stop-begin),destination+begin-offset);
            }
            row=end;
        }
    };
    return result;
}

std::optional<HfPackedQuantBinding> hf_channel_fp8_binding(const HfSafetensorsSource& source,
        const nlohmann::json& config,const TensorMetadata& weight) {
    if ((weight.dtype != "F8_E4M3" && weight.dtype != "F8_E4M3FN") ||
        !config.is_object() || config.value("format",std::string{}) != "float-quantized") return {};
    if (config.value("quant_method",std::string{}) != "compressed-tensors" ||
        weight.name.size() < 7 || weight.name.substr(weight.name.size()-7) != ".weight")
        throw std::invalid_argument("unsupported channel FP8 quantization contract");
    const auto module = weight.name.substr(0,weight.name.size()-7);
    const nlohmann::json* group = nullptr;
    for (const auto& candidate : config.at("config_groups")) {
        bool match = false;
        for (const auto& value : candidate.at("targets")) {
            const auto target = value.get<std::string>();
            match |= target == "Linear" || target == module ||
                (target.rfind("re:",0)==0 && std::regex_match(module,std::regex(target.substr(3))));
        }
        if (match) {
            if (group) throw std::invalid_argument("FP8 weight matches multiple quantization groups");
            group = &candidate;
        }
    }
    if (!group) throw std::invalid_argument("FP8 weight has no matching quantization group");
    const auto& weights = group->at("weights");
    if (weights.value("num_bits",0)!=8 || weights.value("type",std::string{})!="float" ||
        weights.value("strategy",std::string{})!="channel" || !weights.value("symmetric",false) ||
        weights.value("dynamic",false) || !weight.shape || weight.shape->size()!=2)
        throw std::invalid_argument("unsupported FP8 weight scale scheme");
    bool dynamic = false;
    if (group->contains("input_activations") && !group->at("input_activations").is_null()) {
        const auto& input = group->at("input_activations");
        if (input.value("num_bits",0)!=8 || input.value("type",std::string{})!="float" ||
            input.value("strategy",std::string{})!="token" || !input.value("symmetric",false) ||
            !input.value("dynamic",false))
            throw std::invalid_argument("unsupported FP8 input activation scheme");
        dynamic = true;
    }
    if (group->contains("output_activations") && !group->at("output_activations").is_null())
        throw std::invalid_argument("FP8 output activation quantization is unsupported");
    const auto scale_name = module + ".weight_scale";
    const auto* scale = source.find_tensor(scale_name);
    const auto rows = weight.shape->at(0), width = weight.shape->at(1);
    if (!scale || !scale->shape ||
        (*scale->shape != std::vector<std::int64_t>{rows} && *scale->shape != std::vector<std::int64_t>{rows,1}))
        throw std::invalid_argument("channel FP8 scale shape must match output channels");
    using namespace channel_fp8;
    const auto kind = scale->dtype=="BF16" ? ScaleKind::Bf16 : scale->dtype=="F16" ? ScaleKind::F16 :
        scale->dtype=="F32" ? ScaleKind::F32 : throw std::invalid_argument("unsupported channel FP8 scale dtype");
    const auto image = layout(rows,width,kind,dynamic);
    if (weight.nbytes!=std::uint64_t(rows)*width || scale->nbytes!=std::uint64_t(rows)*scale_bytes(kind))
        throw std::invalid_argument("channel FP8 source size mismatch");
    HfPackedQuantBinding result;
    result.tensor = {weight.name,std::string(dtype),weight.dtype,weight.shape,image.bytes};
    result.auxiliaries = {scale_name};
    result.read = [&source,image,preamble=prefix(image),name=weight.name,scale_name](
            std::uint64_t offset,std::byte* destination,std::size_t count) {
        if (offset>image.bytes || count>image.bytes-offset || (count && !destination))
            throw std::out_of_range("channel FP8 tensor read is out of bounds");
        const auto stop = offset+count;
        const auto copy = [&](std::uint64_t first,std::uint64_t last,const std::string& tensor) {
            const auto begin = std::max(offset,first), end = std::min(stop,last);
            if (begin>=end) return;
            if (tensor.empty()) std::copy_n(preamble.data()+begin-first,end-begin,destination+begin-offset);
            else source.read_range_into(tensor,begin-first,destination+begin-offset,end-begin);
        };
        copy(0,header_bytes,{}); copy(header_bytes,image.scales,name); copy(image.scales,image.bytes,scale_name);
    };
    return result;
}

} // namespace mfq
