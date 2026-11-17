#include "tyloinfer/tq_model_layout.h"
#include "tyloinfer/tq_rows.h"
#include "nlohmann/json.hpp"
#include <cstring>

namespace mfq::tq_source {
std::vector<std::int64_t> logical_shape(const mfq::ModelSource& source,const std::string& name) {
    const auto* record=source.find_tensor(name);
    if(record && record->dtype!="TQ")throw std::invalid_argument("expected TQ tensor: "+name);
    std::vector<std::int64_t> shape;
    bool found=false;
    for(const auto* key:{"tq_shapes","tensor_shapes"}) {
        const auto metadata=source.metadata().find(key);
        if(metadata==source.metadata().end())continue;
        const auto shapes=nlohmann::json::parse(metadata->second);
        if(!shapes.is_object())throw std::invalid_argument("invalid TQ shape metadata");
        if(shapes.contains(name)) {
            shape=shapes.at(name).get<std::vector<std::int64_t>>();found=true;break;
        }
    }
    if(!found && record && record->shape)shape=*record->shape;
    if(shape.empty() && record)
        return mfq::tq_tensor_shape(record->nbytes,source.tensor_reader(name));
    if(shape.size()!=2 && shape.size()!=3)
        throw std::invalid_argument("TQ logical tensor shape must have rank 2 or 3: "+name);
    std::uint64_t count=1;
    for(const auto dimension:shape) {
        if(dimension<1 || count>UINT64_MAX/std::uint64_t(dimension))
            throw std::invalid_argument("invalid TQ logical tensor shape: "+name);
        count*=dimension;
    }
    return shape;
}
std::vector<std::string> row_parts(const mfq::ModelSource& source,const std::string& name) {
    for(const auto* key:{"tq_row_shards","tq.segments"}) {
        const auto entry=source.metadata().find(key);
        if(entry==source.metadata().end())continue;
        const auto tables=nlohmann::json::parse(entry->second);
        if(!tables.is_object())throw std::invalid_argument("invalid TQ row-shard manifest");
        if(!tables.contains(name))continue;
        if(std::string_view(key)=="tq_row_shards" && source.find_tensor(name))
            throw std::invalid_argument("ambiguous TQ row table: "+name);
        const auto names=tables.at(name).get<std::vector<std::string>>();
        if(names.empty())throw std::invalid_argument("empty TQ row table: "+name);
        return names;
    }
    return {name};
}
std::vector<std::uint8_t> join_rows(const std::vector<std::vector<std::uint8_t>>& blobs) {
    if(blobs.empty())throw std::invalid_argument("empty TQ row selection");
    const mfq::TqLayout first(blobs[0].data(),blobs[0].size());
    std::uint64_t rows=0,packed_bytes=0;
    for(const auto& blob:blobs) {
        const mfq::TqLayout layout(blob.data(),blob.size());
        if(layout.width!=first.width || layout.columns!=first.columns || layout.gs!=first.gs)
            throw std::invalid_argument("incompatible TQ row segments");
        if(first.columns && std::memcmp(blobs[0].data()+first.scales,blob.data()+layout.scales,first.scale_bytes))
            throw std::invalid_argument("TQ row segments have different column scales");
        rows+=layout.rows;packed_bytes+=layout.packed_bytes;
    }
    if(rows>UINT32_MAX || packed_bytes>UINT32_MAX)throw std::overflow_error("selected TQ rows exceed RTRELL05 limits");
    const auto scale_bytes=first.columns?first.scale_bytes:(rows*first.groups+1)/2;
    const auto scale_offset=64+rows*8,packed_offset=scale_offset+scale_bytes;
    std::vector<std::uint8_t> result(packed_offset+packed_bytes);
    std::memcpy(result.data(),blobs[0].data(),64);
    mfq::TqLayout::integer(result.data()+12,rows,4);
    mfq::TqLayout::integer(result.data()+28,packed_bytes,4);
    std::uint64_t row_offset=0,byte_offset=0;
    if(first.columns)std::memcpy(result.data()+scale_offset,blobs[0].data()+first.scales,scale_bytes);
    for(const auto& blob:blobs) {
        const mfq::TqLayout m(blob.data(),blob.size());
        for(std::size_t row=0;row<m.rows;++row) {
            auto descriptor=mfq::TqLayout::integer(blob.data()+64+row*8,8);
            descriptor=m.rebase(descriptor,std::uint64_t(m.offset(descriptor))+byte_offset/4);
            mfq::TqLayout::integer(result.data()+64+(row_offset+row)*8,descriptor,8);
        }
        if(!m.columns)for(std::size_t i=0;i<std::size_t(m.rows)*m.groups;++i) {
            const auto selector=(blob[m.scales+i/2]>>((i&1)*4))&15;
            const auto index=row_offset*m.groups+i;
            result[scale_offset+index/2]|=selector<<((index&1)*4);
        }
        std::memcpy(result.data()+packed_offset+byte_offset,blob.data()+m.packed,m.packed_bytes);
        row_offset+=m.rows;byte_offset+=m.packed_bytes;
    }
    return result;
}
std::vector<std::uint8_t> tq_image(const std::vector<std::uint8_t>& blob,const mfq::TqLayout& m) {
    // The serialized scale stream can end at an odd byte. Align the packed
    // word stream in one compressed image instead of uploading fields apart.
    const auto words_offset=(m.packed+3)&~std::size_t(3);
    const auto relative_offset=words_offset+m.packed_bytes;
    std::vector<std::uint8_t> image(relative_offset+32);
    std::memcpy(image.data(),blob.data(),m.packed);
    std::memcpy(image.data()+words_offset,blob.data()+m.packed,m.packed_bytes);
    // Exact FP16 RELATIVE table from the TQ format, independent of device.
    constexpr std::uint16_t relative[]{0x3400,0x34d0,0x35ca,0x36f7,0x3830,0x390a,0x3a10,0x3b4b,
        0x3c63,0x3d47,0x3e59,0x3fa4,0x4098,0x4187,0x42a6,0x4400};
    std::memcpy(image.data()+relative_offset,relative,sizeof(relative));
    return image;
}

bool is_row_table(const mfq::ModelSource& source,const std::string& name) {
    const auto* record=source.find_tensor(name);
    for(const auto* key:{"tq_row_shards","tq.segments"}) {
        const auto entry=source.metadata().find(key);
        if(entry!=source.metadata().end()) {
            const auto tables=nlohmann::json::parse(entry->second);
            if(!tables.is_object())throw std::invalid_argument("invalid TQ row-shard manifest");
            if(tables.contains(name)) {
                if(std::string_view(key)=="tq_row_shards" && record)
                    throw std::invalid_argument("ambiguous TQ row table: "+name);
                return true;
            }
        }
    }
    return record && record->dtype=="TQ";
}
}
