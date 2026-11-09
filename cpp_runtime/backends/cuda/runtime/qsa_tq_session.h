#pragma once
#include "mfq_tensor_backend.h"
#include "tyloinfer/qsa_ram_store.h"
#include <stdexcept>
#include <cstring>
#include <map>

namespace mfq::cuda {
// A serialized model worker owns updates to these small immutable archives.
// Prefix persistence can replace their backing without retaining a RAM copy.
struct QsaIndexArchive {
    struct Slice {const QsaIndexArchive* archive;std::uint8_t* destination;};
    std::shared_ptr<const std::vector<std::uint8_t>> resident;
    std::shared_ptr<mfq::cache::QsaKvStore> store;
    mfq::cache::QsaKvStore::BlockPtr block;
    std::shared_ptr<const tyloinfer::qsa::RamKvReadSource> source;
    std::size_t offset=0,bytes=0;
    std::shared_ptr<const std::vector<std::uint8_t>> read()const {
        if(resident)return resident;
        if(source)return std::make_shared<const std::vector<std::uint8_t>>(source->read(offset,bytes));
        if(store && block)return store->read(block);
        throw std::runtime_error("QSA index archive has no retained backing");
    }
    // Restore the active INT16 suffix directly without copying its retained
    // FP32 companion. Validate the complete archive geometry before any read.
    void read_slice_into(std::size_t at,std::uint8_t* destination,std::size_t count,std::size_t expected_bytes)const {
        if(at>expected_bytes || count>expected_bytes-at)throw std::out_of_range("QSA index slice exceeds archive");
        if(count && !destination)throw std::invalid_argument("QSA index destination is null");
        if(source && !resident) {
            if(bytes!=expected_bytes)throw std::runtime_error("Incompatible QSA snapshot index archive length");
            source->validate(offset,bytes);source->read_into(offset+at,destination,count);return;
        }
        const auto data=read();
        if(data->size()!=expected_bytes)throw std::runtime_error("Incompatible QSA snapshot index archive length");
        if(count)std::memcpy(destination,data->data()+at,count);
    }
    static void read_slices_into(const std::vector<Slice>& slices,std::size_t at,
            std::size_t count,std::size_t expected_bytes) {
        if(at>expected_bytes || count>expected_bytes-at)throw std::out_of_range("QSA index slice exceeds archive");
        std::vector<FileReadRange> files;files.reserve(slices.size());
        std::map<mfq::cache::QsaKvStore*,std::vector<mfq::cache::QsaKvStore::ReadRequest>> stores;
        for(const auto& slice:slices) {
            if(!slice.archive || (count && !slice.destination))throw std::invalid_argument("Invalid QSA index slice");
            const auto& archive=*slice.archive;
            if(archive.resident) {
                if(archive.resident->size()!=expected_bytes)throw std::runtime_error("Incompatible QSA snapshot index archive length");
            }else if(archive.source) {
                if(archive.bytes!=expected_bytes)throw std::runtime_error("Incompatible QSA snapshot index archive length");
                archive.source->validate(archive.offset,archive.bytes);
                if(archive.source->file_reader) {
                    const auto offset=archive.offset+at;
                    if(offset>UINT64_MAX-archive.source->file_offset)throw std::out_of_range("QSA index file offset overflows");
                    files.push_back({archive.source->file_reader,{archive.source->file_offset+offset,
                        reinterpret_cast<std::byte*>(slice.destination),count}});
                }
            }else {
                if(!archive.store || !archive.block)throw std::runtime_error("QSA index archive has no retained backing");
                if(archive.store->read_size(archive.block)!=expected_bytes)throw std::runtime_error("Incompatible QSA snapshot index archive length");
                if(!at && count==expected_bytes)
                    stores[archive.store.get()].push_back({archive.block,slice.destination,count});
            }
        }
        for(const auto& slice:slices) {
            const auto& archive=*slice.archive;
            if(!archive.resident && archive.source && archive.source->file_reader)continue;
            if(!archive.resident && !archive.source && archive.store && !at && count==expected_bytes)continue;
            archive.read_slice_into(at,slice.destination,count,expected_bytes);
        }
        for(const auto& [store,requests]:stores)store->read_into_batch(requests);
        if(!files.empty())FileRangeReader::read_many(files);
    }
    void bind(std::shared_ptr<const tyloinfer::qsa::RamKvReadSource> reader,std::size_t at,std::size_t count) {
        if(!reader)throw std::invalid_argument("QSA index prefix reader is empty");
        reader->validate(at,count);
        source=std::move(reader);offset=at;bytes=count;
        resident.reset();block.reset();store.reset();
    }
};
struct QsaTqSession {
    std::int64_t tokens=0;
    int heads=0,kv_heads=0,width=0,padded=0,index_width=0,ram_bits=0;
    std::shared_ptr<tyloinfer::qsa::RamKvSnapshot> kv;
    std::vector<std::shared_ptr<QsaIndexArchive>> index;
    bool durable_prefix=false,borrow_prefix=false;
    mfq_tensor_backend::Tensor tail_keys,tail_values,tail_index;
    std::size_t resident_bytes()const {
        std::size_t bytes=sizeof(*this)+index.size()*sizeof(QsaIndexArchive);
        if(kv)for(const auto& b:kv->blocks){bytes+=sizeof(b);if(b.image)bytes+=b.image->bytes();}
        for(const auto& b:index)if(b && b->resident)bytes+=b->resident->size();
        for(const auto* t:{&tail_keys,&tail_values,&tail_index})if(t->defined())bytes+=std::size_t(t->numel())*t->element_size();
        return bytes;
    }
};
} // namespace mfq::cuda
