#include "tyloinfer/tq_rows.h"
#include "tyloinfer/tq_model_layout.h"
#include "tyloinfer/encoded_row_cache.h"
#include <atomic>
#include <thread>
#include <future>
#include "tyloinfer/format_features.h"
#include "tyloinfer/tensor_dtype.h"

#include <cassert>
#include <iostream>

namespace {
using Bytes = std::vector<std::uint8_t>;

Bytes tensor(bool columns, std::uint32_t salt, unsigned width = 256, unsigned gs = 16) {
    constexpr unsigned rows = 3;
    const unsigned bits[]{1, 3, 8};
    const auto scale_bytes = columns ? width * 2 : (rows * ((width + gs - 1) / gs) + 1) / 2;
    const auto packed_bytes = width * (bits[0] + bits[1] + bits[2]) / 8;
    Bytes result(64 + rows * 8 + scale_bytes + packed_bytes, 0);
    std::memcpy(result.data(), "RTRELL05", 8);
    for (auto value : std::vector<std::pair<unsigned, unsigned>>{
            {8, 5}, {12, rows}, {16, width}, {20, unsigned(columns)}, {24, gs}, {28, packed_bytes}})
        mfq::TqLayout::integer(result.data() + value.first, value.second, 4);
    unsigned offset = 0;
    for (unsigned row = 0; row < rows; ++row) {
        mfq::TqLayout::integer(result.data() + 64 + row * 8,
            offset | (std::uint64_t(bits[row] - 1) << 32) | (std::uint64_t(0x3c00) << 40), 8);
        offset += width * bits[row] / 32;
    }
    auto* scale = result.data() + 64 + rows * 8;
    if (columns) for (unsigned col = 0; col < width; ++col)
        mfq::TqLayout::integer(scale + col * 2, 0x3800 + salt, 2);
    else std::fill(scale, scale + scale_bytes, 0x89);
    for (unsigned i = 0; i < packed_bytes; ++i) scale[scale_bytes + i] = std::uint8_t(i + salt);
    return result;
}

Bytes bank(bool columns) {
    auto a = tensor(columns, 1), b = tensor(columns, 2);
    Bytes result(96 + a.size() + b.size());
    std::memcpy(result.data(), "RTQEXP01", 8);
    for (auto value : std::vector<std::pair<unsigned, unsigned>>{{8, 1}, {12, 2}, {16, 3}, {20, 251}, {24, 16}})
        mfq::TqLayout::integer(result.data() + value.first, value.second, 4);
    mfq::TqLayout::integer(result.data() + 64, 96, 8);
    mfq::TqLayout::integer(result.data() + 72, a.size(), 8);
    mfq::TqLayout::integer(result.data() + 80, 96 + a.size(), 8);
    mfq::TqLayout::integer(result.data() + 88, b.size(), 8);
    std::copy(a.begin(), a.end(), result.begin() + 96);
    std::copy(b.begin(), b.end(), result.begin() + 96 + a.size());
    return result;
}

mfq::ModelSource::TensorReader reader(const Bytes& value, std::size_t& read_bytes) {
    return [&value, &read_bytes](std::uint64_t offset, std::byte* target, std::size_t size) {
        if (offset > value.size() || size > value.size() - offset) throw std::out_of_range("test read");
        read_bytes += size;
        std::memcpy(target, value.data() + offset, size);
    };
}

void check(bool columns) {
    const auto value = bank(columns);
    std::size_t bytes = 0;
    auto parts = mfq::tq_row_segments(value.size(), reader(value, bytes), 251);
    assert(parts.size() == 2 && bytes == 64 + 32 + 2 * 64);
    assert(parts[0]->layout().rows == 3 && parts[1]->layout().width == 256);
    assert((mfq::tq_tensor_shape(value.size(), reader(value, bytes)) ==
        std::vector<std::int64_t>{2, 3, 251}));
    const auto flat = tensor(columns, 3);
    assert((mfq::tq_tensor_shape(flat.size(), reader(flat, bytes), {1, 3, 251}) ==
        std::vector<std::int64_t>{1, 3, 251}));
    for (const auto& shape : std::vector<std::vector<std::int64_t>>{
            {3, 251}, {2, 2, 251}, {2, 3, 250}, {2, 3, 0}, {2, -3, 251}, {2, 3, 1, 251}}) {
        bool rejected = false;
        try { (void)mfq::tq_tensor_shape(value.size(), reader(value, bytes), shape); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    for (unsigned expert = 0; expert < 2; ++expert) {
        const auto expected = tensor(columns, expert + 1);
        assert(parts[expert]->slice_rows(0, 3) == expected);
        auto partial = parts[expert]->slice_rows(1, 3);
        mfq::TqLayout layout(partial.data(), partial.size());
        assert(layout.rows == 2 && mfq::TqLayout::integer(partial.data() + 64, 4) == 0);
        assert(mfq::TqLayout::integer(partial.data() + 72, 4) == 24);
        const mfq::TqLayout original(expected.data(), expected.size());
        assert(std::equal(partial.begin() + layout.packed, partial.end(), expected.begin() + original.packed + 32));
    }
    for (int mutation = 0; mutation < 7; ++mutation) {
        auto bad = value;
        if (mutation == 0) bad[8] = 2;
        if (mutation == 1) bad[24] = 8;
        if (mutation == 2) bad[32] = 1;
        if (mutation == 3) bad[64] = 95;
        if (mutation == 4) bad.pop_back();
        if (mutation == 5) bad[96 + 12] = 4;
        if (mutation == 6) bad[20] = 250;
        bool refused = false;
        try { (void)mfq::tq_row_segments(bad.size(), reader(bad, bytes), 251); }
        catch (const std::exception&) { refused = true; }
        assert(refused);
    }
}

void check_columns(bool columns, unsigned gs) {
    auto value = tensor(columns, 3, 1024, gs);
    const mfq::TqLayout original(value.data(), value.size());
    if (columns) {
        for (unsigned col = 0; col < original.width; ++col)
            mfq::TqLayout::integer(value.data() + original.scales + col * 2, 0x3800 + col, 2);
    } else {
        for (std::size_t group = 0; group < original.rows * original.groups; ++group) {
            auto& byte = value[original.scales + group / 2];
            const unsigned shift = (group & 1) * 4;
            byte = (byte & ~(15u << shift)) | (((group * 7 + group / 3) & 15) << shift);
        }
    }
    std::size_t read_bytes = 0;
    const mfq::TqRows source(value.size(), reader(value, read_bytes));
    for (const auto [begin, end] : {std::pair<unsigned, unsigned>{0, 256}, {256, 768}, {768, 1024}, {0, 1024}}) {
        const auto part = source.slice_columns(begin, end);
        const mfq::TqLayout layout(part.data(), part.size());
        assert(layout.rows == original.rows && layout.width == end - begin && layout.columns == columns);
        std::uint64_t packed_end = 0;
        for (unsigned row = 0; row < layout.rows; ++row) {
            const auto descriptor = mfq::TqLayout::integer(value.data() + 64 + row * 8, 8);
            const auto sliced = mfq::TqLayout::integer(part.data() + 64 + row * 8, 8);
            const auto bits = ((descriptor >> 32) & 7) + 1;
            assert((descriptor >> 32) == (sliced >> 32) && std::uint32_t(sliced) * 4 == packed_end);
            const auto bytes = (end - begin) * bits / 8;
            assert(std::equal(part.begin() + layout.packed + packed_end,
                part.begin() + layout.packed + packed_end + bytes,
                value.begin() + original.packed + std::uint32_t(descriptor) * 4 + begin * bits / 8));
            packed_end += bytes;
            for (unsigned col = 0; col < layout.width; ++col) {
                if (columns) {
                    assert(mfq::TqLayout::integer(part.data() + layout.scales + col * 2, 2) ==
                        mfq::TqLayout::integer(value.data() + original.scales + (begin + col) * 2, 2));
                } else {
                    const auto from = row * original.groups + (begin + col) / original.gs;
                    const auto to = row * layout.groups + col / layout.gs;
                    assert(((part[layout.scales + to / 2] >> ((to & 1) * 4)) & 15) ==
                        ((value[original.scales + from / 2] >> ((from & 1) * 4)) & 15));
                }
            }
        }
        assert(packed_end == layout.packed_bytes);
    }
    for (const auto [begin, end] : {std::pair<unsigned, unsigned>{1, 256}, {0, 257}, {256, 256}, {0, 1280}}) {
        bool rejected = false;
        try { (void)source.slice_columns(begin, end); }
        catch (const std::out_of_range&) { rejected = true; }
        assert(rejected);
    }
    value[64] = 1;
    bool rejected = false;
    try { (void)source.slice_columns(256, 512); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

void check_cached_metadata(bool columns, unsigned gs,unsigned width=256,bool compact=false) {
    auto value=tensor(columns,7,width,gs);
    if(compact) {
        std::memcpy(value.data(),"RTRELL06",8);mfq::TqLayout::integer(value.data()+8,6,4);
        for(unsigned row=0;row<3;++row) {
            auto descriptor=mfq::TqLayout::integer(value.data()+64+row*8,8);
            const auto bits=((descriptor>>32)&7)+1;
            const auto seed=(0x1234u+row)&((1u<<(16-bits))-1);
            descriptor|=std::uint64_t(seed&31)<<35;
            descriptor|=std::uint64_t((seed>>5)&255)<<56;
            descriptor|=(seed>>13)&1;
            descriptor|=std::uint64_t((seed>>14)&1)<<31;
            mfq::TqLayout::integer(value.data()+64+row*8,descriptor,8);
        }
    }
    const mfq::TqLayout original(value.data(),value.size());
    std::size_t read_bytes=0;
    mfq::TqRows source(value.size(),reader(value,read_bytes));
    source.cache_metadata();
    const auto before=read_bytes;
    source.cache_metadata();assert(read_bytes==before);
    for(unsigned begin=0;begin<original.rows;++begin)for(unsigned end=begin+1;end<=original.rows;++end) {
        const auto prior=read_bytes;
        const auto selected=source.slice_rows(begin,end);
        const mfq::TqLayout layout(selected.data(),selected.size());
        assert(read_bytes-prior==layout.packed_bytes);
        if(columns)assert(std::equal(selected.begin()+layout.scales,selected.begin()+layout.packed,
            value.begin()+original.scales));
        else for(std::size_t i=0;i<std::size_t(end-begin)*original.groups;++i) {
            const auto from=std::size_t(begin)*original.groups+i;
            assert(((selected[layout.scales+i/2]>>((i&1)*4))&15)==
                ((value[original.scales+from/2]>>((from&1)*4))&15));
        }
        const auto offset=original.offset(mfq::TqLayout::integer(value.data()+64+begin*8,8))*4;
        assert(std::equal(selected.begin()+layout.packed,selected.end(),value.begin()+original.packed+offset));
        Bytes row(72+original.scale_bytes+original.packed_bytes);
        const auto prior_row=read_bytes;
        const auto count=source.slice_row_into(begin,row.data(),row.size());
        const mfq::TqLayout row_layout(row.data(),count);
        assert(read_bytes-prior_row==row_layout.packed_bytes);
        Bytes staged(count+35,0xa5);
        const auto prior_stage=read_bytes;
        const auto image=source.stage_row_image(begin,staged.data(),staged.size());
        assert(read_bytes-prior_stage==row_layout.packed_bytes);
        assert(image.packed_offset==((row_layout.packed+3)&~std::size_t(3)));
        assert(image.relative_offset==image.packed_offset+row_layout.packed_bytes);
        assert(image.bytes==image.relative_offset+32 && image.bytes<=staged.size());
        assert(std::equal(row.begin(),row.begin()+row_layout.packed,staged.begin()));
        assert(std::equal(row.begin()+row_layout.packed,row.begin()+count,staged.begin()+image.packed_offset));
        for(auto i=row_layout.packed;i<image.packed_offset;++i)assert(staged[i]==0);
        bool overflow=false;
        try {source.stage_row_image(begin,staged.data(),image.bytes-1);}
        catch(const std::overflow_error&){overflow=true;}
        assert(overflow);
    }
}
}

static void check_encoded_row_cache() {
    using Value=std::vector<std::uint8_t>;
    std::atomic<std::size_t> bytes{0};
    std::shared_ptr<const Value> retained;
    const auto account=[&](std::size_t count,bool insert) {
        if(insert)bytes.fetch_add(count);else bytes.fetch_sub(count);
    };
    {
        mfq::EncodedRowCache<Value> cache(268,[](const Value& value){return value.size();},account);
        cache.insert(1,Value(6,1));cache.insert(2,Value(6,2));
        auto second=cache.find(2);retained=cache.find(1);
        cache.insert(3,Value(6,3));
        assert(!cache.find(2) && cache.find(1) && cache.find(3) && bytes.load()==18);
        second.reset();assert(bytes.load()==12);
        cache.insert(1,Value(6,9));assert(cache.find(1)->front()==1);
        cache.insert(4,Value(141,4));assert(!cache.find(4) && cache.find(3));
    }
    assert(bytes.load()==6 && retained->front()==1);
    retained.reset();assert(bytes.load()==0);
    {
        mfq::EncodedRowCache<Value> cache(3*134,[](const Value& value){return value.size();},account);
        std::vector<std::thread> threads;
        for(int worker=0;worker<4;++worker)threads.emplace_back([&,worker] {
            for(int i=0;i<1000;++i) {
                const auto key=std::uint8_t((i+worker)%8);
                cache.insert(key,Value(6,key));
                if(const auto value=cache.find(key))assert(value->size()==6 && value->front()==key);
            }
        });
        for(auto& thread:threads)thread.join();
        assert(bytes.load()<=18);
    }
    assert(bytes.load()==0);
    mfq::EncodedRowCache<Value> overflow(SIZE_MAX,[](const Value&){return SIZE_MAX;},account);
    overflow.insert(0,{});assert(!overflow.find(0) && bytes.load()==0);
}

class LayoutSource final:public mfq::ModelSource {
    std::shared_ptr<Bytes> bytes_;
    std::vector<std::filesystem::path> paths_;
    std::vector<std::string> assets_;
public:
    std::unordered_map<std::string,std::string> properties;
    std::vector<mfq::TensorMetadata> records;
    explicit LayoutSource(Bytes bytes):bytes_(std::make_shared<Bytes>(std::move(bytes))) {
        mfq::TensorMetadata record;record.name="weight";record.dtype=record.stored_dtype="TQ";
        record.nbytes=bytes_->size();records.push_back(record);
    }
    const std::vector<std::filesystem::path>& source_paths()const noexcept override{return paths_;}
    std::string_view architecture()const noexcept override{return "layout-test";}
    const std::unordered_map<std::string,std::string>& metadata()const noexcept override{return properties;}
    const std::vector<mfq::TensorMetadata>& tensors()const noexcept override{return records;}
    const mfq::TensorMetadata* find_tensor(std::string_view name)const noexcept override {
        for(const auto& record:records)if(record.name==name)return &record;return nullptr;
    }
    TensorReader tensor_reader(std::string_view name)const override {
        if(!find_tensor(name))throw std::out_of_range("unknown layout-test tensor");
        return [bytes=bytes_](std::uint64_t offset,std::byte* output,std::size_t count) {
            if(offset>bytes->size() || count>bytes->size()-offset || (count && !output))
                throw std::out_of_range("layout-test tensor range");
            if(count)std::memcpy(output,bytes->data()+offset,count);
        };
    }
    void read_range_into(std::string_view name,std::uint64_t offset,std::byte* output,std::size_t count)const override {
        tensor_reader(name)(offset,output,count);
    }
    const std::vector<std::string>& assets()const noexcept override{return assets_;}
    bool has_asset(std::string_view)const noexcept override{return false;}
    std::vector<std::byte> read_asset(std::string_view)const override{throw std::out_of_range("no layout-test assets");}
    std::optional<mfq::ModelGraph> model_graph()const override{return {};}
};
void check_model_layout() {
    using namespace mfq::tq_source;
    LayoutSource matrix(tensor(false,1)),experts(bank(false));
    assert((logical_shape(matrix,"weight")==std::vector<std::int64_t>{3,256}));
    assert((logical_shape(experts,"weight")==std::vector<std::int64_t>{2,3,251}));
    matrix.properties["tq_shapes"]=R"({"weight":[3,251],"logical":[3,251]})";
    assert((logical_shape(matrix,"weight")==std::vector<std::int64_t>{3,251}));
    matrix.properties["tq_row_shards"]=R"({"logical":["weight"]})";
    assert(is_row_table(matrix,"logical") && is_row_table(matrix,"weight"));
    assert((row_parts(matrix,"logical")==std::vector<std::string>{"weight"}));
    assert((logical_shape(matrix,"logical")==std::vector<std::int64_t>{3,251}));
    const auto rejects=[](auto&& work) {
        bool failed=false;try{work();}catch(const std::exception&){failed=true;}assert(failed);
    };
    for(const auto* shape:{"[3]","[3,0]","[-1,3]","[1,2,3,4]","[9223372036854775807,9223372036854775807]"}) {
        matrix.properties["tq_shapes"]=std::string("{\"weight\":")+shape+"}";
        rejects([&]{(void)logical_shape(matrix,"weight");});
    }
    matrix.properties["tq_row_shards"]=R"({"weight":["weight"]})";
    rejects([&]{(void)is_row_table(matrix,"weight");});
    rejects([&]{(void)row_parts(matrix,"weight");});
    matrix.properties["tq_row_shards"]="[]";
    rejects([&]{(void)is_row_table(matrix,"missing");});
    matrix.properties["tq_row_shards"]=R"({"logical":[]})";
    rejects([&]{(void)row_parts(matrix,"logical");});
    matrix.records[0].dtype="F16";
    rejects([&]{(void)logical_shape(matrix,"weight");});
    const auto a=tensor(false,1),b=tensor(false,2);
    const auto joined=join_rows({a,b});std::size_t transferred=0;
    mfq::TqRows rows(joined.size(),reader(joined,transferred));
    assert(rows.slice_rows(0,3)==a && rows.slice_rows(3,6)==b);
    const mfq::TqLayout layout(a.data(),a.size());const auto image=tq_image(a,layout);
    assert(image.size()==((layout.packed+3)&~std::size_t(3))+layout.packed_bytes+32);
    assert(std::equal(a.begin(),a.begin()+layout.packed,image.begin()));
    rejects([&]{(void)join_rows({});});
    rejects([&]{(void)join_rows({a,tensor(false,1,512)});});
    rejects([&]{(void)join_rows({tensor(true,1),tensor(true,2)});});
}

static void check_host_cache_registry() {
    using Value=std::vector<std::uint8_t>;
    using Cache=mfq::EncodedRowCache<Value>;
    std::atomic<std::size_t> bytes{0};
    auto cache=std::make_shared<Cache>(3*134,[](const Value& value){return value.size();},
        [&](std::size_t size,bool insert){if(insert)bytes.fetch_add(size);else bytes.fetch_sub(size);});
    mfq::register_host_cache(cache);mfq::register_host_cache(cache);
    for(int i=0;i<3;++i)cache->insert(i,Value(6,std::uint8_t(i)));
    auto lease=cache->find(0);
    assert(mfq::reclaim_host_caches(0)==0 && bytes.load()==18);
    assert(mfq::reclaim_host_caches(7)==12 && bytes.load()==6);
    assert(lease->front()==0 && cache->find(0) && !cache->find(1));
    assert(mfq::reclaim_host_caches(SIZE_MAX)==0 && !cache->find(0) && bytes.load()==6);
    cache.reset();lease.reset();assert(bytes.load()==0 && mfq::reclaim_host_caches(1)==0);
    struct Failure final:mfq::ReclaimableHostCache {
        std::size_t reclaim(std::size_t)override{throw std::runtime_error("injected host reclaim failure");}
    };
    auto failure=std::make_shared<Failure>();mfq::register_host_cache(failure);
    auto next=std::make_shared<Cache>(135,[](const Value& value){return value.size();});
    next->insert(0,Value(7,1));mfq::register_host_cache(next);
    assert(mfq::reclaim_host_caches(1)==7 && !next->find(0));
    struct Blocking final:mfq::ReclaimableHostCache {
        std::promise<void> entered;std::shared_future<void> resume;
        explicit Blocking(std::shared_future<void> future):resume(std::move(future)){}
        std::size_t reclaim(std::size_t)override{entered.set_value();resume.wait();return 11;}
    };
    std::promise<void> release;
    auto blocked=std::make_shared<Blocking>(release.get_future().share());
    auto entered=blocked->entered.get_future();std::weak_ptr<Blocking> lifetime=blocked;
    mfq::register_host_cache(blocked);
    auto work=std::async(std::launch::async,[]{return mfq::reclaim_host_caches(1);});
    assert(entered.wait_for(std::chrono::seconds(2))==std::future_status::ready);
    blocked.reset();assert(!lifetime.expired());
    release.set_value();assert(work.get()==11 && lifetime.expired());
}

static void check_deferred_expert_reads() {
    for(const bool columns:{false,true}) {
        const auto encoded=bank(columns);std::size_t bytes=0;
        const auto read=reader(encoded,bytes);
        const auto parts=mfq::tq_row_segments(encoded.size(),read,251);
        for(const auto& part:parts)part->cache_metadata();
        const auto before=bytes;
        std::vector<mfq::ReadSpan> spans;
        auto second=parts[1]->slice_rows(0,3,&spans);
        auto first=parts[0]->slice_rows(0,3,&spans);
        assert(bytes==before && spans.size()==2);
        for(const auto& span:spans)read(span.offset,span.destination,span.size);
        assert(first==tensor(columns,1) && second==tensor(columns,2));
        spans.clear();
        auto tail=parts[1]->slice_rows(1,3,&spans);
        for(const auto& span:spans)read(span.offset,span.destination,span.size);
        assert(tail==parts[1]->slice_rows(1,3));
    }
}

int main() {
    check_deferred_expert_reads();
    check_model_layout();
    check_host_cache_registry();
    check_encoded_row_cache();
    check(false);
    check(true);
    for(unsigned gs:{16,128,256,512}) {
        check_cached_metadata(false,gs);
        check_cached_metadata(true,gs);
    }
    check_cached_metadata(true,256,384,true);
    check_cached_metadata(true,256,640,true);
    for (unsigned gs : {16, 128, 256, 512, 1024, 2048}) {
        check_columns(false, gs);
        check_columns(true, gs);
    }
    assert(mfq::supported_tensor_dtype("TQ") && mfq::supported_tensor_dtype("MXFP8"));
    for (const auto* dtype : {"NINT", "NVQ", "MFE", "FP8-128SQ", "MXFP4-SQ"})
        assert(mfq::supported_tensor_dtype(dtype) == mfq::mfq_compat_enabled);
    std::cout << "TQ expert spans, independent scales, row reads and admission: ok\n";
}
