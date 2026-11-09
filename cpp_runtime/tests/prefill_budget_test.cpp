#include "prefill_budget.h"

#include <cassert>
#include <limits>
#include <utility>

int main() {
    using namespace mfq::engine;
    constexpr std::size_t row=327680, fixed=std::size_t(2)<<30;
    const auto memory=[&](int tokens) {return fixed+std::size_t(tokens)*row;};
    assert(select_prefill_chunk(131072,8192,memory(8192),fixed,row)==8192);
    assert(select_prefill_chunk(4096,8192,memory(8192),fixed,row)==4096);
    assert(select_prefill_chunk(8192,2048,memory(8192),fixed,row)==2048);
    assert(select_prefill_chunk(8192,8192,memory(4100),fixed,row)==4096);
    assert(select_prefill_chunk(8192,8192,memory(256)-1,fixed,row)==0);
    assert(select_prefill_chunk(8192,8192,fixed-1,fixed,row)==0);
    assert(select_prefill_chunk(17,8192,memory(17),fixed,row)==17);
    assert(select_prefill_chunk(17,8192,memory(16),fixed,row)==0);
    assert(select_prefill_chunk(8192,999999,memory(16384),fixed,row)==8192);
    assert(select_prefill_chunk(1048576,8192,memory(255),fixed,row,64)==192);
    assert(select_prefill_chunk(1048576,8192,memory(128),fixed,row,64)==128);
    assert(select_prefill_chunk(1048576,8192,memory(64),fixed,row,64)==64);
    assert(select_prefill_chunk(1048576,8192,memory(64)-1,fixed,row,64)==0);
    // More transfer storage must reduce the compute tile. A later request can
    // grow back to 8192; the policy never latches an earlier low-memory result.
    assert(select_prefill_chunk(8192,8192,memory(8192),fixed+4096*row,row)==4096);
    assert(select_prefill_chunk(8192,8192,memory(8192),fixed,row)==8192);
    const auto maximum=std::numeric_limits<std::size_t>::max();
    const auto dense=prefill_activation_row_bytes(768,3,3072);
    assert(dense==sizeof(float)*(768*3*8+3072*3));
    assert(prefill_activation_row_bytes(768,3)==sizeof(float)*768*3*8);
    assert(prefill_activation_row_bytes(INT64_MAX,INT64_MAX,INT64_MAX)==maximum);
    assert(prefill_sequence_row_bytes(768,3)==sizeof(float)*768*3*3);
    assert(prefill_sequence_row_bytes(INT64_MAX,INT64_MAX)==maximum);
    for(const auto geometry : {std::pair{768,3},std::pair{2048,2},std::pair{6144,4}}) {
        const auto activation=prefill_activation_row_bytes(geometry.first,geometry.second,geometry.first*3);
        const auto residual=prefill_sequence_row_bytes(geometry.first,geometry.second);
        const auto combined=prefill_budget_add(activation,residual);
        for(int count:{255,256,511,512,1024}) {
            const auto available=dense_prefill_fixed_bytes+std::size_t(count)*combined;
            const auto selected=select_prefill_chunk(1048576,8192,available,dense_prefill_fixed_bytes,combined);
            assert(selected==0 || dense_prefill_fixed_bytes+std::size_t(selected)*activation+
                std::size_t(selected)*residual<=available);
            assert(selected==(count<256?0:count/256*256));
        }
    }
    assert(select_prefill_chunk(1048576,8192,fixed+dense*1024,fixed,dense)==1024);
    assert(prefill_staging_fits(100,60,10,30));
    assert(!prefill_staging_fits(100,61,10,30));
    assert(!prefill_staging_fits(100,0,101,0));
    assert(!prefill_staging_fits(100,0,0,101));
    assert(!prefill_staging_fits(maximum,maximum,1,1));
    assert(prefill_budget_add(maximum-100,101)==maximum);
    assert(select_prefill_chunk(8192,8192,maximum,maximum,row)==0);
    for(int rows=0;rows<10000;++rows) {
        const auto result=select_prefill_chunk(8192,8192,memory(rows),fixed,row);
        assert(result>=0 && result<=8192 && result<=rows);
        assert(result==0 || result%256==0);
    }
}
