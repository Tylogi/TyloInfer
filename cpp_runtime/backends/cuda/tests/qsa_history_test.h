#pragma once
#include "kernels/qsa_tq.h"

inline void check_qsa_recall_history() {
    namespace qsa=tyloinfer::qsa;
    int cases=0;
    for(int blocks:{37,1003,4097})for(int queries:{1,3,32}) {
        const int count=queries==3?19:513,words=(blocks+31)/32;
        std::vector<std::uint32_t> expected(words+2,0);
        expected.front()=0x14538acfu;expected.back()=0xf13579bdu;
        auto output=tb::from_blob(expected.data(),{words+2},tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt32)).to(tb::kCUDA);
        for(int pass=0;pass<2;++pass) {
            std::vector<qsa::SelectedBlock> selected(std::size_t(queries)*count);
            std::vector<int64_t> positions(queries);
            for(int q=0;q<queries;++q) {
                positions[q]=pass && !q?-1:2LL*blocks+q;
                for(int slot=0;slot<count;++slot) {
                    auto& block=selected[std::size_t(q)*count+slot];
                    block.id=(slot*19+q*37+pass*3)%(blocks+9)-3;
                    block.token_mask=std::uint32_t(slot*13+q+pass)&31u;
                    if(block.id<0 || block.id>=blocks)continue;
                    std::uint32_t causal=0;
                    for(int lane=0;lane<4;++lane)if(int64_t(block.id)*4+lane<=positions[q])causal|=1u<<lane;
                    if(block.token_mask&causal)expected[1+block.id/32]|=1u<<unsigned(block.id%32);
                }
            }
            auto ids=tb::from_blob(selected.data(),{int64_t(selected.size()*sizeof(qsa::SelectedBlock))},
                tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8)).to(tb::kCUDA);
            auto absolute=tb::from_blob(positions.data(),{queries},tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt64)).to(tb::kCUDA);
            require(qsa::tq_accumulate_recall_bits(ids.data_ptr<qsa::SelectedBlock>(),absolute.data_ptr<int64_t>(),
                output.data_ptr<std::uint32_t>()+1,queries,count,blocks,mfq_current_cuda_stream())==0,
                "Recall history kernel failed");
            const auto result=output.cpu();
            require(!std::memcmp(result.data_ptr(),expected.data(),expected.size()*sizeof(std::uint32_t)),
                "Recall history changed causal membership, accumulated bits or destination guards");
        }
        ++cases;
    }
    std::cout<<"QSA recall history: independent causal oracle, partial words, invalid IDs, guarded writes and accumulation PASS cases="<<cases<<'\n';
}
