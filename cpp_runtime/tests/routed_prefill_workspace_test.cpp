#include "../backends/cuda/runtime/routed_prefill_workspace_plan.h"
#include "../backends/cuda/runtime/routed_prefill_capacity.h"
#include "../backends/cuda/runtime/routed_workspace_plan.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <limits>

int main() {
    using namespace mfq::cuda;
    {
        using Shares=std::array<std::size_t,3>;
        const Shares weights{1,3,6};
        assert(routed_budget_shares(100,weights)==Shares({10,30,60}));
        assert(routed_budget_shares(100,weights,Shares{20,0,0})==Shares({20,26,54}));
        assert(routed_budget_shares(100,weights,Shares{20,50,0})==Shares({20,50,30}));
        assert(routed_budget_shares(19,Shares{0,4,0},Shares{3,5,0})==Shares({3,16,0}));
        assert(routed_budget_shares(100,Shares{})==Shares{});
        assert(routed_budget_shares(0,weights)==Shares{});
        const auto maximum=std::numeric_limits<std::size_t>::max();
        const auto extreme=routed_budget_shares(maximum,Shares{maximum,maximum,maximum});
        assert(extreme[0]<=maximum-extreme[1]);
        assert(extreme[0]+extreme[1]==maximum-extreme[2]);
        assert(routed_budget_shares(maximum,Shares{0,maximum,0})==Shares({0,maximum,0}));
        bool rejected=false;
        try{(void)routed_budget_shares(60,weights,Shares{20,50,0});}
        catch(const std::runtime_error&){rejected=true;}
        assert(rejected);
        rejected=false;
        try{(void)routed_budget_shares(maximum,weights,Shares{maximum,1,0});}
        catch(const std::runtime_error&){rejected=true;}
        assert(rejected);
        // Shrinking and restoring a budget use the same immutable source mix.
        (void)routed_budget_shares(83,weights,Shares{20,50,0});
        assert(routed_budget_shares(100,weights,Shares{20,50,0})==Shares({20,50,30}));
    }
    {
        const std::vector<std::size_t> counts{129,65,64,17,16,1};
        const auto ends=[&](std::size_t capacity) {
            return routed_prefill_group_ends(counts.size(),capacity,[&](auto i){return counts[i];});
        };
        assert(ends(4)==std::vector<std::size_t>({2,3,5,6}));
        assert(ends(1)==std::vector<std::size_t>({1,2,3,4,5,6}));
        assert(ends(16)==ends(4));
        assert(routed_prefill_group_ends(0,1,[](auto){return 0;}).empty());
        const auto maximum=std::numeric_limits<std::size_t>::max();
        assert(routed_prefill_group_ends(2,2,[&](auto i){return i?maximum/2:maximum;})==
            std::vector<std::size_t>({1,2}));
        bool rejected=false;
        try{(void)ends(0);}catch(const std::invalid_argument&){rejected=true;}
        assert(rejected);
    }
    constexpr std::uint64_t MiB=1ULL<<20;
    const auto make=[](const std::vector<std::size_t>& counts,const std::vector<std::size_t>& ends,
            std::size_t gemm,const RoutedPrefillWorkspaceShape& shape,
            std::uint64_t activation=64ULL<<20,std::uint64_t decoded=256ULL<<20,bool compact=true) {
        return routed_prefill_workspace_plan(ends,gemm,[&](auto i){return counts.at(i);},
            shape,activation,decoded,compact);
    };
    {
        // A frequent expert in its own upload group must not pad later groups
        // or reserve sixteen full decoded images for a four-expert batch.
        const RoutedPrefillWorkspaceShape shape{2560,640,2560};
        const auto p=make({4096,128,128,128,128},{1,5},5,shape);
        assert(p.tiles.size()==2 && p.group_tile_ends==std::vector<std::size_t>({1,2}));
        assert(p.peak_image_batch==4 && p.peak_batch==4 && p.peak_rows==4096);
        assert(p.activation_bytes==60*MiB && p.decoded_bytes==25*MiB);
        assert(p.tiles[1].rows==512 && p.tiles[1].capacity==128);
    }
    {
        const RoutedPrefillWorkspaceShape narrow{33,19,17},wide{33,19,95};
        const auto compact=make({65,33},{2},2,narrow);
        const auto padded=make({65,33},{2},2,narrow,64*MiB,256*MiB,false);
        assert(compact.tiles.size()==1 && compact.tiles[0].compact);
        assert(compact.tiles[0].rows==98 && compact.tiles[0].capacity==80);
        assert(padded.tiles[0].rows==160 && !padded.tiles[0].compact);
        assert(compact.activation_bytes==21376 && padded.activation_bytes==34880);
        // Rectangular Down projections price their actual N, including when
        // it exceeds both the source K and the combined Gate/Up width.
        const auto output=make({65,33},{2},2,wide);
        assert(output.activation_bytes>compact.activation_bytes);
        assert(output.decoded_bytes==2*19*95*2);
        const auto constrained=make({65,33},{2},2,narrow,narrow.activation_bytes(65,true));
        assert(constrained.tiles.size()==2 && constrained.peak_batch==1);
        assert(constrained.activation_bytes==narrow.activation_bytes(65,true));
        // A bounded one-row decoded image keeps the operator's row-tiling
        // fallback available without inflating either activation batch.
        const auto tiled=make({65,33},{2},2,narrow,64*MiB,66);
        assert(tiled.tiles.size()==2 && tiled.decoded_bytes==66);
        assert(tiled.peak_batch==1 && tiled.activation_bytes==narrow.activation_bytes(65,true));
    }
    {
        const RoutedPrefillWorkspaceShape shape{33,19,33};
        const auto p=make({129,128,17,3},{4},2,shape);
        assert(p.tiles.size()==2 && p.tiles[0].end==2 && p.tiles[1].begin==2);
        assert(p.tiles[0].gemm && p.tiles[0].compact && p.tiles[0].rows==257);
        assert(!p.tiles[1].gemm && !p.tiles[1].compact && p.tiles[1].rows==64);
        const auto short_only=make({17,3},{2},0,shape);
        assert(short_only.decoded_bytes==0 && short_only.tiles[0].rows==64);
        // One indivisible expert may exceed the soft budget, but its actual
        // buffer requirement is still recorded and never shared with another.
        const auto small_budget=make({129,128},{2},2,shape,1,1);
        assert(small_budget.tiles.size()==2 && small_budget.peak_batch==1);
        assert(small_budget.activation_bytes==shape.activation_bytes(129,true));
    }
    {
        const RoutedPrefillWorkspaceShape shape{33,19,33};
        assert(make({}, {},0,shape).tiles.empty());
        bool rejected=false;
        try{(void)make({0},{1},1,shape);}catch(const std::invalid_argument&){rejected=true;}
        assert(rejected);
        rejected=false;
        try{(void)make({1},{1,1},1,shape);}catch(const std::invalid_argument&){rejected=true;}
        assert(rejected);
        rejected=false;
        try{(void)make({1},{1},1,{std::numeric_limits<std::uint64_t>::max(),19,33});}
        catch(const std::overflow_error&){rejected=true;}
        assert(rejected);
    }
}
