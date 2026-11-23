#include "../runtime/expert_route_usage.h"
#include "../runtime/routed_moe_order.h"
#include "../runtime/routed_host_cache.h"
#include "../runtime/routed_expert_groups.h"
#include "../runtime/routed_adaptive_slots.h"
#include "../runtime/routed_prefill_routes.h"
#include "../runtime/routed_prefill_capacity.h"
#include "../runtime/routed_completion_order.h"
#include "../runtime/gdn_decode_plan.h"
#include "gguf_speculative_journal_test.h"
#include "../../../core/include/tyloinfer/host_memory.h"
#include <iostream>
#include <map>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace {
void check_gdn_parallel_plan() {
    for(const int processors:{1,2,3,7,16,63,127,1024}) {
        int previous=4;
        const auto target=std::uint64_t(processors)*2;
        for(std::uint64_t heads=1;heads<=target*2;++heads) {
            const int tiles=mfq::cuda::gdn_decode_tiles(heads,processors);
            if((tiles!=1 && tiles!=2 && tiles!=4) || tiles>previous)
                throw std::runtime_error("GDN parallel plan is not monotonic");
            if(heads*4>=target && heads*tiles<target)
                throw std::runtime_error("GDN parallel plan left its producer working set uncovered");
            if(tiles>1 && heads*(tiles/2)>=target)
                throw std::runtime_error("GDN parallel plan repeated unnecessary column splits");
            previous=tiles;
        }
        if(mfq::cuda::gdn_decode_tiles(UINT64_MAX,processors)!=1)
            throw std::runtime_error("large GDN work overflowed its launch plan");
    }
    for(const auto work:{std::pair{std::uint64_t(0),1},std::pair{std::uint64_t(1),0},std::pair{std::uint64_t(1),-1}}) {
        bool rejected=false;try{mfq::cuda::gdn_decode_tiles(work.first,work.second);}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("GDN parallel plan accepted invalid resources");
    }
}
void check_routed_completion_order() {
    using namespace mfq::cuda;
    std::mt19937 random(781);
    for(unsigned pass=0;pass<128;++pass) {
        RoutedCompletionClock clock;
        std::vector<RoutedCompletionPosition> positions;
        std::map<std::pair<int,std::uintptr_t>,std::uint64_t> completed;
        for(int i=0;i<257;++i)positions.push_back(clock.record(i%3,std::uintptr_t(i%5)));
        for(const auto position:positions)if(random()%3==0)
            completed[{position.device,position.stream}]=position.sequence;
        std::shuffle(positions.begin(),positions.end(),random);
        RoutedCompletionProbe probe;
        for(const auto position:positions) {
            const bool expected=position.sequence<=completed[{position.device,position.stream}];
            const auto ready=probe.ready(position,[&]{return expected;});
            if(ready!=expected)throw std::runtime_error("completion order crossed streams or retired an incomplete fence");
        }
    }
    RoutedCompletionClock clock;
    std::vector<RoutedCompletionPosition> ordered;
    for(int i=0;i<257;++i)ordered.push_back(clock.record(0,17));
    unsigned queries=0;
    RoutedCompletionProbe pending;
    for(const auto position:ordered)if(pending.ready(position,[&]{++queries;return false;}))
        throw std::runtime_error("pending stream retired its host payload");
    if(queries!=1)throw std::runtime_error("pending stream queried every later completion fence");
    queries=0;
    RoutedCompletionProbe finished;
    for(auto it=ordered.rbegin();it!=ordered.rend();++it)
        if(!finished.ready(*it,[&]{++queries;return true;}))
            throw std::runtime_error("completed stream retained an earlier fence");
    if(queries!=1)throw std::runtime_error("completed tail queried every earlier completion fence");
    for(const bool expected:{false,true,false})
        if(finished.ready({},[&]{return expected;})!=expected)
            throw std::runtime_error("untracked external event reused an unrelated completion result");
    RoutedCompletionProbe maximum;
    if(!maximum.ready({0,0,UINT64_MAX},[]{return true;}))
        throw std::runtime_error("maximum completion sequence matched an empty pending sentinel");
    std::cout<<"Routed completion order: shuffled device/stream fences, pending tails and external events PASS\n";
}

void check_mfq_journal_heat() {
    RoutedExpertJournal journal;
    const std::vector<std::int32_t> ids{1,1,1,2, 2,2,2,2};
    const auto original=ids;
    std::vector<RoutedExpertTouch> captured,prefix,other;
    unsigned lookups=0;
    journal.collect(std::span<const std::int32_t>(ids),3,captured,[&](auto){++lookups;return false;});
    if(lookups!=2 || captured.size()!=2 || ids!=original)
        throw std::runtime_error("int32 journal changed RAM route order or repeated an expert lookup");
    const std::int32_t replacement[]{0,0,0,0};
    journal.collect(std::span<const std::int32_t>(replacement),3,other,[](auto){return true;});
    // A captured journal owns its counts across subsequent layer publication.
    if(captured[0].expert!=2 || captured[0].uses!=5 || captured[1].expert!=1 || captured[1].uses!=3)
        throw std::runtime_error("later int32 route publication changed a captured journal");
    for(const std::optional<int> mode:{std::optional<int>{},std::optional<int>{0},
            std::optional<int>{1},std::optional<int>{2}})
        for(std::size_t committed=0;committed<=4;++committed) {
            std::vector<float> expected{0,16777216.f,0},actual=expected;
            std::vector<std::uint64_t> seen(1,0);
            const auto expected_visits=mfq::cuda::routed_speculative_heat<std::int32_t>(
                ids,4,committed,expected,seen,mode);
            const auto learned=mode?committed:mfq::cuda::routed_observed_rows(committed,4);
            std::size_t visits=0;
            if(learned) {
                const auto* touches=&captured;
                if(learned<4) {
                    journal.collect(std::span<const std::int32_t>(ids.data(),learned*2),3,prefix,[](auto){return false;});
                    touches=&prefix;
                }
                visits=apply_routed_expert_heat(*touches,actual,mode);
            }
            if(visits!=expected_visits || actual!=expected || ids!=original)
                throw std::runtime_error("journal heat changed executed/prefix selection, rounding, or accepted RAM routes");
        }
    std::vector<float> observations(3,0);
    const std::vector<RoutedExpertTouch> invalid{{0,2,false},{3,1,false}};
    bool rejected=false;
    try{(void)apply_routed_expert_heat(invalid,observations);}
    catch(const std::out_of_range&){rejected=true;}
    if(!rejected || observations!=std::vector<float>(3,0))
        throw std::runtime_error("invalid journal heat modified observations before validation");
}
void check_shared_expert_heat() {
    using namespace mfq::cuda;
    if(expert_usage_interval!=4 || expert_usage_retention!=.7f ||
            expert_usage_decay(0)!=1.f || expert_usage_decay(4)!=.7f ||
            std::abs(expert_usage_decay(8)-.49f)>1e-6f)
        throw std::runtime_error("shared expert heat lost its decay window");
    for(std::uint64_t window=1;window<=32;++window)
        if(routed_admission_due(window)!=(window%4==0) || !routed_admission_due(window,true))
            throw std::runtime_error("shared admission changed the completed-window interval");
    // A 96-row prompt has its highest frequency before the last 32 rows.
    // Every row must contribute, with the same normalized seed as TQ.
    std::vector<std::int32_t> prompt(96,0);
    std::fill(prompt.begin()+64,prompt.end(),1);
    std::vector<std::uint64_t> counts(3,0);
    for(const auto id:prompt)++counts[std::size_t(id)];
    ExpertRouteUsage usage;
    for(std::size_t expert=0;expert<counts.size();++expert) {
        usage.prefill(expert,counts[expert],prompt.size());
        if(usage.score(expert)!=expert_prefill_usage(counts[expert],prompt.size()))
            throw std::runtime_error("expert prefill seed diverged from shared frequency normalization");
    }
    if(usage.score(0)!=2.f*usage.score(1) || usage.score(2)!=0.f || expert_prefill_usage(7,0)!=0.f)
        throw std::runtime_error("prefill heat ignored full prompt frequencies or empty rows");
    const std::vector<std::int32_t> ids{1,1,1,2, 2,2,2,2};
    std::vector<std::uint64_t> seen(1,0);
    for(std::size_t committed=0;committed<=4;++committed) {
        std::vector<float> observations(3,0);
        const auto learned=routed_speculative_heat<std::int32_t>(ids,4,committed,observations,seen);
        if(learned!=(committed?8:0) || observations!=std::vector<float>({0,committed?3.f:0.f,committed?5.f:0.f}))
            throw std::runtime_error("default heat did not learn executed verifier rows after a successful commit");
    }
    for(const int mode:{0,1,2}) {
        std::vector<float> observations(3,0);
        const auto learned=routed_speculative_heat<std::int32_t>(ids,4,2,observations,seen,mode);
        const auto expected=mode==0?std::vector<float>{0,3,1}:
            (mode==1?std::vector<float>{0,1,1}:std::vector<float>{0,2,2});
        if(learned!=std::size_t(mode?2:4) || observations!=expected)
            throw std::runtime_error("explicit legacy heat override changed accepted-prefix semantics");
    }
    for(const auto invalid:std::vector<std::vector<std::int32_t>>{{0,1,2},{0,1,2,3},{0,1,2,-1}}) {
        std::vector<float> observations(3,0);bool rejected=false;
        try{(void)routed_speculative_heat<std::int32_t>(invalid,4,1,observations,seen);}
        catch(const std::exception&){rejected=true;}
        if(!rejected || observations!=std::vector<float>(3,0))
            throw std::runtime_error("invalid speculative heat modified observations before validation");
    }
    std::vector<float> observations(3,0);bool rejected=false;
    try{(void)routed_speculative_heat<std::int32_t>(ids,4,5,observations,seen);}
    catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("speculative heat accepted a prefix beyond executed rows");
}
void check_host_evacuation_policy() {
    using namespace mfq::cuda;
    if(routed_host_evacuation_capacity(100,30,80,20)!=60 ||
            routed_host_evacuation_capacity(100,99,80,20)!=1 ||
            routed_host_evacuation_capacity(100,101,80,20)!=0 ||
            routed_host_evacuation_capacity(100,0,19,20)!=0)
        throw std::runtime_error("host evacuation exceeded existing cache or physical headroom");
    struct Block {const void* allocation_id()const{return this;}};
    struct Image {std::shared_ptr<Block> block;bool ready=true;std::size_t size()const{return 16;}};
    struct Cache:RoutedHostCache<Image> {
        void add(std::uint64_t key,const std::shared_ptr<Image>& image,bool duplicate=false) {
            host_index_insert(key,image,duplicate);
        }
        auto candidates(const void* allocation)const {
            return host_evacuation_candidates(allocation,[](auto key){return key!=5;},
                [](const auto& image){return image.ready;});
        }
    } cache;
    const auto block=std::make_shared<Block>();
    for(std::uint64_t key:{1,2,4,5,6}) {
        auto image=std::make_shared<Image>();if(key!=6)image->block=block;
        image->ready=key!=4;cache.add(key,image,key==2);
    }
    auto active=std::make_shared<Image>();active->block=block;cache.add(3,active);
    if(cache.candidates(block.get())!=std::vector<std::uint64_t>{1} || !cache.candidates(nullptr).empty())
        throw std::runtime_error("host evacuation admitted a duplicate, consumer lease, unready or ineligible image");
    bool blocked=false;
    {RoutedHostEvacuationBlock outer(blocked);{RoutedHostEvacuationBlock inner(blocked);}
        if(!blocked)throw std::runtime_error("nested host evacuation block released its outer owner");}
    if(blocked)throw std::runtime_error("host evacuation block did not restore the previous state");
}
void check_host_allocator_trim() {
    struct Cache:mfq::cuda::RoutedHostCache<int> {using RoutedHostCache::host_headroom_after_allocator_trim;} cache;
    using Clock=std::chrono::steady_clock;
    const auto start=Clock::time_point{}+std::chrono::seconds(2);
    unsigned releases=0,probes=0;
    const auto release=[&]{++releases;};
    const auto probe=[&]{++probes;return std::size_t(64);};
    if(cache.host_headroom_after_allocator_trim(32,32,release,probe,start)!=32 || releases || probes)
        throw std::runtime_error("idle host allocator trim ran without a shortage");
    if(cache.host_headroom_after_allocator_trim(128,0,release,probe,start)!=64 || releases!=1 || probes!=1)
        throw std::runtime_error("idle host allocator trim did not re-sample actual available bytes");
    if(cache.host_headroom_after_allocator_trim(128,0,release,probe,start+std::chrono::milliseconds(999))!=0 ||
            releases!=1 || probes!=1)
        throw std::runtime_error("idle host allocator trim ignored its retry interval");
    if(cache.host_headroom_after_allocator_trim(128,0,release,probe,start+std::chrono::seconds(1))!=64 ||
            releases!=2 || probes!=2)
        throw std::runtime_error("idle host allocator trim did not retry a later shortage");
}
void check_host_memory_headroom() {
    mfq::HostMemorySample memory{600,800,std::nullopt};
    if(mfq::host_memory_headroom(memory)!=800 ||
            mfq::host_memory_headroom(memory,2000)!=800 ||
            mfq::host_memory_headroom(memory,1000)!=400 ||
            mfq::host_memory_headroom(memory,601)!=1 ||
            mfq::host_memory_headroom(memory,600)!=0 ||
            mfq::host_memory_headroom(memory,500)!=0)
        throw std::runtime_error("host headroom lost physical availability or process limit boundaries");
    memory.commit_available_bytes=300;
    if(mfq::host_memory_headroom(memory)!=300 ||
            mfq::host_memory_headroom(memory,1000)!=300 ||
            mfq::host_memory_headroom(memory,800)!=200)
        throw std::runtime_error("host headroom did not intersect commit and process availability");
    memory.commit_available_bytes=900;
    if(mfq::host_memory_headroom(memory)!=800)
        throw std::runtime_error("roomier commit availability exceeded physical availability");
    memory.commit_available_bytes=0;
    if(mfq::host_memory_headroom(memory)!=0)
        throw std::runtime_error("host headroom ignored exhausted commit capacity");
    memory={600,0,std::nullopt};
    if(mfq::host_memory_headroom(memory,1000)!=0)
        throw std::runtime_error("process capacity replaced exhausted physical availability");
    memory={600,1,std::nullopt};
    if(mfq::host_memory_headroom(memory)!=1)
        throw std::runtime_error("automatic host headroom reserved unrequested RAM");
    const auto maximum=std::numeric_limits<std::uint64_t>::max();
    memory={maximum,maximum,std::nullopt};
    if(mfq::host_memory_headroom(memory)!=std::numeric_limits<std::size_t>::max() ||
            mfq::host_memory_headroom(memory,maximum)!=0)
        throw std::runtime_error("host headroom truncated availability or underflowed process capacity");
}
void check_prefill_capacity() {
    using namespace mfq::cuda;
    constexpr std::size_t mib=std::size_t(1)<<20;
    if(routed_prefill_group_capacity(100,8*mib,32*mib)!=8 ||
            routed_prefill_group_capacity(100,8*mib,0)!=8 ||
            routed_prefill_group_capacity(100,128*mib,512*mib)!=1 ||
            routed_prefill_group_capacity(100,8*mib,0,3)!=3 ||
            routed_prefill_group_capacity(100,8*mib,0,0,32*mib)!=2 ||
            routed_prefill_group_capacity(100,8*mib,0,0,0)!=1)
        throw std::runtime_error("routed prefill capacity lost image, override or host staging bounds");
    RoutedPrefillImageGeometry backbone,predictor;
    backbone.add(0,0,4*mib);backbone.add(0,1,mib);backbone.add(0,2,7*mib);
    backbone.add(0,0,2*mib);backbone.add(0,1,5*mib);backbone.add(0,2,mib);
    predictor.add(-1,0,128*mib);
    if(backbone.experts!=3 || backbone.encoded!=8*mib || predictor.experts!=1 ||
            routed_prefill_group_capacity(backbone.experts,backbone.encoded,0)!=3)
        throw std::runtime_error("cohort fields were not summed per expert or predictor changed backbone capacity");
    const auto layout=[&](std::size_t count){return 2*(mib+3*mib*count);};
    if(routed_prefill_workspace_capacity(64,41*mib,layout)!=6 ||
            routed_prefill_workspace_capacity(3,41*mib,layout)!=3 ||
            routed_prefill_workspace_capacity(64,8*mib,layout)!=1 ||
            routed_prefill_workspace_capacity(64,8*mib-1,layout)!=0)
        throw std::runtime_error("routed prefill workspace lost double buffering or cohort dummy overhead");
    const auto maximum=std::numeric_limits<std::size_t>::max();
    const auto identity=[](std::size_t count){return count;};
    if(routed_prefill_workspace_capacity(maximum,maximum-1,identity)!=maximum-1 ||
            routed_prefill_workspace_capacity(maximum,maximum,identity)!=maximum ||
            routed_prefill_workspace_capacity(10,0,identity)!=0)
        throw std::runtime_error("routed prefill workspace rounded or overflowed an integer byte boundary");
    bool visited=false;
    if(routed_prefill_workspace_capacity(0,maximum,[&](auto){visited=true;return 1;})!=0 || visited)
        throw std::runtime_error("empty prefill capacity evaluated a nonexistent stage layout");
    for(const auto decoded:{-1.L,std::numeric_limits<long double>::infinity(),
            std::numeric_limits<long double>::quiet_NaN()}) {
        bool rejected=false;
        try{(void)routed_prefill_group_capacity(3,mib,decoded);}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("routed prefill capacity accepted invalid decoded geometry");
    }
    bool rejected=false;
    try{(void)routed_prefill_workspace_capacity(3,maximum,[](auto){return -1;});}
    catch(const std::overflow_error&){rejected=true;}
    if(!rejected)throw std::runtime_error("routed prefill workspace accepted an invalid signed size");
}
void check_prefill_routes() {
    const std::vector<std::vector<std::int64_t>> groups{{31,5,17},{9},{23,11}};
    const mfq::cuda::RoutedPrefillRoutes plan(groups.size(),[&](auto i)->const auto&{return groups[i];});
    if(plan.bounds!=std::vector<std::int64_t>{0,3,4,6} ||
            plan.slots!=std::vector<std::int64_t>{31,5,17,9,23,11} || plan.rows(1,3)!=3)
        throw std::runtime_error("compact route plan changed original reduction slots or retained padding");
    bool rejected=false;
    try{(void)plan.rows(2,4);}catch(const std::out_of_range&){rejected=true;}
    if(!rejected)throw std::runtime_error("compact route plan accepted an invalid group boundary");
    auto invalid=groups;invalid[1].clear();rejected=false;
    try{mfq::cuda::RoutedPrefillRoutes empty(invalid.size(),[&](auto i)->const auto&{return invalid[i];});}
    catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("compact route plan accepted an empty active expert");
    invalid=groups;invalid[1][0]=-1;rejected=false;
    try{mfq::cuda::RoutedPrefillRoutes negative(invalid.size(),[&](auto i)->const auto&{return invalid[i];});}
    catch(const std::out_of_range&){rejected=true;}
    if(!rejected)throw std::runtime_error("compact route plan accepted a negative output slot");
}
struct Demand {float uses=0;std::size_t bytes=0;};
using Reference=std::unordered_map<std::uint64_t,Demand>;
void check_grouped_admission() {
    using Group=std::array<std::uint64_t,3>;
    const std::array<Group,2> banks{{{1,2,3},{4,5,6}}};
    const mfq::cuda::RoutedExpertBanks grouped_banks(banks);
    {
        mfq::cuda::ExpertRouteUsage variable;
        for(unsigned expert=0;expert<4;++expert)for(const auto source:banks[0]) {
            const auto key=(source<<32)|expert;
            variable.image(key,expert==2?30:expert==3?10:20);
            if(expert<2)variable.observe(key);
        }
        variable.select(120,0,&grouped_banks);
        const auto before=variable.retained_keys();
        if(variable.exchange(1ULL<<32,(1ULL<<32)|2,120,grouped_banks))
            throw std::runtime_error("a larger expert exceeded a single victim's capacity");
        const std::array<std::uint64_t,2> duplicate{1ULL<<32,1ULL<<32};
        if(variable.exchange(duplicate,(1ULL<<32)|2,120,grouped_banks) || variable.retained_keys()!=before)
            throw std::runtime_error("duplicate victims changed a failed atomic exchange");
        std::vector<mfq::cuda::RoutedImageCandidate> candidates{{4,(1ULL<<32)|3,30},{12,(1ULL<<32)|2,90}};
        std::vector<mfq::cuda::RoutedImageCandidate> victims{{1,1ULL<<32,60},{1,(1ULL<<32)|1,60}};
        mfq::cuda::AdaptiveImageSwapStats stats;
        const auto swaps=mfq::cuda::adaptive_image_swaps(variable,candidates,victims,120,grouped_banks,2,&stats);
        if(swaps.size()!=2 || swaps[0].victims.size()!=2 || !swaps[1].victims.empty() ||
                stats.multi_victim!=1 || variable.retained_bytes()!=120 || variable.measured_retained_bytes()!=120 ||
                variable.retained_images()!=6 || variable.retained(1ULL<<32) || !variable.retained((3ULL<<32)|2))
            throw std::runtime_error("variable-size admission failed to combine cold capacity or use its remainder");
        const auto selected=variable.retained_keys();
        std::vector<mfq::cuda::RoutedImageCandidate> expensive{{20,1ULL<<32,60}};
        std::vector<mfq::cuda::RoutedImageCandidate> protected_images{{10,(1ULL<<32)|2,90}};
        if(!mfq::cuda::adaptive_image_swaps(variable,expensive,protected_images,120,grouped_banks,0).empty() ||
                variable.retained_keys()!=selected)
            throw std::runtime_error("zero admission cap changed selected experts");
        expensive[0].score=11;
        if(!mfq::cuda::adaptive_image_swaps(variable,expensive,protected_images,120,grouped_banks).empty() ||
                variable.retained_keys()!=selected)
            throw std::runtime_error("variable-size replacement discarded greater combined reuse");
    }
    {
        mfq::cuda::ExpertRouteUsage slots;
        for(const auto& sources:banks)for(std::uint32_t expert=0;expert<3;++expert)
            for(const auto source:sources) {
                const auto key=(source<<32)|expert;slots.image(key,expert==2?101:20);
                slots.observe(key);if(!expert)slots.observe(key);
            }
        slots.select(120,0,&grouped_banks);
        if(!slots.replace(1ULL<<32,(1ULL<<32)|1,120,grouped_banks) ||
                slots.retained(1ULL<<32) || !slots.retained((3ULL<<32)|1) || slots.retained_images()!=6)
            throw std::runtime_error("same-layer slot replacement separated an expert");
        const auto before=slots.retained_keys();
        if(slots.replace((1ULL<<32)|1,(1ULL<<32)|2,120,grouped_banks) ||
                slots.replace((1ULL<<32)|1,(4ULL<<32)|1,120,grouped_banks) || slots.retained_keys()!=before)
            throw std::runtime_error("slot replacement crossed a layer or exceeded its byte budget");
        if(grouped_banks.groups()!=std::vector<Group>(banks.begin(),banks.end()))
            throw std::runtime_error("routed slot layers are not in source order");
        if(!slots.exchange((1ULL<<32)|1,(4ULL<<32)|1,120,grouped_banks) ||
                slots.retained((1ULL<<32)|1) || !slots.retained((6ULL<<32)|1) || slots.retained_images()!=6)
            throw std::runtime_error("global expert exchange exceeded capacity or separated projections");
    }
    struct Expert {std::uint64_t key;float score;std::size_t bytes;std::array<std::uint64_t,3> images;};
    mfq::cuda::ExpertRouteUsage usage;
    std::vector<Expert> experts;
    for(const auto& sources:banks)for(std::uint32_t expert=0;expert<4;++expert) {
        const auto score=float(9-expert);
        Expert item{(sources[0]<<32)|expert,score,0,{}};
        const bool resident=expert==2;
        if(resident)item.score+=mfq::cuda::expert_replacement_margin;
        for(unsigned role=0;role<3;++role) {
            const auto key=item.images[role]=(sources[role]<<32)|expert;
            const auto bytes=std::size_t(11*sources[role]+7*expert+3);
            usage.image(key,bytes);
            for(unsigned i=0;i<9-expert;++i)usage.observe(key);
            usage.resident(key,resident);item.bytes+=bytes;
        }
        experts.push_back(item);
        for(const auto image:item.images) {
            std::vector<std::uint64_t> victims;
            grouped_banks.visit_expert(image,[&](auto key){victims.push_back(key);});
            if(victims!=std::vector<std::uint64_t>(item.images.begin(),item.images.end()))
                throw std::runtime_error("expert replacement separated projections or mixed expert IDs");
        }
    }
    std::sort(experts.begin(),experts.end(),[](const auto& a,const auto& b) {
        return a.score!=b.score?a.score>b.score:a.key<b.key;
    });
    for(std::size_t budget=0;budget<2500;budget+=17) {
        auto remaining=budget;
        std::unordered_set<std::uint64_t> expected;
        for(const auto& expert:experts)if(expert.bytes<=remaining) {
            remaining-=expert.bytes;expected.insert(expert.images.begin(),expert.images.end());
        }
        usage.select(budget,mfq::cuda::expert_replacement_margin,&grouped_banks);
        const std::unordered_set<std::uint64_t> selected(usage.retained_keys().begin(),usage.retained_keys().end());
        if(selected!=expected || selected.size()!=usage.retained_images())
            throw std::runtime_error("grouped admission differs from complete-expert byte-budget ranking");
        for(const auto& expert:experts)for(const auto key:expert.images)
            if(usage.retained(key)!=bool(expected.count(key)))
                throw std::runtime_error("grouped admission retained an incomplete expert");
    }
    mfq::cuda::ExpertRouteUsage aliased;
    aliased.image(1ULL<<32,31);aliased.observe(1ULL<<32);
    aliased.image(2ULL<<32,29);aliased.observe(2ULL<<32);
    const std::array<Group,1> repeated{{{1,1,2}}};
    const mfq::cuda::RoutedExpertBanks repeated_banks(repeated);
    std::vector<std::uint64_t> repeated_victims;
    repeated_banks.visit_expert(1ULL<<32,[&](auto key){repeated_victims.push_back(key);});
    if(repeated_victims!=std::vector<std::uint64_t>{1ULL<<32,2ULL<<32})
        throw std::runtime_error("expert replacement visited aliased storage twice");
    aliased.select(59,0,&repeated_banks);
    if(aliased.retained_images())throw std::runtime_error("grouped admission exceeded its complete-expert budget");
    aliased.select(60,0,&repeated_banks);
    if(aliased.retained_images()!=2)throw std::runtime_error("aliased projection was charged or retained twice");
    mfq::cuda::ExpertRouteUsage incomplete;
    incomplete.image(1ULL<<32,31);incomplete.observe(1ULL<<32);incomplete.select(SIZE_MAX,0,&repeated_banks);
    if(incomplete.retained_images())throw std::runtime_error("grouped admission retained an expert with missing projection metadata");
    const std::array<Group,2> overlapping{{{1,2,3},{1,4,5}}};
    const mfq::cuda::RoutedExpertBanks overlapping_banks(overlapping);
    std::vector<std::uint64_t> shared_victims;
    overlapping_banks.visit_expert(2ULL<<32,[&](auto key){shared_victims.push_back(key);});
    if(shared_victims!=std::vector<std::uint64_t>{2ULL<<32})
        throw std::runtime_error("expert replacement erased storage belonging to another projection group");
    mfq::cuda::RoutedExpertBanks wide;
    wide.add({0,UINT32_MAX,3});
    bool rejected=false;
    try{wide.add({1,UINT64_MAX,2});}catch(const std::out_of_range&){rejected=true;}
    if(!rejected || !wide.find(0) || wide.find(1))
        throw std::runtime_error("invalid source identifier corrupted an existing expert group");
    for(std::size_t budget:{0u,30u,60u,90u}) {
        auto reference=usage;reference.select(budget);
        usage.select(budget,0,&overlapping_banks);
        const std::unordered_set<std::uint64_t> selected(usage.retained_keys().begin(),usage.retained_keys().end());
        const std::unordered_set<std::uint64_t> expected(reference.retained_keys().begin(),reference.retained_keys().end());
        if(selected!=expected)throw std::runtime_error("overlapping projection groups lost their independent ownership");
    }
}
void check_route_groups() {
    std::mt19937_64 random(781);
    for(const std::int64_t experts:{1LL,7LL,257LL,(1LL<<33)+7})
        for(const std::size_t count:{0ULL,1ULL,4ULL,32ULL,512ULL})
            for(const bool partitioned:{false,true}) {
                const auto begin=partitioned?experts/3:0,end=partitioned?experts*2/3:experts;
                std::vector<std::int64_t> ids(count),expected;
                std::unordered_map<std::int64_t,std::vector<std::int64_t>> locations;
                for(std::size_t i=0;i<count;++i) {
                    const auto id=ids[i]=std::int64_t(random()%experts);
                    if(id<begin || id>=end)continue;
                    if(!locations.count(id))expected.push_back(id);
                    locations[id].push_back(std::int64_t(i));
                }
                std::vector<std::int64_t> unique;
                std::vector<std::vector<std::int64_t>> slots;
                mfq::cuda::group_routed_experts(ids,experts,unique,&slots,begin,end);
                if(unique!=expected || slots.size()!=expected.size())
                    throw std::runtime_error("grouping changed first-use order or local expert selection");
                for(std::size_t i=0;i<unique.size();++i)if(slots[i]!=locations.at(unique[i]))
                    throw std::runtime_error("grouping lost duplicate routes or changed original routing slots");
                mfq::cuda::group_routed_experts(ids,experts,unique,nullptr,begin,end);
                if(unique!=expected)throw std::runtime_error("unique-only selection differs from grouped selection");
            }
    for(const auto invalid:{std::int64_t(-1),std::int64_t(7)})for(const bool grouped:{false,true}) {
        const std::vector<std::int64_t> ids{0,invalid};
        std::vector<std::int64_t> unique;
        std::vector<std::vector<std::int64_t>> slots;
        bool rejected=false;
        try{mfq::cuda::group_routed_experts(ids,7,unique,grouped?&slots:nullptr,0,0);}
        catch(const std::out_of_range&){rejected=true;}
        if(!rejected)throw std::runtime_error("expert partition hid an invalid routing ID");
    }
    std::vector<std::int64_t> unique;
    bool rejected=false;
    try{mfq::cuda::group_routed_experts({},0,unique);}
    catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("grouping accepted an empty expert domain");
}
void compare(mfq::cuda::ExpertRouteUsage& usage,const Reference& reference,std::size_t budget) {
    std::vector<std::uint64_t> order;
    for(const auto& [key,demand]:reference) {
        if(usage.score(key)!=demand.uses)throw std::runtime_error("route score changed");
        if(demand.uses>0 && demand.bytes)order.push_back(key);
    }
    std::sort(order.begin(),order.end(),[&](auto a,auto b) {
        return reference.at(a).uses!=reference.at(b).uses?
            reference.at(a).uses>reference.at(b).uses:a<b;
    });
    const auto original=budget;
    std::unordered_set<std::uint64_t> expected;
    for(const auto key:order)if(reference.at(key).bytes<=budget) {
        expected.insert(key);budget-=reference.at(key).bytes;
    }
    usage.select(original);
    if(usage.retained_images()!=expected.size())throw std::runtime_error("route count differs from full ranking");
    const std::unordered_set<std::uint64_t> selected(usage.retained_keys().begin(),usage.retained_keys().end());
    if(selected!=expected || selected.size()!=usage.retained_keys().size())
        throw std::runtime_error("admission worklist differs from retained expert set");
    for(const auto& [key,_]:reference)if(usage.retained(key)!=bool(expected.count(key)))
        throw std::runtime_error("route retention differs from full ranking");
}
}
int main()try {
    check_gdn_parallel_plan();
    check_routed_completion_order();
    check_mfq_journal_heat();
    check_shared_expert_heat();
    check_host_evacuation_policy();
    check_host_allocator_trim();
    check_host_memory_headroom();
    check_prefill_capacity();
    check_prefill_routes();
    check_grouped_admission();
    check_route_groups();
    gguf_journal_test::check();
    for(std::size_t executed:{std::size_t(1),std::size_t(4),std::size_t(32)})
        for(std::size_t committed=0;committed<=executed;++committed)
            if(mfq::cuda::routed_observed_rows(committed,executed)!=(committed?executed:0))
                throw std::runtime_error("cache heat confused executed verification work with committed KV rows");
    {
        struct Image {std::size_t bytes;bool registered;std::size_t size()const{return bytes;}bool pinned()const{return registered;}};
        struct Cache:mfq::cuda::RoutedHostCache<Image> {
            using RoutedHostCache::host_index_insert;using RoutedHostCache::host_index_erase;
            using RoutedHostCache::host_index_touch;using RoutedHostCache::host_index_duplicate;
            using RoutedHostCache::host_index_victim;
            std::size_t bytes()const{return host_used_;}
            std::size_t pinned_duplicates()const{return pinned_duplicates_.size();}
            std::size_t images()const{return host_.size();}
        } cache;
        auto unique=std::make_shared<Image>(Image{17,true});
        auto duplicate=std::make_shared<Image>(Image{31,true});
        auto pageable=std::make_shared<Image>(Image{53,false});
        cache.host_index_insert(1,unique,false);cache.host_index_insert(2,duplicate,true);
        cache.host_index_insert(3,pageable,true);
        if(cache.bytes()!=101 || cache.pinned_duplicates()!=1 || cache.host_index_victim()!=2)
            throw std::runtime_error("encoded host cache lost bytes or GPU-backup eviction priority");
        bool rejected=false;
        try{cache.host_index_insert(1,unique,false);}catch(const std::logic_error&){rejected=true;}
        if(!rejected || cache.images()!=3 || cache.bytes()!=101)
            throw std::runtime_error("failed host admission corrupted cache accounting");
        cache.host_index_touch(2);
        if(cache.host_index_victim()!=3)throw std::runtime_error("host duplicate LRU ignored a hit");
        duplicate->registered=false;cache.host_index_duplicate(2,true);
        if(cache.pinned_duplicates())throw std::runtime_error("unregistered GPU backup retained its pinned index");
        cache.host_index_duplicate(2,false);cache.host_index_duplicate(3,false);
        if(cache.host_index_victim()!=1)throw std::runtime_error("last GPU eviction retained a duplicate marker");
        auto lease=unique;cache.host_index_erase(1);unique.reset();
        if(cache.bytes()!=84 || lease->bytes!=17)throw std::runtime_error("host eviction invalidated an outstanding transfer lease");
        cache.host_index_erase(2);cache.host_index_erase(3);
        if(cache.images() || cache.bytes())throw std::runtime_error("host cache did not release its byte charge");
    }
    {
        std::vector<int> phases;
        const auto result=mfq::cuda::execute_routed_moe([&]{phases.push_back(0);},[&]{phases.push_back(1);},
            [&]{phases.push_back(2);},[&]{phases.push_back(3);},[&]{phases.push_back(4);},
            [&]{phases.push_back(5);},[&]{phases.push_back(6);},[&]{phases.push_back(7);return 13;});
        if(result!=13 || phases!=std::vector<int>{0,1,2,3,4,5,6,7})
            throw std::runtime_error("resident, DMA joins and streamed work executed out of order");
        for(int fail=0;fail<8;++fail) {
            phases.clear();
            const auto phase=[&](int i){phases.push_back(i);if(i==fail)throw std::runtime_error("injected phase failure");};
            bool caught=false;
            try {
                mfq::cuda::execute_routed_moe([&]{phase(0);},[&]{phase(1);},[&]{phase(2);},[&]{phase(3);},
                    [&]{phase(4);},[&]{phase(5);},[&]{phase(6);},[&]{phase(7);return 0;});
            }catch(const std::runtime_error&){caught=true;}
            if(!caught || phases.size()!=std::size_t(fail+1))
                throw std::runtime_error("failed expert phase continued into a dependent read");
        }
    }
    {
        mfq::cuda::ExpertRouteUsage stable;
        stable.image(0,64);stable.image(1,64);
        for(int i=0;i<5;++i)stable.observe(0);
        for(int i=0;i<6;++i)stable.observe(1);
        stable.resident(0,true);
        stable.select(64,mfq::cuda::expert_replacement_margin);
        if(!stable.retained(0) || stable.retained(1))
            throw std::runtime_error("small heat changes replaced a resident image");
        stable.observe(1);stable.select(64,mfq::cuda::expert_replacement_margin);
        if(stable.retained(0) || !stable.retained(1))
            throw std::runtime_error("sufficient reuse gain failed to replace a resident image");
        stable.resident(0,false);stable.observe(0);stable.observe(0);stable.observe(0);
        stable.select(64,mfq::cuda::expert_replacement_margin);
        if(!stable.retained(0))throw std::runtime_error("evicted image retained a residency bonus");
        stable.select(0,mfq::cuda::expert_replacement_margin);
        if(stable.retained_images())throw std::runtime_error("stability exceeded the byte budget");
    }
    mfq::cuda::ExpertRouteUsage usage;
    Reference reference;
    compare(usage,reference,SIZE_MAX);
    std::mt19937 random(317);
    for(unsigned step=0;step<1600;++step) {
        const auto key=(std::uint64_t(random()%5)<<32)|(random()%257);
        switch(random()%5) {
        case 0: {
            const auto bytes=random()%9*37;
            usage.image(key,bytes);reference[key].bytes=bytes;break;
        }
        case 1:
            usage.observe(key);reference[key].uses+=1.f;break;
        case 2: {
            const auto uses=random()%101,rows=random()%17;
            usage.prefill(key,uses,rows);
            if(rows)reference[key].uses=float(uses)*4.f/(.3f*float(rows));break;
        }
        case 3: {
            const auto source=random()%5;usage.reset_prefill(source);
            for(auto& [id,demand]:reference)if(id>>32==source)demand.uses=0;break;
        }
        default: {
            const auto windows=random()%9;usage.decay(windows);
            const auto factor=mfq::cuda::expert_usage_decay(windows);
            for(auto& [_,demand]:reference)demand.uses*=factor;break;
        }
        }
        if(step%7==0) {
            for(const auto budget:{std::size_t(0),std::size_t(1),std::size_t(773),std::size_t(5903),SIZE_MAX})
                compare(usage,reference,budget);
            auto copy=usage;compare(copy,reference,1103);
            mfq::cuda::ExpertRouteUsage assigned;assigned=copy;
            auto moved=std::move(assigned);compare(moved,reference,337);
            assigned=usage;compare(assigned,reference,SIZE_MAX);
        }
    }
    usage.decay(1000000);for(auto& [_,demand]:reference)demand.uses=0;
    compare(usage,reference,SIZE_MAX);
    std::cout<<"Exact routed-cache ranking passed\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
