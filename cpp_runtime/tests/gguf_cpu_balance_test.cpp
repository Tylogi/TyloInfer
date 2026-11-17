#include "../backends/cuda/ops/gguf/gguf_cpu_balance.h"
#include "../backends/cuda/runtime/routed_admission_executor.h"
#include "../backends/cuda/runtime/routed_layer_admission.h"
#include "../backends/cuda/runtime/routed_workspace_plan.h"
#include <array>
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <vector>

void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
void check_risk_only_reprobe() {
    GgufCpuBalanceModel model;
    model.on=true;model.ncpu=8;model.ngpu_p=4;model.max_observed_cpu=3;
    model.a=.02;model.c=.12;model.d=.01;model.g0=.03;model.g=.06;
    // A transient delay may reject CPU work even though its fitted mean is
    // faster. Preserve that protection, but collect fresh evidence on time.
    model.tail_variance=100.;
    for(int i=0;i<63;++i)require(model.cpu_count(3)==0,"CPU tail protection was bypassed between probes");
    require(model.cpu_count(3)==1,"old tail uncertainty postponed a cheaper CPU lane's recovery probe");
}
void check_relative_tail_risk() {
    const auto trained=[] {
        GgufCpuBalanceModel model;
        for(int i=0;i<600;++i) {
            const int cpu=i%5,pcie=(i/5)%6;
            model.observe(cpu,pcie,cpu?.02+.12*cpu+.01*pcie:0.,.03+.06*pcie);
        }
        model.on=true;return model;
    };
    auto gpu_delayed=trained();
    gpu_delayed.observe(1,2,.16,1.15);
    require(gpu_delayed.tail_variance<1e-8,"GPU-only delay was priced as unexpected CPU lateness");
    auto cpu_delayed=trained();
    cpu_delayed.observe(1,2,1.16,.15);
    require(cpu_delayed.tail_variance>.01,"unexpected CPU lateness lost its risk penalty");
    auto shared_delay=trained();
    shared_delay.observe(1,2,1.16,1.15);
    require(shared_delay.tail_variance<1e-6,"a delay shared by both lanes became CPU tail risk");
}
void check_admission_worker() {
    RoutedAdmissionExecutor worker;
    std::thread::id identity;int completed=0;
    std::vector<std::future<void>> jobs;
    for(int i=0;i<32;++i)jobs.push_back(worker.submit([&,i] {
        require(completed++==i,"cache admission changed submission order");
        if(i==0)identity=std::this_thread::get_id();
        require(identity==std::this_thread::get_id(),"cache admission recreated its host thread");
    }));
    for(auto& job:jobs)job.get();
    auto failed=worker.submit([]{throw std::runtime_error("admission failure");});
    bool propagated=false;
    try{failed.get();}catch(const std::runtime_error&){propagated=true;}
    require(propagated,"cache admission swallowed an asynchronous failure");
    worker.submit([&]{++completed;}).get();
    require(completed==33,"cache admission worker did not recover after a job failed");
    std::atomic<int> drained=0;
    std::promise<void> gate;const auto ready=gate.get_future().share();
    jobs.clear();
    {
        RoutedAdmissionExecutor retiring;
        jobs.push_back(retiring.submit([&]{ready.wait();++drained;}));
        for(int i=0;i<32;++i)jobs.push_back(retiring.submit([&]{++drained;}));
        gate.set_value();
    }
    require(drained==33,"cache unload did not drain accepted admission jobs");
    for(auto& job:jobs)job.get();
    {
        RoutedAdmissionExecutor bounded(2);
        std::promise<void> running,release;
        const auto gate=release.get_future().share();
        auto first=bounded.submit([&]{running.set_value();gate.wait();});
        running.get_future().get();
        auto second=bounded.submit([]{}),third=bounded.submit([]{});
        bool rejected=false;
        try{bounded.submit([]{});}catch(const std::logic_error&){rejected=true;}
        release.set_value();first.get();second.get();third.get();
        require(rejected,"bounded prefill reader accepted more work than its packet ring");
        bounded.submit([]{}).get();
    }
    bool invalid=false;
    try{RoutedAdmissionExecutor empty(0);}catch(const std::invalid_argument&){invalid=true;}
    require(invalid,"invalid worker capacity started an unusable thread");
}
void check_admission_handoff() {
    RoutedAdmission admission;
    std::promise<void> started,release,entering,attempted;
    const auto gate=release.get_future().share();
    std::atomic<bool> completed=false;
    admission.submit([&]{started.set_value();gate.wait();completed=true;});
    started.get_future().get();
    unsigned readers=0,updates=0;
    auto reader=std::async(std::launch::async,[&] {
        admission.transition([&](auto& state) {
            entering.set_value();state.wait();
            require(completed,"cache reader started before admission completed");
            ++readers;
        });
    });
    entering.get_future().get();
    auto writer=std::async(std::launch::async,[&] {
        attempted.set_value();
        admission.transition([&](auto& state) {
            state.wait();
            if(!readers)state.submit([&]{++updates;});
        });
    });
    attempted.get_future().get();release.set_value();
    reader.get();writer.get();admission.wait();
    require(readers==1 && updates==0,"cache admission crossed the next reader handoff");
    admission.submit([]{throw std::runtime_error("admission failure");});
    bool propagated=false;
    try{admission.wait();}catch(const std::runtime_error&){propagated=true;}
    require(propagated,"cache handoff swallowed an asynchronous failure");
    admission.wait();
    admission.submit([&]{++updates;});admission.wait();
    require(updates==1 && !admission.pending(),"failed admission poisoned a later handoff");
    bool rejected=false;
    try{admission.transition([](auto&){throw std::invalid_argument("rejected cache transition");});}
    catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"cache transition swallowed a validation failure");
    admission.submit([&]{++updates;});admission.wait();
    require(updates==2,"rejected cache transition retained its handoff lock");
    std::atomic<unsigned> drained=0;
    {RoutedAdmission retiring;retiring.submit([&]{++drained;});}
    require(drained==1,"cache destruction released storage before its admission finished");
}
void check_gated_admission_handoff() {
    RoutedAdmission admission;
    // MFQ may prepare a cohort transaction before graph completion permits
    // publication. Its release gate must work while a cache reader waits
    // inside the shared admission transition.
    for(const bool cancel:{false,true}) {
        std::promise<void> prepared,finish,entering;
        const auto gate=finish.get_future().share();
        std::atomic<unsigned> phase=0,pins=1;
        admission.transition([&](auto& state) {
            require(!state.pending(),"new gated admission retained an old future");
            state.submit([&] {
                phase=1;prepared.set_value();gate.wait();
                if(cancel) {
                    phase=2;pins=0; // Rollback retires leases before failure.
                    throw std::runtime_error("cancelled cohort publication");
                }
                phase=3;pins=0;
            });
        });
        prepared.get_future().get();
        require(admission.pending() && phase==1 && pins==1,
            "preparation was mistaken for completed cohort publication");
        auto reader=std::async(std::launch::async,[&] {
            return admission.transition([&](auto& state) {
                entering.set_value();bool failed=false;
                try{state.wait();}catch(const std::runtime_error&){failed=true;}
                require(pins==0 && phase==(cancel?2u:3u),
                    "admission returned before publication/rollback retired its leases");
                require(!state.pending(),"gated admission failure retained its future");
                return failed;
            });
        });
        entering.get_future().get();finish.set_value();
        require(reader.get()==cancel,"gated admission lost its publication result");
        admission.wait();
    }
    unsigned recovered=0;
    admission.submit([&]{++recovered;});admission.wait();
    require(recovered==1,"cancelled cohort handoff poisoned later admission");
}
void check_layer_admission() {
    struct View {
        std::array<std::uint64_t,3> sources;
        std::uint32_t flag=1;
        bool valid=true;
        std::uint32_t* admission_host(){return &flag;}
    };
    mfq::cuda::RoutedLayerAdmission<View,int,int> epoch;
    epoch.views.push_back(std::make_shared<View>(View{{1,2,3}}));
    epoch.views.push_back(std::make_shared<View>(View{{4,5,6}}));
    epoch.arm();
    for(const auto& view:epoch.views)require(view->flag==0,"layer admission exposed incomplete uploads");
    epoch.release_unchanged([](const auto& view){return view.sources[0]==1;},[](auto* flag){*flag=1;});
    require(epoch.views[0]->flag==0 && epoch.views[1]->flag==1,"unchanged layers waited for unrelated uploads");
    epoch.require_active();
    unsigned invalidated=0,published=0;
    epoch.abort(std::make_exception_ptr(std::out_of_range("failed expert image")),[&](auto& view) {
        view.valid=false;++invalidated;
    },[&](auto* flag) {
        require(invalidated==epoch.views.size(),"failed admission released readers before invalidating every view");
        *flag=1;++published;
    });
    require(published==2 && epoch.cancelled,"cancelled admission left a reader waiting");
    epoch.abort({},[&](auto&){++invalidated;},[&](auto*){++published;});
    require(invalidated==2 && published==2,"cancelled admission repeated its invalidation");
    bool propagated=false;
    try{epoch.require_active();}catch(const std::out_of_range&){propagated=true;}
    require(propagated,"layer admission replaced the original source error");
    unsigned pauses=0;
    mfq::cuda::wait_routed_layer_admission([&]{return pauses==17;},[&]{++pauses;},
        []{throw std::runtime_error("unexpected admission timeout");});
    require(pauses==17,"layer wait did not release at its own readiness flag");
    bool timed_out=false;
    try {
        mfq::cuda::wait_routed_layer_admission([]{return false;},[]{},
            [&]{timed_out=true;},std::chrono::steady_clock::duration::zero());
    }catch(const std::runtime_error&){}
    require(timed_out,"layer admission timeout did not invoke cancellation");
}
void check_workspace_plan() {
    using mfq::cuda::routed_decode_workspace_reserve;
    using mfq::cuda::routed_cache_capacity;
    constexpr auto mib=std::size_t(1)<<20;
    require(routed_decode_workspace_reserve(13*mib,{}, {},32*mib)==1325*mib,
        "unprepared decoder changed its workspace and transfer allowance");
    require(routed_decode_workspace_reserve(13*mib,0,{},32*mib)==745*mib,
        "prepared decoder lost its driver residency allowance");
    require(routed_decode_workspace_reserve(13*mib,2048*mib,{},32*mib)==2093*mib,
        "large transient workspace was clipped to a default allowance");
    require(routed_decode_workspace_reserve(13*mib,2048*mib,0,32*mib)==13*mib,
        "explicit zero workspace reserve was replaced by automatic defaults");
    require(routed_decode_workspace_reserve(13*mib,2048*mib,256*mib,32*mib)==269*mib,
        "explicit workspace reserve was not shared by every encoding");
    require(routed_decode_workspace_reserve(SIZE_MAX,{},1)==SIZE_MAX,
        "workspace byte count wrapped on overflow");
    require(routed_cache_capacity(100,20,80,70)==130,"free headroom was not returned to the expert cache");
    require(routed_cache_capacity(100,20,30,70)==80,"workspace deficit remained charged to resident experts");
    require(routed_cache_capacity(100,20,30,200)==0,"workspace deficit produced a negative cache capacity");
}
int main() {
    try {
        check_admission_worker();
        check_admission_handoff();
        check_gated_admission_handoff();
        check_layer_admission();
        check_workspace_plan();
        check_relative_tail_risk();
        check_risk_only_reprobe();
        GgufCpuBalanceModel model;
        require(model.cpu_count(0)==0,"empty route set scheduled CPU work");
        for(int i=0;i<4;++i)require(model.cpu_count(1)==(i%2==0?1:0),"sole-miss calibration did not measure both devices");
        model={};model.on=true;model.ncpu=8;model.ngpu_p=4;model.max_observed_cpu=3;
        model.a=.02;model.c=.12;model.d=.01;model.g0=.03;model.g=.06;
        require(model.cpu_count(1)==0,"CPU took a sole miss whose complete GPU cost is lower");
        require(model.cpu_count(3)==1,"cost balance did not hide CPU work behind the remaining uploads");
        model.tail_variance=.04;
        require(model.cpu_count(3)==0,"a slow CPU tail was hidden by its cheaper mean cost");
        model.tail_variance=0;
        model.explore=63;require(model.cpu_count(1)==1,"bounded exploration stopped measuring a rejected lane");
        model.a=8.;model.c=.01;model.d=0.;model.g0=.02;model.g=.01;model.explore=0;
        int expensive_probes=0;double extra_ms=0.;
        for(int i=0;i<20000;++i) {
            const int cpu=model.cpu_count(1);expensive_probes+=cpu>0;
            extra_ms+=cpu?model.price(1,0)-model.price(1,1):0.;
        }
        require(expensive_probes>0 && extra_ms<20000.*model.price(1,1)*.016,
            "rejected CPU wake probes consumed more than their measured exploration budget");
        model.a=.001;model.c=.001;model.explore=0;
        require(model.cpu_count(1)==1,"a previously slow CPU lane could not recover after its measured cost fell");
        model={};model.on=true;model.ncpu=8;model.ngpu_p=4;model.max_observed_cpu=1;
        model.c=.001;model.g=1.;
        require(model.cpu_count(20)<=2,"one-expert calibration extrapolated to a whole CPU layer");
        model={};
        for(int i=0;i<600;++i) {
            const int cpu=i%5,pcie=(i/5)%6;
            model.observe(cpu,pcie,cpu?.02+.12*cpu+.01*pcie:0.,.03+.06*pcie);
        }
        model.on=true;
        require(model.ready(),"both fits did not become usable");
        require(std::abs(model.a-.02)<.01 && std::abs(model.c-.12)<.01 && std::abs(model.d-.01)<.01 &&
            std::abs(model.g0-.03)<.001 && std::abs(model.g-.06)<.001,"measured-cost fit changed its CPU/PCIe slopes");
        require(model.cpu_count(1)==0 && model.cpu_count(3)==1,"learned costs selected a slower split");
        model={};
        for(int i=0;i<300;++i) {
            const int cpu=i%3,pcie=3-cpu;
            model.observe(cpu,pcie,cpu?.02+.12*cpu+.01*pcie:0.,.03+.06*pcie);
        }
        model.on=true;
        require(std::abs(model.a+model.c+2*model.d-.16)<.002 &&
            std::abs(model.a+2*model.c+model.d-.27)<.002,
            "constant miss count made constrained CPU predictions drift");
        require(model.cpu_count(3)==1,"collinear counts made the CPU/PCIe split slower");
        model={};
        for(int i=0;i<300;++i) {
            const int pcie=3+i%6;
            model.sample(0,pcie,0.,.02*pcie-.05);
        }
        model.solve();
        require(model.g0==0 && model.g>0 && model.g<.02,
            "negative GPU intercept was independently clipped without refitting its slope");
        model.on=true;model.ncpu=8;model.max_observed_cpu=3;
        model.a=.045;model.c=.02;model.d=0.;
        require(model.cpu_count(4)==0,"biased GPU cost assigned a slower CPU expert");
        std::cout<<"GGUF CPU/PCIe measured-cost balance, calibration and bounded exploration PASS\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
