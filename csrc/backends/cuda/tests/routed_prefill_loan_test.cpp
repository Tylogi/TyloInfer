#include "../runtime/routed_prefill_loan.h"
#include <cassert>
#include <vector>

namespace {
struct Stream {int value;int stream()const noexcept{return value;}};
struct Storage {
    std::shared_ptr<std::vector<unsigned char>> owner;
    void* data_ptr()const noexcept{return owner->data();}
    std::size_t nbytes()const noexcept{return owner->size();}
};
struct Context {
    std::shared_ptr<void> backing;
    bool active=false,enabled=false,retiring=false;
    unsigned live=0,begins=0,ends=0;
    void begin_workspace_loan(const std::vector<std::pair<void*,std::size_t>>&,int,std::shared_ptr<void> owner) {
        if(active)throw std::logic_error("duplicate loan");
        active=enabled=true;retiring=false;backing=std::move(owner);++begins;
    }
    void suspend_workspace_loan(){enabled=false;}
    void end_workspace_loan() {
        if(live)throw std::logic_error("live tensor");
        active=enabled=false;backing.reset();++ends;
    }
    void retire_workspace_loan()noexcept {
        enabled=false;retiring=true;if(!live){active=false;backing.reset();}
    }
    void release() {
        assert(live);--live;if(!live && retiring){active=false;backing.reset();}
    }
};
using Loan=mfq::cuda::RoutedPrefillLoanState<Context,Storage,Stream>;
}

int main() {
    using namespace mfq::cuda;
    constexpr std::size_t MiB=1ULL<<20;
    // Match real per-allocation placement, including the two independent
    // layer-ring slots and predictor Gate/Up versus Down images.
    assert(!routed_workspace_allocations_fit({128*MiB},{64*MiB,64*MiB},0));
    assert(routed_workspace_allocations_fit({64*MiB,64*MiB},{64*MiB,64*MiB},0));
    assert(!routed_workspace_allocations_fit({64*MiB,64*MiB},{64*MiB},64*MiB-1));
    assert(routed_workspace_allocations_fit({64*MiB,64*MiB},{64*MiB},64*MiB));
    assert(routed_workspace_allocations_fit({256,512},{512,256},0));
    assert(!routed_workspace_allocations_fit({257,1},{512},0));
    assert(routed_workspace_allocations_fit({257,1},{512},1));
    assert(routed_workspace_allocations_fit({0},{},0));
    assert(!routed_workspace_allocations_fit({SIZE_MAX},{SIZE_MAX},0));
    assert(routed_workspace_allocations_fit({SIZE_MAX},{},SIZE_MAX));
    // Several independent 64-MiB holes cannot satisfy a 256-MiB request.
    assert(routed_workspace_full_borrow_request(256*MiB,64*MiB)==0);
    assert(routed_workspace_borrow_request(256*MiB,16*MiB,128*MiB,64*MiB)==0);
    assert(routed_workspace_borrow_request(256*MiB,256*MiB,256*MiB,64*MiB)==0);
    const auto partial=routed_workspace_borrow_request(256*MiB,255*MiB,256*MiB,64*MiB);
    assert(partial==MiB+256 && partial<64*MiB);
    assert(routed_workspace_full_borrow_request(32*MiB,64*MiB)==32*MiB+256);
    assert(routed_workspace_region_bytes(reinterpret_cast<void*>(0x1010),4096)==3840);
    assert(routed_workspace_region_bytes(reinterpret_cast<void*>(0x1000),4096)==4096);
    {
        auto context=std::make_shared<Context>();
        auto bytes=std::make_shared<std::vector<unsigned char>>(4096);
        std::weak_ptr<std::vector<unsigned char>> lifetime=bytes;
        auto loan=Loan::begin(context,Stream{7},Storage{bytes});bytes.reset();
        assert(loan->matches(context,Stream{7}) && !loan->matches(context,Stream{8}));
        // Nested reservations share one owner and one Context begin.
        auto nested=loan;loan.reset();assert(context->active && context->begins==1);
        bool wrong_thread=false;
        std::thread other([&] {
            assert(!nested->matches(context,Stream{7}));
            try{nested->suspend();}catch(const std::logic_error&){wrong_thread=true;}
        });
        other.join();assert(wrong_thread && context->enabled);
        nested->suspend();assert(!context->enabled && !lifetime.expired());
        nested->finish();assert(!context->active && context->ends==1 && lifetime.expired());
    }
    {
        auto context=std::make_shared<Context>();
        auto bytes=std::make_shared<std::vector<unsigned char>>(4096);
        std::weak_ptr<std::vector<unsigned char>> lifetime=bytes;
        auto loan=Loan::begin(context,Stream{7},Storage{bytes});bytes.reset();context->live=1;
        bool rejected=false;
        try{loan->finish();}catch(const std::logic_error&){rejected=true;}
        assert(rejected && loan->active() && context->active && !lifetime.expired());
        // Destruction of the cache/loan owner must not destroy live outputs.
        // Context retains backing until its last live suballocation returns.
        loan.reset();assert(context->retiring && !context->enabled && !lifetime.expired());
        context->release();assert(!context->active && lifetime.expired());
    }
    {
        auto context=std::make_shared<Context>();
        auto bytes=std::make_shared<std::vector<unsigned char>>(4096);
        auto loan=Loan::begin(context,Stream{7},Storage{bytes});context->live=1;
        try{loan->finish();assert(false);}catch(const std::logic_error&){}
        context->live=0;loan->finish();assert(!loan->active() && !context->active);
    }
}
