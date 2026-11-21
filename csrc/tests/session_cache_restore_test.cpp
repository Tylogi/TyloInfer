#include "runtime_config.h"
#include "session_cache.h"

#include <optional>
#include <stdexcept>
#include <iostream>

namespace {
struct PredictorState { int last_target_hidden=0; };
struct Snapshot {
    std::vector<int64_t> tokens;
    std::string input_key;
    std::size_t bytes=0;
    std::optional<PredictorState> mtp;
};
struct Predictor {
    bool supports_session_state()const{return true;}
    void restore_session_state(const PredictorState&){}
    void reset(int){}
};
struct Backend {
    using Snapshot=::Snapshot;
    using Predictor=::Predictor;
    using StateError=std::invalid_argument;
    struct Restore {std::size_t tokens=0;int mtp_last_target_hidden=0;};
    static Snapshot decode(const std::vector<mfq::cache::PagedPrefixPayload>&,
            const std::vector<int64_t>& tokens,std::size_t){return {tokens,"",64,{}};}
    static mfq::cache::PagedPrefixPayload encode(const Snapshot&,std::size_t,std::size_t) {
        return std::make_shared<const std::vector<std::uint8_t>>(64);
    }
};
struct Model {
    bool fail=true;
    int resets=0;
    bool supports_text_session_state()const{return true;}
    bool supports_paged_text_session_state()const{return true;}
    void restore_text_session_state(const Snapshot&) {
        if(fail)throw MfqResourceExhausted("restore capacity exhausted");
    }
    void reset(int){++resets;}
};
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
}
int main()try {
    mfq::engine::SessionCacheConfig sessions;
    mfq::engine::PrefixCacheConfig prefix;
    mfq::engine::SessionCache<Backend> cache(sessions,prefix);
    cache.store("shared",{{1,2},"",64,{}});
    Model model;
    bool rejected=false;
    try{cache.restore_best(model,nullptr,"shared",{1,2,3},2);}
    catch(const MfqResourceExhausted&){rejected=true;}
    require(rejected && model.resets==1,"Resource failure became a cold prefill");
    model.fail=false;
    require(cache.restore_best(model,nullptr,"shared",{1,2,3},2).tokens==2,
        "Resource failure discarded a valid checkpoint");

    mfq::cache::PagedPrefixCacheConfig disk;
    disk.block_size_tokens=2;disk.max_hot_bytes=4096;disk.max_disk_bytes=0;
    disk.cache_dir=std::filesystem::current_path()/"session-cache-restore-test-data";
    disk.compatibility_key="capacity-retry";
    auto pages=std::make_shared<mfq::cache::PagedPrefixCache>(disk);
    mfq::engine::SessionCache<Backend> paged(sessions,prefix,pages);
    paged.store("shared",{{1,2},"",64,{}});
    model.fail=true;rejected=false;
    try{paged.restore_best(model,nullptr,"shared",{1,2,3},2);}
    catch(const MfqResourceExhausted&){rejected=true;}
    require(rejected,"Paged resource failure became a cold prefill");
    model.fail=false;
    require(paged.restore_best(model,nullptr,"shared",{1,2,3},2).tokens==2,
        "Paged resource failure invalidated immutable prefix data");

    prefix.hot_bytes=64;
    mfq::engine::SessionCache<Backend> combined(sessions,prefix,pages,true,0,true);
    combined.store("long",{{1,2,3,4},"",128,{}});
    combined.store("short",{{1,2},"",64,{}});
    require(combined.restore_best(model,nullptr,"short",{1,2,3,4,5},4).tokens==4,
        "A short hot snapshot hid a longer durable prefix");
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
