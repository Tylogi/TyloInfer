#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mfq::cuda {
struct RoutedCompletionPosition {
    int device=-1;
    std::uintptr_t stream=0;
    std::uint64_t sequence=0;
};

// Cache owners serialize event records and position updates under one lock.
class RoutedCompletionClock {
    std::uint64_t sequence_=0;
public:
    RoutedCompletionPosition record(int device,std::uintptr_t stream) {
        if(device<0 || sequence_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("invalid routed completion sequence");
        return {device,stream,++sequence_};
    }
};

// Within one pass, a completed fence proves all earlier records on its stream
// complete. An incomplete fence keeps later records pending without querying.
// Untracked or re-recorded external events must supply sequence zero.
class RoutedCompletionProbe {
    struct Lane {
        int device;
        std::uintptr_t stream;
        std::uint64_t completed=0,pending=0;
    };
    std::vector<Lane> lanes_;
public:
    template<class Query> bool ready(RoutedCompletionPosition position,Query query) {
        if(!position.sequence)return query();
        if(position.device<0)throw std::invalid_argument("routed completion has no device");
        auto found=std::find_if(lanes_.begin(),lanes_.end(),[&](const auto& lane) {
            return lane.device==position.device && lane.stream==position.stream;
        });
        if(found==lanes_.end())found=lanes_.insert(lanes_.end(),Lane{position.device,position.stream});
        if(position.sequence<=found->completed)return true;
        if(found->pending && position.sequence>=found->pending)return false;
        if(query()){found->completed=position.sequence;return true;}
        found->pending=position.sequence;return false;
    }
};
} // namespace mfq::cuda
