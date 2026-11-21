#pragma once
#include "mfq_cuda_context.h"
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <vector>

namespace mfq::cuda {
// Owners serialize access and retain each stream until this table is retired.
// Records on one explicit stream may reuse its event: a later completed
// record also proves every earlier copy on that lane complete. Never share
// an event between independent producer streams.
class RoutedCompletionEvents {
    std::unordered_map<int,std::unordered_map<cudaStream_t,std::shared_ptr<Event>>> lanes_;
public:
    void prepare(int device,cudaStream_t stream) {
        auto& lane=lanes_[device][stream];
        if(!lane){const DeviceGuard guard(device);lane=std::make_shared<Event>(cudaEventDisableTiming);}
    }
    std::shared_ptr<Event> at(int device,cudaStream_t stream)const {
        return lanes_.at(device).at(stream);
    }
};

// Destroying a completed event may enter a blocking driver operation. Keep
// its handle while a graph waits for host work, then take it after the graph
// finishes and release it outside the cache lock. Payload leases retire at
// their ordinary DMA boundary; only the small event owner is deferred.
class RoutedEventRetirement {
    std::vector<std::shared_ptr<Event>> events_;
public:
    void retain(const std::shared_ptr<Event>& event) {
        if(event && (events_.empty() || events_.back()!=event))events_.push_back(event);
    }
    std::size_t size()const noexcept{return events_.size();}
    std::vector<std::shared_ptr<Event>> take()noexcept {
        std::vector<std::shared_ptr<Event>> result;result.swap(events_);return result;
    }
};
} // namespace mfq::cuda
