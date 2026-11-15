#pragma once

#include "../requests/output.h"
#include "sequence.h"
#include <exception>
#include <utility>

namespace tyloi::engine {

using tyloi::StepState;
using Generation = tyloi::StepSequence<EventData>;

// Resource owners release exactly once, including coroutine destruction. The
// scheduler observes teardown failures before publishing a terminal event.
template <class Release> struct ExecutionCleanup {
    std::exception_ptr& failure;
    Release release;
    bool pending = true;
    void finish() {
        if (!std::exchange(pending, false)) return;
        try { release(); }
        catch (...) { failure = std::current_exception(); throw; }
    }
    ~ExecutionCleanup() { try { finish(); } catch (...) {} }
};
template <class Release>
ExecutionCleanup(std::exception_ptr&, Release) -> ExecutionCleanup<Release>;

} // namespace tyloi::engine
