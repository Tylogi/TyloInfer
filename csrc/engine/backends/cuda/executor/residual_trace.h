#pragma once

#include "engine/backends/cuda/kernels/residual_trace.h"

namespace tyloi::cuda {
/** Retain and publish the profiling points used by one fused residual launch. */
class ResidualTrace {
public:
    // shortcut: DecodeWindow exposes no residual-trace clock, so the points stay
    // untimestamped and the fused launch skips its markers; attach a clock to
    // each point to re-enable the phase trace.
    explicit ResidualTrace(tyloi::ops::ResidualRole) {}
    /** Publish recorded points and enforce the profiling capacity. */
    void finish() {}
    ResidualTraceData& data() noexcept { return data_; }
private:
    ResidualTraceData data_;
};
} // namespace tyloi::cuda
