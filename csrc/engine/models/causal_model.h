#pragma once

#include "engine/tensor.h"
#include "engine/metrics.h"
#include "engine/models/metadata.h"
#include "engine/models/input.h"
#include "engine/models/batch.h"
#include "engine/models/state.h"
#include "engine/generation/sequence.h"

#include <cstdint>
#include <memory>
#include <vector>

struct MfqMultimodalInput;

namespace tyloi::models {

enum class LogitPrecision { model, input };

class CausalModel {
public:
    virtual ~CausalModel() = default;
    virtual Info info() const = 0;
    virtual MfqModelCapabilities capabilities() const = 0;
    virtual tyloi::engine::Metrics metrics() const = 0;
    // Borrowed from this model; predictor state belongs to the request.
    virtual Mtp* predictor() const noexcept { return nullptr; }
    virtual std::unique_ptr<State> make_state() const = 0;
    virtual tyloi::StepSequence<Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, State& state) const = 0;
    virtual Tensor embedding(const Tensor& ids, int primary = 0) const = 0;
    virtual Tensor normalize(const Tensor& hidden, std::int64_t batch, std::int64_t tokens) const = 0;
    virtual Tensor softcap(const Tensor& logits) const = 0;
    virtual Tensor logits(const Tensor& hidden, bool capped = true,
        LogitPrecision precision = LogitPrecision::model) const = 0;
    virtual bool supports_speculation() const { return false; }
    virtual bool supports_suffix_speculation() const noexcept { return false; }
    virtual Tensor forward(Input input, State& state, const ForwardOptions& options = {}) const = 0;
    virtual PrefillPlan prefill_plan(const State&, const PrefillOptions&) const { return {}; }
    virtual StepSequence<Prefill> prefill(Input, State&, const ForwardOptions&, const PrefillOptions&) const { return {}; }
    virtual StepSequence<Decoded> decode(Input, State&, const ForwardOptions&, const DecodeOptions&) const { return {}; }
    virtual StepSequence<Decoded> warmup(State&, const DecodeOptions&) const { return {}; }
    virtual std::unique_ptr<Batch> make_batch(const BatchConfig&) const { return {}; }
};

} // namespace tyloi::models
