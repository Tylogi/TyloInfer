#include "engine/models/qwen4_exp/ops.h"

namespace tyloi::models::qwen4_exp {

PrefillPlan Model::prefill_plan(const models::State&, const PrefillOptions&) const { return {}; }

StepSequence<Prefill> Model::prefill(Input, models::State&, const ForwardOptions&,
        const PrefillOptions&) const { return {}; }

StepSequence<Decoded> Model::decode(Input, models::State&, const ForwardOptions&,
        const DecodeOptions&) const { return {}; }

StepSequence<Decoded> Model::warmup(models::State&, const DecodeOptions&) const { return {}; }

std::unique_ptr<Batch> Model::make_batch(const BatchConfig&) const { return {}; }

} // namespace tyloi::models::qwen4_exp
