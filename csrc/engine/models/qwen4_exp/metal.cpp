#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/qwen4_exp/batch.h"

namespace tyloi::models::qwen4_exp {

PrefillPlan Model::prefill_plan(const models::State&, const PrefillOptions&) const { return {}; }

StepSequence<Prefill> Model::prefill(Input, models::State&, const ForwardOptions&,
        const PrefillOptions&) const { return {}; }

StepSequence<Decoded> Model::decode(Input, models::State&, const ForwardOptions&,
        const DecodeOptions&) const { return {}; }

StepSequence<Decoded> Model::warmup(models::State&, const DecodeOptions&) const { return {}; }

std::unique_ptr<models::Batch> Model::make_batch(const BatchConfig& config) const {
    return std::make_unique<qwen4_exp::Batch>(const_cast<Model&>(*this), config, true);
}

} // namespace tyloi::models::qwen4_exp
