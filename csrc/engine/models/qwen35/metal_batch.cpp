#include "engine/models/qwen35/model.h"
#include "engine/models/batch.h"

namespace tyloi::models::qwen35 {

std::unique_ptr<models::Batch> Model::make_batch(const BatchConfig&) const { return {}; }

} // namespace tyloi::models::qwen35
