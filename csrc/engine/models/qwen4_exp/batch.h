#pragma once

#include "engine/models/batch.h"

#include <memory>

namespace tyloi::cuda::qwen4_exp {
struct Model;
std::unique_ptr<tyloi::models::Batch> make_batch(Model&, const tyloi::models::BatchConfig&);
}
