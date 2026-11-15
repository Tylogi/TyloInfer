#pragma once

#include "engine/load.h"

namespace tyloi::cuda {
std::unique_ptr<engine::Engine> load(engine::EngineOptions options);
void reload(engine::Engine& engine, engine::EngineOptions options);
} // namespace tyloi::cuda
