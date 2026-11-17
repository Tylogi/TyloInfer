#pragma once

#include "engine/tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <cstdint>
#include <string>
#include <vector>

namespace tyloi::models {
struct Mtp;
class State;
}
namespace tyloi::state { class Sessions; }

namespace tyloi::engine {

/** Capture and store one committed target session boundary. */
std::size_t save_session(models::State&, state::Sessions&, const std::string& session,
    const std::vector<std::int64_t>& history, const std::string& input_key = {},
    models::Mtp* mtp = nullptr, const Tensor& last_hidden = {});

} // namespace tyloi::engine
