#pragma once

#include "source.h"

#include <optional>
#include <string_view>

namespace tyloi::source {

std::optional<HadamardTransform> hadamard(const Source& source, std::string_view name);

} // namespace tyloi::source
