#pragma once

#include <string_view>

namespace tyloi {

bool supported_tensor_dtype(std::string_view dtype) noexcept;

} // namespace tyloi
