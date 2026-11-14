#pragma once
#include <string_view>
namespace tyloi::metal {
int gguf_format_id(std::string_view dtype) noexcept;
bool gguf_staged_enabled();
} // namespace tyloi::metal
