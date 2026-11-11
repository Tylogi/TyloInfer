#pragma once

#include "source.h"

namespace tyloi {

class NintRows;

const TensorMetadata& require_tensor(const source::Source& source, std::string_view name);
bool has_tensor(const source::Source& source, std::string_view name) noexcept;
bool has_tensor_prefix(const source::Source& source, std::string_view prefix);
std::vector<std::uint8_t> read_tensor(const source::Source& source, std::string_view name);
std::vector<std::uint8_t> read_asset(const source::Source& source, std::string_view name);
std::string read_asset_text(const source::Source& source, std::string_view name);

} // namespace tyloi

std::shared_ptr<tyloi::NintRows> load_nint_row_table(
    const tyloi::source::Source& source, const std::string& name, bool prefer_mapping = false);
