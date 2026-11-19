#pragma once

#include <cstddef>

class SessionBlockReader;
class SessionBlockWriter;

namespace tyloi::state::qsa { struct Session; }

namespace tyloi::models::qwen4_exp {
void write_ram_page(SessionBlockWriter&, const state::qsa::Session&, std::size_t start, std::size_t count, bool exact);
void read_ram_page(SessionBlockReader&, state::qsa::Session&, std::size_t start, std::size_t count, bool exact);
} // namespace tyloi::models::qwen4_exp
