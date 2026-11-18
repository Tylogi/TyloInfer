#pragma once

#include <future>

namespace tyloi::weights {

void submit_row_read(std::packaged_task<void()> job);

} // namespace tyloi::weights
