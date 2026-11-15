#pragma once

namespace tyloi::commands {

int run_runtime(int argc, char** argv);

} // namespace tyloi::commands

namespace tyloi::cuda::commands {
int run_diagnostics(int argc, char** argv);

} // namespace tyloi::cuda::commands
