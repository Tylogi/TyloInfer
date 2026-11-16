#include "cli.h"

#include <iostream>

namespace mfq::cuda::internal {

int with_command_errors(const std::function<int()>& fn) {
    try {
        return fn();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

} // namespace mfq::cuda::internal
