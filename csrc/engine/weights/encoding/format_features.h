#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tyloi {
extern const bool mfq_compat_enabled;

template<class Function>
std::invoke_result_t<Function> with_mfq_compat(Function&& function) {
    if (mfq_compat_enabled) return std::forward<Function>(function)();
    else throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
}

}
