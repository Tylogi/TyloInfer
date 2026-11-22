#pragma once

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace mfq::test {

/** Returns the absolute error and rejects nonfinite numerical results. */
template <typename Actual, typename Expected>
inline auto checked_absolute_error(Actual actual, Expected expected)
    -> std::common_type_t<Actual, Expected> {
    using Value = std::common_type_t<Actual, Expected>;
    const Value error = std::abs(Value(actual) - Value(expected));
    if (!std::isfinite(error)) {
        throw std::runtime_error("nonfinite numerical result");
    }
    return error;
}

} // namespace mfq::test
