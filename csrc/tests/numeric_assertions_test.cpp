#include "numeric_assertions.h"

#include <cmath>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Value>
void require_rejected(Value actual, Value expected) {
    bool rejected = false;
    try {
        (void)mfq::test::checked_absolute_error(actual, expected);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "nonfinite value was accepted");
}

template <typename Value>
void check_nonfinite_operands() {
    for (Value value : {std::numeric_limits<Value>::quiet_NaN(),
                       std::numeric_limits<Value>::infinity(),
                       -std::numeric_limits<Value>::infinity()}) {
        require_rejected(value, Value(1));
        require_rejected(Value(1), value);
    }
    require_rejected(std::numeric_limits<Value>::max(), -std::numeric_limits<Value>::max());
}

} // namespace

int main() try {
    require(mfq::test::checked_absolute_error(1.25f, 1.0f) == .25f,
        "finite absolute error changed");
    require(mfq::test::checked_absolute_error(1.25, 1.0f) == .25,
        "mixed-precision absolute error changed");
    check_nonfinite_operands<float>();
    check_nonfinite_operands<double>();

    bool rejected = false;
    try {
        (void)mfq::test::checked_absolute_error(
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "infinite difference was accepted");
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
