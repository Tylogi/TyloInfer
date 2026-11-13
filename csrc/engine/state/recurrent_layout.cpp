#include "engine/state/recurrent_layout.h"

#include "engine/tensor.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tyloi::state {

RecurrentLayout recurrent_layout(std::int64_t batch, std::int64_t channels,
        std::int64_t kernel, std::int64_t heads, std::int64_t key_dim,
        std::int64_t value_dim, bool transposed) {
    if (batch <= 0 || channels <= 0 || kernel <= 0 || heads <= 0 ||
        key_dim <= 0 || value_dim <= 0)
        throw std::invalid_argument("invalid recurrent state layout");
    RecurrentLayout layout{{batch, kernel - 1, channels},
        {batch, heads, transposed ? value_dim : key_dim,
                       transposed ? key_dim : value_dim}, 0};
    const auto bytes = [](const auto& shape) {
        TensorView view;
        view.rank = static_cast<std::uint8_t>(shape.size());
        view.scalar_type = kFloat32;
        std::copy(shape.begin(), shape.end(), view.sizes.begin());
        return view.nbytes();
    };
    const auto convolution = bytes(layout.convolution);
    const auto recurrent = bytes(layout.recurrent);
    if (convolution > std::numeric_limits<std::size_t>::max() - recurrent)
        throw std::overflow_error("recurrent state size overflow");
    layout.bytes = convolution + recurrent;
    return layout;
}

} // namespace tyloi::state
