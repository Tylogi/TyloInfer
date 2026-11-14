#include "mlx_tq.h"
#include "mfq_container.h"
#include <stdexcept>

namespace tyloi::metal {

MlxTqWeight MlxTqWeight::load(const MfqContainer& model, const std::string& name) {
    if (model.record(name).dtype != "TQ") throw std::runtime_error("tensor is not TQ: " + name);
    const auto shape = model.tq_shape(name);
    const auto mapped = model.map_record(name);
    return from_blob(mapped.view(), int(shape.back()), shape.size() == 3 ? int(shape.front()) : 1);
}


} // namespace tyloi::metal
