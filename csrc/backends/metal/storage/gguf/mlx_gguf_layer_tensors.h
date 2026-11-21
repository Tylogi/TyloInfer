#pragma once

#include "mfq_container.h"
#include "mlx_gguf.h"
#include "mlx_tensor.h"
#include "mlx_transformer.h"
#include "tyloinfer/gguf_model_source.h"

#include <string>

namespace mfq::metal {

class MlxGgufLayerTensors {
public:
    MlxGgufLayerTensors(const MfqContainer& model, const GgufModelSource& source, int layer)
        : model_(model), source_(source), prefix_("blk." + std::to_string(layer) + ".") {}

    bool has(const std::string& name) const {
        return source_.find_tensor(prefix_ + name) || model_.contains(canonical(name, false));
    }
    MlxLinear linear(const std::string& name) const {
        return MlxLinear::load(model_, canonical(name));
    }
    MlxGgufWeight experts(const std::string& name) const {
        return MlxGgufWeight::load(model_, canonical(name));
    }
    mlx::core::array dense(const std::string& name) const {
        const auto key = canonical(name);
        return mlx::core::astype(load_dense_array(model_.record(key).dtype, model_.read(key)), mlx::core::float32);
    }
    MlxRmsNorm norm(const std::string& name, float eps) const {
        return MlxRmsNorm(dense(name), eps);
    }
private:
    std::string canonical(const std::string& name, bool required = true) const {
        try {
            return source_.canonical_tensor_name(prefix_ + name);
        } catch (const std::invalid_argument&) {
            if (required) throw;
            return prefix_ + name;
        }
    }
    const MfqContainer& model_;
    const GgufModelSource& source_;
    std::string prefix_;
};

}  // namespace mfq::metal
