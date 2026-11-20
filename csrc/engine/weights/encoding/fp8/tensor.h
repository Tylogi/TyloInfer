#pragma once

#include "engine/tensor/encoding/encoded.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi {

class ChannelFp8Tensor final : public EncodedTensor {
    std::shared_ptr<EncodedTensor> cloned() const override {
        return std::make_shared<ChannelFp8Tensor>(*this);
    }

public:
    DenseTensor values;
    DenseTensor scales;
    std::int64_t out = 0;
    std::int64_t neuron_len = 0;
    bool dynamic_activations = false;

    std::vector<const DenseTensor*> buffers() const override {
        return {&values, &scales};
    }

protected:
    const DenseTensor& primary() const override { return values; }
    std::int64_t logical_rows() const override { return out; }
    std::int64_t logical_columns() const override { return neuron_len; }
};

} // namespace tyloi
