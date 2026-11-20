#pragma once

#include "engine/tensor/encoding/encoded.h"
#include "components/ggml/include/ggml.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace tyloi {

struct GgmlInputRotation {
    int block = 1;
    std::vector<std::int8_t> signs_cpu;
    DenseTensor signs_device;
};

struct GgmlRowCorrection {
    DenseTensor packed;
    DenseTensor offsets;
    DenseTensor types;
    std::int64_t source_columns = 0;
    std::int64_t first = 0;
    std::int64_t rows = 0;
    std::int64_t column_first = 0;
    std::int64_t columns = 0;
};

class GgmlQuantizedTensor final : public EncodedTensor {
    std::shared_ptr<EncodedTensor> cloned() const override {
        return std::make_shared<GgmlQuantizedTensor>(*this);
    }

public:
    DenseTensor packed;
    ggml_type type = GGML_TYPE_COUNT;
    std::int64_t out = 0;
    std::int64_t neuron_len = 0;
    std::uint64_t row_bytes = 0;
    std::int64_t quad_storage_rows = 0;
    std::int64_t quad_row_begin = 0;
    std::optional<GgmlInputRotation> rotation;
    std::optional<GgmlRowCorrection> correction;

    std::shared_ptr<TensorImpl> relocated(Device target) const override;

    std::vector<const DenseTensor*> buffers() const override {
        std::vector<const DenseTensor*> result{&packed};
        if (rotation && rotation->signs_device.defined())
            result.push_back(&rotation->signs_device);
        if (correction) {
            if (correction->packed.defined()) result.push_back(&correction->packed);
            if (correction->offsets.defined()) result.push_back(&correction->offsets);
            if (correction->types.defined()) result.push_back(&correction->types);
        }
        return result;
    }

protected:
    const DenseTensor& primary() const override { return packed; }
    std::int64_t logical_rows() const override { return out; }
    std::int64_t logical_columns() const override { return neuron_len; }
};

} // namespace tyloi
