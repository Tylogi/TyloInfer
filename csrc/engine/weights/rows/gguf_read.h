#pragma once

#include "engine/tensor/tensor.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi::weights {
class GgufRows;

class GgufRead final {
public:
    GgufRead(std::vector<std::shared_ptr<GgufRows>> tables, std::int64_t rows_per_shard);
    ~GgufRead();
    bool issue(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape);
    DenseTensor collect(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape);
    void clear();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
