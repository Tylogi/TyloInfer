#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mfq {

enum class LoopLayerNumbering {
    physical,
    execution,
};

// A visit owns its cache slot even when it reads another visit's weights.
class LoopLayerPlan {
public:
    LoopLayerPlan(int64_t blocks, int64_t first, int64_t end, int64_t passes,
                  LoopLayerNumbering numbering)
        : first_(first), end_(end), numbering_(numbering) {
        if (first < 0 || end <= first || end > blocks || passes <= 0)
            throw std::invalid_argument("invalid loop layer range or pass count");
        const auto width = end - first;
        if (passes - 1 > (std::numeric_limits<int64_t>::max() - end) / width)
            throw std::invalid_argument("loop layer count overflows");
        extra_ = (passes - 1) * width;
        if (numbering == LoopLayerNumbering::execution) {
            if (extra_ > blocks - end)
                throw std::invalid_argument("loop visits exceed the declared execution layers");
            execution_layers_ = blocks;
            weight_layers_ = blocks - extra_;
        } else {
            if (extra_ > std::numeric_limits<int64_t>::max() - blocks)
                throw std::invalid_argument("loop layer count overflows");
            execution_layers_ = blocks + extra_;
            weight_layers_ = blocks;
        }
    }

    int64_t execution_layers() const noexcept { return execution_layers_; }
    int64_t weight_layers() const noexcept { return weight_layers_; }

    int64_t first_visit(int64_t layer) const {
        if (layer < 0 || layer >= weight_layers_)
            throw std::out_of_range("loop weight layer is out of range");
        return layer < end_ ? layer : layer + extra_;
    }

    int64_t physical_layer(int64_t visit) const {
        if (visit < 0 || visit >= execution_layers_)
            throw std::out_of_range("loop layer visit is out of range");
        if (visit < end_) return visit;
        if (visit < end_ + extra_) return first_ + (visit - first_) % (end_ - first_);
        return visit - extra_;
    }

    int64_t source_layer(int64_t visit) const {
        const auto physical = physical_layer(visit);
        return numbering_ == LoopLayerNumbering::execution && visit >= end_ + extra_ ? visit : physical;
    }

private:
    int64_t first_, end_, extra_, execution_layers_, weight_layers_;
    LoopLayerNumbering numbering_;
};

} // namespace mfq
