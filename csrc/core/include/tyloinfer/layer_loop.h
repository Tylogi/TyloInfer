#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mfq {

// Visits own runtime state. Several visits can refer to the same weight layer.
class LayerLoop {
public:
    LayerLoop(int64_t weight_layers, int64_t begin, int64_t end, int64_t passes)
        : weight_layers_(weight_layers), begin_(begin), end_(end), passes_(passes) {
        if (weight_layers <= 0 || begin < 0 || begin >= end || end > weight_layers ||
            passes <= 0 || (passes - 1) >
                (std::numeric_limits<int64_t>::max() - weight_layers) / (end - begin)) {
            throw std::invalid_argument("invalid repeated layer geometry");
        }
    }

    int64_t weight_layers() const noexcept { return weight_layers_; }
    int64_t extra_visits() const noexcept { return (end_ - begin_) * (passes_ - 1); }
    int64_t visits() const noexcept { return weight_layers_ + extra_visits(); }

    int64_t weight_layer(int64_t visit) const {
        if (visit < 0 || visit >= visits()) throw std::out_of_range("layer visit is out of range");
        if (visit < begin_) return visit;
        if (visit >= end_ + extra_visits()) return visit - extra_visits();
        return begin_ + (visit - begin_) % (end_ - begin_);
    }

    int64_t first_visit(int64_t layer) const {
        if (layer < 0 || layer >= weight_layers_) throw std::out_of_range("weight layer is out of range");
        return layer < end_ ? layer : layer + extra_visits();
    }

private:
    int64_t weight_layers_;
    int64_t begin_;
    int64_t end_;
    int64_t passes_;
};

}  // namespace mfq
