#pragma once

#include "backend.h"

#include <cstdint>

enum class KlMmqMode {
    Default,
    Nint8One,
    Fp16,
};

struct KlMmqState {
    KlMmqMode mode = KlMmqMode::Default;
    int64_t activation_quantize_calls = 0;
    int64_t dense_calls = 0;
    int64_t moe_calls = 0;
    int64_t fallback_calls = 0;

    ::tyloi::DenseTensor prepare_activation(
        ::tyloi::DenseTensor input);
    void reset() noexcept;
};

const char* kl_mmq_mode_name(KlMmqMode mode);

struct KlMmqScope {
    KlMmqState& state;
    KlMmqState previous;

    KlMmqScope(KlMmqMode mode, KlMmqState& state);
    ~KlMmqScope();
};
