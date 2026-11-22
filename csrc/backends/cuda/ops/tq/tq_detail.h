#pragma once
#include "tq.h"
#include "tyloinfer/tq_rows.h"
#include "tyloinfer/tq_model_layout.h"

namespace tq_detail {
using mfq::tq_source::logical_shape;
using mfq::tq_source::row_parts;
using mfq::tq_source::join_rows;
using mfq::tq_source::tq_image;
inline void status(int code) {
    if(code<0)throw std::invalid_argument("invalid TQ CUDA projection arguments");
    if(code>=1000)throw std::runtime_error("TQ cuBLAS status "+std::to_string(code-1000));
    TYLOINFER_CUDA_CHECK(static_cast<cudaError_t>(code));
}
}
