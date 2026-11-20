#include "backend.h"
#include "env.h"
#include "engine/backends/cuda/kernels/moe/raw.h"
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <cstdint>
#include <vector>
namespace {
void check_same_device(const ::tyloi::DenseTensor& first,const ::tyloi::DenseTensor& value,const char* name) {
    if (!(value.device()==first.device())) {
        std::ostringstream failure;
        failure << name << " must be on " << first.device();
        throw std::runtime_error(failure.str());
    }
}
}
std::vector<::tyloi::DenseTensor> moe_topk_cuda(
        ::tyloi::DenseTensor logits,
        int64_t top_k,
        bool use_sigmoid,
        bool use_sqrt_softplus,
        bool normalize,
        bool delayed_softmax,
        MfqOptional<::tyloi::DenseTensor> bias,
        double norm_floor,
        double scale) {
    if (!(logits.is_cuda() && logits.is_contiguous() && logits.dim() == 2))
        throw std::runtime_error("logits must be contiguous CUDA [tokens, experts]");
    if (!(logits.scalar_type() == ::tyloi::kFloat32 || logits.scalar_type() == ::tyloi::kFloat16 ||
        logits.scalar_type() == ::tyloi::kBFloat16))
        throw std::runtime_error("logits must be float16, bfloat16 or float32");
    const int rows = static_cast<int>(logits.size(0));
    const int experts = static_cast<int>(logits.size(1));
    if (!(rows > 0 && experts > 0))
        throw std::runtime_error("logits dimensions must be nonzero");
    if (!(top_k >= 1 && top_k <= 32 && top_k <= experts))
        throw std::runtime_error("top_k must be in [1, min(32, experts)]");
    if (!(!(normalize && delayed_softmax)))
        throw std::runtime_error("selected-weight normalization and delayed softmax are mutually exclusive");
    if (!(!(use_sigmoid && delayed_softmax)))
        throw std::runtime_error("sigmoid routing and delayed softmax are mutually exclusive");
    if (!(!(use_sqrt_softplus && delayed_softmax)))
        throw std::runtime_error("sqrt-softplus routing and delayed softmax are mutually exclusive");
    if (!(!(use_sigmoid && use_sqrt_softplus)))
        throw std::runtime_error("sigmoid and sqrt-softplus routing are mutually exclusive");
    const float * bias_ptr = nullptr;
    int bias_stride=0;
    if (bias.has_value()) {
        auto & value = bias.value();
        if (!(value.is_cuda() && value.is_contiguous() && value.scalar_type() == ::tyloi::kFloat32 &&
            (value.numel()==experts ||
             (value.dim()==2 && value.size(0)==rows && value.size(1)==experts))))
            throw std::runtime_error("bias must be contiguous CUDA float32 [experts] or [tokens, experts]");
        check_same_device(logits, value, "bias");
        bias_ptr = value.data_ptr<float>();
        bias_stride=value.numel()==experts?0:experts;
    }

    auto ids = ::tyloi::cuda::empty({rows, top_k}, logits.options().dtype(::tyloi::kInt32));
    auto weights = ::tyloi::cuda::empty({rows, top_k}, logits.options().dtype(::tyloi::kFloat32));
    const cudaStream_t stream = mfq_current_cuda_stream();
    static const bool disable_topk_sort = [] {
        const char* value=tyloi::cuda::env_lookup("TYLOINFER_DISABLE_MOE_TOPK_SORT");
        return value && std::atoi(value)!=0;
    }();
    using Kernel = tyloi::cuda::moe_raw::RouterKernel;
    Kernel kernel=Kernel::generic;
    if (!disable_topk_sort && rows==1 && experts==128 && top_k==8 &&
        logits.scalar_type()==::tyloi::kFloat32 && !bias_ptr && !use_sigmoid && !normalize && delayed_softmax && scale==1.0)
        kernel=Kernel::fixed_128;
    else if (!disable_topk_sort && rows==1 && experts==256 && top_k==8 &&
        logits.scalar_type()==::tyloi::kFloat32 && !bias_ptr && !use_sigmoid && !use_sqrt_softplus && !normalize && delayed_softmax && scale==1.0)
        kernel=Kernel::fixed_256;
    else if (!disable_topk_sort && rows==1 && experts==256 && top_k==6 &&
        logits.scalar_type()==::tyloi::kFloat32 &&
        logits.scalar_type()==::tyloi::kFloat32 && !use_sigmoid && use_sqrt_softplus && normalize && !delayed_softmax)
        kernel=Kernel::fixed_sqrt_256;
    else {
        const char* flag=tyloi::cuda::env_lookup("TYLOINFER_MOE_TOPK_512");
        const char* batch=tyloi::cuda::env_lookup("TYLOINFER_MOE_TOPK_512_BATCH");
        if ((rows==1 || (rows<=6 && (!batch || batch[0]!='0'))) && experts==512 && top_k==10 && !bias_ptr &&
            !use_sigmoid && !use_sqrt_softplus && normalize && !delayed_softmax && (!flag || std::atoi(flag)!=0))
            kernel=Kernel::fixed_512;
        else {
            const char* cache=tyloi::cuda::env_lookup("TYLOINFER_MOE_TOPK_CACHE");
            if(experts<=512 && (!cache || std::atoi(cache)!=0)) kernel=Kernel::cached;
        }
    }
    tyloi::cuda::moe_raw::Router args{logits.data_ptr(),bias_ptr,ids.data_ptr<int32_t>(),weights.data_ptr<float>(),
        rows,experts,bias_stride,int(top_k),use_sigmoid,use_sqrt_softplus,normalize,delayed_softmax,
        float(norm_floor),float(scale),logits.scalar_type()==::tyloi::kFloat16 ? tyloi::cuda::NumericType::f16 :
        logits.scalar_type()==::tyloi::kBFloat16 ? tyloi::cuda::NumericType::bf16 : tyloi::cuda::NumericType::f32,kernel};
    tyloi::cuda::moe_raw::topk(args,stream);
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return {ids,weights};
}

::tyloi::DenseTensor moe_sqrtsoftplus_weights_cuda(
        ::tyloi::DenseTensor logits,
        ::tyloi::DenseTensor ids,
        double norm_floor,
        double scale) {
    if (!(logits.is_cuda() && logits.is_contiguous() && logits.dim() == 2))
        throw std::runtime_error("logits must be contiguous CUDA [tokens, experts]");
    if (!(logits.scalar_type() == ::tyloi::kFloat32 || logits.scalar_type() == ::tyloi::kFloat16 ||
        logits.scalar_type() == ::tyloi::kBFloat16))
        throw std::runtime_error("logits must be float16, bfloat16 or float32");
    if (!(ids.is_cuda() && ids.is_contiguous() && ids.dim() == 2 &&
        ids.scalar_type() == ::tyloi::kInt32))
        throw std::runtime_error("ids must be contiguous CUDA int32 [tokens, top_k]");
    check_same_device(logits, ids, "ids");
    if (!(ids.size(0) == logits.size(0)))
        throw std::runtime_error("ids and logits must have the same token count");
    const int rows = static_cast<int>(logits.size(0));
    const int experts = static_cast<int>(logits.size(1));
    const int top_k = static_cast<int>(ids.size(1));
    if (!(rows > 0 && experts > 0))
        throw std::runtime_error("logits dimensions must be nonzero");
    if (!(top_k >= 1 && top_k <= 32 && top_k <= experts))
        throw std::runtime_error("top_k must be in [1, min(32, experts)]");

    auto weights = ::tyloi::cuda::empty(ids.sizes(), logits.options().dtype(::tyloi::kFloat32));
    tyloi::cuda::moe_raw::sqrt_weights(logits.data_ptr(),ids.data_ptr<int32_t>(),weights.data_ptr<float>(),rows,experts,top_k,
        float(norm_floor),float(scale),logits.scalar_type()==::tyloi::kFloat16 ? tyloi::cuda::NumericType::f16 :
        logits.scalar_type()==::tyloi::kBFloat16 ? tyloi::cuda::NumericType::bf16 : tyloi::cuda::NumericType::f32,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return weights;
}
::tyloi::DenseTensor moe_weighted_reduce_cuda(::tyloi::DenseTensor pair_output, ::tyloi::DenseTensor weights) {
    const auto dtype = pair_output.scalar_type();
    if (!(pair_output.is_cuda() && pair_output.is_contiguous() &&
        (dtype == ::tyloi::kFloat16 || dtype == ::tyloi::kBFloat16 ||
         dtype == ::tyloi::kFloat32) && pair_output.dim() == 3))
        throw std::runtime_error("pair_output must be contiguous CUDA f16/bf16/f32 [tokens, routes, width]");
    if (!(weights.is_cuda() && weights.is_contiguous() &&
        weights.scalar_type() == ::tyloi::kFloat32 && weights.dim() == 2))
        throw std::runtime_error("weights must be contiguous CUDA float32 [tokens, routes]");
    if (!(pair_output.size(0) == weights.size(0) && pair_output.size(1) == weights.size(1)))
        throw std::runtime_error("pair_output and weights leading dimensions must match");
    check_same_device(pair_output, weights, "weights");
    const int tokens = static_cast<int>(pair_output.size(0));
    const int routes = static_cast<int>(pair_output.size(1));
    const int width = static_cast<int>(pair_output.size(2));
    auto output = ::tyloi::cuda::empty({tokens, width}, pair_output.options());
    const auto type=dtype==::tyloi::kFloat16 ? tyloi::cuda::NumericType::f16 : dtype==::tyloi::kBFloat16 ? tyloi::cuda::NumericType::bf16 : tyloi::cuda::NumericType::f32;
    tyloi::cuda::moe_raw::reduce(pair_output.data_ptr(),weights.data_ptr<float>(),output.data_ptr(),tokens,routes,width,type,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

::tyloi::DenseTensor moe_swiglu_sigmoid_table_cuda() {
    auto table=::tyloi::cuda::empty({2,65536},
        ::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kFloat16));
    tyloi::cuda::moe_raw::sigmoid_table(table.data_ptr(),mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();return table;
}

::tyloi::DenseTensor moe_swiglu_rounded_cuda(
        ::tyloi::DenseTensor gate,::tyloi::DenseTensor up,
        ::tyloi::DenseTensor sigmoid_table) {
    if (!(gate.is_cuda() && up.is_cuda() && gate.get_device()==up.get_device() &&
        gate.is_contiguous() && up.is_contiguous() && gate.scalar_type()==::tyloi::kFloat16 &&
        up.scalar_type()==::tyloi::kFloat16 && gate.dim()==3 && gate.sizes()==up.sizes()))
        throw std::runtime_error("rounded SwiGLU requires matching contiguous CUDA float16 [tokens,routes,width] projections");
    if (!(gate.size(0)>0 && gate.size(1)>0 && gate.size(2)>0))
        throw std::runtime_error("rounded SwiGLU dimensions must be positive");
    if (!(sigmoid_table.is_cuda() && sigmoid_table.get_device()==gate.get_device() &&
        sigmoid_table.is_contiguous() && sigmoid_table.scalar_type()==::tyloi::kFloat16 &&
        sigmoid_table.dim()==2 && sigmoid_table.size(0)==2 && sigmoid_table.size(1)==65536))
        throw std::runtime_error("rounded SwiGLU requires its canonical float/double sigmoid table on the projection device");
    auto output=::tyloi::cuda::empty(gate.sizes(),gate.options());
    tyloi::cuda::moe_raw::rounded(gate.data_ptr(),up.data_ptr(),sigmoid_table.data_ptr(),output.data_ptr(),gate.numel(),gate.size(0)*gate.size(1)>1,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

::tyloi::DenseTensor moe_swiglu_split_cuda(::tyloi::DenseTensor gate_up) {
    if (!(gate_up.is_cuda() && gate_up.is_contiguous() &&
        gate_up.scalar_type() == ::tyloi::kFloat16 && gate_up.dim() == 3))
        throw std::runtime_error("gate_up must be contiguous CUDA float16 [tokens, routes, 2 * width]");
    if (!(gate_up.size(2) > 0 && gate_up.size(2) % 2 == 0))
        throw std::runtime_error("gate_up width must be positive and even");
    const int tokens = static_cast<int>(gate_up.size(0));
    const int routes = static_cast<int>(gate_up.size(1));
    const int width = static_cast<int>(gate_up.size(2) / 2);
    if (!(tokens > 0 && routes > 0))
        throw std::runtime_error("gate_up leading dimensions must be nonzero");
    auto output = ::tyloi::cuda::empty({tokens, routes, width}, gate_up.options());
    tyloi::cuda::moe_raw::glu(gate_up.data_ptr(),output.data_ptr(),tokens*routes,width,false,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

::tyloi::DenseTensor moe_geglu_split_cuda(::tyloi::DenseTensor gate_up) {
    if (!(gate_up.is_cuda() && gate_up.is_contiguous() &&
        gate_up.scalar_type() == ::tyloi::kFloat16 && gate_up.dim() >= 2))
        throw std::runtime_error("gate_up must be contiguous CUDA float16 [..., 2 * width]");
    if (!(gate_up.size(-1) > 0 && gate_up.size(-1) % 2 == 0))
        throw std::runtime_error("gate_up width must be positive and even");
    const int width = static_cast<int>(gate_up.size(-1) / 2);
    const int64_t rows64 = gate_up.numel() / (2 * width);
    if (!(rows64 > 0 && rows64 <= INT_MAX))
        throw std::runtime_error("gate_up row count is unsupported");
    auto shape = gate_up.sizes().vec();
    shape.back() = width;
    auto output = ::tyloi::cuda::empty(shape, gate_up.options());
    tyloi::cuda::moe_raw::glu(gate_up.data_ptr(),output.data_ptr(),static_cast<int>(rows64),width,true,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

::tyloi::DenseTensor moe_apply_expert_scale_cuda(
        ::tyloi::DenseTensor weights, ::tyloi::DenseTensor ids, ::tyloi::DenseTensor scales) {
    if (!(weights.is_cuda() && weights.is_contiguous() &&
        weights.scalar_type() == ::tyloi::kFloat32 && weights.dim() == 2))
        throw std::runtime_error("weights must be contiguous CUDA float32 [tokens, routes]");
    if (!(ids.is_cuda() && ids.is_contiguous() &&
        ids.scalar_type() == ::tyloi::kInt32 && ids.sizes() == weights.sizes()))
        throw std::runtime_error("ids must be contiguous CUDA int32 with the same shape as weights");
    if (!(scales.is_cuda() && scales.is_contiguous() &&
        scales.scalar_type() == ::tyloi::kFloat32 && scales.dim() == 1))
        throw std::runtime_error("scales must be contiguous CUDA float32 [experts]");
    check_same_device(weights, ids, "ids");
    check_same_device(weights, scales, "scales");
    const int64_t total64 = weights.numel();
    if (!(total64 <= INT_MAX))
        throw std::runtime_error("expert scale tensor is too large");
    tyloi::cuda::moe_raw::expert_scale(weights.data_ptr<float>(),ids.data_ptr<int32_t>(),scales.data_ptr<float>(),static_cast<int>(total64),mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();return weights;
}

::tyloi::DenseTensor moe_add_shared_gate_cuda(
        ::tyloi::DenseTensor routed,
        ::tyloi::DenseTensor shared,
        ::tyloi::DenseTensor gate_logits) {
    if (!(routed.is_cuda() && routed.is_contiguous() &&
        routed.scalar_type() == ::tyloi::kFloat16 && routed.dim() == 2))
        throw std::runtime_error("routed must be contiguous CUDA float16 [tokens, width]");
    if (!(shared.is_cuda() && shared.is_contiguous() &&
        shared.scalar_type() == ::tyloi::kFloat16 && shared.sizes() == routed.sizes()))
        throw std::runtime_error("shared must match routed as contiguous CUDA float16");
    if (!(gate_logits.is_cuda() && gate_logits.is_contiguous() &&
        gate_logits.scalar_type() == ::tyloi::kFloat32 && gate_logits.dim() == 2 &&
        gate_logits.size(0) == routed.size(0) && gate_logits.size(1) == 1))
        throw std::runtime_error("gate_logits must be contiguous CUDA float32 [tokens, 1]");
    check_same_device(routed, shared, "shared");
    check_same_device(routed, gate_logits, "gate_logits");
    const int rows = static_cast<int>(routed.size(0));
    const int width = static_cast<int>(routed.size(1));
    auto output = ::tyloi::cuda::empty_like(routed);
    tyloi::cuda::moe_raw::add_shared(routed.data_ptr(),shared.data_ptr(),gate_logits.data_ptr<float>(),output.data_ptr(),rows,width,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

::tyloi::DenseTensor moe_weighted_reduce_shared_gate_cuda(
        ::tyloi::DenseTensor pair_output,
        ::tyloi::DenseTensor weights,
        ::tyloi::DenseTensor shared,
        ::tyloi::DenseTensor gate_logits) {
    if (!(pair_output.is_cuda() && pair_output.is_contiguous() &&
        pair_output.scalar_type() == ::tyloi::kFloat16 && pair_output.dim() == 3))
        throw std::runtime_error("pair_output must be contiguous CUDA float16 [tokens, routes, width]");
    if (!(weights.is_cuda() && weights.is_contiguous() &&
        weights.scalar_type() == ::tyloi::kFloat32 && weights.dim() == 2 &&
        pair_output.size(0) == weights.size(0) && pair_output.size(1) == weights.size(1)))
        throw std::runtime_error("weights must match pair_output as contiguous CUDA float32 [tokens, routes]");
    if (!(shared.is_cuda() && shared.is_contiguous() &&
        shared.scalar_type() == ::tyloi::kFloat16 && shared.dim() == 2 &&
        shared.size(0) == pair_output.size(0) && shared.size(1) == pair_output.size(2)))
        throw std::runtime_error("shared must be contiguous CUDA float16 [tokens, width]");
    if (!(gate_logits.is_cuda() && gate_logits.is_contiguous() &&
        gate_logits.scalar_type() == ::tyloi::kFloat32 && gate_logits.dim() == 2 &&
        gate_logits.size(0) == pair_output.size(0) && gate_logits.size(1) == 1))
        throw std::runtime_error("gate_logits must be contiguous CUDA float32 [tokens, 1]");
    check_same_device(pair_output, weights, "weights");
    check_same_device(pair_output, shared, "shared");
    check_same_device(pair_output, gate_logits, "gate_logits");
    const int tokens = static_cast<int>(pair_output.size(0));
    const int routes = static_cast<int>(pair_output.size(1));
    const int width = static_cast<int>(pair_output.size(2));
    auto output = ::tyloi::cuda::empty({tokens, width}, pair_output.options());
    tyloi::cuda::moe_raw::reduce_shared(pair_output.data_ptr(),weights.data_ptr<float>(),shared.data_ptr(),gate_logits.data_ptr<float>(),output.data_ptr(),tokens,routes,width,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

namespace {
void build_expert_map(
        const ::tyloi::DenseTensor & ids,
        int experts,
        int tile_m,
        ::tyloi::DenseTensor & counts,
        ::tyloi::DenseTensor & cursors,
        ::tyloi::DenseTensor & ids_dst,
        ::tyloi::DenseTensor & expert_bounds,
        ::tyloi::DenseTensor & tile_bounds,
        ::tyloi::DenseTensor & tile_experts,
        cudaStream_t stream,
        ::tyloi::DenseTensor * secondary_tile_bounds = nullptr,
        ::tyloi::DenseTensor * secondary_tile_experts = nullptr,
        int secondary_tile_m = 0,
        ::tyloi::DenseTensor * tertiary_tile_bounds = nullptr,
        ::tyloi::DenseTensor * tertiary_tile_experts = nullptr,
        int tertiary_tile_m = 0) {
    const int pairs = static_cast<int>(ids.numel());
    const bool has_secondary = secondary_tile_bounds != nullptr;
    const bool has_tertiary = tertiary_tile_bounds != nullptr;
    if (!(has_secondary == (secondary_tile_experts != nullptr)))
        throw std::runtime_error("secondary tile bounds and experts must be provided together");
    if (!(!has_secondary || secondary_tile_m > 0))
        throw std::runtime_error("secondary_tile_m must be positive");
    if (!(has_tertiary == (tertiary_tile_experts != nullptr)))
        throw std::runtime_error("tertiary tile bounds and experts must be provided together");
    if (!(!has_tertiary || tertiary_tile_m > 0))
        throw std::runtime_error("tertiary_tile_m must be positive");
    if (!(counts.is_cuda() && counts.is_contiguous() && counts.scalar_type() == ::tyloi::kInt32 &&
        counts.numel() >= experts))
        throw std::runtime_error("counts workspace is too small");
    if (!(cursors.is_cuda() && cursors.is_contiguous() && cursors.scalar_type() == ::tyloi::kInt32 &&
        cursors.numel() >= experts))
        throw std::runtime_error("cursors workspace is too small");
    if (!(ids_dst.is_cuda() && ids_dst.is_contiguous() && ids_dst.scalar_type() == ::tyloi::kInt32 &&
        ids_dst.numel() >= pairs))
        throw std::runtime_error("ids_dst workspace is too small");
    if (!(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
        expert_bounds.scalar_type() == ::tyloi::kInt32 && expert_bounds.numel() >= experts + 1))
        throw std::runtime_error("expert_bounds workspace is too small");
    if (!(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
        tile_bounds.scalar_type() == ::tyloi::kInt32 && tile_bounds.numel() >= experts + 1))
        throw std::runtime_error("tile_bounds workspace is too small");
    if (!(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
        tile_experts.scalar_type() == ::tyloi::kInt32 && tile_experts.numel() >= pairs))
        throw std::runtime_error("tile_experts workspace is too small");
    if (has_secondary) {
        if (!(secondary_tile_bounds->is_cuda() &&
            secondary_tile_bounds->is_contiguous() &&
            secondary_tile_bounds->scalar_type() == ::tyloi::kInt32 &&
            secondary_tile_bounds->numel() >= experts + 1))
            throw std::runtime_error("secondary tile_bounds workspace is too small");
        if (!(secondary_tile_experts->is_cuda() &&
            secondary_tile_experts->is_contiguous() &&
            secondary_tile_experts->scalar_type() == ::tyloi::kInt32 &&
            secondary_tile_experts->numel() >= pairs))
            throw std::runtime_error("secondary tile_experts workspace is too small");
    }
    if (has_tertiary) {
        if (!(tertiary_tile_bounds->is_cuda() &&
            tertiary_tile_bounds->is_contiguous() &&
            tertiary_tile_bounds->scalar_type() == ::tyloi::kInt32 &&
            tertiary_tile_bounds->numel() >= experts + 1))
            throw std::runtime_error("tertiary tile_bounds workspace is too small");
        if (!(tertiary_tile_experts->is_cuda() &&
            tertiary_tile_experts->is_contiguous() &&
            tertiary_tile_experts->scalar_type() == ::tyloi::kInt32 &&
            tertiary_tile_experts->numel() >= pairs))
            throw std::runtime_error("tertiary tile_experts workspace is too small");
    }
    check_same_device(ids, counts, "counts");
    check_same_device(ids, cursors, "cursors");
    check_same_device(ids, ids_dst, "ids_dst");
    check_same_device(ids, expert_bounds, "expert_bounds");
    check_same_device(ids, tile_bounds, "tile_bounds");
    check_same_device(ids, tile_experts, "tile_experts");
    if (has_secondary) {
        check_same_device(ids, *secondary_tile_bounds, "secondary_tile_bounds");
        check_same_device(ids, *secondary_tile_experts, "secondary_tile_experts");
    }
    if (has_tertiary) {
        check_same_device(ids, *tertiary_tile_bounds, "tertiary_tile_bounds");
        check_same_device(ids, *tertiary_tile_experts, "tertiary_tile_experts");
    }

    int32_t * secondary_bounds_ptr = has_secondary
        ? secondary_tile_bounds->data_ptr<int32_t>() : nullptr;
    int32_t * secondary_experts_ptr = has_secondary
        ? secondary_tile_experts->data_ptr<int32_t>() : nullptr;
    int32_t * tertiary_bounds_ptr = has_tertiary
        ? tertiary_tile_bounds->data_ptr<int32_t>() : nullptr;
    int32_t * tertiary_experts_ptr = has_tertiary
        ? tertiary_tile_experts->data_ptr<int32_t>() : nullptr;

    tyloi::cuda::moe_raw::RouteMap map{ids.data_ptr<int32_t>(),counts.data_ptr<int32_t>(),cursors.data_ptr<int32_t>(),
        ids_dst.data_ptr<int32_t>(),expert_bounds.data_ptr<int32_t>(),tile_bounds.data_ptr<int32_t>(),tile_experts.data_ptr<int32_t>(),
        secondary_bounds_ptr,secondary_experts_ptr,tertiary_bounds_ptr,tertiary_experts_ptr,pairs,experts,tile_m,secondary_tile_m,tertiary_tile_m};
    tyloi::cuda::moe_raw::build_map(map,stream);
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
}
}
std::vector<::tyloi::DenseTensor> moe_build_expert_map_cuda(
        ::tyloi::DenseTensor ids,
        int64_t n_experts,
        int64_t tile_m) {
    if (!(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type() == ::tyloi::kInt32 && ids.dim() == 2))
        throw std::runtime_error("ids must be contiguous CUDA int32 [tokens, routes]");
    if (!(n_experts > 0 && n_experts <= 4096))
        throw std::runtime_error("n_experts must be in [1, 4096]");
    if (!(tile_m > 0 && tile_m <= 1024))
        throw std::runtime_error("tile_m must be in [1, 1024]");
    const int experts = static_cast<int>(n_experts);
    const int pairs = static_cast<int>(ids.numel());
    auto options = ids.options();
    auto counts = ::tyloi::cuda::empty({experts}, options);
    auto cursors = ::tyloi::cuda::empty({experts}, options);
    auto ids_dst = ::tyloi::cuda::empty({pairs}, options);
    auto expert_bounds = ::tyloi::cuda::empty({experts + 1}, options);
    auto tile_bounds = ::tyloi::cuda::empty({experts + 1}, options);
    auto tile_experts = ::tyloi::cuda::empty({pairs}, options);
    const cudaStream_t stream = mfq_current_cuda_stream();
    build_expert_map(ids, experts, static_cast<int>(tile_m), counts, cursors, ids_dst,
        expert_bounds, tile_bounds, tile_experts, stream);
    return {ids_dst, expert_bounds, tile_bounds, tile_experts, counts};
}
std::vector<::tyloi::DenseTensor> moe_build_expert_maps_cuda(
        ::tyloi::DenseTensor ids,
        int64_t n_experts,
        int64_t tile_m,
        int64_t secondary_tile_m,
        int64_t tertiary_tile_m) {
    if (!(ids.is_cuda() && ids.is_contiguous() &&
        ids.scalar_type() == ::tyloi::kInt32 && ids.dim() == 2))
        throw std::runtime_error("ids must be contiguous CUDA int32 [tokens, routes]");
    if (!(n_experts > 0 && n_experts <= 4096))
        throw std::runtime_error("n_experts must be in [1, 4096]");
    if (!(tile_m > 0 && tile_m <= 1024 &&
        secondary_tile_m > 0 && secondary_tile_m <= 1024))
        throw std::runtime_error("tile sizes must be in [1, 1024]");
    if (!(tertiary_tile_m >= 0 && tertiary_tile_m <= 1024))
        throw std::runtime_error("tertiary tile size must be zero or in [1, 1024]");
    const int experts = static_cast<int>(n_experts);
    const int pairs = static_cast<int>(ids.numel());
    auto options = ids.options();
    auto counts = ::tyloi::cuda::empty({experts}, options);
    auto cursors = ::tyloi::cuda::empty({experts}, options);
    auto ids_dst = ::tyloi::cuda::empty({pairs}, options);
    auto expert_bounds = ::tyloi::cuda::empty({experts + 1}, options);
    auto tile_bounds = ::tyloi::cuda::empty({experts + 1}, options);
    auto tile_experts = ::tyloi::cuda::empty({pairs}, options);
    auto secondary_tile_bounds = ::tyloi::cuda::empty(
        {experts + 1}, options);
    auto secondary_tile_experts = ::tyloi::cuda::empty({pairs}, options);
    const bool has_tertiary = tertiary_tile_m > 0;
    ::tyloi::DenseTensor tertiary_tile_bounds;
    ::tyloi::DenseTensor tertiary_tile_experts;
    if (has_tertiary) {
        tertiary_tile_bounds = ::tyloi::cuda::empty({experts + 1}, options);
        tertiary_tile_experts = ::tyloi::cuda::empty({pairs}, options);
    }
    const cudaStream_t stream = mfq_current_cuda_stream();
    build_expert_map(ids, experts, static_cast<int>(tile_m), counts,
        cursors, ids_dst, expert_bounds, tile_bounds, tile_experts, stream,
        &secondary_tile_bounds, &secondary_tile_experts,
        static_cast<int>(secondary_tile_m),
        has_tertiary ? &tertiary_tile_bounds : nullptr,
        has_tertiary ? &tertiary_tile_experts : nullptr,
        static_cast<int>(tertiary_tile_m));
    std::vector<::tyloi::DenseTensor> result{
        ids_dst, expert_bounds, tile_bounds, tile_experts, counts,
        secondary_tile_bounds, secondary_tile_experts};
    if (has_tertiary) {
        result.push_back(tertiary_tile_bounds);
        result.push_back(tertiary_tile_experts);
    }
    return result;
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
void moe_quantize_shared_input_cuda(::tyloi::DenseTensor input,
        ::tyloi::DenseTensor descriptors,int64_t total_groups) {
    if (!(input.is_cuda() && input.is_contiguous() && input.scalar_type()==::tyloi::kFloat16 &&
        input.dim()==2 && input.size(0)>0 && input.size(0)<=65535 && input.size(1)>0 && input.size(1)<=INT_MAX &&
        descriptors.is_cuda() && descriptors.device()==input.device() && descriptors.is_contiguous() &&
        descriptors.scalar_type()==::tyloi::kInt64 && descriptors.dim()==2 &&
        descriptors.size(0)>0 && descriptors.size(1)==4 && total_groups>0 && total_groups<=INT_MAX))
        throw std::runtime_error("shared MoE input quantization geometry/device disagree");
    const MfqCudaGuard guard(input.device());
    tyloi::cuda::moe_raw::quantize_shared(input.data_ptr(),descriptors.data_ptr<int64_t>(),int(input.size(0)),int(input.size(1)),int(total_groups),mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
}
void moe_quantize_shared_nint_rows_cuda(::tyloi::DenseTensor input,
        ::tyloi::DenseTensor descriptors,int64_t total_groups,bool quantize,
        ::tyloi::DenseTensor offsets,::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor minima,::tyloi::DenseTensor row_metadata,
        ::tyloi::DenseTensor expert_local,::tyloi::DenseTensor ids,
        int64_t output_rows,int64_t local_experts) {
    const int64_t count=output_rows*local_experts;
    if (!(input.is_cuda() && input.is_contiguous() && input.scalar_type()==::tyloi::kFloat16 &&
        input.dim()==2 && input.size(0)>0 && input.size(0)<=65535 && input.size(1)>0 && input.size(1)<=INT_MAX &&
        output_rows>0 && output_rows<=INT_MAX-63 && local_experts>0 && count>0 && count<=INT_MAX))
        throw std::runtime_error("NINT row workspace input/geometry disagree");
    if (!(offsets.is_cuda() && offsets.is_contiguous() && offsets.scalar_type()==::tyloi::kInt64 && offsets.numel()==count &&
        scales.is_cuda() && scales.is_contiguous() && scales.scalar_type()==::tyloi::kFloat32 && scales.numel()==count &&
        minima.is_cuda() && minima.is_contiguous() && minima.scalar_type()==::tyloi::kFloat32 && minima.numel()==count &&
        row_metadata.is_cuda() && row_metadata.is_contiguous() && row_metadata.scalar_type()==::tyloi::kInt32 &&
        row_metadata.dim()==2 && row_metadata.size(0)==count && row_metadata.size(1)==4 &&
        offsets.device()==input.device() && scales.device()==input.device() && minima.device()==input.device() &&
        row_metadata.device()==input.device()))
        throw std::runtime_error("NINT row workspace storage/device disagree");
    if (!(expert_local.is_cuda() && expert_local.is_contiguous() && expert_local.scalar_type()==::tyloi::kInt32 &&
        expert_local.dim()==1 && expert_local.numel()>0 && expert_local.numel()<=4096 &&
        ids.is_cuda() && ids.is_contiguous() && ids.scalar_type()==::tyloi::kInt32 && ids.dim()==2 &&
        ids.numel()>0 && ids.numel()<=65535 && ids.device()==input.device() && expert_local.device()==input.device()))
        throw std::runtime_error("NINT row workspace route/device disagree");
    if (!(!quantize || (descriptors.is_cuda() && descriptors.device()==input.device() &&
        descriptors.is_contiguous() && descriptors.scalar_type()==::tyloi::kInt64 && descriptors.dim()==2 &&
        descriptors.size(0)>0 && descriptors.size(1)==4 && total_groups>0 && total_groups<=INT_MAX/input.size(0))))
        throw std::runtime_error("NINT fused quantization descriptors disagree");
    const int64_t metadata_blocks=((output_rows+63)/64)*ids.numel();
    const int64_t quantize_blocks=quantize?input.size(0)*total_groups:0;
    if (!(metadata_blocks<=INT_MAX))
        throw std::runtime_error("NINT row workspace grid is too large");
    const MfqCudaGuard guard(input.device());
    tyloi::cuda::moe_raw::quantize_nint_rows(input.data_ptr(),quantize?descriptors.data_ptr<int64_t>():nullptr,int(input.size(0)),int(input.size(1)),quantize?int(total_groups):0,quantize,offsets.data_ptr<int64_t>(),scales.data_ptr<float>(),minima.data_ptr<float>(),row_metadata.data_ptr(),expert_local.data_ptr<int32_t>(),ids.data_ptr<int32_t>(),int(ids.numel()),int(expert_local.numel()),int(output_rows),int(local_experts),int(std::max(metadata_blocks,quantize_blocks)),mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

#if TYLOINFER_ENABLE_MFQ_COMPAT
void launch_nint_matmul_routed_cuda(
    ::tyloi::DenseTensor bitstream,
    ::tyloi::DenseTensor row_q_bits,
    ::tyloi::DenseTensor row_q_bit_offsets,
    ::tyloi::DenseTensor subgroup_scale,
    ::tyloi::DenseTensor subgroup_minimum,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor neuron_minimum,
    ::tyloi::DenseTensor quantized_input,
    ::tyloi::DenseTensor input_scale,
    ::tyloi::DenseTensor input,
    ::tyloi::DenseTensor route_ids,
    ::tyloi::DenseTensor expert_local,
    bool route_map_ready,
    ::tyloi::DenseTensor ids_dst,
    ::tyloi::DenseTensor expert_bounds,
    ::tyloi::DenseTensor tile_bounds,
    ::tyloi::DenseTensor tile_experts,
    ::tyloi::DenseTensor output,
    int tokens,
    int routes,
    int experts,
    int output_rows,
    int groups,
    int padded_width,
    int input_width,
    int group_size,
    int q_expert_stride,
    int route_tile_m,
    int pool_phase,
    bool masked_experts,
    int epilogue_mode,
    bool routed_input,
    cudaStream_t stream);
::tyloi::DenseTensor mfe_nint_matmul_ws_cuda(
        ::tyloi::DenseTensor q_packed,
        ::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,
        ::tyloi::DenseTensor sub_scale,
        ::tyloi::DenseTensor sub_min,
        ::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_min,
        ::tyloi::DenseTensor x,
        ::tyloi::DenseTensor ids,
        ::tyloi::DenseTensor expert_local,
        int64_t n_experts,
        int64_t n_local_experts,
        int64_t out_per_expert,
        int64_t gs,
        int64_t epilogue_mode,
        bool route_map_ready,
        bool input_quantized,
        ::tyloi::DenseTensor out,
        ::tyloi::DenseTensor qx,
        ::tyloi::DenseTensor xscale,
        ::tyloi::DenseTensor ids_dst,
        ::tyloi::DenseTensor expert_bounds,
        ::tyloi::DenseTensor tile_bounds,
        ::tyloi::DenseTensor tile_experts,
        int64_t route_tile_m,
        int64_t pool_phase) {
    if (!(n_experts > 0 && n_experts <= 4096))
        throw std::runtime_error("n_experts must be in [1, 4096]");
    if (!(n_local_experts > 0 && n_local_experts <= INT_MAX))
        throw std::runtime_error("n_local_experts must be a positive CUDA slot count");
    if (!(out_per_expert > 0 && out_per_expert <= INT_MAX))
        throw std::runtime_error("out_per_expert must be positive");
    if (!(gs >= 4 && gs <= 64))
        throw std::runtime_error("MFE NINT group size must be in [4,64]");
    if (!(epilogue_mode >= 0 && epilogue_mode <= 2))
        throw std::runtime_error("MFE NINT epilogue mode must be none, SwiGLU, or GeGLU");
    if (!(epilogue_mode == 0 || (out_per_expert % 2) == 0))
        throw std::runtime_error("MFE NINT GLU projection width must be even");
    if (!(route_tile_m == 8 || route_tile_m == 16 || route_tile_m == 32 ||
            route_tile_m == 64 || route_tile_m == 128))
        throw std::runtime_error("MFE NINT route tile must be 8, 16, 32, 64, or 128");
    if (!(pool_phase >= 0 && pool_phase <= INT_MAX))
        throw std::runtime_error("MFE NINT pool phase exceeds the CUDA index range");
    const int experts = static_cast<int>(n_experts);
    const int local_experts = static_cast<int>(n_local_experts);
    const int output_width = static_cast<int>(out_per_expert);
    const int result_width = epilogue_mode == 0
        ? output_width
        : output_width / 2;
    if (!(local_experts <= INT_MAX / output_width))
        throw std::runtime_error("MFE NINT row count exceeds the CUDA index range");
    const int weight_rows = local_experts * output_width;
    if (!(q_packed.is_cuda() && q_packed.is_contiguous() &&
        q_packed.scalar_type() == ::tyloi::kUInt8 &&
        (q_packed.dim() == 1 ||
         (q_packed.dim() == 2 && q_packed.size(0) == local_experts &&
          q_packed.size(1) >= 8 && q_packed.size(1) <= INT_MAX))))
        throw std::runtime_error("MFE NINT values must be canonical contiguous CUDA uint8");
    if (!(row_q_bits.is_cuda() && row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == ::tyloi::kUInt8 &&
        row_q_bits.numel() == weight_rows))
        throw std::runtime_error("MFE NINT row widths must be contiguous CUDA uint8");
    if (!(row_q_bit_offsets.is_cuda() && row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == ::tyloi::kInt64 &&
        row_q_bit_offsets.numel() == weight_rows))
        throw std::runtime_error("MFE NINT row offsets must be contiguous CUDA int64");
    if (!(sub_scale.is_cuda() && sub_scale.is_contiguous() &&
        sub_scale.scalar_type() == ::tyloi::kUInt8 &&
        sub_scale.dim() == 2 && sub_scale.size(0) == weight_rows))
        throw std::runtime_error("MFE NINT subgroup scales must be CUDA uint8 [rows,groups]");
    if (!(sub_min.is_cuda() && sub_min.is_contiguous() &&
        sub_min.scalar_type() == ::tyloi::kUInt8 &&
        sub_min.sizes() == sub_scale.sizes()))
        throw std::runtime_error("MFE NINT subgroup minima shape mismatch");
    if (!(sub_scale.size(1) > 0 &&
        sub_scale.size(1) <= INT_MAX / gs))
        throw std::runtime_error("MFE NINT group geometry exceeds the CUDA index range");
    const int groups = static_cast<int>(sub_scale.size(1));
    if (!(neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == ::tyloi::kFloat32 &&
        neuron_scale.numel() == weight_rows &&
        neuron_min.is_cuda() && neuron_min.is_contiguous() &&
        neuron_min.scalar_type() == ::tyloi::kFloat32 &&
        neuron_min.numel() == weight_rows))
        throw std::runtime_error("MFE NINT neuron metadata is invalid");
    if (!(ids.is_cuda() && ids.is_contiguous() &&
        ids.scalar_type() == ::tyloi::kInt32 && ids.dim() == 2))
        throw std::runtime_error("ids must be contiguous CUDA int32 [tokens,routes]");
    if (!(expert_local.is_cuda() && expert_local.is_contiguous() &&
        expert_local.scalar_type() == ::tyloi::kInt32 &&
        expert_local.numel() == experts))
        throw std::runtime_error("expert_local must be contiguous CUDA int32 [experts]");
    if (!(x.is_cuda() && x.is_contiguous() &&
        x.scalar_type() == ::tyloi::kFloat16 &&
        (x.dim() == 2 || x.dim() == 3)))
        throw std::runtime_error("MFE NINT input must be contiguous CUDA float16");
    if (!(ids.size(0) > 0 && ids.size(0) <= INT_MAX &&
        ids.size(1) > 0 && ids.size(1) <= INT_MAX &&
        ids.size(0) <= INT_MAX / ids.size(1)))
        throw std::runtime_error("MFE NINT route geometry exceeds the CUDA index range");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const bool routed_input = x.dim() == 3;
    if (routed_input) {
        if (!(x.size(0) == tokens && x.size(1) == routes))
            throw std::runtime_error("routed input must have [tokens,routes,K] leading dimensions");
    } else {
        if (!(x.size(0) == tokens))
            throw std::runtime_error("shared input must have one row per token");
    }
    const int input_rows = routed_input ? tokens * routes : tokens;
    const int k_pad = groups * static_cast<int>(gs);
    const int q_expert_stride = q_packed.dim() == 2
        ? static_cast<int>(q_packed.size(1)) : 0;
    if (!(x.size(-1) >= 0 && x.size(-1) <= INT_MAX))
        throw std::runtime_error("MFE NINT input width exceeds the CUDA index range");
    const int source_width = static_cast<int>(x.size(-1));
    if (!(source_width > 0 && source_width <= k_pad))
        throw std::runtime_error("MFE NINT input width exceeds the packed weight width");
    const int k_real = input_quantized
        ? k_pad : static_cast<int>(x.size(-1));
    if (!(k_real <= k_pad))
        throw std::runtime_error("input width exceeds MFE NINT width");
    if (!(qx.is_cuda() && qx.is_contiguous() &&
        qx.scalar_type() == ::tyloi::kInt8 &&
        qx.dim() == 2 && qx.size(0) >= input_rows && qx.size(1) >= k_pad))
        throw std::runtime_error("MFE NINT qx workspace is too small");
    if (!(xscale.is_cuda() && xscale.is_contiguous() &&
        xscale.scalar_type() == ::tyloi::kFloat32 &&
        xscale.dim() == 2 && xscale.size(0) >= input_rows &&
        xscale.size(1) >= groups))
        throw std::runtime_error("MFE NINT xscale workspace is too small");
    if (!(out.is_cuda() && out.is_contiguous() &&
        out.scalar_type() == ::tyloi::kFloat16 &&
        out.sizes() == ::tyloi::IntArrayRef(
            {tokens, routes, result_width})))
        throw std::runtime_error("MFE NINT output shape mismatch");
    check_same_device(q_packed, row_q_bits, "row_q_bits");
    check_same_device(q_packed, row_q_bit_offsets, "row_q_bit_offsets");
    check_same_device(q_packed, sub_scale, "sub_scale");
    check_same_device(q_packed, sub_min, "sub_min");
    check_same_device(q_packed, neuron_scale, "neuron_scale");
    check_same_device(q_packed, neuron_min, "neuron_min");
    check_same_device(q_packed, x, "x");
    check_same_device(q_packed, ids, "ids");
    check_same_device(q_packed, expert_local, "expert_local");
    check_same_device(q_packed, qx, "qx");
    check_same_device(q_packed, xscale, "xscale");
    check_same_device(q_packed, out, "out");
    const bool use_compact = tokens > 8 && route_map_ready;
    const bool use_tiled_prefill = use_compact && route_tile_m > 8;
    if (!(route_tile_m == 8 || use_tiled_prefill))
        throw std::runtime_error("coarse MFE NINT route tiles require a compact route map");
    if (use_compact) {
        if (!(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
            ids_dst.scalar_type() == ::tyloi::kInt32 &&
            ids_dst.numel() == ids.numel()))
            throw std::runtime_error("MFE NINT compact route ids must cover every token-route pair");
        if (!(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
            expert_bounds.scalar_type() == ::tyloi::kInt32 &&
            expert_bounds.numel() >= experts + 1 &&
            tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
            tile_bounds.scalar_type() == ::tyloi::kInt32 &&
            tile_bounds.numel() >= experts + 1))
            throw std::runtime_error("MFE NINT compact expert bounds are invalid");
        if (!(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
            tile_experts.scalar_type() == ::tyloi::kInt32 &&
            tile_experts.numel() >= ids.numel()))
            throw std::runtime_error("MFE NINT compact tile map is invalid");
        check_same_device(q_packed, ids_dst, "ids_dst");
        check_same_device(q_packed, expert_bounds, "expert_bounds");
        check_same_device(q_packed, tile_bounds, "tile_bounds");
        check_same_device(q_packed, tile_experts, "tile_experts");
    }
    const cudaStream_t stream = mfq_current_cuda_stream();

    if (!input_quantized && !use_tiled_prefill) {
        tyloi::cuda::moe_raw::quantize(x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),
            input_rows,k_real,k_pad,groups,static_cast<int>(gs),stream);
    }
    launch_nint_matmul_routed_cuda(
        q_packed, row_q_bits, row_q_bit_offsets, sub_scale, sub_min,
        neuron_scale, neuron_min, qx, xscale, x, ids, expert_local,
        use_compact, ids_dst, expert_bounds, tile_bounds, tile_experts, out,
        tokens, routes, experts, output_width, groups, k_pad, source_width,
        static_cast<int>(gs), q_expert_stride, static_cast<int>(route_tile_m),
        // A shared arena can contain more slots than this source has IDs.
        static_cast<int>(pool_phase), local_experts != experts,
        static_cast<int>(epilogue_mode), routed_input, stream);
    return out;
}
#endif

#if TYLOINFER_ENABLE_MFQ_COMPAT
namespace {
constexpr int kRouteTile=8, kRowsPerBlock=4, kMoeMmaBn=64;
void launch_nint8_zero_moe_mma(
        const ::tyloi::DenseTensor & q,
        const ::tyloi::DenseTensor & scale,
        const ::tyloi::DenseTensor & expert_local,
        const ::tyloi::DenseTensor & x,
        const ::tyloi::DenseTensor & ids_dst,
        const ::tyloi::DenseTensor & expert_bounds,
        const ::tyloi::DenseTensor & tile_bounds,
        const ::tyloi::DenseTensor & tile_experts,
        ::tyloi::DenseTensor & out,
        int tokens,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int k_real,
        bool routed_input,
        int route_tile_m,
        cudaStream_t stream) {
    const int ntiles_n =
        (out_per_expert + kMoeMmaBn - 1) / kMoeMmaBn;
    const int pairs = tokens * routes;
    if (!(route_tile_m == kRouteTile || route_tile_m == 16 ||
        route_tile_m == 32 || route_tile_m == 64))
        throw std::runtime_error("NINT8-0 route tile must be 8, 16, 32, or 64");
    const int64_t max_tiles =
        (pairs + route_tile_m - 1) / route_tile_m + experts;
    const int64_t max_tasks = max_tiles * ntiles_n;
    static const int block_cap = [] {
        const char * value = tyloi::cuda::env_lookup("TYLOINFER_MOE_PREFILL_MMA_BLOCKS");
        return value == nullptr ? 0 : std::max(1, std::atoi(value));
    }();
    static const int forced_bm = [] {
        const char * value = tyloi::cuda::env_lookup("TYLOINFER_MOE_PREFILL_MMA_BM");
        if (value == nullptr) return 0;
        const int parsed = std::atoi(value);
        return parsed == 16 || parsed == 32 || parsed == 64 ? parsed : 0;
    }();
    const int effective_block_cap = block_cap != 0 ? block_cap :
        (pairs >= 32768 ? 12288 : 4096);
    const int blocks = static_cast<int>(std::max<int64_t>(
        1, std::min<int64_t>(effective_block_cap, max_tasks)));
    const dim3 threads(32, 8);
    const int rows_per_expert = std::max(
        1, (pairs + experts - 1) / experts);
    const int bm = route_tile_m != kRouteTile ? route_tile_m :
        (forced_bm != 0 ? forced_bm :
            (rows_per_expert <= 16 ? 16 :
             rows_per_expert <= 32 ? 32 : 64));
    const bool coarse_tiles = route_tile_m == bm;
    if (!(route_tile_m == kRouteTile || coarse_tiles))
        throw std::runtime_error("coarse NINT8-0 route tile must match the MMA row tile");
    tyloi::cuda::moe_raw::Nint8 a{q.data_ptr<uint8_t>(),scale.data_ptr(),x.data_ptr(),nullptr,nullptr,
        nullptr,expert_local.data_ptr<int32_t>(),ids_dst.data_ptr<int32_t>(),expert_bounds.data_ptr<int32_t>(),
        tile_bounds.data_ptr<int32_t>(),tile_experts.data_ptr<int32_t>(),out.data_ptr(),tokens,routes,experts,out_per_expert,groups,k_real,routed_input};
    tyloi::cuda::moe_raw::nint8_mma(a,blocks,bm,coarse_tiles,stream);
}
void launch_nint8_zero_grouped_matmul(
        const ::tyloi::DenseTensor & q,
        const ::tyloi::DenseTensor & scale,
        const ::tyloi::DenseTensor & qx,
        const ::tyloi::DenseTensor & xscale,
        const ::tyloi::DenseTensor & ids,
        const ::tyloi::DenseTensor & expert_local,
        const ::tyloi::DenseTensor & ids_dst,
        const ::tyloi::DenseTensor & expert_bounds,
        const ::tyloi::DenseTensor & tile_bounds,
        const ::tyloi::DenseTensor & tile_experts,
        ::tyloi::DenseTensor & out,
        int tokens,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int k_pad,
        bool routed_input,
        cudaStream_t stream) {
    static const bool disable_token_warp = [] {
        const char * value = tyloi::cuda::env_lookup("TYLOINFER_DISABLE_MOE_TOKEN_WARP");
        return value != nullptr && std::atoi(value) != 0;
    }();
    static const int token_warps = [] {
        const char * value = tyloi::cuda::env_lookup("TYLOINFER_MOE_TOKEN_WARPS");
        if (value == nullptr) return 16;
        const int parsed = std::atoi(value);
        return parsed == 8 || parsed == 16 || parsed == 32 ? parsed : 16;
    }();
    tyloi::cuda::moe_raw::Nint8 a{q.data_ptr<uint8_t>(),scale.data_ptr(),nullptr,qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),
        ids.data_ptr<int32_t>(),expert_local.data_ptr<int32_t>(),ids_dst.defined()?ids_dst.data_ptr<int32_t>():nullptr,
        expert_bounds.defined()?expert_bounds.data_ptr<int32_t>():nullptr,tile_bounds.defined()?tile_bounds.data_ptr<int32_t>():nullptr,
        tile_experts.defined()?tile_experts.data_ptr<int32_t>():nullptr,out.data_ptr(),tokens,routes,experts,out_per_expert,groups,k_pad,routed_input};
    const auto launch_direct = [&](int items) { tyloi::cuda::moe_raw::nint8_direct(a,items,stream); };
    if (!disable_token_warp && tokens > 8 && tokens <= 64) {
        launch_direct(token_warps);
        return;
    }
    if (tokens <= 8) {
        launch_direct(tokens == 1 ? 1 : tokens <= 2 ? 2 : tokens <= 4 ? 4 : 8);
        return;
    }

    const int pairs = tokens * routes;
    const int row_tiles =
        (out_per_expert + kRowsPerBlock - 1) / kRowsPerBlock;
    static const bool disable_persistent = [] {
        const char * value = tyloi::cuda::env_lookup("TYLOINFER_DISABLE_MOE_PERSISTENT");
        return value != nullptr && std::atoi(value) != 0;
    }();
    if (!disable_persistent && tokens <= 32) {
        const int64_t max_tasks =
            static_cast<int64_t>(pairs) * row_tiles;
        constexpr int persistent_blocks = 4096;
        const int blocks = static_cast<int>(
            std::min<int64_t>(max_tasks, persistent_blocks));
        tyloi::cuda::moe_raw::nint8_persistent(a,blocks,stream);
        return;
    }

    const int max_tiles =
        (pairs + kRouteTile - 1) / kRouteTile + experts;
    const int64_t blocks =
        static_cast<int64_t>(row_tiles) * max_tiles;
    if (!(blocks <= INT_MAX))
        throw std::runtime_error("grouped NINT8-0 launch grid is too large");
    tyloi::cuda::moe_raw::nint8_grouped(a,static_cast<int>(blocks),max_tiles,stream);
}
}
::tyloi::DenseTensor nint8_zero_moe_grouped_matmul_pool_ws_cuda(
        ::tyloi::DenseTensor q,
        ::tyloi::DenseTensor scale,
        ::tyloi::DenseTensor x,
        ::tyloi::DenseTensor ids,
        ::tyloi::DenseTensor expert_local,
        int64_t n_experts,
        int64_t n_local_experts,
        int64_t out_per_expert,
        bool route_map_ready,
        bool input_quantized,
        bool use_f16_mma,
        ::tyloi::DenseTensor out,
        ::tyloi::DenseTensor qx,
        ::tyloi::DenseTensor xscale,
        ::tyloi::DenseTensor counts,
        ::tyloi::DenseTensor cursors,
        ::tyloi::DenseTensor ids_dst,
        ::tyloi::DenseTensor expert_bounds,
        ::tyloi::DenseTensor tile_bounds,
        ::tyloi::DenseTensor tile_experts,
        int64_t route_tile_m) {
    if (!(n_experts > 0 && n_experts <= 4096))
        throw std::runtime_error("n_experts must be in [1, 4096]");
    if (!(n_local_experts > 0 && n_local_experts <= INT_MAX))
        throw std::runtime_error("n_local_experts must be a positive CUDA slot count");
    if (!(out_per_expert > 0 && out_per_expert <= INT_MAX))
        throw std::runtime_error("out_per_expert must be positive");
    const int experts = static_cast<int>(n_experts);
    const int local_experts = static_cast<int>(n_local_experts);
    const int output_width = static_cast<int>(out_per_expert);
    if (!(local_experts <= INT_MAX / output_width))
        throw std::runtime_error("NINT8-0 MoE row count exceeds the CUDA index range");
    if (!(q.is_cuda() && q.is_contiguous() &&
        q.scalar_type() == ::tyloi::kUInt8 && q.dim() == 3 &&
        q.size(0) == static_cast<int64_t>(local_experts) * output_width &&
        q.size(2) == 32))
        throw std::runtime_error("NINT8-0 MoE q must be contiguous CUDA uint8 "
        "[local_experts*out,groups,32]");
    if (!(scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type() == ::tyloi::kFloat16 && scale.dim() == 2 &&
        scale.size(0) == q.size(0) && scale.size(1) == q.size(1)))
        throw std::runtime_error("NINT8-0 MoE scale must be contiguous CUDA f16 "
        "[local_experts*out,groups]");
    check_same_device(q, scale, "scale");

    if (!(ids.is_cuda() && ids.is_contiguous() &&
        ids.scalar_type() == ::tyloi::kInt32 && ids.dim() == 2))
        throw std::runtime_error("ids must be contiguous CUDA int32 [tokens, routes]");
    if (!(expert_local.is_cuda() && expert_local.is_contiguous() &&
        expert_local.scalar_type() == ::tyloi::kInt32 &&
        expert_local.dim() == 1 && expert_local.numel() == experts))
        throw std::runtime_error("expert_local must be contiguous CUDA int32 [n_experts]");
    if (!(x.is_cuda() && x.is_contiguous() &&
        x.scalar_type() == ::tyloi::kFloat16 &&
        (x.dim() == 2 || x.dim() == 3)))
        throw std::runtime_error("x must be contiguous CUDA float16 [T,K] or [T,R,K]");
    check_same_device(q, x, "x");
    check_same_device(q, ids, "ids");
    check_same_device(q, expert_local, "expert_local");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    if (!(tokens > 0 && routes > 0))
        throw std::runtime_error("ids dimensions must be nonzero");
    const bool routed_input = x.dim() == 3;
    if (routed_input) {
        if (!(x.size(0) == tokens && x.size(1) == routes))
            throw std::runtime_error("routed x must have [tokens, routes, K] leading dimensions");
    } else {
        if (!(x.size(0) == tokens))
            throw std::runtime_error("shared x must have one row per token");
    }
    const int input_rows = routed_input ? tokens * routes : tokens;
    const int groups = static_cast<int>(q.size(1));
    const int k_pad = groups * 32;
    const int input_width = static_cast<int>(x.size(-1));
    const int k_real =
        input_quantized ? k_pad : static_cast<int>(x.size(-1));
    if (!(input_width <= k_pad))
        throw std::runtime_error("x K exceeds packed NINT8-0 K");
    if (!(qx.is_cuda() && qx.is_contiguous() &&
        qx.scalar_type() == ::tyloi::kInt8 && qx.dim() == 2 &&
        qx.size(0) >= input_rows && qx.size(1) >= k_pad))
        throw std::runtime_error("qx workspace is too small");
    if (!(xscale.is_cuda() && xscale.is_contiguous() &&
        xscale.scalar_type() == ::tyloi::kFloat32 && xscale.dim() == 2 &&
        xscale.size(0) >= input_rows && xscale.size(1) >= groups))
        throw std::runtime_error("xscale workspace is too small");
    check_same_device(q, qx, "qx");
    check_same_device(q, xscale, "xscale");
    if (!(out.is_cuda() && out.is_contiguous() &&
        out.scalar_type() == ::tyloi::kFloat16 && out.dim() == 3 &&
        out.size(0) == tokens && out.size(1) == routes &&
        out.size(2) == output_width))
        throw std::runtime_error("out must be contiguous CUDA float16 "
        "[tokens, routes, out_per_expert]");
    check_same_device(q, out, "out");

    const cudaStream_t stream = mfq_current_cuda_stream();
    if (tokens > 8 && !route_map_ready) {
        build_expert_map(
            ids, experts, kRouteTile, counts, cursors, ids_dst,
            expert_bounds, tile_bounds, tile_experts, stream);
    }
    if (tokens > 8) {
        if (!(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
            tile_experts.scalar_type() == ::tyloi::kInt32 &&
            tile_experts.numel() >= tokens * routes))
            throw std::runtime_error("tile_experts workspace is too small");
        check_same_device(ids, tile_experts, "tile_experts");
    }
    if (use_f16_mma) {
        if (!(tokens > 8))
            throw std::runtime_error("NINT8-0 grouped MMA requires more than eight tokens");
        launch_nint8_zero_moe_mma(
            q, scale, expert_local, x, ids_dst, expert_bounds, tile_bounds,
            tile_experts, out, tokens, routes, experts, output_width, groups,
            input_width, routed_input, static_cast<int>(route_tile_m), stream);
        TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
    if (!input_quantized) {
        tyloi::cuda::moe_raw::quantize(x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),input_rows,k_real,k_pad,groups,32,stream);
    }
    launch_nint8_zero_grouped_matmul(
        q, scale, qx, xscale, ids, expert_local, ids_dst,
        expert_bounds, tile_bounds, tile_experts, out, tokens, routes,
        experts, output_width, groups, k_pad, routed_input, stream);
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
#endif
