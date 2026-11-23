#include "app/server/kernels/cuda/index_pool.h"

#include <climits>
#include <cmath>
#include <math_constants.h>

namespace mfq::cuda {
namespace {

namespace tb = mfq_tensor_backend;
constexpr int kThreads = 128;

template<int FixedGroup> __device__ float pool_channel(
    const float* values, const float* logits, const float* bias, int width, int channel, int groupsize) {
    const int group_size = FixedGroup ? FixedGroup : groupsize;
    float maximum = -CUDART_INF_F;
#pragma unroll
    for (int slot = 0; slot < group_size; ++slot) {
        const auto index = int64_t(slot) * width + channel;
        maximum = fmaxf(maximum, logits[index] + bias[index]);
    }
    float numerator = 0;
    float denominator = 0;
#pragma unroll
    for (int slot = 0; slot < group_size; ++slot) {
        const auto index = int64_t(slot) * width + channel;
        const float weight = expf(logits[index] + bias[index] - maximum);
        numerator = fmaf(weight, values[index], numerator);
        denominator += weight;
    }
    return numerator / denominator;
}

template<int FixedGroup> __global__ void pool_and_rotate(
    const float* values, const float* logits, const float* bias, const float* norm,
    const float* cosine, const float* sine, float* output, int group_size, int width, int rotary, float eps) {
    extern __shared__ float pooled[];
    __shared__ float warp_squares[kThreads / 32];
    __shared__ float inverse;
    const auto offset = int64_t(blockIdx.x) * group_size * width;
    const int thread = threadIdx.x;
    float squares = 0;
    for (int channel = thread; channel < width; channel += kThreads) {
        const float value = pool_channel<FixedGroup>(values + offset, logits + offset, bias, width, channel, group_size);
        pooled[channel] = value;
        squares = fmaf(value, value, squares);
    }
    for (int delta = 16; delta; delta /= 2)
        squares += __shfl_down_sync(0xffffffff, squares, delta);
    if (thread % 32 == 0) warp_squares[thread / 32] = squares;
    __syncthreads();
    if (thread == 0) {
        float sum = 0;
#pragma unroll
        for (int warp = 0; warp < kThreads / 32; ++warp) sum += warp_squares[warp];
        inverse = rsqrtf(sum / float(width) + eps);
    }
    __syncthreads();

    const int pass = width - rotary;
    const auto row = int64_t(blockIdx.x);
    float* destination = output + row * width;
    for (int channel = thread; channel < pass; channel += kThreads)
        destination[channel] = (pooled[channel] * inverse) * norm[channel];
    for (int pair = thread; pair < rotary / 2; pair += kThreads) {
        const int channel = pass + 2 * pair;
        const auto r = row * (rotary / 2) + pair;
        const float a = (pooled[channel] * inverse) * norm[channel];
        const float b = (pooled[channel + 1] * inverse) * norm[channel + 1];
        destination[channel] = fmaf(-b, sine[r], a * cosine[r]);
        destination[channel + 1] = fmaf(a, sine[r], b * cosine[r]);
    }
}

bool float_on(const tb::Tensor& tensor, const tb::Tensor& owner) {
    return tensor.is_cuda() && tensor.is_contiguous() && tensor.device() == owner.device() &&
           tensor.scalar_type() == tb::kFloat32;
}

__global__ void complete_groups(const float* values, const float* logits, const int64_t* positions,
    const float* pending_values, const float* pending_logits, const float* bias, const float* norm,
    const float* cosine,const float* sine,float* keys,int64_t* slots,int tokens,int groups,int group_width,
    int width,int record_capacity,int rotary_pairs,int rope_rows,bool packed,float epsilon) {
    const int batch=blockIdx.y,candidate=blockIdx.x,thread=threadIdx.x;
    const int64_t first=positions[packed?int64_t(batch)*tokens:0];
    const int64_t group=first>=0?first/group_width+candidate:-1;
    const int64_t base=group*group_width;
    const int64_t row=int64_t(batch)*groups+candidate;
    const bool complete=group>=0 && group<record_capacity && base+group_width<=first+tokens && base<rope_rows;
    if (thread==0) slots[row]=complete?int64_t(batch)*record_capacity+group:-1;
    if (!complete) return;
    values+=int64_t(batch)*tokens*width;
    logits+=int64_t(batch)*tokens*width;
    pending_values+=int64_t(batch)*group_width*width;
    pending_logits+=int64_t(batch)*group_width*width;
    keys+=row*width;
    extern __shared__ float pooled[];
    __shared__ float squares[kThreads/32],inverse;
    float square=0;
    for (int c=thread;c<width;c+=kThreads) {
        float maximum=-CUDART_INF_F;
        for (int s=0;s<group_width;++s) {
            const int64_t offset=base+s-first;
            const float logit=offset<0?pending_logits[s*width+c]:logits[offset*width+c];
            maximum=fmaxf(maximum,logit+bias[s*width+c]);
        }
        float total=0,result=0;
        for (int s=0;s<group_width;++s) {
            const int64_t offset=base+s-first;
            const float logit=offset<0?pending_logits[s*width+c]:logits[offset*width+c];
            const float value=offset<0?pending_values[s*width+c]:values[offset*width+c];
            const float weight=expf(logit+bias[s*width+c]-maximum);
            result=fmaf(weight,value,result);
            total+=weight;
        }
        const float average=result/total;
        pooled[c]=average;
        square=fmaf(average,average,square);
    }
    for (int shift=16;shift;shift/=2) square+=__shfl_down_sync(0xffffffff,square,shift);
    if (thread%32==0) squares[thread/32]=square;
    __syncthreads();
    if (thread==0) {
        float sum=0;
        for (int warp=0;warp<kThreads/32;++warp) sum+=squares[warp];
        inverse=rsqrtf(sum/width+epsilon);
    }
    __syncthreads();
    const int unrotated=width-2*rotary_pairs;
    for (int c=thread;c<unrotated;c+=kThreads) keys[c]=pooled[c]*inverse*norm[c];
    for (int p=thread;p<rotary_pairs;p+=kThreads) {
        const int c=unrotated+2*p;
        const auto angle=base*rotary_pairs+p;
        const float x=pooled[c]*inverse*norm[c],y=pooled[c+1]*inverse*norm[c+1];
        keys[c]=fmaf(-y,sine[angle],x*cosine[angle]);
        keys[c+1]=fmaf(x,sine[angle],y*cosine[angle]);
    }
}

__global__ void save_pending_group(const float* values,const float* logits,const int64_t* positions,
    float* pending_values,float* pending_logits,int tokens,int group_width,int width,bool packed) {
    const auto element=blockIdx.x*blockDim.x+threadIdx.x;
    if (element>=group_width*width) return;
    const auto batch=blockIdx.y;
    const auto first=positions[packed?int64_t(batch)*tokens:0];
    if (first<0) return;
    const auto end=first+tokens;
    const auto offset=end-end%group_width+element/width-first;
    if (offset<0 || offset>=tokens) return;
    const auto at=(int64_t(batch)*tokens+offset)*width+element%width;
    const auto tail=int64_t(batch)*group_width*width+element;
    pending_values[tail]=values[at];
    pending_logits[tail]=logits[at];
}

__global__ void retain_pending_group(const float* values,const float* logits,const int64_t* positions,
    const int64_t* lengths,const float* saved_values,const float* saved_logits,float* pending_values,
    float* pending_logits,int tokens,int group,int width) {
    const int element=blockIdx.x*blockDim.x+threadIdx.x;
    const int batch=blockIdx.y;
    const auto kept=lengths[batch];
    if (element>=group*width || kept>=tokens) return;
    const auto first=positions[int64_t(batch)*tokens];
    const auto end=first+kept;
    const auto offset=end-end%group+element/width-first;
    const auto tail=int64_t(batch)*group*width+element;
    if (offset>=0 && offset<kept) {
        const auto source=(int64_t(batch)*tokens+offset)*width+element%width;
        pending_values[tail]=values[source];
        pending_logits[tail]=logits[source];
    } else {
        pending_values[tail]=saved_values[tail];
        pending_logits[tail]=saved_logits[tail];
    }
}

__global__ void draft_history(const int32_t* selected, const int64_t* confirmed_lengths,
    int32_t* history, int requests, int top_k, int recent, int width, int64_t elements) {
    const int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= elements) return;
    const int64_t row = at / width;
    const int column = int(at % width);
    const int64_t confirmed = confirmed_lengths[row % requests];
    int64_t position = -1;
    if (confirmed > 0 && confirmed <= INT32_MAX) {
        const int64_t start = confirmed - recent;
        if (column < top_k) {
            const auto candidate = selected[row * top_k + column];
            if (candidate >= 0 && candidate < start) position = candidate;
        } else {
            const auto local = start + column - top_k;
            if (local >= 0) position = local;
        }
    }
    history[at] = static_cast<int32_t>(position);
}

}  // namespace

tb::Tensor index_pool_keys(const tb::Tensor& values, const tb::Tensor& logits,
    const tb::Tensor& position_bias, const tb::Tensor& norm, const tb::Tensor& cosine,
    const tb::Tensor& sine, float epsilon) {
    TYLOINFER_RUNTIME_CHECK(values.dim() == 3 && float_on(values, values),
        "IndexPool values must be contiguous CUDA f32 [groups,group_size,width]");
    const auto groups = values.size(0), slots = values.size(1), width = values.size(2);
    TYLOINFER_RUNTIME_CHECK(groups <= INT_MAX && slots >= 2 && slots <= INT_MAX && width > 0 && width <= INT_MAX &&
        float_on(logits, values) && logits.sizes() == values.sizes() && float_on(position_bias, values) &&
        position_bias.dim() == 2 && position_bias.size(0) == slots && position_bias.size(1) == width &&
        float_on(norm, values) && norm.numel() == width && std::isfinite(epsilon) && epsilon > 0,
        "invalid IndexPool group geometry or RMS parameters");
    const bool rotate = cosine.defined() || sine.defined();
    if (rotate) {
        TYLOINFER_RUNTIME_CHECK(float_on(cosine, values) && float_on(sine, values) && cosine.dim() == 2 &&
            cosine.sizes() == sine.sizes() && cosine.size(0) == groups && cosine.size(1) > 0 &&
            cosine.size(1) <= width / 2, "invalid IndexPool group rotary geometry");
    }
    auto output = tb::empty({groups, width}, values.options());
    if (!groups) return output;
    const int rotary = rotate ? int(2 * cosine.size(1)) : 0;
    const auto launch = [&](auto fixed) {
        pool_and_rotate<decltype(fixed)::value><<<int(groups), kThreads, width * sizeof(float), mfq_current_cuda_stream()>>>(
            values.data_ptr<float>(), logits.data_ptr<float>(), position_bias.data_ptr<float>(), norm.data_ptr<float>(),
            rotate ? cosine.data_ptr<float>() : nullptr, rotate ? sine.data_ptr<float>() : nullptr, output.data_ptr<float>(),
            int(slots), int(width), rotary, epsilon);
    };
    if (slots == 4) launch(std::integral_constant<int, 4>{});
    else launch(std::integral_constant<int, 0>{});
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

tb::Tensor index_pool_draft_history(const tb::Tensor& selected, const tb::Tensor& confirmed_lengths,
                                    int recent_window) {
    TYLOINFER_RUNTIME_CHECK(selected.is_cuda() && selected.is_contiguous() && selected.dim() == 3 &&
        selected.scalar_type() == tb::kInt32 && selected.size(1) > 0 && selected.size(1) <= INT_MAX &&
        confirmed_lengths.is_cuda() && confirmed_lengths.device() == selected.device() &&
        confirmed_lengths.is_contiguous() && confirmed_lengths.scalar_type() == tb::kInt64 &&
        confirmed_lengths.dim() == 1 && confirmed_lengths.size(0) == selected.size(1) && recent_window >= 0 &&
        selected.size(2) <= INT_MAX - recent_window,
        "draft IndexPool history expects int32 selections [layers,requests,top_k] and int64 confirmed lengths");
    const auto layers = selected.size(0), requests = selected.size(1), top_k = selected.size(2);
    const auto width = top_k + recent_window;
    auto history = tb::empty({layers, requests, width}, selected.options());
    const auto elements = history.numel();
    if (!elements) return history;
    TYLOINFER_RUNTIME_CHECK((elements - 1) / kThreads < INT_MAX, "draft IndexPool history is too large");
    draft_history<<<int((elements - 1) / kThreads + 1), kThreads, 0, mfq_current_cuda_stream()>>>(
        selected.data_ptr<int32_t>(), confirmed_lengths.data_ptr<int64_t>(), history.data_ptr<int32_t>(),
        int(requests), int(top_k), recent_window, int(width), elements);
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return history;
}

IndexPoolAppend index_pool_append(const tb::Tensor& values,const tb::Tensor& logits,const tb::Tensor& positions,
    tb::Tensor& pending_values,tb::Tensor& pending_logits,const tb::Tensor& bias,const tb::Tensor& norm,
    const tb::Tensor& cosine,const tb::Tensor& sine,int capacity,float epsilon) {
    TYLOINFER_RUNTIME_CHECK(values.dim()==3 && float_on(values,values) && float_on(logits,values) &&
        logits.sizes()==values.sizes() && values.size(0)>0 && values.size(0)<=65535 &&
        values.size(1)>0 && values.size(1)<=INT_MAX && values.size(2)>0 && values.size(2)<=4096 &&
        pending_values.dim()==3 && float_on(pending_values,values) && float_on(pending_logits,values) &&
        pending_values.sizes()==pending_logits.sizes() && pending_values.size(0)==values.size(0) &&
        pending_values.size(1)>=2 && pending_values.size(1)<=64 && pending_values.size(2)==values.size(2) &&
        float_on(bias,values) && bias.dim()==2 && bias.size(0)==pending_values.size(1) && bias.size(1)==values.size(2) &&
        float_on(norm,values) && norm.numel()==values.size(2) && std::isfinite(epsilon) && epsilon>0 &&
        float_on(cosine,values) && float_on(sine,values) && cosine.dim()==2 && sine.sizes()==cosine.sizes() &&
        cosine.size(0)>0 && cosine.size(0)<=INT_MAX && cosine.size(1)<=values.size(2)/2 &&
        capacity>0 && int64_t(capacity)*values.size(0)<=INT_MAX,
        "invalid IndexPool append geometry");
    const auto b=values.size(0),tokens=values.size(1),width=values.size(2),group=pending_values.size(1);
    TYLOINFER_RUNTIME_CHECK(positions.is_cuda() && positions.device()==values.device() && positions.is_contiguous() &&
        positions.scalar_type()==tb::kInt64 && ((positions.dim()==1 && positions.size(0)==tokens) ||
            (positions.dim()==2 && positions.size(0)==b && positions.size(1)==tokens)),"invalid IndexPool positions");
    const auto groups=(tokens+group-1)/group;
    IndexPoolAppend result{tb::empty({b,groups,width},values.options()),
        tb::empty({b,groups},values.options().dtype(tb::kInt64))};
    const auto stream=mfq_current_cuda_stream();
    complete_groups<<<dim3(unsigned(groups),unsigned(b)),kThreads,width*sizeof(float),stream>>>(values.data_ptr<float>(),
        logits.data_ptr<float>(),positions.data_ptr<int64_t>(),pending_values.data_ptr<float>(),pending_logits.data_ptr<float>(),
        bias.data_ptr<float>(),norm.data_ptr<float>(),cosine.data_ptr<float>(),sine.data_ptr<float>(),result.keys.data_ptr<float>(),
        result.record_positions.data_ptr<int64_t>(),int(tokens),int(groups),int(group),int(width),capacity,int(cosine.size(1)),
        int(cosine.size(0)),positions.dim()==2,epsilon);
    save_pending_group<<<dim3(unsigned((group*width+kThreads-1)/kThreads),unsigned(b)),kThreads,0,stream>>>(
        values.data_ptr<float>(),logits.data_ptr<float>(),positions.data_ptr<int64_t>(),pending_values.data_ptr<float>(),
        pending_logits.data_ptr<float>(),int(tokens),int(group),int(width),positions.dim()==2);
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

void index_pool_retain_prefix(const tb::Tensor& values,const tb::Tensor& logits,const tb::Tensor& positions,
    const tb::Tensor& lengths,const tb::Tensor& saved_values,const tb::Tensor& saved_logits,
    tb::Tensor& pending_values,tb::Tensor& pending_logits) {
    const auto batch=values.size(0),tokens=values.size(1),width=values.size(2),group=pending_values.size(1);
    retain_pending_group<<<dim3(unsigned((group*width+kThreads-1)/kThreads),unsigned(batch)),kThreads,0,
        mfq_current_cuda_stream()>>>(values.data_ptr<float>(),logits.data_ptr<float>(),positions.data_ptr<int64_t>(),
        lengths.data_ptr<int64_t>(),saved_values.data_ptr<float>(),saved_logits.data_ptr<float>(),
        pending_values.data_ptr<float>(),pending_logits.data_ptr<float>(),int(tokens),int(group),int(width));
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace mfq::cuda
