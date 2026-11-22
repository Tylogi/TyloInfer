#include "app/studio/kernels/cuda/loop_mixing.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <math_constants.h>

namespace mfq::cuda {
namespace {

namespace tb = mfq_tensor_backend;
constexpr int kMixThreads = 128;

struct PackedDepthStates {
    const float* values;
    __device__ const float* state(int row,int layer,int count,int width) const {
        return values+(int64_t(row)*count+layer)*width;
    }
};

struct DepthStateRefs {
    static constexpr int capacity=64;
    const float* values[capacity];
    __device__ const float* state(int row,int layer,int,int width) const {
        return values[layer]+int64_t(row)*width;
    }
};

template<class States>
__global__ void mix_depth(States states, const float* query, float* output,
    int width, int count, int query_count, float epsilon) {
    const int row=blockIdx.x;
    const int head=blockIdx.y;
    const int thread=threadIdx.x;
    query+=int64_t(head)*width;
    output+=(int64_t(row)*query_count+head)*width;
    extern __shared__ float coefficients[];
    __shared__ float squares[kMixThreads/32],dots[kMixThreads/32];
    __shared__ float denominator;
    const float score_scale=rsqrtf(float(width));
    for (int layer=0;layer<count;++layer) {
        const auto* state=states.state(row,layer,count,width);
        float square=0,dot=0;
        for (int i=thread;i<width;i+=kMixThreads) {
            square=fmaf(state[i],state[i],square);
            dot=fmaf(state[i],query[i],dot);
        }
        for (int shift=16;shift;shift/=2) {
            square+=__shfl_down_sync(0xffffffff,square,shift);
            dot+=__shfl_down_sync(0xffffffff,dot,shift);
        }
        if (thread%32==0) { squares[thread/32]=square; dots[thread/32]=dot; }
        __syncthreads();
        if (thread==0) {
            float square_sum=0,dot_sum=0;
            for (int warp=0;warp<kMixThreads/32;++warp) {
                square_sum+=squares[warp];
                dot_sum+=dots[warp];
            }
            const float norm=rsqrtf(square_sum/float(width)+epsilon);
            coefficients[layer]=dot_sum*norm*score_scale;
            coefficients[count+layer]=norm;
        }
        __syncthreads();
    }
    if (thread==0) {
        float maximum=-CUDART_INF_F,total=0;
        for (int layer=0;layer<count;++layer) maximum=fmaxf(maximum,coefficients[layer]);
        for (int layer=0;layer<count;++layer) {
            const float weight=expf(coefficients[layer]-maximum);
            total+=weight;
            coefficients[layer]=weight*coefficients[count+layer];
        }
        denominator=total;
    }
    __syncthreads();
    for (int i=thread;i<width;i+=kMixThreads) {
        float mixed=0;
        for (int layer=0;layer<count;++layer)
            mixed=fmaf(coefficients[layer],states.state(row,layer,count,width)[i],mixed);
        output[i]=mixed/denominator;
    }
}

__global__ void normalize_projection(const float* input,const float* multiplier,float* output,
    int64_t rows,int heads,int width,int multiplier_stride,float value,bool clipping) {
    const int64_t row=blockIdx.x;
    if (row>=rows) return;
    input+=row*width;
    output+=row*width;
    if (multiplier) multiplier+=(row%heads)*multiplier_stride;
    float squares=0;
    for (int i=threadIdx.x;i<width;i+=kMixThreads) {
        const auto element=input[i]*(multiplier?multiplier[i]:1.f);
        squares+=element*element;
    }
    for (int offset=16;offset;offset/=2) squares+=__shfl_down_sync(0xffffffff,squares,offset);
    __shared__ float warp_squares[kMixThreads/32],scale;
    if (threadIdx.x%32==0) warp_squares[threadIdx.x/32]=squares;
    __syncthreads();
    if (threadIdx.x==0) {
        float sum=0;
        for (const auto part:warp_squares) sum+=part;
        scale=clipping?fminf(value/(sqrtf(sum)+1e-6f),1.f):rsqrtf(sum/float(width)+value);
    }
    __syncthreads();
    for (int i=threadIdx.x;i<width;i+=kMixThreads)
        output[i]=input[i]*(multiplier?multiplier[i]:1.f)*scale;
}

struct LoopRows {
    const float* values;
    int64_t batch_stride,token_stride;
    __device__ const float* row(int batch,int token) const {
        return values+int64_t(batch)*batch_stride+int64_t(token)*token_stride;
    }
};

__global__ void blend_message(LoopRows current,LoopRows message,LoopRows gain,LoopRows forget,
    float* output,int width,int tokens,int64_t elements,bool contiguous_gain,bool contiguous_forget) {
    const auto at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (at>=elements) return;
    const auto row=at/width;
    const int batch=int(row/tokens),token=int(row%tokens),column=int(at%width);
    // Match the native tensor tanh precision for contiguous and strided inputs.
    const auto keep=forget.row(batch,token)[column],add=gain.row(batch,token)[column];
    const auto retained=__fsub_rn(1.f,contiguous_forget?::tanhf(keep):float(::tanh(double(keep))));
    const auto added=contiguous_gain?::tanhf(add):float(::tanh(double(add)));
    output[at]=__fadd_rn(__fmul_rn(current.row(batch,token)[column],retained),
        __fmul_rn(message.row(batch,token)[column],added));
}

template<bool Decode> __global__ void convolve_residual(float* history, const float* input,
    const float* weights, const float* dynamic, float* output, int channels, int tokens,
    int taps, int heads, float scale) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if (c>=channels) return;
    const auto row=int64_t(blockIdx.z)*tokens+blockIdx.y;
    const int token=blockIdx.y;
    input+=int64_t(blockIdx.z)*tokens*channels;
    history+=int64_t(blockIdx.z)*(taps-1)*channels;
    const auto* coeffs=weights+int64_t(c)*taps;
    if (dynamic) dynamic+=(row*heads+c/(channels/heads))*taps;
    const float current=input[int64_t(token)*channels+c];
    float result=current;
    for (int i=0;i<taps;++i) {
        const int position=token+i-(taps-1);
        const float sample=position>=0 ? input[int64_t(position)*channels+c] :
            history[int64_t(token+i)*channels+c];
        const float weight=coeffs[i]+(dynamic?dynamic[taps-1-i]:0.0f);
        result=fmaf(sample,weight,result);
    }
    output[row*channels+c]=result*scale;
    if constexpr (Decode) {
        for (int i=0;i<taps-2;++i)
            history[int64_t(i)*channels+c]=history[int64_t(i+1)*channels+c];
        history[int64_t(taps-2)*channels+c]=current;
    }
}

template<bool Partial> __global__ void advance_history(float* history, const float* input, int channels,
    int tokens, int past,const float* previous,const int64_t* accepted) {
    const int channel=blockIdx.x*blockDim.x+threadIdx.x;
    if (channel>=channels) return;
    history+=int64_t(blockIdx.y)*past*channels;
    input+=int64_t(blockIdx.y)*tokens*channels;
    const auto requested=Partial?accepted[blockIdx.y]:tokens;
    const int kept=requested<0?0:requested>tokens?tokens:int(requested);
    if constexpr (Partial) previous+=int64_t(blockIdx.y)*past*channels;
    else previous=history;
    for (int i=0;i<past;++i) {
        const int position=kept+i-past;
        history[int64_t(i)*channels+channel]=position>=0 ? input[int64_t(position)*channels+channel] :
            previous[int64_t(i+kept)*channels+channel];
    }
}

bool float_on(const tb::Tensor& tensor, const tb::Tensor& owner) {
    return tensor.defined() && tensor.is_cuda() && tensor.is_contiguous() &&
        tensor.scalar_type()==tb::kFloat32 && tensor.device()==owner.device();
}
}

tb::Tensor residual_depth_mix(const tb::Tensor& states, const tb::Tensor& queries, float epsilon) {
    TYLOINFER_RUNTIME_CHECK(float_on(states,states) && float_on(queries,states) && states.dim()==3 &&
        queries.dim()==2 && states.size(0)>0 && states.size(0)<=INT_MAX && states.size(1)>0 &&
        states.size(1)<=1024 && states.size(2)>0 && states.size(2)<=INT_MAX &&
        queries.size(0)>0 && queries.size(0)<=65535 && queries.size(1)==states.size(2) &&
        epsilon>0 && std::isfinite(epsilon), "invalid residual depth mixing geometry");
    const auto rows=states.size(0),count=states.size(1),width=states.size(2),heads=queries.size(0);
    auto output=tb::empty({rows,heads,width},states.options());
    mix_depth<<<dim3(unsigned(rows),unsigned(heads)),kMixThreads,2*count*sizeof(float),mfq_current_cuda_stream()>>>(
        PackedDepthStates{states.data_ptr<float>()},queries.data_ptr<float>(),output.data_ptr<float>(),
        int(width),int(count),int(heads),epsilon);
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    return output;
}

tb::Tensor residual_depth_mix(const std::vector<tb::Tensor>& bank,const tb::Tensor& hidden,
    const tb::Tensor& queries,float epsilon) {
    const auto rows=hidden.size(0)*hidden.size(1),width=hidden.size(2),heads=queries.size(0);
    if (bank.size()>=DepthStateRefs::capacity || !float_on(hidden,hidden) ||
        !std::all_of(bank.begin(),bank.end(),[&](const auto& value) {
            return float_on(value,hidden) && value.sizes()==hidden.sizes();
        })) {
        auto states=bank;
        states.push_back(hidden);
        return residual_depth_mix(tb::stack(states,2).reshape({rows,int64_t(states.size()),width}).contiguous(),
            queries,epsilon);
    }
    TYLOINFER_RUNTIME_CHECK(rows>0 && rows<=INT_MAX && width>0 && width<=INT_MAX &&
        float_on(queries,hidden) && queries.dim()==2 && heads>0 && heads<=65535 && queries.size(1)==width,
        "invalid residual depth mixing geometry");
    DepthStateRefs sources{};
    for (size_t i=0;i<bank.size();++i) sources.values[i]=bank[i].data_ptr<float>();
    sources.values[bank.size()]=hidden.data_ptr<float>();
    const auto count=int(bank.size()+1);
    auto output=tb::empty({rows,heads,width},hidden.options());
    mix_depth<<<dim3(unsigned(rows),unsigned(heads)),kMixThreads,2*count*sizeof(float),mfq_current_cuda_stream()>>>(
        sources,queries.data_ptr<float>(),output.data_ptr<float>(),int(width),count,int(heads),epsilon);
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

tb::Tensor normalize_loop_attention_projection(const tb::Tensor& input,const tb::Tensor& weights,
    float value,bool clip) {
    const auto heads=input.size(-2),width=input.size(-1),rows=input.numel()/width;
    const bool per_head=weights.defined() && weights.dim()==2 && weights.size(0)==heads && weights.size(1)==width;
    const bool shared=weights.defined() && weights.dim()==1 && weights.size(0)==width;
    if (!float_on(input,input) || (weights.defined() && (!float_on(weights,input) || (!shared && !per_head))))
        return {};
    auto output=tb::empty_like(input);
    normalize_projection<<<unsigned(rows),kMixThreads,0,mfq_current_cuda_stream()>>>(input.data_ptr<float>(),
        weights.defined()?weights.data_ptr<float>():nullptr,output.data_ptr<float>(),
        rows,int(heads),int(width),per_head?int(width):0,value,clip);
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

tb::Tensor blend_loop_message(const tb::Tensor& current,const tb::Tensor& message,
    const tb::Tensor& gain,const tb::Tensor& forget) {
    const auto valid=[&](const tb::Tensor& tensor) {
        return tensor.dim()==3 && tensor.sizes()==current.sizes() && tensor.device()==current.device() &&
            tensor.scalar_type()==tb::kFloat32 && tensor.stride(2)==1;
    };
    if (!current.is_cuda() || !valid(current) || !valid(message) || !valid(gain) || !valid(forget))
        return current*(1.0-tb::tanh(forget))+message*tb::tanh(gain);
    MfqCudaGuard guard(current.device());
    auto output=tb::empty(current.sizes(),current.options());
    const auto elements=output.numel(),blocks=(elements+kMixThreads-1)/kMixThreads;
    TYLOINFER_RUNTIME_CHECK(blocks<=INT_MAX && current.size(1)<=INT_MAX && current.size(2)<=INT_MAX,
        "loop blend grid overflow");
    if (!elements) return output;
    const auto rows=[](const tb::Tensor& tensor) {
        return LoopRows{tensor.data_ptr<float>(),tensor.stride(0),tensor.stride(1)};
    };
    blend_message<<<unsigned(blocks),kMixThreads,0,mfq_current_cuda_stream()>>>(rows(current),rows(message),
        rows(gain),rows(forget),output.data_ptr<float>(),int(current.size(2)),int(current.size(1)),elements,
        gain.is_contiguous(),forget.is_contiguous());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

tb::Tensor residual_short_convolution(tb::Tensor& history, const tb::Tensor& input,
    const tb::Tensor& weights, const tb::Tensor& dynamic) {
    TYLOINFER_RUNTIME_CHECK(float_on(input,input) && float_on(weights,input) && float_on(history,input) &&
        input.dim()==3 && history.dim()==3 && weights.dim()==2 && weights.size(1)>1 && weights.size(1)<=64 &&
        history.size(0)==input.size(0) && history.size(1)==weights.size(1)-1 &&
        history.size(2)==input.size(2) && weights.size(0)==input.size(2) &&
        input.size(0)>0 && input.size(0)<=65535 && input.size(1)>0 && input.size(1)<=65535 &&
        input.size(2)>0 && input.size(2)<=INT_MAX-kMixThreads, "invalid residual convolution geometry");
    const auto batch=input.size(0),tokens=input.size(1),channels=input.size(2),taps=weights.size(1);
    if (dynamic.defined()) {
        TYLOINFER_RUNTIME_CHECK(float_on(dynamic,input) && dynamic.dim()==4 && dynamic.size(0)==batch &&
            dynamic.size(1)==tokens && dynamic.size(2)>0 && channels%dynamic.size(2)==0 && dynamic.size(3)==taps,
            "invalid residual convolution dynamic taps");
    }
    auto output=tb::empty(input.sizes(),input.options());
    const int blocks=int((channels+kMixThreads-1)/kMixThreads);
    const dim3 grid(blocks,unsigned(tokens),unsigned(batch));
    const auto stream=mfq_current_cuda_stream();
    const int heads=dynamic.defined()?int(dynamic.size(2)):0;
    const auto* coeffs=dynamic.defined()?dynamic.data_ptr<float>():nullptr;
    const float scale=1/std::sqrt(float((heads?2:1)*taps+1));
    if (tokens==1) convolve_residual<true><<<grid,kMixThreads,0,stream>>>(history.data_ptr<float>(),
        input.data_ptr<float>(),weights.data_ptr<float>(),coeffs,output.data_ptr<float>(),int(channels),1,int(taps),heads,scale);
    else {
        convolve_residual<false><<<grid,kMixThreads,0,stream>>>(history.data_ptr<float>(),input.data_ptr<float>(),
            weights.data_ptr<float>(),coeffs,output.data_ptr<float>(),int(channels),int(tokens),int(taps),heads,scale);
        advance_history<false><<<dim3(blocks,unsigned(batch)),kMixThreads,0,stream>>>(history.data_ptr<float>(),
            input.data_ptr<float>(),int(channels),int(tokens),int(taps-1),nullptr,nullptr);
    }
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    return output;
}

void retain_convolution_prefix(tb::Tensor& history,const tb::Tensor& previous,const tb::Tensor& inputs,
    const tb::Tensor& accepted) {
    TYLOINFER_RUNTIME_CHECK(float_on(previous,history) && float_on(inputs,history) &&
        history.dim()==3 && history.sizes()==previous.sizes() && inputs.dim()==3 &&
        inputs.size(0)==history.size(0) && inputs.size(1)>0 && inputs.size(2)==history.size(2) &&
        accepted.device()==history.device() && accepted.scalar_type()==tb::kInt64 && accepted.is_contiguous() &&
        accepted.numel()==history.size(0),"invalid retained convolution prefix");
    const auto channels=history.size(2);
    const auto grid=dim3(unsigned((channels+kMixThreads-1)/kMixThreads),unsigned(history.size(0)));
    advance_history<true><<<grid,kMixThreads,0,mfq_current_cuda_stream()>>>(history.data_ptr<float>(),
        inputs.data_ptr<float>(),int(channels),int(inputs.size(1)),int(history.size(1)),
        previous.data_ptr<float>(),accepted.data_ptr<int64_t>());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace mfq::cuda
