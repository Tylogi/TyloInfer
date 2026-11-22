#include "mfq_cuda_gdn_replay_ops.h"
#include "mfq_cuda_context.h"
#include "gdn_update.cuh"
#include "gdn_input.cuh"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>

namespace {
namespace tb=mfq_tensor_backend;

template<bool Transposed,bool Vectorized>
__global__ void gdn_replay_updates_kernel(
    const void* __restrict__ qkv,const float* __restrict__ updates,
    const float* initial_state,float* next_state,
    const float* initial_conv,float* next_conv,
    int64_t qkv_stride,int64_t qkv_batch_stride,int64_t heads,int64_t state_elements,int64_t channels,
    int64_t state_blocks,int64_t total_blocks,int tokens,int retained,bool qkv_float,
    const int64_t* retained_by_sequence) {
    constexpr int Width=Vectorized?4:1,Threads=256;
    for(int64_t tile=blockIdx.x;tile<total_blocks;tile+=gridDim.x) {
        const int64_t item=tile*Threads+threadIdx.x;
        if(tile>=state_blocks) {
            const int64_t at=(tile-state_blocks)*Threads+threadIdx.x;
            const int64_t batch=retained_by_sequence?at/channels:0,channel=retained_by_sequence?at%channels:at;
            if(retained_by_sequence?batch>=state_elements/(heads*128*128):channel>=channels)continue;
            const int keep=retained_by_sequence?int(retained_by_sequence[batch]):retained;
            if(!keep && initial_conv==next_conv)continue;
            const auto base=batch*3*channels;
            // A thread owns this channel's entire history. Load every source
            // before any store, including when the history commits in place.
            float history[3];
            #pragma unroll
            for(int tap=0;tap<3;++tap) {
                const int from=keep+tap-3;
                history[tap]=from<0?initial_conv[base+int64_t(from+3)*channels+channel]:
                    gdn_qkv_value(qkv,batch*qkv_batch_stride+int64_t(from)*qkv_stride+channel,qkv_float);
            }
            #pragma unroll
            for(int tap=0;tap<3;++tap)next_conv[base+int64_t(tap)*channels+channel]=history[tap];
            continue;
        }
        const int64_t index=item*Width;
        if(index>=state_elements)continue;
        const int64_t head=index/(128*128);
        const auto batch=retained_by_sequence?head/heads:0;
        const int keep=retained_by_sequence?int(retained_by_sequence[batch]):retained;
        if(!keep && initial_state==next_state)continue;
        const int offset=int(index%(128*128));
        const int row=Transposed?offset%128:offset/128;
        const int column=Transposed?offset/128:offset%128;
        float values[Width];
        if constexpr(Vectorized) {
            const float4 v=reinterpret_cast<const float4*>(initial_state)[item];
            values[0]=v.x;values[1]=v.y;values[2]=v.z;values[3]=v.w;
        }else values[0]=initial_state[index];
        for(int t=0;t<keep;++t) {
            const auto head_in_batch=head-batch*heads;
            const auto* record=updates+((batch*tokens+t)*heads+head_in_batch)*257;
            const float decay=record[256];
            #pragma unroll
            for(int part=0;part<Width;++part)
                values[part]=mfq_gdn_update(values[part],decay,
                    record[row+(Transposed?part:0)],record[128+column+(Transposed?0:part)]);
        }
        if constexpr(Vectorized)
            reinterpret_cast<float4*>(next_state)[item]=make_float4(values[0],values[1],values[2],values[3]);
        else next_state[index]=values[0];
    }
}

bool overlaps(const void* a,uint64_t a_bytes,const void* b,uint64_t b_bytes) {
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<=y?y-x<a_bytes:x-y<b_bytes;
}
}

bool try_gdn_replay_updates_cuda(const tb::Tensor& qkv,const tb::Tensor& updates,
    const tb::Tensor& initial_conv,const tb::Tensor& initial_state,
    tb::Tensor& next_conv,tb::Tensor& next_state,int64_t keep,bool transposed,const tb::Tensor& retained_by_sequence) {
    if(!qkv.defined() || !qkv.is_cuda() || qkv.dim()!=3 || qkv.size(0)<1 ||
       qkv.size(1)<1 || qkv.size(1)>8 || qkv.size(2)<1 || qkv.stride(2)!=1 ||
       qkv.stride(1)<qkv.size(2) || keep<1 || keep>qkv.size(1) ||
       (qkv.scalar_type()!=tb::kFloat16 && qkv.scalar_type()!=tb::kFloat32))return false;
    const auto real=[&](const tb::Tensor& x) {
        return x.defined() && x.is_cuda() && x.device()==qkv.device() &&
            x.scalar_type()==tb::kFloat32 && x.is_contiguous();
    };
    const auto batch=qkv.size(0),tokens=qkv.size(1);
    const bool batched=retained_by_sequence.defined();
    if(batched && (!retained_by_sequence.is_cuda() || retained_by_sequence.device()!=qkv.device() ||
       retained_by_sequence.scalar_type()!=tb::kInt64 || !retained_by_sequence.is_contiguous() ||
       retained_by_sequence.sizes().vec()!=std::vector<int64_t>({batch})))return false;
    if(!real(updates) || !real(initial_conv) || !real(initial_state) ||
       !real(next_conv) || !real(next_state) || (batched?updates.dim()!=4:updates.dim()!=3 || batch!=1))return false;
    const auto heads=updates.size(batched?2:1);
    if(heads<1 || (batched?updates.sizes().vec()!=std::vector<int64_t>({batch,tokens,heads,257}):
        updates.sizes().vec()!=std::vector<int64_t>({tokens,heads,257})) ||
       initial_conv.sizes().vec()!=std::vector<int64_t>({batch,3,qkv.size(2)}) ||
       initial_state.sizes().vec()!=std::vector<int64_t>({batch,heads,128,128}) ||
       next_conv.sizes()!=initial_conv.sizes() || next_state.sizes()!=initial_state.sizes())return false;
    const int bytes_per_value=qkv.scalar_type()==tb::kFloat32?4:2;
    const auto safe=uint64_t(std::numeric_limits<int64_t>::max())/bytes_per_value;
    if(uint64_t(qkv.size(2))>safe || (qkv.size(1)>1 &&
       uint64_t(qkv.stride(1))>(safe-uint64_t(qkv.size(2)))/uint64_t(qkv.size(1)-1)))return false;
    const auto sequence_elements=(tokens-1)*uint64_t(qkv.stride(1))+qkv.size(2);
    if(batch>1 && uint64_t(qkv.stride(0))>(safe-sequence_elements)/uint64_t(batch-1))return false;
    const uint64_t qkv_bytes=((batch-1)*uint64_t(qkv.stride(0))+sequence_elements)*bytes_per_value;
    const auto overlap=[](const tb::Tensor& a,const tb::Tensor& b) {
        return overlaps(a.data_ptr(),a.nbytes(),b.data_ptr(),b.nbytes());
    };
    if((initial_state.data_ptr()!=next_state.data_ptr() && overlap(initial_state,next_state)) ||
       (initial_conv.data_ptr()!=next_conv.data_ptr() && overlap(initial_conv,next_conv)) ||
       overlap(next_conv,next_state) || overlap(initial_state,next_conv) || overlap(initial_conv,next_state) ||
       overlap(updates,next_conv) || overlap(updates,next_state) ||
       overlaps(qkv.data_ptr(),qkv_bytes,next_conv.data_ptr(),next_conv.nbytes()) ||
       overlaps(qkv.data_ptr(),qkv_bytes,next_state.data_ptr(),next_state.nbytes()))return false;
    const MfqCudaGuard guard(qkv.device());
    const bool vectorized=((reinterpret_cast<uintptr_t>(initial_state.data_ptr())|
        reinterpret_cast<uintptr_t>(next_state.data_ptr()))&15u)==0;
    const int64_t items=initial_state.numel()/(vectorized?4:1);
    const int64_t state_blocks=(items+255)/256,total_blocks=state_blocks+(batch*qkv.size(2)+255)/256;
    const auto launch=[&]<bool Transposed,bool Vectorized>() {
        gdn_replay_updates_kernel<Transposed,Vectorized><<<unsigned(std::min<int64_t>(total_blocks,65535)),256,0,mfq_current_cuda_stream()>>>(
            qkv.data_ptr(),updates.data_ptr<float>(),initial_state.data_ptr<float>(),next_state.data_ptr<float>(),
            initial_conv.data_ptr<float>(),next_conv.data_ptr<float>(),qkv.stride(1),qkv.stride(0),heads,initial_state.numel(),
            qkv.size(2),state_blocks,total_blocks,int(tokens),int(keep),qkv.scalar_type()==tb::kFloat32,
            batched?retained_by_sequence.data_ptr<int64_t>():nullptr);
    };
    if(transposed){if(vectorized)launch.template operator()<true,true>();else launch.template operator()<true,false>();}
    else {if(vectorized)launch.template operator()<false,true>();else launch.template operator()<false,false>();}
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    static std::atomic<bool> reported{false};
    if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
        "runtime_gdn_rank_updates enabled=1 kernels=1 repeated_conv=0 repeated_norm=0 vectorized=%d\n",int(vectorized));
    return true;
}
