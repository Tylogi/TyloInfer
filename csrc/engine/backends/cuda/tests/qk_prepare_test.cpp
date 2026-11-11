#include "qk_prepare_test.h"
#include "engine/backends/cuda/executor/graph_pool_lease.h"

int main() try {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    const auto context=tyloi::cuda::default_context(mfq_current_cuda_device());
    const auto pool=tyloi::cuda::GraphPoolLease::acquire(context,mfq_current_cuda_stream());
    ::tyloi::cuda::NoGradGuard guard;
    qk_prepare_case();
    return 0;
}catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
}
