#pragma once
#include "models/qwen4_exp/model.h"

inline void check_qwen4_host_recurrent_restore() {
    namespace q4=mfq::cuda::qwen4_exp;
    const auto sample=[](const std::vector<int64_t>& shape,int salt) {
        auto tensor=tb::empty(shape,tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        for(int64_t i=0;i<tensor.numel();++i)tensor.data_ptr<float>()[i]=std::sin(float(i)*.017f+salt);
        return tensor;
    };
    int cases=0;
    for(int width:{32,128})for(bool tiled:{false,true})for(auto dtype:{tb::kFloat16,tb::kBFloat16,tb::kFloat32}) {
        constexpr int nk=2,nv=6,batch=2;
        q4::GdnWeights weights;weights.norm=tb::ones({width},tb::TensorOptions().device(tb::kCUDA).dtype(dtype));
        q4::Gdn gdn(std::move(weights),nk,nv,width,4,1e-6,false,true,true,true,true,true,true,tiled);
        auto convolution=sample({batch,3,(tiled?3*nv:2*nk+nv)*width},3);
        auto recurrent=sample({batch,nv,width,width},7);
        auto expected=convolution;
        if(tiled)expected=tb::cat({convolution.narrow(2,0,nk*width),
            convolution.narrow(2,nv*width,nk*width),convolution.narrow(2,2*nv*width,nv*width)},2);
        gdn.restore_session(convolution,recurrent);
        auto pointers=gdn.graph_state();
        require(pointers[0]->is_cuda() && pointers[1]->is_cuda(),"Host recurrent restore left CPU model state");
        same(expected,*pointers[0]);same(recurrent,*pointers[1]);
        const auto* conv_address=pointers[0]->data_ptr();
        const auto* state_address=pointers[1]->data_ptr();
        pointers[0]->zero_();pointers[1]->zero_();
        gdn.restore_session(convolution,recurrent,true);
        require(pointers[0]->data_ptr()==conv_address && pointers[1]->data_ptr()==state_address,
            "Host recurrent restore changed captured graph addresses");
        same(expected,*pointers[0]);same(recurrent,*pointers[1]);
        auto expected_state=recurrent.clone(),expected_conv=expected.clone();
        convolution.zero_();recurrent.zero_();
        same(expected_conv,*pointers[0]);same(expected_state,*pointers[1]);
        auto device_conv=expected_conv.to(tb::kCUDA),device_state=expected_state.to(tb::kCUDA);
        gdn.restore_session(device_conv,device_state);
        device_conv.zero_();device_state.zero_();
        pointers=gdn.graph_state();same(expected_conv,*pointers[0]);same(expected_state,*pointers[1]);
        ++cases;
    }
    std::cout<<"Host recurrent session restore: exact FP32 state, mixed norm dtypes, legacy head layout, stable graph addresses and independent GPU snapshots PASS cases="<<cases<<'\n';
}
