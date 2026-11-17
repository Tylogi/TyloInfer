#pragma once
#include "prefill_projection.h"
#include "quant_linear.h"
#include "core/ffn.h"
#include "mfq_cuda_activation_ops.h"
#include "mfq_cuda_norm_ops.h"
#include "runtime/decode_window.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace mfq::cuda::test {
inline void check_common_prefill_projection(int device) {
    namespace tb=mfq_tensor_backend;
    const MfqCudaGuard guard(device);
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kFloat32);
    if(!gguf::grouped_f16_available(device))return;
    const auto require=[](bool valid,const char* message){if(!valid)throw std::runtime_error(message);};
    const auto matrix=[&](int rows,int columns,int seed,bool wide=false) {
        std::vector<float> values(std::size_t(rows)*columns);
        for(int row=0;row<rows;++row)for(int column=0;column<columns;++column)
            values[std::size_t(row)*columns+column]=row==0?0.f:
                std::ldexp(float(std::sin((row*columns+column)*.173+seed)*.13),wide?(row%5-2)*25:0);
        return tb::tensor(values).reshape({rows,columns}).to(tb::Device(tb::kCUDA,device));
    };
    double worst_relative=0,worst_backward=0;int cases=0,residual_cases=0;
    const auto residual_oracle=[&](const tb::Tensor& actual,const tb::Tensor& expected) {
        const auto a=actual.to(tb::kFloat32).cpu().contiguous(),b=expected.to(tb::kFloat32).cpu().contiguous();
        require(a.sizes()==b.sizes(),"common residual projection changed output shape");
        double error=0,energy=0;
        for(int64_t i=0;i<a.numel();++i) {
            const double value=a.data_ptr<float>()[i],reference=b.data_ptr<float>()[i];
            require(std::isfinite(value),"common residual projection produced a nonfinite value");
            error+=(value-reference)*(value-reference);energy+=reference*reference;
        }
        require(error<=energy*4e-6+1e-12,"common residual projection changed FP16 GEMM rounding excessively");
        ++residual_cases;
    };
    const auto oracle=[&](const tb::Tensor& actual,const tb::Tensor& input,const tb::Tensor& weight) {
        const auto result=actual.to(tb::kFloat32).cpu().contiguous();
        const auto x=input.to(tb::kFloat32).cpu().contiguous(),w=weight.to(tb::kFloat32).cpu().contiguous();
        require(result.sizes().vec()==std::vector<int64_t>{input.size(0),weight.size(0)},"common prefill output shape changed");
        double error=0,energy=0,absolute_energy=0;
        for(int64_t row:{int64_t(0),int64_t(1),input.size(0)/2,input.size(0)-1})
        for(int64_t output=0;output<weight.size(0);++output) {
            double expected=0,absolute=0;
            for(int64_t column=0;column<input.size(1);++column) {
                const double term=double(x.data_ptr<float>()[row*input.size(1)+column])*w.data_ptr<float>()[output*input.size(1)+column];
                expected+=term;absolute+=std::abs(term);
            }
            const double value=result.data_ptr<float>()[row*weight.size(0)+output];
            require(std::isfinite(value),"common prefill produced a nonfinite value");
            if(!expected)require(value==0,"common prefill changed an exact zero projection");
            error+=(value-expected)*(value-expected);energy+=expected*expected;absolute_energy+=absolute*absolute;
        }
        const double relative=energy?std::sqrt(error/energy):0;
        // Normalize the rounding error by absolute contributions. A nearly
        // cancelled dot has no stable output-relative error bound.
        const double backward=absolute_energy?std::sqrt(error/absolute_energy):0;
        if(backward>=2e-3)throw std::runtime_error("common prefill differs from FP64: M="+
            std::to_string(input.size(0))+" N="+std::to_string(weight.size(0))+" K="+
            std::to_string(input.size(1))+" relative_L2="+std::to_string(relative));
        worst_relative=std::max(worst_relative,relative);worst_backward=std::max(worst_backward,backward);++cases;
    };
    for(int columns:{128,320})for(int outputs:{1,17,65})for(int rows:{512,513}) {
        GgufWeight weight;weight.type=GGML_TYPE_Q8_0;weight.out=outputs;weight.neuron_len=columns;
        weight.row_bytes=std::uint64_t(columns/32)*34;
        std::vector<std::uint8_t> bytes(std::size_t(outputs)*weight.row_bytes);
        for(std::size_t offset=0;offset<bytes.size();offset+=34) {
            bytes[offset]=0;bytes[offset+1]=0x2c; // FP16 1/16.
            for(int i=0;i<32;++i)bytes[offset+2+i]=std::uint8_t(int((offset+i*13)%255)-127);
        }
        weight.packed=tb::tensor(bytes).to(tb::Device(tb::kCUDA,device));
        const auto ids=tb::arange(outputs,options.dtype(tb::kInt64));
        const auto decoded=weight.embedding(ids).to(tb::kFloat32).contiguous();
        auto input=matrix(rows,columns,29,true);
        const auto direct=prefill_materialize(prefill_product(prefill_projection_weight(weight),input));
        const auto generic=prefill_materialize(prefill_product(prefill_decoded_projection(weight.packed,
            outputs,columns,[decoded]{return decoded;}),input));
        const auto a=direct.cpu().contiguous(),b=generic.cpu().contiguous();
        require(a.nbytes()==b.nbytes() && !std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()),
            "common decoded projection changed GGUF prefill arithmetic");
        oracle(generic,input,decoded);
    }
    for(int columns:{1,33,128,320})for(int outputs:{1,17,65})for(int rows:{512,513}) {
        auto weights=matrix(outputs,columns,3,true),input=matrix(rows,columns,5,true);
        const auto plan=prefill_decoded_projection(weights,outputs,columns,[weights]{return weights;});
        require(bool(plan) && plan.decode_row_bytes==std::uint64_t(columns)*6,
            "common decoder did not price simultaneous FP32 and FP16 matrices");
        const auto product=prefill_product(plan,input);
        require(product.defined(),"common FP32 decoder did not enter the shared projection");
        oracle(prefill_materialize(product),input,weights);
        if(columns==33 && outputs==17 && rows==513) {
            tb::Tensor output;DecodeWindow graph(mfq_current_cuda_stream());
            graph.capture([&]{output=prefill_materialize(prefill_product(plan,input));},[] {},[&]{output={};});
            for(int step=0;step<3;++step) {
                weights.copy_(matrix(outputs,columns,7+step,step!=1));
                input.copy_(step==0?tb::zeros_like(input):matrix(rows,columns,11+step,true));
                graph.run();oracle(output,input,weights);
            }
        }
    }
    for(int columns:{128,320})for(int outputs:{33,65}) {
        QuantLinear low,high;low.kind=QuantLinearKind::Mxfp4;high.kind=QuantLinearKind::Mxfp8;
        std::vector<std::uint8_t> q4(std::size_t(outputs)*columns/2),q8(std::size_t(outputs)*columns);
        for(std::size_t i=0;i<q4.size();++i)q4[i]=std::uint8_t(i*31+7);
        for(std::size_t i=0;i<q8.size();++i)q8[i]=std::uint8_t((i*37)%120|((i&1)<<7));
        low.mxfp4.weight.values=tb::tensor(q4).reshape({outputs,columns/2}).to(tb::Device(tb::kCUDA,device));
        low.mxfp4.weight.scales=tb::full({outputs,columns/32},123,options.dtype(tb::kUInt8));
        low.mxfp4.weight.out=outputs;low.mxfp4.weight.neuron_len=columns;
        high.mxfp8.weight.values=tb::tensor(q8).reshape({outputs,columns}).to(tb::Device(tb::kCUDA,device));
        high.mxfp8.weight.scales=tb::full({(outputs+31)/32,columns/32},121,options.dtype(tb::kUInt8));
        high.mxfp8.weight.out=outputs;high.mxfp8.weight.neuron_len=columns;
        CudaExecutionContext execution;
        auto input=matrix(513,columns,17);
        for(const auto& linear:{low,high}) {
            auto weight=quant_linear_reference_weight(linear).to(tb::kFloat32);
            require(bool(linear.prefill_weight()),"encoded projection excluded the common prefill descriptor");
            const auto expected=prefill_materialize(prefill_product(linear.prefill_weight(),input));
            const auto actual=linear.forward(execution,input);
            oracle(actual,input,weight);
            const auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
            require(a.nbytes()==b.nbytes() && !std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()),
                "encoded projection bypassed its shared prefill product");
            const auto half_input=input.to(tb::kFloat16),half_weight=weight.to(tb::kFloat16);
            auto residual=matrix(513,outputs,41).to(tb::kFloat16);
            const auto immutable=residual.cpu();
            require(linear.can_forward_prefill_residual(513,residual),"encoded projection excluded shared residual GEMM");
            auto dot=tb::empty({513,outputs},half_input.options());
            auto& blas=default_context(device)->blas(mfq_current_cuda_stream());
            gguf::prefill_gemm_f16_accum(half_input.data_ptr(),half_weight.data_ptr(),nullptr,dot.data_ptr(),
                true,nullptr,nullptr,513,outputs,columns,outputs,blas.get(),mfq_current_cuda_stream());
            const auto expected_residual=acc_cuda(residual,dot);
            residual_oracle(linear.forward_prefill_residual(half_input,residual),expected_residual);
            require(!linear.can_forward_prefill_residual(4,residual.narrow(0,0,4)),
                "small decode matrix entered common residual prefill");
            {
                tb::Tensor result;DecodeWindow graph(mfq_current_cuda_stream());
                graph.capture([&]{result=linear.forward_prefill_residual(half_input,residual);},[]{});
                for(float value:{-.125f,.25f}) {
                    residual.fill_(value);graph.run();
                    const auto reference=linear.forward_prefill_residual(half_input,residual);
                    const auto actual=result.cpu(),fresh=reference.cpu();
                    require(!std::memcmp(actual.data_ptr(),fresh.data_ptr(),actual.nbytes()),
                        "common residual graph retained a stale addend");
                }
            }
            residual.copy_(immutable.to(tb::Device(tb::kCUDA,device)));
            const int stride=outputs+7,first=3;
            auto padded_residual=matrix(513,stride,43).to(tb::kFloat16);
            auto padded_output=tb::full({513,stride},.4375,half_input.options());tb::Tensor workspace;
            prefill_residual_gemm(half_input,half_weight,padded_residual,padded_output,first,outputs,blas.get(),workspace);
            residual_oracle(padded_output.narrow(1,first,outputs).contiguous(),
                acc_cuda(padded_residual.narrow(1,first,outputs).contiguous(),dot));
            const auto padded=padded_output.to(tb::kFloat32).cpu();
            for(int row=0;row<513;++row)for(int column=0;column<stride;++column)
                if(column<first || column>=first+outputs)require(padded.data_ptr<float>()[row*stride+column]==.4375f,
                    "common residual GEMM overwrote an output guard column");
            const auto after=residual.cpu();
            require(!std::memcmp(after.data_ptr(),immutable.data_ptr(),after.nbytes()),
                "common residual projection modified its addend");
            FFN ffn;QuantLinear gate,up;gate.kind=up.kind=QuantLinearKind::Dense;
            gate.dense=matrix(columns,outputs,47);up.dense=matrix(columns,outputs,53);
            ffn.gate_up.layers={gate,up};ffn.gate_up.outs={columns,columns};ffn.down=linear;
            auto hidden_input=matrix(513,outputs,59).to(tb::kFloat16);
            const auto ordinary=acc_cuda(residual,ffn.forward(execution,hidden_input).to(tb::kFloat16).contiguous());
            const auto fused=ffn.try_forward_prefill_residual(execution,hidden_input,residual);
            require(fused && fused->defined(),"mixed dense/encoded FFN excluded shared residual dispatch");
            residual_oracle(*fused,ordinary);
        }
        if(gguf::rms_prefill_available(513,columns,device)) {
            QuantLinearGroup group;group.layers={low,high};group.outs={outputs,outputs};
            tb::Tensor normalized;
            auto scope=group.prepare_prefill_rms_norm_input(execution,input,tb::ones({columns},options),1e-6,0,normalized);
            require(bool(scope) && normalized.defined(),"mixed encoded group excluded shared RMS preparation");
            tb::Tensor packed,exponents;
            require(PrefillInputScope::lookup(normalized,packed,exponents),"mixed encoded group lost its prepared activation");
            const auto values=group.forward(execution,normalized);
            require(values.size()==2,"mixed encoded group changed output count");
            oracle(values[0],normalized,quant_linear_reference_weight(low));
            oracle(values[1],normalized,quant_linear_reference_weight(high));
        }
    }
    {
        auto weight=matrix(17,33,19),input=matrix(512,33,23);int decodes=0;
        auto plan=prefill_decoded_projection(weight,17,33,[&]{++decodes;return weight;});
        plan.decode_row_bytes=std::numeric_limits<std::uint64_t>::max();
        require(!prefill_product(plan,input).defined() && !decodes,"overflowing decoder peak invoked a weight decode");
        require(!prefill_decoded_projection(weight,1,std::numeric_limits<std::int64_t>::max(),[&]{++decodes;return weight;}) && !decodes,
            "overflowing decoder geometry was accepted");
        const auto half_input=input.to(tb::kFloat16),residual=tb::zeros({512,17},half_input.options());
        plan.decode_row_bytes=0;
        require(!can_prefill_residual(plan,512,residual) &&
            !prefill_residual_projection(plan,half_input,residual).defined() && !decodes,
            "zero-sized decoder budget entered residual prefill");
        plan.decode_row_bytes=std::numeric_limits<std::uint64_t>::max();
        require(!can_prefill_residual(plan,512,residual) && !decodes,
            "unbounded decoder peak entered residual prefill");
        for(const auto& bad:{weight,weight.transpose(0,1).to(tb::kFloat16).contiguous(),weight.cpu().to(tb::kFloat16)}) {
            auto invalid=prefill_decoded_projection(weight,17,33,[weight]{return weight;});
            invalid.decode_f16=[bad]{return bad;};bool rejected=false;
            try {(void)prefill_residual_projection(invalid,half_input,residual);}
            catch(const std::runtime_error&){rejected=true;}
            require(rejected,"malformed FP16 decoder entered residual prefill");
        }
        auto half_weight=weight.to(tb::kFloat16),output=tb::full({512,17},.4375,half_input.options());
        auto& blas=default_context(device)->blas(mfq_current_cuda_stream());tb::Tensor workspace;
        for(const auto bounds:{std::pair<int64_t,int64_t>{-1,17},{0,18},{17,1},{0,0}}) {
            bool rejected=false;
            try {prefill_residual_gemm(half_input,half_weight,residual,output,bounds.first,bounds.second,blas.get(),workspace);}
            catch(const std::runtime_error&){rejected=true;}
            require(rejected,"invalid residual output tile was submitted");
        }
        const auto untouched=output.to(tb::kFloat32).cpu();
        for(int64_t i=0;i<untouched.numel();++i)require(untouched.data_ptr<float>()[i]==.4375f,
            "rejected residual output tile changed its destination");
        for(const auto& bad:{weight.to(tb::kFloat16),weight.transpose(0,1),weight.cpu()}) {
            bool rejected=false;
            try {(void)prefill_product(prefill_decoded_projection(weight,17,33,[bad]{return bad;}),input);}
            catch(const std::runtime_error&){rejected=true;}
            require(rejected,"malformed decoded weight entered a shared GEMM");
        }
    }
    std::cout<<"Common decoded prefill: FP64 oracles, MXFP4/MXFP8 dispatch, shared RMS, changing graphs and bounds cases="
        <<cases<<" residual_cases="<<residual_cases<<" relative_L2="<<worst_relative
        <<" backward_error="<<worst_backward<<" GGUF_bytes=exact PASS\n";
}
} // namespace mfq::cuda::test
