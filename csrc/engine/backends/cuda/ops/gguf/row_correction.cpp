#include "row_correction.h"
#include "../../kernels/projection/gguf/residual/row_residual.h"
#include "projection.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include <climits>


namespace tyloi::cuda::gguf {

::tyloi::DenseTensor add_correction(const GgufCorrection& c, ::tyloi::DenseTensor input,::tyloi::DenseTensor output,::tyloi::DenseTensor ids,std::int64_t experts,bool routed) {
    if(!input.is_cuda() && !output.is_cuda() && !ids.defined() && experts==1) {
        if(input.size(-1)!=c.columns || output.size(-1)!=c.rows)
            throw std::invalid_argument("CPU residual projection geometry differs");
        auto x=input.to(::tyloi::kFloat32).contiguous(), y=output.to(::tyloi::kFloat32).contiguous(),delta=dequantize_correction(c,-1);
        const auto tokens=x.numel()/c.columns;
        if(y.numel()!=tokens*c.rows)throw std::invalid_argument("CPU residual token geometry differs");
        for(std::int64_t token=0;token<tokens;++token)for(std::int64_t row=0;row<c.rows;++row) {
            float sum=0;
            for(std::int64_t column=0;column<c.columns;++column)
                sum+=x.data_ptr<float>()[token*c.columns+column]*delta.data_ptr<float>()[row*c.columns+column];
            y.data_ptr<float>()[token*c.rows+row]+=sum;
        }
        return y.to(output.scalar_type());
    }
    if(input.size(-1)!=c.columns || experts<1 || c.rows%experts || input.device()!=output.device() ||
        !input.is_cuda() || input.get_device()!=c.packed.get_device())
        throw std::invalid_argument("GGUF row correction projection geometry differs");
    const auto dtype=output.scalar_type();
    auto x=input.to(::tyloi::kFloat32).contiguous(), y=output.to(::tyloi::kFloat32).contiguous();
    const auto rows=c.rows/experts;
    if(y.size(-1)!=rows || (ids.defined() && (ids.dim()!=2 || ids.device()!=input.device())))
        throw std::invalid_argument("GGUF residual expert/output geometry differs");
    const auto routes=ids.defined()?ids.size(1):1;
    const auto tokens=ids.defined()?ids.size(0):input.numel()/c.columns;
    if(tokens<0 || routes<1 || tokens>INT64_MAX/routes ||
        y.numel()/rows!=tokens*routes || x.numel()/c.columns!=(routed?tokens*routes:tokens))
        throw std::invalid_argument("GGUF residual token geometry differs");
    if(ids.defined())ids=ids.to(::tyloi::kInt64).contiguous();
    const auto stream=mfq_get_current_cuda_stream(c.packed.get_device());
    for(const auto& v:{x,y,ids,c.packed,c.offsets,c.types})
        if(v.defined())mfq_cuda_record_stream(v,stream);
    tyloi::cuda::gguf::add_row_residual(c.packed.data_ptr(),
        reinterpret_cast<const std::uint64_t*>(c.offsets.data_ptr()),c.types.data_ptr<std::int32_t>(),
        c.first,rows,c.column_first,c.columns,x.data_ptr<float>(),y.data_ptr<float>(),tokens,
        ids.defined()?ids.data_ptr<std::int64_t>():nullptr,routes,experts,routed,stream.stream());
    return y.to(dtype);
}
::tyloi::DenseTensor add_correction_embedding(const GgufCorrection& c, ::tyloi::DenseTensor ids,::tyloi::DenseTensor output) {
    if(!output.is_cuda()) {
        auto full=dequantize_correction(c,-1),flat=ids.to(::tyloi::kInt64).cpu().contiguous().reshape({-1});
        auto y=output.to(::tyloi::kFloat32).contiguous();
        if(output.size(-1)!=c.columns || flat.numel()!=output.numel()/c.columns)
            throw std::invalid_argument("CPU residual lookup geometry differs");
        for(std::int64_t i=0;i<flat.numel();++i) {
            const auto row=flat.data_ptr<std::int64_t>()[i];
            if(row>=0 && row<c.rows)for(std::int64_t column=0;column<c.columns;++column)
                y.data_ptr<float>()[i*c.columns+column]+=full.data_ptr<float>()[row*c.columns+column];
        }
        return y.to(output.scalar_type());
    }
    if(!output.is_cuda() || output.get_device()!=c.packed.get_device() || output.size(-1)!=c.columns)
        throw std::invalid_argument("GGUF residual embedding device/width differs");
    const auto dtype=output.scalar_type();auto y=output.to(::tyloi::kFloat32).contiguous();
    if(ids.defined()) {
        if(ids.device()!=output.device())throw std::invalid_argument("GGUF residual embedding IDs are on another device");
        ids=ids.to(::tyloi::kInt64).contiguous();
        if(ids.numel()!=output.numel()/c.columns)throw std::invalid_argument("GGUF residual embedding ID count differs");
    }
    const auto stream=mfq_get_current_cuda_stream(c.packed.get_device());
    for(const auto& v:{y,ids,c.packed,c.offsets,c.types})if(v.defined())mfq_cuda_record_stream(v,stream);
    tyloi::cuda::gguf::add_residual_embedding(c.packed.data_ptr(),
        reinterpret_cast<const std::uint64_t*>(c.offsets.data_ptr()),c.types.data_ptr<std::int32_t>(),
        c.first,c.rows,c.column_first,c.columns,ids.defined()?ids.data_ptr<std::int64_t>():nullptr,
        y.data_ptr<float>(),output.numel()/c.columns,stream.stream());
    return y.to(dtype);
}
::tyloi::DenseTensor dequantize_correction(const GgufCorrection& c, int device) {
    if(device>=0) {
        if(device!=c.packed.get_device())throw std::invalid_argument("GGUF residual belongs to another device");
        return add_correction_embedding(c,{},::tyloi::cuda::zeros({c.rows,c.columns},c.offsets.options().dtype(::tyloi::kFloat32)));
    }
    auto out=::tyloi::cuda::zeros({c.rows,c.columns},::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kFloat32));
    const auto packed=c.packed.cpu().contiguous();
    const auto offsets=c.offsets.cpu().contiguous();
    const auto types=c.types.cpu().contiguous();
    std::vector<float> row((c.source_columns+127)/128*128);
    for(std::int64_t i=0;i<c.rows;++i) {
        const auto index=c.first+i;
        const auto type=types.data_ptr<std::int32_t>()[index];
        if(!type)continue;
        decode_cpu(ggml_type(type),packed.data_ptr<std::uint8_t>()+
            offsets.data_ptr<std::int64_t>()[index],row.data(),std::int64_t(row.size()));
        std::copy_n(row.data()+c.column_first,c.columns,out.data_ptr<float>()+i*c.columns);
    }
    return out;
}

} // namespace tyloi::cuda::gguf
