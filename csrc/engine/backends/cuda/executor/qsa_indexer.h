#pragma once
#include "../kernels/qsa_tq.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/native/context.h"
#include <climits>

namespace tyloi::cuda::qsa_indexer {

inline void check(int status) {TYLOINFER_RUNTIME_CHECK(!status,"QSA INT16 affine Indexer kernel failed");}
inline ::tyloi::DenseTensor encode(const ::tyloi::DenseTensor& source) {
    TYLOINFER_RUNTIME_CHECK(source.is_cuda() && source.dim()>0 && source.size(-1)>0,"invalid Indexer source");
    if(source.scalar_type()==::tyloi::kInt16){TYLOINFER_RUNTIME_CHECK(source.size(-1)>2,"invalid encoded Indexer width");return source;}
    TYLOINFER_RUNTIME_CHECK(source.scalar_type()==::tyloi::kFloat16 || source.scalar_type()==::tyloi::kFloat32,"unsupported Indexer source dtype="+std::to_string(int(source.scalar_type())));
    const auto width=source.size(-1),rows=source.numel()/width;
    TYLOINFER_RUNTIME_CHECK(source.is_cuda() && width>0 && width<=INT_MAX-2 && rows>0 && rows<=INT_MAX,"invalid Indexer geometry");
    const auto input=source.to(::tyloi::kFloat32).contiguous();auto shape=source.sizes().vec();shape.back()+=2;
    auto output=::tyloi::cuda::empty(shape,source.options().dtype(::tyloi::kInt16));
    check(tyloinfer::qsa::indexer_affine16_pack(input.data_ptr<float>(),output.data_ptr<int16_t>(),int(rows),int(width),mfq_current_cuda_stream()));
    return output;
}
inline ::tyloi::DenseTensor decode(const ::tyloi::DenseTensor& source) {
    if(source.scalar_type()!=::tyloi::kInt16)return source.to(::tyloi::kFloat32);
    TYLOINFER_RUNTIME_CHECK(source.is_cuda() && source.dim()>0 && source.size(-1)>2,"invalid encoded Indexer geometry");
    const auto width=source.size(-1)-2,rows=source.numel()/(width+2);auto shape=source.sizes().vec();shape.back()-=2;
    auto input=source.contiguous(),output=::tyloi::cuda::empty(shape,source.options().dtype(::tyloi::kFloat32));
    check(tyloinfer::qsa::indexer_affine16_unpack(input.data_ptr<int16_t>(),output.data_ptr<float>(),int(rows),int(width),mfq_current_cuda_stream()));return output;
}
inline ::tyloi::DenseTensor scores(const ::tyloi::DenseTensor& query,const ::tyloi::DenseTensor& keys,const ::tyloi::DenseTensor& positions={},int64_t pool=4) {
    auto records=encode(keys).contiguous(),q=query.to(::tyloi::kFloat32).contiguous();
    TYLOINFER_RUNTIME_CHECK(q.dim()==4 && records.dim()==3 && q.size(0)==records.size(0) && q.size(3)+2==records.size(2),"invalid Indexer score geometry");
    auto output=::tyloi::cuda::empty({q.size(0),q.size(1),records.size(1)},q.options());
    if(!output.numel())return output;
    const auto p=positions.defined()?positions.to(::tyloi::kInt64).contiguous(): ::tyloi::DenseTensor{};
    for(int64_t batch=0;batch<q.size(0);++batch)
        check(tyloinfer::qsa::indexer_affine16_record_scores(q.data_ptr<float>()+batch*q.stride(0),
            records.data_ptr<int16_t>()+batch*records.stride(0),p.defined()?p.data_ptr<int64_t>():nullptr,
            output.data_ptr<float>()+batch*output.stride(0),int(q.size(1)),int(q.size(2)),int(q.size(3)),int(records.size(1)),int(pool),mfq_current_cuda_stream()));
    return output;
}
}
