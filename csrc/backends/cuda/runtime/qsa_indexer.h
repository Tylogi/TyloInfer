#pragma once
#include "../kernels/qsa_tq.h"
#include "mfq_tensor_backend.h"
#include "mfq_cuda_context.h"
#include <algorithm>
#include <climits>

namespace mfq::cuda::qsa_indexer {
namespace tb=mfq_tensor_backend;
inline void check(int status) {TYLOINFER_RUNTIME_CHECK(!status,"QSA INT16 affine Indexer kernel failed");}
inline tb::Tensor encode(const tb::Tensor& source) {
    TYLOINFER_RUNTIME_CHECK(source.is_cuda() && source.dim()>0 && source.size(-1)>0,"invalid Indexer source");
    if(source.scalar_type()==tb::kInt16){TYLOINFER_RUNTIME_CHECK(source.size(-1)>2,"invalid encoded Indexer width");return source;}
    TYLOINFER_RUNTIME_CHECK(source.scalar_type()==tb::kFloat16 || source.scalar_type()==tb::kFloat32,"unsupported Indexer source dtype="+std::to_string(int(source.scalar_type())));
    const auto width=source.size(-1),rows=source.numel()/width;
    TYLOINFER_RUNTIME_CHECK(source.is_cuda() && width>0 && width<=INT_MAX-2 && rows>0 && rows<=INT_MAX,"invalid Indexer geometry");
    const auto input=source.to(tb::kFloat32).contiguous();auto shape=source.sizes().vec();shape.back()+=2;
    auto output=tb::empty(shape,source.options().dtype(tb::kInt16));
    check(tyloinfer::qsa::indexer_affine16_pack(input.data_ptr<float>(),output.data_ptr<int16_t>(),int(rows),int(width),mfq_current_cuda_stream()));
    return output;
}
inline bool pack_at(const tb::Tensor& source,const tb::Tensor& cache,const tb::Tensor& positions,int64_t size) {
    if(!source.is_cuda() || !cache.defined() || !positions.defined() || source.device()!=cache.device() || source.device()!=positions.device() ||
            source.dim()!=3 || source.size(0)!=1 || source.stride(2)!=1 ||
            (source.scalar_type()!=tb::kFloat16 && source.scalar_type()!=tb::kFloat32) ||
            cache.dim()!=3 || cache.size(0)!=1 || !cache.is_contiguous() || cache.scalar_type()!=tb::kInt16 ||
            positions.dim()!=1 || !positions.is_contiguous() || positions.numel()!=source.size(1) ||
            (positions.scalar_type()!=tb::kInt64 && positions.scalar_type()!=tb::kInt32))return false;
    const auto rows=source.size(1),width=source.size(2),capacity=cache.size(1);
    if(rows<=0 || rows>capacity-size || capacity>INT_MAX || width<=0 || width>INT_MAX-2 ||
            cache.size(2)!=width+2 || (rows>1 && source.stride(1)<width))return false;
    check(tyloinfer::qsa::indexer_affine16_pack_at(source.data_ptr(),source.scalar_type()==tb::kFloat16,
        std::max(width,source.stride(1)),cache.data_ptr<int16_t>(),positions.data_ptr(),positions.scalar_type()==tb::kInt64,
        int(rows),int(width),int(capacity),mfq_current_cuda_stream()));
    return true;
}
inline tb::Tensor decode(const tb::Tensor& source) {
    if(source.scalar_type()!=tb::kInt16)return source.to(tb::kFloat32);
    TYLOINFER_RUNTIME_CHECK(source.is_cuda() && source.dim()>0 && source.size(-1)>2,"invalid encoded Indexer geometry");
    const auto width=source.size(-1)-2,rows=source.numel()/(width+2);auto shape=source.sizes().vec();shape.back()-=2;
    auto input=source.contiguous(),output=tb::empty(shape,source.options().dtype(tb::kFloat32));
    check(tyloinfer::qsa::indexer_affine16_unpack(input.data_ptr<int16_t>(),output.data_ptr<float>(),int(rows),int(width),mfq_current_cuda_stream()));return output;
}
inline tb::Tensor scores(const tb::Tensor& query,const tb::Tensor& keys,const tb::Tensor& positions={},int64_t pool=4) {
    auto records=encode(keys).contiguous(),q=query.to(tb::kFloat32).contiguous();
    TYLOINFER_RUNTIME_CHECK(q.dim()==4 && records.dim()==3 && q.size(0)==records.size(0) && q.size(3)+2==records.size(2),"invalid Indexer score geometry");
    auto output=tb::empty({q.size(0),q.size(1),records.size(1)},q.options());
    if(!output.numel())return output;
    const auto p=positions.defined()?positions.to(tb::kInt64).contiguous():tb::Tensor{};
    for(int64_t batch=0;batch<q.size(0);++batch)
        check(tyloinfer::qsa::indexer_affine16_record_scores(q.data_ptr<float>()+batch*q.stride(0),
            records.data_ptr<int16_t>()+batch*records.stride(0),p.defined()?p.data_ptr<int64_t>():nullptr,
            output.data_ptr<float>()+batch*output.stride(0),int(q.size(1)),int(q.size(2)),int(q.size(3)),int(records.size(1)),int(pool),mfq_current_cuda_stream()));
    return output;
}
}
