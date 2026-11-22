#pragma once

#include "models/deepseek_v41/engram.h"
#include "models/hy4_flash/config.h"
#include "quant_linear.h"
#include "runtime/row_packet_prefetch.h"
#include "tyloinfer/model_source.h"

namespace mfq::cuda::hy4 {

struct EngramState {
    using Hash=mfq::models::deepseek_v41::EngramHashState;
    using HashBatch=mfq::models::deepseek_v41::EngramHashBatch;
    std::unique_ptr<Hash> hash;
    HashBatch batch;
    mfq::ModelSource::TensorBatchReader read_rows;
    uint64_t table_rows=0;
    int64_t row_width=0;

    void load(const mfq::ModelSource&,const mfq::models::hy4_flash::EngramConfig&);
    void prepare(const mfq_tensor_backend::Tensor& token_ids,int64_t position);
    void prepare(HashBatch hashes);
    bool prefetch(const mfq_tensor_backend::Tensor& host_ids,int64_t position);
    std::pair<mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor> lookup(int layer,mfq_tensor_backend::Device);
private:
    void enqueue(const HashBatch&);
    bool parallel_reads_=false;
    bool packed_valid_=false;
    mfq_tensor_backend::Tensor packed_;
    RowPacketPrefetch prefetch_;
};

struct EngramLayer {
    int slot=-1;
    QuantLinear projection;
    mfq_tensor_backend::Tensor lookup_norm, projection_bias, gamma, gate, hc_scale, unit;
    mfq_tensor_backend::Tensor forward(CudaExecutionContext&,EngramState&,
        mfq_tensor_backend::Tensor streams) const;
    mfq_tensor_backend::Tensor forward_rows(CudaExecutionContext&,mfq_tensor_backend::Tensor streams,
        mfq_tensor_backend::Tensor rows,const mfq_tensor_backend::Tensor& valid) const;
};

}  // namespace mfq::cuda::hy4
