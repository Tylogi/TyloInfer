#pragma once
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/weights/encoding/row_residual.h"
#include <memory>

class GgufRowCorrection {
    struct Storage;
    std::shared_ptr<Storage> storage_;
    std::int64_t first_=0,rows_=0,column_first_=0,columns_=0;
public:
    GgufRowCorrection(std::shared_ptr<const tyloi::RowResidualFile>,int device,bool mapped_host=false);
    std::shared_ptr<const GgufRowCorrection> slice(std::int64_t begin,std::int64_t end,
        std::int64_t column_begin,std::int64_t column_end) const;
    ::tyloi::DenseTensor add(::tyloi::DenseTensor input,::tyloi::DenseTensor output,
        ::tyloi::DenseTensor ids={},std::int64_t experts=1,bool routed_input=false) const;
    ::tyloi::DenseTensor add_embedding(::tyloi::DenseTensor ids,::tyloi::DenseTensor output) const;
    ::tyloi::DenseTensor dense(int device) const;
};
