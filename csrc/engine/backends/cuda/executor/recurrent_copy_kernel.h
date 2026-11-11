#pragma once
#include "dma_copy_batch.h"

namespace tyloi::cuda {
void copy_recurrent_states(const DmaCopy* copies,std::size_t count,cudaStream_t stream);
}
