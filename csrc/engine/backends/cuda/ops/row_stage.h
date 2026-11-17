#pragma once

#include "engine/weights/rows/stage.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/native/stream.h"

#include <memory>
#include <stdexcept>

namespace tyloi::cuda {

// Host row packets are reusable storage.  Eager consumers leave an event
// behind; graph owners fence their replay externally and must not make that
// event host-synchronizable.
class RowStageFence {
    std::unique_ptr<Event> done_;
    cudaStream_t stream_ = nullptr;
    bool pending_ = false;

public:
    void initialize() {
        if (!done_) done_ = std::make_unique<Event>();
    }

    void before_write() {
        if (pending_) done_->synchronize();
        pending_ = false;
    }

    void owner_fenced() noexcept { pending_ = false; }

    void record_transfer(cudaStream_t stream) {
        try {
            initialize();
            done_->record(stream);
        } catch (...) {
            (void)cudaStreamSynchronize(stream);
            throw;
        }
        stream_ = stream;
        pending_ = true;
    }

    cudaStreamCaptureStatus join(cudaStream_t stream) const {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (pending_ && stream != stream_) {
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(
                stream, done_->get(),
                capture == cudaStreamCaptureStatusNone ? 0 : cudaEventWaitExternal));
        }
        return capture;
    }

    void record_consumer(cudaStream_t stream, cudaStreamCaptureStatus capture) {
        if (capture == cudaStreamCaptureStatusNone) record_transfer(stream);
    }
};

std::shared_ptr<RowStage> make_dense_row_stage(const std::vector<Tensor>& tables,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& shape, Device device);

} // namespace tyloi::cuda
