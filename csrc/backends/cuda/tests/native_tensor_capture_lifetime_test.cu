#include "mfq_cuda_context.h"
#include "mfq_native_tensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mfq::cuda::Tensor;
using mfq::cuda::TensorOptions;

constexpr std::size_t kElements = 32;
constexpr std::int32_t kExpected = static_cast<std::int32_t>(0x3c3c3c3c);

struct EventOwner {
    cudaEvent_t value = nullptr;

    ~EventOwner() {
        if (value != nullptr) (void)cudaEventDestroy(value);
    }
};

void check_output(const Tensor& output, const char* name) {
    std::vector<std::int32_t> host(kElements);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpy(
        host.data(), output.data_ptr(), output.nbytes(), cudaMemcpyDeviceToHost));
    for (const auto value : host) {
        if (value != kExpected)
            throw std::runtime_error(std::string(name) + " produced an incorrect replay value");
    }
}

Tensor make_output(const mfq::cuda::StreamHandle& stream) {
    mfq::cuda::StreamGuard guard(stream);
    auto output = mfq::cuda::empty(
        {static_cast<std::int64_t>(kElements)},
        TensorOptions{}.device(mfq::cuda::kCUDA).dtype(mfq::cuda::kInt32));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    return output;
}

void create_event(EventOwner& event) {
    TYLOINFER_NATIVE_CUDA_CHECK(
        cudaEventCreateWithFlags(&event.value, cudaEventDisableTiming));
}

void run_ordinary_cross_stream_case() {
    const auto origin = mfq::cuda::stream_from_pool(false);
    const auto side = mfq::cuda::stream_from_pool(false);
    auto output = make_output(origin);
    EventOwner ready;
    create_event(ready);
    {
        mfq::cuda::StreamGuard side_guard(side);
        auto source = mfq::cuda::empty(
            {static_cast<std::int64_t>(kElements)},
            TensorOptions{}.device(mfq::cuda::kCUDA).dtype(mfq::cuda::kInt32));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaMemsetAsync(
            source.data_ptr(), 0x2a, source.nbytes(), side.stream()));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(ready.value, side.stream()));
        {
            mfq::cuda::StreamGuard origin_guard(origin);
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(origin.stream(), ready.value, 0));
            source.record_stream(
                reinterpret_cast<std::uintptr_t>(origin.stream()), origin.lifetime());
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemsetAsync(
                source.data_ptr(), 0x3c, source.nbytes(), origin.stream()));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(
                output.data_ptr(), source.data_ptr(), source.nbytes(),
                cudaMemcpyDeviceToDevice, origin.stream()));
            source = {};
        }
    }
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(origin.stream()));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(side.stream()));
    check_output(output, "ordinary cross-stream release");
}

void run_capture_case(bool second_consumer, bool participant_pool) {
    const auto origin = mfq::cuda::stream_from_pool(false);
    const auto side = mfq::cuda::stream_from_pool(false);
    const auto second = mfq::cuda::stream_from_pool(false);
    auto output = make_output(origin);
    auto second_output = second_consumer ? make_output(origin) : Tensor{};

    EventOwner fork, ready, origin_done, second_done;
    create_event(fork);
    create_event(ready);
    if (second_consumer) {
        create_event(origin_done);
        create_event(second_done);
    }

    mfq::cuda::StreamGuard origin_guard(origin);
    mfq::cuda::Graph graph;
    if (participant_pool) {
        graph.prepare_memory({side});
        {
            mfq::cuda::StreamGuard side_guard(side);
            auto warmup = mfq::cuda::empty(
                {static_cast<std::int64_t>(kElements)},
                TensorOptions{}.device(mfq::cuda::kCUDA).dtype(mfq::cuda::kInt32));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(side.stream()));
        }
    } else {
        graph.prepare_memory();
    }
    graph.capture_begin();
    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(fork.value, origin.stream()));

    {
        mfq::cuda::StreamGuard side_guard(side);
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(side.stream(), fork.value, 0));
        auto source = mfq::cuda::empty(
            {static_cast<std::int64_t>(kElements)},
            TensorOptions{}.device(mfq::cuda::kCUDA).dtype(mfq::cuda::kInt32));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaMemsetAsync(
            source.data_ptr(), 0x2a, source.nbytes(), side.stream()));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(ready.value, side.stream()));
        {
            mfq::cuda::StreamGuard consumer_guard(origin);
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(origin.stream(), ready.value, 0));
            source.record_stream(
                reinterpret_cast<std::uintptr_t>(origin.stream()), origin.lifetime());
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemsetAsync(
                source.data_ptr(), 0x3c, source.nbytes(), origin.stream()));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(
                output.data_ptr(), source.data_ptr(), source.nbytes(),
                cudaMemcpyDeviceToDevice, origin.stream()));
            if (second_consumer) {
                TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(origin_done.value, origin.stream()));
                {
                    mfq::cuda::StreamGuard second_guard(second);
                    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(
                        second.stream(), origin_done.value, 0));
                    source.record_stream(
                        reinterpret_cast<std::uintptr_t>(second.stream()), second.lifetime());
                    TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(
                        second_output.data_ptr(), source.data_ptr(), source.nbytes(),
                        cudaMemcpyDeviceToDevice, second.stream()));
                    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(
                        second_done.value, second.stream()));
                }
                TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(
                    origin.stream(), second_done.value, 0));
            }
            // The owner was allocated on side and is released while origin captures.
            source = {};
        }
    }

    graph.capture_end();
    for (int replay = 0; replay < 3; ++replay) {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaMemsetAsync(
            output.data_ptr(), 0, output.nbytes(), origin.stream()));
        if (second_consumer) TYLOINFER_NATIVE_CUDA_CHECK(cudaMemsetAsync(
            second_output.data_ptr(), 0, second_output.nbytes(), origin.stream()));
        graph.replay();
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(origin.stream()));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(side.stream()));
        if (second_consumer)
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(second.stream()));
        check_output(output, second_consumer ? "capture origin consumer" : "capture consumer");
        if (second_consumer) check_output(second_output, "capture second consumer");
    }
    std::cout << "capture_case=" << (second_consumer ? "multiple" : "single")
              << " pool=" << (participant_pool ? 1 : 0)
              << " replay_rounds=3 PASS\n";
}

} // namespace

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        run_ordinary_cross_stream_case();
        std::cout << "capture_case=ordinary replay_rounds=1 PASS\n";
        run_capture_case(false, false);
        run_capture_case(true, false);
        run_capture_case(false, true);
        run_capture_case(true, true);
        std::cout << "native tensor capture lifetime cases=5 PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native tensor capture lifetime failed: " << error.what() << '\n';
        return 1;
    }
}
