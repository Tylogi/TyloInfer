#pragma once

#include "generation_step.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <variant>

namespace tyloi::metal {

// The native MLX pipeline runs on one device-owning thread. Only owned token
// and timing values cross into the scheduler; no response writer runs there.
class MlxGenerationJob {
public:
    using Event = std::variant<std::int64_t, MfqPrefillTiming>;
    using Work = std::function<void(MlxGenerationJob&)>;
    explicit MlxGenerationJob(Work work);
    ~MlxGenerationJob();
    MlxGenerationJob(const MlxGenerationJob&) = delete;
    MlxGenerationJob& operator=(const MlxGenerationJob&) = delete;

    bool token(std::int64_t token);
    void prefill(MfqPrefillTiming timing);
    bool cancelled() const;
    void cancel();
    std::optional<Event> take();
    bool done() const;
    void join();
    tyloi::engine::mtp::GenerationStats mtp;

private:
    bool push(Event event);
    mutable std::mutex mutex_;
    std::condition_variable space_;
    std::deque<Event> events_;
    bool cancelled_ = false, done_ = false;
    std::exception_ptr failure_;
    std::thread worker_;
    static constexpr std::size_t capacity_ = 4;
};

tyloi::engine::Generation consume_mlx_generation(
    MlxGenerationJob::Work work, tyloi::engine::InferenceOutput& output);

} // namespace tyloi::metal
