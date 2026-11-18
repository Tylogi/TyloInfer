#include "engine/weights/rows/prefetch.h"

#include <array>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

namespace tyloi::weights {
namespace {
class Workers {
public:
    Workers() {
        try {
            for (auto& worker : workers_) worker = std::thread([this] { work(); });
        } catch (...) {
            { std::lock_guard lock(mutex_); stopping_ = true; }
            ready_.notify_all();
            for (auto& worker : workers_) if (worker.joinable()) worker.join();
            throw;
        }
    }
    ~Workers() {
        { std::lock_guard lock(mutex_); stopping_ = true; }
        ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
    void submit(std::packaged_task<void()> job) {
        { std::lock_guard lock(mutex_); jobs_.push(std::move(job)); }
        ready_.notify_one();
    }
private:
    void work() {
        for (;;) {
            std::packaged_task<void()> job;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
                if (jobs_.empty()) return;
                job = std::move(jobs_.front());
                jobs_.pop();
            }
            job();
        }
    }
    std::mutex mutex_;
    std::condition_variable ready_;
    std::queue<std::packaged_task<void()>> jobs_;
    std::array<std::thread, 2> workers_;
    bool stopping_ = false;
};
} // namespace

void submit_row_read(std::packaged_task<void()> job) {
    static Workers workers;
    workers.submit(std::move(job));
}

} // namespace tyloi::weights
