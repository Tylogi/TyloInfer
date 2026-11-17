#pragma once
#include <condition_variable>
#include <deque>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

// Cache admission is serialized outside decode windows. Reuse its host thread
// and CUDA thread state across windows; destruction drains accepted work.
class RoutedAdmissionExecutor {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::packaged_task<void()>> queue_;
    bool stopping_=false;
    std::size_t capacity_;
    std::thread worker_;
    static std::size_t checked_capacity(std::size_t capacity) {
        if(!capacity)throw std::invalid_argument("routed worker queue capacity must be positive");
        return capacity;
    }
    void run() {
        for(;;) {
            std::packaged_task<void()> task;
            {
                std::unique_lock lock(mutex_);
                changed_.wait(lock,[&]{return stopping_ || !queue_.empty();});
                if(queue_.empty())return;
                task=std::move(queue_.front());queue_.pop_front();
            }
            task(); // packaged_task forwards failures to the caller's future.
        }
    }
public:
    explicit RoutedAdmissionExecutor(std::size_t capacity=std::numeric_limits<std::size_t>::max())
        :capacity_(checked_capacity(capacity)),worker_([this]{run();}){}
    RoutedAdmissionExecutor(const RoutedAdmissionExecutor&)=delete;
    RoutedAdmissionExecutor& operator=(const RoutedAdmissionExecutor&)=delete;
    ~RoutedAdmissionExecutor() {
        {std::lock_guard lock(mutex_);stopping_=true;}
        changed_.notify_one();worker_.join();
    }
    template<class Function> std::future<void> submit(Function&& function) {
        std::packaged_task<void()> task(std::forward<Function>(function));
        auto result=task.get_future();
        {
            std::lock_guard lock(mutex_);
            if(stopping_)throw std::logic_error("cache admission worker has stopped");
            if(queue_.size()>=capacity_)throw std::logic_error("routed worker queue is full");
            queue_.push_back(std::move(task));
        }
        changed_.notify_one();return result;
    }
};

// Serialize the handoff from admission to cache readers, including preparation
// of the next job. The worker never takes this lock or waits for a reader.
class RoutedAdmission {
    RoutedAdmissionExecutor executor_;
    std::mutex mutex_;
    std::future<void> pending_;
public:
    class Transition {
        friend class RoutedAdmission;
        RoutedAdmission& owner_;
        explicit Transition(RoutedAdmission& owner):owner_(owner){}
    public:
        bool pending()const{return owner_.pending_.valid();}
        void wait() {
            auto job=std::exchange(owner_.pending_,{});
            if(job.valid())job.get();
        }
        template<class Function> void submit(Function&& function) {
            if(pending())throw std::logic_error("cache admission already pending");
            owner_.pending_=owner_.executor_.submit(std::forward<Function>(function));
        }
    };
    template<class Function> auto transition(Function&& function) {
        std::lock_guard lock(mutex_);
        Transition state(*this);
        return std::forward<Function>(function)(state);
    }
    bool pending(){return transition([](auto& state){return state.pending();});}
    void wait(){transition([](auto& state){state.wait();});}
    template<class Function> void submit(Function&& function) {
        transition([&](auto& state){state.wait();state.submit(std::forward<Function>(function));});
    }
    ~RoutedAdmission(){try{wait();}catch(...) {}}
};
