#pragma once

#include "engine/tensor.h"

#include <memory>
#include <chrono>
#include <initializer_list>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tyloi {

class Executor {
public:
    struct Memory {
        Device device;
        // Allocator accounting, followed by device physical free and total bytes.
        std::size_t allocated, reserved, available, total;
        // Process limit (0 means unlimited), charged residency, allocatable bytes and reusable workspace.
        std::size_t limit = 0, resident = 0, allocatable = 0, workspace = 0;
    };

    class Scope {
    public:
        virtual ~Scope() = default;
    };

    // Starts on creation; stop records completion and elapsed_ms waits for it.
    class Timer {
    public:
        virtual ~Timer() = default;
        virtual void stop() = 0;
        // Lazy backends evaluate these results at the timing boundary.
        virtual void stop(std::initializer_list<Tensor>) { stop(); }
        virtual double elapsed_ms() const = 0;
    };

    // Call finish once on success; destruction restores profiling after an exception.
    class Profile {
    public:
        virtual ~Profile() = default;
        virtual void finish() = 0;
        virtual void finish(std::initializer_list<Tensor>) { finish(); }
    };

    // The parent Executor outlives Graph; do not keep its Scope across suspension.
    // Invalidate before prepare, restore after each warmup, then begin and end capture.
    // end retains backend scratch; matches compares it along with the supplied owners.
    // invalidate waits for all consumers before releasing owners; failure retains them.
    // A backend must not release unproven in-flight owners during destruction.
    class Graph {
    public:
        virtual ~Graph() = default;
        virtual std::unique_ptr<Scope> enter() = 0;
        virtual bool matches(std::span<const Tensor> owners) const = 0;
        virtual void prepare(std::span<Tensor* const> recurrent) = 0;
        virtual void restore() = 0;
        virtual void begin() = 0;
        virtual void end(std::span<const Tensor> owners) = 0;
        virtual void replay() = 0;
        virtual void wait() = 0;
        virtual void invalidate() = 0;
    };

    virtual ~Executor() = default;
    // Destroy scopes in reverse order on the entering thread, before their resources.
    // Exiting a scope does not synchronize device work.
    virtual std::unique_ptr<Scope> enter() = 0;
    virtual std::unique_ptr<Timer> timer() = 0;
    virtual std::unique_ptr<Profile> profile(const char* name) = 0;
    virtual std::unique_ptr<Graph> graph() = 0;
    // Query within this executor's scope; one snapshot per requested device, in the same order.
    virtual std::vector<Memory> memory(std::span<const Device> devices) const = 0;
    // Call within this executor's scope; stop submissions before changing a limit.
    virtual void reclaim(Device device) = 0;
    virtual void set_memory_limit(Device device, std::size_t bytes) = 0;
    // Stop concurrent submissions and call within this executor's scope.
    // Wait for executor work; owners coordinate weights and offload separately.
    virtual void drain() = 0;
    // After draining and destroying graphs, clear execution caches within the scope.
    virtual void clear() = 0;
};

} // namespace tyloi

namespace tyloi::profiling {

struct Timing {
    double elapsed_ms = 0;
    std::size_t evaluations = 0;
};

class Components {
public:
    void record(std::string_view component, double elapsed_ms);
    const std::map<std::string, Timing>& timings() const noexcept;
    double evaluated_ms() const noexcept;

private:
    std::map<std::string, Timing> timings_;
};

// Scopes must be destroyed in reverse order on the thread that created them.
class ProfileScope {
public:
    explicit ProfileScope(Components* profile) noexcept;
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
    ~ProfileScope();

private:
    static thread_local Components* active_;
    Components* previous_ = nullptr;
    friend bool active() noexcept;
    friend void mark(std::string_view);
    template<class Function> friend void component(std::string_view, Function&&);
};

class EvaluationScope {
public:
    explicit EvaluationScope(double* elapsed_ms) noexcept;
    EvaluationScope(const EvaluationScope&) = delete;
    EvaluationScope& operator=(const EvaluationScope&) = delete;
    ~EvaluationScope();

private:
    static thread_local double* active_;
    double* previous_ = nullptr;
    template<class Function> friend void measure(Function&&);
};

bool active() noexcept;
void mark(std::string_view component);

// Lazy graph construction is excluded; only explicit evaluation is timed.
template<class Function>
void measure(Function&& function) {
    if (EvaluationScope::active_ == nullptr) {
        std::forward<Function>(function)();
        return;
    }
    const auto started = std::chrono::steady_clock::now();
    std::forward<Function>(function)();
    *EvaluationScope::active_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

template<class Function>
void component(std::string_view name, Function&& function) {
    if (ProfileScope::active_ == nullptr) {
        return;
    }
    const auto started = std::chrono::steady_clock::now();
    std::forward<Function>(function)();
    ProfileScope::active_->record(name, std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count());
}

} // namespace tyloi::profiling
