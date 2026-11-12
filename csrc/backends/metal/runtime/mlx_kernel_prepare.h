#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <mlx/mlx.h>
#include "engine/backends/metal/native/kernel.h"
#include "mlx_sampling.h"

namespace mfq::metal {

class MlxKernelPreparation {
public:
    MlxKernelPreparation(int context, int chunk_size);
    ~MlxKernelPreparation();
    MlxKernelPreparation(const MlxKernelPreparation&) = delete;
    MlxKernelPreparation& operator=(const MlxKernelPreparation&) = delete;
    static MlxKernelPreparation* current() noexcept;
    static bool collecting() noexcept;
    void collect(const std::function<void()>& build);
    void add(const mlx::core::array& output);
    void finish();
    void collect_sampling(int vocab, const MlxSamplingParams& defaults);
    void use_model_paths() noexcept { model_paths_ = true; }
    bool model_paths() const noexcept { return model_paths_; }
    const std::vector<int>& row_buckets() const noexcept { return rows_; }
    int context() const noexcept { return context_; }
    void set_routes(int routes) noexcept { routes_ = routes; }
    int routes() const noexcept { return routes_; }
    void set_activation_dtype(mlx::core::Dtype dtype) noexcept { activation_dtype_ = dtype; }
    mlx::core::Dtype activation_dtype() const noexcept { return activation_dtype_; }
    std::size_t size() const noexcept { return jobs_.size() + compiled_jobs_.size(); }
    const std::unordered_set<std::string>& unsupported() const noexcept { return unsupported_; }

private:
    static thread_local MlxKernelPreparation* active_;
    MlxKernelPreparation* previous_;
    bool collecting_ = false;
    bool model_paths_ = false;
    int routes_ = 0;
    int context_ = 0;
    mlx::core::Dtype activation_dtype_ = mlx::core::float16;
    std::vector<int> rows_;
    std::unordered_set<std::string> keys_;
    std::unordered_set<std::string> unsupported_;
    std::vector<std::function<void()>> jobs_;
    std::vector<std::function<void()>> compiled_jobs_;
};

} // namespace mfq::metal
