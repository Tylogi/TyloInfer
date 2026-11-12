#pragma once

#include <functional>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <mlx/backend/metal/device.h>

namespace tyloi::metal {

class KernelDevice {
public:
    explicit KernelDevice(mlx::core::metal::Device& device) : device_(device) {}
    MTL::Library* get_library(const std::string& name,
        const mlx::core::CompileOptions& options, const std::function<std::string()>& source);
    MTL::Library* get_library(const std::string& name, const std::function<std::string()>& source) {
        return get_library(name, {}, source);
    }
    MTL::ComputePipelineState* get_kernel(const std::string& name, MTL::Library* library);
    MTL::ComputePipelineState* get_kernel(const std::string& name, MTL::Library* library,
        const std::string& key, const mlx::core::metal::MTLFCList& constants);
    MTL::ComputePipelineState* get_kernel(const std::string& name) { return device_.get_kernel(name); }
    const std::string& get_architecture() const { return device_.get_architecture(); }
    int get_architecture_gen() const { return device_.get_architecture_gen(); }
private:
    mlx::core::metal::Device& device_;
};

KernelDevice& kernel_device(mlx::core::Device device);

class PreparableKernel {
public:
    virtual ~PreparableKernel() = default;
    virtual std::string preparation_key() const = 0;
    virtual void prepare_gpu() = 0;
};

std::string kernel_key(const mlx::core::Primitive& primitive);
void prepare_kernel(mlx::core::Primitive& primitive);

mlx::core::array row_contiguous_view(const mlx::core::array& input, mlx::core::Stream stream);

mlx::core::fast::CustomKernelFunction custom_kernel(
    const std::string& name, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs, const std::string& source,
    const std::string& header = "", bool contiguous = true, bool atomic = false,
    const mlx::core::CompileOptions& options = {});

} // namespace tyloi::metal
