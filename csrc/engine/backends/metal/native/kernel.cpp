#include "kernel.h"
#include "mfq_mlx_utils_embedded.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <CommonCrypto/CommonDigest.h>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <mach-o/dyld.h>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mlx/backend/metal/utils.h>
#include <mlx/fast_primitives.h>
#include <mlx/utils.h>
#include <mlx/version.h>

namespace tyloi::metal {
namespace {

template <typename T> struct Cache {
    std::shared_mutex mutex;
    std::unordered_map<std::string, std::shared_future<NS::SharedPtr<T>>> entries;
};

template <typename T, typename Build>
T* cached(Cache<T>& cache, const std::string& key, Build build) {
    std::shared_future<NS::SharedPtr<T>> future;
    {
        std::shared_lock lock(cache.mutex);
        if (const auto found = cache.entries.find(key); found != cache.entries.end())
            future = found->second;
    }
    if (future.valid()) return future.get().get();
    std::promise<NS::SharedPtr<T>> promise;
    {
        std::unique_lock lock(cache.mutex);
        const auto [entry, inserted] = cache.entries.try_emplace(key, promise.get_future().share());
        future = entry->second;
        if (!inserted) {
            lock.unlock();
            return future.get().get();
        }
    }
    try { promise.set_value(build()); }
    catch (...) {
        promise.set_exception(std::current_exception());
        std::unique_lock lock(cache.mutex);
        cache.entries.erase(key);
    }
    return future.get().get();
}

MTL::LanguageVersion language_version() {
    if (__builtin_available(macOS 27, iOS 27, tvOS 27, visionOS 27, *))
        return static_cast<MTL::LanguageVersion>((4 << 16) + 1);
    if (__builtin_available(macOS 26, iOS 26, tvOS 26, visionOS 26, *))
        return MTL::LanguageVersion4_0;
    if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *))
        return MTL::LanguageVersion3_2;
    return MTL::LanguageVersion3_1;
}

MTL::Library* precompiled_library(MTL::Device* device, const std::string& source, int mode) {
    const auto fingerprint = source + '\0' + std::to_string(mode) + '\0'
        + std::to_string(static_cast<unsigned>(language_version()));
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(fingerprint.data(), static_cast<CC_LONG>(fingerprint.size()), digest);
    std::string filename;
    for (const auto byte : digest) {
        filename += "0123456789abcdef"[byte >> 4];
        filename += "0123456789abcdef"[byte & 15];
    }
    filename += ".metallib";
    static const auto roots = [] {
        std::vector<std::filesystem::path> paths;
        if (const auto* value = std::getenv("TYLOINFER_METAL_LIBRARY_DIR")) paths.emplace_back(value);
        std::uint32_t size = 0;
        _NSGetExecutablePath(nullptr, &size);
        std::string executable(size, '\0');
        if (_NSGetExecutablePath(executable.data(), &size) == 0) {
            const auto parent = std::filesystem::weakly_canonical(executable.c_str()).parent_path();
            paths.push_back(parent / "lib" / "mfq-metal");
            paths.push_back(parent / ".." / "Resources" / "mfq-metal");
        }
        return paths;
    }();
    for (const auto& root : roots) {
        const auto path = root / filename;
        if (!std::filesystem::is_regular_file(path)) continue;
        NS::Error* error = nullptr;
        auto* library = device->newLibrary(NS::String::string(path.c_str(), NS::UTF8StringEncoding), &error);
        if (!library) throw std::runtime_error("Cannot load bundled Metal library: " + path.string()
            + ": " + (error ? error->localizedDescription()->utf8String() : "invalid library"));
        return library;
    }
    return nullptr;
}

}

KernelDevice& kernel_device(mlx::core::Device device) {
    static auto* cache = new KernelDevice(mlx::core::metal::device(device));
    return *cache;
}

MTL::Library* KernelDevice::get_library(const std::string& name,
    const mlx::core::CompileOptions& options, const std::function<std::string()>& source) {
    const auto mode = options.serialize();
    if (mode < 0 || mode > 2) throw std::invalid_argument("invalid Metal math mode");
    static auto* libraries = new std::array<Cache<MTL::Library>, 3>;
    return cached(libraries->at(mode), name, [&] {
        auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        const auto text = source();
        if (auto* library = precompiled_library(device_.mtl_device(), text, mode))
            return NS::TransferPtr(library);
        auto config = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
        config->setLanguageVersion(language_version());
        if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)) {
            const MTL::MathMode modes[]{MTL::MathModeSafe, MTL::MathModeRelaxed, MTL::MathModeFast};
            config->setMathMode(modes[mode]);
        } else {
            if (options.math_mode == mlx::core::MathMode::Relaxed)
                throw std::runtime_error("relaxed Metal math requires macOS 15");
            config->setFastMathEnabled(options.math_mode == mlx::core::MathMode::Fast);
        }
#ifndef NDEBUG
        if (config->languageVersion() >= MTL::LanguageVersion3_2) config->setEnableLogging(true);
#endif
        NS::Error* error = nullptr;
        auto library = NS::TransferPtr(device_.mtl_device()->newLibrary(
            NS::String::string(text.c_str(), NS::UTF8StringEncoding), config.get(), &error));
        if (!library) throw std::runtime_error("Metal library " + name + ": "
            + (error ? error->localizedDescription()->utf8String() : "compilation failed"));
        return library;
    });
}

MTL::ComputePipelineState* KernelDevice::get_kernel(const std::string& name, MTL::Library* library) {
    return get_kernel(name, library, name, {});
}

MTL::ComputePipelineState* KernelDevice::get_kernel(const std::string& name, MTL::Library* library,
        const std::string& key, const mlx::core::metal::MTLFCList& constants) {
    static auto* mutex = new std::mutex;
    static auto* pipelines = new std::unordered_map<MTL::Library*, std::unique_ptr<Cache<MTL::ComputePipelineState>>>;
    Cache<MTL::ComputePipelineState>* cache;
    {
        std::lock_guard lock(*mutex);
        auto& entry = (*pipelines)[library];
        if (!entry) entry = std::make_unique<Cache<MTL::ComputePipelineState>>();
        cache = entry.get();
    }
    return cached(*cache, key, [&] {
        auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        NS::Error* error = nullptr;
        NS::SharedPtr<MTL::Function> function;
        const auto entry = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
        if (constants.empty()) function = NS::TransferPtr(library->newFunction(entry));
        else {
            auto values = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
            for (const auto& [value, type, index] : constants) values->setConstantValue(value, type, index);
            function = NS::TransferPtr(library->newFunction(entry, values.get(), &error));
        }
        if (!function) throw std::runtime_error("Metal function " + name + ": "
            + (error ? error->localizedDescription()->utf8String() : "not found"));
        auto pipeline = NS::TransferPtr(device_.mtl_device()->newComputePipelineState(function.get(), &error));
        if (!pipeline) throw std::runtime_error("Metal pipeline " + name + ": "
            + (error ? error->localizedDescription()->utf8String() : "compilation failed"));
        return pipeline;
    });
}

std::string kernel_key(const mlx::core::Primitive& primitive) {
    if (const auto* native = dynamic_cast<const PreparableKernel*>(&primitive))
        return native->preparation_key();
    if (std::string_view(primitive.name()) != "CustomKernel")
        throw std::invalid_argument("Metal kernel does not support preparation");
    const auto state = static_cast<const mlx::core::fast::CustomKernel&>(primitive).state();
    std::ostringstream key;
    key << std::get<0>(state);
#if MLX_VERSION_NUMERIC >= 32001
    key << '_' << std::hex << std::hash<std::string>{}(std::get<1>(state))
        << '_' << std::dec << std::get<10>(state);
#endif
    return key.str();
}

void prepare_kernel(mlx::core::Primitive& primitive) {
    if (auto* native = dynamic_cast<PreparableKernel*>(&primitive)) {
        native->prepare_gpu();
        return;
    }
    const auto key = kernel_key(primitive);
    const auto state = static_cast<mlx::core::fast::CustomKernel&>(primitive).state();
    auto& device = mlx::core::metal::device(primitive.stream().device);
    auto* library = device.get_library(key, mlx::core::CompileOptions(std::get<10>(state)),
        [&] { return std::string(mfq::metal::detail::kMlxUtilsSource) + std::get<1>(state); });
    (void)device.get_kernel(std::get<0>(state), library);
}

namespace {
class RowMajorView final : public mlx::core::Primitive {
public:
    RowMajorView(const mlx::core::array& input, mlx::core::Stream stream)
        : Primitive(stream), copy_(mlx::core::contiguous(input, false, stream).primitive_ptr()) {}
    const char* name() const override { return "MlxRowMajorView"; }
    void eval_cpu(const std::vector<mlx::core::array>& inputs, std::vector<mlx::core::array>& outputs) override {
        if (inputs.front().flags().row_contiguous) outputs.front().copy_shared_buffer(inputs.front());
        else copy_->eval_cpu(inputs, outputs);
    }
    void eval_gpu(const std::vector<mlx::core::array>& inputs, std::vector<mlx::core::array>& outputs) override {
        if (inputs.front().flags().row_contiguous) outputs.front().copy_shared_buffer(inputs.front());
        else copy_->eval_gpu(inputs, outputs);
    }
private:
    std::shared_ptr<mlx::core::Primitive> copy_;
};
}

mlx::core::array row_contiguous_view(const mlx::core::array& input, mlx::core::Stream stream) {
    return mlx::core::array(input.shape(), input.dtype(),
        std::make_shared<RowMajorView>(input, stream), {input});
}

#if MLX_VERSION_NUMERIC >= 32001
namespace {
using namespace mlx::core;

using Arguments = std::vector<std::pair<std::string, fast::TemplateArg>>;
using Geometry = std::tuple<int, int, int>;

unsigned binding(const array& value) {
    return value.ndim() == 0 ? 0 : value.size() < 8 ? 1 : 2;
}

struct Description {
    Device device;
    std::vector<Dtype> inputs, outputs;
    std::vector<unsigned> bindings;
    Arguments arguments;
    std::string name, source, library_name;
    std::vector<std::tuple<bool, bool, bool>> metadata;
    bool contiguous;
    CompileOptions::Data options;
    std::mutex mutex;
    NS::SharedPtr<MTL::ComputePipelineState> owner;
    std::atomic<MTL::ComputePipelineState*> pipeline{nullptr};

    Description(Device device, const std::vector<array>& values,
        const std::vector<Dtype>& outputs, Arguments arguments,
        const fast::CustomKernel& prototype)
        : device(device), outputs(outputs), arguments(std::move(arguments)) {
        for (const auto& value : values) {
            inputs.push_back(value.dtype());
            bindings.push_back(binding(value));
        }
        auto state = prototype.state();
        name = std::move(std::get<0>(state));
        source = std::move(std::get<1>(state));
        metadata = std::move(std::get<4>(state));
        contiguous = std::get<5>(state);
        options = std::get<10>(state);
        std::ostringstream key;
        key << name << '_' << std::hex << std::hash<std::string>{}(source)
            << '_' << std::dec << options;
        library_name = key.str();
    }

    bool matches(Device target, const std::vector<array>& values,
        const std::vector<Dtype>& types, const Arguments& args) const {
        if (target != device || outputs != types || arguments != args) return false;
        for (std::size_t i = 0; i < values.size(); ++i)
            if (inputs[i] != values[i].dtype() || bindings[i] != binding(values[i])) return false;
        return true;
    }

    MTL::ComputePipelineState* prepare() {
        if (auto* ready = pipeline.load(std::memory_order_acquire)) return ready;
        std::lock_guard lock(mutex);
        if (auto* ready = pipeline.load(std::memory_order_relaxed)) return ready;
        auto& backend = mlx::core::metal::device(device);
        auto* library = backend.get_library(library_name, CompileOptions(options),
            [&] { return std::string(mfq::metal::detail::kMlxUtilsSource) + source; });
        owner = NS::RetainPtr(backend.get_kernel(name, library));
        pipeline.store(owner.get(), std::memory_order_release);
        return owner.get();
    }
};

class CachedKernel final : public Primitive, public PreparableKernel {
public:
    CachedKernel(Stream stream, std::shared_ptr<Description> description,
        Geometry grid, Geometry group, std::optional<float> initial)
        : Primitive(stream), description_(std::move(description)), grid_(grid),
            group_(group), initial_(initial) {
        if (initial_) initializer_ = full({1}, 0, float32, stream).primitive_ptr();
    }

    const char* name() const override { return "MlxCachedKernel"; }
    std::string preparation_key() const override {
        return description_->library_name + ':' + std::to_string(description_->device.index);
    }
    void prepare_gpu() override { (void)description_->prepare(); }
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("Metal kernels require a GPU stream");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto* pipeline = description_->prepare();
        std::vector<array> temporaries;
        for (auto& output : outputs) {
            if (initial_) {
                temporaries.emplace_back(*initial_, output.dtype());
                std::vector<array> filled{output};
                initializer_->eval_gpu({temporaries.back()}, filled);
                output = std::move(filled.front());
            } else output.set_data(allocator::malloc(output.nbytes()));
        }
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(pipeline);
        int slot = 0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto& value = inputs[i];
            encoder.set_input_array(value, slot++);
            if (value.ndim() > 0) {
                const auto [shape, strides, dimensions] = description_->metadata[i];
                if (shape) encoder.set_vector_bytes(value.shape(), value.ndim(), slot++);
                if (strides) encoder.set_vector_bytes(value.strides(), value.ndim(), slot++);
                if (dimensions) encoder.set_bytes(value.ndim(), slot++);
            }
        }
        for (auto& output : outputs) encoder.set_output_array(output, slot++);
        const auto [tx, ty, tz] = group_;
        if (std::size_t(tx) * ty * tz > pipeline->maxTotalThreadsPerThreadgroup())
            throw std::invalid_argument("Metal threadgroup exceeds this kernel's thread limit");
        const auto [gx, gy, gz] = grid_;
        encoder.dispatch_threads(MTL::Size(gx, gy, gz),
            MTL::Size(std::min(tx, gx), std::min(ty, gy), std::min(tz, gz)));
        encoder.add_temporaries(std::move(temporaries));
    }
private:
    std::shared_ptr<Description> description_;
    std::shared_ptr<Primitive> initializer_;
    Geometry grid_, group_;
    std::optional<float> initial_;
};

struct Factory {
    fast::CustomKernelFunction original;
    std::mutex mutex;
    std::vector<std::shared_ptr<Description>> descriptions;
};
}
#endif

mlx::core::fast::CustomKernelFunction custom_kernel(
    const std::string& name, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs, const std::string& source,
    const std::string& header, bool contiguous, bool atomic,
    const mlx::core::CompileOptions& options) {
    auto original = mlx::core::fast::metal_kernel(name, inputs, outputs, source,
        header, contiguous, atomic, options);
#if MLX_VERSION_NUMERIC < 32001
    return original;
#else
    auto factory = std::make_shared<Factory>();
    factory->original = std::move(original);
    return [factory, input_count = inputs.size(), output_count = outputs.size()](
        const std::vector<array>& values, const std::vector<Shape>& shapes,
        const std::vector<Dtype>& types, Geometry grid, Geometry group,
        Arguments arguments, std::optional<float> initial, bool verbose, StreamOrDevice target) {
        if (values.size() != input_count || shapes.size() != output_count || types.size() != output_count)
            throw std::invalid_argument("Metal kernel input/output counts do not match its signature");
        const auto stream = std::holds_alternative<std::monostate>(target)
            ? default_stream(Device::gpu) : to_stream(target);
        if (stream.device != Device::gpu)
            throw std::invalid_argument("Metal kernels require a GPU stream");
        std::shared_ptr<Description> description;
        {
            std::lock_guard lock(factory->mutex);
            for (const auto& item : factory->descriptions) {
                if (item->matches(stream.device, values, types, arguments)) {
                    description = item;
                    break;
                }
            }
            if (!description) {
                auto prototype = factory->original(values, shapes, types, grid, group,
                    arguments, initial, false, stream);
                description = std::make_shared<Description>(stream.device, values, types,
                    std::move(arguments), static_cast<const fast::CustomKernel&>(prototype.front().primitive()));
                factory->descriptions.push_back(description);
            }
        }
        if (verbose) std::cout << description->source << '\n';
        std::vector<array> checked = values;
        if (description->contiguous)
            for (auto& value : checked)
                if (!value.is_available() || !value.flags().row_contiguous)
                    value = row_contiguous_view(value, stream);
        return array::make_arrays(shapes, types,
            std::make_shared<CachedKernel>(stream, std::move(description), grid, group, initial), std::move(checked));
    };
#endif
}


} // namespace tyloi::metal
