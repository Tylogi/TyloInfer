#include "device.h"
#include "memory.h"

#include <mlx/mlx.h>
#include <mach-o/dyld.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

namespace tyloi::metal {
namespace {
std::filesystem::path executable_path() {
    std::uint32_t size = 0;
    (void)_NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size + 1, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        throw std::runtime_error("failed to resolve executable path");
    }
    std::error_code error;
    const auto resolved =
        std::filesystem::weakly_canonical(buffer.data(), error);
    return error ? std::filesystem::path(buffer.data()) : resolved;
}
}

void initialize() {
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        if (std::getenv("MLX_MAX_MB_PER_BUFFER") == nullptr) {
            ::setenv("MLX_MAX_MB_PER_BUFFER", "512", 0);
        }
        std::vector<std::filesystem::path> candidates;
        if (const char* configured = std::getenv("TYLOINFER_MLX_METALLIB")) {
            if (*configured != '\0') {
                candidates.emplace_back(configured);
            }
        }
        candidates.push_back(
            executable_path().parent_path() / "lib" / "mlx.metallib");
#ifdef TYLOINFER_MLX_METALLIB_DEFAULT
        candidates.emplace_back(TYLOINFER_MLX_METALLIB_DEFAULT);
#endif
        for (const auto& candidate : candidates) {
            std::error_code error;
            if (std::filesystem::is_regular_file(candidate, error) && !error) {
                mlx::core::metal::set_metallib_path(candidate.string());
                mlx::core::set_default_device(mlx::core::Device::gpu);
                // Set residency before loading weights: otherwise macOS can
                // compress early layers while later layers are being loaded.
                // This is a ceiling, not an allocation or a system-wide change.
                try {
                    const auto& info = mlx::core::device_info();
                    const auto limit = std::get<std::size_t>(
                        info.at("max_recommended_working_set_size"));
                    if (limit == 0) {
                        throw std::runtime_error("device returned a zero limit");
                    }
                    tyloi::metal::set_wired_limit(limit);
                    std::cerr << "Metal wired memory limit: " << limit
                              << " bytes (device recommendation)\n";
                } catch (const std::exception& error) {
                    if (__builtin_available(macOS 15, *)) {
                        throw;
                    }
                    std::cerr << "Warning: Metal memory residency unavailable: "
                              << error.what() << "\n";
                }
                return;
            }
        }
        throw std::runtime_error(
            "cannot locate mlx.metallib; set TYLOINFER_MLX_METALLIB or keep "
            "lib/mlx.metallib beside the runtime executable");
    });
}

int metal_apple_gpu_core_count() noexcept {
    static const int cores = [] {
        const auto service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"));
        if (!service) return 0;
        const auto value = IORegistryEntryCreateCFProperty(service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        IOObjectRelease(service);
        if (!value) return 0;
        int count = 0;
        if (CFGetTypeID(value) == CFNumberGetTypeID())
            CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberIntType, &count);
        CFRelease(value);
        return count > 0 && count <= 1024 ? count : 0;
    }();
    return cores;
}
}
