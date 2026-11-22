#pragma once

#include "mfq_cuda_context.h"
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace mfq::cuda {

constexpr bool device_has_separate_memory(bool integrated, bool host_page_tables) noexcept {
    // Host page-table access identifies a shared CPU/GPU memory system; UVA
    // alone does not imply shared physical memory.
    return !integrated && !host_page_tables;
}

inline std::optional<bool> separate_memory_override(const char* value) {
    if (!value || !*value) return std::nullopt;
    const std::string_view setting(value);
    if (setting == "unknown") return std::nullopt;
    if (setting == "discrete") return true;
    if (setting == "unified") return false;
    throw std::invalid_argument("invalid TYLOINFER_MEMORY_ARCHITECTURE: expected discrete, unified or unknown");
}

inline bool device_has_separate_memory(int device) {
    if (const auto override = separate_memory_override(std::getenv("TYLOINFER_MEMORY_ARCHITECTURE")))
        return *override;
    int integrated = 0, host_page_tables = 0;
    TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device));
    TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(
        &host_page_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, device));
    return device_has_separate_memory(integrated != 0, host_page_tables != 0);
}

} // namespace mfq::cuda
