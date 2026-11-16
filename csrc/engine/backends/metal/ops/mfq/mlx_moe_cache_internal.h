#pragma once
#include "mlx_moe.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace tyloi::metal::mfe_internal {
inline std::atomic<std::size_t> pooled_expert_staging{0};
inline constexpr int kDescriptorSize = 32;
inline constexpr unsigned kMfeSmallMReadWorkers = 16;
inline constexpr std::size_t kMfeDirectRangeMaxExperts = 2;
inline constexpr int kFamilyNint = 0;
inline constexpr int kFamilyMxfp4 = 3;
inline constexpr int kFamilyMxfp8 = 4;
inline constexpr int kFamilyVq = 1;
inline constexpr int kVqProfileGeneric = 0;
inline constexpr int kVqProfileJsc4 = 1;
inline constexpr int kVqProfileNpqS = 2;
inline constexpr int kVqProfileNvq1L = 3;
inline constexpr int kVqProfileJsc8 = 4;
inline constexpr int kVqProfileNpqL = 5;
inline constexpr int kVqProfileNvq1S = 6;
inline constexpr int kVqProfileJscExtended8 = 7;
inline constexpr int kVqProfileJscExtended4 = 8;

struct MfeDirectReadTarget {
    std::uint64_t offset;
    std::uint8_t* wire;
    std::size_t bytes;
    std::uint8_t* destination = nullptr;
};
inline thread_local std::vector<MfeDirectReadTarget>* direct_read_targets = nullptr;
enum class MfeMetadataRead { Planned, Eager, DeferredMxScales };
inline thread_local MfeMetadataRead direct_read_metadata = MfeMetadataRead::Planned;
struct MfeDirectReadScope {
    explicit MfeDirectReadScope(std::vector<MfeDirectReadTarget>& targets)
        : previous(std::exchange(direct_read_targets, &targets)) {}
    ~MfeDirectReadScope() { direct_read_targets = previous; }
    std::vector<MfeDirectReadTarget>* previous;
};

std::size_t checked_product(std::size_t, std::size_t, const char*);
std::size_t checked_add(std::size_t, std::size_t, const char*);
std::size_t checked_packed_size(std::size_t, int, const char*);
std::int32_t checked_int(std::size_t, const char*);
mlx::core::array allocate_packed_array(std::size_t, mlx::core::Dtype);
mlx::core::array allocate_wire_array(std::size_t, bool account = false);
bool uses_indirect_expert_pages(std::uint32_t);
void prepare_packed_copy();
void collect_packed_copy();
}
