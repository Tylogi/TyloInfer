#include "qsa_indexer_cpu.h"
#include "gguf_cpu.h"
#include "tyloinfer/host_parallel.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace mfq::cpu {
#if defined(TYLOINFER_GGUF_CPU_AVX2)
float qsa_affine16_score_avx2(const float*, const float*, int, const std::int16_t*, int, int, float);
void qsa_affine16_score_pair_avx2(const float*, const float*, const std::int16_t*, int, int, float, int, float, float*);
#endif
namespace {
struct Candidate { float score; int id; };

bool better(const Candidate& a, const Candidate& b) {
    return a.score != b.score ? a.score > b.score : a.id < b.id;
}

Candidate select(std::vector<Candidate>& values, int count) {
    if (values.size() <= std::size_t(count))
        return {-std::numeric_limits<float>::infinity(), std::numeric_limits<int>::max()};
    auto boundary = values.begin() + count - 1;
    std::nth_element(values.begin(), boundary, values.end(), better);
    auto limit = *boundary;
    values.resize(count);
    return limit;
}

float scale_value(std::uint16_t bits) {
    const auto exponent = (bits >> 10) & 31;
    const auto mantissa = bits & 1023;
    if (!exponent) return float(mantissa) * (1.f / 16777216.f);
    const std::uint32_t fp32 = (std::uint32_t(exponent == 31 ? 255 : exponent + 112) << 23) | (std::uint32_t(mantissa) << 13);
    float result;
    std::memcpy(&result, &fp32, sizeof(result));
    return result;
}

float score_scalar(const float* query, const float* sums, int heads, const std::int16_t* code,
                   int width, int zero, float scale) {
    float score = 0;
    for (int h = 0; h < heads; ++h) {
        float dot = 0;
        for (int k = 0; k < width; ++k) dot += float(code[k]) * query[h * width + k];
        score += std::max((dot - float(zero) * sums[h]) * scale, 0.f);
    }
    return score;
}
} // namespace

void qsa_indexer_topk(const float* queries, const std::int64_t* positions, int query_count, int heads,
                      int width, const std::int16_t* codes, const std::uint16_t* scales,
                      const std::int16_t* zero, int pools, int budget, int threads, std::int32_t* selected) {
    const int top = budget / 4, slots = top + 1;
    std::vector<float> sums(std::size_t(query_count) * heads);
    std::vector<int> visible(query_count);
    int longest = 0;
    for (int q = 0; q < query_count; ++q) {
        const auto live = std::max<std::int64_t>(0, positions[q] + 1);
        auto* out = selected + std::size_t(q) * slots * 2;
        for (int slot = 0; slot < slots; ++slot) { out[2 * slot] = -1; out[2 * slot + 1] = 0; }
        if (live <= budget) {
            for (int slot = 0; slot * 4 < live; ++slot) {
                out[slot * 2] = slot;
                out[slot * 2 + 1] = (1 << std::min<std::int64_t>(4, live - slot * 4)) - 1;
            }
            continue;
        }
        visible[q] = int(std::min<std::int64_t>(pools, live / 4));
        longest = std::max(longest, visible[q]);
        if (live % 4) { out[top * 2] = int(live / 4); out[top * 2 + 1] = (1 << (live % 4)) - 1; }
        for (int h = 0; h < heads; ++h) {
            const auto* row = queries + (std::size_t(q) * heads + h) * width;
            sums[std::size_t(q) * heads + h] = std::accumulate(row, row + width, 0.f);
        }
    }
    if (!longest) return;
    const int workers = std::max(1, std::min(threads, 1 + (longest - 1) / 1024));
    std::vector<std::vector<Candidate>> local(std::size_t(workers) * query_count);
    auto score = score_scalar;
#if defined(TYLOINFER_GGUF_CPU_AVX2)
    const bool use_avx2 = gguf::available() && heads == 4 && width % 8 == 0;
    if (use_avx2) score = qsa_affine16_score_avx2;
#endif
    mfq::host_parallel_for(0, workers, 1, workers, [&](std::int64_t begin, std::int64_t end) {
        std::vector<Candidate> cutoffs(query_count, {-std::numeric_limits<float>::infinity(), std::numeric_limits<int>::max()});
        for (auto worker = begin; worker < end; ++worker) {
            for (int q = 0; q < query_count; ++q) local[worker * query_count + q].reserve(std::size_t(top) * 2);
            std::fill(cutoffs.begin(), cutoffs.end(), Candidate{-std::numeric_limits<float>::infinity(), std::numeric_limits<int>::max()});
            const auto consider = [&](int q, Candidate candidate) {
                if (!better(candidate, cutoffs[q])) return;
                auto& list = local[worker * query_count + q];
                list.push_back(candidate);
                if (list.size() == std::size_t(top) * 2) cutoffs[q] = select(list, top);
            };
            auto id = int(longest * worker / workers);
            const int last = int(longest * (worker + 1) / workers);
#if defined(TYLOINFER_GGUF_CPU_AVX2)
            if (use_avx2 && query_count == 1) {
                for (; id + 1 < last; id += 2) {
                    float scores[2];
                    qsa_affine16_score_pair_avx2(queries, sums.data(), codes + std::size_t(id) * width, width,
                        zero[id], scale_value(scales[id]), zero[id + 1], scale_value(scales[id + 1]), scores);
                    consider(0, {scores[0], id});
                    consider(0, {scores[1], id + 1});
                }
            }
#endif
            for (; id < last; ++id) {
                const float scale = scale_value(scales[id]);
                const auto* row = codes + std::size_t(id) * width;
                for (int q = 0; q < query_count; ++q) {
                    if (id >= visible[q]) continue;
                    const auto query_head = std::size_t(q) * heads;
                    consider(q, {score(queries + query_head * width, sums.data() + query_head,
                        heads, row, width, zero[id], scale), id});
                }
            }
            for (int q = 0; q < query_count; ++q) select(local[worker * query_count + q], top);
        }
    });
    std::vector<Candidate> merged;
    for (int q = 0; q < query_count; ++q) {
        if (!visible[q]) continue;
        merged.clear();
        for (int w = 0; w < workers; ++w) {
            const auto& part = local[w * query_count + q];
            merged.insert(merged.end(), part.begin(), part.end());
        }
        select(merged, top);
        std::sort(merged.begin(), merged.end(), better);
        auto* out = selected + std::size_t(q) * slots * 2;
        for (std::size_t i = 0; i < merged.size(); ++i) { out[2 * i] = merged[i].id; out[2 * i + 1] = 15; }
    }
}

} // namespace mfq::cpu
