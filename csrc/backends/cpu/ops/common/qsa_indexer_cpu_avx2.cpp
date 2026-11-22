#include <algorithm>
#include <cstdint>
#include <immintrin.h>

namespace mfq::cpu {
namespace {
float sum(__m256 input) {
    auto lanes = _mm_add_ps(_mm256_castps256_ps128(input), _mm256_extractf128_ps(input, 1));
    lanes = _mm_hadd_ps(lanes, lanes);
    return _mm_cvtss_f32(_mm_hadd_ps(lanes, lanes));
}
struct QueryDots {
    __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;

    void accumulate(__m256 key, __m256 qa, __m256 qb, __m256 qc, __m256 qd) {
        a = _mm256_fmadd_ps(key, qa, a);
        b = _mm256_fmadd_ps(key, qb, b);
        c = _mm256_fmadd_ps(key, qc, c);
        d = _mm256_fmadd_ps(key, qd, d);
    }

    float score(const float* sums, int zero, float scale) const {
        const auto term = [&](int h, __m256 acc) {
            return std::max((sum(acc) - float(zero) * sums[h]) * scale, 0.f);
        };
        return term(0, a) + term(1, b) + term(2, c) + term(3, d);
    }
};

__m256 unpack(const std::int16_t* codes) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(codes))));
}

template<bool Pair>
void dot_rows(const float* query, const float* sums, const std::int16_t* codes, int width,
              int zero, float scale, int next_zero, float next_scale, float* scores) {
    QueryDots first, second;
    for (int j = 0; j < width; j += 8) {
        const auto qa = _mm256_loadu_ps(query + j);
        const auto qb = _mm256_loadu_ps(query + width + j);
        const auto qc = _mm256_loadu_ps(query + 2 * width + j);
        const auto qd = _mm256_loadu_ps(query + 3 * width + j);
        first.accumulate(unpack(codes + j), qa, qb, qc, qd);
        if constexpr (Pair) second.accumulate(unpack(codes + width + j), qa, qb, qc, qd);
    }
    scores[0] = first.score(sums, zero, scale);
    if constexpr (Pair) scores[1] = second.score(sums, next_zero, next_scale);
}
} // namespace

float qsa_affine16_score_avx2(const float* query, const float* sums, int, const std::int16_t* codes,
                             int width, int zero, float scale) {
    float score;
    dot_rows<false>(query, sums, codes, width, zero, scale, 0, 0.f, &score);
    return score;
}

void qsa_affine16_score_pair_avx2(const float* query, const float* sums, const std::int16_t* codes, int width,
                                 int zero, float scale, int next_zero, float next_scale, float* scores) {
    dot_rows<true>(query, sums, codes, width, zero, scale, next_zero, next_scale, scores);
}

float index_pool_affine16_score_avx2(const float* query,const float* sums,const float* weights,int heads,
    const std::int16_t* codes,int width,int zero,float scale) {
    float score=0;
    for (int head=0;head<heads;head+=4) {
        __m256 accumulators[4]{_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps()};
        const int count=std::min(4,heads-head);
        for (int col=0;col<width;col+=8) {
            const auto key=unpack(codes+col);
            for (int i=0;i<count;++i)
                accumulators[i]=_mm256_fmadd_ps(key,_mm256_loadu_ps(query+(head+i)*width+col),accumulators[i]);
        }
        for (int i=0;i<count;++i) score+=weights[head+i]*std::max(
            (sum(accumulators[i])-float(zero)*sums[head+i])*scale,0.f);
    }
    return score;
}

} // namespace mfq::cpu
