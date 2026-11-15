#include "mlx_sampling.h"

#include "engine/backends/metal/native/stream.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tyloi::metal {
namespace {
constexpr int kMaxTopK = 1024;
using mlx::core::Shape;
using mlx::core::array;
int checked_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string("sampling ") + name + " exceeds MLX limits");
    return static_cast<int>(value);
}

int sampling_rows(const array& logits) {
    if (logits.ndim() < 1 || logits.shape(-1) <= 0)
        throw std::invalid_argument("sampling logits must end in a non-empty vocabulary");
    const auto rows = logits.size() / static_cast<std::size_t>(logits.shape(-1));
    if (!rows) throw std::invalid_argument("sampling logits cannot have zero rows");
    return checked_int(rows, "row count");
}

array make_cpu_random(std::mt19937_64& rng, int rows) {
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::vector<float> values(static_cast<std::size_t>(rows));
    for (auto& value : values) value = uniform(rng);
    return array(values.begin(), Shape{rows}, mlx::core::float32);
}
} // namespace

array mlx_last_token_logits(
    const array& logits,
    int vocab) {
    if (vocab <= 0 || logits.ndim() != 3 ||
        logits.shape(0) <= 0 || logits.shape(1) <= 0 ||
        logits.shape(2) != vocab) {
        throw std::runtime_error(
            "generation logits must have [batch,tokens,vocab] shape");
    }
    return mlx::core::reshape(
        mlx::core::slice(
            logits,
            Shape{0, logits.shape(1) - 1, 0},
            Shape{logits.shape(0), logits.shape(1), vocab}),
        Shape{logits.shape(0), vocab});
}

void mlx_validate_token_set(
    std::span<const std::int64_t> token_ids,
    int vocab) {
    if (vocab <= 0) {
        throw std::invalid_argument("vocabulary size must be positive");
    }
    for (const auto token : token_ids) {
        if (token < 0 || token >= vocab) {
            throw std::invalid_argument("token set contains an invalid ID");
        }
    }
}

bool mlx_token_in_set(
    std::span<const std::int64_t> token_ids,
    std::int64_t token) noexcept {
    return std::find(token_ids.begin(), token_ids.end(), token) !=
        token_ids.end();
}

std::vector<float> host_sampling_distribution(
    const array& logits,
    const MlxSamplingParams& params) {
    if (params.greedy()) {
        throw std::invalid_argument(
            "host sampling distribution requires stochastic sampling");
    }
    if (!std::isfinite(params.temperature) || params.temperature <= 0.0 ||
        !std::isfinite(params.top_p) || params.top_p <= 0.0 ||
        params.top_p > 1.0 || params.top_k < 0 || params.top_k > kMaxTopK) {
        throw std::invalid_argument(
            "invalid host sampling distribution parameters");
    }
    auto values = mlx::core::contiguous(mlx::core::astype(
        mlx::core::reshape(logits, Shape{checked_int(logits.size(), "logits size")}),
        mlx::core::float32));
    tyloi::metal::eval(values);
    const int vocab = static_cast<int>(values.size());
    if (vocab <= 0 || (params.top_k > 0 && params.top_k > vocab)) {
        throw std::invalid_argument(
            "host sampling distribution has invalid vocabulary geometry");
    }
    const auto* source = values.data<float>();
    std::vector<int> order(static_cast<std::size_t>(vocab));
    for (int token = 0; token < vocab; ++token) order[token] = token;
    const auto before = [source](int left, int right) {
        const float lhs = std::isnan(source[left])
            ? -std::numeric_limits<float>::infinity() : source[left];
        const float rhs = std::isnan(source[right])
            ? -std::numeric_limits<float>::infinity() : source[right];
        return lhs > rhs || (lhs == rhs && left < right);
    };
    const int selected = params.top_k > 0 ? params.top_k : vocab;
    const bool needs_order = params.top_k > 0 || params.top_p < 1.0;
    if (needs_order) {
        if (selected < vocab) {
            std::partial_sort(
                order.begin(), order.begin() + selected, order.end(), before);
            order.resize(static_cast<std::size_t>(selected));
        } else {
            std::sort(order.begin(), order.end(), before);
        }
    }

    float maximum = -std::numeric_limits<float>::infinity();
    if (needs_order) {
        maximum = source[order.front()];
    } else {
        for (int token = 0; token < vocab; ++token) {
            const float value = std::isnan(source[token])
                ? -std::numeric_limits<float>::infinity() : source[token];
            maximum = std::max(maximum, value);
        }
    }
    if (!std::isfinite(maximum)) {
        throw std::runtime_error("sampling logits contain no finite value");
    }

    std::vector<float> probabilities(static_cast<std::size_t>(vocab), 0.0f);
    std::vector<double> masses(static_cast<std::size_t>(selected));
    double total = 0.0;
    for (int index = 0; index < selected; ++index) {
        const int token = needs_order ? order[static_cast<std::size_t>(index)] : index;
        const float value = std::isnan(source[token])
            ? -std::numeric_limits<float>::infinity() : source[token];
        const double mass = std::exp(
            (static_cast<double>(value) - maximum) / params.temperature);
        masses[static_cast<std::size_t>(index)] = mass;
        total += mass;
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error("sampling distribution is not normalizable");
    }
    int keep = selected;
    double keep_mass = total;
    if (params.top_p < 1.0) {
        const double cutoff = params.top_p * total;
        keep_mass = 0.0;
        for (int index = 0; index < selected; ++index) {
            keep_mass += masses[static_cast<std::size_t>(index)];
            if (keep_mass >= cutoff) {
                keep = index + 1;
                break;
            }
        }
    }
    for (int index = 0; index < keep; ++index) {
        const int token = needs_order ? order[static_cast<std::size_t>(index)] : index;
        probabilities[static_cast<std::size_t>(token)] = static_cast<float>(
            masses[static_cast<std::size_t>(index)] / keep_mass);
    }
    return probabilities;
}

std::int32_t sample_host_distribution(
    const std::vector<float>& probabilities,
    double uniform) {
    if (probabilities.empty() || !std::isfinite(uniform)) {
        throw std::invalid_argument("invalid host sampling input");
    }
    uniform = std::clamp(uniform, 0.0, std::nextafter(1.0, 0.0));
    double cumulative = 0.0;
    std::int32_t fallback = -1;
    for (std::size_t token = 0; token < probabilities.size(); ++token) {
        const float probability = probabilities[token];
        if (!std::isfinite(probability) || probability < 0.0f) {
            throw std::invalid_argument("host sampling probability is invalid");
        }
        if (probability <= 0.0f) continue;
        fallback = static_cast<std::int32_t>(token);
        cumulative += probability;
        if (uniform < cumulative) return fallback;
    }
    if (fallback < 0) {
        throw std::invalid_argument("host sampling distribution is empty");
    }
    return fallback;
}

array sample(const array& logits, const MlxSamplingParams& params,
        const std::optional<array>& random) {
    if (params.top_k < 0) throw std::invalid_argument("top_k must be non-negative");
    if (params.greedy()) return sample_greedy(logits);
    if (!std::isfinite(params.top_p) || params.top_p <= 0.0 || params.top_p > 1.0)
        throw std::invalid_argument("top_p must be in (0,1]");
    const auto rows = sampling_rows(logits);
    std::mt19937_64 rng(params.seed);
    auto uniforms = random ? *random : make_cpu_random(rng, rows);
    if (params.top_k > 0)
        return sample_top_k_top_p(logits, uniforms, params.temperature, params.top_k, params.top_p);
    if (params.top_p >= 1.0) return sample_softmax(logits, uniforms, params.temperature);
    return sample_nucleus(logits, uniforms, params.temperature, params.top_p);
}

MlxSampler::MlxSampler(MlxSamplingParams params)
    : params_(std::move(params)),
      rng_(params_.seed) {
    if (params_.top_k < 0) {
        throw std::invalid_argument("top_k must be non-negative");
    }
}

void MlxSampler::reset_seed(std::uint64_t seed) {
    params_.seed = seed;
    rng_.seed(seed);
}

void MlxSampler::discard_random(std::uint64_t draws) {
    rng_.discard(draws);
}

array MlxSampler::next_random(const array& logits) {
    return make_cpu_random(rng_, sampling_rows(logits));
}

array MlxSampler::sample(const array& logits) {
    if (params_.greedy()) {
        return tyloi::metal::sample(logits, params_);
    }
    return tyloi::metal::sample(
        logits,
        params_,
        next_random(logits));
}

array MlxSampler::sample(
    const array& logits,
    const array& counts) {
    if (!params_.has_penalties()) {
        return sample(logits);
    }
    return sample(apply_penalties(logits, counts));
}

array MlxSampler::apply_penalties(
    const array& logits,
    const array& counts) const {
    return sample_apply_penalties(
        logits,
        counts,
        params_.presence_penalty,
        params_.frequency_penalty,
        params_.repetition_penalty);
}

double MlxSampler::next_uniform() {
    return std::generate_canonical<double, 53>(rng_);
}

} // namespace tyloi::metal
