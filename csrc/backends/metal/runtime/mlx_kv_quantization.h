#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <string>

namespace mfq::metal {

struct MlxKvQuantization {
    double bits = 0;
    bool enabled() const noexcept { return bits != 0; }
    int key_bits() const noexcept;
    int value_bits() const noexcept;
    bool operator==(const MlxKvQuantization&) const = default;
};

MlxKvQuantization mlx_kv_quantization();
bool mlx_kv_bits_valid(double bits);
int mlx_kv_packed_width(int dimension, int bits);
std::string mlx_kv_quantization_tag();
mlx::core::array mlx_kv_encode(const mlx::core::array& input, int bits, bool value);
mlx::core::array mlx_kv_decode(const mlx::core::array& input, int dimension, int bits,
    bool value, mlx::core::Dtype dtype = mlx::core::float16);

// In MLA the low-rank value is also the non-rotary part of each key.
// Store that shared vector once and encode the rotary coordinates separately.
struct MlxMlaKvFormat {
    int latent_dimension;
    int rotary_dimension;
    MlxKvQuantization quantization;

    int width() const;
    mlx::core::Dtype dtype() const noexcept;
    mlx::core::array pack(const mlx::core::array& rows) const;
    mlx::core::array unpack(const mlx::core::array& rows) const;
};

mlx::core::array mlx_kv_sparse_attention(const mlx::core::array& query,
    const mlx::core::array& key, const mlx::core::array& value,
    const std::optional<mlx::core::array>& blocks, MlxKvQuantization quantization,
    int position, int query_offset, int ratio, int budget,
    const std::optional<mlx::core::array>& key_mask = std::nullopt);

}
