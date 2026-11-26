/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2026 The LCZero Authors

 Leela Chess is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Leela Chess is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
 */

// Compiled with AVX2/AVX-VNNI flags; includes only POD headers (no STL) so no
// shared COMDAT functions can acquire VEX instructions.

#include <immintrin.h>

#include <cstring>

#include "neural/backends/blas/int8_vnni_kernels.h"

namespace lczero {
namespace blas_backend {
namespace {

constexpr size_t kSquares = 64;
constexpr size_t kBoardSize = 8;
constexpr size_t kPadSize = 10;
constexpr size_t kNr = 16;

inline size_t MinSize(size_t a, size_t b) { return a < b ? a : b; }

inline float ClampFloat(float value, float low, float high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

inline int32_t LoadU32(const uint8_t* ptr) {
  int32_t value;
  std::memcpy(&value, ptr, sizeof(value));
  return value;
}

inline int32_t LoadTailU32(const uint8_t* ptr, size_t remaining) {
  uint32_t value = 0;
  for (size_t i = 0; i < remaining; ++i) {
    value |= static_cast<uint32_t>(ptr[i]) << (8 * i);
  }
  return static_cast<int32_t>(value);
}

// VEX AVX-VNNI vpdpbusd (safe on non-AVX512 CPUs such as Alder/Raptor Lake).
inline __m256i DpbusdAvx(__m256i acc, __m256i a_u8, __m256i b_s8) {
#if defined(_MSC_VER) || defined(__clang__) || defined(__GNUC__)
  return _mm256_dpbusd_avx_epi32(acc, a_u8, b_s8);
#else
  return _mm256_dpbusd_epi32(acc, a_u8, b_s8);
#endif
}

// Fast AVX2 exp(x), accurate to ~1-2 ULP across [-87, 88].
inline __m256 ExpPsAvx(__m256 x) {
  const __m256 min_x = _mm256_set1_ps(-87.33654475f);
  const __m256 max_x = _mm256_set1_ps(88.37626266f);
  x = _mm256_max_ps(_mm256_min_ps(x, max_x), min_x);

  const __m256 log2_e = _mm256_set1_ps(1.44269504088896341f);
  const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
  const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);

  const __m256 k = _mm256_round_ps(
      _mm256_mul_ps(x, log2_e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(k, ln2_hi, x);
  r = _mm256_fnmadd_ps(k, ln2_lo, r);

  __m256 poly = _mm256_set1_ps(1.38888889e-3f);
  poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(8.33333333e-3f));
  poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(4.16666667e-2f));
  poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(1.66666667e-1f));
  poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(0.5f));
  poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(1.0f));
  poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(1.0f));

  const __m256i ki = _mm256_cvtps_epi32(k);
  const __m256i exp_bits =
      _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(poly, _mm256_castsi256_ps(exp_bits));
}

inline __m256 MishPsAvx(__m256 x) {
  const __m256 cap = _mm256_set1_ps(20.0f);
  const __m256 two = _mm256_set1_ps(2.0f);
  const __m256 threshold = _mm256_set1_ps(-0.125f);
  const __m256 large_mask = _mm256_cmp_ps(x, cap, _CMP_GT_OQ);
  const __m256 clamped = _mm256_min_ps(x, cap);
  const __m256 exp_val = ExpPsAvx(clamped);
  const __m256 num =
      _mm256_fmadd_ps(exp_val, exp_val, _mm256_add_ps(exp_val, exp_val));
  const __m256 denom = _mm256_div_ps(clamped, _mm256_add_ps(num, two));
  const __m256 neg_mask = _mm256_cmp_ps(clamped, threshold, _CMP_LE_OQ);
  const __m256 pos_branch = _mm256_fnmadd_ps(two, denom, clamped);
  const __m256 neg_branch = _mm256_mul_ps(num, denom);
  const __m256 result = _mm256_blendv_ps(pos_branch, neg_branch, neg_mask);
  return _mm256_blendv_ps(result, x, large_mask);
}

inline __m256 SigmoidPsAvx(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
  return _mm256_div_ps(one, _mm256_add_ps(one, ExpPsAvx(neg_x)));
}

inline __m256 ApplyActivationAvx(__m256 values, ActivationFunction activation) {
  switch (activation) {
    case ACTIVATION_NONE:
      return values;
    case ACTIVATION_RELU:
      return _mm256_max_ps(values, _mm256_setzero_ps());
    case ACTIVATION_RELU_2: {
      __m256 relu = _mm256_max_ps(values, _mm256_setzero_ps());
      return _mm256_mul_ps(relu, relu);
    }
    case ACTIVATION_MISH:
      return MishPsAvx(values);
    case ACTIVATION_SWISH: {
      const __m256 one = _mm256_set1_ps(1.0f);
      return _mm256_div_ps(
          values, _mm256_add_ps(one, ExpPsAvx(_mm256_sub_ps(_mm256_setzero_ps(),
                                                            values))));
    }
    default: {
      alignas(32) float tmp[8];
      _mm256_store_ps(tmp, values);
      for (int i = 0; i < 8; ++i) tmp[i] = Activate(tmp[i], activation);
      return _mm256_load_ps(tmp);
    }
  }
}

inline void StoreQuantized16(__m256 v0, __m256 v1, __m256 inverse_scale,
                             __m256 zero_point, uint8_t* destination) {
  const __m256 zero = _mm256_setzero_ps();
  const __m256 max_val = _mm256_set1_ps(255.0f);
  v0 = _mm256_min_ps(
      _mm256_max_ps(_mm256_fmadd_ps(v0, inverse_scale, zero_point), zero),
      max_val);
  v1 = _mm256_min_ps(
      _mm256_max_ps(_mm256_fmadd_ps(v1, inverse_scale, zero_point), zero),
      max_val);
  const __m256i i0 = _mm256_cvtps_epi32(
      _mm256_round_ps(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  const __m256i i1 = _mm256_cvtps_epi32(
      _mm256_round_ps(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  const __m256i packed16 =
      _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);
  const __m128i lo = _mm256_castsi256_si128(packed16);
  const __m128i hi = _mm256_extracti128_si256(packed16, 1);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(destination),
                   _mm_packus_epi16(lo, hi));
}

inline void Microkernel4x16Accum(const uint8_t* input_base, size_t row_stride,
                                 const int8_t* weights_ptr, size_t k_full,
                                 size_t k_rem, __m256i acc[4][2]) {
  const uint8_t* a0 = input_base + 0 * row_stride;
  const uint8_t* a1 = input_base + 1 * row_stride;
  const uint8_t* a2 = input_base + 2 * row_stride;
  const uint8_t* a3 = input_base + 3 * row_stride;

  __m256i c00 = acc[0][0], c01 = acc[0][1];
  __m256i c10 = acc[1][0], c11 = acc[1][1];
  __m256i c20 = acc[2][0], c21 = acc[2][1];
  __m256i c30 = acc[3][0], c31 = acc[3][1];

  for (size_t k_group = 0; k_group < k_full; ++k_group) {
    const __m256i b0 =
        _mm256_load_si256(reinterpret_cast<const __m256i*>(weights_ptr));
    const __m256i b1 =
        _mm256_load_si256(reinterpret_cast<const __m256i*>(weights_ptr + 32));
    const size_t offset = k_group * 4;
    __m256i av;
    av = _mm256_set1_epi32(LoadU32(a0 + offset));
    c00 = DpbusdAvx(c00, av, b0);
    c01 = DpbusdAvx(c01, av, b1);
    av = _mm256_set1_epi32(LoadU32(a1 + offset));
    c10 = DpbusdAvx(c10, av, b0);
    c11 = DpbusdAvx(c11, av, b1);
    av = _mm256_set1_epi32(LoadU32(a2 + offset));
    c20 = DpbusdAvx(c20, av, b0);
    c21 = DpbusdAvx(c21, av, b1);
    av = _mm256_set1_epi32(LoadU32(a3 + offset));
    c30 = DpbusdAvx(c30, av, b0);
    c31 = DpbusdAvx(c31, av, b1);
    weights_ptr += 64;
  }
  if (k_rem > 0) {
    const __m256i b0 =
        _mm256_load_si256(reinterpret_cast<const __m256i*>(weights_ptr));
    const __m256i b1 =
        _mm256_load_si256(reinterpret_cast<const __m256i*>(weights_ptr + 32));
    const size_t offset = k_full * 4;
    __m256i av;
    av = _mm256_set1_epi32(LoadTailU32(a0 + offset, k_rem));
    c00 = DpbusdAvx(c00, av, b0);
    c01 = DpbusdAvx(c01, av, b1);
    av = _mm256_set1_epi32(LoadTailU32(a1 + offset, k_rem));
    c10 = DpbusdAvx(c10, av, b0);
    c11 = DpbusdAvx(c11, av, b1);
    av = _mm256_set1_epi32(LoadTailU32(a2 + offset, k_rem));
    c20 = DpbusdAvx(c20, av, b0);
    c21 = DpbusdAvx(c21, av, b1);
    av = _mm256_set1_epi32(LoadTailU32(a3 + offset, k_rem));
    c30 = DpbusdAvx(c30, av, b0);
    c31 = DpbusdAvx(c31, av, b1);
  }

  acc[0][0] = c00;
  acc[0][1] = c01;
  acc[1][0] = c10;
  acc[1][1] = c11;
  acc[2][0] = c20;
  acc[2][1] = c21;
  acc[3][0] = c30;
  acc[3][1] = c31;
}

inline void FillPadBorderAvx(uint8_t* pad, size_t padded_channels,
                             uint8_t zero_point) {
  std::memset(pad, zero_point, kPadSize * padded_channels);
  std::memset(pad + (kPadSize - 1) * kPadSize * padded_channels, zero_point,
              kPadSize * padded_channels);
  for (size_t y = 1; y <= kBoardSize; ++y) {
    uint8_t* row = pad + y * kPadSize * padded_channels;
    std::memset(row, zero_point, padded_channels);
    std::memset(row + (kPadSize - 1) * padded_channels, zero_point,
                padded_channels);
  }
}

inline void QuantizeBoardToPadAvx(size_t channels, size_t padded_channels,
                                  const float* input_nhwc, float inverse_scale,
                                  uint8_t zero_point, uint8_t* pad) {
  FillPadBorderAvx(pad, padded_channels, zero_point);
  const float zero_point_f = static_cast<float>(zero_point);
  if (padded_channels == channels) {
    for (size_t y = 0; y < kBoardSize; ++y) {
      QuantizeToUint8Avx(kBoardSize * channels,
                         input_nhwc + y * kBoardSize * channels, inverse_scale,
                         zero_point_f,
                         pad + ((y + 1) * kPadSize + 1) * padded_channels);
    }
  } else {
    for (size_t y = 0; y < kBoardSize; ++y) {
      for (size_t x = 0; x < kBoardSize; ++x) {
        uint8_t* destination =
            pad + ((y + 1) * kPadSize + (x + 1)) * padded_channels;
        QuantizeToUint8Avx(channels,
                           input_nhwc + (y * kBoardSize + x) * channels,
                           inverse_scale, zero_point_f, destination);
        std::memset(destination + channels, 0, padded_channels - channels);
      }
    }
  }
}

inline float DotProductAvx(const float* a, const float* b, size_t count) {
  __m256 sum = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, sum);
  float result =
      tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; i < count; ++i) result += a[i] * b[i];
  return result;
}

inline void LoadPartial16I32(const int32_t* ptr, size_t remaining, __m256i& v0,
                             __m256i& v1) {
  alignas(32) int32_t tmp[16] = {};
  std::memcpy(tmp, ptr, MinSize(16, remaining) * sizeof(int32_t));
  v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
  v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp + 8));
}

inline void LoadPartial16F32(const float* ptr, size_t remaining, __m256& v0,
                             __m256& v1) {
  alignas(32) float tmp[16] = {};
  std::memcpy(tmp, ptr, MinSize(16, remaining) * sizeof(float));
  v0 = _mm256_load_ps(tmp);
  v1 = _mm256_load_ps(tmp + 8);
}

inline void StorePartial16F32(float* destination, size_t remaining, __m256 v0,
                              __m256 v1) {
  alignas(32) float tmp[16];
  _mm256_store_ps(tmp, v0);
  _mm256_store_ps(tmp + 8, v1);
  std::memcpy(destination, tmp, MinSize(16, remaining) * sizeof(float));
}

}  // namespace

void QuantizeToUint8Avx(size_t count, const float* input, float inverse_scale,
                        float zero_point, uint8_t* output) {
  const __m256 inverse_scale_vec = _mm256_set1_ps(inverse_scale);
  const __m256 zero_point_vec = _mm256_set1_ps(zero_point);
  size_t i = 0;
  for (; i + 16 <= count; i += 16) {
    StoreQuantized16(_mm256_loadu_ps(input + i), _mm256_loadu_ps(input + i + 8),
                     inverse_scale_vec, zero_point_vec, output + i);
  }
  for (; i < count; ++i) {
    const float value =
        ClampFloat(input[i] * inverse_scale + zero_point, 0.0f, 255.0f);
    const __m128 rounded =
        _mm_round_ss(_mm_setzero_ps(), _mm_set_ss(value),
                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    output[i] = static_cast<uint8_t>(_mm_cvtss_si32(rounded));
  }
}

void GemmEpilogueAvx(size_t row_begin, size_t row_end, size_t outputs,
                     size_t inputs, const uint8_t* quantized_input,
                     const int8_t* packed_weights, const float* output_scales,
                     const int32_t* zero_point_corrections, const float* biases,
                     ActivationFunction activation, float* output) {
  const size_t k_full = inputs / 4;
  const size_t k_rem = inputs & 3;
  const size_t k_groups = (inputs + 3) / 4;
  const size_t n_blocks = (outputs + kNr - 1) / kNr;
  const size_t weights_block_stride = k_groups * 64;

  constexpr size_t kMc = 64;
  constexpr size_t kNBlockChunk = 8;

  for (size_t block_base = 0; block_base < n_blocks;
       block_base += kNBlockChunk) {
    const size_t block_end = MinSize(n_blocks, block_base + kNBlockChunk);
    for (size_t row_tile = row_begin; row_tile < row_end; row_tile += kMc) {
      const size_t row_tile_end = MinSize(row_end, row_tile + kMc);
      for (size_t row = row_tile; row < row_tile_end; row += 4) {
        const size_t row_count = MinSize(4, row_tile_end - row);
        for (size_t n_block = block_base; n_block < block_end; ++n_block) {
          const size_t n_offset = n_block * kNr;
          const size_t n_remaining = outputs - n_offset;
          const bool full_n = (n_remaining >= kNr);

          __m256i zp_corr0, zp_corr1;
          __m256 scale0, scale1, bias0, bias1;
          if (full_n) {
            zp_corr0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                zero_point_corrections + n_offset));
            zp_corr1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                zero_point_corrections + n_offset + 8));
            scale0 = _mm256_loadu_ps(output_scales + n_offset);
            scale1 = _mm256_loadu_ps(output_scales + n_offset + 8);
            bias0 = biases ? _mm256_loadu_ps(biases + n_offset)
                           : _mm256_setzero_ps();
            bias1 = biases ? _mm256_loadu_ps(biases + n_offset + 8)
                           : _mm256_setzero_ps();
          } else {
            LoadPartial16I32(zero_point_corrections + n_offset, n_remaining,
                             zp_corr0, zp_corr1);
            LoadPartial16F32(output_scales + n_offset, n_remaining, scale0,
                             scale1);
            if (biases) {
              LoadPartial16F32(biases + n_offset, n_remaining, bias0, bias1);
            } else {
              bias0 = _mm256_setzero_ps();
              bias1 = _mm256_setzero_ps();
            }
          }

          const int8_t* weights_ptr =
              packed_weights + n_block * weights_block_stride;
          __m256i acc[4][2];
          for (size_t r = 0; r < row_count; ++r) {
            acc[r][0] = _mm256_setzero_si256();
            acc[r][1] = _mm256_setzero_si256();
          }

          if (row_count == 4) {
            Microkernel4x16Accum(quantized_input + row * inputs, inputs,
                                 weights_ptr, k_full, k_rem, acc);
          } else {
            for (size_t k_group = 0; k_group < k_full; ++k_group) {
              const __m256i b0 = _mm256_load_si256(
                  reinterpret_cast<const __m256i*>(weights_ptr));
              const __m256i b1 = _mm256_load_si256(
                  reinterpret_cast<const __m256i*>(weights_ptr + 32));
              const size_t offset = k_group * 4;
              for (size_t r = 0; r < row_count; ++r) {
                const __m256i av = _mm256_set1_epi32(
                    LoadU32(quantized_input + (row + r) * inputs + offset));
                acc[r][0] = DpbusdAvx(acc[r][0], av, b0);
                acc[r][1] = DpbusdAvx(acc[r][1], av, b1);
              }
              weights_ptr += 64;
            }
            if (k_rem > 0) {
              const __m256i b0 = _mm256_load_si256(
                  reinterpret_cast<const __m256i*>(weights_ptr));
              const __m256i b1 = _mm256_load_si256(
                  reinterpret_cast<const __m256i*>(weights_ptr + 32));
              const size_t offset = k_full * 4;
              for (size_t r = 0; r < row_count; ++r) {
                const __m256i av = _mm256_set1_epi32(LoadTailU32(
                    quantized_input + (row + r) * inputs + offset, k_rem));
                acc[r][0] = DpbusdAvx(acc[r][0], av, b0);
                acc[r][1] = DpbusdAvx(acc[r][1], av, b1);
              }
            }
          }

          for (size_t r = 0; r < row_count; ++r) {
            __m256 v0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][0], zp_corr0)),
                scale0, bias0);
            __m256 v1 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][1], zp_corr1)),
                scale1, bias1);
            v0 = ApplyActivationAvx(v0, activation);
            v1 = ApplyActivationAvx(v1, activation);
            float* destination = output + (row + r) * outputs + n_offset;
            if (full_n) {
              _mm256_storeu_ps(destination, v0);
              _mm256_storeu_ps(destination + 8, v1);
            } else {
              StorePartial16F32(destination, n_remaining, v0, v1);
            }
          }
        }
      }
    }
  }
}

void ResidualBlockBoardAvx(
    size_t channels, float* board_io, const int8_t* conv1_packed_weights,
    const float* conv1_output_scales,
    const int32_t* conv1_zero_point_corrections, const float* conv1_biases,
    float conv1_inverse_scale, uint8_t conv1_zero_point,
    const int8_t* conv2_packed_weights, const float* conv2_output_scales,
    const int32_t* conv2_zero_point_corrections, const float* conv2_biases,
    float conv2_inverse_scale, uint8_t conv2_zero_point,
    ActivationFunction activation, const ResidualSeParams& se,
    const Conv3x3ScratchView& scratch) {
  const size_t channel_groups = (channels + 3) / 4;
  const size_t padded_channels = channel_groups * 4;
  const size_t n_blocks = (channels + kNr - 1) / kNr;
  const size_t tap_stride = channel_groups * 64;

  uint8_t* pad1 = scratch.pad1;
  uint8_t* pad2 = scratch.pad2;

  QuantizeBoardToPadAvx(channels, padded_channels, board_io,
                        conv1_inverse_scale, conv1_zero_point, pad1);
  FillPadBorderAvx(pad2, padded_channels, conv2_zero_point);
  if (padded_channels != channels) {
    for (size_t y = 1; y <= kBoardSize; ++y) {
      for (size_t x = 1; x <= kBoardSize; ++x) {
        std::memset(pad2 + (y * kPadSize + x) * padded_channels + channels, 0,
                    padded_channels - channels);
      }
    }
  }

  const __m256 conv2_inv_vec = _mm256_set1_ps(conv2_inverse_scale);
  const __m256 conv2_zp_vec =
      _mm256_set1_ps(static_cast<float>(conv2_zero_point));

  for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
    const size_t n_offset = n_block * kNr;
    const size_t n_remaining = channels - n_offset;
    const bool full_n = (n_remaining >= kNr);
    __m256i zp_corr0, zp_corr1;
    __m256 scale0, scale1, bias0, bias1;
    if (full_n) {
      zp_corr0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          conv1_zero_point_corrections + n_offset));
      zp_corr1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          conv1_zero_point_corrections + n_offset + 8));
      scale0 = _mm256_loadu_ps(conv1_output_scales + n_offset);
      scale1 = _mm256_loadu_ps(conv1_output_scales + n_offset + 8);
      bias0 = _mm256_loadu_ps(conv1_biases + n_offset);
      bias1 = _mm256_loadu_ps(conv1_biases + n_offset + 8);
    } else {
      LoadPartial16I32(conv1_zero_point_corrections + n_offset, n_remaining,
                       zp_corr0, zp_corr1);
      LoadPartial16F32(conv1_output_scales + n_offset, n_remaining, scale0,
                       scale1);
      LoadPartial16F32(conv1_biases + n_offset, n_remaining, bias0, bias1);
    }
    const int8_t* weights_block =
        conv1_packed_weights + n_block * 9 * tap_stride;

    for (size_t y = 0; y < kBoardSize; ++y) {
      for (size_t x0 = 0; x0 < kBoardSize; x0 += 4) {
        __m256i acc[4][2];
        for (int r = 0; r < 4; ++r) {
          acc[r][0] = _mm256_setzero_si256();
          acc[r][1] = _mm256_setzero_si256();
        }
        for (size_t dy = 0; dy < 3; ++dy) {
          for (size_t dx = 0; dx < 3; ++dx) {
            const size_t tap = dy * 3 + dx;
            const uint8_t* input_row =
                pad1 + ((y + dy) * kPadSize + dx + x0) * padded_channels;
            Microkernel4x16Accum(input_row, padded_channels,
                                 weights_block + tap * tap_stride,
                                 channel_groups, 0, acc);
          }
        }
        for (size_t r = 0; r < 4; ++r) {
          __m256 v0 = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][0], zp_corr0)), scale0,
              bias0);
          __m256 v1 = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][1], zp_corr1)), scale1,
              bias1);
          v0 = ApplyActivationAvx(v0, activation);
          v1 = ApplyActivationAvx(v1, activation);
          uint8_t* destination =
              pad2 + ((y + 1) * kPadSize + (x0 + r + 1)) * padded_channels +
              n_offset;
          if (full_n) {
            StoreQuantized16(v0, v1, conv2_inv_vec, conv2_zp_vec, destination);
          } else {
            alignas(16) uint8_t tmp[16];
            StoreQuantized16(v0, v1, conv2_inv_vec, conv2_zp_vec, tmp);
            std::memcpy(destination, tmp, n_remaining);
          }
        }
      }
    }
  }

  if (se.has_se) {
    float* conv2_output = scratch.conv2_output;
    float* avg_pool = scratch.avg_pool;
    const __m256 inv64 = _mm256_set1_ps(1.0f / 64.0f);

    for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
      const size_t n_offset = n_block * kNr;
      const size_t n_remaining = channels - n_offset;
      const bool full_n = (n_remaining >= kNr);
      __m256i zp_corr0, zp_corr1;
      __m256 scale0, scale1, bias0, bias1;
      if (full_n) {
        zp_corr0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            conv2_zero_point_corrections + n_offset));
        zp_corr1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            conv2_zero_point_corrections + n_offset + 8));
        scale0 = _mm256_loadu_ps(conv2_output_scales + n_offset);
        scale1 = _mm256_loadu_ps(conv2_output_scales + n_offset + 8);
        bias0 = _mm256_loadu_ps(conv2_biases + n_offset);
        bias1 = _mm256_loadu_ps(conv2_biases + n_offset + 8);
      } else {
        LoadPartial16I32(conv2_zero_point_corrections + n_offset, n_remaining,
                         zp_corr0, zp_corr1);
        LoadPartial16F32(conv2_output_scales + n_offset, n_remaining, scale0,
                         scale1);
        LoadPartial16F32(conv2_biases + n_offset, n_remaining, bias0, bias1);
      }
      const int8_t* weights_block =
          conv2_packed_weights + n_block * 9 * tap_stride;
      __m256 pool0 = _mm256_setzero_ps();
      __m256 pool1 = _mm256_setzero_ps();

      for (size_t y = 0; y < kBoardSize; ++y) {
        for (size_t x0 = 0; x0 < kBoardSize; x0 += 4) {
          __m256i acc[4][2];
          for (int r = 0; r < 4; ++r) {
            acc[r][0] = _mm256_setzero_si256();
            acc[r][1] = _mm256_setzero_si256();
          }
          for (size_t dy = 0; dy < 3; ++dy) {
            for (size_t dx = 0; dx < 3; ++dx) {
              const size_t tap = dy * 3 + dx;
              const uint8_t* input_row =
                  pad2 + ((y + dy) * kPadSize + dx + x0) * padded_channels;
              Microkernel4x16Accum(input_row, padded_channels,
                                   weights_block + tap * tap_stride,
                                   channel_groups, 0, acc);
            }
          }
          for (size_t r = 0; r < 4; ++r) {
            __m256 v0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][0], zp_corr0)),
                scale0, bias0);
            __m256 v1 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][1], zp_corr1)),
                scale1, bias1);
            float* destination =
                conv2_output + (y * kBoardSize + x0 + r) * channels + n_offset;
            if (full_n) {
              _mm256_storeu_ps(destination, v0);
              _mm256_storeu_ps(destination + 8, v1);
            } else {
              StorePartial16F32(destination, n_remaining, v0, v1);
            }
            pool0 = _mm256_add_ps(pool0, v0);
            pool1 = _mm256_add_ps(pool1, v1);
          }
        }
      }
      pool0 = _mm256_mul_ps(pool0, inv64);
      pool1 = _mm256_mul_ps(pool1, inv64);
      if (full_n) {
        _mm256_storeu_ps(avg_pool + n_offset, pool0);
        _mm256_storeu_ps(avg_pool + n_offset + 8, pool1);
      } else {
        StorePartial16F32(avg_pool + n_offset, n_remaining, pool0, pool1);
      }
    }

    float* fc1 = scratch.se_fc1;
    for (size_t o = 0; o < se.se_fc_outputs; ++o) {
      fc1[o] = Activate(
          DotProductAvx(se.w1 + o * channels, avg_pool, channels) + se.b1[o],
          activation);
    }
    float* fc2 = scratch.se_fc2;
    for (size_t o = 0; o < 2 * channels; ++o) {
      fc2[o] =
          DotProductAvx(se.w2 + o * se.se_fc_outputs, fc1, se.se_fc_outputs) +
          se.b2[o];
    }
    size_t c_sig = 0;
    for (; c_sig + 8 <= channels; c_sig += 8) {
      _mm256_storeu_ps(fc2 + c_sig, SigmoidPsAvx(_mm256_loadu_ps(fc2 + c_sig)));
    }
    if (c_sig < channels) {
      alignas(32) float tmp[8] = {};
      const size_t remaining = channels - c_sig;
      std::memcpy(tmp, fc2 + c_sig, remaining * sizeof(float));
      _mm256_store_ps(tmp, SigmoidPsAvx(_mm256_load_ps(tmp)));
      std::memcpy(fc2 + c_sig, tmp, remaining * sizeof(float));
    }

    const float* gamma = fc2;
    const float* beta = fc2 + channels;
    for (size_t square = 0; square < kSquares; ++square) {
      const float* in_row = conv2_output + square * channels;
      float* io_row = board_io + square * channels;
      size_t c = 0;
      for (; c + 8 <= channels; c += 8) {
        const __m256 v = _mm256_fmadd_ps(
            _mm256_loadu_ps(gamma + c), _mm256_loadu_ps(in_row + c),
            _mm256_add_ps(_mm256_loadu_ps(io_row + c),
                          _mm256_loadu_ps(beta + c)));
        _mm256_storeu_ps(io_row + c, ApplyActivationAvx(v, activation));
      }
      for (; c < channels; ++c) {
        io_row[c] =
            Activate(gamma[c] * in_row[c] + io_row[c] + beta[c], activation);
      }
    }
  } else {
    for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
      const size_t n_offset = n_block * kNr;
      const size_t n_remaining = channels - n_offset;
      const bool full_n = (n_remaining >= kNr);
      __m256i zp_corr0, zp_corr1;
      __m256 scale0, scale1, bias0, bias1;
      if (full_n) {
        zp_corr0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            conv2_zero_point_corrections + n_offset));
        zp_corr1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            conv2_zero_point_corrections + n_offset + 8));
        scale0 = _mm256_loadu_ps(conv2_output_scales + n_offset);
        scale1 = _mm256_loadu_ps(conv2_output_scales + n_offset + 8);
        bias0 = _mm256_loadu_ps(conv2_biases + n_offset);
        bias1 = _mm256_loadu_ps(conv2_biases + n_offset + 8);
      } else {
        LoadPartial16I32(conv2_zero_point_corrections + n_offset, n_remaining,
                         zp_corr0, zp_corr1);
        LoadPartial16F32(conv2_output_scales + n_offset, n_remaining, scale0,
                         scale1);
        LoadPartial16F32(conv2_biases + n_offset, n_remaining, bias0, bias1);
      }
      const int8_t* weights_block =
          conv2_packed_weights + n_block * 9 * tap_stride;

      for (size_t y = 0; y < kBoardSize; ++y) {
        for (size_t x0 = 0; x0 < kBoardSize; x0 += 4) {
          __m256i acc[4][2];
          for (int r = 0; r < 4; ++r) {
            acc[r][0] = _mm256_setzero_si256();
            acc[r][1] = _mm256_setzero_si256();
          }
          for (size_t dy = 0; dy < 3; ++dy) {
            for (size_t dx = 0; dx < 3; ++dx) {
              const size_t tap = dy * 3 + dx;
              const uint8_t* input_row =
                  pad2 + ((y + dy) * kPadSize + dx + x0) * padded_channels;
              Microkernel4x16Accum(input_row, padded_channels,
                                   weights_block + tap * tap_stride,
                                   channel_groups, 0, acc);
            }
          }
          for (size_t r = 0; r < 4; ++r) {
            float* io =
                board_io + (y * kBoardSize + x0 + r) * channels + n_offset;
            __m256 skip0, skip1;
            if (full_n) {
              skip0 = _mm256_loadu_ps(io);
              skip1 = _mm256_loadu_ps(io + 8);
            } else {
              LoadPartial16F32(io, n_remaining, skip0, skip1);
            }
            __m256 v0 = _mm256_add_ps(
                _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][0], zp_corr0)),
                    scale0, bias0),
                skip0);
            __m256 v1 = _mm256_add_ps(
                _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][1], zp_corr1)),
                    scale1, bias1),
                skip1);
            v0 = ApplyActivationAvx(v0, activation);
            v1 = ApplyActivationAvx(v1, activation);
            if (full_n) {
              _mm256_storeu_ps(io, v0);
              _mm256_storeu_ps(io + 8, v1);
            } else {
              StorePartial16F32(io, n_remaining, v0, v1);
            }
          }
        }
      }
    }
  }
}

void Conv3x3NhwcBoardAvx(size_t input_channels, size_t output_channels,
                         const float* input_nhwc, const int8_t* packed_weights,
                         const float* output_scales,
                         const int32_t* zero_point_corrections,
                         float inverse_scale, uint8_t zero_point,
                         float* output_nhwc,
                         const Conv3x3ScratchView& scratch) {
  const size_t channel_groups = (input_channels + 3) / 4;
  const size_t padded_channels = channel_groups * 4;
  const size_t n_blocks = (output_channels + kNr - 1) / kNr;
  const size_t tap_stride = channel_groups * 64;

  uint8_t* pad1 = scratch.pad1;
  QuantizeBoardToPadAvx(input_channels, padded_channels, input_nhwc,
                        inverse_scale, zero_point, pad1);

  for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
    const size_t n_offset = n_block * kNr;
    const size_t n_remaining = output_channels - n_offset;
    const bool full_n = (n_remaining >= kNr);
    __m256i zp_corr0, zp_corr1;
    __m256 scale0, scale1;
    if (full_n) {
      zp_corr0 = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(zero_point_corrections + n_offset));
      zp_corr1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          zero_point_corrections + n_offset + 8));
      scale0 = _mm256_loadu_ps(output_scales + n_offset);
      scale1 = _mm256_loadu_ps(output_scales + n_offset + 8);
    } else {
      LoadPartial16I32(zero_point_corrections + n_offset, n_remaining, zp_corr0,
                       zp_corr1);
      LoadPartial16F32(output_scales + n_offset, n_remaining, scale0, scale1);
    }
    const int8_t* weights_block = packed_weights + n_block * 9 * tap_stride;

    for (size_t y = 0; y < kBoardSize; ++y) {
      for (size_t x0 = 0; x0 < kBoardSize; x0 += 4) {
        __m256i acc[4][2];
        for (int r = 0; r < 4; ++r) {
          acc[r][0] = _mm256_setzero_si256();
          acc[r][1] = _mm256_setzero_si256();
        }
        for (size_t dy = 0; dy < 3; ++dy) {
          for (size_t dx = 0; dx < 3; ++dx) {
            const size_t tap = dy * 3 + dx;
            const uint8_t* input_row =
                pad1 + ((y + dy) * kPadSize + dx + x0) * padded_channels;
            Microkernel4x16Accum(input_row, padded_channels,
                                 weights_block + tap * tap_stride,
                                 channel_groups, 0, acc);
          }
        }
        for (size_t r = 0; r < 4; ++r) {
          __m256 v0 = _mm256_mul_ps(
              _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][0], zp_corr0)),
              scale0);
          __m256 v1 = _mm256_mul_ps(
              _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][1], zp_corr1)),
              scale1);
          float* destination = output_nhwc +
                               (y * kBoardSize + x0 + r) * output_channels +
                               n_offset;
          if (full_n) {
            _mm256_storeu_ps(destination, v0);
            _mm256_storeu_ps(destination + 8, v1);
          } else {
            StorePartial16F32(destination, n_remaining, v0, v1);
          }
        }
      }
    }
  }
}

}  // namespace blas_backend
}  // namespace lczero
