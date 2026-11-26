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

// Compiled with AVX-512 VNNI flags; includes only POD headers (no STL) so no
// shared COMDAT functions can acquire EVEX/VEX instructions.

#include <immintrin.h>

#include <cstring>

#include "neural/backends/blas/int8_vnni_kernels.h"

namespace lczero {
namespace blas_backend {
namespace {

constexpr size_t kSquares = 64;
constexpr size_t kBoardSize = 8;
constexpr size_t kPadSize = 10;
constexpr size_t kNr = 32;

inline size_t MinSize(size_t a, size_t b) { return a < b ? a : b; }

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

// Fast, overflow-safe AVX-512 exp(x) accurate to ~1-2 ULP across [-87, 88].
inline __m512 ExpPsAvx512(__m512 x) {
  const __m512 min_x = _mm512_set1_ps(-87.33654475f);
  const __m512 max_x = _mm512_set1_ps(88.37626266f);
  x = _mm512_max_ps(_mm512_min_ps(x, max_x), min_x);

  const __m512 log2_e = _mm512_set1_ps(1.44269504088896341f);
  const __m512 ln2_hi = _mm512_set1_ps(0.693359375f);
  const __m512 ln2_lo = _mm512_set1_ps(-2.12194440e-4f);

  const __m512 k =
      _mm512_roundscale_ps(_mm512_mul_ps(x, log2_e), _MM_FROUND_TO_NEAREST_INT);
  __m512 r = _mm512_fnmadd_ps(k, ln2_hi, x);
  r = _mm512_fnmadd_ps(k, ln2_lo, r);

  // Degree-6 polynomial on [-ln2/2, ln2/2].
  __m512 poly = _mm512_set1_ps(1.38888889e-3f);
  poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(8.33333333e-3f));
  poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(4.16666667e-2f));
  poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(1.66666667e-1f));
  poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(0.5f));
  poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(1.0f));
  poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(1.0f));

  return _mm512_scalef_ps(poly, k);
}

inline __m512 RcpNrPsAvx512(__m512 d) {
  const __m512 two = _mm512_set1_ps(2.0f);
  const __m512 r0 = _mm512_rcp14_ps(d);
  return _mm512_mul_ps(r0, _mm512_fnmadd_ps(d, r0, two));
}

inline __m512 MishPsAvx512(__m512 x) {
  const __m512 cap = _mm512_set1_ps(20.0f);
  const __m512 two = _mm512_set1_ps(2.0f);
  const __m512 threshold = _mm512_set1_ps(-0.125f);
  const __mmask16 large_mask = _mm512_cmp_ps_mask(x, cap, _CMP_GT_OQ);
  const __m512 clamped = _mm512_min_ps(x, cap);
  const __m512 exp_val = ExpPsAvx512(clamped);
  const __m512 num =
      _mm512_fmadd_ps(exp_val, exp_val, _mm512_add_ps(exp_val, exp_val));
  const __m512 denom =
      _mm512_mul_ps(clamped, RcpNrPsAvx512(_mm512_add_ps(num, two)));
  const __mmask16 neg_mask = _mm512_cmp_ps_mask(clamped, threshold, _CMP_LE_OQ);
  const __m512 pos_branch = _mm512_fnmadd_ps(two, denom, clamped);
  const __m512 neg_branch = _mm512_mul_ps(num, denom);
  const __m512 result = _mm512_mask_blend_ps(neg_mask, pos_branch, neg_branch);
  return _mm512_mask_blend_ps(large_mask, result, x);
}

inline __m512 SwishPsAvx512(__m512 x) {
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
  return _mm512_mul_ps(x,
                       RcpNrPsAvx512(_mm512_add_ps(one, ExpPsAvx512(neg_x))));
}

inline __m512 SigmoidPsAvx512(__m512 x) {
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
  return RcpNrPsAvx512(_mm512_add_ps(one, ExpPsAvx512(neg_x)));
}

inline __m512 ApplyActivationVec(__m512 values, ActivationFunction activation) {
  switch (activation) {
    case ACTIVATION_NONE:
      return values;
    case ACTIVATION_RELU:
      return _mm512_max_ps(values, _mm512_setzero_ps());
    case ACTIVATION_RELU_2: {
      __m512 relu = _mm512_max_ps(values, _mm512_setzero_ps());
      return _mm512_mul_ps(relu, relu);
    }
    case ACTIVATION_MISH:
      return MishPsAvx512(values);
    case ACTIVATION_SWISH:
      return SwishPsAvx512(values);
    default: {
      alignas(64) float tmp[16];
      _mm512_store_ps(tmp, values);
      for (int i = 0; i < 16; ++i) tmp[i] = Activate(tmp[i], activation);
      return _mm512_load_ps(tmp);
    }
  }
}

inline void QuantizeStore16(uint8_t* destination, __m512 values,
                            __m512 inverse_scale, __m512 zero_point) {
  const __m512 zero = _mm512_setzero_ps();
  const __m512 max_val = _mm512_set1_ps(255.0f);
  __m512 scaled = _mm512_fmadd_ps(values, inverse_scale, zero_point);
  scaled = _mm512_min_ps(_mm512_max_ps(scaled, zero), max_val);
  __m512i quantized = _mm512_cvt_roundps_epi32(
      scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  // Store immediately via masked memory-form intrinsic to avoid an MSVC 19.51
  // /O2 /arch:AVX512 codegen bug where multiple live _mm512_cvtusepi32_epi8
  // temporaries alias the same zmm register when fused into vpmovusdb stores.
  _mm512_mask_cvtusepi32_storeu_epi8(destination, 0xFFFF, quantized);
}

inline void QuantizeStoreMasked(uint8_t* destination, __mmask16 mask,
                                __m512 values, __m512 inverse_scale,
                                __m512 zero_point) {
  const __m512 zero = _mm512_setzero_ps();
  const __m512 max_val = _mm512_set1_ps(255.0f);
  __m512 scaled = _mm512_fmadd_ps(values, inverse_scale, zero_point);
  scaled = _mm512_min_ps(_mm512_max_ps(scaled, zero), max_val);
  __m512i quantized = _mm512_cvt_roundps_epi32(
      scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  _mm512_mask_cvtusepi32_storeu_epi8(destination, mask, quantized);
}

inline void Transpose8x8(const float* source, size_t source_stride,
                         float* destination, size_t destination_stride) {
  __m256 r0 = _mm256_loadu_ps(source + 0 * source_stride);
  __m256 r1 = _mm256_loadu_ps(source + 1 * source_stride);
  __m256 r2 = _mm256_loadu_ps(source + 2 * source_stride);
  __m256 r3 = _mm256_loadu_ps(source + 3 * source_stride);
  __m256 r4 = _mm256_loadu_ps(source + 4 * source_stride);
  __m256 r5 = _mm256_loadu_ps(source + 5 * source_stride);
  __m256 r6 = _mm256_loadu_ps(source + 6 * source_stride);
  __m256 r7 = _mm256_loadu_ps(source + 7 * source_stride);

  __m256 t0 = _mm256_unpacklo_ps(r0, r1);
  __m256 t1 = _mm256_unpackhi_ps(r0, r1);
  __m256 t2 = _mm256_unpacklo_ps(r2, r3);
  __m256 t3 = _mm256_unpackhi_ps(r2, r3);
  __m256 t4 = _mm256_unpacklo_ps(r4, r5);
  __m256 t5 = _mm256_unpackhi_ps(r4, r5);
  __m256 t6 = _mm256_unpacklo_ps(r6, r7);
  __m256 t7 = _mm256_unpackhi_ps(r6, r7);

  __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
  __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
  __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
  __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
  __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
  __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
  __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
  __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

  _mm256_storeu_ps(destination + 0 * destination_stride,
                   _mm256_permute2f128_ps(s0, s4, 0x20));
  _mm256_storeu_ps(destination + 1 * destination_stride,
                   _mm256_permute2f128_ps(s1, s5, 0x20));
  _mm256_storeu_ps(destination + 2 * destination_stride,
                   _mm256_permute2f128_ps(s2, s6, 0x20));
  _mm256_storeu_ps(destination + 3 * destination_stride,
                   _mm256_permute2f128_ps(s3, s7, 0x20));
  _mm256_storeu_ps(destination + 4 * destination_stride,
                   _mm256_permute2f128_ps(s0, s4, 0x31));
  _mm256_storeu_ps(destination + 5 * destination_stride,
                   _mm256_permute2f128_ps(s1, s5, 0x31));
  _mm256_storeu_ps(destination + 6 * destination_stride,
                   _mm256_permute2f128_ps(s2, s6, 0x31));
  _mm256_storeu_ps(destination + 7 * destination_stride,
                   _mm256_permute2f128_ps(s3, s7, 0x31));
}

#if defined(_MSC_VER)
#define LC0_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define LC0_FORCEINLINE __attribute__((always_inline)) inline
#else
#define LC0_FORCEINLINE inline
#endif

// Accumulates 8 contiguous rows (with row stride `row_stride`) x 32 packed
// columns over `k_full` 4-byte groups plus optional `k_rem` tail bytes.
LC0_FORCEINLINE void Microkernel8x32Accum(const uint8_t* input_base,
                                          size_t row_stride,
                                          const int8_t* weights_ptr,
                                          size_t k_full, size_t k_rem,
                                          __m512i acc[8][2]) {
  const uint8_t* a0 = input_base + 0 * row_stride;
  const uint8_t* a1 = input_base + 1 * row_stride;
  const uint8_t* a2 = input_base + 2 * row_stride;
  const uint8_t* a3 = input_base + 3 * row_stride;
  const uint8_t* a4 = input_base + 4 * row_stride;
  const uint8_t* a5 = input_base + 5 * row_stride;
  const uint8_t* a6 = input_base + 6 * row_stride;
  const uint8_t* a7 = input_base + 7 * row_stride;

  __m512i c00 = acc[0][0], c01 = acc[0][1];
  __m512i c10 = acc[1][0], c11 = acc[1][1];
  __m512i c20 = acc[2][0], c21 = acc[2][1];
  __m512i c30 = acc[3][0], c31 = acc[3][1];
  __m512i c40 = acc[4][0], c41 = acc[4][1];
  __m512i c50 = acc[5][0], c51 = acc[5][1];
  __m512i c60 = acc[6][0], c61 = acc[6][1];
  __m512i c70 = acc[7][0], c71 = acc[7][1];

  size_t k_group = 0;
  for (; k_group + 1 < k_full; k_group += 2) {
    _mm_prefetch(reinterpret_cast<const char*>(weights_ptr + 512), _MM_HINT_T0);
    {
      const __m512i b0 = _mm512_load_si512(weights_ptr);
      const __m512i b1 = _mm512_load_si512(weights_ptr + 64);
      const size_t offset = k_group * 4;
      __m512i av;
      av = _mm512_set1_epi32(LoadU32(a0 + offset));
      c00 = _mm512_dpbusd_epi32(c00, av, b0);
      c01 = _mm512_dpbusd_epi32(c01, av, b1);
      av = _mm512_set1_epi32(LoadU32(a1 + offset));
      c10 = _mm512_dpbusd_epi32(c10, av, b0);
      c11 = _mm512_dpbusd_epi32(c11, av, b1);
      av = _mm512_set1_epi32(LoadU32(a2 + offset));
      c20 = _mm512_dpbusd_epi32(c20, av, b0);
      c21 = _mm512_dpbusd_epi32(c21, av, b1);
      av = _mm512_set1_epi32(LoadU32(a3 + offset));
      c30 = _mm512_dpbusd_epi32(c30, av, b0);
      c31 = _mm512_dpbusd_epi32(c31, av, b1);
      av = _mm512_set1_epi32(LoadU32(a4 + offset));
      c40 = _mm512_dpbusd_epi32(c40, av, b0);
      c41 = _mm512_dpbusd_epi32(c41, av, b1);
      av = _mm512_set1_epi32(LoadU32(a5 + offset));
      c50 = _mm512_dpbusd_epi32(c50, av, b0);
      c51 = _mm512_dpbusd_epi32(c51, av, b1);
      av = _mm512_set1_epi32(LoadU32(a6 + offset));
      c60 = _mm512_dpbusd_epi32(c60, av, b0);
      c61 = _mm512_dpbusd_epi32(c61, av, b1);
      av = _mm512_set1_epi32(LoadU32(a7 + offset));
      c70 = _mm512_dpbusd_epi32(c70, av, b0);
      c71 = _mm512_dpbusd_epi32(c71, av, b1);
    }
    {
      const __m512i b0 = _mm512_load_si512(weights_ptr + 128);
      const __m512i b1 = _mm512_load_si512(weights_ptr + 192);
      const size_t offset = (k_group + 1) * 4;
      __m512i av;
      av = _mm512_set1_epi32(LoadU32(a0 + offset));
      c00 = _mm512_dpbusd_epi32(c00, av, b0);
      c01 = _mm512_dpbusd_epi32(c01, av, b1);
      av = _mm512_set1_epi32(LoadU32(a1 + offset));
      c10 = _mm512_dpbusd_epi32(c10, av, b0);
      c11 = _mm512_dpbusd_epi32(c11, av, b1);
      av = _mm512_set1_epi32(LoadU32(a2 + offset));
      c20 = _mm512_dpbusd_epi32(c20, av, b0);
      c21 = _mm512_dpbusd_epi32(c21, av, b1);
      av = _mm512_set1_epi32(LoadU32(a3 + offset));
      c30 = _mm512_dpbusd_epi32(c30, av, b0);
      c31 = _mm512_dpbusd_epi32(c31, av, b1);
      av = _mm512_set1_epi32(LoadU32(a4 + offset));
      c40 = _mm512_dpbusd_epi32(c40, av, b0);
      c41 = _mm512_dpbusd_epi32(c41, av, b1);
      av = _mm512_set1_epi32(LoadU32(a5 + offset));
      c50 = _mm512_dpbusd_epi32(c50, av, b0);
      c51 = _mm512_dpbusd_epi32(c51, av, b1);
      av = _mm512_set1_epi32(LoadU32(a6 + offset));
      c60 = _mm512_dpbusd_epi32(c60, av, b0);
      c61 = _mm512_dpbusd_epi32(c61, av, b1);
      av = _mm512_set1_epi32(LoadU32(a7 + offset));
      c70 = _mm512_dpbusd_epi32(c70, av, b0);
      c71 = _mm512_dpbusd_epi32(c71, av, b1);
    }
    weights_ptr += 256;
  }
  for (; k_group < k_full; ++k_group) {
    const __m512i b0 = _mm512_load_si512(weights_ptr);
    const __m512i b1 = _mm512_load_si512(weights_ptr + 64);
    const size_t offset = k_group * 4;
    __m512i av;
    av = _mm512_set1_epi32(LoadU32(a0 + offset));
    c00 = _mm512_dpbusd_epi32(c00, av, b0);
    c01 = _mm512_dpbusd_epi32(c01, av, b1);
    av = _mm512_set1_epi32(LoadU32(a1 + offset));
    c10 = _mm512_dpbusd_epi32(c10, av, b0);
    c11 = _mm512_dpbusd_epi32(c11, av, b1);
    av = _mm512_set1_epi32(LoadU32(a2 + offset));
    c20 = _mm512_dpbusd_epi32(c20, av, b0);
    c21 = _mm512_dpbusd_epi32(c21, av, b1);
    av = _mm512_set1_epi32(LoadU32(a3 + offset));
    c30 = _mm512_dpbusd_epi32(c30, av, b0);
    c31 = _mm512_dpbusd_epi32(c31, av, b1);
    av = _mm512_set1_epi32(LoadU32(a4 + offset));
    c40 = _mm512_dpbusd_epi32(c40, av, b0);
    c41 = _mm512_dpbusd_epi32(c41, av, b1);
    av = _mm512_set1_epi32(LoadU32(a5 + offset));
    c50 = _mm512_dpbusd_epi32(c50, av, b0);
    c51 = _mm512_dpbusd_epi32(c51, av, b1);
    av = _mm512_set1_epi32(LoadU32(a6 + offset));
    c60 = _mm512_dpbusd_epi32(c60, av, b0);
    c61 = _mm512_dpbusd_epi32(c61, av, b1);
    av = _mm512_set1_epi32(LoadU32(a7 + offset));
    c70 = _mm512_dpbusd_epi32(c70, av, b0);
    c71 = _mm512_dpbusd_epi32(c71, av, b1);
    weights_ptr += 128;
  }
  if (k_rem > 0) {
    const __m512i b0 = _mm512_load_si512(weights_ptr);
    const __m512i b1 = _mm512_load_si512(weights_ptr + 64);
    const size_t offset = k_full * 4;
    __m512i av;
    av = _mm512_set1_epi32(LoadTailU32(a0 + offset, k_rem));
    c00 = _mm512_dpbusd_epi32(c00, av, b0);
    c01 = _mm512_dpbusd_epi32(c01, av, b1);
    av = _mm512_set1_epi32(LoadTailU32(a1 + offset, k_rem));
    c10 = _mm512_dpbusd_epi32(c10, av, b0);
    c11 = _mm512_dpbusd_epi32(c11, av, b1);
    av = _mm512_set1_epi32(LoadTailU32(a2 + offset, k_rem));
    c20 = _mm512_dpbusd_epi32(c20, av, b0);
    c21 = _mm512_dpbusd_epi32(c21, av, b1);
    av = _mm512_set1_epi32(LoadTailU32(a3 + offset, k_rem));
    c30 = _mm512_dpbusd_epi32(c30, av, b0);
    c31 = _mm512_dpbusd_epi32(c31, av, b1);
    av = _mm512_set1_epi32(LoadTailU32(a4 + offset, k_rem));
    c40 = _mm512_dpbusd_epi32(c40, av, b0);
    c41 = _mm512_dpbusd_epi32(c41, av, b1);
    av = _mm512_set1_epi32(LoadTailU32(a5 + offset, k_rem));
    c50 = _mm512_dpbusd_epi32(c50, av, b0);
    c51 = _mm512_dpbusd_epi32(c51, av, b1);
    av = _mm512_set1_epi32(LoadTailU32(a6 + offset, k_rem));
    c60 = _mm512_dpbusd_epi32(c60, av, b0);
    c61 = _mm512_dpbusd_epi32(c61, av, b1);
    av = _mm512_set1_epi32(LoadTailU32(a7 + offset, k_rem));
    c70 = _mm512_dpbusd_epi32(c70, av, b0);
    c71 = _mm512_dpbusd_epi32(c71, av, b1);
  }

  acc[0][0] = c00;
  acc[0][1] = c01;
  acc[1][0] = c10;
  acc[1][1] = c11;
  acc[2][0] = c20;
  acc[2][1] = c21;
  acc[3][0] = c30;
  acc[3][1] = c31;
  acc[4][0] = c40;
  acc[4][1] = c41;
  acc[5][0] = c50;
  acc[5][1] = c51;
  acc[6][0] = c60;
  acc[6][1] = c61;
  acc[7][0] = c70;
  acc[7][1] = c71;
}

// Runs all 9 taps of a 3x3 convolution for one 8-square rank x 32 output
// channels, keeping all 16 accumulator ZMMs in registers across all 9 taps.
void Microkernel8x32Conv3x3(const uint8_t* pad_rank_y, size_t padded_channels,
                            const int8_t* weights_block, size_t channel_groups,
                            __m512i acc[8][2]) {
  __m512i c00 = _mm512_setzero_si512(), c01 = _mm512_setzero_si512();
  __m512i c10 = _mm512_setzero_si512(), c11 = _mm512_setzero_si512();
  __m512i c20 = _mm512_setzero_si512(), c21 = _mm512_setzero_si512();
  __m512i c30 = _mm512_setzero_si512(), c31 = _mm512_setzero_si512();
  __m512i c40 = _mm512_setzero_si512(), c41 = _mm512_setzero_si512();
  __m512i c50 = _mm512_setzero_si512(), c51 = _mm512_setzero_si512();
  __m512i c60 = _mm512_setzero_si512(), c61 = _mm512_setzero_si512();
  __m512i c70 = _mm512_setzero_si512(), c71 = _mm512_setzero_si512();

  const int8_t* weights_ptr = weights_block;
  for (size_t dy = 0; dy < 3; ++dy) {
    const uint8_t* row_dy = pad_rank_y + dy * kPadSize * padded_channels;
    for (size_t dx = 0; dx < 3; ++dx) {
      const uint8_t* a0 = row_dy + (dx + 0) * padded_channels;
      const uint8_t* a1 = row_dy + (dx + 1) * padded_channels;
      const uint8_t* a2 = row_dy + (dx + 2) * padded_channels;
      const uint8_t* a3 = row_dy + (dx + 3) * padded_channels;
      const uint8_t* a4 = row_dy + (dx + 4) * padded_channels;
      const uint8_t* a5 = row_dy + (dx + 5) * padded_channels;
      const uint8_t* a6 = row_dy + (dx + 6) * padded_channels;
      const uint8_t* a7 = row_dy + (dx + 7) * padded_channels;

      size_t k_group = 0;
      for (; k_group + 1 < channel_groups; k_group += 2) {
        _mm_prefetch(reinterpret_cast<const char*>(weights_ptr + 512),
                     _MM_HINT_T0);
        {
          const __m512i b0 = _mm512_load_si512(weights_ptr);
          const __m512i b1 = _mm512_load_si512(weights_ptr + 64);
          const size_t offset = k_group * 4;
          __m512i av;
          av = _mm512_set1_epi32(LoadU32(a0 + offset));
          c00 = _mm512_dpbusd_epi32(c00, av, b0);
          c01 = _mm512_dpbusd_epi32(c01, av, b1);
          av = _mm512_set1_epi32(LoadU32(a1 + offset));
          c10 = _mm512_dpbusd_epi32(c10, av, b0);
          c11 = _mm512_dpbusd_epi32(c11, av, b1);
          av = _mm512_set1_epi32(LoadU32(a2 + offset));
          c20 = _mm512_dpbusd_epi32(c20, av, b0);
          c21 = _mm512_dpbusd_epi32(c21, av, b1);
          av = _mm512_set1_epi32(LoadU32(a3 + offset));
          c30 = _mm512_dpbusd_epi32(c30, av, b0);
          c31 = _mm512_dpbusd_epi32(c31, av, b1);
          av = _mm512_set1_epi32(LoadU32(a4 + offset));
          c40 = _mm512_dpbusd_epi32(c40, av, b0);
          c41 = _mm512_dpbusd_epi32(c41, av, b1);
          av = _mm512_set1_epi32(LoadU32(a5 + offset));
          c50 = _mm512_dpbusd_epi32(c50, av, b0);
          c51 = _mm512_dpbusd_epi32(c51, av, b1);
          av = _mm512_set1_epi32(LoadU32(a6 + offset));
          c60 = _mm512_dpbusd_epi32(c60, av, b0);
          c61 = _mm512_dpbusd_epi32(c61, av, b1);
          av = _mm512_set1_epi32(LoadU32(a7 + offset));
          c70 = _mm512_dpbusd_epi32(c70, av, b0);
          c71 = _mm512_dpbusd_epi32(c71, av, b1);
        }
        {
          const __m512i b0 = _mm512_load_si512(weights_ptr + 128);
          const __m512i b1 = _mm512_load_si512(weights_ptr + 192);
          const size_t offset = (k_group + 1) * 4;
          __m512i av;
          av = _mm512_set1_epi32(LoadU32(a0 + offset));
          c00 = _mm512_dpbusd_epi32(c00, av, b0);
          c01 = _mm512_dpbusd_epi32(c01, av, b1);
          av = _mm512_set1_epi32(LoadU32(a1 + offset));
          c10 = _mm512_dpbusd_epi32(c10, av, b0);
          c11 = _mm512_dpbusd_epi32(c11, av, b1);
          av = _mm512_set1_epi32(LoadU32(a2 + offset));
          c20 = _mm512_dpbusd_epi32(c20, av, b0);
          c21 = _mm512_dpbusd_epi32(c21, av, b1);
          av = _mm512_set1_epi32(LoadU32(a3 + offset));
          c30 = _mm512_dpbusd_epi32(c30, av, b0);
          c31 = _mm512_dpbusd_epi32(c31, av, b1);
          av = _mm512_set1_epi32(LoadU32(a4 + offset));
          c40 = _mm512_dpbusd_epi32(c40, av, b0);
          c41 = _mm512_dpbusd_epi32(c41, av, b1);
          av = _mm512_set1_epi32(LoadU32(a5 + offset));
          c50 = _mm512_dpbusd_epi32(c50, av, b0);
          c51 = _mm512_dpbusd_epi32(c51, av, b1);
          av = _mm512_set1_epi32(LoadU32(a6 + offset));
          c60 = _mm512_dpbusd_epi32(c60, av, b0);
          c61 = _mm512_dpbusd_epi32(c61, av, b1);
          av = _mm512_set1_epi32(LoadU32(a7 + offset));
          c70 = _mm512_dpbusd_epi32(c70, av, b0);
          c71 = _mm512_dpbusd_epi32(c71, av, b1);
        }
        weights_ptr += 256;
      }
      for (; k_group < channel_groups; ++k_group) {
        const __m512i b0 = _mm512_load_si512(weights_ptr);
        const __m512i b1 = _mm512_load_si512(weights_ptr + 64);
        const size_t offset = k_group * 4;
        __m512i av;
        av = _mm512_set1_epi32(LoadU32(a0 + offset));
        c00 = _mm512_dpbusd_epi32(c00, av, b0);
        c01 = _mm512_dpbusd_epi32(c01, av, b1);
        av = _mm512_set1_epi32(LoadU32(a1 + offset));
        c10 = _mm512_dpbusd_epi32(c10, av, b0);
        c11 = _mm512_dpbusd_epi32(c11, av, b1);
        av = _mm512_set1_epi32(LoadU32(a2 + offset));
        c20 = _mm512_dpbusd_epi32(c20, av, b0);
        c21 = _mm512_dpbusd_epi32(c21, av, b1);
        av = _mm512_set1_epi32(LoadU32(a3 + offset));
        c30 = _mm512_dpbusd_epi32(c30, av, b0);
        c31 = _mm512_dpbusd_epi32(c31, av, b1);
        av = _mm512_set1_epi32(LoadU32(a4 + offset));
        c40 = _mm512_dpbusd_epi32(c40, av, b0);
        c41 = _mm512_dpbusd_epi32(c41, av, b1);
        av = _mm512_set1_epi32(LoadU32(a5 + offset));
        c50 = _mm512_dpbusd_epi32(c50, av, b0);
        c51 = _mm512_dpbusd_epi32(c51, av, b1);
        av = _mm512_set1_epi32(LoadU32(a6 + offset));
        c60 = _mm512_dpbusd_epi32(c60, av, b0);
        c61 = _mm512_dpbusd_epi32(c61, av, b1);
        av = _mm512_set1_epi32(LoadU32(a7 + offset));
        c70 = _mm512_dpbusd_epi32(c70, av, b0);
        c71 = _mm512_dpbusd_epi32(c71, av, b1);
        weights_ptr += 128;
      }
    }
  }

  acc[0][0] = c00;
  acc[0][1] = c01;
  acc[1][0] = c10;
  acc[1][1] = c11;
  acc[2][0] = c20;
  acc[2][1] = c21;
  acc[3][0] = c30;
  acc[3][1] = c31;
  acc[4][0] = c40;
  acc[4][1] = c41;
  acc[5][0] = c50;
  acc[5][1] = c51;
  acc[6][0] = c60;
  acc[6][1] = c61;
  acc[7][0] = c70;
  acc[7][1] = c71;
}

// Runs all 9 taps of a 3x3 convolution for one 8-square rank x 48 output
// channels (24 ZMM accumulators + 3 weight ZMMs + 1 broadcast ZMM = 28/32
// registers, zero per-tap spills or calls).
void Microkernel8x48Conv3x3(const uint8_t* pad_rank_y, size_t padded_channels,
                            const int8_t* weights_block, size_t channel_groups,
                            __m512i acc[8][3]) {
  __m512i c00 = _mm512_setzero_si512(), c01 = _mm512_setzero_si512(),
          c02 = _mm512_setzero_si512();
  __m512i c10 = _mm512_setzero_si512(), c11 = _mm512_setzero_si512(),
          c12 = _mm512_setzero_si512();
  __m512i c20 = _mm512_setzero_si512(), c21 = _mm512_setzero_si512(),
          c22 = _mm512_setzero_si512();
  __m512i c30 = _mm512_setzero_si512(), c31 = _mm512_setzero_si512(),
          c32 = _mm512_setzero_si512();
  __m512i c40 = _mm512_setzero_si512(), c41 = _mm512_setzero_si512(),
          c42 = _mm512_setzero_si512();
  __m512i c50 = _mm512_setzero_si512(), c51 = _mm512_setzero_si512(),
          c52 = _mm512_setzero_si512();
  __m512i c60 = _mm512_setzero_si512(), c61 = _mm512_setzero_si512(),
          c62 = _mm512_setzero_si512();
  __m512i c70 = _mm512_setzero_si512(), c71 = _mm512_setzero_si512(),
          c72 = _mm512_setzero_si512();

  const int8_t* weights_ptr = weights_block;
  for (size_t dy = 0; dy < 3; ++dy) {
    const uint8_t* row_dy = pad_rank_y + dy * kPadSize * padded_channels;
    for (size_t dx = 0; dx < 3; ++dx) {
      const uint8_t* a0 = row_dy + (dx + 0) * padded_channels;
      const uint8_t* a1 = row_dy + (dx + 1) * padded_channels;
      const uint8_t* a2 = row_dy + (dx + 2) * padded_channels;
      const uint8_t* a3 = row_dy + (dx + 3) * padded_channels;
      const uint8_t* a4 = row_dy + (dx + 4) * padded_channels;
      const uint8_t* a5 = row_dy + (dx + 5) * padded_channels;
      const uint8_t* a6 = row_dy + (dx + 6) * padded_channels;
      const uint8_t* a7 = row_dy + (dx + 7) * padded_channels;

      for (size_t k_group = 0; k_group < channel_groups; ++k_group) {
        _mm_prefetch(reinterpret_cast<const char*>(weights_ptr + 576),
                     _MM_HINT_T0);
        const __m512i b0 = _mm512_load_si512(weights_ptr);
        const __m512i b1 = _mm512_load_si512(weights_ptr + 64);
        const __m512i b2 = _mm512_load_si512(weights_ptr + 128);
        const size_t offset = k_group * 4;
        __m512i av;
        av = _mm512_set1_epi32(LoadU32(a0 + offset));
        c00 = _mm512_dpbusd_epi32(c00, av, b0);
        c01 = _mm512_dpbusd_epi32(c01, av, b1);
        c02 = _mm512_dpbusd_epi32(c02, av, b2);
        av = _mm512_set1_epi32(LoadU32(a1 + offset));
        c10 = _mm512_dpbusd_epi32(c10, av, b0);
        c11 = _mm512_dpbusd_epi32(c11, av, b1);
        c12 = _mm512_dpbusd_epi32(c12, av, b2);
        av = _mm512_set1_epi32(LoadU32(a2 + offset));
        c20 = _mm512_dpbusd_epi32(c20, av, b0);
        c21 = _mm512_dpbusd_epi32(c21, av, b1);
        c22 = _mm512_dpbusd_epi32(c22, av, b2);
        av = _mm512_set1_epi32(LoadU32(a3 + offset));
        c30 = _mm512_dpbusd_epi32(c30, av, b0);
        c31 = _mm512_dpbusd_epi32(c31, av, b1);
        c32 = _mm512_dpbusd_epi32(c32, av, b2);
        av = _mm512_set1_epi32(LoadU32(a4 + offset));
        c40 = _mm512_dpbusd_epi32(c40, av, b0);
        c41 = _mm512_dpbusd_epi32(c41, av, b1);
        c42 = _mm512_dpbusd_epi32(c42, av, b2);
        av = _mm512_set1_epi32(LoadU32(a5 + offset));
        c50 = _mm512_dpbusd_epi32(c50, av, b0);
        c51 = _mm512_dpbusd_epi32(c51, av, b1);
        c52 = _mm512_dpbusd_epi32(c52, av, b2);
        av = _mm512_set1_epi32(LoadU32(a6 + offset));
        c60 = _mm512_dpbusd_epi32(c60, av, b0);
        c61 = _mm512_dpbusd_epi32(c61, av, b1);
        c62 = _mm512_dpbusd_epi32(c62, av, b2);
        av = _mm512_set1_epi32(LoadU32(a7 + offset));
        c70 = _mm512_dpbusd_epi32(c70, av, b0);
        c71 = _mm512_dpbusd_epi32(c71, av, b1);
        c72 = _mm512_dpbusd_epi32(c72, av, b2);
        weights_ptr += 192;
      }
    }
  }

  acc[0][0] = c00;
  acc[0][1] = c01;
  acc[0][2] = c02;
  acc[1][0] = c10;
  acc[1][1] = c11;
  acc[1][2] = c12;
  acc[2][0] = c20;
  acc[2][1] = c21;
  acc[2][2] = c22;
  acc[3][0] = c30;
  acc[3][1] = c31;
  acc[3][2] = c32;
  acc[4][0] = c40;
  acc[4][1] = c41;
  acc[4][2] = c42;
  acc[5][0] = c50;
  acc[5][1] = c51;
  acc[5][2] = c52;
  acc[6][0] = c60;
  acc[6][1] = c61;
  acc[6][2] = c62;
  acc[7][0] = c70;
  acc[7][1] = c71;
  acc[7][2] = c72;
}

inline void FillPadBorder(uint8_t* pad, size_t padded_channels,
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

inline void QuantizeBoardToPad(size_t channels, size_t padded_channels,
                               const float* input_nhwc, float inverse_scale,
                               uint8_t zero_point, uint8_t* pad) {
  FillPadBorder(pad, padded_channels, zero_point);
  const float zero_point_f = static_cast<float>(zero_point);
  if (padded_channels == channels) {
    for (size_t y = 0; y < kBoardSize; ++y) {
      QuantizeToUint8Avx512(kBoardSize * channels,
                            input_nhwc + y * kBoardSize * channels,
                            inverse_scale, zero_point_f,
                            pad + ((y + 1) * kPadSize + 1) * padded_channels);
    }
  } else {
    for (size_t y = 0; y < kBoardSize; ++y) {
      for (size_t x = 0; x < kBoardSize; ++x) {
        uint8_t* destination =
            pad + ((y + 1) * kPadSize + (x + 1)) * padded_channels;
        QuantizeToUint8Avx512(channels,
                              input_nhwc + (y * kBoardSize + x) * channels,
                              inverse_scale, zero_point_f, destination);
        std::memset(destination + channels, 0, padded_channels - channels);
      }
    }
  }
}

inline float DotProductAvx512(const float* a, const float* b, size_t count) {
  __m512 sum0 = _mm512_setzero_ps();
  __m512 sum1 = _mm512_setzero_ps();
  size_t i = 0;
  for (; i + 32 <= count; i += 32) {
    sum0 =
        _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), sum0);
    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16),
                           _mm512_loadu_ps(b + i + 16), sum1);
  }
  for (; i + 16 <= count; i += 16) {
    sum0 =
        _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), sum0);
  }
  if (i < count) {
    const __mmask16 mask = static_cast<__mmask16>((1u << (count - i)) - 1u);
    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, a + i),
                           _mm512_maskz_loadu_ps(mask, b + i), sum0);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
}

template <ActivationFunction Act>
inline __m512 ApplyActivationStatic(__m512 values,
                                    ActivationFunction runtime_activation) {
  if constexpr (Act == ACTIVATION_NONE) {
    return values;
  } else if constexpr (Act == ACTIVATION_RELU) {
    return _mm512_max_ps(values, _mm512_setzero_ps());
  } else if constexpr (Act == ACTIVATION_RELU_2) {
    __m512 relu = _mm512_max_ps(values, _mm512_setzero_ps());
    return _mm512_mul_ps(relu, relu);
  } else if constexpr (Act == ACTIVATION_MISH) {
    return MishPsAvx512(values);
  } else if constexpr (Act == ACTIVATION_SWISH) {
    return SwishPsAvx512(values);
  } else {
    return ApplyActivationVec(values, runtime_activation);
  }
}

template <ActivationFunction Act, bool kQuantizeOut>
void GemmEpilogueAvx512Impl(
    size_t row_begin, size_t row_end, size_t outputs, size_t inputs,
    const uint8_t* quantized_input, const int8_t* packed_weights,
    const float* output_scales, const int32_t* zero_point_corrections,
    const float* biases, ActivationFunction runtime_activation, float* output,
    float out_inverse_scale, float out_zero_point, uint8_t* quantized_output) {
  const size_t k_full = inputs / 4;
  const size_t k_rem = inputs & 3;
  const size_t k_groups = (inputs + 3) / 4;
  const size_t n_blocks = (outputs + kNr - 1) / kNr;
  const size_t weights_block_stride = k_groups * 128;
  const __m512 out_inv_vec = _mm512_set1_ps(out_inverse_scale);
  const __m512 out_zp_vec = _mm512_set1_ps(out_zero_point);

  constexpr size_t kMc = 64;
  constexpr size_t kNBlockChunk = 4;

  for (size_t block_base = 0; block_base < n_blocks;
       block_base += kNBlockChunk) {
    const size_t block_end = MinSize(n_blocks, block_base + kNBlockChunk);
    for (size_t row_tile = row_begin; row_tile < row_end; row_tile += kMc) {
      const size_t row_tile_end = MinSize(row_end, row_tile + kMc);
      for (size_t row = row_tile; row < row_tile_end; row += 8) {
        const size_t row_count = MinSize(8, row_tile_end - row);
        for (size_t n_block = block_base; n_block < block_end; ++n_block) {
          const size_t n_offset = n_block * kNr;
          const size_t n_remaining = outputs - n_offset;
          const bool full_n = (n_remaining >= kNr);
          const __mmask16 mask0 =
              full_n ? 0xFFFF
                     : static_cast<__mmask16>(n_remaining >= 16
                                                  ? 0xFFFF
                                                  : ((1u << n_remaining) - 1u));
          const __mmask16 mask1 =
              full_n ? 0xFFFF
                     : static_cast<__mmask16>(
                           n_remaining > 16 ? ((1u << (n_remaining - 16)) - 1u)
                                            : 0u);

          const __m512i zp_corr0 =
              full_n ? _mm512_loadu_si512(zero_point_corrections + n_offset)
                     : _mm512_maskz_loadu_epi32(
                           mask0, zero_point_corrections + n_offset);
          const __m512i zp_corr1 =
              full_n
                  ? _mm512_loadu_si512(zero_point_corrections + n_offset + 16)
                  : _mm512_maskz_loadu_epi32(
                        mask1, zero_point_corrections + n_offset + 16);
          const __m512 scale0 =
              full_n ? _mm512_loadu_ps(output_scales + n_offset)
                     : _mm512_maskz_loadu_ps(mask0, output_scales + n_offset);
          const __m512 scale1 =
              full_n
                  ? _mm512_loadu_ps(output_scales + n_offset + 16)
                  : _mm512_maskz_loadu_ps(mask1, output_scales + n_offset + 16);
          const __m512 bias0 =
              biases
                  ? (full_n ? _mm512_loadu_ps(biases + n_offset)
                            : _mm512_maskz_loadu_ps(mask0, biases + n_offset))
                  : _mm512_setzero_ps();
          const __m512 bias1 =
              biases
                  ? (full_n
                         ? _mm512_loadu_ps(biases + n_offset + 16)
                         : _mm512_maskz_loadu_ps(mask1, biases + n_offset + 16))
                  : _mm512_setzero_ps();

          const int8_t* weights_ptr =
              packed_weights + n_block * weights_block_stride;
          __m512i acc[8][2];
          for (size_t r = 0; r < row_count; ++r) {
            acc[r][0] = _mm512_setzero_si512();
            acc[r][1] = _mm512_setzero_si512();
          }

          if (row_count == 8) {
            Microkernel8x32Accum(quantized_input + row * inputs, inputs,
                                 weights_ptr, k_full, k_rem, acc);
          } else {
            for (size_t k_group = 0; k_group < k_full; ++k_group) {
              const __m512i b0 = _mm512_load_si512(weights_ptr);
              const __m512i b1 = _mm512_load_si512(weights_ptr + 64);
              const size_t offset = k_group * 4;
              for (size_t r = 0; r < row_count; ++r) {
                const __m512i av = _mm512_set1_epi32(
                    LoadU32(quantized_input + (row + r) * inputs + offset));
                acc[r][0] = _mm512_dpbusd_epi32(acc[r][0], av, b0);
                acc[r][1] = _mm512_dpbusd_epi32(acc[r][1], av, b1);
              }
              weights_ptr += 128;
            }
            if (k_rem > 0) {
              const __m512i b0 = _mm512_load_si512(weights_ptr);
              const __m512i b1 = _mm512_load_si512(weights_ptr + 64);
              const size_t offset = k_full * 4;
              for (size_t r = 0; r < row_count; ++r) {
                const __m512i av = _mm512_set1_epi32(LoadTailU32(
                    quantized_input + (row + r) * inputs + offset, k_rem));
                acc[r][0] = _mm512_dpbusd_epi32(acc[r][0], av, b0);
                acc[r][1] = _mm512_dpbusd_epi32(acc[r][1], av, b1);
              }
            }
          }

          if (full_n) {
            for (size_t r = 0; r < row_count; ++r) {
              __m512 v0 = _mm512_fmadd_ps(
                  _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[r][0], zp_corr0)),
                  scale0, bias0);
              __m512 v1 = _mm512_fmadd_ps(
                  _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[r][1], zp_corr1)),
                  scale1, bias1);
              v0 = ApplyActivationStatic<Act>(v0, runtime_activation);
              v1 = ApplyActivationStatic<Act>(v1, runtime_activation);
              if constexpr (kQuantizeOut) {
                uint8_t* destination =
                    quantized_output + (row + r) * outputs + n_offset;
                QuantizeStore16(destination, v0, out_inv_vec, out_zp_vec);
                QuantizeStore16(destination + 16, v1, out_inv_vec, out_zp_vec);
              } else {
                float* destination = output + (row + r) * outputs + n_offset;
                _mm512_storeu_ps(destination, v0);
                _mm512_storeu_ps(destination + 16, v1);
              }
            }
          } else {
            for (size_t r = 0; r < row_count; ++r) {
              __m512 v0 = _mm512_fmadd_ps(
                  _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[r][0], zp_corr0)),
                  scale0, bias0);
              __m512 v1 = _mm512_fmadd_ps(
                  _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[r][1], zp_corr1)),
                  scale1, bias1);
              v0 = ApplyActivationStatic<Act>(v0, runtime_activation);
              v1 = ApplyActivationStatic<Act>(v1, runtime_activation);
              if constexpr (kQuantizeOut) {
                uint8_t* destination =
                    quantized_output + (row + r) * outputs + n_offset;
                QuantizeStoreMasked(destination, mask0, v0, out_inv_vec,
                                    out_zp_vec);
                if (mask1) {
                  QuantizeStoreMasked(destination + 16, mask1, v1, out_inv_vec,
                                      out_zp_vec);
                }
              } else {
                float* destination = output + (row + r) * outputs + n_offset;
                _mm512_mask_storeu_ps(destination, mask0, v0);
                if (mask1) _mm512_mask_storeu_ps(destination + 16, mask1, v1);
              }
            }
          }
        }
      }
    }
  }
}

// Processes `num_boards` boards in NHWC layout using zero-spill y-outer 8x32
// VNNI accumulation (all 16 accumulator ZMMs stay in registers across all 9
// filter taps).
template <ActivationFunction Act>
void ResidualBlockBoardsAvx512Impl(
    size_t num_boards, size_t channels, float* boards_io,
    const int8_t* conv1_packed_weights, const float* conv1_output_scales,
    const int32_t* conv1_zero_point_corrections, const float* conv1_biases,
    float conv1_inverse_scale, uint8_t conv1_zero_point,
    const int8_t* conv2_packed_weights, const float* conv2_output_scales,
    const int32_t* conv2_zero_point_corrections, const float* conv2_biases,
    float conv2_inverse_scale, uint8_t conv2_zero_point,
    ActivationFunction runtime_activation, const ResidualSeParams& se,
    const Conv3x3ScratchView& scratch) {
  const size_t channel_groups = (channels + 3) / 4;
  const size_t padded_channels = channel_groups * 4;
  const size_t n_blocks = (channels + kNr - 1) / kNr;
  const size_t tap_stride = channel_groups * 128;
  const size_t io_board_stride = kSquares * channels;

  uint8_t* pad1 = scratch.pad1;
  uint8_t* pad2 = scratch.pad2;
  const __m512 conv2_inv_vec = _mm512_set1_ps(conv2_inverse_scale);
  const __m512 conv2_zp_vec =
      _mm512_set1_ps(static_cast<float>(conv2_zero_point));

  for (size_t b = 0; b < num_boards; ++b) {
    float* board_io = boards_io + b * io_board_stride;

    QuantizeBoardToPad(channels, padded_channels, board_io, conv1_inverse_scale,
                       conv1_zero_point, pad1);

    FillPadBorder(pad2, padded_channels, conv2_zero_point);
    if (padded_channels != channels) {
      for (size_t y = 1; y <= kBoardSize; ++y) {
        for (size_t x = 1; x <= kBoardSize; ++x) {
          std::memset(pad2 + (y * kPadSize + x) * padded_channels + channels, 0,
                      padded_channels - channels);
        }
      }
    }

    // Conv1 with fused bias, activation, and requantization into pad2.
    for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
      const size_t n_offset = n_block * kNr;
      const size_t n_remaining = channels - n_offset;
      const bool full_n = (n_remaining >= kNr);
      const __mmask16 mask0 =
          full_n ? 0xFFFF
                 : static_cast<__mmask16>(
                       n_remaining >= 16 ? 0xFFFF : ((1u << n_remaining) - 1u));
      const __mmask16 mask1 =
          full_n
              ? 0xFFFF
              : static_cast<__mmask16>(
                    n_remaining > 16 ? ((1u << (n_remaining - 16)) - 1u) : 0u);

      const __m512i zp_corr0 =
          full_n ? _mm512_loadu_si512(conv1_zero_point_corrections + n_offset)
                 : _mm512_maskz_loadu_epi32(
                       mask0, conv1_zero_point_corrections + n_offset);
      const __m512i zp_corr1 =
          full_n
              ? _mm512_loadu_si512(conv1_zero_point_corrections + n_offset + 16)
              : _mm512_maskz_loadu_epi32(
                    mask1, conv1_zero_point_corrections + n_offset + 16);
      const __m512 scale0 =
          full_n ? _mm512_loadu_ps(conv1_output_scales + n_offset)
                 : _mm512_maskz_loadu_ps(mask0, conv1_output_scales + n_offset);
      const __m512 scale1 =
          full_n ? _mm512_loadu_ps(conv1_output_scales + n_offset + 16)
                 : _mm512_maskz_loadu_ps(mask1,
                                         conv1_output_scales + n_offset + 16);
      const __m512 bias0 =
          full_n ? _mm512_loadu_ps(conv1_biases + n_offset)
                 : _mm512_maskz_loadu_ps(mask0, conv1_biases + n_offset);
      const __m512 bias1 =
          full_n ? _mm512_loadu_ps(conv1_biases + n_offset + 16)
                 : _mm512_maskz_loadu_ps(mask1, conv1_biases + n_offset + 16);

      const int8_t* weights_block =
          conv1_packed_weights + n_block * 9 * tap_stride;

      for (size_t y = 0; y < kBoardSize; ++y) {
        __m512i acc[8][2];
        Microkernel8x32Conv3x3(pad1 + y * kPadSize * padded_channels,
                               padded_channels, weights_block, channel_groups,
                               acc);

        uint8_t* destination_row =
            pad2 + ((y + 1) * kPadSize + 1) * padded_channels + n_offset;
        if (full_n) {
          for (size_t x = 0; x < kBoardSize; ++x) {
            __m512 v0 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][0], zp_corr0)),
                scale0, bias0);
            __m512 v1 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][1], zp_corr1)),
                scale1, bias1);
            v0 = ApplyActivationStatic<Act>(v0, runtime_activation);
            v1 = ApplyActivationStatic<Act>(v1, runtime_activation);
            uint8_t* destination = destination_row + x * padded_channels;
            QuantizeStore16(destination, v0, conv2_inv_vec, conv2_zp_vec);
            QuantizeStore16(destination + 16, v1, conv2_inv_vec, conv2_zp_vec);
          }
        } else {
          for (size_t x = 0; x < kBoardSize; ++x) {
            __m512 v0 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][0], zp_corr0)),
                scale0, bias0);
            __m512 v1 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][1], zp_corr1)),
                scale1, bias1);
            v0 = ApplyActivationStatic<Act>(v0, runtime_activation);
            v1 = ApplyActivationStatic<Act>(v1, runtime_activation);
            uint8_t* destination = destination_row + x * padded_channels;
            QuantizeStoreMasked(destination, mask0, v0, conv2_inv_vec,
                                conv2_zp_vec);
            if (mask1) {
              QuantizeStoreMasked(destination + 16, mask1, v1, conv2_inv_vec,
                                  conv2_zp_vec);
            }
          }
        }
      }
    }

    // Conv2 with optional fused global average pooling and SE unit.
    if (se.has_se) {
      float* conv2_output = scratch.conv2_output;
      float* avg_pool = scratch.avg_pool;
      const __m512 inv64 = _mm512_set1_ps(1.0f / 64.0f);

      for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
        const size_t n_offset = n_block * kNr;
        const size_t n_remaining = channels - n_offset;
        const bool full_n = (n_remaining >= kNr);
        const __mmask16 mask0 =
            full_n
                ? 0xFFFF
                : static_cast<__mmask16>(
                      n_remaining >= 16 ? 0xFFFF : ((1u << n_remaining) - 1u));
        const __mmask16 mask1 =
            full_n
                ? 0xFFFF
                : static_cast<__mmask16>(n_remaining > 16
                                             ? ((1u << (n_remaining - 16)) - 1u)
                                             : 0u);

        const __m512i zp_corr0 =
            full_n ? _mm512_loadu_si512(conv2_zero_point_corrections + n_offset)
                   : _mm512_maskz_loadu_epi32(
                         mask0, conv2_zero_point_corrections + n_offset);
        const __m512i zp_corr1 =
            full_n ? _mm512_loadu_si512(conv2_zero_point_corrections +
                                        n_offset + 16)
                   : _mm512_maskz_loadu_epi32(
                         mask1, conv2_zero_point_corrections + n_offset + 16);
        const __m512 scale0 =
            full_n
                ? _mm512_loadu_ps(conv2_output_scales + n_offset)
                : _mm512_maskz_loadu_ps(mask0, conv2_output_scales + n_offset);
        const __m512 scale1 =
            full_n ? _mm512_loadu_ps(conv2_output_scales + n_offset + 16)
                   : _mm512_maskz_loadu_ps(mask1,
                                           conv2_output_scales + n_offset + 16);
        const __m512 bias0 =
            full_n ? _mm512_loadu_ps(conv2_biases + n_offset)
                   : _mm512_maskz_loadu_ps(mask0, conv2_biases + n_offset);
        const __m512 bias1 =
            full_n ? _mm512_loadu_ps(conv2_biases + n_offset + 16)
                   : _mm512_maskz_loadu_ps(mask1, conv2_biases + n_offset + 16);

        const int8_t* weights_block =
            conv2_packed_weights + n_block * 9 * tap_stride;
        __m512 pool0 = _mm512_setzero_ps();
        __m512 pool1 = _mm512_setzero_ps();

        for (size_t y = 0; y < kBoardSize; ++y) {
          __m512i acc[8][2];
          Microkernel8x32Conv3x3(pad2 + y * kPadSize * padded_channels,
                                 padded_channels, weights_block, channel_groups,
                                 acc);

          if (full_n) {
            for (size_t x = 0; x < kBoardSize; ++x) {
              __m512 v0 = _mm512_fmadd_ps(
                  _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][0], zp_corr0)),
                  scale0, bias0);
              __m512 v1 = _mm512_fmadd_ps(
                  _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][1], zp_corr1)),
                  scale1, bias1);
              float* destination =
                  conv2_output + (y * kBoardSize + x) * channels + n_offset;
              _mm512_storeu_ps(destination, v0);
              _mm512_storeu_ps(destination + 16, v1);
              pool0 = _mm512_add_ps(pool0, v0);
              pool1 = _mm512_add_ps(pool1, v1);
            }
          } else {
            for (size_t x = 0; x < kBoardSize; ++x) {
              __m512 v0 = _mm512_fmadd_ps(
                  _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][0], zp_corr0)),
                  scale0, bias0);
              __m512 v1 = _mm512_fmadd_ps(
                  _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][1], zp_corr1)),
                  scale1, bias1);
              float* destination =
                  conv2_output + (y * kBoardSize + x) * channels + n_offset;
              _mm512_mask_storeu_ps(destination, mask0, v0);
              if (mask1) _mm512_mask_storeu_ps(destination + 16, mask1, v1);
              pool0 = _mm512_add_ps(pool0, v0);
              pool1 = _mm512_add_ps(pool1, v1);
            }
          }
        }

        pool0 = _mm512_mul_ps(pool0, inv64);
        pool1 = _mm512_mul_ps(pool1, inv64);
        if (full_n) {
          _mm512_storeu_ps(avg_pool + n_offset, pool0);
          _mm512_storeu_ps(avg_pool + n_offset + 16, pool1);
        } else {
          _mm512_mask_storeu_ps(avg_pool + n_offset, mask0, pool0);
          if (mask1) {
            _mm512_mask_storeu_ps(avg_pool + n_offset + 16, mask1, pool1);
          }
        }
      }

      float* fc1 = scratch.se_fc1;
      for (size_t o = 0; o < se.se_fc_outputs; ++o) {
        const float value =
            DotProductAvx512(se.w1 + o * channels, avg_pool, channels) +
            se.b1[o];
        fc1[o] = Activate(value, runtime_activation);
      }

      float* fc2 = scratch.se_fc2;
      for (size_t o = 0; o < 2 * channels; ++o) {
        fc2[o] = DotProductAvx512(se.w2 + o * se.se_fc_outputs, fc1,
                                  se.se_fc_outputs) +
                 se.b2[o];
      }
      for (size_t c = 0; c < channels; c += 16) {
        const size_t remaining = channels - c;
        if (remaining >= 16) {
          _mm512_storeu_ps(fc2 + c, SigmoidPsAvx512(_mm512_loadu_ps(fc2 + c)));
        } else {
          const __mmask16 mask = static_cast<__mmask16>((1u << remaining) - 1u);
          _mm512_mask_storeu_ps(
              fc2 + c, mask,
              SigmoidPsAvx512(_mm512_maskz_loadu_ps(mask, fc2 + c)));
        }
      }

      const float* gamma = fc2;
      const float* beta = fc2 + channels;
      for (size_t square = 0; square < kSquares; ++square) {
        const float* in_row = conv2_output + square * channels;
        float* io_row = board_io + square * channels;
        size_t c = 0;
        for (; c + 16 <= channels; c += 16) {
          const __m512 g = _mm512_loadu_ps(gamma + c);
          const __m512 bt = _mm512_loadu_ps(beta + c);
          const __m512 in_v = _mm512_loadu_ps(in_row + c);
          const __m512 skip_v = _mm512_loadu_ps(io_row + c);
          const __m512 v = _mm512_fmadd_ps(g, in_v, _mm512_add_ps(skip_v, bt));
          _mm512_storeu_ps(io_row + c,
                           ApplyActivationStatic<Act>(v, runtime_activation));
        }
        if (c < channels) {
          const __mmask16 mask =
              static_cast<__mmask16>((1u << (channels - c)) - 1u);
          const __m512 g = _mm512_maskz_loadu_ps(mask, gamma + c);
          const __m512 bt = _mm512_maskz_loadu_ps(mask, beta + c);
          const __m512 in_v = _mm512_maskz_loadu_ps(mask, in_row + c);
          const __m512 skip_v = _mm512_maskz_loadu_ps(mask, io_row + c);
          const __m512 v = _mm512_fmadd_ps(g, in_v, _mm512_add_ps(skip_v, bt));
          _mm512_mask_storeu_ps(
              io_row + c, mask,
              ApplyActivationStatic<Act>(v, runtime_activation));
        }
      }
    } else {
      for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
        const size_t n_offset = n_block * kNr;
        const size_t n_remaining = channels - n_offset;
        const bool full_n = (n_remaining >= kNr);
        const __mmask16 mask0 =
            full_n
                ? 0xFFFF
                : static_cast<__mmask16>(
                      n_remaining >= 16 ? 0xFFFF : ((1u << n_remaining) - 1u));
        const __mmask16 mask1 =
            full_n
                ? 0xFFFF
                : static_cast<__mmask16>(n_remaining > 16
                                             ? ((1u << (n_remaining - 16)) - 1u)
                                             : 0u);

        const __m512i zp_corr0 =
            full_n ? _mm512_loadu_si512(conv2_zero_point_corrections + n_offset)
                   : _mm512_maskz_loadu_epi32(
                         mask0, conv2_zero_point_corrections + n_offset);
        const __m512i zp_corr1 =
            full_n ? _mm512_loadu_si512(conv2_zero_point_corrections +
                                        n_offset + 16)
                   : _mm512_maskz_loadu_epi32(
                         mask1, conv2_zero_point_corrections + n_offset + 16);
        const __m512 scale0 =
            full_n
                ? _mm512_loadu_ps(conv2_output_scales + n_offset)
                : _mm512_maskz_loadu_ps(mask0, conv2_output_scales + n_offset);
        const __m512 scale1 =
            full_n ? _mm512_loadu_ps(conv2_output_scales + n_offset + 16)
                   : _mm512_maskz_loadu_ps(mask1,
                                           conv2_output_scales + n_offset + 16);
        const __m512 bias0 =
            full_n ? _mm512_loadu_ps(conv2_biases + n_offset)
                   : _mm512_maskz_loadu_ps(mask0, conv2_biases + n_offset);
        const __m512 bias1 =
            full_n ? _mm512_loadu_ps(conv2_biases + n_offset + 16)
                   : _mm512_maskz_loadu_ps(mask1, conv2_biases + n_offset + 16);

        const int8_t* weights_block =
            conv2_packed_weights + n_block * 9 * tap_stride;

        for (size_t y = 0; y < kBoardSize; ++y) {
          __m512i acc[8][2];
          Microkernel8x32Conv3x3(pad2 + y * kPadSize * padded_channels,
                                 padded_channels, weights_block, channel_groups,
                                 acc);

          if (full_n) {
            for (size_t x = 0; x < kBoardSize; ++x) {
              float* io = board_io + (y * kBoardSize + x) * channels + n_offset;
              __m512 v0 = _mm512_fmadd_ps(
                  _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][0], zp_corr0)),
                  scale0, bias0);
              __m512 v1 = _mm512_fmadd_ps(
                  _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][1], zp_corr1)),
                  scale1, bias1);
              v0 = _mm512_add_ps(v0, _mm512_loadu_ps(io));
              v1 = _mm512_add_ps(v1, _mm512_loadu_ps(io + 16));
              _mm512_storeu_ps(
                  io, ApplyActivationStatic<Act>(v0, runtime_activation));
              _mm512_storeu_ps(
                  io + 16, ApplyActivationStatic<Act>(v1, runtime_activation));
            }
          } else {
            for (size_t x = 0; x < kBoardSize; ++x) {
              float* io = board_io + (y * kBoardSize + x) * channels + n_offset;
              __m512 v0 = _mm512_fmadd_ps(
                  _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][0], zp_corr0)),
                  scale0, bias0);
              __m512 v1 = _mm512_fmadd_ps(
                  _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][1], zp_corr1)),
                  scale1, bias1);
              v0 = _mm512_add_ps(v0, _mm512_maskz_loadu_ps(mask0, io));
              v1 = _mm512_add_ps(v1, _mm512_maskz_loadu_ps(mask1, io + 16));
              _mm512_mask_storeu_ps(
                  io, mask0,
                  ApplyActivationStatic<Act>(v0, runtime_activation));
              if (mask1) {
                _mm512_mask_storeu_ps(
                    io + 16, mask1,
                    ApplyActivationStatic<Act>(v1, runtime_activation));
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace

void QuantizeToUint8Avx512(size_t count, const float* input,
                           float inverse_scale, float zero_point,
                           uint8_t* output) {
  const __m512 inverse_scale_vec = _mm512_set1_ps(inverse_scale);
  const __m512 zero_point_vec = _mm512_set1_ps(zero_point);
  size_t i = 0;
  for (; i + 64 <= count; i += 64) {
    QuantizeStore16(output + i, _mm512_loadu_ps(input + i), inverse_scale_vec,
                    zero_point_vec);
    QuantizeStore16(output + i + 16, _mm512_loadu_ps(input + i + 16),
                    inverse_scale_vec, zero_point_vec);
    QuantizeStore16(output + i + 32, _mm512_loadu_ps(input + i + 32),
                    inverse_scale_vec, zero_point_vec);
    QuantizeStore16(output + i + 48, _mm512_loadu_ps(input + i + 48),
                    inverse_scale_vec, zero_point_vec);
  }
  for (; i + 16 <= count; i += 16) {
    QuantizeStore16(output + i, _mm512_loadu_ps(input + i), inverse_scale_vec,
                    zero_point_vec);
  }
  if (i < count) {
    const size_t remaining = count - i;
    const __mmask16 mask = static_cast<__mmask16>((1u << remaining) - 1u);
    QuantizeStoreMasked(output + i, mask,
                        _mm512_maskz_loadu_ps(mask, input + i),
                        inverse_scale_vec, zero_point_vec);
  }
}

void GemmEpilogueAvx512(size_t row_begin, size_t row_end, size_t outputs,
                        size_t inputs, const uint8_t* quantized_input,
                        const int8_t* packed_weights,
                        const float* output_scales,
                        const int32_t* zero_point_corrections,
                        const float* biases, ActivationFunction activation,
                        float* output) {
  switch (activation) {
    case ACTIVATION_NONE:
      GemmEpilogueAvx512Impl<ACTIVATION_NONE, false>(
          row_begin, row_end, outputs, inputs, quantized_input, packed_weights,
          output_scales, zero_point_corrections, biases, activation, output,
          0.0f, 0.0f, nullptr);
      break;
    case ACTIVATION_RELU:
      GemmEpilogueAvx512Impl<ACTIVATION_RELU, false>(
          row_begin, row_end, outputs, inputs, quantized_input, packed_weights,
          output_scales, zero_point_corrections, biases, activation, output,
          0.0f, 0.0f, nullptr);
      break;
    case ACTIVATION_RELU_2:
      GemmEpilogueAvx512Impl<ACTIVATION_RELU_2, false>(
          row_begin, row_end, outputs, inputs, quantized_input, packed_weights,
          output_scales, zero_point_corrections, biases, activation, output,
          0.0f, 0.0f, nullptr);
      break;
    case ACTIVATION_MISH:
      GemmEpilogueAvx512Impl<ACTIVATION_MISH, false>(
          row_begin, row_end, outputs, inputs, quantized_input, packed_weights,
          output_scales, zero_point_corrections, biases, activation, output,
          0.0f, 0.0f, nullptr);
      break;
    case ACTIVATION_SWISH:
      GemmEpilogueAvx512Impl<ACTIVATION_SWISH, false>(
          row_begin, row_end, outputs, inputs, quantized_input, packed_weights,
          output_scales, zero_point_corrections, biases, activation, output,
          0.0f, 0.0f, nullptr);
      break;
    default:
      GemmEpilogueAvx512Impl<ACTIVATION_DEFAULT, false>(
          row_begin, row_end, outputs, inputs, quantized_input, packed_weights,
          output_scales, zero_point_corrections, biases, activation, output,
          0.0f, 0.0f, nullptr);
      break;
  }
}

void GemmEpilogueQuantizeAvx512(
    size_t row_begin, size_t row_end, size_t outputs, size_t inputs,
    const uint8_t* quantized_input, const int8_t* packed_weights,
    const float* output_scales, const int32_t* zero_point_corrections,
    const float* biases, ActivationFunction activation, float out_inverse_scale,
    float out_zero_point, uint8_t* quantized_output) {
  switch (activation) {
    case ACTIVATION_NONE:
      GemmEpilogueAvx512Impl<ACTIVATION_NONE, true>(
          row_begin, row_end, outputs, inputs, quantized_input, packed_weights,
          output_scales, zero_point_corrections, biases, activation, nullptr,
          out_inverse_scale, out_zero_point, quantized_output);
      break;
    case ACTIVATION_RELU:
      GemmEpilogueAvx512Impl<ACTIVATION_RELU, true>(
          row_begin, row_end, outputs, inputs, quantized_input, packed_weights,
          output_scales, zero_point_corrections, biases, activation, nullptr,
          out_inverse_scale, out_zero_point, quantized_output);
      break;
    case ACTIVATION_RELU_2:
      GemmEpilogueAvx512Impl<ACTIVATION_RELU_2, true>(
          row_begin, row_end, outputs, inputs, quantized_input, packed_weights,
          output_scales, zero_point_corrections, biases, activation, nullptr,
          out_inverse_scale, out_zero_point, quantized_output);
      break;
    case ACTIVATION_MISH:
      GemmEpilogueAvx512Impl<ACTIVATION_MISH, true>(
          row_begin, row_end, outputs, inputs, quantized_input, packed_weights,
          output_scales, zero_point_corrections, biases, activation, nullptr,
          out_inverse_scale, out_zero_point, quantized_output);
      break;
    case ACTIVATION_SWISH:
      GemmEpilogueAvx512Impl<ACTIVATION_SWISH, true>(
          row_begin, row_end, outputs, inputs, quantized_input, packed_weights,
          output_scales, zero_point_corrections, biases, activation, nullptr,
          out_inverse_scale, out_zero_point, quantized_output);
      break;
    default:
      GemmEpilogueAvx512Impl<ACTIVATION_DEFAULT, true>(
          row_begin, row_end, outputs, inputs, quantized_input, packed_weights,
          output_scales, zero_point_corrections, biases, activation, nullptr,
          out_inverse_scale, out_zero_point, quantized_output);
      break;
  }
}

void LayerNorm2DWithSkipAndQuantizeAvx512(
    size_t rows, size_t channels, float* data, float alpha, const float* skip,
    const float* gammas, const float* betas, float epsilon,
    uint8_t* quantized_out, float inverse_scale, float zero_point) {
  const __m512 alpha_vec = _mm512_set1_ps(alpha);
  const __m512 inv_scale_vec = _mm512_set1_ps(inverse_scale);
  const __m512 zp_vec = _mm512_set1_ps(zero_point);
  const float inv_c = 1.0f / static_cast<float>(channels);

  if (channels == 256) {
    // 256 floats fit exactly in 16 ZMM registers -> single memory pass!
    for (size_t r = 0; r < rows; ++r) {
      float* row = data + r * 256;
      __m512 v[16];
      __m512 s0 = _mm512_setzero_ps();
      __m512 s1 = _mm512_setzero_ps();
      __m512 s2 = _mm512_setzero_ps();
      __m512 s3 = _mm512_setzero_ps();

      if (skip != nullptr) {
        const float* srow = skip + r * 256;
        for (size_t k = 0; k < 16; k += 4) {
          v[k + 0] =
              _mm512_fmadd_ps(_mm512_loadu_ps(row + (k + 0) * 16), alpha_vec,
                              _mm512_loadu_ps(srow + (k + 0) * 16));
          v[k + 1] =
              _mm512_fmadd_ps(_mm512_loadu_ps(row + (k + 1) * 16), alpha_vec,
                              _mm512_loadu_ps(srow + (k + 1) * 16));
          v[k + 2] =
              _mm512_fmadd_ps(_mm512_loadu_ps(row + (k + 2) * 16), alpha_vec,
                              _mm512_loadu_ps(srow + (k + 2) * 16));
          v[k + 3] =
              _mm512_fmadd_ps(_mm512_loadu_ps(row + (k + 3) * 16), alpha_vec,
                              _mm512_loadu_ps(srow + (k + 3) * 16));
          s0 = _mm512_add_ps(s0, v[k + 0]);
          s1 = _mm512_add_ps(s1, v[k + 1]);
          s2 = _mm512_add_ps(s2, v[k + 2]);
          s3 = _mm512_add_ps(s3, v[k + 3]);
        }
      } else {
        for (size_t k = 0; k < 16; k += 4) {
          v[k + 0] =
              _mm512_mul_ps(_mm512_loadu_ps(row + (k + 0) * 16), alpha_vec);
          v[k + 1] =
              _mm512_mul_ps(_mm512_loadu_ps(row + (k + 1) * 16), alpha_vec);
          v[k + 2] =
              _mm512_mul_ps(_mm512_loadu_ps(row + (k + 2) * 16), alpha_vec);
          v[k + 3] =
              _mm512_mul_ps(_mm512_loadu_ps(row + (k + 3) * 16), alpha_vec);
          s0 = _mm512_add_ps(s0, v[k + 0]);
          s1 = _mm512_add_ps(s1, v[k + 1]);
          s2 = _mm512_add_ps(s2, v[k + 2]);
          s3 = _mm512_add_ps(s3, v[k + 3]);
        }
      }

      const float mean = _mm512_reduce_add_ps(_mm512_add_ps(
                             _mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3))) *
                         inv_c;
      const __m512 mean_vec = _mm512_set1_ps(mean);

      __m512 var0 = _mm512_setzero_ps();
      __m512 var1 = _mm512_setzero_ps();
      __m512 var2 = _mm512_setzero_ps();
      __m512 var3 = _mm512_setzero_ps();
      for (size_t k = 0; k < 16; k += 4) {
        v[k + 0] = _mm512_sub_ps(v[k + 0], mean_vec);
        v[k + 1] = _mm512_sub_ps(v[k + 1], mean_vec);
        v[k + 2] = _mm512_sub_ps(v[k + 2], mean_vec);
        v[k + 3] = _mm512_sub_ps(v[k + 3], mean_vec);
        var0 = _mm512_fmadd_ps(v[k + 0], v[k + 0], var0);
        var1 = _mm512_fmadd_ps(v[k + 1], v[k + 1], var1);
        var2 = _mm512_fmadd_ps(v[k + 2], v[k + 2], var2);
        var3 = _mm512_fmadd_ps(v[k + 3], v[k + 3], var3);
      }

      const float var =
          _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(var0, var1),
                                             _mm512_add_ps(var2, var3))) *
          inv_c;
      const float inv_std = _mm_cvtss_f32(
          _mm_div_ss(_mm_set_ss(1.0f), _mm_sqrt_ss(_mm_set_ss(var + epsilon))));
      const __m512 inv_std_vec = _mm512_set1_ps(inv_std);

      if (quantized_out != nullptr) {
        uint8_t* qrow = quantized_out + r * 256;
        for (size_t k = 0; k < 16; ++k) {
          const size_t c = k * 16;
          __m512 norm = _mm512_mul_ps(v[k], inv_std_vec);
          __m512 out = _mm512_fmadd_ps(norm, _mm512_loadu_ps(gammas + c),
                                       _mm512_loadu_ps(betas + c));
          _mm512_storeu_ps(row + c, out);
          QuantizeStore16(qrow + c, out, inv_scale_vec, zp_vec);
        }
      } else {
        for (size_t k = 0; k < 16; ++k) {
          const size_t c = k * 16;
          __m512 norm = _mm512_mul_ps(v[k], inv_std_vec);
          __m512 out = _mm512_fmadd_ps(norm, _mm512_loadu_ps(gammas + c),
                                       _mm512_loadu_ps(betas + c));
          _mm512_storeu_ps(row + c, out);
        }
      }
    }
    return;
  }

  // General AVX-512 path for arbitrary channel counts.
  const size_t full_c = channels & ~static_cast<size_t>(15);
  const size_t rem_c = channels - full_c;
  const __mmask16 rem_mask =
      rem_c ? static_cast<__mmask16>((1u << rem_c) - 1u) : 0u;

  for (size_t r = 0; r < rows; ++r) {
    float* row = data + r * channels;
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    if (skip != nullptr) {
      const float* srow = skip + r * channels;
      size_t c = 0;
      for (; c + 32 <= channels; c += 32) {
        __m512 a = _mm512_fmadd_ps(_mm512_loadu_ps(row + c), alpha_vec,
                                   _mm512_loadu_ps(srow + c));
        __m512 b = _mm512_fmadd_ps(_mm512_loadu_ps(row + c + 16), alpha_vec,
                                   _mm512_loadu_ps(srow + c + 16));
        _mm512_storeu_ps(row + c, a);
        _mm512_storeu_ps(row + c + 16, b);
        sum0 = _mm512_add_ps(sum0, a);
        sum1 = _mm512_add_ps(sum1, b);
      }
      for (; c + 16 <= channels; c += 16) {
        __m512 a = _mm512_fmadd_ps(_mm512_loadu_ps(row + c), alpha_vec,
                                   _mm512_loadu_ps(srow + c));
        _mm512_storeu_ps(row + c, a);
        sum0 = _mm512_add_ps(sum0, a);
      }
      if (rem_c) {
        __m512 a =
            _mm512_fmadd_ps(_mm512_maskz_loadu_ps(rem_mask, row + c), alpha_vec,
                            _mm512_maskz_loadu_ps(rem_mask, srow + c));
        _mm512_mask_storeu_ps(row + c, rem_mask, a);
        sum0 = _mm512_mask_add_ps(sum0, rem_mask, sum0, a);
      }
    } else {
      size_t c = 0;
      for (; c + 32 <= channels; c += 32) {
        __m512 a = _mm512_mul_ps(_mm512_loadu_ps(row + c), alpha_vec);
        __m512 b = _mm512_mul_ps(_mm512_loadu_ps(row + c + 16), alpha_vec);
        _mm512_storeu_ps(row + c, a);
        _mm512_storeu_ps(row + c + 16, b);
        sum0 = _mm512_add_ps(sum0, a);
        sum1 = _mm512_add_ps(sum1, b);
      }
      for (; c + 16 <= channels; c += 16) {
        __m512 a = _mm512_mul_ps(_mm512_loadu_ps(row + c), alpha_vec);
        _mm512_storeu_ps(row + c, a);
        sum0 = _mm512_add_ps(sum0, a);
      }
      if (rem_c) {
        __m512 a =
            _mm512_mul_ps(_mm512_maskz_loadu_ps(rem_mask, row + c), alpha_vec);
        _mm512_mask_storeu_ps(row + c, rem_mask, a);
        sum0 = _mm512_mask_add_ps(sum0, rem_mask, sum0, a);
      }
    }

    const float mean = _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1)) * inv_c;
    const __m512 mean_vec = _mm512_set1_ps(mean);

    __m512 var0 = _mm512_setzero_ps();
    __m512 var1 = _mm512_setzero_ps();
    size_t c = 0;
    for (; c + 32 <= channels; c += 32) {
      __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(row + c), mean_vec);
      __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(row + c + 16), mean_vec);
      var0 = _mm512_fmadd_ps(d0, d0, var0);
      var1 = _mm512_fmadd_ps(d1, d1, var1);
    }
    for (; c + 16 <= channels; c += 16) {
      __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(row + c), mean_vec);
      var0 = _mm512_fmadd_ps(d0, d0, var0);
    }
    if (rem_c) {
      __m512 d0 =
          _mm512_sub_ps(_mm512_maskz_loadu_ps(rem_mask, row + c), mean_vec);
      var0 = _mm512_mask3_fmadd_ps(d0, d0, var0, rem_mask);
    }

    const float var = _mm512_reduce_add_ps(_mm512_add_ps(var0, var1)) * inv_c;
    const float inv_std = _mm_cvtss_f32(
        _mm_div_ss(_mm_set_ss(1.0f), _mm_sqrt_ss(_mm_set_ss(var + epsilon))));
    const __m512 inv_std_vec = _mm512_set1_ps(inv_std);

    uint8_t* qrow = quantized_out ? (quantized_out + r * channels) : nullptr;
    c = 0;
    for (; c + 16 <= channels; c += 16) {
      __m512 d = _mm512_sub_ps(_mm512_loadu_ps(row + c), mean_vec);
      __m512 out = _mm512_fmadd_ps(_mm512_mul_ps(d, inv_std_vec),
                                   _mm512_loadu_ps(gammas + c),
                                   _mm512_loadu_ps(betas + c));
      _mm512_storeu_ps(row + c, out);
      if (qrow) QuantizeStore16(qrow + c, out, inv_scale_vec, zp_vec);
    }
    if (rem_c) {
      __m512 d =
          _mm512_sub_ps(_mm512_maskz_loadu_ps(rem_mask, row + c), mean_vec);
      __m512 out = _mm512_fmadd_ps(_mm512_mul_ps(d, inv_std_vec),
                                   _mm512_maskz_loadu_ps(rem_mask, gammas + c),
                                   _mm512_maskz_loadu_ps(rem_mask, betas + c));
      _mm512_mask_storeu_ps(row + c, rem_mask, out);
      if (qrow) {
        QuantizeStoreMasked(qrow + c, rem_mask, out, inv_scale_vec, zp_vec);
      }
    }
  }
}

void Softmax64Avx512(size_t rows, float* data) {
  for (size_t r = 0; r < rows; ++r) {
    float* row = data + r * 64;
    __m512 v0 = _mm512_loadu_ps(row + 0);
    __m512 v1 = _mm512_loadu_ps(row + 16);
    __m512 v2 = _mm512_loadu_ps(row + 32);
    __m512 v3 = _mm512_loadu_ps(row + 48);

    const __m512 max_v =
        _mm512_max_ps(_mm512_max_ps(v0, v1), _mm512_max_ps(v2, v3));
    const __m512 max_s = _mm512_set1_ps(_mm512_reduce_max_ps(max_v));

    v0 = ExpPsAvx512(_mm512_sub_ps(v0, max_s));
    v1 = ExpPsAvx512(_mm512_sub_ps(v1, max_s));
    v2 = ExpPsAvx512(_mm512_sub_ps(v2, max_s));
    v3 = ExpPsAvx512(_mm512_sub_ps(v3, max_s));

    const __m512 sum_v =
        _mm512_add_ps(_mm512_add_ps(v0, v1), _mm512_add_ps(v2, v3));
    const float sum = _mm512_reduce_add_ps(sum_v);
    const __m512 inv_sum = _mm512_set1_ps(1.0f / sum);

    _mm512_storeu_ps(row + 0, _mm512_mul_ps(v0, inv_sum));
    _mm512_storeu_ps(row + 16, _mm512_mul_ps(v1, inv_sum));
    _mm512_storeu_ps(row + 32, _mm512_mul_ps(v2, inv_sum));
    _mm512_storeu_ps(row + 48, _mm512_mul_ps(v3, inv_sum));
  }
}

void QuantizeDynamicRowsAvx512(size_t rows, size_t inputs, const float* input,
                               uint8_t* q_out, float* row_scales,
                               float* row_mins) {
  const size_t rem = inputs & 15;
  const __mmask16 rem_mask =
      rem ? static_cast<__mmask16>((1u << rem) - 1u) : 0u;

  for (size_t r = 0; r < rows; ++r) {
    const float* row = input + r * inputs;
    uint8_t* qrow = q_out + r * inputs;

    if (inputs == 256) {
      __m512 v[16];
      __m512 mn0 = _mm512_loadu_ps(row + 0);
      __m512 mx0 = mn0;
      v[0] = mn0;
      for (size_t k = 1; k < 16; ++k) {
        v[k] = _mm512_loadu_ps(row + k * 16);
        mn0 = _mm512_min_ps(mn0, v[k]);
        mx0 = _mm512_max_ps(mx0, v[k]);
      }
      const float mn = _mm512_reduce_min_ps(mn0);
      const float mx = _mm512_reduce_max_ps(mx0);
      const float range = mx - mn;
      const float scale = (range > 1e-12f) ? (range * (1.0f / 255.0f)) : 0.0f;
      const float inv_scale = (range > 1e-12f) ? (255.0f / range) : 0.0f;
      row_scales[r] = scale;
      row_mins[r] = mn;
      const __m512 inv_vec = _mm512_set1_ps(inv_scale);
      const __m512 offset_vec = _mm512_set1_ps(-mn * inv_scale);
      for (size_t k = 0; k < 16; ++k) {
        QuantizeStore16(qrow + k * 16, v[k], inv_vec, offset_vec);
      }
      continue;
    }

    __m512 mn0 = _mm512_set1_ps(row[0]);
    __m512 mx0 = mn0;
    size_t c = 0;
    for (; c + 16 <= inputs; c += 16) {
      __m512 v = _mm512_loadu_ps(row + c);
      mn0 = _mm512_min_ps(mn0, v);
      mx0 = _mm512_max_ps(mx0, v);
    }
    if (rem) {
      __m512 v = _mm512_mask_loadu_ps(mn0, rem_mask, row + c);
      mn0 = _mm512_min_ps(mn0, v);
      mx0 = _mm512_max_ps(mx0, v);
    }
    const float mn = _mm512_reduce_min_ps(mn0);
    const float mx = _mm512_reduce_max_ps(mx0);
    const float range = mx - mn;
    const float scale = (range > 1e-12f) ? (range * (1.0f / 255.0f)) : 0.0f;
    const float inv_scale = (range > 1e-12f) ? (255.0f / range) : 0.0f;
    row_scales[r] = scale;
    row_mins[r] = mn;

    const __m512 inv_vec = _mm512_set1_ps(inv_scale);
    const __m512 offset_vec = _mm512_set1_ps(-mn * inv_scale);
    c = 0;
    for (; c + 16 <= inputs; c += 16) {
      QuantizeStore16(qrow + c, _mm512_loadu_ps(row + c), inv_vec, offset_vec);
    }
    if (rem) {
      QuantizeStoreMasked(qrow + c, rem_mask,
                          _mm512_maskz_loadu_ps(rem_mask, row + c), inv_vec,
                          offset_vec);
    }
  }
}

namespace {
template <ActivationFunction Act>
void GemmDynamicEpilogueAvx512Impl(
    size_t row_begin, size_t row_end, size_t n_block_begin, size_t n_block_end,
    size_t outputs, size_t inputs, const uint8_t* quantized_input,
    const float* row_scales, const float* row_mins,
    const int8_t* packed_weights, const float* weight_scales,
    const float* weight_sum_scaled, const float* biases,
    ActivationFunction runtime_activation, float* output) {
  const size_t k_full = inputs / 4;
  const size_t k_rem = inputs & 3;
  const size_t k_groups = (inputs + 3) / 4;
  const size_t weights_block_stride = k_groups * 128;

  constexpr size_t kMc = 64;
  constexpr size_t kNBlockChunk = 4;

  for (size_t block_base = n_block_begin; block_base < n_block_end;
       block_base += kNBlockChunk) {
    const size_t block_end = MinSize(n_block_end, block_base + kNBlockChunk);
    for (size_t row_tile = row_begin; row_tile < row_end; row_tile += kMc) {
      const size_t row_tile_end = MinSize(row_end, row_tile + kMc);
      for (size_t row = row_tile; row < row_tile_end; row += 8) {
        const size_t row_count = MinSize(8, row_tile_end - row);
        for (size_t n_block = block_base; n_block < block_end; ++n_block) {
          const size_t n_offset = n_block * kNr;
          const size_t n_remaining = outputs - n_offset;
          const bool full_n = (n_remaining >= kNr);
          const __mmask16 mask0 =
              full_n ? 0xFFFF
                     : static_cast<__mmask16>(n_remaining >= 16
                                                  ? 0xFFFF
                                                  : ((1u << n_remaining) - 1u));
          const __mmask16 mask1 =
              full_n ? 0xFFFF
                     : static_cast<__mmask16>(
                           n_remaining > 16 ? ((1u << (n_remaining - 16)) - 1u)
                                            : 0u);

          const __m512 ws0 =
              full_n ? _mm512_loadu_ps(weight_scales + n_offset)
                     : _mm512_maskz_loadu_ps(mask0, weight_scales + n_offset);
          const __m512 ws1 =
              full_n
                  ? _mm512_loadu_ps(weight_scales + n_offset + 16)
                  : _mm512_maskz_loadu_ps(mask1, weight_scales + n_offset + 16);
          const __m512 wsum0 =
              full_n
                  ? _mm512_loadu_ps(weight_sum_scaled + n_offset)
                  : _mm512_maskz_loadu_ps(mask0, weight_sum_scaled + n_offset);
          const __m512 wsum1 =
              full_n ? _mm512_loadu_ps(weight_sum_scaled + n_offset + 16)
                     : _mm512_maskz_loadu_ps(mask1,
                                             weight_sum_scaled + n_offset + 16);
          const __m512 bias0 =
              biases
                  ? (full_n ? _mm512_loadu_ps(biases + n_offset)
                            : _mm512_maskz_loadu_ps(mask0, biases + n_offset))
                  : _mm512_setzero_ps();
          const __m512 bias1 =
              biases
                  ? (full_n
                         ? _mm512_loadu_ps(biases + n_offset + 16)
                         : _mm512_maskz_loadu_ps(mask1, biases + n_offset + 16))
                  : _mm512_setzero_ps();

          const int8_t* weights_ptr =
              packed_weights + n_block * weights_block_stride;
          __m512i acc[8][2];
          for (size_t r = 0; r < row_count; ++r) {
            acc[r][0] = _mm512_setzero_si512();
            acc[r][1] = _mm512_setzero_si512();
          }

          if (row_count == 8) {
            Microkernel8x32Accum(quantized_input + row * inputs, inputs,
                                 weights_ptr, k_full, k_rem, acc);
          } else {
            for (size_t k_group = 0; k_group < k_full; ++k_group) {
              const __m512i b0 = _mm512_load_si512(weights_ptr);
              const __m512i b1 = _mm512_load_si512(weights_ptr + 64);
              const size_t offset = k_group * 4;
              for (size_t r = 0; r < row_count; ++r) {
                const __m512i av = _mm512_set1_epi32(
                    LoadU32(quantized_input + (row + r) * inputs + offset));
                acc[r][0] = _mm512_dpbusd_epi32(acc[r][0], av, b0);
                acc[r][1] = _mm512_dpbusd_epi32(acc[r][1], av, b1);
              }
              weights_ptr += 128;
            }
            if (k_rem > 0) {
              const __m512i b0 = _mm512_load_si512(weights_ptr);
              const __m512i b1 = _mm512_load_si512(weights_ptr + 64);
              const size_t offset = k_full * 4;
              for (size_t r = 0; r < row_count; ++r) {
                const __m512i av = _mm512_set1_epi32(LoadTailU32(
                    quantized_input + (row + r) * inputs + offset, k_rem));
                acc[r][0] = _mm512_dpbusd_epi32(acc[r][0], av, b0);
                acc[r][1] = _mm512_dpbusd_epi32(acc[r][1], av, b1);
              }
            }
          }

          for (size_t r = 0; r < row_count; ++r) {
            const __m512 rs = _mm512_set1_ps(row_scales[row + r]);
            const __m512 rm = _mm512_set1_ps(row_mins[row + r]);
            __m512 v0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[r][0]),
                                        _mm512_mul_ps(rs, ws0),
                                        _mm512_fmadd_ps(rm, wsum0, bias0));
            __m512 v1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[r][1]),
                                        _mm512_mul_ps(rs, ws1),
                                        _mm512_fmadd_ps(rm, wsum1, bias1));
            v0 = ApplyActivationStatic<Act>(v0, runtime_activation);
            v1 = ApplyActivationStatic<Act>(v1, runtime_activation);
            float* destination = output + (row + r) * outputs + n_offset;
            if (full_n) {
              _mm512_storeu_ps(destination, v0);
              _mm512_storeu_ps(destination + 16, v1);
            } else {
              _mm512_mask_storeu_ps(destination, mask0, v0);
              if (mask1) _mm512_mask_storeu_ps(destination + 16, mask1, v1);
            }
          }
        }
      }
    }
  }
}
}  // namespace

void GemmDynamicEpilogueAvx512(
    size_t row_begin, size_t row_end, size_t n_block_begin, size_t n_block_end,
    size_t outputs, size_t inputs, const uint8_t* quantized_input,
    const float* row_scales, const float* row_mins,
    const int8_t* packed_weights, const float* weight_scales,
    const float* weight_sum_scaled, const float* biases,
    ActivationFunction activation, float* output) {
  switch (activation) {
    case ACTIVATION_NONE:
      GemmDynamicEpilogueAvx512Impl<ACTIVATION_NONE>(
          row_begin, row_end, n_block_begin, n_block_end, outputs, inputs,
          quantized_input, row_scales, row_mins, packed_weights, weight_scales,
          weight_sum_scaled, biases, activation, output);
      break;
    case ACTIVATION_RELU:
      GemmDynamicEpilogueAvx512Impl<ACTIVATION_RELU>(
          row_begin, row_end, n_block_begin, n_block_end, outputs, inputs,
          quantized_input, row_scales, row_mins, packed_weights, weight_scales,
          weight_sum_scaled, biases, activation, output);
      break;
    case ACTIVATION_RELU_2:
      GemmDynamicEpilogueAvx512Impl<ACTIVATION_RELU_2>(
          row_begin, row_end, n_block_begin, n_block_end, outputs, inputs,
          quantized_input, row_scales, row_mins, packed_weights, weight_scales,
          weight_sum_scaled, biases, activation, output);
      break;
    case ACTIVATION_MISH:
      GemmDynamicEpilogueAvx512Impl<ACTIVATION_MISH>(
          row_begin, row_end, n_block_begin, n_block_end, outputs, inputs,
          quantized_input, row_scales, row_mins, packed_weights, weight_scales,
          weight_sum_scaled, biases, activation, output);
      break;
    case ACTIVATION_SWISH:
      GemmDynamicEpilogueAvx512Impl<ACTIVATION_SWISH>(
          row_begin, row_end, n_block_begin, n_block_end, outputs, inputs,
          quantized_input, row_scales, row_mins, packed_weights, weight_scales,
          weight_sum_scaled, biases, activation, output);
      break;
    default:
      GemmDynamicEpilogueAvx512Impl<ACTIVATION_DEFAULT>(
          row_begin, row_end, n_block_begin, n_block_end, outputs, inputs,
          quantized_input, row_scales, row_mins, packed_weights, weight_scales,
          weight_sum_scaled, biases, activation, output);
      break;
  }
}

// Specialized zero-spill 8x48 residual block path for `channels % 48 == 0`
// (e.g. T79 C=192).
template <ActivationFunction Act>
void ResidualBlockBoardsAvx512Nr48Impl(
    size_t num_boards, size_t channels, float* boards_io,
    const int8_t* conv1_packed_weights, const float* conv1_output_scales,
    const int32_t* conv1_zero_point_corrections, const float* conv1_biases,
    float conv1_inverse_scale, uint8_t conv1_zero_point,
    const int8_t* conv2_packed_weights, const float* conv2_output_scales,
    const int32_t* conv2_zero_point_corrections, const float* conv2_biases,
    float conv2_inverse_scale, uint8_t conv2_zero_point,
    ActivationFunction runtime_activation, const ResidualSeParams& se,
    const Conv3x3ScratchView& scratch) {
  constexpr size_t kNr48 = 48;
  const size_t channel_groups = channels / 4;
  const size_t padded_channels = channels;
  const size_t n_blocks = channels / kNr48;
  const size_t tap_stride = channel_groups * 192;
  const size_t io_board_stride = kSquares * channels;

  uint8_t* pad1 = scratch.pad1;
  uint8_t* pad2 = scratch.pad2;
  const __m512 conv2_inv_vec = _mm512_set1_ps(conv2_inverse_scale);
  const __m512 conv2_zp_vec =
      _mm512_set1_ps(static_cast<float>(conv2_zero_point));

  for (size_t b = 0; b < num_boards; ++b) {
    float* board_io = boards_io + b * io_board_stride;

    QuantizeBoardToPad(channels, padded_channels, board_io, conv1_inverse_scale,
                       conv1_zero_point, pad1);
    FillPadBorder(pad2, padded_channels, conv2_zero_point);

    // Conv1 with fused bias, activation, and requantization into pad2.
    for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
      const size_t n_offset = n_block * kNr48;
      const __m512i zp0 =
          _mm512_loadu_si512(conv1_zero_point_corrections + n_offset);
      const __m512i zp1 =
          _mm512_loadu_si512(conv1_zero_point_corrections + n_offset + 16);
      const __m512i zp2 =
          _mm512_loadu_si512(conv1_zero_point_corrections + n_offset + 32);
      const __m512 s0 = _mm512_loadu_ps(conv1_output_scales + n_offset);
      const __m512 s1 = _mm512_loadu_ps(conv1_output_scales + n_offset + 16);
      const __m512 s2 = _mm512_loadu_ps(conv1_output_scales + n_offset + 32);
      const __m512 b0 = _mm512_loadu_ps(conv1_biases + n_offset);
      const __m512 b1 = _mm512_loadu_ps(conv1_biases + n_offset + 16);
      const __m512 b2 = _mm512_loadu_ps(conv1_biases + n_offset + 32);

      const int8_t* weights_block =
          conv1_packed_weights + n_block * 9 * tap_stride;

      for (size_t y = 0; y < kBoardSize; ++y) {
        __m512i acc[8][3];
        Microkernel8x48Conv3x3(pad1 + y * kPadSize * padded_channels,
                               padded_channels, weights_block, channel_groups,
                               acc);

        uint8_t* destination_row =
            pad2 + ((y + 1) * kPadSize + 1) * padded_channels + n_offset;
        for (size_t x = 0; x < kBoardSize; ++x) {
          __m512 v0 = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][0], zp0)), s0, b0);
          __m512 v1 = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][1], zp1)), s1, b1);
          __m512 v2 = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][2], zp2)), s2, b2);
          v0 = ApplyActivationStatic<Act>(v0, runtime_activation);
          v1 = ApplyActivationStatic<Act>(v1, runtime_activation);
          v2 = ApplyActivationStatic<Act>(v2, runtime_activation);
          uint8_t* destination = destination_row + x * padded_channels;
          QuantizeStore16(destination, v0, conv2_inv_vec, conv2_zp_vec);
          QuantizeStore16(destination + 16, v1, conv2_inv_vec, conv2_zp_vec);
          QuantizeStore16(destination + 32, v2, conv2_inv_vec, conv2_zp_vec);
        }
      }
    }

    // Conv2 with optional fused global average pooling and SE unit.
    if (se.has_se) {
      float* conv2_output = scratch.conv2_output;
      float* avg_pool = scratch.avg_pool;
      const __m512 inv64 = _mm512_set1_ps(1.0f / 64.0f);

      for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
        const size_t n_offset = n_block * kNr48;
        const __m512i zp0 =
            _mm512_loadu_si512(conv2_zero_point_corrections + n_offset);
        const __m512i zp1 =
            _mm512_loadu_si512(conv2_zero_point_corrections + n_offset + 16);
        const __m512i zp2 =
            _mm512_loadu_si512(conv2_zero_point_corrections + n_offset + 32);
        const __m512 s0 = _mm512_loadu_ps(conv2_output_scales + n_offset);
        const __m512 s1 = _mm512_loadu_ps(conv2_output_scales + n_offset + 16);
        const __m512 s2 = _mm512_loadu_ps(conv2_output_scales + n_offset + 32);
        const __m512 b0 = _mm512_loadu_ps(conv2_biases + n_offset);
        const __m512 b1 = _mm512_loadu_ps(conv2_biases + n_offset + 16);
        const __m512 b2 = _mm512_loadu_ps(conv2_biases + n_offset + 32);

        const int8_t* weights_block =
            conv2_packed_weights + n_block * 9 * tap_stride;
        __m512 pool0 = _mm512_setzero_ps();
        __m512 pool1 = _mm512_setzero_ps();
        __m512 pool2 = _mm512_setzero_ps();

        for (size_t y = 0; y < kBoardSize; ++y) {
          __m512i acc[8][3];
          Microkernel8x48Conv3x3(pad2 + y * kPadSize * padded_channels,
                                 padded_channels, weights_block, channel_groups,
                                 acc);

          for (size_t x = 0; x < kBoardSize; ++x) {
            __m512 v0 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][0], zp0)), s0, b0);
            __m512 v1 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][1], zp1)), s1, b1);
            __m512 v2 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][2], zp2)), s2, b2);
            float* destination =
                conv2_output + (y * kBoardSize + x) * channels + n_offset;
            _mm512_storeu_ps(destination, v0);
            _mm512_storeu_ps(destination + 16, v1);
            _mm512_storeu_ps(destination + 32, v2);
            pool0 = _mm512_add_ps(pool0, v0);
            pool1 = _mm512_add_ps(pool1, v1);
            pool2 = _mm512_add_ps(pool2, v2);
          }
        }

        _mm512_storeu_ps(avg_pool + n_offset, _mm512_mul_ps(pool0, inv64));
        _mm512_storeu_ps(avg_pool + n_offset + 16, _mm512_mul_ps(pool1, inv64));
        _mm512_storeu_ps(avg_pool + n_offset + 32, _mm512_mul_ps(pool2, inv64));
      }

      float* fc1 = scratch.se_fc1;
      for (size_t o = 0; o < se.se_fc_outputs; ++o) {
        const float value =
            DotProductAvx512(se.w1 + o * channels, avg_pool, channels) +
            se.b1[o];
        fc1[o] = Activate(value, runtime_activation);
      }
      float* fc2 = scratch.se_fc2;
      for (size_t o = 0; o < 2 * channels; ++o) {
        fc2[o] = DotProductAvx512(se.w2 + o * se.se_fc_outputs, fc1,
                                  se.se_fc_outputs) +
                 se.b2[o];
      }
      for (size_t c = 0; c < channels; c += 16) {
        _mm512_storeu_ps(fc2 + c, SigmoidPsAvx512(_mm512_loadu_ps(fc2 + c)));
      }
      const float* gamma = fc2;
      const float* beta = fc2 + channels;
      for (size_t square = 0; square < kSquares; ++square) {
        const float* in_row = conv2_output + square * channels;
        float* io_row = board_io + square * channels;
        for (size_t c = 0; c < channels; c += 16) {
          const __m512 g = _mm512_loadu_ps(gamma + c);
          const __m512 bt = _mm512_loadu_ps(beta + c);
          const __m512 in_v = _mm512_loadu_ps(in_row + c);
          const __m512 skip_v = _mm512_loadu_ps(io_row + c);
          const __m512 v = _mm512_fmadd_ps(g, in_v, _mm512_add_ps(skip_v, bt));
          _mm512_storeu_ps(io_row + c,
                           ApplyActivationStatic<Act>(v, runtime_activation));
        }
      }
    } else {
      for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
        const size_t n_offset = n_block * kNr48;
        const __m512i zp0 =
            _mm512_loadu_si512(conv2_zero_point_corrections + n_offset);
        const __m512i zp1 =
            _mm512_loadu_si512(conv2_zero_point_corrections + n_offset + 16);
        const __m512i zp2 =
            _mm512_loadu_si512(conv2_zero_point_corrections + n_offset + 32);
        const __m512 s0 = _mm512_loadu_ps(conv2_output_scales + n_offset);
        const __m512 s1 = _mm512_loadu_ps(conv2_output_scales + n_offset + 16);
        const __m512 s2 = _mm512_loadu_ps(conv2_output_scales + n_offset + 32);
        const __m512 b0 = _mm512_loadu_ps(conv2_biases + n_offset);
        const __m512 b1 = _mm512_loadu_ps(conv2_biases + n_offset + 16);
        const __m512 b2 = _mm512_loadu_ps(conv2_biases + n_offset + 32);

        const int8_t* weights_block =
            conv2_packed_weights + n_block * 9 * tap_stride;

        for (size_t y = 0; y < kBoardSize; ++y) {
          __m512i acc[8][3];
          Microkernel8x48Conv3x3(pad2 + y * kPadSize * padded_channels,
                                 padded_channels, weights_block, channel_groups,
                                 acc);

          for (size_t x = 0; x < kBoardSize; ++x) {
            float* io = board_io + (y * kBoardSize + x) * channels + n_offset;
            __m512 v0 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][0], zp0)), s0, b0);
            __m512 v1 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][1], zp1)), s1, b1);
            __m512 v2 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][2], zp2)), s2, b2);
            v0 = _mm512_add_ps(v0, _mm512_loadu_ps(io));
            v1 = _mm512_add_ps(v1, _mm512_loadu_ps(io + 16));
            v2 = _mm512_add_ps(v2, _mm512_loadu_ps(io + 32));
            _mm512_storeu_ps(
                io, ApplyActivationStatic<Act>(v0, runtime_activation));
            _mm512_storeu_ps(
                io + 16, ApplyActivationStatic<Act>(v1, runtime_activation));
            _mm512_storeu_ps(
                io + 32, ApplyActivationStatic<Act>(v2, runtime_activation));
          }
        }
      }
    }
  }
}

void ResidualBlockBoardsAvx512(
    size_t num_boards, size_t channels, float* boards_io,
    const int8_t* conv1_packed_weights, const float* conv1_output_scales,
    const int32_t* conv1_zero_point_corrections, const float* conv1_biases,
    float conv1_inverse_scale, uint8_t conv1_zero_point,
    const int8_t* conv2_packed_weights, const float* conv2_output_scales,
    const int32_t* conv2_zero_point_corrections, const float* conv2_biases,
    float conv2_inverse_scale, uint8_t conv2_zero_point,
    ActivationFunction activation, const ResidualSeParams& se,
    const Conv3x3ScratchView& scratch, size_t conv_nr) {
  if (conv_nr == 48 && (channels % 48) == 0) {
    switch (activation) {
      case ACTIVATION_RELU:
        ResidualBlockBoardsAvx512Nr48Impl<ACTIVATION_RELU>(
            num_boards, channels, boards_io, conv1_packed_weights,
            conv1_output_scales, conv1_zero_point_corrections, conv1_biases,
            conv1_inverse_scale, conv1_zero_point, conv2_packed_weights,
            conv2_output_scales, conv2_zero_point_corrections, conv2_biases,
            conv2_inverse_scale, conv2_zero_point, activation, se, scratch);
        return;
      case ACTIVATION_MISH:
        ResidualBlockBoardsAvx512Nr48Impl<ACTIVATION_MISH>(
            num_boards, channels, boards_io, conv1_packed_weights,
            conv1_output_scales, conv1_zero_point_corrections, conv1_biases,
            conv1_inverse_scale, conv1_zero_point, conv2_packed_weights,
            conv2_output_scales, conv2_zero_point_corrections, conv2_biases,
            conv2_inverse_scale, conv2_zero_point, activation, se, scratch);
        return;
      case ACTIVATION_NONE:
        ResidualBlockBoardsAvx512Nr48Impl<ACTIVATION_NONE>(
            num_boards, channels, boards_io, conv1_packed_weights,
            conv1_output_scales, conv1_zero_point_corrections, conv1_biases,
            conv1_inverse_scale, conv1_zero_point, conv2_packed_weights,
            conv2_output_scales, conv2_zero_point_corrections, conv2_biases,
            conv2_inverse_scale, conv2_zero_point, activation, se, scratch);
        return;
      default:
        ResidualBlockBoardsAvx512Nr48Impl<ACTIVATION_DEFAULT>(
            num_boards, channels, boards_io, conv1_packed_weights,
            conv1_output_scales, conv1_zero_point_corrections, conv1_biases,
            conv1_inverse_scale, conv1_zero_point, conv2_packed_weights,
            conv2_output_scales, conv2_zero_point_corrections, conv2_biases,
            conv2_inverse_scale, conv2_zero_point, activation, se, scratch);
        return;
    }
  }
  switch (activation) {
    case ACTIVATION_RELU:
      ResidualBlockBoardsAvx512Impl<ACTIVATION_RELU>(
          num_boards, channels, boards_io, conv1_packed_weights,
          conv1_output_scales, conv1_zero_point_corrections, conv1_biases,
          conv1_inverse_scale, conv1_zero_point, conv2_packed_weights,
          conv2_output_scales, conv2_zero_point_corrections, conv2_biases,
          conv2_inverse_scale, conv2_zero_point, activation, se, scratch);
      break;
    case ACTIVATION_MISH:
      ResidualBlockBoardsAvx512Impl<ACTIVATION_MISH>(
          num_boards, channels, boards_io, conv1_packed_weights,
          conv1_output_scales, conv1_zero_point_corrections, conv1_biases,
          conv1_inverse_scale, conv1_zero_point, conv2_packed_weights,
          conv2_output_scales, conv2_zero_point_corrections, conv2_biases,
          conv2_inverse_scale, conv2_zero_point, activation, se, scratch);
      break;
    case ACTIVATION_NONE:
      ResidualBlockBoardsAvx512Impl<ACTIVATION_NONE>(
          num_boards, channels, boards_io, conv1_packed_weights,
          conv1_output_scales, conv1_zero_point_corrections, conv1_biases,
          conv1_inverse_scale, conv1_zero_point, conv2_packed_weights,
          conv2_output_scales, conv2_zero_point_corrections, conv2_biases,
          conv2_inverse_scale, conv2_zero_point, activation, se, scratch);
      break;
    default:
      ResidualBlockBoardsAvx512Impl<ACTIVATION_DEFAULT>(
          num_boards, channels, boards_io, conv1_packed_weights,
          conv1_output_scales, conv1_zero_point_corrections, conv1_biases,
          conv1_inverse_scale, conv1_zero_point, conv2_packed_weights,
          conv2_output_scales, conv2_zero_point_corrections, conv2_biases,
          conv2_inverse_scale, conv2_zero_point, activation, se, scratch);
      break;
  }
}

void ResidualBlockBoardAvx512(
    size_t channels, float* board_io, const int8_t* conv1_packed_weights,
    const float* conv1_output_scales,
    const int32_t* conv1_zero_point_corrections, const float* conv1_biases,
    float conv1_inverse_scale, uint8_t conv1_zero_point,
    const int8_t* conv2_packed_weights, const float* conv2_output_scales,
    const int32_t* conv2_zero_point_corrections, const float* conv2_biases,
    float conv2_inverse_scale, uint8_t conv2_zero_point,
    ActivationFunction activation, const ResidualSeParams& se,
    const Conv3x3ScratchView& scratch) {
  ResidualBlockBoardsAvx512(
      1, channels, board_io, conv1_packed_weights, conv1_output_scales,
      conv1_zero_point_corrections, conv1_biases, conv1_inverse_scale,
      conv1_zero_point, conv2_packed_weights, conv2_output_scales,
      conv2_zero_point_corrections, conv2_biases, conv2_inverse_scale,
      conv2_zero_point, activation, se, scratch);
}

void Conv3x3NhwcBoardAvx512(
    size_t input_channels, size_t output_channels, const float* input_nhwc,
    const int8_t* packed_weights, const float* output_scales,
    const int32_t* zero_point_corrections, float inverse_scale,
    uint8_t zero_point, float* output_nhwc, const Conv3x3ScratchView& scratch,
    const float* biases, ActivationFunction activation) {
  const size_t channel_groups = (input_channels + 3) / 4;
  const size_t padded_channels = channel_groups * 4;
  const size_t n_blocks = (output_channels + kNr - 1) / kNr;
  const size_t tap_stride = channel_groups * 128;

  uint8_t* pad1 = scratch.pad1;
  QuantizeBoardToPad(input_channels, padded_channels, input_nhwc, inverse_scale,
                     zero_point, pad1);

  for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
    const size_t n_offset = n_block * kNr;
    const size_t n_remaining = output_channels - n_offset;
    const bool full_n = (n_remaining >= kNr);
    const __mmask16 mask0 =
        full_n ? 0xFFFF
               : static_cast<__mmask16>(
                     n_remaining >= 16 ? 0xFFFF : ((1u << n_remaining) - 1u));
    const __mmask16 mask1 =
        full_n ? 0xFFFF
               : static_cast<__mmask16>(
                     n_remaining > 16 ? ((1u << (n_remaining - 16)) - 1u) : 0u);

    const __m512i zp_corr0 =
        full_n ? _mm512_loadu_si512(zero_point_corrections + n_offset)
               : _mm512_maskz_loadu_epi32(mask0,
                                          zero_point_corrections + n_offset);
    const __m512i zp_corr1 =
        full_n ? _mm512_loadu_si512(zero_point_corrections + n_offset + 16)
               : _mm512_maskz_loadu_epi32(
                     mask1, zero_point_corrections + n_offset + 16);
    const __m512 scale0 =
        full_n ? _mm512_loadu_ps(output_scales + n_offset)
               : _mm512_maskz_loadu_ps(mask0, output_scales + n_offset);
    const __m512 scale1 =
        full_n ? _mm512_loadu_ps(output_scales + n_offset + 16)
               : _mm512_maskz_loadu_ps(mask1, output_scales + n_offset + 16);
    const __m512 bias0 =
        biases ? (full_n ? _mm512_loadu_ps(biases + n_offset)
                         : _mm512_maskz_loadu_ps(mask0, biases + n_offset))
               : _mm512_setzero_ps();
    const __m512 bias1 =
        biases ? (full_n ? _mm512_loadu_ps(biases + n_offset + 16)
                         : _mm512_maskz_loadu_ps(mask1, biases + n_offset + 16))
               : _mm512_setzero_ps();

    const int8_t* weights_block = packed_weights + n_block * 9 * tap_stride;

    for (size_t y = 0; y < kBoardSize; ++y) {
      __m512i acc[8][2];
      Microkernel8x32Conv3x3(pad1 + y * kPadSize * padded_channels,
                             padded_channels, weights_block, channel_groups,
                             acc);

      for (size_t x = 0; x < kBoardSize; ++x) {
        __m512 v0 = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][0], zp_corr0)), scale0,
            bias0);
        __m512 v1 = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[x][1], zp_corr1)), scale1,
            bias1);
        if (activation != ACTIVATION_NONE) {
          v0 = ApplyActivationVec(v0, activation);
          v1 = ApplyActivationVec(v1, activation);
        }
        float* destination =
            output_nhwc + (y * kBoardSize + x) * output_channels + n_offset;
        if (full_n) {
          _mm512_storeu_ps(destination, v0);
          _mm512_storeu_ps(destination + 16, v1);
        } else {
          _mm512_mask_storeu_ps(destination, mask0, v0);
          if (mask1) _mm512_mask_storeu_ps(destination + 16, mask1, v1);
        }
      }
    }
  }
}

void NchwToNhwcAvx512(size_t batch_size, size_t channels, const float* input,
                      float* output) {
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const float* source = input + batch * channels * kSquares;
    float* destination = output + batch * kSquares * channels;
    size_t channel = 0;
    for (; channel + 8 <= channels; channel += 8) {
      for (size_t square = 0; square < kSquares; square += 8) {
        Transpose8x8(source + channel * kSquares + square, kSquares,
                     destination + square * channels + channel, channels);
      }
    }
    for (; channel < channels; ++channel) {
      for (size_t square = 0; square < kSquares; ++square) {
        destination[square * channels + channel] =
            source[channel * kSquares + square];
      }
    }
  }
}

void NhwcToNchwAvx512(size_t batch_size, size_t channels, const float* input,
                      float* output) {
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const float* source = input + batch * kSquares * channels;
    float* destination = output + batch * channels * kSquares;
    size_t channel = 0;
    for (; channel + 8 <= channels; channel += 8) {
      for (size_t square = 0; square < kSquares; square += 8) {
        Transpose8x8(source + square * channels + channel, channels,
                     destination + channel * kSquares + square, kSquares);
      }
    }
    for (; channel < channels; ++channel) {
      for (size_t square = 0; square < kSquares; ++square) {
        destination[channel * kSquares + square] =
            source[square * channels + channel];
      }
    }
  }
}

}  // namespace blas_backend
}  // namespace lczero
