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

#pragma once

// Minimal POD-only header included by ISA-specific translation units
// (int8_vnni_avx512.cc, int8_vnni_avx.cc). Intentionally includes no STL
// headers or inline/template definitions so per-file /arch or -m flags cannot
// emit VEX/EVEX instructions into shared COMDAT functions across the binary.

#include <cstddef>
#include <cstdint>

#include "neural/tables/activation_function.h"

namespace lczero {

// Declared here without pulling in activation.h (<vector>) for the rare
// fallback activation path inside ISA translation units.
float Activate(float val, ActivationFunction activation);

namespace blas_backend {

// Parameters for fused residual-block SE unit (NHWC).
struct ResidualSeParams {
  bool has_se = false;
  size_t se_fc_outputs = 0;
  const float* w1 = nullptr;  // [se_fc_outputs][channels] row-major
  const float* b1 = nullptr;
  const float* w2 = nullptr;  // [2 * channels][se_fc_outputs] row-major
  const float* b2 = nullptr;
  const float* w1_t = nullptr;  // optional [channels][se_fc_outputs]
  const float* w2_t = nullptr;  // optional [se_fc_outputs][2 * channels]
};

// Pre-sized raw scratch pointers passed into the ISA residual-block kernels so
// those translation units never allocate or instantiate AlignedBuffer methods.
struct Conv3x3ScratchView {
  uint8_t* pad1 = nullptr;        // 10 * 10 * padded_channels
  uint8_t* pad2 = nullptr;        // 10 * 10 * padded_channels
  float* conv2_output = nullptr;  // 64 * channels
  float* avg_pool = nullptr;      // channels
  float* se_fc1 = nullptr;        // se_fc_outputs
  float* se_fc2 = nullptr;        // 2 * channels
  float* nhwc_board = nullptr;    // optional 64 * input_channels staging
};

#ifdef LC0_HAS_AVX512_VNNI
void QuantizeToUint8Avx512(size_t count, const float* input,
                           float inverse_scale, float zero_point,
                           uint8_t* output);

void GemmEpilogueAvx512(size_t row_begin, size_t row_end, size_t outputs,
                        size_t inputs, const uint8_t* quantized_input,
                        const int8_t* packed_weights,
                        const float* output_scales,
                        const int32_t* zero_point_corrections,
                        const float* biases, ActivationFunction activation,
                        float* output);

void GemmEpilogueQuantizeAvx512(
    size_t row_begin, size_t row_end, size_t outputs, size_t inputs,
    const uint8_t* quantized_input, const int8_t* packed_weights,
    const float* output_scales, const int32_t* zero_point_corrections,
    const float* biases, ActivationFunction activation, float out_inverse_scale,
    float out_zero_point, uint8_t* quantized_output);

// Single-pass AVX-512 LayerNorm with optional skip connection and optional
// fused uint8 quantization into `quantized_out` (pass nullptr to skip), plus
// optional dynamic per-row uint8 quantization into `dyn_quantized_out`.
void LayerNorm2DWithSkipAndQuantizeAvx512(
    size_t rows, size_t channels, float* data, float alpha, const float* skip,
    const float* gammas, const float* betas, float epsilon,
    uint8_t* quantized_out, float inverse_scale, float zero_point,
    uint8_t* dyn_quantized_out = nullptr, float* dyn_scales = nullptr,
    float* dyn_mins = nullptr);

// Single-pass AVX-512 LayerNorm (alpha=1, no skip, read-only input) fused
// directly with per-group dynamic uint8 quantization (used by Smolgen LN1/LN2).
void LayerNormAndDynamicQuantizeAvx512(size_t rows, size_t channels,
                                       size_t quant_group_size,
                                       const float* input, const float* gammas,
                                       const float* betas, float epsilon,
                                       uint8_t* q_out, float* group_scales,
                                       float* group_mins);

// In-place AVX-512 Softmax over `rows` rows of exactly 64 floats each.
void Softmax64Avx512(size_t rows, float* data);

// Per-row dynamic asymmetric uint8 quantization: x[r][k] ~= row_mins[r] +
// row_scales[r] * q_out[r][k].
void QuantizeDynamicRowsAvx512(size_t rows, size_t inputs, const float* input,
                               uint8_t* q_out, float* row_scales,
                               float* row_mins);

// Dynamic per-row W8A8 GEMM over [row_begin, row_end) x [n_block_begin,
// n_block_end) (in 32-column blocks). Computes:
//   y[r][o] = act((row_scales[r] * weight_scales[o]) * acc[r][o]
//                 + row_mins[r] * weight_sum_scaled[o] + biases[o]).
void GemmDynamicEpilogueAvx512(
    size_t row_begin, size_t row_end, size_t n_block_begin, size_t n_block_end,
    size_t outputs, size_t inputs, const uint8_t* quantized_input,
    const float* row_scales, const float* row_mins,
    const int8_t* packed_weights, const float* weight_scales,
    const float* weight_sum_scaled, const float* biases,
    ActivationFunction activation, float* output);

void ResidualBlockBoardsAvx512(
    size_t num_boards, size_t channels, float* boards_io,
    const int8_t* conv1_packed_weights, const float* conv1_output_scales,
    const int32_t* conv1_zero_point_corrections, const float* conv1_biases,
    float conv1_inverse_scale, uint8_t conv1_zero_point,
    const int8_t* conv2_packed_weights, const float* conv2_output_scales,
    const int32_t* conv2_zero_point_corrections, const float* conv2_biases,
    float conv2_inverse_scale, uint8_t conv2_zero_point,
    ActivationFunction activation, const ResidualSeParams& se,
    const Conv3x3ScratchView& scratch, size_t conv_nr = 32);

void ResidualBlockBoardAvx512(
    size_t channels, float* board_io, const int8_t* conv1_packed_weights,
    const float* conv1_output_scales,
    const int32_t* conv1_zero_point_corrections, const float* conv1_biases,
    float conv1_inverse_scale, uint8_t conv1_zero_point,
    const int8_t* conv2_packed_weights, const float* conv2_output_scales,
    const int32_t* conv2_zero_point_corrections, const float* conv2_biases,
    float conv2_inverse_scale, uint8_t conv2_zero_point,
    ActivationFunction activation, const ResidualSeParams& se,
    const Conv3x3ScratchView& scratch);

void Conv3x3NhwcBoardAvx512(
    size_t input_channels, size_t output_channels, const float* input_nhwc,
    const int8_t* packed_weights, const float* output_scales,
    const int32_t* zero_point_corrections, float inverse_scale,
    uint8_t zero_point, float* output_nhwc, const Conv3x3ScratchView& scratch,
    const float* biases = nullptr,
    ActivationFunction activation = ACTIVATION_NONE, size_t conv_nr = 32,
    bool input_is_nchw = false, float plane109_scale = 1.0f);

void GemmFp32Avx512(size_t row_begin, size_t row_end, size_t col_begin,
                    size_t col_end, size_t outputs, size_t inputs,
                    const float* input, const float* weights_transposed,
                    const float* biases, ActivationFunction activation,
                    float* output);

void NchwToNhwcAvx512(size_t batch_size, size_t channels, const float* input,
                      float* output);
void NhwcToNchwAvx512(size_t batch_size, size_t channels, const float* input,
                      float* output);

void PrepAttentionInputPeMapAvx512(size_t batch_size, size_t input_channels,
                                   size_t pe_channels, const float* input_nchw,
                                   const float* pos_encoding,
                                   float* output_nhwc);

void ApplyInputGatingAvx512(size_t batch_size, size_t embedding_size,
                            float* flow_nhwc, const float* mult_nhwc,
                            const float* add_nhwc);

void AttentionPolicyPromoAvx512(size_t policy_d_model, const float* k_board,
                                const float* ip4_pol_w, float* logits_board);

// Computes one attention head (head_idx, depth == 32) on a single 64-square
// board: attn_h = Softmax(Q_h * K_h^T * scaling [+ smolgen_bias]) * V_h.
// `attn_board` may alias `q_board` because heads write disjoint 128-byte
// slices after reading their Q row tile.
void MhaAttentionHead32Avx512(size_t d_model, float scaling,
                              const float* q_board, const float* k_board,
                              const float* v_board, const float* smolgen_bias,
                              size_t head_idx, float* attn_board);
#endif

#ifdef LC0_HAS_AVX_VNNI
void QuantizeToUint8Avx(size_t count, const float* input, float inverse_scale,
                        float zero_point, uint8_t* output);

void GemmEpilogueAvx(size_t row_begin, size_t row_end, size_t outputs,
                     size_t inputs, const uint8_t* quantized_input,
                     const int8_t* packed_weights, const float* output_scales,
                     const int32_t* zero_point_corrections, const float* biases,
                     ActivationFunction activation, float* output);

void ResidualBlockBoardAvx(
    size_t channels, float* board_io, const int8_t* conv1_packed_weights,
    const float* conv1_output_scales,
    const int32_t* conv1_zero_point_corrections, const float* conv1_biases,
    float conv1_inverse_scale, uint8_t conv1_zero_point,
    const int8_t* conv2_packed_weights, const float* conv2_output_scales,
    const int32_t* conv2_zero_point_corrections, const float* conv2_biases,
    float conv2_inverse_scale, uint8_t conv2_zero_point,
    ActivationFunction activation, const ResidualSeParams& se,
    const Conv3x3ScratchView& scratch);

void Conv3x3NhwcBoardAvx(size_t input_channels, size_t output_channels,
                         const float* input_nhwc, const int8_t* packed_weights,
                         const float* output_scales,
                         const int32_t* zero_point_corrections,
                         float inverse_scale, uint8_t zero_point,
                         float* output_nhwc, const Conv3x3ScratchView& scratch);
#endif

}  // namespace blas_backend
}  // namespace lczero
