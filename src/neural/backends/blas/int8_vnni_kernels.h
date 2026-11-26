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
    uint8_t zero_point, float* output_nhwc, const Conv3x3ScratchView& scratch);

void NchwToNhwcAvx512(size_t batch_size, size_t channels, const float* input,
                      float* output);
void NhwcToNchwAvx512(size_t batch_size, size_t channels, const float* input,
                      float* output);
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
