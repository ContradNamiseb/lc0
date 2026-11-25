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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "neural/backends/shared/activation.h"

namespace lczero {
namespace blas_backend {

// Whether this build can run INT8 layers. They need MKL's integer matrix
// multiply (cblas_gemm_s8u8s32); other builds run the dequantized weights.
#ifdef USE_MKL
constexpr bool kInt8Available = true;
#else
constexpr bool kInt8Available = false;
#endif

// Temporaries of the INT8 layers, kept in the computation's workspace.
struct Int8Scratch {
  std::vector<uint8_t> quantized_input;
  // The 3x3 convolution's input, one square's neighbourhood per row.
  std::vector<uint8_t> columns;
  std::vector<int32_t> accumulators;
};

// One W8A8 layer: int8 weights with one scale per output channel, and the
// scale and zero point that quantize the layer's input to uint8. Computes
//   y[o] = output_scales[o] * (sum_k q[k] * w[o][k] - zero_point * sum_k
//   w[o][k])
// with q = clamp(rint(x / input_scale + zero_point), 0, 255), which equals
// input_scale * weight_scale[o] * sum_k (q[k] - zero_point) * w[o][k].
class Int8Weights {
 public:
  // `weights` is row-major [outputs][inputs] and must outlive this object.
  Int8Weights(const std::vector<int8_t>& weights,
              const std::vector<float>& weight_scales, float input_scale,
              uint8_t input_zero_point);

  size_t inputs() const { return inputs_; }
  size_t outputs() const { return outputs_; }

  // Quantizes `count` floats of this layer's input.
  void QuantizeInput(size_t count, const float* input, uint8_t* output) const;

  // output[r][o] = activation(y[r][o] + biases[o]) for `rows` rows of an
  // already quantized input [rows][inputs].
  void Forward(size_t rows, const uint8_t* quantized_input, const float* biases,
               ActivationFunction activation, float* output,
               Int8Scratch& scratch) const;

  // Quantizes `input` [rows][inputs] and runs Forward().
  void Forward(size_t rows, const float* input, const float* biases,
               ActivationFunction activation, float* output,
               Int8Scratch& scratch) const;

  // A 3x3 convolution of an NCHW tensor with these weights ([outputs]
  // [inputs / 9][3][3]), without bias or activation, like
  // WinogradConvolution3::Forward().
  void Convolve3(size_t batch_size, const float* input, float* output,
                 Int8Scratch& scratch) const;

 private:
  // accumulators[rows][outputs] = quantized_input[rows][inputs] * weights^T.
  void MatrixMultiply(size_t rows, const uint8_t* quantized_input,
                      int32_t* accumulators) const;

  const int8_t* weights_;
  size_t inputs_;
  size_t outputs_;
  std::vector<float> output_scales_;
  std::vector<int32_t> zero_point_corrections_;
  float inverse_input_scale_;
  uint8_t input_zero_point_;
};

}  // namespace blas_backend
}  // namespace lczero
