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

#include "neural/backends/blas/int8_vnni.h"
#include "neural/backends/shared/activation.h"

namespace lczero {
namespace blas_backend {

// Whether this build has MKL's integer matrix multiply (cblas_gemm_s8u8s32).
#ifdef USE_MKL
constexpr bool kMklInt8Available = true;
#else
constexpr bool kMklInt8Available = false;
#endif

// Backward-compatible compile-time/runtime helper: true if MKL or VNNI can run.
bool IsInt8SupportedOnHost(bool allow_mkl);

// Temporaries of the INT8 layers, kept in the computation's workspace.
struct Int8Scratch {
  AlignedBuffer<uint8_t> quantized_input;
  // The 3x3 convolution's input (MKL im2col fallback), one square's
  // neighbourhood per row.
  std::vector<uint8_t> columns;
  std::vector<int32_t> accumulators;
  // NHWC staging buffers for single-layer NCHW Convolve3 on VNNI builds.
  AlignedBuffer<float> nhwc_input;
  AlignedBuffer<float> nhwc_output;
  // Per-thread scratch for direct 9-tap NHWC VNNI convolutions.
  std::vector<Conv3x3ThreadScratch> conv3_scratch;
  // Non-owning pointer to the network's INT8 thread pool (nullptr for 1
  // thread).
  Int8ThreadPool* pool = nullptr;
};

// Fast NCHW <-> NHWC conversion helpers (uses AVX-512 8x8 tile transpose when
// available and parallelizes across boards via `pool`).
void NchwToNhwcFast(size_t batch_size, size_t channels, const float* input,
                    float* output, Int8ThreadPool* pool = nullptr);
void NhwcToNchwFast(size_t batch_size, size_t channels, const float* input,
                    float* output, Int8ThreadPool* pool = nullptr);

// One W8A8 layer: int8 weights with one scale per output channel, and the
// scale and zero point that quantize the layer's input to uint8. Computes
//   y[o] = output_scales[o] * (sum_k q[k] * w[o][k] - zero_point * sum_k
//   w[o][k])
// with q = clamp(rint(x / input_scale + zero_point), 0, 255), which equals
// input_scale * weight_scale[o] * sum_k (q[k] - zero_point) * w[o][k].
class Int8Weights {
 public:
  // `weights` is row-major [outputs][inputs] and must outlive this object.
  // When `is_conv3x3` is true and a VNNI kernel is selected, packs the 3x3
  // weights for direct 9-tap NHWC convolution instead of GEMM.
  Int8Weights(const std::vector<int8_t>& weights,
              const std::vector<float>& weight_scales, float input_scale,
              uint8_t input_zero_point, Int8Kernel kernel = Int8Kernel::kMkl,
              bool is_conv3x3 = false);

  size_t inputs() const { return inputs_; }
  size_t outputs() const { return outputs_; }
  Int8Kernel kernel() const { return kernel_; }
  bool uses_vnni() const {
    return kernel_ == Int8Kernel::kVnniAvx512 ||
           kernel_ == Int8Kernel::kVnniAvx;
  }

  // Quantizes `count` floats of this layer's input.
  void QuantizeInput(size_t count, const float* input, uint8_t* output,
                     Int8ThreadPool* pool = nullptr) const;

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

  // A 3x3 convolution on an NHWC tensor [batch_size][64][in_c] -> [64][out_c].
  void Convolve3Nhwc(size_t batch_size, const float* input_nhwc,
                     float* output_nhwc, Int8Scratch& scratch) const;

  // Runs a complete fused NHWC residual block (conv1 -> bias1 + act ->
  // requant -> conv2 -> optional SE + skip + act) in-place on `tensor_nhwc`.
  static void RunResidualBlockNhwc(size_t batch_size, size_t channels,
                                   float* tensor_nhwc, const Int8Weights& conv1,
                                   const float* conv1_biases,
                                   const Int8Weights& conv2,
                                   const float* conv2_biases,
                                   ActivationFunction activation,
                                   const ResidualSeParams& se,
                                   Int8Scratch& scratch);

 private:
  // accumulators[rows][outputs] = quantized_input[rows][inputs] * weights^T.
  void MatrixMultiply(size_t rows, const uint8_t* quantized_input,
                      int32_t* accumulators) const;

  const int8_t* weights_;
  size_t inputs_;
  size_t outputs_;
  Int8Kernel kernel_;
  AlignedBuffer<int8_t> packed_weights_;
  std::vector<float> output_scales_;
  std::vector<int32_t> zero_point_corrections_;
  float inverse_input_scale_;
  uint8_t input_zero_point_;
};

}  // namespace blas_backend
}  // namespace lczero
