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

#include "neural/backends/blas/int8_layers.h"

#include <algorithm>
#include <array>

#include "neural/backends/blas/blas.h"
#include "neural/backends/blas/layers.h"
#include "utils/exception.h"

namespace lczero {
namespace blas_backend {
namespace {

constexpr int kBoardSize = 8;
constexpr int kFilterTaps = 9;

// For each square, the squares its 3x3 filter reads, in filter order; -1
// marks a tap off the board, which reads the zero point (the quantized 0.0).
constexpr std::array<std::array<int, kFilterTaps>, kSquares> MakeNeighbours() {
  std::array<std::array<int, kFilterTaps>, kSquares> neighbours{};
  for (int y = 0; y < kBoardSize; y++) {
    for (int x = 0; x < kBoardSize; x++) {
      for (int dy = 0; dy < 3; dy++) {
        for (int dx = 0; dx < 3; dx++) {
          const int row = y + dy - 1;
          const int column = x + dx - 1;
          const bool on_board = row >= 0 && row < kBoardSize && column >= 0 &&
                                column < kBoardSize;
          neighbours[y * kBoardSize + x][dy * 3 + dx] =
              on_board ? row * kBoardSize + column : -1;
        }
      }
    }
  }
  return neighbours;
}
constexpr auto kNeighbours = MakeNeighbours();

// Adding and subtracting 1.5 * 2^23 rounds a float in [0, 2^22) to the
// nearest integer, ties to even, like rint(), in a form compilers vectorize.
constexpr float kRoundingConstant = 12582912.0f;

}  // namespace

Int8Weights::Int8Weights(const std::vector<int8_t>& weights,
                         const std::vector<float>& weight_scales,
                         float input_scale, uint8_t input_zero_point)
    : weights_(weights.data()),
      outputs_(weight_scales.size()),
      inverse_input_scale_(1.0f / input_scale),
      input_zero_point_(input_zero_point) {
  if (outputs_ == 0 || weights.size() % outputs_ != 0) {
    throw Exception("INT8 weights don't match their scales.");
  }
  inputs_ = weights.size() / outputs_;
  output_scales_.resize(outputs_);
  zero_point_corrections_.resize(outputs_);
  for (size_t output = 0; output < outputs_; output++) {
    int32_t weight_sum = 0;
    for (size_t input = 0; input < inputs_; input++) {
      weight_sum += weights_[output * inputs_ + input];
    }
    zero_point_corrections_[output] =
        static_cast<int32_t>(input_zero_point) * weight_sum;
    output_scales_[output] = input_scale * weight_scales[output];
  }
}

void Int8Weights::QuantizeInput(size_t count, const float* input,
                                uint8_t* output) const {
  const float inverse_scale = inverse_input_scale_;
  const float zero_point = input_zero_point_;
  for (size_t i = 0; i < count; i++) {
    const float value =
        std::min(std::max(input[i] * inverse_scale + zero_point, 0.0f), 255.0f);
    output[i] = static_cast<uint8_t>(
        static_cast<int>((value + kRoundingConstant) - kRoundingConstant));
  }
}

void Int8Weights::MatrixMultiply(size_t rows, const uint8_t* quantized_input,
                                 int32_t* accumulators) const {
#ifdef USE_MKL
  // MKL's first matrix is the unsigned one, the second the signed one.
  const MKL_INT32 no_offset = 0;
  cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset,
                     static_cast<MKL_INT>(rows), static_cast<MKL_INT>(outputs_),
                     static_cast<MKL_INT>(inputs_), 1.0f, quantized_input,
                     static_cast<MKL_INT>(inputs_), 0, weights_,
                     static_cast<MKL_INT>(inputs_), 0, 0.0f, accumulators,
                     static_cast<MKL_INT>(outputs_), &no_offset);
#else
  (void)rows;
  (void)quantized_input;
  (void)accumulators;
  throw Exception("INT8 layers need a blas build with MKL.");
#endif
}

void Int8Weights::Forward(size_t rows, const uint8_t* quantized_input,
                          const float* biases, ActivationFunction activation,
                          float* output, Int8Scratch& scratch) const {
  GrowBuffer(scratch.accumulators, rows * outputs_);
  MatrixMultiply(rows, quantized_input, scratch.accumulators.data());
  for (size_t row = 0; row < rows; row++) {
    const int32_t* accumulators = &scratch.accumulators[row * outputs_];
    float* values = output + row * outputs_;
    for (size_t i = 0; i < outputs_; i++) {
      values[i] =
          output_scales_[i] *
          static_cast<float>(accumulators[i] - zero_point_corrections_[i]);
    }
    Activate(outputs_, values, biases, values, activation);
  }
}

void Int8Weights::Forward(size_t rows, const float* input, const float* biases,
                          ActivationFunction activation, float* output,
                          Int8Scratch& scratch) const {
  GrowBuffer(scratch.quantized_input, rows * inputs_);
  QuantizeInput(rows * inputs_, input, scratch.quantized_input.data());
  Forward(rows, scratch.quantized_input.data(), biases, activation, output,
          scratch);
}

void Int8Weights::Convolve3(size_t batch_size, const float* input,
                            float* output, Int8Scratch& scratch) const {
  const size_t channels = inputs_ / kFilterTaps;
  const size_t rows = batch_size * kSquares;

  // Quantize the input, then lay out each square's 3x3 neighbourhood in all
  // channels as one row (im2col), so the convolution is one matrix multiply.
  GrowBuffer(scratch.quantized_input, batch_size * channels * kSquares);
  QuantizeInput(batch_size * channels * kSquares, input,
                scratch.quantized_input.data());
  GrowBuffer(scratch.columns, rows * inputs_);
  for (size_t batch = 0; batch < batch_size; batch++) {
    const uint8_t* planes =
        &scratch.quantized_input[batch * channels * kSquares];
    for (int square = 0; square < kSquares; square++) {
      uint8_t* row = &scratch.columns[(batch * kSquares + square) * inputs_];
      const auto& neighbours = kNeighbours[square];
      for (size_t channel = 0; channel < channels; channel++) {
        const uint8_t* plane = planes + channel * kSquares;
        uint8_t* taps = row + channel * kFilterTaps;
        for (int tap = 0; tap < kFilterTaps; tap++) {
          const int source = neighbours[tap];
          taps[tap] = source >= 0 ? plane[source] : input_zero_point_;
        }
      }
    }
  }

  GrowBuffer(scratch.accumulators, rows * outputs_);
  MatrixMultiply(rows, scratch.columns.data(), scratch.accumulators.data());

  // The products come out as [square][channel]; write them back as NCHW.
  for (size_t batch = 0; batch < batch_size; batch++) {
    const int32_t* accumulators =
        &scratch.accumulators[batch * kSquares * outputs_];
    for (size_t channel = 0; channel < outputs_; channel++) {
      float* plane = output + (batch * outputs_ + channel) * kSquares;
      const float scale = output_scales_[channel];
      const int32_t correction = zero_point_corrections_[channel];
      for (int square = 0; square < kSquares; square++) {
        plane[square] =
            scale * static_cast<float>(
                        accumulators[square * outputs_ + channel] - correction);
      }
    }
  }
}

}  // namespace blas_backend
}  // namespace lczero
