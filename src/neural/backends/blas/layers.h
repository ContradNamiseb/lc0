/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018-2026 The LCZero Authors

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
#include <memory>
#include <vector>

#include "neural/backends/blas/winograd_convolution3.h"
#include "neural/backends/shared/activation.h"
#include "neural/network_legacy.h"

namespace lczero {
namespace blas_backend {

constexpr int kSquares = 64;
constexpr size_t kNumOutputPolicy = 1858;

// Grows a buffer to at least `size` elements. Buffers never shrink, so a
// workspace reaches its final size after the first full batch.
void GrowBuffer(std::vector<float>& buffer, size_t size);

// Memory one computation owns while it runs the network. The network keeps
// a pool of these, so the buffers are allocated once and reused. Layers hold
// only weights; the tensors a layer reads and writes are passed to Eval, and
// a layer uses these buffers for its temporaries.
template <bool use_eigen>
class Workspace {
 public:
  Workspace(size_t max_convolution_input_channels,
            size_t max_convolution_output_channels);

  // The Winograd convolution scratch, big enough for `batch_size`.
  WinogradConvolution3<use_eigen>& Winograd(size_t batch_size);

  // The tensors the network passes between layers; the first holds the
  // encoded planes.
  std::vector<float> tensors[3];
  // Scratch for the layers' temporaries.
  std::vector<float> buffers[4];

 private:
  const size_t max_convolution_input_channels_;
  const size_t max_convolution_output_channels_;
  size_t winograd_batch_size_ = 0;
  std::unique_ptr<WinogradConvolution3<use_eigen>> winograd_;
};

// One stage of the network. Layers in the residual tower use NCHW tensors
// (batch, channels, 64 squares); layers in the attention body use NHWC
// tensors (batch, 64 squares, channels).
//
// Layers keep references into the network's MultiHeadWeights, which outlive
// them. Weights the backend transforms, such as Winograd filters, are
// transformed there once before the layers are built.
template <bool use_eigen>
class BaseLayer {
 public:
  explicit BaseLayer(size_t output_size) : output_size_(output_size) {}
  virtual ~BaseLayer() = default;

  // Number of floats this layer writes per sample.
  size_t GetOutputSize() const { return output_size_; }

  // `input2` is an optional second input (the planes for the attention body,
  // unused elsewhere).
  virtual void Eval(size_t batch_size, float* output, const float* input,
                    const float* input2,
                    Workspace<use_eigen>& workspace) const = 0;

 private:
  const size_t output_size_;
};

// A 3x3 convolution (Winograd) or 1x1 convolution, followed by bias and
// activation. NCHW in and out.
template <bool use_eigen>
class ConvLayer : public BaseLayer<use_eigen> {
 public:
  ConvLayer(const MultiHeadWeights::ConvBlock& weights, size_t input_channels,
            size_t filter_size, ActivationFunction activation);

  void Eval(size_t batch_size, float* output, const float* input,
            const float* input2,
            Workspace<use_eigen>& workspace) const override;

 private:
  const MultiHeadWeights::ConvBlock& weights_;
  const size_t input_channels_;
  const size_t output_channels_;
  const size_t filter_size_;
  const ActivationFunction activation_;
};

// Two 3x3 convolutions, the optional squeeze-and-excitation unit and the skip
// connection. NCHW in and out; `output` may be the same tensor as `input`.
template <bool use_eigen>
class ResidualBlock : public BaseLayer<use_eigen> {
 public:
  ResidualBlock(const MultiHeadWeights::Residual& weights, size_t channels,
                ActivationFunction activation);

  void Eval(size_t batch_size, float* output, const float* input,
            const float* input2,
            Workspace<use_eigen>& workspace) const override;

 private:
  const MultiHeadWeights::Residual& weights_;
  const size_t channels_;
  const ActivationFunction activation_;
};

// A fully connected layer over each sample's whole input.
template <bool use_eigen>
class DenseLayer : public BaseLayer<use_eigen> {
 public:
  DenseLayer(const std::vector<float>& weights,
             const std::vector<float>& biases, size_t input_size,
             ActivationFunction activation);

  void Eval(size_t batch_size, float* output, const float* input,
            const float* input2,
            Workspace<use_eigen>& workspace) const override;

 private:
  const std::vector<float>& weights_;
  const std::vector<float>& biases_;
  const size_t input_size_;
  const ActivationFunction activation_;
};

// A fully connected layer applied to every square of an NHWC tensor.
template <bool use_eigen>
class EmbeddingLayer : public BaseLayer<use_eigen> {
 public:
  EmbeddingLayer(const std::vector<float>& weights,
                 const std::vector<float>& biases, size_t input_channels,
                 ActivationFunction activation);

  void Eval(size_t batch_size, float* output, const float* input,
            const float* input2,
            Workspace<use_eigen>& workspace) const override;

 private:
  const std::vector<float>& weights_;
  const std::vector<float>& biases_;
  const size_t input_channels_;
  const ActivationFunction activation_;
};

// Picks the 1858 policy outputs out of the head's raw output, using one of
// the policy map tables (-1 marks an unused entry).
template <bool use_eigen>
class PolicyMapLayer : public BaseLayer<use_eigen> {
 public:
  PolicyMapLayer(const short* map, size_t used_inputs, size_t input_stride);

  void Eval(size_t batch_size, float* output, const float* input,
            const float* input2,
            Workspace<use_eigen>& workspace) const override;

 private:
  const short* map_;
  const size_t used_inputs_;
  const size_t input_stride_;
};

// One transformer encoder layer (smolgen, multi-head attention, feed-forward
// network, two layer norms). Not a BaseLayer: it runs in place on the
// attention body's or policy head's NHWC tensor.
template <bool use_eigen>
class EncoderBlock {
 public:
  EncoderBlock(const MultiHeadWeights::EncoderLayer& weights,
               const std::vector<float>& smolgen_global_weights,
               size_t embedding_size, size_t heads,
               ActivationFunction smolgen_activation,
               ActivationFunction ffn_activation, float alpha,
               float default_epsilon);

  // The result is left in `flow`. The other buffers are scratch; the vectors
  // may be swapped with `flow`.
  void Eval(size_t batch_size, std::vector<float>& flow,
            std::vector<float>& buffer2, std::vector<float>& buffer3,
            std::vector<float>& buffer4) const;

 private:
  const MultiHeadWeights::EncoderLayer& weights_;
  const std::vector<float>& smolgen_global_weights_;
  const size_t embedding_size_;
  const size_t heads_;
  const ActivationFunction smolgen_activation_;
  const ActivationFunction ffn_activation_;
  const float alpha_;
  const float default_epsilon_;
};

// The whole attention body: input preprocessing and position encoding, the
// input embedding, gating, the embedding feed-forward network and the encoder
// stack. Reads the planes (NCHW) or a residual tower's output (NCHW), writes
// NHWC.
template <bool use_eigen>
class AttentionBody : public BaseLayer<use_eigen> {
 public:
  AttentionBody(const MultiHeadWeights& weights, size_t input_channels,
                bool has_residual_tower, bool is_pe_dense_embedding,
                ActivationFunction default_activation,
                ActivationFunction smolgen_activation,
                ActivationFunction ffn_activation);

  void Eval(size_t batch_size, float* output, const float* input,
            const float* input2,
            Workspace<use_eigen>& workspace) const override;

 private:
  const MultiHeadWeights& weights_;
  const size_t input_channels_;
  const size_t embedding_size_;
  const size_t position_encoding_channels_;
  const bool has_residual_tower_;
  const bool is_pe_dense_embedding_;
  const ActivationFunction default_activation_;
  const ActivationFunction ffn_activation_;
  const float alpha_;
  std::vector<EncoderBlock<use_eigen>> encoders_;
};

// The attention policy head up to the 64*64 + 8*24 raw move logits: the
// policy embedding, the policy encoders, Q and K, and the promotion offsets.
// Reads NCHW from a residual tower or NHWC from an attention body.
template <bool use_eigen>
class AttentionPolicyHead : public BaseLayer<use_eigen> {
 public:
  AttentionPolicyHead(const MultiHeadWeights::PolicyHead& weights,
                      const std::vector<float>& smolgen_global_weights,
                      size_t input_channels, bool attention_body,
                      ActivationFunction default_activation,
                      ActivationFunction smolgen_activation,
                      ActivationFunction ffn_activation);

  void Eval(size_t batch_size, float* output, const float* input,
            const float* input2,
            Workspace<use_eigen>& workspace) const override;

 private:
  const MultiHeadWeights::PolicyHead& weights_;
  const size_t input_channels_;
  const bool attention_body_;
  const ActivationFunction default_activation_;
  std::vector<EncoderBlock<use_eigen>> encoders_;
};

// The value head: a 1x1 convolution (residual tower) or an embedding
// (attention body), two fully connected layers, then softmax for WDL or tanh.
// Writes 3 floats per sample for WDL, 1 otherwise.
template <bool use_eigen>
class ValueHead : public BaseLayer<use_eigen> {
 public:
  ValueHead(const MultiHeadWeights::ValueHead& weights, size_t input_channels,
            bool attention_body, bool wdl,
            ActivationFunction default_activation);

  void Eval(size_t batch_size, float* output, const float* input,
            const float* input2,
            Workspace<use_eigen>& workspace) const override;

 private:
  const MultiHeadWeights::ValueHead& weights_;
  const size_t input_channels_;
  const bool attention_body_;
  const bool wdl_;
  const ActivationFunction default_activation_;
};

}  // namespace blas_backend
}  // namespace lczero
