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

#include "neural/backends/blas/layers.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

#include "neural/backends/blas/blas.h"
#include "neural/backends/blas/convolution1.h"
#include "neural/backends/blas/encoder.h"
#include "neural/backends/blas/fully_connected_layer.h"
#include "neural/backends/blas/se_unit.h"
#include "neural/tables/attention_policy_map.h"
#include "utils/exception.h"

#ifdef USE_ISPC
#include "activation_ispc.h"
#endif

namespace lczero {
namespace blas_backend {
namespace {

template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenStridedMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0,
               Eigen::OuterStride<>>;
template <typename T>
using ConstEigenStridedMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0,
               Eigen::OuterStride<>>;

// The 64*64 move logits plus 8*24 promotion logits of the attention policy.
constexpr size_t kAttentionPolicyOutputs = 64 * 64 + 8 * 24;

// Copies an NCHW tensor into NHWC order.
void NchwToNhwc(size_t batch_size, size_t channels, const float* input,
                float* output) {
  for (size_t batch = 0; batch < batch_size; batch++) {
    for (size_t square = 0; square < kSquares; square++) {
      for (size_t channel = 0; channel < channels; channel++) {
        output[batch * kSquares * channels + square * channels + channel] =
            input[batch * kSquares * channels + channel * kSquares + square];
      }
    }
  }
}

}  // namespace

template <bool use_eigen>
Workspace<use_eigen>::Workspace(size_t max_convolution_input_channels,
                                size_t max_convolution_output_channels)
    : max_convolution_input_channels_(max_convolution_input_channels),
      max_convolution_output_channels_(max_convolution_output_channels) {}

template <bool use_eigen>
WinogradConvolution3<use_eigen>& Workspace<use_eigen>::Winograd(
    size_t batch_size) {
  if (batch_size > winograd_batch_size_) {
    winograd_ = std::make_unique<WinogradConvolution3<use_eigen>>(
        batch_size, max_convolution_input_channels_,
        max_convolution_output_channels_);
    winograd_batch_size_ = batch_size;
  }
  return *winograd_;
}

template <bool use_eigen>
ConvLayer<use_eigen>::ConvLayer(const MultiHeadWeights::ConvBlock& weights,
                                size_t input_channels, size_t filter_size,
                                ActivationFunction activation)
    : BaseLayer<use_eigen>(weights.biases.size() * kSquares),
      weights_(weights),
      input_channels_(input_channels),
      output_channels_(weights.biases.size()),
      filter_size_(filter_size),
      activation_(activation) {
  assert(filter_size == 1 || filter_size == 3);
}

template <bool use_eigen>
void ConvLayer<use_eigen>::Eval(size_t batch_size, float* output,
                                const float* input, const float* /*input2*/,
                                Workspace<use_eigen>& workspace) const {
  if (filter_size_ == 3) {
    workspace.Winograd(batch_size)
        .Forward(batch_size, input_channels_, output_channels_, input,
                 weights_.weights.data(), output);
  } else {
    Convolution1<use_eigen>::Forward(batch_size, input_channels_,
                                     output_channels_, input,
                                     weights_.weights.data(), output);
  }
  BiasActivate(batch_size, output_channels_, output, weights_.biases.data(),
               activation_);
}

template <bool use_eigen>
ResidualBlock<use_eigen>::ResidualBlock(
    const MultiHeadWeights::Residual& weights, size_t channels,
    ActivationFunction activation, bool use_int8)
    : BaseLayer<use_eigen>(channels * kSquares),
      weights_(weights),
      channels_(channels),
      activation_(activation) {
  if (!use_int8) return;
  const auto& conv1 = weights.conv1;
  const auto& conv2 = weights.conv2;
  if (conv1.has_int8) {
    conv1_int8_.emplace(conv1.weights_i8, conv1.w_scales, conv1.act_scale,
                        conv1.act_zp);
  }
  if (conv2.has_int8) {
    conv2_int8_.emplace(conv2.weights_i8, conv2.w_scales, conv2.act_scale,
                        conv2.act_zp);
  }
}

template <bool use_eigen>
void ResidualBlock<use_eigen>::Eval(size_t batch_size, float* output,
                                    const float* input, const float* /*input2*/,
                                    Workspace<use_eigen>& workspace) const {
  const size_t tensor_size = batch_size * channels_ * kSquares;
  // The skip connection is added into `output` in place.
  if (output != input) std::memcpy(output, input, tensor_size * sizeof(float));
  std::vector<float>& conv1_output = workspace.buffers[0];
  std::vector<float>& conv2_output = workspace.buffers[1];
  GrowBuffer(conv1_output, tensor_size);
  GrowBuffer(conv2_output, tensor_size);
  auto& winograd = workspace.Winograd(batch_size);

  const auto& conv1 = weights_.conv1;
  const auto& conv2 = weights_.conv2;
  if (conv1_int8_) {
    conv1_int8_->Convolve3(batch_size, output, conv1_output.data(),
                           workspace.int8);
  } else {
    winograd.Forward(batch_size, channels_, channels_, output,
                     conv1.weights.data(), conv1_output.data());
  }
  BiasActivate(batch_size, channels_, conv1_output.data(), conv1.biases.data(),
               activation_);
  if (conv2_int8_) {
    conv2_int8_->Convolve3(batch_size, conv1_output.data(), conv2_output.data(),
                           workspace.int8);
  } else {
    winograd.Forward(batch_size, channels_, channels_, conv1_output.data(),
                     conv2.weights.data(), conv2_output.data());
  }

  if (weights_.has_se) {
    // No activation here: the SE unit adds the bias and the skip connection
    // and activates.
    const auto& se = weights_.se;
    ApplySEUnit<use_eigen>(batch_size, channels_, se.b1.size(),
                           conv2_output.data(), conv2.biases.data(), output,
                           se.w1.data(), se.b1.data(), se.w2.data(),
                           se.b2.data(), output, activation_);
  } else {
    BiasResidual(batch_size, channels_, output, conv2.biases.data(),
                 conv2_output.data(), activation_);
  }
}

template <bool use_eigen>
DenseLayer<use_eigen>::DenseLayer(const std::vector<float>& weights,
                                  const std::vector<float>& biases,
                                  size_t input_size,
                                  ActivationFunction activation)
    : BaseLayer<use_eigen>(biases.size()),
      weights_(weights),
      biases_(biases),
      input_size_(input_size),
      activation_(activation) {}

template <bool use_eigen>
void DenseLayer<use_eigen>::Eval(size_t batch_size, float* output,
                                 const float* input, const float* /*input2*/,
                                 Workspace<use_eigen>& /*workspace*/) const {
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size, input_size_, biases_.size(), input, weights_.data(),
      biases_.data(), activation_, output);
}

template <bool use_eigen>
EmbeddingLayer<use_eigen>::EmbeddingLayer(const std::vector<float>& weights,
                                          const std::vector<float>& biases,
                                          size_t input_channels,
                                          ActivationFunction activation)
    : BaseLayer<use_eigen>(biases.size() * kSquares),
      weights_(weights),
      biases_(biases),
      input_channels_(input_channels),
      activation_(activation) {}

template <bool use_eigen>
void EmbeddingLayer<use_eigen>::Eval(
    size_t batch_size, float* output, const float* input,
    const float* /*input2*/, Workspace<use_eigen>& /*workspace*/) const {
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, input_channels_, biases_.size(), input,
      weights_.data(), biases_.data(), activation_, output);
}

template <bool use_eigen>
PolicyMapLayer<use_eigen>::PolicyMapLayer(const short* map, size_t used_inputs,
                                          size_t input_stride)
    : BaseLayer<use_eigen>(kNumOutputPolicy),
      map_(map),
      used_inputs_(used_inputs),
      input_stride_(input_stride) {}

template <bool use_eigen>
void PolicyMapLayer<use_eigen>::Eval(
    size_t batch_size, float* output, const float* input,
    const float* /*input2*/, Workspace<use_eigen>& /*workspace*/) const {
  for (size_t batch = 0; batch < batch_size; batch++) {
    float* policy = output + batch * kNumOutputPolicy;
    const float* logits = input + batch * input_stride_;
    std::fill(policy, policy + kNumOutputPolicy, 0.0f);
    for (size_t i = 0; i < used_inputs_; i++) {
      const auto j = map_[i];
      if (j >= 0) policy[j] = logits[i];
    }
  }
}

template <bool use_eigen>
EncoderBlock<use_eigen>::EncoderBlock(
    const MultiHeadWeights::EncoderLayer& weights,
    const std::vector<float>& smolgen_global_weights, size_t embedding_size,
    size_t heads, ActivationFunction smolgen_activation,
    ActivationFunction ffn_activation, float alpha, float default_epsilon,
    bool use_int8)
    : weights_(weights),
      smolgen_global_weights_(smolgen_global_weights),
      embedding_size_(embedding_size),
      heads_(heads),
      smolgen_activation_(smolgen_activation),
      ffn_activation_(ffn_activation),
      alpha_(alpha),
      default_epsilon_(default_epsilon) {
  if (!use_int8) return;
  const auto& mha = weights.mha;
  if (mha.has_int8) {
    q_int8_.emplace(mha.q_w_i8, mha.q_w_scales, mha.qkv_act_scale,
                    mha.qkv_act_zp);
    k_int8_.emplace(mha.k_w_i8, mha.k_w_scales, mha.qkv_act_scale,
                    mha.qkv_act_zp);
    v_int8_.emplace(mha.v_w_i8, mha.v_w_scales, mha.qkv_act_scale,
                    mha.qkv_act_zp);
    dense_int8_.emplace(mha.dense_w_i8, mha.dense_w_scales, mha.dense_act_scale,
                        mha.dense_act_zp);
  }
  const auto& ffn = weights.ffn;
  if (ffn.has_int8) {
    ffn_dense1_int8_.emplace(ffn.dense1_w_i8, ffn.dense1_w_scales,
                             ffn.dense1_act_scale, ffn.dense1_act_zp);
    ffn_dense2_int8_.emplace(ffn.dense2_w_i8, ffn.dense2_w_scales,
                             ffn.dense2_act_scale, ffn.dense2_act_zp);
  }
}

template <bool use_eigen>
void EncoderBlock<use_eigen>::Eval(size_t batch_size, std::vector<float>& flow,
                                   std::vector<float>& buffer2,
                                   std::vector<float>& buffer3,
                                   std::vector<float>& buffer4,
                                   Int8Scratch& int8_scratch) const {
  const auto& mha = weights_.mha;
  const auto& ffn = weights_.ffn;
  const size_t d_model = mha.q_b.size();
  const size_t dff_size = ffn.dense1_b.size();
  const size_t hidden_channels =
      mha.has_smolgen ? mha.smolgen.compress.size() / embedding_size_ : 0;
  const size_t hidden_size = mha.has_smolgen ? mha.smolgen.dense1_b.size() : 0;
  const size_t generated_size =
      mha.has_smolgen ? mha.smolgen.dense2_b.size() : 0;
  const size_t heads = heads_;

  GrowBuffer(flow, batch_size * d_model * kSquares);
  GrowBuffer(buffer2, batch_size * std::max(std::max(d_model, hidden_channels) *
                                                kSquares,
                                            generated_size));
  GrowBuffer(buffer3, batch_size * std::max(d_model * kSquares, hidden_size));
  GrowBuffer(buffer4,
             batch_size * kSquares * std::max(kSquares * heads, dff_size));

  // Smolgen.
  if (mha.has_smolgen) {
    float* qk = buffer4.data();

    // Compress.
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size * kSquares, embedding_size_, hidden_channels, flow.data(),
        mha.smolgen.compress.data(), (const float*)nullptr, ACTIVATION_NONE,
        buffer2.data());

    // Dense 1.
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size, kSquares * hidden_channels, hidden_size, buffer2.data(),
        mha.smolgen.dense1_w.data(), mha.smolgen.dense1_b.data(),
        smolgen_activation_, buffer3.data());
    // Layer Norm.
    LayerNorm2DWithSkipConnection(
        batch_size, hidden_size, buffer3.data(), 1.0f, (const float*)nullptr,
        mha.smolgen.ln1_gammas.data(), mha.smolgen.ln1_betas.data(), 1e-3f);

    // Dense 2.
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size, hidden_size, generated_size, buffer3.data(),
        mha.smolgen.dense2_w.data(), mha.smolgen.dense2_b.data(),
        smolgen_activation_, buffer2.data());
    // Layer Norm.
    LayerNorm2DWithSkipConnection(
        batch_size, generated_size, buffer2.data(), 1.0f, (const float*)nullptr,
        mha.smolgen.ln2_gammas.data(), mha.smolgen.ln2_betas.data(), 1e-3f);

    // Global smolgen weights.
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size * heads, generated_size / heads, kSquares * kSquares,
        buffer2.data(), smolgen_global_weights_.data(), (const float*)nullptr,
        ACTIVATION_NONE, qk);
  }

  // Q
  const size_t rows = batch_size * kSquares;
  if (q_int8_) {
    q_int8_->Forward(rows, flow.data(), mha.q_b.data(), ACTIVATION_NONE,
                     buffer2.data(), int8_scratch);
  } else {
    FullyConnectedLayer<use_eigen>::Forward1D(
        rows, embedding_size_, d_model, flow.data(), mha.q_w.data(),
        mha.q_b.data(), ACTIVATION_NONE, buffer2.data());
  }
  // K
  if (k_int8_) {
    k_int8_->Forward(rows, flow.data(), mha.k_b.data(), ACTIVATION_NONE,
                     buffer3.data(), int8_scratch);
  } else {
    FullyConnectedLayer<use_eigen>::Forward1D(
        rows, embedding_size_, d_model, flow.data(), mha.k_w.data(),
        mha.k_b.data(), ACTIVATION_NONE, buffer3.data());
  }

  // MHA (Q, K, V)
  const int depth = static_cast<int>(d_model / heads);
  const int row_stride = static_cast<int>(d_model);  // heads * depth
  const float scaling = 1.0f / sqrtf(static_cast<float>(depth));

  // MHA is done per batch since there's a fourth dimension introduced.
  for (size_t batch = 0; batch < batch_size; batch++) {
    const size_t batch_start = batch * kSquares * d_model;

    float* qk = &buffer4[batch * kSquares * kSquares * heads];

    const float* q = &buffer2[batch_start];
    const float* k = &buffer3[batch_start];

    // matmul(Q, K) for all heads per batch.
    for (size_t h = 0; h < heads; h++) {
      const float* a = &q[h * depth];
      const float* b = &k[h * depth];
      float* c = &qk[h * kSquares * kSquares];
      const float beta = mha.has_smolgen ? 1.0f : 0.0f;
      if (use_eigen) {
        auto c_mat = EigenMatrixMap<float>(c, kSquares, kSquares);
        c_mat.noalias() =
            beta * c_mat +
            scaling *
                ConstEigenStridedMatrixMap<float>(
                    b, depth, kSquares, Eigen::OuterStride<>(row_stride))
                    .transpose() *
                ConstEigenStridedMatrixMap<float>(
                    a, depth, kSquares, Eigen::OuterStride<>(row_stride));
      } else {
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, kSquares, kSquares,
                    depth, scaling, a, row_stride, b, row_stride, beta, c,
                    kSquares);
#else
        // Should never get here.
        throw Exception("Blas backend internal error");
#endif
      }
    }
  }

  // Apply Softmax.
  float* qk = buffer4.data();
  for (size_t h = 0; h < batch_size * heads * kSquares * kSquares;
       h += kSquares) {
#if defined(USE_ISPC)
    if (!use_eigen) {
      ispc::SoftmaxActivation(kSquares, qk + h, qk + h);
      continue;
    }
#endif
    SoftmaxActivation(kSquares, qk + h, qk + h);
  }

  // V
  if (v_int8_) {
    v_int8_->Forward(rows, flow.data(), mha.v_b.data(), ACTIVATION_NONE,
                     buffer3.data(), int8_scratch);
  } else {
    FullyConnectedLayer<use_eigen>::Forward1D(
        rows, embedding_size_, d_model, flow.data(), mha.v_w.data(),
        mha.v_b.data(), ACTIVATION_NONE, buffer3.data());
  }

  for (size_t batch = 0; batch < batch_size; batch++) {
    const size_t batch_start = batch * kSquares * d_model;
    // matmul(softmax(QK), V) for all heads per batch.
    float* attention = &buffer2[batch_start];
    const float* v = &buffer3[batch_start];
    const float* batch_qk = &buffer4[batch * kSquares * kSquares * heads];
    for (size_t h = 0; h < heads; h++) {
      const float* a = &batch_qk[h * kSquares * kSquares];
      const float* b = &v[h * depth];
      float* c = &attention[h * depth];
      if (use_eigen) {
        auto c_mat = EigenStridedMatrixMap<float>(
            c, depth, kSquares, Eigen::OuterStride<>(row_stride));
        c_mat.noalias() =
            ConstEigenStridedMatrixMap<float>(
                b, depth, kSquares, Eigen::OuterStride<>(row_stride)) *
            ConstEigenMatrixMap<float>(a, kSquares, kSquares);
      } else {
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kSquares, depth,
                    kSquares, 1.0f, a, kSquares, b, row_stride, 0.0f, c,
                    row_stride);
#endif
      }
    }
  }

  // Fully connected final MHA layer.
  if (dense_int8_) {
    dense_int8_->Forward(rows, buffer2.data(), mha.dense_b.data(),
                         ACTIVATION_NONE, buffer3.data(), int8_scratch);
  } else {
    FullyConnectedLayer<use_eigen>::Forward1D(
        rows, d_model, embedding_size_, buffer2.data(), mha.dense_w.data(),
        mha.dense_b.data(), ACTIVATION_NONE, buffer3.data());
  }

  // Layer Norm + skip connection.
  LayerNorm2DWithSkipConnection(batch_size * kSquares, embedding_size_,
                                buffer3.data(), alpha_, flow.data(),
                                weights_.ln1_gammas.data(),
                                weights_.ln1_betas.data(), default_epsilon_);
  std::swap(buffer3, flow);

  // FFN.
  if (ffn_dense1_int8_) {
    ffn_dense1_int8_->Forward(rows, flow.data(), ffn.dense1_b.data(),
                              ffn_activation_, buffer4.data(), int8_scratch);
  } else {
    FullyConnectedLayer<use_eigen>::Forward1D(
        rows, embedding_size_, dff_size, flow.data(), ffn.dense1_w.data(),
        ffn.dense1_b.data(), ffn_activation_, buffer4.data());
  }

  if (ffn_dense2_int8_) {
    ffn_dense2_int8_->Forward(rows, buffer4.data(), ffn.dense2_b.data(),
                              ACTIVATION_NONE, buffer3.data(), int8_scratch);
  } else {
    FullyConnectedLayer<use_eigen>::Forward1D(
        rows, dff_size, ffn.dense2_b.size(), buffer4.data(),
        ffn.dense2_w.data(), ffn.dense2_b.data(), ACTIVATION_NONE,
        buffer3.data());
  }

  // Layer Norm + skip connection.
  LayerNorm2DWithSkipConnection(batch_size * kSquares, embedding_size_,
                                buffer3.data(), alpha_, flow.data(),
                                weights_.ln2_gammas.data(),
                                weights_.ln2_betas.data(), default_epsilon_);
  std::swap(buffer3, flow);
}

template <bool use_eigen>
AttentionBody<use_eigen>::AttentionBody(const MultiHeadWeights& weights,
                                        size_t input_channels,
                                        bool has_residual_tower,
                                        bool is_pe_dense_embedding,
                                        ActivationFunction default_activation,
                                        ActivationFunction smolgen_activation,
                                        ActivationFunction ffn_activation,
                                        bool use_int8)
    : BaseLayer<use_eigen>(weights.ip_emb_b.size() * kSquares),
      weights_(weights),
      input_channels_(input_channels),
      embedding_size_(weights.ip_emb_b.size()),
      position_encoding_channels_(
          is_pe_dense_embedding ? weights.ip_emb_preproc_b.size() / kSquares
                                : kNumPosEncodingChannels),
      has_residual_tower_(has_residual_tower),
      is_pe_dense_embedding_(is_pe_dense_embedding),
      default_activation_(default_activation),
      ffn_activation_(ffn_activation),
      alpha_(static_cast<float>(pow(2.0 * weights.encoder.size(), -0.25))) {
  assert(embedding_size_ > 0);
  for (const auto& layer : weights.encoder) {
    encoders_.emplace_back(layer, weights.smolgen_w, embedding_size_,
                           weights.encoder_head_count, smolgen_activation,
                           ffn_activation, alpha_,
                           is_pe_dense_embedding ? 1e-3f : 1e-6f, use_int8);
  }
}

template <bool use_eigen>
void AttentionBody<use_eigen>::Eval(size_t batch_size, float* output,
                                    const float* input, const float* /*input2*/,
                                    Workspace<use_eigen>& workspace) const {
  std::vector<float>& flow = workspace.buffers[0];
  std::vector<float>& buffer2 = workspace.buffers[1];
  std::vector<float>& buffer3 = workspace.buffers[2];
  std::vector<float>& buffer4 = workspace.buffers[3];

  // The embedding's input: the residual tower's output, or the planes with
  // the position encoding appended, in NHWC order.
  const size_t embedding_input_size =
      has_residual_tower_ ? input_channels_
                          : input_channels_ + position_encoding_channels_;
  GrowBuffer(buffer3, batch_size * kSquares * embedding_input_size);
  float* embedding_input = buffer3.data();

  if (has_residual_tower_) {
    NchwToNhwc(batch_size, input_channels_, input, embedding_input);
  } else {
    if (is_pe_dense_embedding_) {
      // NCHW to NHWC conversion of 12-channel slice of input.
      GrowBuffer(buffer4, batch_size * kSquares * 12);
      for (size_t batch = 0; batch < batch_size; batch++) {
        for (size_t i = 0; i < kSquares; i++) {
          for (size_t j = 0; j < 12; j++) {
            buffer4[batch * kSquares * 12 + i * 12 + j] =
                input[batch * kSquares * input_channels_ + j * kSquares + i];
          }
        }
      }
      // Dense embedding preprocess layer.
      GrowBuffer(buffer2, batch_size * weights_.ip_emb_preproc_b.size());
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size, kSquares * 12, weights_.ip_emb_preproc_b.size(),
          buffer4.data(), weights_.ip_emb_preproc_w.data(),
          weights_.ip_emb_preproc_b.data(), ACTIVATION_NONE, buffer2.data());
    }

    const size_t channels = position_encoding_channels_;
    for (size_t batch = 0; batch < batch_size; batch++) {
      for (size_t i = 0; i < kSquares; i++) {
        float* row =
            embedding_input + (batch * kSquares + i) * embedding_input_size;
        // Input NCHW to NHWC conversion.
        for (size_t j = 0; j < input_channels_; j++) {
          row[j] = input[batch * kSquares * input_channels_ + j * kSquares + i];
        }
        // Position encoding concat.
        for (size_t j = 0; j < channels; j++) {
          row[input_channels_ + j] =
              is_pe_dense_embedding_
                  ? buffer2[batch * kSquares * channels + i * channels + j]
                  : kPosEncoding[i][j];
        }
      }
    }
  }

  // Input embedding.
  GrowBuffer(flow, batch_size * kSquares * embedding_size_);
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, embedding_input_size, embedding_size_,
      embedding_input, weights_.ip_emb_w.data(), weights_.ip_emb_b.data(),
      default_activation_, flow.data());

  // Layer norm for new encoding.
  if (is_pe_dense_embedding_) {
    LayerNorm2DWithSkipConnection(batch_size * kSquares, embedding_size_,
                                  flow.data(), 1.0f, (const float*)nullptr,
                                  weights_.ip_emb_ln_gammas.data(),
                                  weights_.ip_emb_ln_betas.data(), 1e-3f);
  }

  // Input gating
  if (weights_.ip_mult_gate.size() > 0 && weights_.ip_add_gate.size() > 0) {
    for (size_t batch = 0; batch < batch_size; batch++) {
      for (size_t i = 0; i < kSquares; i++) {
        for (size_t j = 0; j < embedding_size_; j++) {
          const size_t index =
              batch * kSquares * embedding_size_ + i * embedding_size_ + j;
          flow[index] = flow[index] * weights_.ip_mult_gate[j * kSquares + i] +
                        weights_.ip_add_gate[j * kSquares + i];
        }
      }
    }
  }

  // FFN in embedding for new encoding.
  if (is_pe_dense_embedding_) {
    const auto& ffn = weights_.ip_emb_ffn;
    const size_t dff_size = ffn.dense1_b.size();
    GrowBuffer(buffer3, batch_size * kSquares * dff_size);
    GrowBuffer(buffer2, batch_size * kSquares * ffn.dense2_b.size());
    // FFN dense 1.
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size * kSquares, embedding_size_, dff_size, flow.data(),
        ffn.dense1_w.data(), ffn.dense1_b.data(), ffn_activation_,
        buffer3.data());

    // FFN dense 2.
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size * kSquares, dff_size, ffn.dense2_b.size(), buffer3.data(),
        ffn.dense2_w.data(), ffn.dense2_b.data(), ACTIVATION_NONE,
        buffer2.data());

    // Layer Norm.
    LayerNorm2DWithSkipConnection(batch_size * kSquares, ffn.dense2_b.size(),
                                  buffer2.data(), alpha_, flow.data(),
                                  weights_.ip_emb_ffn_ln_gammas.data(),
                                  weights_.ip_emb_ffn_ln_betas.data(), 1e-3f);

    std::swap(flow, buffer2);
  }

  // Attention body encoders.
  for (const auto& encoder : encoders_) {
    encoder.Eval(batch_size, flow, buffer2, buffer3, buffer4, workspace.int8);
  }

  std::memcpy(output, flow.data(),
              batch_size * kSquares * embedding_size_ * sizeof(float));
}

template <bool use_eigen>
AttentionPolicyHead<use_eigen>::AttentionPolicyHead(
    const MultiHeadWeights::PolicyHead& weights,
    const std::vector<float>& smolgen_global_weights, size_t input_channels,
    bool attention_body, ActivationFunction default_activation,
    ActivationFunction smolgen_activation, ActivationFunction ffn_activation,
    bool use_int8)
    : BaseLayer<use_eigen>(kAttentionPolicyOutputs),
      weights_(weights),
      input_channels_(input_channels),
      attention_body_(attention_body),
      default_activation_(default_activation) {
  const size_t policy_embedding_size = weights.ip_pol_b.size();
  for (const auto& layer : weights.pol_encoder) {
    encoders_.emplace_back(
        layer, smolgen_global_weights, policy_embedding_size,
        weights.pol_encoder_head_count,
        attention_body ? smolgen_activation : ACTIVATION_NONE,
        attention_body ? ffn_activation : ACTIVATION_SELU, 1.0f, 1e-6f,
        use_int8);
  }
}

template <bool use_eigen>
void AttentionPolicyHead<use_eigen>::Eval(
    size_t batch_size, float* output, const float* input,
    const float* /*input2*/, Workspace<use_eigen>& workspace) const {
  std::vector<float>& flow = workspace.buffers[0];
  std::vector<float>& buffer2 = workspace.buffers[1];
  std::vector<float>& buffer3 = workspace.buffers[2];
  std::vector<float>& buffer4 = workspace.buffers[3];

  const float* embedding_input = input;
  if (!attention_body_) {
    GrowBuffer(buffer2, batch_size * kSquares * input_channels_);
    NchwToNhwc(batch_size, input_channels_, input, buffer2.data());
    embedding_input = buffer2.data();
  }

  const size_t policy_embedding_size = weights_.ip_pol_b.size();
  GrowBuffer(flow, batch_size * kSquares * policy_embedding_size);
  // Policy Embedding.
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, input_channels_, policy_embedding_size,
      embedding_input, weights_.ip_pol_w.data(), weights_.ip_pol_b.data(),
      attention_body_
          ? default_activation_
          : ACTIVATION_SELU,  // SELU activation hardcoded for apmish nets.
      flow.data());

  for (const auto& encoder : encoders_) {
    encoder.Eval(batch_size, flow, buffer2, buffer3, buffer4, workspace.int8);
  }

  const size_t policy_d_model = weights_.ip2_pol_b.size();
  GrowBuffer(buffer2, batch_size * kSquares * policy_d_model);
  GrowBuffer(buffer3, batch_size * kSquares * policy_d_model);

  // Q
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, policy_embedding_size, policy_d_model, flow.data(),
      weights_.ip2_pol_w.data(), weights_.ip2_pol_b.data(), ACTIVATION_NONE,
      buffer2.data());
  // K
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, policy_embedding_size, policy_d_model, flow.data(),
      weights_.ip3_pol_w.data(), weights_.ip3_pol_b.data(), ACTIVATION_NONE,
      buffer3.data());
  const int d_model = static_cast<int>(policy_d_model);
  const float scaling = 1.0f / sqrtf(static_cast<float>(d_model));
  for (size_t batch = 0; batch < batch_size; batch++) {
    const float* a = &buffer2[batch * kSquares * policy_d_model];
    const float* b = &buffer3[batch * kSquares * policy_d_model];
    float* c = output + batch * kAttentionPolicyOutputs;
    if (use_eigen) {
      auto c_mat = EigenMatrixMap<float>(c, kSquares, kSquares);
      c_mat.noalias() =
          scaling *
          ConstEigenMatrixMap<float>(b, d_model, kSquares).transpose() *
          ConstEigenMatrixMap<float>(a, d_model, kSquares);
    } else {
#ifdef USE_BLAS
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, kSquares, kSquares,
                  d_model, scaling, a, d_model, b, d_model, 0.0f, c, kSquares);
#else
      // Should never get here.
      throw Exception("Blas backend internal error");
#endif
    }
  }
  // Promotion offset calculation.
  for (size_t batch = 0; batch < batch_size; batch++) {
    float* logits = output + batch * kAttentionPolicyOutputs;
    float promotion_offsets[4][8];
    // This is so small that SGEMM seems slower.
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 8; j++) {
        float sum = 0;
        for (size_t k = 0; k < policy_d_model; k++) {
          sum += buffer3[batch * kSquares * policy_d_model +
                         (56 + j) * policy_d_model + k] *
                 weights_.ip4_pol_w[i * policy_d_model + k];
        }
        promotion_offsets[i][j] = sum;
      }
    }
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 8; j++) {
        promotion_offsets[i][j] += promotion_offsets[3][j];
      }
    }
    for (int k = 0; k < 8; k++) {      // y in cuda
      for (int j = 0; j < 8; j++) {    // w in cuda
        for (int i = 0; i < 3; i++) {  // c in cuda
          logits[64 * 64 + 24 * k + 3 * j + i] =
              logits[(48 + k) * 64 + 56 + j] + promotion_offsets[i][j];
        }
      }
    }
  }
}

template <bool use_eigen>
ValueHead<use_eigen>::ValueHead(const MultiHeadWeights::ValueHead& weights,
                                size_t input_channels, bool attention_body,
                                bool wdl, ActivationFunction default_activation)
    : BaseLayer<use_eigen>(wdl ? 3 : 1),
      weights_(weights),
      input_channels_(input_channels),
      attention_body_(attention_body),
      wdl_(wdl),
      default_activation_(default_activation) {}

template <bool use_eigen>
void ValueHead<use_eigen>::Eval(size_t batch_size, float* output,
                                const float* input, const float* /*input2*/,
                                Workspace<use_eigen>& workspace) const {
  const size_t input_planes =
      attention_body_ ? weights_.ip_val_b.size() : weights_.value.biases.size();
  const size_t value_channels = weights_.ip1_val_b.size();
  std::vector<float>& head = workspace.buffers[0];
  std::vector<float>& hidden = workspace.buffers[1];
  GrowBuffer(head, batch_size * input_planes * kSquares);
  GrowBuffer(hidden, batch_size * value_channels);

  if (attention_body_) {
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size * kSquares, input_channels_, input_planes, input,
        weights_.ip_val_w.data(), weights_.ip_val_b.data(), default_activation_,
        head.data());
  } else {
    Convolution1<use_eigen>::Forward(batch_size, input_channels_, input_planes,
                                     input, weights_.value.weights.data(),
                                     head.data());
    BiasActivate(batch_size, input_planes, head.data(),
                 weights_.value.biases.data(), default_activation_);
  }

  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size, input_planes * kSquares, value_channels, head.data(),
      weights_.ip1_val_w.data(), weights_.ip1_val_b.data(),
      default_activation_,  // Activation On
      hidden.data());

  // Now get the score
  if (wdl_) {
    std::vector<float>& wdl = workspace.buffers[2];
    GrowBuffer(wdl, 3 * batch_size);
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size, value_channels, 3, hidden.data(), weights_.ip2_val_w.data(),
        weights_.ip2_val_b.data(),
        ACTIVATION_NONE,  // Activation Off
        wdl.data());
    for (size_t j = 0; j < batch_size; j++) {
      SoftmaxActivation(3, &wdl[j * 3], output + j * 3);
    }
  } else {
    for (size_t j = 0; j < batch_size; j++) {
      const double winrate = FullyConnectedLayer<use_eigen>::Forward0D(
                                 value_channels, weights_.ip2_val_w.data(),
                                 &hidden[j * value_channels]) +
                             weights_.ip2_val_b[0];
      output[j] = static_cast<float>(std::tanh(winrate));
    }
  }
}

template class Workspace<true>;
template class ConvLayer<true>;
template class ResidualBlock<true>;
template class DenseLayer<true>;
template class EmbeddingLayer<true>;
template class PolicyMapLayer<true>;
template class EncoderBlock<true>;
template class AttentionBody<true>;
template class AttentionPolicyHead<true>;
template class ValueHead<true>;
#ifdef USE_BLAS
template class Workspace<false>;
template class ConvLayer<false>;
template class ResidualBlock<false>;
template class DenseLayer<false>;
template class EmbeddingLayer<false>;
template class PolicyMapLayer<false>;
template class EncoderBlock<false>;
template class AttentionBody<false>;
template class AttentionPolicyHead<false>;
template class ValueHead<false>;
#endif

}  // namespace blas_backend
}  // namespace lczero
