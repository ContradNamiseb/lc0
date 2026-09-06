/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018-2023 The LCZero Authors

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

#include <algorithm>
#include <fstream>
#include <cassert>
#include <cmath>
#include <iostream>

#include "neural/backends/blas/blas.h"
#include "neural/backends/blas/convolution1.h"
#include "neural/backends/blas/encoder.h"
#include "neural/backends/blas/fully_connected_layer.h"
#include "neural/backends/blas/se_unit.h"
#include "neural/backends/blas/winograd_convolution3.h"
#include "neural/backends/shared/activation.h"
#include "neural/backends/shared/winograd_filter.h"
#include "neural/factory.h"
#include "neural/network.h"
#include "neural/kda_directions.h"
#include "neural/network_legacy.h"
#include "neural/tables/attention_policy_map.h"
#include "neural/tables/policy_map.h"
#include "utils/numa.h"

#ifdef USE_DNNL
#include <omp.h>
#endif

#ifdef USE_ISPC
#include "activation_ispc.h"
#endif

namespace lczero {
namespace {

struct Buffers {
  std::vector<float> buffer1;
  std::vector<float> buffer2;
  std::vector<float> buffer3;
  std::vector<float> buffer4;
};

template <bool use_eigen>
class BlasNetwork;

template <bool use_eigen>
class BlasComputation : public NetworkComputation {
 public:
  BlasComputation(BlasNetwork<use_eigen>* network,
                  const MultiHeadWeights& weights,
                  const std::string policy_head, const std::string value_head,
                  const size_t max_batch_size, const bool wdl,
                  const bool moves_left, const bool conv_policy,
                  const ActivationFunction default_activation,
                  const ActivationFunction smolgen_activation,
                  const ActivationFunction ffn_activation,
                  const bool attn_policy, const bool attn_body,
                  bool is_pe_dense_embedding, std::vector<int> kda_directions,
                  int threads);

  virtual ~BlasComputation() {}

  // Adds a sample to the batch.
  void AddInput(InputPlanes&& input) override { planes_.emplace_back(input); }

  // Do the computation.
  void ComputeBlocking() override;

  // Returns how many times AddInput() was called.
  int GetBatchSize() const override { return static_cast<int>(planes_.size()); }

  // Returns Q value of @sample.
  float GetQVal(int sample) const override {
    if (wdl_) {
      auto w = q_values_[3 * sample + 0];
      auto l = q_values_[3 * sample + 2];
      return w - l;
    } else {
      return q_values_[sample];
    }
  }

  float GetDVal(int sample) const override {
    if (wdl_) {
      auto d = q_values_[3 * sample + 1];
      return d;
    } else {
      return 0.0f;
    }
  }

  float GetMVal(int sample) const override {
    if (moves_left_) {
      return m_values_[sample];
    } else {
      return 0.0f;
    }
  }

  // Returns P value @move_id of @sample.
  float GetPVal(int sample, int move_id) const override {
    return policies_[sample][move_id];
  }

 private:
  void EncodePlanes(const InputPlanes& sample, float* buffer);
  void ForwardEncoderLayer(
      std::vector<float>& encoder_buffer, std::vector<float>& encoder_buffer2,
      std::vector<float>& encoder_buffer3, std::vector<float>& encoder_buffer4,
      size_t batch_size, const MultiHeadWeights::EncoderLayer& layer,
      int embedding_size, int heads, ActivationFunction smolgen_activation,
      ActivationFunction ffn_activation, float alpha, float default_eps);
  void ForwardKdaMixer(const std::vector<float>& input,
                       std::vector<float>& output, size_t batch_size,
                       const MultiHeadWeights::EncoderLayer& layer,
                       int embedding_size, int heads);

  static constexpr auto kWidth = 8;
  static constexpr auto kHeight = 8;
  static constexpr auto kSquares = kWidth * kHeight;
  static constexpr auto kPolicyOutputs = 1858;
  // Number of used planes with convolutional policy.
  // The real number of planes is higher because of padding.
  static constexpr auto kPolicyUsedPlanes = 73;

  const MultiHeadWeights& weights_;
  size_t max_batch_size_;
  std::vector<InputPlanes> planes_;
  std::vector<std::vector<float>> policies_;
  std::vector<float> q_values_;
  std::vector<float> m_values_;
  bool wdl_;
  bool moves_left_;
  bool conv_policy_;
  bool attn_policy_;
  bool attn_body_;
  bool is_pe_dense_embedding_;
  std::vector<int> kda_directions_;
  ActivationFunction default_activation_;
  ActivationFunction smolgen_activation_;
  ActivationFunction ffn_activation_;
  std::string policy_head_;
  std::string value_head_;
  BlasNetwork<use_eigen>* network_;
};

template <bool use_eigen>
class BlasNetwork : public Network {
 public:
  BlasNetwork(const WeightsFile& weights, const OptionsDict& options);
  virtual ~BlasNetwork() {};

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<BlasComputation<use_eigen>>(
        this, weights_, policy_head_, value_head_, max_batch_size_, wdl_,
        moves_left_, conv_policy_, default_activation_, smolgen_activation_,
        ffn_activation_, attn_policy_, attn_body_, is_pe_dense_embedding_,
        kda_directions_, threads_);
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  int GetMiniBatchSize() const override { return 7; }

  bool IsCpu() const override { return true; }

  void InitThread(int id) override { Numa::BindThread(id); }

  std::unique_ptr<Buffers> GetBuffers() {
    std::lock_guard<std::mutex> lock(buffers_lock_);
    if (free_buffers_.empty()) {
      return std::make_unique<Buffers>();
    } else {
      auto buffers = std::move(free_buffers_.back());
      free_buffers_.pop_back();
      return buffers;
    }
  }

  void ReleaseBuffers(std::unique_ptr<Buffers> buffers) {
    std::lock_guard<std::mutex> lock(buffers_lock_);
    free_buffers_.push_back(std::move(buffers));
  }

 private:
  // A cap on the max batch size since it consumes a lot of memory
  static constexpr auto kHardMaxBatchSize = 2048;

  const NetworkCapabilities capabilities_;
  MultiHeadWeights weights_;
  size_t max_batch_size_;
  int threads_;
  bool wdl_;
  bool moves_left_;
  bool conv_policy_;
  bool attn_policy_;
  bool attn_body_;
  bool is_pe_dense_embedding_;
  std::vector<int> kda_directions_;
  ActivationFunction default_activation_ = ACTIVATION_NONE;
  ActivationFunction smolgen_activation_ = ACTIVATION_NONE;
  ActivationFunction ffn_activation_ = ACTIVATION_NONE;
  std::string policy_head_;
  std::string value_head_;
  std::mutex buffers_lock_;
  std::vector<std::unique_ptr<Buffers>> free_buffers_;
};

template <bool use_eigen>
BlasComputation<use_eigen>::BlasComputation(
    BlasNetwork<use_eigen>* network, const MultiHeadWeights& weights,
    const std::string policy_head, const std::string value_head,
    const size_t max_batch_size, const bool wdl, const bool moves_left,
    const bool conv_policy, const ActivationFunction default_activation,
    const ActivationFunction smolgen_activation,
    const ActivationFunction ffn_activation, const bool attn_policy,
    const bool attn_body, bool is_pe_dense_embedding,
    std::vector<int> kda_directions, [[maybe_unused]] int threads)
    : weights_(weights),
      max_batch_size_(max_batch_size),
      policies_(0),
      q_values_(0),
      wdl_(wdl),
      moves_left_(moves_left),
      conv_policy_(conv_policy),
      attn_policy_(attn_policy),
      attn_body_(attn_body),
      is_pe_dense_embedding_(is_pe_dense_embedding),
      kda_directions_(std::move(kda_directions)),
      default_activation_(default_activation),
      smolgen_activation_(smolgen_activation),
      ffn_activation_(ffn_activation),
      policy_head_(policy_head),
      value_head_(value_head),
      network_(network) {
#ifdef USE_DNNL
  omp_set_num_threads(threads);
#endif
}

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

void vec_adjust(std::vector<float>& vec, size_t size) {
  if (vec.size() < size) {
    vec.clear();
    vec.resize(size);
  }
}

// Traversal order of the 64 squares for each KdaDirection value. Must match
// KDA_TRAVERSALS in the trainer.
// Reference implementation of the gated delta rule, written as the plain
// sequential recurrence the trainer's chunkwise-parallel form is equivalent to.
template <bool use_eigen>
void BlasComputation<use_eigen>::ForwardKdaMixer(
    const std::vector<float>& input, std::vector<float>& output,
    size_t batch_size, const MultiHeadWeights::EncoderLayer& layer,
    int embedding_size, int heads) {
  // Mirrors KDA_LOG_DECAY_FLOOR in the trainer.
  constexpr float kLogDecayFloor = -10.0f;
  const auto& kda = layer.kda;
  const int key_dim = kda.key_dim;
  const int value_dim = kda.value_dim;
  const int gate_rank = kda.gate_rank;
  const int key_depth = heads * key_dim;
  const int value_depth = heads * value_dim;
  const size_t tokens = batch_size * kSquares;
  const int direction_count = static_cast<int>(kda_directions_.size());

  if (key_dim <= 0 || value_dim <= 0 || gate_rank <= 0) {
    throw Exception("Invalid KDA dimensions.");
  }
  if (direction_count <= 0 || heads % direction_count != 0) {
    throw Exception("KDA directions must evenly divide the encoder heads.");
  }
  if (kda.output_rms_norm && kda.out_norm_gammas.empty()) {
    throw Exception(
        "KDA output RMS norm requested but out_norm_gammas is missing.");
  }
  const int heads_per_direction = heads / direction_count;

  auto bias = [](const MultiHeadWeights::Vec& v) {
    return v.empty() ? nullptr : v.data();
  };

  // Apply the optional 3x3 depthwise board convolution (residual) before
  // the Q/K/V projections. Matches kda_local_conv in the trainer.
  // local_conv_w layout: [emb_size, 1, 3, 3] flattened =
  //   w[c * 9 + kr * 3 + kf] for channel c, kernel row offset kr, col kf.
  const std::vector<float>* proj_input_ptr = &input;
  std::vector<float> conv_input;
  if (kda.local_conv && !kda.local_conv_w.empty()) {
    conv_input = input;
    for (size_t batch = 0; batch < batch_size; ++batch) {
      for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
          const int square = static_cast<int>(batch) * kSquares + rank * 8 + file;
          for (int c = 0; c < embedding_size; ++c) {
            float sum = 0.0f;
            for (int kr = 0; kr < 3; ++kr) {
              const int nr = rank + kr - 1;
              if (nr < 0 || nr > 7) continue;
              for (int kf = 0; kf < 3; ++kf) {
                const int nf = file + kf - 1;
                if (nf < 0 || nf > 7) continue;
                const int nbr = static_cast<int>(batch) * kSquares +
                                nr * 8 + nf;
                sum += kda.local_conv_w[c * 9 + kr * 3 + kf] *
                       input[nbr * embedding_size + c];
              }
            }
            if (!kda.local_conv_b.empty()) {
              sum += kda.local_conv_b[c];
            }
            conv_input[square * embedding_size + c] += sum;
          }
        }
      }
    }
    proj_input_ptr = &conv_input;
  }
  const std::vector<float>& proj_input = *proj_input_ptr;

  std::vector<float> q(tokens * key_depth);
  std::vector<float> k(tokens * key_depth);
  std::vector<float> v(tokens * value_depth);
  std::vector<float> raw_decay(tokens * key_depth);
  std::vector<float> beta(tokens * heads);
  std::vector<float> hidden(tokens * gate_rank);
  std::vector<float> mixed(tokens * value_depth);

  // The trainer's qkv_silu applies SiLU (= Swish, x*sigmoid(x)) to the
  // q/k/v projections. Matches kda.py F.silu when qkv_silu is set. (Ported
  // from the sycl-openvino-fix lineage so this branch's BLAS reference
  // evaluates trained qkv_silu nets correctly.)
  const ActivationFunction qkv_act =
      kda.qkv_silu ? ACTIVATION_SWISH : ACTIVATION_NONE;

  FullyConnectedLayer<use_eigen>::Forward1D(
      tokens, embedding_size, key_depth, proj_input.data(), kda.q_w.data(),
      bias(kda.q_b), qkv_act, q.data());
  FullyConnectedLayer<use_eigen>::Forward1D(
      tokens, embedding_size, key_depth, proj_input.data(), kda.k_w.data(),
      bias(kda.k_b), qkv_act, k.data());
  FullyConnectedLayer<use_eigen>::Forward1D(
      tokens, embedding_size, value_depth, proj_input.data(), kda.v_w.data(),
      bias(kda.v_b), qkv_act, v.data());

  FullyConnectedLayer<use_eigen>::Forward1D(
      tokens, embedding_size, gate_rank, proj_input.data(),
      kda.decay_a_w.data(), bias(kda.decay_a_b), ACTIVATION_NONE,
      hidden.data());
  FullyConnectedLayer<use_eigen>::Forward1D(
      tokens, gate_rank, key_depth, hidden.data(), kda.decay_b_w.data(),
      bias(kda.decay_b_b), ACTIVATION_NONE, raw_decay.data());

  FullyConnectedLayer<use_eigen>::Forward1D(
      tokens, embedding_size, heads, proj_input.data(), kda.beta_w.data(),
      bias(kda.beta_b), ACTIVATION_NONE, beta.data());

  std::vector<float> state(static_cast<size_t>(key_dim) * value_dim);
  for (size_t batch = 0; batch < batch_size; batch++) {
    for (int head = 0; head < heads; head++) {
      const int direction = kda_directions_[head / heads_per_direction];
      const float decay_scale = std::exp(kda.a_log[head]);
      const float scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
      std::fill(state.begin(), state.end(), 0.0f);

      for (int token = 0; token < kSquares; token++) {
        const size_t square = batch * kSquares + KdaSquareForToken(
                                                     direction, token);
        const size_t key_offset = square * key_depth + head * key_dim;
        const size_t value_offset = square * value_depth + head * value_dim;

        float q_norm_sq = 0.0f;
        float k_norm_sq = 0.0f;
        for (int key = 0; key < key_dim; key++) {
          q_norm_sq += q[key_offset + key] * q[key_offset + key];
          k_norm_sq += k[key_offset + key] * k[key_offset + key];
        }
        const float q_norm = 1.0f / std::sqrt(std::max(q_norm_sq, 1.0e-12f));
        const float k_norm = 1.0f / std::sqrt(std::max(k_norm_sq, 1.0e-12f));

        for (int key = 0; key < key_dim; key++) {
          const float x = raw_decay[key_offset + key] +
                          kda.dt_bias[head * key_dim + key];
          const float softplus =
              std::max(x, 0.0f) + std::log1p(std::exp(-std::abs(x)));
          const float decay =
              std::exp(std::max(-decay_scale * softplus, kLogDecayFloor));
          for (int value = 0; value < value_dim; value++) {
            state[key * value_dim + value] *= decay;
          }
        }

        const float beta_raw = beta[square * heads + head];
        const float update_rate = 1.0f / (1.0f + std::exp(-beta_raw));
        for (int value = 0; value < value_dim; value++) {
          float prediction = 0.0f;
          for (int key = 0; key < key_dim; key++) {
            prediction += k[key_offset + key] * k_norm *
                          state[key * value_dim + value];
          }
          const float delta =
              update_rate * (v[value_offset + value] - prediction);
          float sum = 0.0f;
          for (int key = 0; key < key_dim; key++) {
            const int index = key * value_dim + value;
            state[index] += k[key_offset + key] * k_norm * delta;
            sum += q[key_offset + key] * q_norm * scale * state[index];
          }
          mixed[value_offset + value] = sum;
        }
      }
    }
  }

  if (kda.output_rms_norm) {
    for (size_t token = 0; token < tokens; token++) {
      float sum_squares = 0.0f;
      for (int channel = 0; channel < value_depth; channel++) {
        const float value = mixed[token * value_depth + channel];
        sum_squares += value * value;
      }
      const float factor =
          1.0f / std::sqrt(sum_squares / value_depth + kda.rms_norm_epsilon);
      for (int channel = 0; channel < value_depth; channel++) {
        mixed[token * value_depth + channel] *=
            factor * kda.out_norm_gammas[channel];
      }
    }
  }

  if (kda.output_gate) {
    std::vector<float> gate(tokens * value_depth);
    FullyConnectedLayer<use_eigen>::Forward1D(
        tokens, embedding_size, gate_rank, proj_input.data(),
        kda.gate_a_w.data(), bias(kda.gate_a_b), ACTIVATION_NONE,
        hidden.data());
    FullyConnectedLayer<use_eigen>::Forward1D(
        tokens, gate_rank, value_depth, hidden.data(), kda.gate_b_w.data(),
        bias(kda.gate_b_b), ACTIVATION_NONE, gate.data());
    for (size_t i = 0; i < mixed.size(); i++) {
      mixed[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
    }
  }

  vec_adjust(output, tokens * embedding_size);
  FullyConnectedLayer<use_eigen>::Forward1D(
      tokens, value_depth, embedding_size, mixed.data(), kda.dense_w.data(),
      bias(kda.dense_b), ACTIVATION_NONE, output.data());
}

template <bool use_eigen>
void BlasComputation<use_eigen>::ForwardEncoderLayer(
    std::vector<float>& encoder_buffer, std::vector<float>& encoder_buffer2,
    std::vector<float>& encoder_buffer3, std::vector<float>& encoder_buffer4,
    size_t batch_size, const MultiHeadWeights::EncoderLayer& layer,
    int embedding_size, int heads, ActivationFunction smolgen_activation,
    ActivationFunction ffn_activation, float alpha, float default_eps) {
  // Derived from the weights rather than the bias: nets trained with
  // omit_qkv_biases have no q_b at all.
  const int d_model = layer.mha.q_w.size() / embedding_size;
  const int dff_size = layer.ffn.dense1_b.size();
  const int hidden_channels =
      layer.mha.has_smolgen ? layer.mha.smolgen.compress.size() / embedding_size
                            : 0;
  const int hidden_sz =
      layer.mha.has_smolgen ? layer.mha.smolgen.dense1_b.size() : 0;
  const int gen_sz_outputs =
      layer.mha.has_smolgen ? layer.mha.smolgen.dense2_b.size() : 0;

  const int largest_batch_size = std::min(max_batch_size_, planes_.size());

  vec_adjust(encoder_buffer, largest_batch_size * d_model * kSquares);
  vec_adjust(encoder_buffer2,
             largest_batch_size *
                 std::max(std::max(d_model, hidden_channels) * kSquares,
                          gen_sz_outputs));
  vec_adjust(encoder_buffer3,
             largest_batch_size * std::max(d_model * kSquares, hidden_sz));
  vec_adjust(encoder_buffer4,
             batch_size * kSquares * std::max(kSquares * heads, dff_size));

  // Both mixers leave their pre-normalization output in encoder_buffer3. The
  // MHA branch below is left at its original indentation to keep the diff
  // reviewable.
  if (layer.is_kda) {
    ForwardKdaMixer(encoder_buffer, encoder_buffer3, batch_size, layer,
                    embedding_size, heads);
  } else {

  // Smolgen.
  if (layer.mha.has_smolgen) {
    const float* input = &encoder_buffer[0];
    float* QK = &encoder_buffer4[0];

    // Compress.
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size * kSquares, embedding_size, hidden_channels, input,
        layer.mha.smolgen.compress.data(), (const float*)nullptr,
        ACTIVATION_NONE, encoder_buffer2.data());

    // Dense 1.
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size, kSquares * hidden_channels, hidden_sz,
        encoder_buffer2.data(), layer.mha.smolgen.dense1_w.data(),
        layer.mha.smolgen.dense1_b.data(), smolgen_activation,
        encoder_buffer3.data());
    // Layer Norm.
    LayerNorm2DWithSkipConnection(batch_size, hidden_sz, encoder_buffer3.data(),
                                  1.0f, (const float*)nullptr,
                                  layer.mha.smolgen.ln1_gammas.data(),
                                  layer.mha.smolgen.ln1_betas.data(), 1e-3);

    // Dense 2.
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size, hidden_sz, gen_sz_outputs, encoder_buffer3.data(),
        layer.mha.smolgen.dense2_w.data(), layer.mha.smolgen.dense2_b.data(),
        smolgen_activation, encoder_buffer2.data());
    // Layer Norm.
    LayerNorm2DWithSkipConnection(
        batch_size, gen_sz_outputs, encoder_buffer2.data(), 1.0f,
        (const float*)nullptr, layer.mha.smolgen.ln2_gammas.data(),
        layer.mha.smolgen.ln2_betas.data(), 1e-3);

    // Global smolgen weights.
    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size * heads, gen_sz_outputs / heads, kSquares * kSquares,
        encoder_buffer2.data(), weights_.smolgen_w.data(),
        (const float*)nullptr, ACTIVATION_NONE, QK);
  }

  // Q
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, embedding_size, d_model, encoder_buffer.data(),
      layer.mha.q_w.data(), layer.mha.q_b.data(), ACTIVATION_NONE,
      encoder_buffer2.data());
  // K
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, embedding_size, d_model, encoder_buffer.data(),
      layer.mha.k_w.data(), layer.mha.k_b.data(), ACTIVATION_NONE,
      encoder_buffer3.data());

  // MHA (Q, K, V)
  const int depth = d_model / heads;
  const float scaling = 1.0f / sqrtf(depth);

  // MHA is done per batch since there's a fourth dimension introduced.
  for (auto batch = size_t{0}; batch < batch_size; batch++) {
    auto batchStart = batch * kSquares * d_model;

    float* QK = &encoder_buffer4[batch * kSquares * kSquares * heads];

    const float* Q = &encoder_buffer2[batchStart];
    const float* K = &encoder_buffer3[batchStart];

    // matmul(Q, K) for all heads per batch.

    for (auto h = 0; h < heads; h++) {
      const float* A = &Q[h * depth];
      const float* B = &K[h * depth];
      float* C = &QK[h * kSquares * kSquares];
      const float beta = layer.mha.has_smolgen ? 1.0f : 0.0f;
      if (use_eigen) {
        auto C_mat = EigenMatrixMap<float>(C, kSquares, kSquares);
        C_mat.noalias() =
            beta * C_mat +
            scaling *
                ConstEigenStridedMatrixMap<float>(
                    B, depth, kSquares, Eigen::OuterStride<>(heads * depth))
                    .transpose() *
                ConstEigenStridedMatrixMap<float>(
                    A, depth, kSquares, Eigen::OuterStride<>(heads * depth));
      } else {
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, kSquares, kSquares,
                    depth, scaling, A, heads * depth, B, heads * depth, beta, C,
                    kSquares);
#else
        // Should never get here.
        throw Exception("Blas backend internal error");
#endif
      }
    }
  }

  // Apply Softmax.
  float* QK = &encoder_buffer4[0];
  for (size_t h = 0; h < batch_size * heads * kSquares * kSquares;
       h += kSquares) {
#if defined(USE_ISPC)
    if (!use_eigen) {
      ispc::SoftmaxActivation(kSquares, QK + h, QK + h);
      continue;
    }
#endif
    SoftmaxActivation(kSquares, QK + h, QK + h);
  }

  // V
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, embedding_size, d_model, encoder_buffer.data(),
      layer.mha.v_w.data(), layer.mha.v_b.data(), ACTIVATION_NONE,
      encoder_buffer3.data());

  for (auto batch = size_t{0}; batch < batch_size; batch++) {
    auto batchStart = batch * kSquares * d_model;
    // matmul(softmax(QK), V) for all heads per batch.
    float* attn = &encoder_buffer2[batchStart];
    const float* V = &encoder_buffer3[batchStart];
    const float* QK = &encoder_buffer4[batch * kSquares * kSquares * heads];
    for (auto h = 0; h < heads; h++) {
      const float* A = &QK[h * kSquares * kSquares];
      const float* B = &V[h * depth];
      float* C = &attn[h * depth];
      if (use_eigen) {
        auto C_mat = EigenStridedMatrixMap<float>(
            C, depth, kSquares, Eigen::OuterStride<>(heads * depth));
        C_mat.noalias() =
            ConstEigenStridedMatrixMap<float>(
                B, depth, kSquares, Eigen::OuterStride<>(heads * depth)) *
            ConstEigenMatrixMap<float>(A, kSquares, kSquares);
      } else {
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kSquares, depth,
                    kSquares, 1.0f, A, kSquares, B, heads * depth, 0.0f, C,
                    heads * depth);
#endif
      }
    }
  }

  // Fully connected final MHA layer.
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, d_model, embedding_size, encoder_buffer2.data(),
      layer.mha.dense_w.data(), layer.mha.dense_b.data(), ACTIVATION_NONE,
      encoder_buffer3.data());

  }  // End of the MHA mixer branch.

  // Layer Norm + skip connection.
  LayerNorm2DWithSkipConnection(batch_size * kSquares, embedding_size,
                                encoder_buffer3.data(), alpha,
                                encoder_buffer.data(), layer.ln1_gammas.data(),
                                layer.ln1_betas.data(), default_eps);
  std::swap(encoder_buffer3, encoder_buffer);

  // FFN.
  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, embedding_size, dff_size, encoder_buffer.data(),
      layer.ffn.dense1_w.data(), layer.ffn.dense1_b.data(), ffn_activation,
      encoder_buffer4.data());

  FullyConnectedLayer<use_eigen>::Forward1D(
      batch_size * kSquares, dff_size, layer.ffn.dense2_b.size(),
      encoder_buffer4.data(), layer.ffn.dense2_w.data(),
      layer.ffn.dense2_b.data(), ACTIVATION_NONE, encoder_buffer3.data());

  // Layer Norm + skip connection.
  LayerNorm2DWithSkipConnection(batch_size * kSquares, embedding_size,
                                encoder_buffer3.data(), alpha,
                                encoder_buffer.data(), layer.ln2_gammas.data(),
                                layer.ln2_betas.data(), default_eps);
  std::swap(encoder_buffer3, encoder_buffer);
}

template <bool use_eigen>
void BlasComputation<use_eigen>::ComputeBlocking() {
  const auto& value_head = weights_.value_heads.at(value_head_);
  const auto& policy_head = weights_.policy_heads.at(policy_head_);
  // Retrieve network key dimensions from the weights structure.
  const auto num_value_channels = value_head.ip1_val_b.size();
  const auto num_moves_channels = weights_.ip1_mov_b.size();
  const auto num_value_input_planes =
      attn_body_ ? value_head.ip_val_b.size() : value_head.value.biases.size();
  const auto num_policy_input_planes = policy_head.policy.biases.size();
  const auto num_moves_input_planes =
      attn_body_ ? weights_.ip_mov_b.size() : weights_.moves_left.biases.size();
  const auto num_output_policy = static_cast<size_t>(kPolicyOutputs);
  const auto output_channels =
      attn_body_ ? weights_.ip_emb_b.size() : weights_.input.biases.size();
  const auto num_res_blocks = weights_.residual.size();

  // max_channels is the maximum number of input channels of any
  // convolution.
  // Residual blocks are identical, but the first convolution might be bigger
  // when the network has very few filters
  // For attention body nets, input channel size is increased by positional
  // encoding.
  const auto enc_channels = is_pe_dense_embedding_
                                ? weights_.ip_emb_preproc_b.size() / 64
                                : kNumPosEncodingChannels;
  const auto input_channels =
      static_cast<size_t>(kInputPlanes + (attn_body_ ? enc_channels : 0));
  const auto input_embed_dff = weights_.ip_emb_ffn.dense1_b.size();

  const auto max_channels =
      std::max(std::max(output_channels, input_channels), input_embed_dff);

  // The policy head may increase convolution max output size.
  const auto max_output_channels =
      (conv_policy_ && policy_head.policy.biases.size() > output_channels)
          ? policy_head.policy.biases.size()
          : output_channels;

  // Determine the largest batch for allocations.
  const auto total_batches = planes_.size();
  const auto largest_batch_size = std::min(max_batch_size_, total_batches);

  /* Typically
   input_channels = 112
   position encoding = 64 (512 for new encoding)
   output_channels = 192
   max_channels = 192
   num_value_input_planes = 32
   num_policy_input_planes = 32
   num_value_channels = 128
   num_output_policy = 1858
   */

  size_t max_fc_channels = std::max(
      num_value_channels, std::max(num_output_policy, num_moves_channels));
  size_t max_head_planes =
      std::max(num_policy_input_planes,
               std::max(num_value_input_planes, num_moves_input_planes));
  if (attn_policy_) {
    max_head_planes = std::max(std::max(max_head_planes, size_t{67}),
                               policy_head.ip_pol_b.size());
  }

  // The attention policy head writes into buffer1/buffer2/buffer3 with its
  // OWN widths, which are head quantities and have nothing to do with the
  // body. max_channels above stays exactly what it is -- a body/Winograd
  // bound -- because inflating it globally would enlarge unrelated
  // allocations and hide the coupling. What has to grow is only the width
  // the three shared buffers are sized from:
  //
  //   - the policy embedding writes batch * kSquares * policy_embedding_size
  //     floats into buffer2 (see the Forward1D below "Policy Embedding");
  //   - Q and K write batch * kSquares * policy_d_model into buffer1 and
  //     buffer3, and are then INDEXED with a policy_d_model stride, as is
  //     the promotion-offset loop.
  //
  // Either width exceeding max_channels overruns the allocation. The symptom
  // is heap corruption reported at some later, unrelated allocation, which
  // points nowhere near the cause.
  size_t buffer_width = max_channels;
  if (attn_policy_) {
    const size_t policy_embedding_size = policy_head.ip_pol_b.size();
    // policy_d_model is derived the same way the head derives it, but the
    // division is guarded here: this runs during allocation, before any of
    // the head's own validity assumptions have been exercised.
    const size_t policy_d_model =
        policy_embedding_size > 0
            ? policy_head.ip2_pol_w.size() / policy_embedding_size
            : 0;
    buffer_width = std::max(buffer_width,
                            std::max(policy_embedding_size, policy_d_model));
  }

  std::unique_ptr<Buffers> buffers = network_->GetBuffers();

  // Allocate data for the whole batch.
  std::vector<float>& buffer1 = buffers->buffer1;
  vec_adjust(buffer1, largest_batch_size * buffer_width * kSquares);
  std::vector<float>& buffer2 = buffers->buffer2;
  vec_adjust(buffer2, largest_batch_size * buffer_width * kSquares);
  std::vector<float>& buffer3 = buffers->buffer3;
  vec_adjust(buffer3, largest_batch_size *
                          std::max(buffer_width * kSquares, max_fc_channels));
  std::vector<float>& head_buffer = buffers->buffer4;
  vec_adjust(head_buffer, largest_batch_size * max_head_planes * kSquares);

  // Output values.
  q_values_.reserve(wdl_ ? 3 * total_batches : total_batches);
  policies_.reserve(total_batches);
  if (moves_left_) m_values_.resize(total_batches);

  WinogradConvolution3<use_eigen> convolve3(largest_batch_size, max_channels,
                                            max_output_channels);

  for (size_t start = 0; start < total_batches; start += largest_batch_size) {
    const auto batch_size = std::min(total_batches - start, largest_batch_size);
    for (size_t j = 0; j < batch_size; j++) {
      EncodePlanes(planes_[start + j], &buffer1[j * kSquares * kInputPlanes]);
    }

    if (num_res_blocks > 0) {
      // Input convolution
      convolve3.Forward(batch_size, kInputPlanes, output_channels,
                        buffer1.data(), weights_.input.weights.data(),
                        buffer2.data());

      BiasActivate(batch_size, output_channels, buffer2.data(),
                   weights_.input.biases.data(), default_activation_);

      // Residual tower
      for (auto& residual : weights_.residual) {
        const auto& conv1 = residual.conv1;
        const auto& conv2 = residual.conv2;
        const auto& se = residual.se;

        convolve3.Forward(batch_size, output_channels, output_channels,
                          buffer2.data(), conv1.weights.data(), buffer1.data());

        BiasActivate(batch_size, output_channels, buffer1.data(),
                     conv1.biases.data(), default_activation_);

        convolve3.Forward(batch_size, output_channels, output_channels,
                          buffer1.data(), conv2.weights.data(), buffer3.data());

        if (residual.has_se) {
          // No relu if followed by SE-unit and residual/bias is added later
          auto se_fc_outputs = se.b1.size();
          ApplySEUnit<use_eigen>(
              batch_size, output_channels, se_fc_outputs, buffer3.data(),
              conv2.biases.data(), buffer2.data(), se.w1.data(), se.b1.data(),
              se.w2.data(), se.b2.data(), buffer2.data(), default_activation_);
        } else {
          BiasResidual(batch_size, output_channels, buffer2.data(),
                       conv2.biases.data(), buffer3.data(),
                       default_activation_);
        }
      }
    }

    if (attn_body_) {
      const auto embedding_size = weights_.ip_emb_b.size();
      assert(embedding_size > 0);
      const auto input_size =
          num_res_blocks == 0 ? input_channels : weights_.input.biases.size();

      if (num_res_blocks == 0) {
        // No residual means pure transformer, so process input position
        // encoding.
        if (is_pe_dense_embedding_) {
          // NCHW to NHWC conversion of 12-channel slice of input.
          for (auto batch = size_t{0}; batch < batch_size; batch++) {
            for (auto i = 0; i < kSquares; i++) {
              for (size_t j = 0; j < 12; j++) {
                buffer3[batch * kSquares * 12 + i * 12 + j] =
                    buffer1[batch * kSquares * kInputPlanes + j * kSquares + i];
              }
            }
          }
          // Dense embedding preprocess layer.
          FullyConnectedLayer<use_eigen>::Forward1D(
              batch_size, kSquares * 12, weights_.ip_emb_preproc_b.size(),
              buffer3.data(), weights_.ip_emb_preproc_w.data(),
              weights_.ip_emb_preproc_b.data(), ACTIVATION_NONE,
              buffer2.data());
        }

        // Preprocess for attention body.
        for (auto batch = size_t{0}; batch < batch_size; batch++) {
          for (auto i = 0; i < kSquares; i++) {
            // Input NCHW to NHWC conversion.
            for (size_t j = 0; j < kInputPlanes; j++) {
              buffer3[batch * kSquares * input_size + i * input_size + j] =
                  buffer1[batch * kSquares * kInputPlanes + j * kSquares + i];
            }
            // Position encoding concat.
            if (is_pe_dense_embedding_) {
              for (size_t j = kInputPlanes; j < input_size; j++) {
                buffer3[batch * kSquares * input_size + i * input_size + j] =
                    buffer2[batch * kSquares * enc_channels + i * enc_channels +
                            j - kInputPlanes];
              }
            } else {
              for (size_t j = kInputPlanes; j < input_size; j++) {
                buffer3[batch * kSquares * input_size + i * input_size + j] =
                    kPosEncoding[i][j - kInputPlanes];
              }
            }
          }
        }
      }

      // Input embedding.
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, input_size, embedding_size, buffer3.data(),
          weights_.ip_emb_w.data(), weights_.ip_emb_b.data(),
          default_activation_, buffer1.data());

      // Layer norm for new encoding.
      if (is_pe_dense_embedding_) {
        LayerNorm2DWithSkipConnection(
            batch_size * kSquares, embedding_size, buffer1.data(), 1.0f,
            (const float*)nullptr, weights_.ip_emb_ln_gammas.data(),
            weights_.ip_emb_ln_betas.data(), 1e-3);
      }

      // Input gating
      if (weights_.ip_mult_gate.size() > 0 && weights_.ip_add_gate.size() > 0) {
        int idx;
        for (auto batch = size_t{0}; batch < batch_size; batch++) {
          for (auto i = 0; i < kSquares; i++) {
            for (size_t j = 0; j < embedding_size; j++) {
              idx = batch * kSquares * embedding_size + i * embedding_size + j;
              buffer1[idx] =
                  buffer1[idx] * weights_.ip_mult_gate[j * kSquares + i] +
                  weights_.ip_add_gate[j * kSquares + i];
            }
          }
        };
      }

      float alpha = (float)pow(2.0 * weights_.encoder.size(), -0.25);

      // FFN in embedding for new encoding.
      if (is_pe_dense_embedding_) {
        const auto dff_size = weights_.ip_emb_ffn.dense1_b.size();
        // FFN dense 1.
        FullyConnectedLayer<use_eigen>::Forward1D(
            batch_size * kSquares, embedding_size, dff_size, buffer1.data(),
            weights_.ip_emb_ffn.dense1_w.data(),
            weights_.ip_emb_ffn.dense1_b.data(), ffn_activation_,
            buffer3.data());

        // FFN dense 2.
        FullyConnectedLayer<use_eigen>::Forward1D(
            batch_size * kSquares, dff_size,
            weights_.ip_emb_ffn.dense2_b.size(), buffer3.data(),
            weights_.ip_emb_ffn.dense2_w.data(),
            weights_.ip_emb_ffn.dense2_b.data(), ACTIVATION_NONE,
            buffer2.data());

        // Layer Norm.
        LayerNorm2DWithSkipConnection(
            batch_size * kSquares, weights_.ip_emb_ffn.dense2_b.size(),
            buffer2.data(), alpha, buffer1.data(),
            weights_.ip_emb_ffn_ln_gammas.data(),
            weights_.ip_emb_ffn_ln_betas.data(), 1e-3);

        std::swap(buffer1, buffer2);
      }

      // Attention body encoders.
      // LC0_DUMP_BODY=<prefix> writes the embedding output and each encoder's
      // output as raw floats, so another backend's dumps can be diffed
      // against them to find which layer diverges. Debug aid only.
      const char* dump_prefix = getenv("LC0_DUMP_BODY");
      auto dump_stage = [&](const char* stage, const std::vector<float>& buf) {
        if (!dump_prefix) return;
        const std::string path =
            std::string(dump_prefix) + ".blas." + stage + ".bin";
        const size_t n = std::min<size_t>(buf.size(), 64 * 1024);
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(buf.data()),
                n * sizeof(float));
      };
      dump_stage("emb", buffer1);
      int dump_index = 0;
      for (auto& layer : weights_.encoder) {
        ForwardEncoderLayer(buffer1, buffer2, buffer3, head_buffer, batch_size,
                            layer, embedding_size, weights_.encoder_head_count,
                            smolgen_activation_, ffn_activation_, alpha,
                            is_pe_dense_embedding_ ? 1e-3 : 1e-6);
        dump_stage(("enc" + std::to_string(dump_index++)).c_str(), buffer1);
      }
    }

    // Preserve buffer1 and buffer2, used for policy and moves left heads.
    // Value head
    if (attn_body_) {
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, weights_.ip_emb_b.size(),
          num_value_input_planes, buffer1.data(), value_head.ip_val_w.data(),
          value_head.ip_val_b.data(), default_activation_, head_buffer.data());
    } else {
      Convolution1<use_eigen>::Forward(
          batch_size, output_channels, num_value_input_planes, buffer2.data(),
          value_head.value.weights.data(), head_buffer.data());

      BiasActivate(batch_size, num_value_input_planes, &head_buffer[0],
                   value_head.value.biases.data(), default_activation_);
    }

    FullyConnectedLayer<use_eigen>::Forward1D(
        batch_size, num_value_input_planes * kSquares, num_value_channels,
        head_buffer.data(), value_head.ip1_val_w.data(),
        value_head.ip1_val_b.data(),
        default_activation_,  // Activation On
        buffer3.data());

    // Now get the score
    if (wdl_) {
      std::vector<float> wdl(3 * batch_size);
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size, num_value_channels, 3, buffer3.data(),
          value_head.ip2_val_w.data(), value_head.ip2_val_b.data(),
          ACTIVATION_NONE,  // Activation Off
          wdl.data());

      for (size_t j = 0; j < batch_size; j++) {
        std::vector<float> wdl_softmax(3);
        SoftmaxActivation(3, &wdl[j * 3], wdl_softmax.data());

        q_values_.emplace_back(wdl_softmax[0]);
        q_values_.emplace_back(wdl_softmax[1]);
        q_values_.emplace_back(wdl_softmax[2]);
      }
    } else {
      for (size_t j = 0; j < batch_size; j++) {
        double winrate = FullyConnectedLayer<use_eigen>::Forward0D(
                             num_value_channels, value_head.ip2_val_w.data(),
                             &buffer3[j * num_value_channels]) +
                         value_head.ip2_val_b[0];

        q_values_.emplace_back(std::tanh(winrate));
      }
    }

    // Moves left head.
    if (moves_left_) {
      if (attn_body_) {
        FullyConnectedLayer<use_eigen>::Forward1D(
            batch_size * kSquares, weights_.ip_emb_b.size(),
            num_moves_input_planes, buffer1.data(), weights_.ip_mov_w.data(),
            weights_.ip_mov_b.data(), default_activation_, head_buffer.data());
      } else {
        Convolution1<use_eigen>::Forward(
            batch_size, output_channels, num_moves_input_planes, buffer2.data(),
            weights_.moves_left.weights.data(), head_buffer.data());

        BiasActivate(batch_size, num_moves_input_planes, &head_buffer[0],
                     weights_.moves_left.biases.data(), default_activation_);
      }

      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size, num_moves_input_planes * kSquares, num_moves_channels,
          head_buffer.data(), weights_.ip1_mov_w.data(),
          weights_.ip1_mov_b.data(),
          default_activation_,  // Activation On
          buffer3.data());

      std::vector<float> output_moves_left(batch_size);
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size, num_moves_channels, 1, buffer3.data(),
          weights_.ip2_mov_w.data(), weights_.ip2_mov_b.data(),
          ACTIVATION_RELU,  // Specifically Relu
          &m_values_[start]);
    }

    // Policy head.
    if (attn_policy_) {
      if (!attn_body_) {
        // NCHW to NHWC conversion.
        for (auto batch = size_t{0}; batch < batch_size; batch++) {
          for (auto i = 0; i < kSquares; i++) {
            for (size_t j = 0; j < output_channels; j++) {
              buffer1[batch * kSquares * output_channels + i * output_channels +
                      j] = buffer2[batch * kSquares * output_channels +
                                   j * kSquares + i];
            }
          }
        }
      }
      const size_t policy_embedding_size = policy_head.ip_pol_b.size();
      // Policy Embedding.
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, output_channels, policy_embedding_size,
          buffer1.data(), policy_head.ip_pol_w.data(),
          policy_head.ip_pol_b.data(),
          attn_body_
              ? default_activation_
              : ACTIVATION_SELU,  // SELU activation hardcoded for apmish nets.
          buffer2.data());

      const size_t policy_d_model =
          policy_head.ip2_pol_w.size() / policy_embedding_size;

      for (auto& layer : policy_head.pol_encoder) {
        ForwardEncoderLayer(
            buffer2, buffer1, buffer3, head_buffer, batch_size, layer,
            policy_embedding_size, policy_head.pol_encoder_head_count,
            attn_body_ ? smolgen_activation_ : ACTIVATION_NONE,
            attn_body_ ? ffn_activation_ : ACTIVATION_SELU, 1.0f, 1e-6);
      }

      // Q
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, policy_embedding_size, policy_d_model,
          buffer2.data(), policy_head.ip2_pol_w.data(),
          policy_head.ip2_pol_b.data(), ACTIVATION_NONE, buffer1.data());
      // K
      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size * kSquares, policy_embedding_size, policy_d_model,
          buffer2.data(), policy_head.ip3_pol_w.data(),
          policy_head.ip3_pol_b.data(), ACTIVATION_NONE, buffer3.data());
      const float scaling = 1.0f / sqrtf(policy_d_model);
      for (auto batch = size_t{0}; batch < batch_size; batch++) {
        const float* A = &buffer1[batch * 64 * policy_d_model];
        const float* B = &buffer3[batch * 64 * policy_d_model];
        float* C = &head_buffer[batch * (64 * 64 + 8 * 24)];
        if (use_eigen) {
          auto C_mat = EigenMatrixMap<float>(C, kSquares, kSquares);
          C_mat.noalias() =
              scaling *
              ConstEigenMatrixMap<float>(B, policy_d_model, kSquares)
                  .transpose() *
              ConstEigenMatrixMap<float>(A, policy_d_model, kSquares);
        } else {
#ifdef USE_BLAS
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, kSquares,
                      kSquares, policy_d_model, scaling, A, policy_d_model, B,
                      policy_d_model, 0.0f, C, 64);
#else
          // Should never get here.
          throw Exception("Blas backend internal error");
#endif
        }
      }
      // Promotion offset calculation.
      for (auto batch = size_t{0}; batch < batch_size; batch++) {
        float promotion_offsets[4][8];
        // This is so small that SGEMM seems slower.
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 8; j++) {
            float sum = 0;
            for (size_t k = 0; k < policy_d_model; k++) {
              sum += buffer3[batch * kSquares * policy_d_model +
                             (56 + j) * policy_d_model + k] *
                     policy_head.ip4_pol_w[i * policy_d_model + k];
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
              head_buffer[batch * (64 * 64 + 8 * 24) + 64 * 64 + 24 * k +
                          3 * j + i] = head_buffer[batch * (64 * 64 + 8 * 24) +
                                                   (48 + k) * 64 + 56 + j] +
                                       promotion_offsets[i][j];
            }
          }
        }
      }
      // Mapping from attention policy to lc0 policy
      for (auto batch = size_t{0}; batch < batch_size; batch++) {
        std::vector<float> policy(num_output_policy);
        for (auto i = 0; i < 64 * 64 + 8 * 24; i++) {
          auto j = kAttnPolicyMap[i];
          if (j >= 0) {
            policy[j] = head_buffer[batch * (64 * 64 + 8 * 24) + i];
          }
        }
        policies_.emplace_back(std::move(policy));
      }
    } else if (conv_policy_) {
      assert(!attn_body_);  // not supported with attention body
      convolve3.Forward(batch_size, output_channels, output_channels,
                        buffer2.data(), policy_head.policy1.weights.data(),
                        buffer1.data());

      BiasActivate(batch_size, output_channels, buffer1.data(),
                   policy_head.policy1.biases.data(), default_activation_);

      convolve3.Forward(batch_size, output_channels, num_policy_input_planes,
                        buffer1.data(), policy_head.policy.weights.data(),
                        head_buffer.data());

      BiasActivate(batch_size, num_policy_input_planes, head_buffer.data(),
                   policy_head.policy.biases.data(), ACTIVATION_NONE);

      // Mapping from convolutional policy to lc0 policy
      for (auto batch = size_t{0}; batch < batch_size; batch++) {
        std::vector<float> policy(num_output_policy);
        for (auto i = 0; i < kPolicyUsedPlanes * kSquares; i++) {
          auto j = kConvPolicyMap[i];
          if (j >= 0) {
            policy[j] =
                head_buffer[batch * num_policy_input_planes * kSquares + i];
          }
        }
        policies_.emplace_back(std::move(policy));
      }

    } else {
      assert(!attn_body_);  // not supported with attention body
      Convolution1<use_eigen>::Forward(
          batch_size, output_channels, num_policy_input_planes, buffer2.data(),
          policy_head.policy.weights.data(), head_buffer.data());

      BiasActivate(batch_size, num_policy_input_planes, &head_buffer[0],
                   policy_head.policy.biases.data(), default_activation_);

      FullyConnectedLayer<use_eigen>::Forward1D(
          batch_size, num_policy_input_planes * kSquares, num_output_policy,
          head_buffer.data(), policy_head.ip_pol_w.data(),
          policy_head.ip_pol_b.data(),
          ACTIVATION_NONE,  // Activation Off
          buffer3.data());

      for (size_t j = 0; j < batch_size; j++) {
        std::vector<float> policy(num_output_policy);

        // Get the moves
        policy.assign(buffer3.begin() + j * num_output_policy,
                      buffer3.begin() + (j + 1) * num_output_policy);
        policies_.emplace_back(std::move(policy));
      }
    }
  }
  network_->ReleaseBuffers(std::move(buffers));
}

template <bool use_eigen>
void BlasComputation<use_eigen>::EncodePlanes(const InputPlanes& sample,
                                              float* buffer) {
  for (const InputPlane& plane : sample) {
    const float value = plane.value;
    for (auto i = 0; i < kSquares; i++)
      *(buffer++) = (plane.mask & (((uint64_t)1) << i)) != 0 ? value : 0;
  }
}

template <bool use_eigen>
BlasNetwork<use_eigen>::BlasNetwork(const WeightsFile& file,
                                    const OptionsDict& options)
    : capabilities_{file.format().network_format().input(),
                    file.format().network_format().output(),
                    file.format().network_format().moves_left()},
      weights_(file.weights()) {
  Numa::Init();

  max_batch_size_ =
      static_cast<size_t>(options.GetOrDefault<int>("batch_size", 256));
  threads_ = options.GetOrDefault<int>("threads", 1);

  auto nf = file.format().network_format();
  using NF = pblczero::NetworkFormat;
  wdl_ = nf.value() == NF::VALUE_WDL;

  moves_left_ = (nf.moves_left() == NF::MOVES_LEFT_V1) &&
                options.GetOrDefault<bool>("mlh", true);

  conv_policy_ = nf.policy() == NF::POLICY_CONVOLUTION;

  attn_policy_ = nf.policy() == NF::POLICY_ATTENTION;

  attn_body_ = nf.network() == NF::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT ||
               nf.network() == NF::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT ||
               nf.network() == NF::NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT;

  kda_directions_.assign(nf.kda_directions().begin(),
                         nf.kda_directions().end());
  for (const int direction : kda_directions_) {
    if (direction < 1 || direction > 16) {
      throw Exception("Unsupported KDA traversal direction.");
    }
  }

  default_activation_ = nf.default_activation() == NF::DEFAULT_ACTIVATION_MISH
                            ? ACTIVATION_MISH
                            : ACTIVATION_RELU;

  is_pe_dense_embedding_ =
      static_cast<InputEmbedding>(
          file.format().network_format().input_embedding()) ==
      InputEmbedding::INPUT_EMBEDDING_PE_DENSE;

  // Both embedding LayerNorm calls below are selected by this flag and read
  // their gamma/beta unconditionally, so validate them against the widths
  // their own consumers use before any of that runs.
  ValidateEmbeddingNormWeights(weights_, is_pe_dense_embedding_);

  if (attn_body_) {
    const auto smol_act = nf.smolgen_activation();
    smolgen_activation_ = smol_act == NF::ACTIVATION_DEFAULT
                              ? default_activation_
                              : static_cast<ActivationFunction>(smol_act);
    const auto ffn_act = nf.ffn_activation();
    ffn_activation_ = ffn_act == NF::ACTIVATION_DEFAULT
                          ? default_activation_
                          : static_cast<ActivationFunction>(ffn_act);
  }

  if (max_batch_size_ > kHardMaxBatchSize) {
    max_batch_size_ = kHardMaxBatchSize;
  }

  const auto inputChannels = kInputPlanes;
  const auto channels = static_cast<int>(weights_.input.biases.size());
  const auto residual_blocks = weights_.residual.size();

  weights_.input.weights =
      WinogradFilterTransformF(weights_.input.weights, channels, inputChannels);

  // residual blocks
  for (size_t i = 0; i < residual_blocks; i++) {
    auto& residual = weights_.residual[i];
    auto& conv1 = residual.conv1;
    auto& conv2 = residual.conv2;

    conv1.weights = WinogradFilterTransformF(conv1.weights, channels, channels);
    conv2.weights = WinogradFilterTransformF(conv2.weights, channels, channels);
  }

  policy_head_ = options.GetOrDefault<std::string>("policy_head", "vanilla");
  // Check that selected policy head exists.
  if (weights_.policy_heads.count(policy_head_) == 0) {
    throw Exception("The policy head you specified '" + policy_head_ +
                    "' does not exist in this net.");
  }

  value_head_ = options.GetOrDefault<std::string>("value_head", "winner");
  // Check that selected value head exists.
  if (weights_.value_heads.count(value_head_) == 0) {
    throw Exception("The value head you specified '" + value_head_ +
                    "' does not exist in this net.");
  }

  if (conv_policy_) {
    auto& policy_head = weights_.policy_heads.at("vanilla");
    policy_head.policy1.weights = WinogradFilterTransformF(
        policy_head.policy1.weights, channels, channels);
    auto pol_channels = policy_head.policy.biases.size();
    policy_head.policy.weights = WinogradFilterTransformF(
        policy_head.policy.weights, pol_channels, channels);
  }

  if (use_eigen) {
    CERR << "Using Eigen version " << EIGEN_WORLD_VERSION << "."
         << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION;
    CERR << "Eigen max batch size is " << max_batch_size_ << ".";
  } else {
#ifdef USE_OPENBLAS
    int num_procs = openblas_get_num_procs();
    openblas_set_num_threads(threads_);
    const char* core_name = openblas_get_corename();
    const char* config = openblas_get_config();
    CERR << "BLAS vendor: OpenBLAS.";
    CERR << "OpenBLAS [" << config << "].";
    CERR << "OpenBLAS found " << num_procs << " " << core_name << " core(s).";
#endif

#ifdef USE_MKL
    mkl_set_num_threads(threads_);
    CERR << "BLAS vendor: MKL.";
    constexpr int len = 256;
    char versionbuf[len];
    mkl_get_version_string(versionbuf, len);
    CERR << "MKL " << versionbuf << ".";
    MKLVersion version;
    mkl_get_version(&version);
    CERR << "MKL platform: " << version.Platform
         << ", processor: " << version.Processor << ".";
#endif

#ifdef USE_DNNL
    const dnnl_version_t* ver = dnnl_version();
    CERR << "BLAS functions from DNNL version " << ver->major << "."
         << ver->minor << "." << ver->patch;
#endif

#ifdef USE_ACCELERATE
    CERR << "BLAS vendor: Apple vecLib.";
#endif
    CERR << "BLAS max batch size is " << max_batch_size_ << ".";
  }
}

template <bool use_eigen>
std::unique_ptr<Network> MakeBlasNetwork(const std::optional<WeightsFile>& w,
                                         const OptionsDict& options) {
  if (!w) {
    throw Exception("The " + std::string(use_eigen ? "eigen" : "blas") +
                    " backend requires a network file.");
  }
  const WeightsFile& weights = *w;
  auto nf = weights.format().network_format();
  using NF = pblczero::NetworkFormat;
  switch (nf.network()) {
    case NF::NETWORK_CLASSICAL_WITH_HEADFORMAT:
    case NF::NETWORK_SE_WITH_HEADFORMAT:
    case NF::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT:
    case NF::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT:
    case NF::NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT:
      break;
    default:
      throw Exception("Network format " +
                      NF::NetworkStructure_Name(nf.network()) +
                      " is not supported by the BLAS backend.");
  }
  switch (nf.policy()) {
    case NF::POLICY_CLASSICAL:
    case NF::POLICY_CONVOLUTION:
    case NF::POLICY_ATTENTION:
      break;
    default:
      throw Exception("Policy format " + NF::PolicyFormat_Name(nf.policy()) +
                      " is not supported by the BLAS backend.");
  }
  switch (nf.value()) {
    case NF::VALUE_CLASSICAL:
    case NF::VALUE_WDL:
      break;
    default:
      throw Exception("Value format " + NF::ValueFormat_Name(nf.value()) +
                      " is not supported by the BLAS backend.");
  }
  switch (nf.moves_left()) {
    case NF::MOVES_LEFT_NONE:
    case NF::MOVES_LEFT_V1:
      break;
    default:
      throw Exception("Moves left head format " +
                      NF::MovesLeftFormat_Name(nf.moves_left()) +
                      " is not supported by the BLAS backend.");
  }
  switch (nf.default_activation()) {
    case NF::DEFAULT_ACTIVATION_RELU:
    case NF::DEFAULT_ACTIVATION_MISH:
      break;
    default:
      throw Exception("Default activation " +
                      NF::DefaultActivation_Name(nf.default_activation()) +
                      " is not supported by the BLAS backend.");
  }
  switch (nf.input_embedding()) {
    case NF::INPUT_EMBEDDING_NONE:
    case NF::INPUT_EMBEDDING_PE_MAP:
    case NF::INPUT_EMBEDDING_PE_DENSE:
      break;
    default:
      throw Exception("Input embedding " +
                      NF::InputEmbeddingFormat_Name(nf.input_embedding()) +
                      " is not supported by the BLAS backend.");
  }
  return std::make_unique<BlasNetwork<use_eigen>>(weights, options);
}

#ifdef USE_BLAS
REGISTER_NETWORK("blas", MakeBlasNetwork<false>, 50)
#endif
REGISTER_NETWORK("eigen", MakeBlasNetwork<true>, 49)

}  // namespace
}  // namespace lczero
