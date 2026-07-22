/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors
  Copyright (C) 2023 Intel Corporation

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
   
  SPDX-License-Identifier:GNU General Public License v3.0 or later
*/

#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <memory>

#include "sycl_common.h"
#include "neural/backends/shared/activation.h"
#include "neural/network_legacy.h"

namespace lczero {
namespace sycldnn_backend {
/**
 * @brief Base class for all neural network layers in the SYCL backend.
 * Layer objects hold memory for weights and biases on the GPU device;
 * input/output tensor memory is passed per Eval call.
 */
template <typename DataType>
class BaseLayer {
 public:
  int GetC() const { return C; }
  int GetH() const { return H; }
  int GetW() const { return W; }
  sycl::queue& GetSycl_Queue() { return sycl_queue_;}

  bool isNHWC() const { return nhwc_; }

  BaseLayer(int c, int h, int w, BaseLayer* ip, sycl::queue& sycl_queue);
  BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc, sycl::queue& sycl_queue);
  virtual ~BaseLayer() = default;

  /**
   * @brief Returns output buffer size in bytes for batch size N.
   * @param N Batch size.
   */
  size_t GetOutputSize(int N) const { return sizeof(DataType) * N * C * H * W; }

  /**
   * @brief Evaluates the layer for batch size N.
   * @param N Batch size.
   * @param output Pointer to output device memory.
   * @param input Pointer to input device memory.
   * @param input2 Optional skip-connection input memory.
   * @param scratch Scratch device memory pointer.
   * @param scratch_size Size of scratch memory in bytes.
   * @param sycl_queue SYCL queue to execute operations on.
   */
  virtual void Eval(int N, DataType* output, const DataType* input,
                    const DataType* input2, void* scratch, size_t scratch_size,
                    sycl::queue &sycl_queue, DataType*** = nullptr) = 0;

 protected:
  BaseLayer* input_;
  sycl::queue& sycl_queue_;

  int C;  // Output tensor dimensions.
  int H;
  int W;

  bool nhwc_;  // tensor layout

  void cublasRowMajorMatrixMul(const DataType* A, const DataType* B,
                               DataType* Out, int M, int N, int K,
                               int batchSize, sycl::queue &sycl_queue);
};

/**
 * @brief Fully Connected (FC) Layer implementation.
 */
template <typename DataType>
class FCLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::nhwc_;
  using BaseLayer<DataType>::sycl_queue_;

 public:
  FCLayer(BaseLayer<DataType>* ip, int C, int H, int W, bool bias,
          ActivationFunction activation, sycl::queue &sycl_queue);
  ~FCLayer();

  void LoadWeights(float* cpuWeight, float* cpuBias, void* scratch);

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            sycl::queue &sycl_queue, DataType*** = nullptr) override;

 private:
  const bool use_bias_;
  const ActivationFunction act_;
  DataType* weights_ = nullptr;
  DataType* biases_ = nullptr;
};

/**
 * @brief Policy Mapping Layer converting tensor representations to policy logits.
 */
template <typename DataType>
class PolicyMapLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::nhwc_;
  using BaseLayer<DataType>::sycl_queue_;

 public:
  PolicyMapLayer(BaseLayer<DataType>* ip, int C, int H, int W, int usedSize,
                 bool attention, sycl::queue &sycl_queue);
  ~PolicyMapLayer();

  void LoadWeights(const short* cpuWeight, void* scratch);

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size, sycl::queue &sycl_queue,
            DataType*** = nullptr) override;

 private:
  int used_size_;  // Size of the input without padding
  const bool attention_map_;
  short* weights_ = nullptr;
};

/**
 * @brief Squeeze-and-Excitation (SE) Layer implementation.
 */
template <typename DataType>
class SELayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::nhwc_;
  using BaseLayer<DataType>::sycl_queue_;

 public:
  SELayer(BaseLayer<DataType>* ip, int numFc1Out, bool addPrevLayerBias,
          ActivationFunction activation, sycl::queue &sycl_queue);
  ~SELayer();

  void LoadWeights(float* w1, float* b1, float* w2, float* b2,
                   float* prevLayerBias, void* scratch);

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            sycl::queue &sycl_queue, DataType*** = nullptr) override;

 private:
  DataType* w1_ = nullptr;
  DataType* w1_t_ = nullptr;  // transposed copy used by fused SE kernel
  DataType* b1_ = nullptr;
  DataType* w2_ = nullptr;
  DataType* w2_t_ = nullptr;
  DataType* b2_ = nullptr;
  DataType* bPrev_ = nullptr;
  int numFc1Out_;
  bool addPrevLayerBias_;
  const ActivationFunction act_;
};

/**
 * @brief Multi-pass Winograd Convolution fused with optional Squeeze-and-Excitation.
 */
template <typename DataType>
class FusedWinogradConvSELayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::nhwc_;
  using BaseLayer<DataType>::sycl_queue_;

 public:
  FusedWinogradConvSELayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           int Cin, ActivationFunction activation, bool bias,
                           bool skipAdd, bool se, int se_k, 
                           sycl::queue &sycl_queue, bool op_nhcw = false);

  ~FusedWinogradConvSELayer();
  void LoadWeights(float* pfilter, float* pBias, void* scratch);
  void LoadSEWeights(float* w1, float* b1, float* w2, float* b2, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            sycl::queue &sycl_queue, DataType*** = nullptr) override;

 private:
  const int c_input_;
  const ActivationFunction act_;
  const bool use_bias_;
  const bool skip_add_;
  const bool has_se_;
  const int se_k_;
  const bool op_nhcw_;

  DataType* biases_ = nullptr;
  DataType* transformed_weights_ = nullptr;  // After winograd transform.

  // Weights and Biases for (optional) SE.
  DataType* w1_ = nullptr;
  DataType* w2_ = nullptr;
  DataType* b1_ = nullptr;
  DataType* b2_ = nullptr;
};

template <typename DataType>
class Conv1Layer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::nhwc_;
  using BaseLayer<DataType>::sycl_queue_;

 public:
  Conv1Layer(BaseLayer<DataType>* ip, int C, int H, int W, int Cin,
             ActivationFunction activation, bool bias, sycl::queue &sycl_queue);

  ~Conv1Layer();
  void LoadWeights(float* pfilter, float* pBias, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            sycl::queue &sycl_queue, DataType*** = nullptr) override;

 private:
  const int c_input_;
  const ActivationFunction act_;
  const bool use_bias_;

  DataType* biases_ = nullptr;
  DataType* weights_ = nullptr;

  void cublasSpecialMatrixMul(const DataType* A, const DataType* B,
                              DataType* Out, int M, int N, int K, int batchSize, sycl::queue &sycl_queue);
};

template <typename DataType>
class ResidualBlock : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::sycl_queue_;

 public:
  ResidualBlock(BaseLayer<DataType>* ip, int C, bool se, int se_k,
                bool first, bool last,
                ActivationFunction activation, int shared_mem_size, sycl::queue &sycl_queue);

  ~ResidualBlock();
  void LoadWeights0(float* pfilter, float* pBias, void* scratch);
  void LoadWeights1(float* pfilter, float* pBias, void* scratch);
  void LoadSEWeights(float* w1, float* b1, float* w2, float* b2, void* scratch);

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            sycl::queue &sycl_queue, DataType*** = nullptr) override;

 private:
  const bool has_se_;
  const int se_k_;
  const int c_input_;
  const bool first_block_;
  const bool last_block_;
  const int shared_mem_size_;
  const ActivationFunction act_;

  DataType* biases0_ = nullptr;
  DataType* biases1_ = nullptr;
  DataType* transformed_weights0_ = nullptr;
  DataType* transformed_weights1_ = nullptr;

  DataType* w1_ = nullptr;
  DataType* w2_ = nullptr;
  DataType* b1_ = nullptr;
  DataType* b2_ = nullptr;
};

/**
 * @brief Single Transformer Encoder block (Multi-Head Attention + Feed Forward Network).
 */
template <typename DataType>
class EncoderBlock {
 public:
  EncoderBlock(const MultiHeadWeights::EncoderLayer& cpu_weights, void* scratch,
               int heads, int size, float alpha,
               DataType* smolgen_global_scratch, int smolgen_global_size,
               int max_batch_size, ActivationFunction smolgen_act,
               ActivationFunction ffn_act, float default_eps, sycl::queue &sycl_queue);
  ~EncoderBlock();

  void Eval(int N, DataType* inpop, DataType* scratch0, DataType* scratch1,
            DataType* scratch2, sycl::queue &sycl_queue,
            DataType*** offset_pointers);

 private:
  // GPU side device memory pointers
  DataType *mha_q_w = nullptr, *mha_q_b = nullptr;
  DataType *mha_k_w = nullptr, *mha_k_b = nullptr;
  DataType *mha_v_w = nullptr, *mha_v_b = nullptr;
  DataType *mha_qkv_w = nullptr, *mha_qkv_b = nullptr;
  DataType *mha_dense_w = nullptr, *mha_dense_b = nullptr;

  DataType *ln1_gammas = nullptr, *ln1_betas = nullptr;

  DataType *ffn_dense1_w = nullptr, *ffn_dense1_b = nullptr;
  DataType *ffn_dense2_w = nullptr, *ffn_dense2_b = nullptr;

  DataType *ln2_gammas = nullptr, *ln2_betas = nullptr;

  DataType *smol_compress = nullptr;
  DataType *smol_dense1_w = nullptr, *smol_dense1_b = nullptr;
  DataType *smol_dense2_w = nullptr, *smol_dense2_b = nullptr;
  DataType *smol_ln1_gammas = nullptr, *smol_ln1_betas = nullptr;
  DataType *smol_ln2_gammas = nullptr, *smol_ln2_betas = nullptr;
  DataType *smol_global = nullptr;

  int mha_q_size_;
  int mha_k_size_;
  int mha_v_size_;
  int mha_dense_size_;

  int ffn_dense1_size_;
  int ffn_dense2_size_;

  int embedding_op_size_;
  int encoder_heads_;

  float alpha_;  // scale to apply to skip connection add
  float default_eps_;  // value of epsilon where it wasn't specified in training

  const bool has_smolgen_;
  const ActivationFunction smolgen_activation_;
  const ActivationFunction ffn_activation_;

  int smol_compress_size_;
  int smol_dense_1_size_;
  int smol_dense_2_size_;
  int smol_global_size_;

  const int max_batch_size_;

  sycl::queue& sycl_queue_;
};

/**
 * @brief Attention Policy Head implementation.
 */
template <typename DataType>
class AttentionPolicyHead : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::sycl_queue_;

 public:
  AttentionPolicyHead(BaseLayer<DataType>* ip,
                      const MultiHeadWeights::PolicyHead& weights,
                      void* scratch, bool attention_body,
                      ActivationFunction act, int max_batch_size, sycl::queue &sycl_queue);

  ~AttentionPolicyHead();
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            sycl::queue &sycl_queue, DataType*** = nullptr) override;

 private:
  DataType *ip_pol_w_ = nullptr, *ip_pol_b_ = nullptr;
  DataType *ip2_pol_w_ = nullptr, *ip2_pol_b_ = nullptr;
  DataType *ip3_pol_w_ = nullptr, *ip3_pol_b_ = nullptr;
  DataType *ip4_pol_w_ = nullptr;

  DataType *wqk_w_ = nullptr, *wqk_b_ = nullptr;

  int embedding_op_size_;
  int wq_op_size_;
  int wk_op_size_;

  int encoder_heads_;
  int policy_d_model_;
  bool attention_body_;
  ActivationFunction act_;

  std::vector<std::unique_ptr<EncoderBlock<DataType>>> encoder_weights_;
};

template <typename DataType>
class EmbeddingLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::sycl_queue_;

 public:
  EmbeddingLayer(BaseLayer<DataType>* ip, const std::vector<float>& weights,
                 const std::vector<float>& biases, void* scratch,
                 ActivationFunction activation, sycl::queue &sycl_queue);
  ~EmbeddingLayer();

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            sycl::queue &sycl_queue, DataType*** = nullptr) override;

 private:
  DataType *weights_ = nullptr, *biases_ = nullptr;
  ActivationFunction act_;
};

/**
 * @brief Attention Body implementation containing transformer encoder stack.
 */
template <typename DataType>
class AttentionBody : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::sycl_queue_;

 public:
  AttentionBody(const MultiHeadWeights& weights, void* scratch,
                Activations activations, int num_res_blocks, int input_c,
                int max_batch_size, bool is_pe_dense_embedding,
                sycl::queue &sycl_queue);
  ~AttentionBody();

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            sycl::queue &sycl_queue, DataType*** = nullptr) override;

 private:
  DataType *ip_emb_pre_w_ = nullptr, *ip_emb_pre_b_ = nullptr;
  DataType *ip_emb_w_ = nullptr, *ip_emb_b_ = nullptr;
  DataType *ip_emb_ln_g_ = nullptr, *ip_emb_ln_b_ = nullptr;
  DataType *ip_mult_gate_ = nullptr, *ip_add_gate_ = nullptr;
  DataType *ip_emb_ffn_d1_w_ = nullptr, *ip_emb_ffn_d1_b_ = nullptr;
  DataType *ip_emb_ffn_d2_w_ = nullptr, *ip_emb_ffn_d2_b_ = nullptr;
  DataType *ip_emb_ffn_ln_g_ = nullptr, *ip_emb_ffn_ln_b_ = nullptr;
  DataType *smolgen_global_ = nullptr;
  bool is_pe_dense_embedding_;
  DataType *pos_encoding_ = nullptr;
  int embedding_dense_size_;
  int embedding_op_size_;
  int embedding_ffn_size_;
  int embedding_ffn_dff_;
  int encoder_head_count_;
  std::vector<std::unique_ptr<EncoderBlock<DataType>>> encoder_weights_;
  Activations activations_;
  int num_resi_blocks_;
  int input_c_;
  int smolgen_global_size_;
  const bool has_gating_;
  const bool has_smolgen_;
};

/**
 * @brief Value Head implementation (WDL/Classic).
 */
template <typename DataType>
class ValueHead : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::sycl_queue_;

 public:
  ValueHead(BaseLayer<DataType>* ip, const MultiHeadWeights::ValueHead& weights,
            void* scratch, bool attention_body, bool wdl, ActivationFunction act,
            int max_batch_size, sycl::queue &sycl_queue);
  ~ValueHead();
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            sycl::queue &sycl_queue, DataType*** = nullptr) override;

 private:
  std::unique_ptr<Conv1Layer<DataType>> conv_;

  DataType *ip_val_w_ = nullptr, *ip_val_b_ = nullptr;
  DataType *ip1_val_w_ = nullptr, *ip1_val_b_ = nullptr;
  DataType *ip2_val_w_ = nullptr, *ip2_val_b_ = nullptr;
  DataType *ip_val_err_w_ = nullptr, *ip_val_err_b_ = nullptr;

  int embedding_size_;
  int value_hidden_size_;
  bool wdl_;
  bool attention_body_;
  ActivationFunction act_;
};

}  // namespace sycldnn_backend
}  // namespace lczero
