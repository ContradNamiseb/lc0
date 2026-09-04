/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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
*/
#pragma once

// Layer classes for the DirectML backend, following the CUDA backend's
// architecture (cuda/layers.h) as closely as DirectML's execution model
// allows:
//
//   - namespace lczero::directml_backend, mirroring lczero::cudnn_backend;
//   - a BaseLayer<DataType> contract with one Eval() per layer that takes
//     (N, output, input, input2, scratch, ...) exactly like the CUDA/SYCL
//     Eval signatures -- the pointers are DmlPtrs (resource + offset) into
//     the shared arenas because D3D12 binding needs the resource identity;
//   - layers own only weights and compiled operators; activation memory is
//     provided by the caller, "The Layer objects only hold memory for
//     weights, biases, etc; memory for input and output tensors is provided
//     by caller of Eval" (cuda/layers.h);
//   - the network holds the topology as a flat vector of layers built in
//     execution order and calls Eval sequentially with tensor_mem_[3] style
//     buffer rotation.
//
// What is fundamentally different from CUDA: a DirectML layer's math is not
// hand-written kernels but a dml::Graph built once per (layer, batch size)
// and compiled into an IDMLCompiledOperator -- DirectML has no graph capture,
// so per-N compilation (lazy, cached) replaces CUDA graph replay. The two
// pieces that have no DirectML primitive stay hand-written HLSL compute
// shaders in the CUDA kernel tradition: the KDA recurrence (a sequential
// scan) and the attention policy finalize (an interleave). The
// expand-planes and attention-input-preprocess shuffles also run as HLSL
// here rather than as DML transpose graphs, mirroring the SYCL kernels they
// are ported from.

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <d3d12.h>
#include <directml.h>
#include <wrl/client.h>

#include "neural/backends/directml/dml_common.h"
#include "neural/backends/shared/activation.h"
#include "neural/network_legacy.h"

namespace lczero {
namespace directml_backend {

using Microsoft::WRL::ComPtr;

template <typename DataType>
class EncoderBlock;

// ---------------------------------------------------------------------------
// Minimal host-side fp16: lc0's other backends use cuda_fp16/sycl::half; this
// backend needs the same "storage type with float conversion" and nothing
// more (weights are converted on the host at load, activations live on the
// GPU in DML FLOAT16 tensors).
struct alignas(2) DmlHalf {
  uint16_t bits;
  DmlHalf() = default;
  explicit DmlHalf(float f);
  operator float() const;
};
static_assert(sizeof(DmlHalf) == 2);

template <typename DataType>
constexpr DML_TENSOR_DATA_TYPE DmlTensorType() {
  return std::is_same<DataType, DmlHalf>::value
             ? DML_TENSOR_DATA_TYPE_FLOAT16
             : DML_TENSOR_DATA_TYPE_FLOAT32;
}

// ---------------------------------------------------------------------------
// Collects weight tensors during layer construction and flushes them to the
// GPU in one command-list pass after the whole topology is built -- the
// DirectML equivalent of the CUDA backend's scratch-staged weight uploads
// (fp32 host data converted to DataType at upload time).
class DmlWeightUploader {
 public:
  struct Pending {
    DmlPtr dest;
    // Owned copy of the source bytes. The uploader must own them because the
    // flush happens after the whole topology is built -- borrowing the
    // caller's pointer dangles when the caller passes a temporary (this
    // exact bug shipped garbage gather indices that hung debugging for a
    // day: a ctor-local vector freed before FlushWeights ran).
    std::vector<uint8_t> owned;
    size_t bytes;
    bool is_float;  // convert fp32 -> DataType at flush; else raw bytes
  };

  explicit DmlWeightUploader(DmlArena* arena) : arena_(arena) {}

  DmlPtr Add(const std::vector<float>& floats) {
    DmlPtr p = arena_->Allocate(floats.size() * sizeof(float));
    if (!floats.empty()) {
      Pending entry;
      entry.dest = p;
      const size_t bytes = floats.size() * sizeof(float);
      entry.owned.assign(reinterpret_cast<const uint8_t*>(floats.data()),
                         reinterpret_cast<const uint8_t*>(floats.data()) + bytes);
      entry.bytes = bytes;
      entry.is_float = true;
      pending_.push_back(std::move(entry));
    }
    return p;
  }
  // Raw (already-typed) bytes, e.g. the uint32 policy-map indices.
  DmlPtr AddRaw(const void* data, size_t bytes) {
    DmlPtr p = arena_->Allocate(bytes);
    Pending entry;
    entry.dest = p;
    entry.owned.assign(static_cast<const uint8_t*>(data),
                       static_cast<const uint8_t*>(data) + bytes);
    entry.bytes = bytes;
    entry.is_float = false;
    pending_.push_back(std::move(entry));
    return p;
  }

  const std::vector<Pending>& pending() const { return pending_; }
  size_t entry_count() const { return pending_.size(); }

 private:
  DmlArena* arena_;
  std::vector<Pending> pending_;
};

// ---------------------------------------------------------------------------
// Base class for all layers: the cuda/layers.h BaseLayer contract with the
// sycl::queue replaced by a DmlExecScope (command list + transient arena).
template <typename DataType>
class BaseLayer {
 public:
  int GetC() const { return C; }
  int GetH() const { return H; }
  int GetW() const { return W; }
  size_t GetOutputSize(int N) const {
    return sizeof(DataType) * (size_t)N * C * H * W;
  }

  BaseLayer(int c, int h, int w, BaseLayer* ip) : C(c), H(h), W(w), input_(ip) {}
  virtual ~BaseLayer() = default;

  virtual void Eval(int N, DmlPtr output, DmlPtr input, DmlPtr input2,
                    DmlPtr scratch, size_t scratch_size,
                    DmlExecScope& scope) = 0;

  // Compiles this layer's per-batch-size graphs WITHOUT recording GPU work.
  // forwardEval runs this over the whole network before the first Eval:
  // on this driver, interleaving diverse-shape graph builds with bound
  // dispatches poisons later CreateOperator calls (see
  // docs/directml-handoff.md section 3), so all compilation must finish
  // before dispatching begins. Eval falls back to this when it finds no
  // compiled op, so nothing breaks if a call site skips the pre-pass.
  virtual void EnsureCompiled(int N, DmlExecScope& scope) {}

 protected:
  BaseLayer* input_;
  int C;  // Output tensor dimensions.
  int H;
  int W;
};

// Fully connected layer over a flattened [N, C*H*W] input, exactly the
// cuda/layers.h FCLayer semantics: out = act(x @ W^T + b).
template <typename DataType>
class FCLayer : public BaseLayer<DataType> {
 public:
  FCLayer(BaseLayer<DataType>* ip, int C, int H, int W, bool bias,
          ActivationFunction activation, const std::vector<float>& weights,
          const std::vector<float>& biases, DmlWeightUploader& uploader);
  void Eval(int N, DmlPtr output, DmlPtr input, DmlPtr input2, DmlPtr scratch,
            size_t scratch_size, DmlExecScope& scope) override;
  void EnsureCompiled(int N, DmlExecScope& scope) override;

 private:
  const bool use_bias_;
  const ActivationFunction act_;
  DmlPtr weights_;  // [M, K] row-major
  DmlPtr biases_;   // [M]
  std::unordered_map<int, DmlCompiledOp> compiled_;
};

// Square-embedding layer: GEMM over tokens without flattening H/W --
// [N*64, K] -> [N*64, M], the sycl/layers.h EmbeddingLayer semantics. Used
// as the moves-left embedding for attention-body nets.
template <typename DataType>
class EmbeddingLayer : public BaseLayer<DataType> {
 public:
  EmbeddingLayer(BaseLayer<DataType>* ip, int C, int H, int W, bool bias,
                 ActivationFunction activation,
                 const std::vector<float>& weights,
                 const std::vector<float>& biases,
                 DmlWeightUploader& uploader);
  void Eval(int N, DmlPtr output, DmlPtr input, DmlPtr input2, DmlPtr scratch,
            size_t scratch_size, DmlExecScope& scope) override;
  void EnsureCompiled(int N, DmlExecScope& scope) override;

 private:
  const bool use_bias_;
  const ActivationFunction act_;
  DmlPtr weights_;
  DmlPtr biases_;
  std::unordered_map<int, DmlCompiledOp> compiled_;
};

// Maps the attention policy rows ([N, 64*64 + 8*24]) or conv policy planes
// to 1858 policy logits via a uint32 gather table (the short map converted
// at load), replacing the CUDA PolicyMapLayer kernel.
template <typename DataType>
class PolicyMapLayer : public BaseLayer<DataType> {
 public:
  PolicyMapLayer(BaseLayer<DataType>* ip, int usedSize, bool attention,
                 const short* cpuWeight, DmlWeightUploader& uploader);
  void Eval(int N, DmlPtr output, DmlPtr input, DmlPtr input2, DmlPtr scratch,
            size_t scratch_size, DmlExecScope& scope) override;
  void EnsureCompiled(int N, DmlExecScope& scope) override;

 private:
  int used_size_;
  // Host-owned copy of the inverse gather table. DmlWeightUploader::AddRaw
  // only records the pointer (upload happens later in FlushWeights), so the
  // bytes must outlive layer construction -- a ctor-local vector dangles.
  std::vector<uint32_t> indices_host_;
  DmlPtr indices_;  // uint32 [kNumOutputPolicy]
  std::unordered_map<int, DmlCompiledOp> compiled_;
};

// The attention-body "embedding stage": NHWC conversion + positional-encoding
// concat (HLSL kernel, port of the SYCL preprocess kernel), the square
// embedding GEMM (+LN for dense embeddings), input gating, and for PE_DENSE
// nets the preprocess GEMM + embedding FFN + LN. Contains the encoder stack
// like sycl/layers.h AttentionBody.
template <typename DataType>
class AttentionBody : public BaseLayer<DataType> {
 public:
  AttentionBody(const MultiHeadWeights& weights, DmlWeightUploader& uploader,
                Activations activations, int num_res_blocks, int input_c,
                int max_batch_size, bool is_pe_dense_embedding,
                const std::vector<int>& kda_directions,
                DmlDeviceContext& ctx, bool fp16);
  void Eval(int N, DmlPtr output, DmlPtr input, DmlPtr input2, DmlPtr scratch,
            size_t scratch_size, DmlExecScope& scope) override;
  void EnsureCompiled(int N, DmlExecScope& scope) override;

 private:
  DmlPtr ip_emb_w_, ip_emb_b_;
  DmlPtr ip_emb_pre_w_, ip_emb_pre_b_;
  DmlPtr ip_emb_ln_g_, ip_emb_ln_b_;
  DmlPtr ip_emb_ffn_d1_w_, ip_emb_ffn_d1_b_;
  DmlPtr ip_emb_ffn_d2_w_, ip_emb_ffn_d2_b_;
  DmlPtr ip_emb_ffn_ln_g_, ip_emb_ffn_ln_b_;
  DmlPtr ip_mult_gate_, ip_add_gate_;
  // Host-owned copy of the positional-encoding table for the same lifetime
  // reason as PolicyMapLayer::indices_host_ below.
  std::vector<float> pos_encoding_host_;
  DmlPtr pos_encoding_;
  int embedding_op_size_;
  int embedding_dense_size_;
  int embedding_ffn_size_;
  int embedding_ffn_dff_;
  int input_c_;
  bool is_pe_dense_embedding_;
  bool has_gating_;
  Activations activations_;
  int num_resi_blocks_;
  std::vector<std::unique_ptr<EncoderBlock<DataType>>> encoder_weights_;

  // The preprocess HLSL pipeline state (root signature + PSO), compiled once.
  ComPtr<ID3D12RootSignature> preprocess_root_signature_;
  ComPtr<ID3D12PipelineState> preprocess_pso_;
  ComPtr<ID3D12Resource> preprocess_const_buffer_;  // persistently mapped
  void* preprocess_const_mapped_ = nullptr;

  std::unordered_map<int, DmlCompiledOp> compiled_;      // embedding graph(s)
  std::unordered_map<int, DmlCompiledOp> pre_compiled_;  // PE_DENSE pre gemm
};

// One transformer encoder block: MHA (+optional smolgen) or KDA mixer, LN1,
// FFN, LN2 -- the sycl/layers.h EncoderBlock split into a "projections"
// graph, the hand-written KDA recurrence shader (KDA only), and a
// "normalization tail" graph.
template <typename DataType>
class EncoderBlock {
 public:
  EncoderBlock(const MultiHeadWeights::EncoderLayer& cpu_weights,
               DmlWeightUploader& uploader, int heads, int size, float alpha,
               DmlPtr smolgen_global, int smolgen_global_size,
               int max_batch_size, ActivationFunction smolgen_act,
               ActivationFunction ffn_act, float default_eps,
               const std::vector<int>& kda_directions, DmlDeviceContext& ctx,
               bool fp16);

  // in_out_tensor is updated in place (input on entry, output on exit), like
  // the SYCL EncoderBlock::Eval; scratch/buffer1/buffer2 are scratch.
  void Eval(int N, DmlPtr in_out_tensor, DmlPtr scratch, DmlPtr buffer1,
            DmlPtr buffer2, DmlPtr ln_scratch, DmlExecScope& scope);

  // Two-phase compile support: builds this block's per-N graphs without
  // recording (see BaseLayer::EnsureCompiled).
  void EnsureCompiled(int N, DmlExecScope& scope);

 private:
  // Idempotent builders for the split LN/FFN tail graphs, shared by
  // EnsureCompiled and the Eval paths so the two cannot diverge.
  void BuildKdaTails(int N, DmlExecScope& scope);
  void BuildMhaTails(int N, DmlExecScope& scope);

  void EvalKda(int N, DmlPtr in_out_tensor, DmlPtr scratch, DmlPtr buffer1,
               DmlPtr buffer2, DmlPtr ln_scratch, DmlExecScope& scope);
  void EvalMha(int N, DmlPtr in_out_tensor, DmlPtr scratch, DmlPtr buffer1,
               DmlPtr buffer2, DmlPtr ln_scratch, DmlExecScope& scope);

  // MHA weights.
  DmlPtr mha_q_w_, mha_q_b_, mha_k_w_, mha_k_b_;
  DmlPtr mha_v_w_, mha_v_b_, mha_dense_w_, mha_dense_b_;
  DmlPtr ln1_gammas_, ln1_betas_;
  DmlPtr ffn_dense1_w_, ffn_dense1_b_, ffn_dense2_w_, ffn_dense2_b_;
  DmlPtr ln2_gammas_, ln2_betas_;

  // KDA weights (mirroring the MultiHeadWeights::KDA fields).
  DmlPtr kda_q_w_, kda_q_b_, kda_k_w_, kda_k_b_, kda_v_w_, kda_v_b_;
  DmlPtr kda_decay_a_w_, kda_decay_a_b_, kda_decay_b_w_, kda_decay_b_b_;
  DmlPtr kda_beta_w_, kda_beta_b_, kda_a_log_, kda_dt_bias_;
  DmlPtr kda_gate_a_w_, kda_gate_a_b_, kda_gate_b_w_, kda_gate_b_b_;
  DmlPtr kda_out_norm_gammas_;
  DmlPtr kda_dense_w_, kda_dense_b_;
  DmlPtr kda_local_conv_w_, kda_local_conv_b_;

  // Smolgen weights.
  DmlPtr smol_compress_, smol_dense1_w_, smol_dense1_b_;
  DmlPtr smol_dense2_w_, smol_dense2_b_;
  DmlPtr smol_ln1_gammas_, smol_ln1_betas_, smol_ln2_gammas_, smol_ln2_betas_;
  DmlPtr smolgen_global_;
  int smolgen_global_size_ = 0;

  int embedding_op_size_;
  int encoder_heads_;
  int mha_q_size_, ffn_dense1_size_;
  bool is_kda_;
  int kda_key_dim_, kda_value_dim_, kda_gate_rank_;
  float kda_rms_norm_epsilon_;
  bool kda_output_gate_, kda_output_rms_norm_, kda_local_conv_, kda_qkv_silu_;
  std::array<int, 16> kda_directions_{};
  int kda_direction_count_;
  float alpha_;
  float default_eps_;
  bool has_smolgen_;
  ActivationFunction smolgen_activation_, ffn_activation_;
  int smol_compress_size_, smol_dense_1_size_, smol_dense_2_size_;
  const int max_batch_size_;
  DmlDeviceContext& ctx_;
  bool fp16_;

  // Per-N compiled graphs. MHA: a qkv graph, an attention graph (dense
  // [N*H,1]-batch; the head interleave runs in the transpose shader), and
  // a tail graph per N. KDA: a projections graph and a tail graph per N,
  // with the recurrence shader in between.
  std::unordered_map<int, DmlCompiledOp> mha_qkv_compiled_;
  std::unordered_map<int, DmlCompiledOp> mha_mlp_compiled_;
  std::unordered_map<int, DmlCompiledOp> mha_mlp2_compiled_;
  std::unordered_map<int, DmlCompiledOp> mha_mlp3_compiled_;
  std::unordered_map<int, DmlCompiledOp> mha_attn_compiled_;
  // The LN/FFN tails split at each fused LayerNorm: "1" is the graph up to
  // the first LN, "2" the FFN sandwiched between the two LNs. The LNs
  // themselves are LayerNormLayer dispatches (HLSL cannot live inside a
  // dml::Graph), so an encoder is alternating GEMM graphs and kernels.
  std::unordered_map<int, DmlCompiledOp> mha_tail1_compiled_;
  std::unordered_map<int, DmlCompiledOp> mha_tail2_compiled_;
  std::unordered_map<int, DmlCompiledOp> kda_proj_compiled_;
  std::unordered_map<int, DmlCompiledOp> kda_tail1_compiled_;
  std::unordered_map<int, DmlCompiledOp> kda_tail2_compiled_;

  // The KDA recurrence compute layer (compiled once at construction).
  std::unique_ptr<class KdaRecurrenceLayer> kda_recurrence_;
  // The MHA head-split/merge transpose (compiled once at construction for
  // MHA blocks; see mha_transpose.hlsl for why a shader is needed).
  std::unique_ptr<class MhaTransposeLayer> mha_transpose_;
  std::unique_ptr<class SmolgenBiasLayer> smolgen_bias_;
  std::unique_ptr<class KdaLocalConvLayer> kda_local_conv_layer_;
  // Fused LayerNorm, shared by this block's two LN sites.
  std::unique_ptr<class LayerNormLayer> layer_norm_;
};

// Attention policy head: ip_pol embedding, optional encoder stack, wq/wk
// gemms, scaled scores, and the policy-finalize HLSL kernel that interleaves
// promotion logits. The final policy map is a separate layer (as in CUDA).
template <typename DataType>
class AttentionPolicyHead : public BaseLayer<DataType> {
 public:
  AttentionPolicyHead(BaseLayer<DataType>* ip,
                      const MultiHeadWeights::PolicyHead& weights,
                      DmlWeightUploader& uploader, bool attention_body,
                      ActivationFunction act, int max_batch_size,
                      DmlDeviceContext& ctx, bool fp16,
                      const std::vector<int>& kda_directions);
  void Eval(int N, DmlPtr output, DmlPtr input, DmlPtr input2, DmlPtr scratch,
            size_t scratch_size, DmlExecScope& scope) override;
  void EnsureCompiled(int N, DmlExecScope& scope) override;

  // Output layout is [N, 64*64 + 8*24] (the attention policy map's input).
 private:
  DmlPtr ip_pol_w_, ip_pol_b_;
  DmlPtr ip2_pol_w_, ip2_pol_b_;  // wq
  DmlPtr ip3_pol_w_, ip3_pol_b_;  // wk
  DmlPtr ip4_pol_w_;              // ppo (promotion offsets)
  int embedding_op_size_;
  int policy_d_model_;
  int encoder_heads_;
  bool attention_body_;
  ActivationFunction act_;
  std::vector<std::unique_ptr<EncoderBlock<DataType>>> encoder_weights_;

  ComPtr<ID3D12RootSignature> finalize_root_signature_;
  ComPtr<ID3D12PipelineState> finalize_pso_;
  ComPtr<ID3D12Resource> finalize_const_buffer_;
  void* finalize_const_mapped_ = nullptr;

  std::unordered_map<int, DmlCompiledOp> compiled_;  // gemms + scores
};

// Value head for attention-body nets: ip_val embedding GEMM -> ip1 GEMM ->
// ip2 GEMM (WDL [N,3] logits, softmaxed on the host like the CUDA backend;
// or classical [N,1] tanh).
template <typename DataType>
class ValueHead : public BaseLayer<DataType> {
 public:
  ValueHead(BaseLayer<DataType>* ip, const MultiHeadWeights::ValueHead& weights,
            DmlWeightUploader& uploader, bool wdl, ActivationFunction act);
  void Eval(int N, DmlPtr output, DmlPtr input, DmlPtr input2, DmlPtr scratch,
            size_t scratch_size, DmlExecScope& scope) override;
  void EnsureCompiled(int N, DmlExecScope& scope) override;

 private:
  DmlPtr ip_val_w_, ip_val_b_, ip1_val_w_, ip1_val_b_, ip2_val_w_, ip2_val_b_;
  int embedding_size_;
  int value_hidden_size_;
  bool wdl_;
  ActivationFunction act_;
  std::unordered_map<int, DmlCompiledOp> compiled_;
};

// ---------------------------------------------------------------------------
// MHA head transpose layer: one HLSL kernel (compiled once) run in two
// modes -- split token-major [T,H*D] to batch-major [B=N*H,64,D] before the
// attention GEMMs, merge back after. Record() only appends commands.
class MhaTransposeLayer {
 public:
  struct Params {
    uint32_t batch_size;  // N
    uint32_t heads;       // H
    uint32_t head_dim;    // D
    uint32_t mode;        // 0 = split, 1 = merge
  };

  MhaTransposeLayer(ID3D12Device* device, bool fp16);

  void Record(ID3D12GraphicsCommandList* command_list, const Params& params,
              DmlPtr input, DmlPtr output);

 private:
  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12RootSignature> root_signature_;
  ComPtr<ID3D12PipelineState> pso_;
  bool fp16_;
};

// KDA local 3x3 depthwise board conv + residual
// (shaders/kda_local_conv.hlsl) -- see SmolgenBiasLayer for the
// build-once/record-only contract.
class KdaLocalConvLayer {
 public:
  struct Params {
    uint32_t tokens;  // N * 64
    uint32_t emb;     // embedding width
  };

  KdaLocalConvLayer(ID3D12Device* device, bool fp16);

  void Record(ID3D12GraphicsCommandList* command_list, const Params& params,
              DmlPtr input, DmlPtr weights, DmlPtr bias, DmlPtr output);

 private:
  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12RootSignature> root_signature_;
  ComPtr<ID3D12PipelineState> pso_;
  bool fp16_;
};

// Smolgen generated-bias matmul (shaders/smolgen_bias.hlsl): computes
// bias[n,h,i] = sum_g table[h,i,g] * d2[n,h,g] -- the per-(batch, head)
// attention bias -- as a hand-written kernel, because the equivalent DML
// graph needs a 5-D broadcast GEMM operand this driver rejects. Same
// build-once/record-only contract as MhaTransposeLayer.
class SmolgenBiasLayer {
 public:
  struct Params {
    uint32_t batch;  // N
    uint32_t heads;  // H
    uint32_t gen;    // per-head generated-vector width
  };

  SmolgenBiasLayer(ID3D12Device* device, bool fp16);

  void Record(ID3D12GraphicsCommandList* command_list, const Params& params,
              DmlPtr table, DmlPtr d2, DmlPtr bias_out);

 private:
  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12RootSignature> root_signature_;
  ComPtr<ID3D12PipelineState> pso_;
  bool fp16_;
};

// Fused layer normalization (shaders/layer_norm.hlsl):
//   y = gamma * (act(input + bias) * alpha + skip - mean) / sqrt(var + eps)
//       + beta
// replacing the ~10-node composed DML expression LayerNormExpr builds. Each
// of those nodes is a separate dispatch streaming the whole [rows, channels]
// tensor, so the composed form costs an order of magnitude more memory
// traffic than the arithmetic warrants; this is one dispatch. Same
// build-once/record-only contract as SmolgenBiasLayer.
//
// Because HLSL cannot live inside a dml::Graph, every call site splits its
// graph here -- an encoder becomes alternating GEMM graphs and kernels,
// which is the shape the SYCL backend already has.
class LayerNormLayer {
 public:
  struct Params {
    uint32_t rows;      // token count, N * 64
    uint32_t channels;  // normalized dimension
    bool has_bias;      // fold the previous gemm's bias in
    bool has_skip;      // residual added after alpha
    ActivationFunction act;
    float alpha;
    float eps;
  };

  LayerNormLayer(ID3D12Device* device, bool fp16);

  // bias/skip may be empty DmlPtrs when the matching Params flag is false;
  // they are then bound to `input` so the root SRV stays valid (a null root
  // SRV removes the device on this driver).
  void Record(ID3D12GraphicsCommandList* command_list, const Params& params,
              DmlPtr input, DmlPtr bias, DmlPtr skip, DmlPtr gammas,
              DmlPtr betas, DmlPtr output);

 private:
  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12RootSignature> root_signature_;
  ComPtr<ID3D12PipelineState> pso_;
  bool fp16_;
};

// ---------------------------------------------------------------------------
// KDA recurrence layer (see test_kda_recurrence.cc). Construction compiles
// the HLSL once, specialised to this net's (key_dim, value_dim); Record()
// only appends commands.
class KdaRecurrenceLayer {
 public:
  struct Params {
    uint32_t batch_size;
    uint32_t heads;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t direction_count;
    std::array<int32_t, 16> directions;  // exactly 16 entries
    bool use_fused_qkv;
    uint32_t qkv_stride;
    float log_decay_floor;
    bool fp16;
  };

  // key_dim/value_dim are baked into the shader at PSO creation, so one
  // instance serves exactly one geometry -- Record() rejects any other.
  KdaRecurrenceLayer(ID3D12Device* device, bool fp16, uint32_t key_dim,
                     uint32_t value_dim);

  void Record(ID3D12GraphicsCommandList* command_list, const Params& params,
              DmlPtr qkv, DmlPtr q, DmlPtr k, DmlPtr v, DmlPtr raw_decay,
              DmlPtr dt_bias, DmlPtr a_log, DmlPtr beta, DmlPtr mixed_out);

 private:
  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12RootSignature> root_signature_;
  ComPtr<ID3D12PipelineState> pso_;
  bool fp16_;
  uint32_t key_dim_;
  uint32_t value_dim_;
};

}  // namespace directml_backend
}  // namespace lczero
