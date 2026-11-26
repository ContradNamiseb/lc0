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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>

#include "neural/backends/blas/blas.h"
#include "neural/backends/blas/layers.h"
#include "neural/backends/shared/activation.h"
#include "neural/backends/shared/winograd_filter.h"
#include "neural/factory.h"
#include "neural/network.h"
#include "neural/network_legacy.h"
#include "neural/tables/attention_policy_map.h"
#include "neural/tables/policy_map.h"
#include "utils/numa.h"

#ifdef USE_DNNL
#include <omp.h>
#endif

namespace lczero {
namespace blas_backend {
namespace {

// Number of layers stored with INT8 weights.
size_t CountInt8Layers(const MultiHeadWeights& weights,
                       const MultiHeadWeights::PolicyHead& policy_head) {
  size_t layers = 0;
  for (const auto& residual : weights.residual) {
    layers += residual.conv1.has_int8 + residual.conv2.has_int8;
  }
  const auto count_encoders =
      [&layers](const std::vector<MultiHeadWeights::EncoderLayer>& encoders) {
        for (const auto& encoder : encoders) {
          layers += 4 * encoder.mha.has_int8 + 2 * encoder.ffn.has_int8;
        }
      };
  count_encoders(weights.encoder);
  count_encoders(policy_head.pol_encoder);
  return layers;
}

template <bool use_eigen>
class BlasNetwork;

template <bool use_eigen>
class BlasComputation : public NetworkComputation {
 public:
  BlasComputation(BlasNetwork<use_eigen>* network, int threads);

  // Adds a sample to the batch.
  void AddInput(InputPlanes&& input) override { planes_.emplace_back(input); }

  // Do the computation.
  void ComputeBlocking() override;

  // Returns how many times AddInput() was called.
  int GetBatchSize() const override { return static_cast<int>(planes_.size()); }

  // Returns Q value of @sample.
  float GetQVal(int sample) const override {
    if (wdl_) {
      return values_[3 * sample + 0] - values_[3 * sample + 2];
    } else {
      return values_[sample];
    }
  }

  float GetDVal(int sample) const override {
    return wdl_ ? values_[3 * sample + 1] : 0.0f;
  }

  float GetMVal(int sample) const override {
    return moves_left_ ? moves_left_values_[sample] : 0.0f;
  }

  // Returns P value @move_id of @sample.
  float GetPVal(int sample, int move_id) const override {
    return policies_[sample * kNumOutputPolicy + move_id];
  }

 private:
  BlasNetwork<use_eigen>* network_;
  const bool wdl_;
  const bool moves_left_;
  std::vector<InputPlanes> planes_;
  std::vector<float> policies_;
  std::vector<float> values_;
  std::vector<float> moves_left_values_;
};

template <bool use_eigen>
class BlasNetwork : public Network {
 public:
  BlasNetwork(const WeightsFile& weights, const OptionsDict& options);

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<BlasComputation<use_eigen>>(this, threads_);
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  int GetMiniBatchSize() const override { return 7; }

  bool IsCpu() const override { return true; }

  void InitThread(int id) override { Numa::BindThread(id); }

  bool wdl() const { return wdl_; }
  bool moves_left() const { return moves_left_; }

  // Runs the network on `batch_size` samples whose planes are already encoded
  // into the first of the workspace's tensors.
  void Forward(size_t batch_size, float* policies, float* values,
               float* moves_left_values, Workspace<use_eigen>& workspace) const;

  std::unique_ptr<Workspace<use_eigen>> GetWorkspace();
  void ReleaseWorkspace(std::unique_ptr<Workspace<use_eigen>> workspace);

  // Largest batch evaluated in one pass; bigger batches are split.
  size_t max_batch_size() const { return max_batch_size_; }

  // Floats each of the three tensors holds per sample.
  size_t tensor_size() const { return tensor_size_; }

 private:
  // A cap on the max batch size since it consumes a lot of memory
  static constexpr auto kHardMaxBatchSize = 2048;

  BaseLayer<use_eigen>* AddLayer(std::unique_ptr<BaseLayer<use_eigen>> layer) {
    network_.emplace_back(std::move(layer));
    return network_.back().get();
  }

  const NetworkCapabilities capabilities_;
  MultiHeadWeights weights_;
  size_t max_batch_size_;
  int threads_;
  bool wdl_;
  bool moves_left_;
  bool conv_policy_;
  bool attn_policy_;
  bool attn_body_;
  size_t num_residual_blocks_;
  size_t residual_channels_ = 0;
  bool use_nhwc_residual_tower_ = false;
  bool use_int8_input_conv_ = false;
  bool use_int8_conv_policy_ = false;
  Int8Kernel int8_kernel_ = Int8Kernel::kMkl;
  std::unique_ptr<Int8ThreadPool> int8_pool_;
  size_t max_convolution_input_channels_ = 0;
  size_t max_convolution_output_channels_ = 0;
  size_t tensor_size_ = 0;
  // The layers, in the order Forward() runs them. Declared after weights_,
  // since the layers hold references into it.
  std::vector<std::unique_ptr<BaseLayer<use_eigen>>> network_;
  std::mutex workspaces_lock_;
  std::vector<std::unique_ptr<Workspace<use_eigen>>> free_workspaces_;
};

template <bool use_eigen>
BlasComputation<use_eigen>::BlasComputation(BlasNetwork<use_eigen>* network,
                                            [[maybe_unused]] int threads)
    : network_(network),
      wdl_(network->wdl()),
      moves_left_(network->moves_left()) {
#ifdef USE_DNNL
  omp_set_num_threads(threads);
#endif
}

template <bool use_eigen>
void BlasComputation<use_eigen>::ComputeBlocking() {
  const size_t total_batches = planes_.size();
  const size_t largest_batch_size =
      std::min(network_->max_batch_size(), total_batches);

  policies_.resize(total_batches * kNumOutputPolicy);
  values_.resize(total_batches * (wdl_ ? 3 : 1));
  if (moves_left_) moves_left_values_.resize(total_batches);

  auto workspace = network_->GetWorkspace();
  for (auto& tensor : workspace->tensors) {
    GrowBuffer(tensor, largest_batch_size * network_->tensor_size());
  }

  for (size_t start = 0; start < total_batches; start += largest_batch_size) {
    const size_t batch_size =
        std::min(total_batches - start, largest_batch_size);
    float* planes = workspace->tensors[0].data();
    for (size_t j = 0; j < batch_size; j++) {
      for (const InputPlane& plane : planes_[start + j]) {
        const float value = plane.value;
        for (int i = 0; i < kSquares; i++) {
          *(planes++) = (plane.mask & (uint64_t{1} << i)) != 0 ? value : 0;
        }
      }
    }
    network_->Forward(batch_size, &policies_[start * kNumOutputPolicy],
                      &values_[start * (wdl_ ? 3 : 1)],
                      moves_left_ ? &moves_left_values_[start] : nullptr,
                      *workspace);
  }
  network_->ReleaseWorkspace(std::move(workspace));
}

template <bool use_eigen>
void BlasNetwork<use_eigen>::Forward(size_t batch_size, float* policies,
                                     float* values, float* moves_left_values,
                                     Workspace<use_eigen>& workspace) const {
  float* tensors[3] = {workspace.tensors[0].data(), workspace.tensors[1].data(),
                       workspace.tensors[2].data()};
  size_t layer = 0;

  // The planes are in tensors[0]; `flow` is the body's output.
  int flow = 0;
  if (num_residual_blocks_ > 0) {
    if (use_int8_input_conv_) {
      // SmoothQuant-109 INT8 input conv writes NHWC directly into tensors[2].
      network_[layer++]->Eval(batch_size, tensors[2], tensors[0], nullptr,
                              workspace);
      for (size_t block = 0; block < num_residual_blocks_; block++) {
        network_[layer++]->Eval(batch_size, tensors[2], tensors[2], nullptr,
                                workspace);  // residual block, in place NHWC
      }
      NhwcToNchwFast(batch_size, residual_channels_, tensors[2], tensors[1],
                     workspace.int8.pool);
    } else if (use_nhwc_residual_tower_) {
      network_[layer++]->Eval(batch_size, tensors[1], tensors[0], nullptr,
                              workspace);  // input conv
      NchwToNhwcFast(batch_size, residual_channels_, tensors[1], tensors[2],
                     workspace.int8.pool);
      for (size_t block = 0; block < num_residual_blocks_; block++) {
        network_[layer++]->Eval(batch_size, tensors[2], tensors[2], nullptr,
                                workspace);  // residual block, in place NHWC
      }
      NhwcToNchwFast(batch_size, residual_channels_, tensors[2], tensors[1],
                     workspace.int8.pool);
    } else {
      network_[layer++]->Eval(batch_size, tensors[1], tensors[0], nullptr,
                              workspace);  // input conv
      for (size_t block = 0; block < num_residual_blocks_; block++) {
        network_[layer++]->Eval(batch_size, tensors[1], tensors[1], nullptr,
                                workspace);  // residual block, in place
      }
    }
    flow = 1;
  }
  if (attn_body_) {
    network_[layer++]->Eval(batch_size, tensors[2], tensors[flow], nullptr,
                            workspace);  // entire attention body
    flow = 2;
  }
  float* body = tensors[flow];
  float* spare1 = tensors[(flow + 1) % 3];
  float* spare2 = tensors[(flow + 2) % 3];

  // Policy head.
  if (attn_policy_) {
    // When the residual tower ran in NHWC and there is no attention body,
    // `tensors[2]` (which is `spare1`) still holds the NHWC body output, so
    // pass it directly and write logits into `spare2` to skip NCHW->NHWC.
    const bool direct_nhwc = use_nhwc_residual_tower_ && !attn_body_;
    const float* pol_in = direct_nhwc ? spare1 : body;
    float* pol_out = direct_nhwc ? spare2 : spare1;
    network_[layer++]->Eval(batch_size, pol_out, pol_in, nullptr,
                            workspace);  // attention policy head
    network_[layer++]->Eval(batch_size, policies, pol_out, nullptr,
                            workspace);  // policy map
  } else if (conv_policy_) {
    if (use_int8_conv_policy_) {
      // `spare1` (`tensors[2]`) still holds the NHWC residual tower output.
      // policy1: NHWC spare1 -> NHWC spare2; policy: NHWC spare2 -> NCHW spare1
      // (converted inside ConvLayer), then PolicyMapLayer reads spare1 ->
      // policies.
      network_[layer++]->Eval(batch_size, spare2, spare1, nullptr,
                              workspace);  // policy conv1 (NHWC -> NHWC)
      network_[layer++]->Eval(batch_size, spare1, spare2, nullptr,
                              workspace);  // policy conv2 (NHWC -> NCHW)
      network_[layer++]->Eval(batch_size, policies, spare1, nullptr,
                              workspace);  // policy map
    } else {
      network_[layer++]->Eval(batch_size, spare1, body, nullptr,
                              workspace);  // policy conv1
      network_[layer++]->Eval(batch_size, spare2, spare1, nullptr,
                              workspace);  // policy conv2
      network_[layer++]->Eval(batch_size, policies, spare2, nullptr,
                              workspace);  // policy map
    }
  } else {
    network_[layer++]->Eval(batch_size, spare1, body, nullptr,
                            workspace);  // policy conv
    network_[layer++]->Eval(batch_size, policies, spare1, nullptr,
                            workspace);  // policy dense
  }

  // Value head.
  network_[layer++]->Eval(batch_size, values, body, nullptr, workspace);

  // Moves left head.
  if (moves_left_) {
    network_[layer++]->Eval(batch_size, spare1, body, nullptr,
                            workspace);  // moves left conv or embedding
    network_[layer++]->Eval(batch_size, spare2, spare1, nullptr,
                            workspace);  // moves left dense 1
    network_[layer++]->Eval(batch_size, moves_left_values, spare2, nullptr,
                            workspace);  // moves left dense 2
  }
  assert(layer == network_.size());
}

template <bool use_eigen>
std::unique_ptr<Workspace<use_eigen>> BlasNetwork<use_eigen>::GetWorkspace() {
  std::lock_guard<std::mutex> lock(workspaces_lock_);
  if (free_workspaces_.empty()) {
    auto workspace = std::make_unique<Workspace<use_eigen>>(
        max_convolution_input_channels_, max_convolution_output_channels_);
    workspace->int8.pool = int8_pool_.get();
    return workspace;
  }
  auto workspace = std::move(free_workspaces_.back());
  free_workspaces_.pop_back();
  workspace->int8.pool = int8_pool_.get();
  return workspace;
}

template <bool use_eigen>
void BlasNetwork<use_eigen>::ReleaseWorkspace(
    std::unique_ptr<Workspace<use_eigen>> workspace) {
  std::lock_guard<std::mutex> lock(workspaces_lock_);
  free_workspaces_.push_back(std::move(workspace));
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
               nf.network() == NF::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT;

  const ActivationFunction default_activation =
      nf.default_activation() == NF::DEFAULT_ACTIVATION_MISH ? ACTIVATION_MISH
                                                             : ACTIVATION_RELU;

  const bool is_pe_dense_embedding =
      static_cast<InputEmbedding>(nf.input_embedding()) ==
      InputEmbedding::INPUT_EMBEDDING_PE_DENSE;

  ActivationFunction smolgen_activation = ACTIVATION_NONE;
  ActivationFunction ffn_activation = ACTIVATION_NONE;
  if (attn_body_) {
    const auto smolgen_setting = nf.smolgen_activation();
    smolgen_activation = smolgen_setting == NF::ACTIVATION_DEFAULT
                             ? default_activation
                             : static_cast<ActivationFunction>(smolgen_setting);
    const auto ffn_setting = nf.ffn_activation();
    ffn_activation = ffn_setting == NF::ACTIVATION_DEFAULT
                         ? default_activation
                         : static_cast<ActivationFunction>(ffn_setting);
  }

  if (max_batch_size_ > kHardMaxBatchSize) {
    max_batch_size_ = kHardMaxBatchSize;
  }

  // Layers stored in INT8 run in INT8 unless int8=false or neither VNNI nor
  // MKL is available; then they use the dequantized FP32 weights.
  const bool int8_requested = options.GetOrDefault<bool>("int8", true);
  const std::string int8_kernel_opt =
      options.GetOrDefault<std::string>("int8_kernel", "auto");
  const bool allow_mkl = !use_eigen && kMklInt8Available;
  const bool int8_supported = IsInt8SupportedOnHost(allow_mkl);
  const bool use_int8 = int8_requested && int8_supported;
  if (use_int8) {
    int8_kernel_ = SelectInt8Kernel(int8_kernel_opt, allow_mkl);
  }

  const std::string policy_head =
      options.GetOrDefault<std::string>("policy_head", "vanilla");
  // Check that selected policy head exists.
  if (weights_.policy_heads.count(policy_head) == 0) {
    throw Exception("The policy head you specified '" + policy_head +
                    "' does not exist in this net.");
  }

  const std::string value_head =
      options.GetOrDefault<std::string>("value_head", "winner");
  // Check that selected value head exists.
  if (weights_.value_heads.count(value_head) == 0) {
    throw Exception("The value head you specified '" + value_head +
                    "' does not exist in this net.");
  }

  auto& policy_weights = weights_.policy_heads.at(policy_head);
  const size_t int8_layers = CountInt8Layers(weights_, policy_weights);
  const bool net_uses_int8 = use_int8 && int8_layers > 0;

  const bool vnni_kernel = (int8_kernel_ == Int8Kernel::kVnniAvx512 ||
                            int8_kernel_ == Int8Kernel::kVnniAvx);
  if (net_uses_int8 && vnni_kernel && threads_ > 1) {
    int8_pool_ = std::make_unique<Int8ThreadPool>(threads_);
  }

  // 1. Transform the 3x3 convolution filters for Winograd, in place.
  const size_t channels = weights_.input.biases.size();
  residual_channels_ = channels;
  num_residual_blocks_ = weights_.residual.size();
  if (num_residual_blocks_ > 0) {
    bool all_res_int8 = use_int8 && vnni_kernel;
    for (const auto& residual : weights_.residual) {
      if (!residual.conv1.has_int8 || !residual.conv2.has_int8) {
        all_res_int8 = false;
        break;
      }
    }
    use_nhwc_residual_tower_ = all_res_int8;
#ifdef LC0_HAS_AVX512_VNNI
    use_int8_input_conv_ = use_nhwc_residual_tower_ &&
                           int8_kernel_ == Int8Kernel::kVnniAvx512 &&
                           CpuSupportsAvx512Vnni();
#endif

    if (!use_int8_input_conv_) {
      weights_.input.weights = WinogradFilterTransformF(weights_.input.weights,
                                                        channels, kInputPlanes);
    }
    for (auto& residual : weights_.residual) {
      if (!(use_int8 && residual.conv1.has_int8)) {
        residual.conv1.weights = WinogradFilterTransformF(
            residual.conv1.weights, channels, channels);
      }
      if (!(use_int8 && residual.conv2.has_int8)) {
        residual.conv2.weights = WinogradFilterTransformF(
            residual.conv2.weights, channels, channels);
      }
    }
    max_convolution_input_channels_ = std::max<size_t>(kInputPlanes, channels);
    max_convolution_output_channels_ = channels;
  }
  if (conv_policy_) {
#ifdef LC0_HAS_AVX512_VNNI
    use_int8_conv_policy_ = use_nhwc_residual_tower_ && !attn_body_ &&
                            int8_kernel_ == Int8Kernel::kVnniAvx512 &&
                            CpuSupportsAvx512Vnni();
#endif
    const size_t policy_channels = policy_weights.policy.biases.size();
    if (!use_int8_conv_policy_) {
      policy_weights.policy1.weights = WinogradFilterTransformF(
          policy_weights.policy1.weights, channels, channels);
      policy_weights.policy.weights = WinogradFilterTransformF(
          policy_weights.policy.weights, policy_channels, channels);
    }
    max_convolution_output_channels_ =
        std::max(max_convolution_output_channels_, policy_channels);
  }

  // 2. Build the network.
  if (num_residual_blocks_ > 0) {
    if (use_int8_input_conv_) {
      const bool hectoplies =
          nf.input() >= NF::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES;
      AddLayer(std::make_unique<ConvLayer<use_eigen>>(
          weights_.input, kInputPlanes, 3, default_activation,
          &weights_.input.weights, hectoplies ? 1.0f : 0.01f));
    } else {
      AddLayer(std::make_unique<ConvLayer<use_eigen>>(
          weights_.input, kInputPlanes, 3, default_activation));
    }
    for (const auto& residual : weights_.residual) {
      AddLayer(std::make_unique<ResidualBlock<use_eigen>>(
          residual, channels, default_activation, use_int8, int8_kernel_,
          use_nhwc_residual_tower_));
    }
  }

  size_t body_channels = channels;
  if (attn_body_) {
    AddLayer(std::make_unique<AttentionBody<use_eigen>>(
        weights_, num_residual_blocks_ > 0 ? channels : kInputPlanes,
        num_residual_blocks_ > 0, is_pe_dense_embedding, default_activation,
        smolgen_activation, ffn_activation, use_int8, int8_kernel_));
    body_channels = weights_.ip_emb_b.size();
  }

  // Policy head.
  if (attn_policy_) {
    AddLayer(std::make_unique<AttentionPolicyHead<use_eigen>>(
        policy_weights, weights_.smolgen_w, body_channels, attn_body_,
        default_activation, smolgen_activation, ffn_activation, net_uses_int8,
        int8_kernel_, use_nhwc_residual_tower_ && !attn_body_));
    AddLayer(std::make_unique<PolicyMapLayer<use_eigen>>(
        kAttnPolicyMap, 64 * 64 + 8 * 24, 64 * 64 + 8 * 24));
  } else if (conv_policy_) {
    assert(!attn_body_);  // not supported with attention body
    if (use_int8_conv_policy_) {
      AddLayer(std::make_unique<ConvLayer<use_eigen>>(
          policy_weights.policy1, channels, 3, default_activation,
          &policy_weights.policy1.weights, 1.0f, /*dynamic_int8_nhwc=*/true,
          /*convert_output_to_nchw=*/false));
      const auto* policy_conv = AddLayer(std::make_unique<ConvLayer<use_eigen>>(
          policy_weights.policy, channels, 3, ACTIVATION_NONE,
          &policy_weights.policy.weights, 1.0f, /*dynamic_int8_nhwc=*/true,
          /*convert_output_to_nchw=*/true));
      constexpr size_t kPolicyUsedPlanes = 73;
      AddLayer(std::make_unique<PolicyMapLayer<use_eigen>>(
          kConvPolicyMap, kPolicyUsedPlanes * kSquares,
          policy_conv->GetOutputSize()));
    } else {
      AddLayer(std::make_unique<ConvLayer<use_eigen>>(
          policy_weights.policy1, channels, 3, default_activation));
      const auto* policy_conv = AddLayer(std::make_unique<ConvLayer<use_eigen>>(
          policy_weights.policy, channels, 3, ACTIVATION_NONE));
      // Number of used planes with convolutional policy. The real number of
      // planes is higher because of padding.
      constexpr size_t kPolicyUsedPlanes = 73;
      AddLayer(std::make_unique<PolicyMapLayer<use_eigen>>(
          kConvPolicyMap, kPolicyUsedPlanes * kSquares,
          policy_conv->GetOutputSize()));
    }
  } else {
    assert(!attn_body_);  // not supported with attention body
    const auto* policy_conv = AddLayer(std::make_unique<ConvLayer<use_eigen>>(
        policy_weights.policy, channels, 1, default_activation));
    AddLayer(std::make_unique<DenseLayer<use_eigen>>(
        policy_weights.ip_pol_w, policy_weights.ip_pol_b,
        policy_conv->GetOutputSize(), ACTIVATION_NONE));
  }

  // Value head.
  AddLayer(std::make_unique<ValueHead<use_eigen>>(
      weights_.value_heads.at(value_head), body_channels, attn_body_, wdl_,
      default_activation));

  // Moves left head.
  if (moves_left_) {
    const BaseLayer<use_eigen>* moves_left_input;
    if (attn_body_) {
      moves_left_input = AddLayer(std::make_unique<EmbeddingLayer<use_eigen>>(
          weights_.ip_mov_w, weights_.ip_mov_b, body_channels,
          default_activation));
    } else {
      moves_left_input = AddLayer(std::make_unique<ConvLayer<use_eigen>>(
          weights_.moves_left, channels, 1, default_activation));
    }
    const auto* moves_left_hidden =
        AddLayer(std::make_unique<DenseLayer<use_eigen>>(
            weights_.ip1_mov_w, weights_.ip1_mov_b,
            moves_left_input->GetOutputSize(), default_activation));
    AddLayer(std::make_unique<DenseLayer<use_eigen>>(
        weights_.ip2_mov_w, weights_.ip2_mov_b,
        moves_left_hidden->GetOutputSize(),
        ACTIVATION_RELU));  // Specifically Relu
  }

  // 3. Size the three tensors: the planes and every layer's output must fit.
  tensor_size_ = kInputPlanes * kSquares;
  for (const auto& layer : network_) {
    tensor_size_ = std::max(tensor_size_, layer->GetOutputSize());
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

  if (int8_layers > 0 && use_int8) {
    CERR << "Using W8A8 INT8 for " << int8_layers << " layers ("
         << Int8KernelName(int8_kernel_) << ").";
  } else if (int8_layers > 0) {
    CERR << "The net has " << int8_layers
         << " INT8 layers; running them in FP32 from the dequantized weights"
         << (int8_requested ? " (no VNNI or MKL INT8 kernel available)."
                            : " (int8=false).");
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
}  // namespace blas_backend
}  // namespace lczero
