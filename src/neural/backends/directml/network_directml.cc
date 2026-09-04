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

// DirectML backend orchestration, following network_cuda.cc's structure:
// DirectMlNetwork<DataType> owns the device context, the weight arena, the
// three rotating tensor buffers and the flat layer list; the computation
// wrapper checks an InputsOutputs out of a free list and calls forwardEval,
// which sequences the layer Evals exactly like the SYCL backend's
// forwardEval (attention body -> attention policy head + policy map ->
// value head -> moves-left head).
//
// Supported net shapes (v1): attention-body and KDA-hybrid nets whose
// policy head is POLICY_ATTENTION -- the shapes the training branch this
// backend accompanies produces. Conv towers (classical/SE nets),
// conv/classical policy heads and smolgen in policy encoders throw clear
// errors rather than guessing. fp16 runs through the same template with
// DmlHalf storage (FXC half caveat applies to the HLSL kernels only).
//
// STATUS (2026-09-04): three of five parity nets PASS at the full 2e-4
// tolerance (KDA+MLH, KDA-hybrid, NoEncoder); the two MHA-encoder nets run
// end to end with small residual drift (~1e-2, under investigation -- see
// docs/directml-handoff.md section 6). The smolgen MLP/bias path is
// implemented (dense graphs + SmolgenBiasLayer HLSL kernel) but the real
// trained net's MLP graph currently fails DML CompileGraph on this driver
// -- the one open blocker for real-net benchmarking; see the handoff's
// section 3/5 for the established driver-rule context.
//   - fp16: compiled, untested.
// The parity tests fail loudly rather than pass at a loosened tolerance;
// treat a red kda_parity_test_directml as the to-do list, not breakage.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <chrono>

#include <span>
#include <version>
#include <DirectMLX.h>

#include "neural/backends/directml/dml_common.h"
#include "neural/backends/directml/inputs_outputs.h"
#include "neural/backends/directml/layers.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "neural/network_legacy.h"
#include "neural/tables/attention_policy_map.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace directml_backend {

template <typename DataType>
class DirectMlNetworkComputation;

// ===========================================================================
// Device bring-up (the network_cuda.cc showInfo/showDeviceInfo analogue).
// ===========================================================================
void DmlDeviceContext::Init(const OptionsDict& options) {
  const int gpu_id = options.GetOrDefault<int>("gpu", 0);
  meta_commands_ = options.GetOrDefault<bool>("meta_commands", true);

  ReportD3DErrors(CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory_)),
                  "CreateDXGIFactory1");
  UINT adapter_index = 0;
  UINT found = 0;
  for (UINT i = 0;
       dxgi_factory_->EnumAdapters1(i, &adapter_) != DXGI_ERROR_NOT_FOUND;
       ++i) {
    DXGI_ADAPTER_DESC1 desc;
    adapter_->GetDesc1(&desc);
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
    if (found == (UINT)gpu_id) {
      std::wstring name(desc.Description);
      CERR << "directml backend selected GPU: "
           << std::string(name.begin(), name.end());
      break;
    }
    adapter_.Reset();
    ++found;
  }
  if (!adapter_) {
    throw Exception("No hardware Direct3D 12 adapter found (directml "
                    "backend, gpu=" +
                    std::to_string(gpu_id) + ").");
  }

  ReportD3DErrors(
      D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_11_0,
                        IID_PPV_ARGS(&device_)),
      "D3D12CreateDevice");

  D3D12_COMMAND_QUEUE_DESC queue_desc = {};
  queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  ReportD3DErrors(device_->CreateCommandQueue(&queue_desc,
                                              IID_PPV_ARGS(&queue_)),
                  "CreateCommandQueue");

  ReportDmlErrors(DMLCreateDevice(device_.Get(),
                                  DML_CREATE_DEVICE_FLAG_NONE,
                                  IID_PPV_ARGS(&dml_device_)),
                  "DMLCreateDevice");
  ReportDmlErrors(
      dml_device_->CreateCommandRecorder(IID_PPV_ARGS(&recorder_)),
      "CreateCommandRecorder");

  ReportD3DErrors(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                       IID_PPV_ARGS(&fence_)),
                  "CreateFence");
  fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (!fence_event_) throw Exception("Failed to create fence event");

  // Descriptor slots are permanently reserved per cached binding table
  // (one per compiled operator, one per ladder batch size). A big net --
  // 10 encoders, 6 graphs each, 8 ladder sizes, ~10 descriptors per table
  // -- needs tens of thousands; 64K slots (~4-8MB heap) covers it.
  descriptors_.Create(device_.Get(), 65536);

  // One-shot command list for weight upload at load.
  ReportD3DErrors(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                  IID_PPV_ARGS(&upload_allocator_)),
                  "CreateCommandAllocator (upload)");
  ReportD3DErrors(device_->CreateCommandList(
                      0, D3D12_COMMAND_LIST_TYPE_DIRECT, upload_allocator_.Get(),
                      nullptr, IID_PPV_ARGS(&upload_list_)),
                  "CreateCommandList (upload)");
  upload_list_->Close();
}

// ===========================================================================
// The network.
// ===========================================================================
template <typename DataType>
class DirectMlNetwork : public Network {
 public:
  DirectMlNetwork(const WeightsFile& file, const OptionsDict& options);
  ~DirectMlNetwork() override = default;

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }
  int GetThreads() const override { return 1; }
  bool IsCpu() const override { return false; }
  int GetMiniBatchSize() const override { return std::min(max_batch_size_, 256); }
  int GetPreferredBatchStep() const override { return 1; }
  std::unique_ptr<NetworkComputation> NewComputation() override;

 private:
  friend class DirectMlNetworkComputation<DataType>;

  void forwardEval(InputsOutputs* io, int batch,
                   const std::vector<InputPlanes>& planes);
  void FlushWeights(DmlWeightUploader& uploader);
  BaseLayer<DataType>* getLastLayer() { return network_.back().get(); }

  NetworkCapabilities capabilities_;
  DmlDeviceContext ctx_;
  MultiHeadWeights weights_;

  DmlArena weight_arena_;
  DmlArena tensor_arena_;    // three rotating slots, sub-allocated
  DmlArena scratch_arena_;   // layer scratch (KDA q/k/v, policy wq/wk...)
  DmlArena transient_arena_; // per-dispatch DirectML internal scratch

  std::vector<std::unique_ptr<BaseLayer<DataType>>> network_;
  BaseLayer<DataType>* encoder_last_ = nullptr;

  bool wdl_ = false;
  bool moves_left_ = false;
  bool attn_body_ = false;
  int max_batch_size_ = 256;
  int min_batch_size_ = 4;
  DmlArena smolgen_arena_;
  uint64_t scratch_bytes_ = 0;
  uint64_t tensor_slot_bytes_ = 0;

  std::mutex eval_lock_;
  std::mutex io_lock_;
  std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;

  std::unique_ptr<InputsOutputs> GetInputsOutputs();
  void ReleaseInputsOutputs(std::unique_ptr<InputsOutputs> io);
};

template <typename DataType>
DirectMlNetwork<DataType>::DirectMlNetwork(const WeightsFile& file,
                                           const OptionsDict& options)
    : weights_(file.weights()) {
  const auto nf = file.format().network_format();
  using NF = pblczero::NetworkFormat;
  capabilities_ = {nf.input(), nf.output(), nf.moves_left()};

  attn_body_ = nf.network() == NF::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT ||
               nf.network() == NF::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT ||
               nf.network() == NF::NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT;
  if (!attn_body_) {
    throw Exception(
        "Network format " + NF::NetworkStructure_Name(nf.network()) +
        " is not supported by the directml backend (attention-body and "
        "KDA-hybrid nets only; conv towers are not implemented).");
  }
  if (nf.policy() != NF::POLICY_ATTENTION) {
    throw Exception("Policy format " + NF::PolicyFormat_Name(nf.policy()) +
                    " is not supported by the directml backend "
                    "(POLICY_ATTENTION only).");
  }

  max_batch_size_ = std::min(1024, std::max(1, options.GetOrDefault<int>(
                                            "max_batch", 256)));
  min_batch_size_ = std::clamp(
      options.GetOrDefault<int>("min_batch", std::min(4, max_batch_size_)), 1,
      max_batch_size_);
  if (max_batch_size_ < min_batch_size_) {
    throw Exception("Max batch must not be less than min_batch setting.");
  }

  ctx_.Init(options);


  if (!weights_.residual.empty()) {
    throw Exception(
        "The directml backend does not support residual conv blocks yet.");
  }

  constexpr bool fp16 = std::is_same<DataType, DmlHalf>::value;
  if (fp16 && !options.GetOrDefault<bool>("allow_broken_fp16", false)) {
    // The fp16 path compiles and runs but its numbers are garbage, not merely
    // imprecise: on MatchesBlasOnKdaMlhNet it returns policy logits up to
    // 2.67e36 against a reference maximum of 0.032, with q -0.4999 and d
    // 0.5000 -- a degenerate softmax over nonsense. fp16 cannot represent
    // 2.67e36 at all (its maximum is 65504), so this is reinterpreted bits
    // somewhere, not rounding.
    //
    // Registering a selectable backend that silently returns garbage
    // evaluations is the same failure mode that cost days on the fp32 path,
    // so it refuses to load until the numerics are fixed. Set
    // backend-opts=allow_broken_fp16=true to run it anyway while working on
    // it. The buffer-sizing work in 1491280/ebc0dcf made fp16 bindings
    // LEGAL; it did not make them correct.
    throw Exception(
        "The directml-fp16 backend is numerically broken (policy magnitudes "
        "~1e36 against a ~0.03 reference) and refuses to load. Use the fp32 "
        "'directml' backend, or pass "
        "backend-opts=allow_broken_fp16=true to work on the fp16 path.");
  }
  CERR << "Initializing directml backend (" << (fp16 ? "fp16" : "fp32")
       << ")";

  wdl_ = nf.value() == NF::VALUE_WDL;
  moves_left_ = nf.moves_left() == NF::MOVES_LEFT_V1 &&
                options.GetOrDefault<bool>("mlh", true);

  const bool mish_net =
      nf.default_activation() == NF::DEFAULT_ACTIVATION_MISH;
  const ActivationFunction act = mish_net ? ACTIVATION_MISH : ACTIVATION_RELU;
  Activations activations;
  activations.default_activation = act;
  activations.smolgen_activation =
      nf.smolgen_activation() == NF::ACTIVATION_DEFAULT
          ? act
          : static_cast<ActivationFunction>(nf.smolgen_activation());
  activations.ffn_activation =
      nf.ffn_activation() == NF::ACTIVATION_DEFAULT
          ? act
          : static_cast<ActivationFunction>(nf.ffn_activation());

  // Head selection, like the CUDA/SYCL backends.
  const std::string policy_head_name =
      options.GetOrDefault<std::string>("policy_head", "vanilla");
  if (weights_.policy_heads.count(policy_head_name) == 0) {
    throw Exception("The policy head you specified '" + policy_head_name +
                    "' does not exist in this net.");
  }
  const std::string value_head_name =
      options.GetOrDefault<std::string>("value_head", "winner");
  if (weights_.value_heads.count(value_head_name) == 0) {
    throw Exception("The value head you specified '" + value_head_name +
                    "' does not exist in this net.");
  }
  auto& policy_head = weights_.policy_heads.at(policy_head_name);
  auto& value_head = weights_.value_heads.at(value_head_name);

  const size_t elem = sizeof(DataType);
  const uint64_t max_tokens = (uint64_t)max_batch_size_ * 64;

  // Scratch estimate: the largest of what the body/policy/encoder phases
  // keep alive simultaneously (mirror of the SYCL backend's
  // getMaxAttentionBodySize/getMaxKdaBodySize reasoning).
  const uint64_t emb_size = weights_.ip_emb_b.size();
  uint64_t scratch_elems = 112 + kNumPosEncodingChannels;
  for (const auto& enc : weights_.encoder) {
    uint64_t need;
    if (enc.is_kda) {
      const uint64_t KD =
          (uint64_t)weights_.encoder_head_count * enc.kda.key_dim;
      const uint64_t VD =
          (uint64_t)weights_.encoder_head_count * enc.kda.value_dim;
      need = 2 * KD + VD + enc.kda.gate_rank + emb_size +
             std::max<uint64_t>(2 * KD, VD + 3 * enc.kda.key_dim);
    } else {
      const uint64_t d_model =
          !enc.mha.q_w.empty() ? enc.mha.q_w.size() / emb_size : emb_size;
      // 3*d_model per token for the q/k/v projections in the scratch arena
      // PLUS 5*d_model for the split/merged head-transpose buffers EvalMha
      // keeps in the buffer1 tensor slot (qt/kt/vt/ctx/merged); the tensor
      // slot is sized >= scratch_bytes_, so folding both into this estimate
      // covers both arenas. An underestimate here is not a failed
      // allocation but an out-of-bounds UAV write and a removed device
      // (DXGI_ERROR_DEVICE_REMOVED surfacing at the next DML call).
      need = 8 * d_model;
    }
    scratch_elems = std::max(scratch_elems, need);
  }
  scratch_elems = std::max(
      {scratch_elems,
       2 * policy_head.ip2_pol_b.size() + 64,
       emb_size + 64 + 64 /* dense concat */,
       // Fused-LayerNorm temporaries. Splitting each encoder tail at its
       // LayerNorms (see LayerNormLayer) leaves two [tokens, C] buffers
       // alive across the split graphs. They are carved out of the scratch
       // arena's second half -- which is `scratch_bytes_` bytes, i.e.
       // max_tokens * scratch_elems elements -- so scratch_elems must cover
       // 2 * C for every C a LayerNorm is applied at. Underestimating here
       // is not a failed allocation but an out-of-bounds UAV write.
       2 * emb_size,
       2 * policy_head.ip_pol_b.size()});
  scratch_bytes_ = max_tokens * scratch_elems * elem;

  // Tensor slots: at least the largest layer output, the raw input, and
  // half the scratch each (buffer1/buffer2 live in the input2 slot, like
  // the SYCL backend's tensor_mem_[3] rotation).
  uint64_t max_layer_bytes = max_tokens * 64 * elem;  // raw NCHW input size
  // (layer outputs are all <= tokens*max(embed, pol_map); compute exactly)
  const uint64_t emb = weights_.ip_emb_b.size();
  max_layer_bytes =
      std::max(max_layer_bytes, max_tokens * std::max<uint64_t>(emb, 4288) * elem);
  tensor_slot_bytes_ = std::max(max_layer_bytes, scratch_bytes_);

  // The sizing above accounts only for layer OUTPUTS, but EncoderBlock::
  // EvalMha carves FIVE max_tokens * d_model regions (qt/kt/vt/ctx/merged)
  // out of buffer1, which is a tensor slot. Nothing made that an invariant,
  // so current nets fit by arithmetic luck rather than by construction, and
  // an overflow here is not a failed allocation -- it is an out-of-bounds
  // UAV write into the neighbouring slot.
  //
  // The bound differs between the two callers, because they hand EvalMha a
  // different buffer1:
  //   AttentionBody::Eval    passes the slot BASE            -> 5 * d_model
  //   AttentionPolicyHead    passes output + scratch/2       -> scratch/2
  //                                                              + 5 * d_model
  // Checked here in elements per token, with the net's real numbers in the
  // message so a future failure names the net rather than the arithmetic.
  {
    const uint64_t slot_elems_per_token = tensor_slot_bytes_ / (max_tokens * elem);
    auto require = [&](uint64_t needed, const char* which, uint64_t d_model) {
      if (needed > slot_elems_per_token) {
        throw Exception(
            std::string("directml backend: this net's ") + which +
            " attention needs " + std::to_string(needed) +
            " elements per token but the tensor slot holds only " +
            std::to_string(slot_elems_per_token) + " (d_model " +
            std::to_string(d_model) + ", max_batch " +
            std::to_string(max_batch_size_) +
            "). Raise the tensor slot sizing to cover the MHA carve-up.");
      }
    };
    for (const auto& enc : weights_.encoder) {
      const uint64_t d_model = enc.mha.q_b.size();
      if (d_model == 0) continue;  // KDA encoders do not take this path
      require(5 * d_model, "body", d_model);
    }
    const uint64_t pol_d_model = policy_head.ip2_pol_b.size();
    if (pol_d_model != 0 && !policy_head.pol_encoder.empty()) {
      require(scratch_elems / 2 + 5 * pol_d_model, "policy-head", pol_d_model);
    }
  }

  const uint64_t tensor_arena_bytes = 3 * tensor_slot_bytes_;

  // Weight arena: sized from the parsed weights' serialized size -- a
  // tight upper bound on the float data (the hand-written per-field
  // estimate missed real-net arrays: smolgen, the unselected value/policy
  // heads, preprocess layers -- and exhausted the arena on real nets).
  const uint64_t weight_arena_bytes =
      (uint64_t)file.weights().OutputAsString().size() * 2 + (16 << 20);
  weight_arena_.Create(ctx_.device(), weight_arena_bytes, "weights",
                       D3D12_RESOURCE_STATE_COPY_DEST);

  // Smolgen intermediates for one encoder at max batch: the compress
  // output, the two MLP stage outputs and the generated bias, all live at
  // once and all crossing dispatch boundaries. Only one encoder's are live
  // at a time, so this is a max over encoders, not a sum.
  uint64_t smolgen_bytes = 0;
  for (const auto& enc : weights_.encoder) {
    if (!enc.mha.has_smolgen || enc.is_kda) continue;
    const uint64_t compress_size =
        emb_size ? enc.mha.smolgen.compress.size() / emb_size : 0;
    const uint64_t need =
        AlignUp(max_tokens * compress_size * elem) +
        AlignUp((uint64_t)max_batch_size_ * enc.mha.smolgen.dense1_b.size() *
                elem) +
        AlignUp((uint64_t)max_batch_size_ * enc.mha.smolgen.dense2_b.size() *
                elem) +
        AlignUp((uint64_t)max_batch_size_ * weights_.encoder_head_count * 64 *
                64 * elem);
    smolgen_bytes = std::max(smolgen_bytes, need);
  }
  // A zero-size committed resource is invalid; keep one aligned block so the
  // arena is always bindable even for nets without smolgen.
  smolgen_arena_.Create(ctx_.device(), std::max<uint64_t>(smolgen_bytes, 256),
                        "smolgen");

  tensor_arena_.Create(ctx_.device(), tensor_arena_bytes, "tensors");
  scratch_arena_.Create(ctx_.device(), scratch_bytes_ * 2, "scratch");
  // 256MB transient: the MHA attention graph's [B=N*H,64,64] scores +
  // softmax temporaries reach ~2*B*4096*4 bytes (134MB at max batch with 16
  // heads); 64MB would throw arena-exhausted on large batches.
  transient_arena_.Create(ctx_.device(), 256 * 1024 * 1024, "transient");

  DmlWeightUploader uploader(&weight_arena_);

  const bool is_pe_dense =
      nf.input_embedding() == NF::INPUT_EMBEDDING_PE_DENSE;
  const std::vector<int> kda_directions(nf.kda_directions().begin(),
                                        nf.kda_directions().end());
  // Reject anything the traversal table does not cover. Without this an
  // unknown direction silently fell through to plain rank order, so a net
  // trained with one would load, run, and return quietly wrong evaluations.
  for (const int direction : kda_directions) {
    if (direction < 1 || direction > 16) {
      throw Exception(
          "directml backend: unsupported KDA traversal direction " +
          std::to_string(direction) + " (expected 1-16).");
    }
  }

  // Build the topology, in execution order (network_cuda.cc's "2. Build the
  // network, and copy the weights to GPU memory").
  {
    auto body = std::make_unique<AttentionBody<DataType>>(
        weights_, uploader, activations, 0, kNumInputPlanes, max_batch_size_,
        is_pe_dense, kda_directions, ctx_, fp16);
    network_.emplace_back(std::move(body));
    encoder_last_ = getLastLayer();
  }
  {
    auto head = std::make_unique<AttentionPolicyHead<DataType>>(
        getLastLayer(), policy_head, uploader, true, act, max_batch_size_,
        ctx_, fp16, kda_directions);
    network_.emplace_back(std::move(head));
    auto policymap = std::make_unique<PolicyMapLayer<DataType>>(
        getLastLayer(), 64 * 64 + 8 * 24, true, kAttnPolicyMap, uploader);
    network_.emplace_back(std::move(policymap));
  }
  {
    auto head = std::make_unique<ValueHead<DataType>>(
        encoder_last_, value_head, uploader, wdl_, act);
    network_.emplace_back(std::move(head));
  }
  if (moves_left_) {
    auto embedded_mov = std::make_unique<EmbeddingLayer<DataType>>(
        encoder_last_, (int)weights_.ip_mov_b.size(), 8, 8, true, act,
        weights_.ip_mov_w, weights_.ip_mov_b, uploader);
    network_.emplace_back(std::move(embedded_mov));
    auto fc1 = std::make_unique<FCLayer<DataType>>(
        getLastLayer(), (int)weights_.ip1_mov_b.size(), 1, 1, true, act,
        weights_.ip1_mov_w, weights_.ip1_mov_b, uploader);
    network_.emplace_back(std::move(fc1));
    auto fc2 = std::make_unique<FCLayer<DataType>>(
        getLastLayer(), 1, 1, 1, true, ACTIVATION_RELU, weights_.ip2_mov_w,
        weights_.ip2_mov_b, uploader);
    network_.emplace_back(std::move(fc2));
  }

  FlushWeights(uploader);

  // Zero the activation arenas once at load, mirroring the SYCL backend's
  // memset of tensor_mem_[i]: padding rows and any region a graph doesn't
  // fully overwrite then read back as zeros instead of undefined memory.
  {
    ReportD3DErrors(ctx_.upload_allocator()->Reset(), "Reset (clear)");
    ReportD3DErrors(
        ctx_.upload_list()->Reset(ctx_.upload_allocator(), nullptr),
        "Reset (clear list)");
    // One zeroed staging buffer shared by both arenas. It MUST outlive the
    // Execute+Wait below: the recorded CopyBufferRegions read from it on the
    // GPU timeline, so releasing it at the end of a per-arena loop iteration
    // (before execution) hands the GPU freed memory and hangs the device.
    static constexpr uint64_t kChunk = 16 << 20;
    ComPtr<ID3D12Resource> zeros = detail::CreateBuffer(
        ctx_.device(), kChunk, D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ);
    {
      uint8_t* z = nullptr;
      zeros->Map(0, nullptr, reinterpret_cast<void**>(&z));
      std::memset(z, 0, kChunk);
      zeros->Unmap(0, nullptr);
    }
    for (DmlArena* arena : {&tensor_arena_, &scratch_arena_}) {
      // No ClearUnorderedAccessView for buffers in D3D12 -- clear with an
      // upload-heap source copy in kChunk pieces from the shared staging.
      D3D12_RESOURCE_BARRIER to_copy = {};
      to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      to_copy.Transition.pResource = arena->resource();
      to_copy.Transition.StateBefore =
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
      ctx_.upload_list()->ResourceBarrier(1, &to_copy);
      for (uint64_t off = 0; off < arena->size(); off += kChunk) {
        const uint64_t bytes = std::min(kChunk, arena->size() - off);
        ctx_.upload_list()->CopyBufferRegion(arena->resource(), off,
                                            zeros.Get(), 0, bytes);
      }
      D3D12_RESOURCE_BARRIER to_uav = to_copy;
      to_uav.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      to_uav.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      ctx_.upload_list()->ResourceBarrier(1, &to_uav);
    }
    ReportD3DErrors(ctx_.upload_list()->Close(), "Close (clear)");
    ID3D12CommandList* lists[] = {ctx_.upload_list()};
    ctx_.queue()->ExecuteCommandLists(1, lists);
    const uint64_t fence_value = ctx_.NextUploadFenceValue();
    ReportD3DErrors(ctx_.queue()->Signal(ctx_.fence(), fence_value),
                    "Signal (clear)");
    ctx_.WaitForFence(ctx_.fence(), fence_value);
  }


  // Pre-compile every layer for a ladder of batch sizes NOW, at load: this
  // driver fails all DML operator creation with bogus errors once
  // dispatches have been recorded (docs/directml-handoff.md section 3), so
  // nothing may be compiled after the first batch runs. forwardEval rounds
  // each batch UP to the ladder.
  for (int b : {min_batch_size_, 8, 16, 32, 64, 128, 256, max_batch_size_}) {
    if (b < min_batch_size_ || b > max_batch_size_) continue;
    DmlExecScope pre(ctx_, ctx_.upload_list(), &transient_arena_,
                     &smolgen_arena_);
    for (auto& layer : network_) layer->EnsureCompiled(b, pre);
  }




  // Pre-allocate one InputsOutputs (the first allocation is slow, like the
  // CUDA backend's note about first cudaMalloc).
  auto io = GetInputsOutputs();
}

template <typename DataType>
void DirectMlNetwork<DataType>::FlushWeights(DmlWeightUploader& uploader) {
  constexpr bool fp16 = std::is_same<DataType, DmlHalf>::value;
  // Gather total bytes, fill one mapped staging buffer (converting fp32 ->
  // DataType for float weight entries; raw entries such as the gather
  // indices are copied verbatim), record one command list of copies, then
  // flip the weight arena to UAV.
  uint64_t total = 0;
  for (const auto& p : uploader.pending()) total += AlignUp(p.bytes);
  ComPtr<ID3D12Resource> staging = detail::CreateBuffer(
      ctx_.device(), total ? total : 256, D3D12_HEAP_TYPE_UPLOAD,
      D3D12_RESOURCE_STATE_GENERIC_READ);
  uint8_t* mapped = nullptr;
  staging->Map(0, nullptr, reinterpret_cast<void**>(&mapped));

  ReportD3DErrors(ctx_.upload_allocator()->Reset(), "Reset (upload)");
  ReportD3DErrors(
      ctx_.upload_list()->Reset(ctx_.upload_allocator(), nullptr),
      "Reset (upload list)");

  uint64_t cursor = 0;
  for (const auto& p : uploader.pending()) {
    uint8_t* dst_ptr = mapped + cursor;
    if (fp16 && p.is_float) {
      const float* fsrc = reinterpret_cast<const float*>(p.owned.data());
      DmlHalf* hdst = reinterpret_cast<DmlHalf*>(dst_ptr);
      const size_t n = p.bytes / 4;
      for (size_t i = 0; i < n; ++i) hdst[i] = DmlHalf(fsrc[i]);
    } else {
      std::memcpy(dst_ptr, p.owned.data(), p.bytes);
    }
    ctx_.upload_list()->CopyBufferRegion(weight_arena_.resource(),
                                         p.dest.offset, staging.Get(), cursor,
                                         p.bytes);
    cursor += AlignUp(p.bytes);
  }
  staging->Unmap(0, nullptr);

  if (total) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = weight_arena_.resource();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    ctx_.upload_list()->ResourceBarrier(1, &barrier);
  }
  ReportD3DErrors(ctx_.upload_list()->Close(), "Close (upload)");
  ID3D12CommandList* lists[] = {ctx_.upload_list()};
  ctx_.queue()->ExecuteCommandLists(1, lists);
  const uint64_t fence_value = ctx_.NextUploadFenceValue();
  ReportD3DErrors(ctx_.queue()->Signal(ctx_.fence(), fence_value),
                  "Signal (upload)");
  ctx_.WaitForFence(ctx_.fence(), fence_value);
}

template <typename DataType>
std::unique_ptr<InputsOutputs> DirectMlNetwork<DataType>::GetInputsOutputs() {
  std::lock_guard<std::mutex> lock(io_lock_);
  if (free_inputs_outputs_.empty()) {
    return std::make_unique<InputsOutputs>(ctx_.device(), max_batch_size_,
                                           wdl_, moves_left_);
  }
  auto io = std::move(free_inputs_outputs_.front());
  free_inputs_outputs_.pop_front();
  return io;
}

template <typename DataType>
void DirectMlNetwork<DataType>::ReleaseInputsOutputs(
    std::unique_ptr<InputsOutputs> io) {
  std::lock_guard<std::mutex> lock(io_lock_);
  free_inputs_outputs_.push_back(std::move(io));
}

template <typename DataType>
void DirectMlNetwork<DataType>::forwardEval(
    InputsOutputs* io, int batch, const std::vector<InputPlanes>& planes) {
  batch = std::max(batch, min_batch_size_);
  // Round UP to the pre-compiled ladder: only these sizes have graphs (see
  // the pre-compile at load -- post-dispatch compilation fails on this
  // driver). The layers are row/batch independent, so extra padding rows
  // are harmless; buffers are sized for max_batch.
  for (int b : {min_batch_size_, 8, 16, 32, 64, 128, 256, max_batch_size_}) {
    if (b >= batch) { batch = std::clamp(b, min_batch_size_, max_batch_size_); break; }
  }
  std::lock_guard<std::mutex> guard(eval_lock_);

  // Expand packed planes to full NCHW float planes on the host -- the CUDA
  // backend's expandPlanes_NCHW kernel's work, done where it is cheapest
  // for one batch; the preprocess HLSL kernel picks it up from here.
  float* input = io->input_mapped_;
  std::memset(input, 0, (size_t)batch * kNumInputPlanes * 64 * sizeof(float));
  for (int n = 0; n < batch; ++n) {
    const auto& sample = planes[std::min((size_t)n, planes.size() - 1)];
    float* dst = input + (size_t)n * kNumInputPlanes * 64;
    size_t c = 0;
    for (const auto& plane : sample) {
      if (c >= kNumInputPlanes) break;
      float* channel = dst + c * 64;
      uint64_t mask = plane.mask;
      while (mask) {
        int bit = 63;
        while (!(mask & (1ULL << bit))) --bit;
        channel[bit] = plane.value;
        mask &= ~(1ULL << bit);
      }
      ++c;
    }
  }

  ReportD3DErrors(io->command_allocator_->Reset(), "Reset allocator");
  ReportD3DErrors(
      io->command_list_->Reset(io->command_allocator_.Get(), nullptr),
      "Reset command list");
  transient_arena_.ResetCursor();

  ID3D12GraphicsCommandList* list = io->command_list_.Get();

  DmlPtr tensor_mem[3];
  for (int i = 0; i < 3; ++i) {
    tensor_mem[i] = DmlPtr(tensor_arena_.resource(),
                           (uint64_t)i * tensor_slot_bytes_);
  }
  DmlPtr scratch(scratch_arena_.resource(), 0);

  DmlExecScope scope(ctx_, list, &transient_arena_, &smolgen_arena_);

  // Two-phase: compile every layer's graphs for this batch size BEFORE any
  // dispatch is recorded. On this driver, interleaving diverse-shape graph
  // builds with bound dispatches poisons later CreateOperator calls (see
  // docs/directml-handoff.md section 3); it also removes lazy-compile
  // stalls from the search loop.
  for (auto& layer : network_) layer->EnsureCompiled(batch, scope);

  DmlPtr flow = tensor_mem[0];
  DmlPtr spare1 = tensor_mem[1];
  DmlPtr spare2 = tensor_mem[2];

  // Attention body: input = tensor_mem[0] (float NCHW planes), output =
  // tensor_mem[1], input2 = tensor_mem[2] (its two halves are buffer1/2).
  // Upload the expanded host planes into tensor slot 0 (the body's NCHW
  // input) before any layer reads it. tensor_arena_ lives in UAV state;
  // transition to COPY_DEST, copy, transition back.
  {
    const uint64_t input_bytes =
        (uint64_t)batch * kNumInputPlanes * 64 * sizeof(float);
    D3D12_RESOURCE_BARRIER to_copy = {};
    to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_copy.Transition.pResource = tensor_arena_.resource();
    to_copy.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    list->ResourceBarrier(1, &to_copy);
    list->CopyBufferRegion(tensor_arena_.resource(), 0,
                           io->input_upload_.Get(), 0, input_bytes);
    D3D12_RESOURCE_BARRIER to_uav = to_copy;
    to_uav.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    to_uav.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    list->ResourceBarrier(1, &to_uav);
  }

  network_[0]->Eval(batch, tensor_mem[1], tensor_mem[0], tensor_mem[2],
                    scratch, scratch_bytes_, scope);


  flow = tensor_mem[1];
  spare1 = tensor_mem[0];
  spare2 = tensor_mem[2];

  int l = 1;

  // Policy head (writes [N, 4288] rows into spare1) + policy map.
  network_[l++]->Eval(batch, spare1, flow, spare2, scratch, scratch_bytes_,
                      scope);
  DmlPtr op_pol(io->policy_gpu_.Get(), 0);
  network_[l++]->Eval(batch, op_pol, spare1, spare2, scratch, scratch_bytes_,
                      scope);

  // Value head via scratch + copy. The value graph's output silently never
  // lands when bound straight to the io readback buffer on this driver
  // (Intel Iris Xe, system DirectML): identical graph + bindings write fine
  // to a scratch-arena target but leave any io-buffer target untouched, with
  // no error. The policy map (no transient resource) writes io buffers fine,
  // so this smells like a transient/internal-output interaction in the
  // driver; route around it with an explicit copy, which is known-good.
  // TODO(directml): re-test direct io output on newer drivers / discrete
  // GPUs and drop the copy if it works there.
  DmlPtr op_val_scratch(scratch_arena_.resource(), 0);
  network_[l++]->Eval(batch, op_val_scratch, flow, spare2, scratch,
                      scratch_bytes_, scope);
  {
    D3D12_RESOURCE_BARRIER vcopy = {};
    vcopy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    vcopy.Transition.pResource = io->value_gpu_.Get();
    vcopy.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    vcopy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    list->ResourceBarrier(1, &vcopy);
    list->CopyBufferRegion(io->value_gpu_.Get(), 0, scratch_arena_.resource(),
                           0,
                           (uint64_t)batch * (wdl_ ? 3 : 1) * sizeof(float));
    vcopy.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    vcopy.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    list->ResourceBarrier(1, &vcopy);
  }

  // Moves left head.
  if (moves_left_) {
    network_[l++]->Eval(batch, spare1, flow, spare2, scratch, scratch_bytes_,
                        scope);
    network_[l++]->Eval(batch, spare2, spare1, spare2, scratch, scratch_bytes_,
                        scope);
    DmlPtr op_mov(io->moves_left_gpu_.Get(), 0);
    network_[l++]->Eval(batch, op_mov, spare2, spare2, scratch, scratch_bytes_,
                        scope);
  }

  // Readback: policy/value/moves-left UAV -> COPY_SOURCE -> readback.
  auto transition = [&](ID3D12Resource* res, D3D12_RESOURCE_STATES before,
                        D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    list->ResourceBarrier(1, &b);
  };
  auto readback = [&](ID3D12Resource* gpu, ID3D12Resource* dest,
                      uint64_t bytes) {
    transition(gpu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(dest, 0, gpu, 0, bytes);
    transition(gpu, D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  };
  readback(io->policy_gpu_.Get(), io->policy_readback_.Get(),
           (uint64_t)batch * kNumOutputPolicy * sizeof(float));
  readback(io->value_gpu_.Get(), io->value_readback_.Get(),
           (uint64_t)batch * (wdl_ ? 3 : 1) * sizeof(float));
  if (moves_left_) {
    readback(io->moves_left_gpu_.Get(), io->moves_left_readback_.Get(),
             (uint64_t)batch * sizeof(float));
  }

  ReportD3DErrors(list->Close(), "Close");
  ID3D12CommandList* lists[] = {list};
  ctx_.queue()->ExecuteCommandLists(1, lists);
  ++io->fence_value_;
  ReportD3DErrors(ctx_.queue()->Signal(io->fence_.Get(), io->fence_value_),
                  "Signal");
  ctx_.WaitForFence(io->fence_.Get(), io->fence_value_);
  if (const char* prefix = getenv("LC0_DUMP_BODY")) {
    for (auto& d : BodyDumps()) {
      const float* p = nullptr;
      d.readback->Map(0, nullptr,
                      const_cast<void**>(reinterpret_cast<const void**>(&p)));
      std::ofstream f(std::string(prefix) + ".dml." + d.stage + ".bin",
                      std::ios::binary);
      f.write(reinterpret_cast<const char*>(p), d.bytes);
      d.readback->Unmap(0, nullptr);
    }
    BodyDumps().clear();
  }
  if (wdl_) {
    // Value softmax done CPU-side, exactly like the CUDA/SYCL finishEval.
    float* v = const_cast<float*>(io->value_mapped_);
    for (int i = 0; i < batch; ++i) {
      float w = v[3 * i + 0];
      float d = v[3 * i + 1];
      float l = v[3 * i + 2];
      const float m = std::max({w, d, l});
      w = std::exp(w - m);
      d = std::exp(d - m);
      l = std::exp(l - m);
      const float sum = w + d + l;
      v[3 * i + 0] = w / sum;
      v[3 * i + 1] = d / sum;
      v[3 * i + 2] = l / sum;
    }
  }
}

// ===========================================================================
// The computation wrapper (cuda's CudaNetworkComputation analogue).
// ===========================================================================
template <typename DataType>
class DirectMlNetworkComputation : public NetworkComputation {
 public:
  DirectMlNetworkComputation(DirectMlNetwork<DataType>* network, bool wdl,
                             bool moves_left)
      : network_(network), wdl_(wdl), moves_left_(moves_left) {
    inputs_outputs_ = network_->GetInputsOutputs();
  }
  ~DirectMlNetworkComputation() override {
    network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
  }

  void AddInput(InputPlanes&& input) override {
    planes_.emplace_back(std::move(input));
  }
  void ComputeBlocking() override {
    network_->forwardEval(inputs_outputs_.get(), GetBatchSize(), planes_);
  }
  int GetBatchSize() const override {
    return static_cast<int>(planes_.size());
  }
  float GetQVal(int sample) const override {
    return wdl_ ? inputs_outputs_->value_mapped_[3 * sample] -
                      inputs_outputs_->value_mapped_[3 * sample + 2]
                : inputs_outputs_->value_mapped_[sample];
  }
  float GetDVal(int sample) const override {
    return wdl_ ? inputs_outputs_->value_mapped_[3 * sample + 1] : 0.0f;
  }
  float GetPVal(int sample, int move_id) const override {
    return inputs_outputs_
        ->policy_mapped_[sample * kNumOutputPolicy + move_id];
  }
  float GetMVal(int sample) const override {
    return moves_left_ ? inputs_outputs_->moves_left_mapped_[sample] : 0.0f;
  }

 private:
  std::vector<InputPlanes> planes_;
  DirectMlNetwork<DataType>* network_;
  std::unique_ptr<InputsOutputs> inputs_outputs_;
  bool wdl_;
  bool moves_left_;
};

template <typename DataType>
std::unique_ptr<NetworkComputation> DirectMlNetwork<DataType>::NewComputation() {
  return std::make_unique<DirectMlNetworkComputation<DataType>>(this, wdl_,
                                                                moves_left_);
}

// ===========================================================================
// Factory (the MakeCudaNetwork format-validation pattern).
// ===========================================================================
template <typename DataType>
std::unique_ptr<Network> MakeDirectMlNetwork(
    const std::optional<WeightsFile>& w, const OptionsDict& options) {
  if (!w) {
    throw Exception(
        "The directml" +
        std::string(std::is_same<DataType, DmlHalf>::value ? "-fp16" : "") +
        " backend requires a network file.");
  }
  const auto nf = w->format().network_format();
  using NF = pblczero::NetworkFormat;
  switch (nf.network()) {
    case NF::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT:
    case NF::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT:
    case NF::NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT:
      break;
    default:
      throw Exception("Network format " +
                      NF::NetworkStructure_Name(nf.network()) +
                      " is not supported by the directml backend.");
  }
  switch (nf.value()) {
    case NF::VALUE_CLASSICAL:
    case NF::VALUE_WDL:
      break;
    default:
      throw Exception("Value format " + NF::ValueFormat_Name(nf.value()) +
                      " is not supported by the directml backend.");
  }
  switch (nf.moves_left()) {
    case NF::MOVES_LEFT_NONE:
    case NF::MOVES_LEFT_V1:
      break;
    default:
      throw Exception("Moves left head format " +
                      NF::MovesLeftFormat_Name(nf.moves_left()) +
                      " is not supported by the directml backend.");
  }
  switch (nf.default_activation()) {
    case NF::DEFAULT_ACTIVATION_RELU:
    case NF::DEFAULT_ACTIVATION_MISH:
      break;
    default:
      throw Exception("Default activation " +
                      NF::DefaultActivation_Name(nf.default_activation()) +
                      " is not supported by the directml backend.");
  }
  switch (nf.input_embedding()) {
    case NF::INPUT_EMBEDDING_NONE:
    case NF::INPUT_EMBEDDING_PE_MAP:
    case NF::INPUT_EMBEDDING_PE_DENSE:
      break;
    default:
      throw Exception("Input embedding " +
                      NF::InputEmbeddingFormat_Name(nf.input_embedding()) +
                      " is not supported by the directml backend.");
  }
  return std::make_unique<DirectMlNetwork<DataType>>(*w, options);
}

REGISTER_NETWORK("directml", MakeDirectMlNetwork<float>, 106)
REGISTER_NETWORK("directml-fp16", MakeDirectMlNetwork<DmlHalf>, 105)

}  // namespace directml_backend
}  // namespace lczero
