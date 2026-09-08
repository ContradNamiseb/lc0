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
#include <array>
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
#include <d3d12sdklayers.h>  // ID3D12Debug (LC0_DML_DEBUG_LAYER, opt-in)

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
DmlDeviceContext::~DmlDeviceContext() {
  // Safe after a partial Init. fence_event_ is null-initialised in the
  // header and is only ever assigned the result of CreateEvent below, so
  // every earlier throw in Init -- no adapter, D3D12CreateDevice, the DML
  // device, the command recorder -- leaves it null and this is a no-op.
  // CreateEvent failing likewise leaves it null before Init throws.
  if (fence_event_) {
    CloseHandle(fence_event_);
    fence_event_ = nullptr;
  }
}

void DmlDeviceContext::Init(const OptionsDict& options) {
  const int gpu_id = options.GetOrDefault<int>("gpu", 0);
  meta_commands_ = options.GetOrDefault<bool>("meta_commands", true);

  // Opt-in D3D12 debug layer (LC0_DML_DEBUG_LAYER=1; agora thread 19 #560,
  // verifying codex-sol's F2 resource-state claims with real validation
  // output rather than reasoning about the D3D12 spec from source alone).
  // Must be enabled before D3D12CreateDevice for it to take effect at all;
  // requires the Windows SDK's Graphics Tools optional feature installed,
  // so this fails soft (logs and continues without validation) rather than
  // throwing -- it is a diagnostic aid, not something normal operation
  // should ever depend on being available.
  if (getenv("LC0_DML_DEBUG_LAYER")) {
    ComPtr<ID3D12Debug> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
      debug->EnableDebugLayer();
      CERR << "directml backend: D3D12 debug layer enabled "
              "(LC0_DML_DEBUG_LAYER)";
      // GPU-Based Validation (agora thread 19 #591/#594/#597): standard
      // EnableDebugLayer() only validates API parameters and CPU-tracked
      // resource states -- it does NOT instrument shaders to catch actual
      // GPU-timeline hazards (e.g. a UAV read-after-write race on a shared
      // buffer). SetEnableGPUBasedValidation is the only path that does,
      // and codex-sol correctly flagged that F2's "zero validation errors"
      // clearance only ever had CPU-layer coverage. Deliberately a SEPARATE
      // opt-in from LC0_DML_DEBUG_LAYER (not folded into it): GBV carries a
      // documented 10x-50x runtime overhead from shader instrumentation, so
      // it must never turn on just because someone wanted the (cheap)
      // message-callback plumbing below.
      if (getenv("LC0_DML_GBV")) {
        ComPtr<ID3D12Debug1> debug1;
        if (SUCCEEDED(debug.As(&debug1))) {
          debug1->SetEnableGPUBasedValidation(TRUE);
          CERR << "directml backend: D3D12 GPU-Based Validation enabled "
                  "(LC0_DML_GBV)";
        } else {
          CERR << "directml backend: LC0_DML_GBV set but ID3D12Debug1 "
                  "unavailable -- continuing without GPU-based validation";
        }
      }
    } else {
      CERR << "directml backend: LC0_DML_DEBUG_LAYER set but "
              "D3D12GetDebugInterface failed (Graphics Tools optional "
              "feature not installed?) -- continuing without validation";
    }
  }

  ReportD3DErrors(CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory_)),
                  "CreateDXGIFactory1");
  UINT adapter_index = 0;
  UINT found = 0;
  // agora thread 19 #620 package A3: EnumAdapters1(i, &adapter_) wrote
  // straight into the member ComPtr every iteration, including right after
  // a SOFTWARE-adapter `continue` that skipped the old adapter_.Reset()
  // below -- the next call took &adapter_ while it still held a live,
  // non-null COM pointer from the skipped iteration (a WRL debug-build
  // assert, and ill-defined behavior in release). Reset() unconditionally
  // at the top of every iteration instead, so &adapter_ is always null
  // going into EnumAdapters1 regardless of which branch the previous
  // iteration took.
  for (UINT i = 0;; ++i) {
    adapter_.Reset();
    if (dxgi_factory_->EnumAdapters1(i, &adapter_) == DXGI_ERROR_NOT_FOUND) {
      break;
    }
    DXGI_ADAPTER_DESC1 desc;
    adapter_->GetDesc1(&desc);
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
    if (found == (UINT)gpu_id) {
      std::wstring name(desc.Description);
      CERR << "directml backend selected GPU: "
           << std::string(name.begin(), name.end());
      break;
    }
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

  // LC0_DML_DEBUG_LAYER, continued: the debug layer's own messages go to
  // OutputDebugString by default, invisible to a console/test-runner
  // capture -- register a real-time callback so validation errors (e.g.
  // codex-sol's F2 resource-state claims, agora thread 19 #560) actually
  // reach CERR instead of requiring an attached debugger to see. Needs
  // ID3D12InfoQueue1 (Windows 10 2004+ / SDK 19041+); fails soft if the
  // interface isn't there, same diagnostic-not-required posture as above.
  if (getenv("LC0_DML_DEBUG_LAYER")) {
    ComPtr<ID3D12InfoQueue1> info_queue;
    if (SUCCEEDED(device_->QueryInterface(IID_PPV_ARGS(&info_queue)))) {
      auto callback = [](D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY sev,
                         D3D12_MESSAGE_ID, LPCSTR description, void*) {
        CERR << "D3D12 debug layer [" << sev << "]: " << description;
      };
      DWORD cookie = 0;
      if (SUCCEEDED(info_queue->RegisterMessageCallback(
              callback, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr,
              &cookie))) {
        CERR << "directml backend: D3D12 debug layer message callback "
                "registered";
      }
    } else {
      CERR << "directml backend: LC0_DML_DEBUG_LAYER set but "
              "ID3D12InfoQueue1 unavailable -- validation messages will "
              "only reach an attached debugger, not this log";
    }
  }

  D3D12_COMMAND_QUEUE_DESC queue_desc = {};
  queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  ReportD3DErrors(device_->CreateCommandQueue(&queue_desc,
                                              IID_PPV_ARGS(&queue_)),
                  "CreateCommandQueue");

  // LC0_DML_PROFILE stage-timing profiler (agora thread 19 #545/#549, Phase
  // 3 Step 2): created only when requested, so the common path pays nothing
  // for it. GetTimestampFrequency needs queue_, hence placed right after it.
  if (getenv("LC0_DML_PROFILE")) {
    D3D12_QUERY_HEAP_DESC query_desc = {};
    query_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    query_desc.Count = kProfileQuerySlots;
    ReportD3DErrors(
        device_->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&profile_heap_)),
        "CreateQueryHeap (profile)");
    ReportD3DErrors(queue_->GetTimestampFrequency(&profile_frequency_),
                    "GetTimestampFrequency");
  }

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
  // The batch sizes that get compiled graphs. The pre-compile at load and
  // the round-up in forwardEval MUST walk the same list: this driver fails
  // all operator creation once dispatches have been recorded, so rounding a
  // batch up to a rung that was never compiled would try to compile after
  // dispatch and fail. Keeping one definition is what makes that
  // impossible -- they were two separate literal lists before.
  std::array<int, 14> BatchLadder() const {
    return {min_batch_size_, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
            max_batch_size_};
  }
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
  // Default 1, not 4 (agora thread 19 #533/#538/#545): at min_batch=4,
  // a batch-1 request evaluated 4 positions on the GPU and discarded 3,
  // measured costing DirectML FP32 batch-1 throughput -26% vs SYCL where
  // min_batch=1 measured +49% vs SYCL on the same net (kda-native-935532,
  // 12 clean rotated A/B runs). --backend-opts=min_batch=N still overrides
  // this for anyone who wants the old batching-for-latency tradeoff.
  min_batch_size_ = std::clamp(
      options.GetOrDefault<int>("min_batch", std::min(1, max_batch_size_)), 1,
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
  // sizeof(float)/elem: 1 in an fp32 network (elem==sizeof(float)), 2 in
  // fp16. Shared by every scratch term below whose underlying buffer is
  // forced to genuine float32 regardless of the network's own DataType
  // (KDA geometry, and now PE_DENSE's nhwc) -- hoisted here (was local to
  // the per-encoder loop below) so the PE_DENSE term added in #476/#478
  // can reuse it too, rather than risk a second, possibly-inconsistent
  // definition.
  const uint64_t scale_rec = sizeof(float) / elem;

  // Scratch estimate: the largest of what the body/policy/encoder phases
  // keep alive simultaneously (mirror of the SYCL backend's
  // getMaxAttentionBodySize/getMaxKdaBodySize reasoning).
  const uint64_t emb_size = weights_.ip_emb_b.size();
  uint64_t scratch_elems = 112 + kNumPosEncodingChannels;
  // agora thread 19 #476/#478 Phase 2: attention_preprocess.hlsl now
  // always writes genuine float32 for a PE_DENSE net's mode-1 output
  // (nhwc), regardless of elem -- this floor is exactly that per-token
  // width (112 = kNumInputPlanes + embedding_dense_size_, mirroring
  // AttentionBody's own derivation, ip_emb_preproc_b.size()/64, since that
  // member isn't available here). Confirmed NOT already covered by the
  // pre-existing floor above (112+kNumPosEncodingChannels=176 elements at
  // native elem width) once nhwc needs scale_rec=2x headroom on an fp16
  // PE_DENSE net -- e.g. MakePeDenseNet's dense_size=32 needs
  // (112+32)*2=288, which 176 does not cover. pos_info (mode 2) is not
  // separately scaled: its own per-token width is only 12 elements
  // (encoding_size argument to record_preprocess), comfortably under every
  // existing floor even doubled.
  if (nf.input_embedding() == NF::INPUT_EMBEDDING_PE_DENSE) {
    const uint64_t dense_size = weights_.ip_emb_preproc_b.size() / 64;
    scratch_elems =
        std::max(scratch_elems, (112 + dense_size) * scale_rec);
  }
  for (const auto& enc : weights_.encoder) {
    uint64_t need;
    if (enc.is_kda) {
      const uint64_t KD =
          (uint64_t)weights_.encoder_head_count * enc.kda.key_dim;
      const uint64_t VD =
          (uint64_t)weights_.encoder_head_count * enc.kda.value_dim;
      // note 2523/2494's fp32 recurrence boundary: q/k/v/raw_decay/beta (and
      // whatever the max(...) term below protects -- undocumented at
      // introduction, 1112af3, and the SYCL analogue this was adapted from,
      // getMaxKdaBodySize, doesn't reduce to the same expression either, so
      // its exact meaning isn't confirmed) are now sizeof(float) wide in a
      // fp16 network rather than elem, since EvalKda's scratch offsets
      // upcast them (see layers.cc). scale_rec is 1 in the fp32 network
      // (elem==sizeof(float)), so this is algebraically identical to the
      // formula it replaces there -- verified on MakeKdaMlhNet's dims
      // (KD=VD=32, key_dim=4, gate_rank=4, emb=32): both give 196. Scaling
      // the whole KDA-geometry term uniformly, including the unexplained
      // max(...) piece, is deliberately conservative rather than trying to
      // cleave exactly which sub-term needs it -- see agora thread 19
      // #431/#433 for the discarded formula that under-sized this by 64
      // elements at scale_rec=1, which would have been a silent
      // out-of-bounds UAV write, not merely a failed allocation.
      // agora thread 19 note 2530/#57: proj_input (the emb_size term) is now
      // float32-width whenever this net is fp16, whether or not THIS
      // encoder uses local_conv (the offset is always reserved, mirroring
      // the pre-existing unconditional-emb_size comment below it), so scale
      // it uniformly like every other term here. local_conv encoders
      // additionally need in_up, a second full emb_size-at-scale_rec
      // staging buffer that non-local-conv KDA encoders never allocate.
      need = (2 * KD + VD +
             std::max<uint64_t>(2 * KD, VD + 3 * enc.kda.key_dim)) *
                 scale_rec +
             enc.kda.gate_rank + emb_size * scale_rec +
             (enc.kda.local_conv ? emb_size * scale_rec : 0);
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
      //
      // agora thread 19 #488-#493: the always-FP32 MHA core makes q/k/v/qt/
      // kt/vt/ctx/merged all genuine float32 in a fp16 network now (see
      // EvalMha/mha_qkv_compiled_/mha_attn_compiled_ in layers.cc), not
      // native elem -- scale by scale_rec like every other FP32-boundary
      // term above. A no-op in the fp32 network (scale_rec == 1 there).
      need = 8 * d_model * scale_rec;
    }
    scratch_elems = std::max(scratch_elems, need);
  }
  // Policy encoders need the same treatment the body loop gives MHA
  // encoders, and did not have it: the terms below size the wq/wk/scores
  // layout and the LayerNorm temporaries, but nothing covered the policy
  // encoder's own q/k/v scratch (3 * d_model) or its buffer1 carve-up
  // (5 * d_model). Unlike the body path, AttentionPolicyHead::Eval starts
  // that carve-up at output + AlignUp(scratch_bytes_ / 2), half a scratch
  // into the slot, so the slot must hold scratch/2 + 5 * d_model. Since the
  // slot is sized >= scratch_bytes_, requiring scratch >= 10 * d_model makes
  // that hold by construction (scratch/2 + 5*d_model <= scratch <= slot) and
  // covers the 3 * d_model of q/k/v as well.
  const uint64_t pol_emb = policy_head.ip_pol_b.size();
  for (const auto& enc : policy_head.pol_encoder) {
    const uint64_t pol_d = (!enc.mha.q_w.empty() && pol_emb != 0)
                               ? enc.mha.q_w.size() / pol_emb
                               : pol_emb;
    // Same always-FP32 MHA core scaling as the body loop's MHA term above --
    // pol_encoder blocks are EncoderBlock<DataType>::EvalMha too.
    scratch_elems = std::max(scratch_elems, 10 * pol_d * scale_rec);
  }
  scratch_elems = std::max(
      {scratch_elems,
       // agora thread 19 #483/#484: wq (d_model), wk (d_model), and
       // scores (64/token) are now ALL genuine float32 in the policy
       // head (previously only wk/scores were) -- scale the whole term
       // by scale_rec like the KDA and PE_DENSE terms above, not just
       // the elements that changed this round, so a future promotion of
       // any remaining native-width term here doesn't need yet another
       // audit of which half of this sum needs it.
       (2 * policy_head.ip2_pol_b.size() + 64) * scale_rec,
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
  // note 2523/2494's fourth fp16 defect candidate (agora thread 19 #453):
  // policy_finalize.hlsl now always writes the 4288-wide policy row as
  // FLOAT32 into this same tensor slot, regardless of the network's own
  // DataType -- size that term at sizeof(float) specifically rather than
  // folding it into the uniform max(emb,4288)*elem multiplication, which
  // would under-size it by half whenever elem==2 (fp16). At elem==4 (fp32)
  // this is algebraically identical to the formula it replaces: both
  // reduce to max(emb, 4288)*4.
  // F7 (agora thread 19 #560/#573, codex-sol's independent review): this
  // term used max_tokens (max_batch_size_ * 64), but policy_finalize.hlsl's
  // output is one [4288]-wide row PER BATCH SAMPLE, not per token/square --
  // the embedding term right above it is genuinely tokens-scaled (each of
  // the 64 squares gets its own row), but the policy row count is batch,
  // never batch*64. At max_batch_size_=256 this was a 64x overestimate
  // (~268MB per tensor slot, ~804MB across the three rotating slots) for
  // no correctness benefit -- an oversizing bug, not a correctness one
  // (more arena than needed doesn't corrupt anything), but real wasted
  // VRAM. Confirmed emb-term and this term are otherwise independent
  // maximands (the comment above about "both reduce to max(emb,4288)*4"
  // describes the fp32-elem-width identity between the two formulas, not
  // a tokens/batch relationship, so this fix doesn't undo that reasoning).
  // agora thread 19 #620 package C2: the "raw NCHW input size" term above
  // (max_layer_bytes' initial value, max_tokens*64*elem) claims to size the
  // raw input upload, but the upload it actually needs to hold is
  // kNumInputPlanes(112) planes wide and ALWAYS float32 -- forwardEval's
  // CopyBufferRegion into this same tensor-arena slot copies exactly
  // `batch * kNumInputPlanes * 64 * sizeof(float)` bytes (see below),
  // independent of the network's own DataType. The existing term uses 64
  // (not 112) planes and scales by `elem` (the network's DataType width,
  // 2 for fp16) instead of sizeof(float) -- the same "always-float,
  // wrongly elem-scaled" bug class F7 already fixed for the policy term
  // above. At fp32 this term happens to be masked by scratch_bytes_'s own
  // floor; a small-embedding fp16 net is not guaranteed the same rescue.
  // Added as an extra maximand (never removes headroom, only adds it if
  // the other terms were already sufficient).
  max_layer_bytes = std::max(
      {max_layer_bytes, max_tokens * emb * elem,
       max_tokens * (uint64_t)kNumInputPlanes * sizeof(float),
       (uint64_t)max_batch_size_ * (uint64_t)4288 * sizeof(float)});
  tensor_slot_bytes_ = std::max(max_layer_bytes, scratch_bytes_);

  // Belt-and-braces on the policy encoder carve-up, in exact bytes.
  //
  // The body path needs no such check: its scratch term is already 8 *
  // d_model (3 for q/k/v, 5 for the buffer1 carve-up) and the slot is sized
  // >= scratch_bytes_, so any 5 * d_model requirement is satisfied by
  // construction. An earlier version of this block asserted that anyway and
  // was therefore a tautology; it also derived a "breaks at d_model >= 858"
  // threshold by holding the slot at 4288 while growing d_model, which is
  // wrong because the slot grows with d_model through exactly that scratch
  // term.
  //
  // The policy path is the one that was uncovered, and the scratch term
  // above now fixes it. This asserts the resulting inequality in the bytes
  // AttentionPolicyHead::Eval actually uses -- AlignUp(scratch_bytes_ / 2),
  // not a truncating scratch_elems / 2 -- so that a future change to either
  // side is caught rather than assumed.
  for (const auto& enc : policy_head.pol_encoder) {
    const uint64_t pol_d = (!enc.mha.q_w.empty() && pol_emb != 0)
                               ? enc.mha.q_w.size() / pol_emb
                               : pol_emb;
    // Independent review, agora thread 19 #37/#512 (muse-spark): this guard
    // was sized in elem (native DataType width) but EvalMha's actual 5
    // scratch regions (qt/kt/vt/ctx/merged) and 3 q/k/v regions are always
    // genuine float32 now (the always-FP32 MHA core, thread 19 #488-#493,
    // layers.cc EvalMha's own S = max_tokens*d_model*sizeof(float)) --
    // understating this guard by up to 2x on fp16 nets with policy-encoder
    // blocks. sizeof(float), not elem, matches what Eval actually writes.
    const uint64_t needed =
        AlignUp(scratch_bytes_ / 2) + 5 * max_tokens * pol_d * sizeof(float);
    if (needed > tensor_slot_bytes_) {
      throw Exception(
          "directml backend: this net's policy encoder needs " +
          std::to_string(needed) + " bytes of tensor slot but only " +
          std::to_string(tensor_slot_bytes_) + " are sized (policy d_model " +
          std::to_string(pol_d) + ", max_batch " +
          std::to_string(max_batch_size_) + ").");
    }
    const uint64_t qkv_needed = 3 * max_tokens * pol_d * sizeof(float);
    if (qkv_needed > scratch_bytes_) {
      throw Exception(
          "directml backend: this net's policy encoder needs " +
          std::to_string(qkv_needed) + " bytes of scratch for q/k/v but only " +
          std::to_string(scratch_bytes_) + " are sized (policy d_model " +
          std::to_string(pol_d) + ").");
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
  // ln_scratch (layers.cc:1789, :3287) starts at scratch + AlignUp(scratch_
  // bytes_) and writes up to scratch_bytes_ more bytes past that offset. A
  // plain scratch_bytes_ * 2 allocation only guarantees scratch_bytes_ bytes
  // remain there when scratch_bytes_ is already a multiple of the 256-byte
  // alignment; otherwise ln_scratch's write can run past the arena by up to
  // AlignUp(scratch_bytes_) - scratch_bytes_ (<=255) bytes -- a real OOB UAV
  // write on an exact-fit net, not just a theoretical one. Sizing the arena
  // to AlignUp(scratch_bytes_) + scratch_bytes_ instead guarantees the full
  // scratch_bytes_ bytes are present after the aligned ln_scratch offset,
  // by construction, for at most 255 bytes of extra VRAM. See agora thread
  // 19/37 (MEDIUM 1, muse-spark independent review, ratified by
  // gemini-antigravity).
  scratch_arena_.Create(ctx_.device(),
                        AlignUp(scratch_bytes_) + scratch_bytes_, "scratch");
  // 256MB transient: the MHA attention graph's [B=N*H,64,64] scores +
  // softmax temporaries reach ~2*B*4096*4 bytes (134MB at max batch with 16
  // heads); 64MB would throw arena-exhausted on large batches.
  transient_arena_.Create(ctx_.device(), 256 * 1024 * 1024, "transient");

  DmlWeightUploader uploader(&weight_arena_);

  const bool is_pe_dense =
      nf.input_embedding() == NF::INPUT_EMBEDDING_PE_DENSE;

  // Same check the reference performs. DirectML uploads these tensors by
  // size (layers.cc:1203, 1209), so an absent one yields a zero-byte arena
  // allocation that a WeightChannel then reads across -- a potential invalid
  // read rather than the reference's outright fault, which is a worse way to
  // find out.
  ValidateEmbeddingNormWeights(weights_, is_pe_dense);
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
      // RR4 (agora thread 19 #650/#652): an unchecked Map leaves z
      // indeterminate on failure, and memset below would then write through
      // a garbage pointer instead of failing cleanly at the Map call site --
      // same class as the InputsOutputs constructor's A2 fix.
      uint8_t* z = nullptr;
      ReportD3DErrors(zeros->Map(0, nullptr, reinterpret_cast<void**>(&z)),
                      "Map (arena-clear zeros)");
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
  for (int b : BatchLadder()) {
    if (b < min_batch_size_ || b > max_batch_size_) continue;
    DmlExecScope pre(ctx_, ctx_.upload_list(), &transient_arena_,
                     &smolgen_arena_);
    for (auto& layer : network_) layer->EnsureCompiled(b, pre);
  }

  // Initialize every compiled operator exactly once, now that the batch
  // ladder above is done and nothing further will ever be compiled (agora
  // thread 19 #591/#594/#602 -- see InitializeCompiledOperators's comment
  // for why this must happen before any operator's first real dispatch).
  // Same Reset/record/Close/Execute/Signal/Wait drill as the arena-clear
  // step above, on the same one-shot upload command list.
  {
    ReportD3DErrors(ctx_.upload_allocator()->Reset(), "Reset (init ops)");
    ReportD3DErrors(
        ctx_.upload_list()->Reset(ctx_.upload_allocator(), nullptr),
        "Reset (init ops list)");
    ctx_.InitializeCompiledOperators(ctx_.upload_list());
    ReportD3DErrors(ctx_.upload_list()->Close(), "Close (init ops)");
    ID3D12CommandList* lists[] = {ctx_.upload_list()};
    ctx_.queue()->ExecuteCommandLists(1, lists);
    const uint64_t fence_value = ctx_.NextUploadFenceValue();
    ReportD3DErrors(ctx_.queue()->Signal(ctx_.fence(), fence_value),
                    "Signal (init ops)");
    ctx_.WaitForFence(ctx_.fence(), fence_value);
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
  // RR4 (agora thread 19 #650/#652): unchecked, same class as the arena-clear
  // and InputsOutputs A2 fixes -- every weight-copy write below through
  // `mapped` would otherwise run against a garbage pointer on failure.
  uint8_t* mapped = nullptr;
  ReportD3DErrors(staging->Map(0, nullptr, reinterpret_cast<void**>(&mapped)),
                  "Map (weight staging)");

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
                                           wdl_, moves_left_,
                                           sizeof(DataType));
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
  // A batch above the top rung would fall straight through the loop below
  // with its size unchanged, and EnsureCompiled would then compile a graph
  // for a size the load-time pre-compile never saw -- compilation after
  // dispatch, which this driver answers with bogus errors for every
  // subsequent operator. The arenas are sized for max_batch too, so this
  // cannot be served by padding either. Fail loudly instead.
  if (batch > max_batch_size_) {
    throw Exception("directml backend asked to evaluate a batch of " +
                    std::to_string(batch) + ", above the configured max_batch of " +
                    std::to_string(max_batch_size_) +
                    "; raise max_batch so its graphs are compiled at load.");
  }
  // Round UP to the pre-compiled ladder: only these sizes have graphs (see
  // the pre-compile at load -- post-dispatch compilation fails on this
  // driver). The layers are row/batch independent, so extra padding rows
  // are harmless; buffers are sized for max_batch.
  for (int b : BatchLadder()) {
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
  // Anchor mark: without this, "embedding" (the first named stage) would
  // have no prior mark to diff against and ProfileStage's own display logic
  // would print it as a meaningless 0us, silently hiding the input-upload
  // barrier/copy below plus the embedding dispatch itself -- exactly the
  // gap Phase 3 Step 2 was supposed to measure. See ProfileMarks' comment.
  ProfileStage("start", scope);

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
  ProfileStage("policy_head", scope);

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
    // Both sides of the copy need a state transition, not just the
    // destination (agora thread 19 #560, codex-sol's F1: the scratch
    // source stayed in UNORDERED_ACCESS through this CopyBufferRegion,
    // invalid D3D12 usage per the resource-state contract regardless of
    // whether this driver happens to tolerate it silently today). scratch
    // is a shared arena other dispatches read/write throughout forwardEval
    // (the moves-left head's own Eval calls immediately below, for one),
    // so it must be back in UNORDERED_ACCESS before anything else touches
    // it -- not left in COPY_SOURCE.
    D3D12_RESOURCE_BARRIER vpre[2] = {};
    vpre[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    vpre[0].Transition.pResource = io->value_gpu_.Get();
    vpre[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    vpre[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    vpre[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    vpre[1].Transition.pResource = scratch_arena_.resource();
    vpre[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    vpre[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(2, vpre);
    list->CopyBufferRegion(io->value_gpu_.Get(), 0, scratch_arena_.resource(),
                           0,
                           (uint64_t)batch * (wdl_ ? 3 : 1) * sizeof(DataType));
    D3D12_RESOURCE_BARRIER vpost[2] = {};
    vpost[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    vpost[0].Transition.pResource = io->value_gpu_.Get();
    vpost[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    vpost[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    vpost[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    vpost[1].Transition.pResource = scratch_arena_.resource();
    vpost[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    vpost[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    list->ResourceBarrier(2, vpost);
  }
  ProfileStage("value_head", scope);

  // Moves left head. Same scratch + copy workaround as the value head above
  // (network_directml.cc:835-843's comment): binding this graph's output
  // straight to io->moves_left_gpu_ hits the identical driver quirk (silent
  // no-op write to an io buffer), just never worked around here before.
  if (moves_left_) {
    network_[l++]->Eval(batch, spare1, flow, spare2, scratch, scratch_bytes_,
                        scope);
    network_[l++]->Eval(batch, spare2, spare1, spare2, scratch, scratch_bytes_,
                        scope);
    DmlPtr op_mov_scratch(scratch_arena_.resource(), 0);
    network_[l++]->Eval(batch, op_mov_scratch, spare2, spare2, scratch,
                        scratch_bytes_, scope);
    // Same F1 fix as the value head's copy above -- both sides transition,
    // scratch restored to UNORDERED_ACCESS after (agora thread 19 #560).
    D3D12_RESOURCE_BARRIER mpre[2] = {};
    mpre[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    mpre[0].Transition.pResource = io->moves_left_gpu_.Get();
    mpre[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    mpre[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    mpre[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    mpre[1].Transition.pResource = scratch_arena_.resource();
    mpre[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    mpre[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(2, mpre);
    list->CopyBufferRegion(io->moves_left_gpu_.Get(), 0,
                           scratch_arena_.resource(), 0,
                           (uint64_t)batch * sizeof(DataType));
    D3D12_RESOURCE_BARRIER mpost[2] = {};
    mpost[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    mpost[0].Transition.pResource = io->moves_left_gpu_.Get();
    mpost[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    mpost[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    mpost[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    mpost[1].Transition.pResource = scratch_arena_.resource();
    mpost[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    mpost[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    list->ResourceBarrier(2, mpost);
    ProfileStage("movesleft_head", scope);
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
           (uint64_t)batch * kNumOutputPolicy * sizeof(DataType));
  readback(io->value_gpu_.Get(), io->value_readback_.Get(),
           (uint64_t)batch * (wdl_ ? 3 : 1) * sizeof(DataType));
  if (moves_left_) {
    readback(io->moves_left_gpu_.Get(), io->moves_left_readback_.Get(),
             (uint64_t)batch * sizeof(DataType));
  }

  // LC0_DML_PROFILE (agora thread 19 #545/#549, Phase 3 Step 2):
  // ResolveQueryData is itself a command list op -- it must be recorded
  // before Close(), unlike the CPU-side readback of its destination buffer
  // below, which has to wait for the fence like everything else GPU-written.
  // agora thread 19 RR3/P3: scope.ProfileMarks() is this call's own vector
  // now (dml_common.h's DmlExecScope), not the process-wide static the R5
  // "original F10 remains" gap used to name -- no lock needed, and no
  // separate-critical-sections dance either, since nothing outside this one
  // forwardEval call can ever see or touch this scope's vectors regardless
  // of what happens across the fence wait below.
  ComPtr<ID3D12Resource> profile_readback;
  const size_t profile_marks = scope.ProfileMarks().size();
  if (profile_marks > 0 && ctx_.profile_heap()) {
    profile_readback = detail::CreateBuffer(
        ctx_.device(), profile_marks * sizeof(UINT64),
        D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    list->ResolveQueryData(ctx_.profile_heap(), D3D12_QUERY_TYPE_TIMESTAMP, 0,
                           static_cast<UINT>(profile_marks),
                           profile_readback.Get(), 0);
  }

  ReportD3DErrors(list->Close(), "Close");
  ID3D12CommandList* lists[] = {list};
  ctx_.queue()->ExecuteCommandLists(1, lists);
  ++io->fence_value_;
  ReportD3DErrors(ctx_.queue()->Signal(io->fence_.Get(), io->fence_value_),
                  "Signal");
  ctx_.WaitForFence(io->fence_.Get(), io->fence_value_);
  {
    // RR3/P3: no lock -- scope.ProfileMarks()/scope.BodyDumps() are this
    // call's own vectors, unreachable from any other DirectMlNetwork
    // instance or forwardEval call. This block only runs after Close/
    // Execute/Signal/WaitForFence all succeeded, so every dump readback
    // here was genuinely written by this call's own GPU work.
    if (profile_readback) {
      // RR4 (agora thread 19 #650/#652): unchecked, same class as the other
      // three sites -- the stamps[] reads below would otherwise dereference
      // a garbage pointer on a failed Map.
      const UINT64* stamps = nullptr;
      ReportD3DErrors(
          profile_readback->Map(
              0, nullptr,
              const_cast<void**>(reinterpret_cast<const void**>(&stamps))),
          "Map (profile readback)");
      const double freq = static_cast<double>(ctx_.profile_frequency());
      CERR << "LC0_DML_PROFILE batch=" << batch << " (" << profile_marks
          << " stage marks, us since previous mark):";
      for (size_t i = 0; i < profile_marks; ++i) {
        const double us =
            i == 0 ? 0.0
                   : (static_cast<double>(stamps[i] - stamps[i - 1]) / freq) *
                         1e6;
        CERR << "  " << scope.ProfileMarks()[i] << ": " << us << " us";
      }
      profile_readback->Unmap(0, nullptr);
      scope.ProfileMarks().clear();
    }
    if (const char* prefix = getenv("LC0_DUMP_BODY")) {
      for (auto& d : scope.BodyDumps()) {
        // RR4 (agora thread 19 #650/#652): unchecked, same class as above --
        // the f.write below would otherwise read a garbage-pointer range on
        // a failed Map.
        const float* p = nullptr;
        ReportD3DErrors(
            d.readback->Map(
                0, nullptr,
                const_cast<void**>(reinterpret_cast<const void**>(&p))),
            "Map (body dump readback)");
        std::ofstream f(std::string(prefix) + ".dml." + d.stage + ".bin",
                        std::ios::binary);
        f.write(reinterpret_cast<const char*>(p), d.bytes);
        d.readback->Unmap(0, nullptr);
      }
      scope.BodyDumps().clear();
    }
  }
  // fp16 output conversion: the readback buffers above just landed raw
  // DataType bits (DmlHalf when fp16); *_mapped_ is defined to always be a
  // float view (see inputs_outputs.h), so convert host-side before anything
  // below reads through it. A no-op for the fp32 network, where *_mapped_
  // already aliases the readback mapping directly.
  if (std::is_same<DataType, DmlHalf>::value) {
    auto convert = [](const void* raw, const float* dst, size_t n) {
      const DmlHalf* h = reinterpret_cast<const DmlHalf*>(raw);
      float* out = const_cast<float*>(dst);
      for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(h[i]);
    };
    convert(io->policy_readback_mapped_, io->policy_mapped_,
            (size_t)batch * kNumOutputPolicy);
    convert(io->value_readback_mapped_, io->value_mapped_,
            (size_t)batch * (wdl_ ? 3 : 1));
    if (moves_left_) {
      convert(io->moves_left_readback_mapped_, io->moves_left_mapped_,
              (size_t)batch);
    }
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
    // F8 (agora thread 19 #560/#573, codex-sol's independent review): an
    // empty computation (NewComputation followed by ComputeBlocking with
    // no AddInput calls) used to reach forwardEval with batch=0 and an
    // empty planes_ vector. forwardEval raises batch up to min_batch_size_
    // internally, but planes_ stays empty, so its per-sample loop's
    // `planes[std::min(n, planes.size() - 1)]` underflows
    // (planes.size() - 1 wraps to SIZE_MAX for an empty vector) and reads
    // out of bounds. ONNX already treats an empty computation as a no-op;
    // mirror that here instead of ever calling forwardEval with nothing to
    // evaluate.
    if (planes_.empty()) return;
    // Safety check added by user, don't remove -- an explicit belt-and-
    // suspenders guard against empty batches, which have shown up in other
    // backends before. GetBatchSize() is planes_.size(), so this can never
    // actually fire given the guard just above (the two conditions are the
    // same test), but it costs nothing and stays as a second line of
    // defense in case that invariant ever changes.
    if (GetBatchSize() == 0) return;
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
