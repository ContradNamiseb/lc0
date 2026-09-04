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

#include "neural/backends/directml/layers.h"

// <span>/<version> first: DirectMLX.h only includes <span> when
// __cpp_lib_span is already visible, and falls back to a detail::span that
// leaves later std::span uses broken under this toolchain.
#include <span>
#include <version>
#include <DirectMLX.h>
#include <d3dcompiler.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "neural/backends/directml/attention_preprocess_shader_source.h"
#include "neural/backends/directml/kda_local_conv_shader_source.h"
#include "neural/backends/directml/kda_recurrence_shader_source.h"
#include "neural/backends/directml/layer_norm_shader_source.h"
#include "neural/backends/directml/mha_transpose_shader_source.h"
#include "neural/backends/directml/policy_finalize_shader_source.h"
#include "neural/backends/directml/smolgen_bias_shader_source.h"
#include "neural/tables/attention_policy_map.h"
#include "utils/exception.h"
#include "utils/logging.h"

#pragma comment(lib, "d3dcompiler.lib")

namespace lczero {
namespace directml_backend {

// ===========================================================================
// Host fp16 conversion (round-to-nearest-even bit tricks, the standard
// half-float algorithm; same idea as FP32toFP16 in cuda/inputs_outputs.h).
// ===========================================================================
namespace {

inline uint16_t F32toF16Bits(float value) {
  uint32_t f;
  std::memcpy(&f, &value, sizeof(f));
  const uint32_t sign = (f >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((f >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = (f >> 13) & 0x3ffu;
  if (((f >> 23) & 0xffu) == 0xffu) {  // Inf/NaN
    return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
  }
  if (exp >= 0x1f) {
    return static_cast<uint16_t>(sign | 0x7c00u);  // overflow -> Inf
  }
  if (exp <= 0) {  // subnormal or zero
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x400u;
    const uint32_t shift = static_cast<uint32_t>(1 - exp);
    const uint32_t half = static_cast<uint32_t>(mant >> shift);
    const uint32_t rem = mant & ((1u << shift) - 1);
    const uint32_t mid = 1u << (shift - 1);
    uint32_t sub = half + ((rem > mid || (rem == mid && (half & 1u))) ? 1u : 0u);
    return static_cast<uint16_t>(sign | sub);
  }
  // Round mantissa to nearest even.
  const uint32_t rem = f & 0x1fffu;
  mant += (rem > 0x1000u || (rem == 0x1000u && (mant & 1u))) ? 1u : 0u;
  if (mant > 0x3ffu) {
    mant = 0;
    ++exp;
    if (exp >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00u);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

inline float F16BitsToF32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {  // subnormal
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3ffu;
      f = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1f) {
    f = sign | 0x7f800000u | (mant << 13);
  } else {
    f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

// ===========================================================================
// Shader compilation (runtime FXC, once per PSO at load) and a generic
// root-signature layout: [root constants][SRVs][UAVs]. The KDA recurrence
// keeps its bespoke 10-parameter layout from the original stub.
// ===========================================================================

ComPtr<ID3DBlob> CompileHlsl(
    const char* source, size_t source_len, const char* filename,
    const char* entry_point, bool fp16,
    const std::vector<std::pair<std::string, std::string>>& extra_defines =
        {}) {
  // Owned storage: D3D_SHADER_MACRO holds bare pointers into whatever the
  // caller passed, and the terminator must stay a {nullptr, nullptr} pair.
  std::vector<D3D_SHADER_MACRO> macros;
  macros.push_back({"INPUT_TYPE", fp16 ? "half" : "float"});
  for (const auto& d : extra_defines) {
    macros.push_back({d.first.c_str(), d.second.c_str()});
  }
  macros.push_back({nullptr, nullptr});
  UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL3;
#ifndef NDEBUG
  flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
  // FXC caps at shader model 5.1 where `half` is min16float; genuine 16-bit
  // storage would need DXC (cs_6_2 + -enable-16bit-types), same caveat as
  // kda_recurrence.hlsl.
  ComPtr<ID3DBlob> shader, errors;
  HRESULT hr = D3DCompile(source, source_len, filename, macros.data(),
                          nullptr /* no shader #includes */, entry_point,
                          "cs_5_1", flags, 0, &shader, &errors);
  if (FAILED(hr)) {
    std::string message = "Failed to compile ";
    message += filename;
    message += ": ";
    if (errors) {
      message.append(static_cast<const char*>(errors->GetBufferPointer()),
                     errors->GetBufferSize());
    } else {
      message += "HRESULT 0x" + std::to_string(static_cast<uint32_t>(hr));
    }
    throw Exception(message);
  }
  return shader;
}

// Root signature: parameter 0 = num_32bit_constants root constants,
// then srv_count root SRVs (t0..), then uav_count root UAVs (u0..).
ComPtr<ID3D12RootSignature> CreateShaderRootSignature(ID3D12Device* device,
                                                      UINT num_32bit_constants,
                                                      UINT srv_count,
                                                      UINT uav_count) {
  std::vector<D3D12_ROOT_PARAMETER> params(!!num_32bit_constants + srv_count +
                                           uav_count);
  UINT next = 0;
  if (num_32bit_constants) {
    params[next].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[next].Constants.ShaderRegister = 0;
    params[next].Constants.RegisterSpace = 0;
    params[next].Constants.Num32BitValues = num_32bit_constants;
    params[next].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    ++next;
  }
  for (UINT i = 0; i < srv_count; ++i, ++next) {
    params[next].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[next].Descriptor.ShaderRegister = i;
    params[next].Descriptor.RegisterSpace = 0;
    params[next].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  for (UINT i = 0; i < uav_count; ++i, ++next) {
    params[next].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[next].Descriptor.ShaderRegister = i;
    params[next].Descriptor.RegisterSpace = 0;
    params[next].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_ROOT_SIGNATURE_DESC desc = {};
  desc.NumParameters = static_cast<UINT>(params.size());
  desc.pParameters = params.data();
  ComPtr<ID3DBlob> signature, errors;
  ReportD3DErrors(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                               &signature, &errors),
                  "SerializeRootSignature");
  ComPtr<ID3D12RootSignature> root;
  ReportD3DErrors(device->CreateRootSignature(0, signature->GetBufferPointer(),
                                              signature->GetBufferSize(),
                                              IID_PPV_ARGS(&root)),
                  "CreateRootSignature");
  return root;
}

ComPtr<ID3D12PipelineState> CreateComputePso(ID3D12Device* device,
                                             ID3D12RootSignature* root,
                                             ID3DBlob* shader) {
  D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
  desc.pRootSignature = root;
  desc.CS = {shader->GetBufferPointer(), shader->GetBufferSize()};
  ComPtr<ID3D12PipelineState> pso;
  ReportD3DErrors(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)),
                  "CreateComputePipelineState");
  return pso;
}

// A persistently-mapped upload buffer for one shader's root constants --
// recorded once per dispatch with CopyBufferRegion + a constant-buffer root
// view would be the alternative; root constants set directly on the command
// list are simpler and avoid the extra copy, so this is only used where the
// constant payload exceeds what SetComputeRoot32BitConstants conveniently
// expresses (nothing today -- kept for symmetry with inputs_outputs.h).
constexpr UINT kMaxRootConstants = 16;

// ===========================================================================
// GraphFactory: wraps dml::Graph construction so every layer records its
// graph inputs (weights and activations) in one ordered list and compiles a
// DmlCompiledOp. The DirectML equivalent of writing a CUDA kernel launcher:
// shapes are baked per batch size, exactly like a CUDA graph per batch size.
// ===========================================================================
template <typename DataType>
class GraphFactory {
 public:
  explicit GraphFactory(DmlDeviceContext& ctx)
      : graph_(ctx.dml_device()), ctx_(ctx) {}

  dml::Graph& graph() { return graph_; }

  using Sizes = std::vector<uint32_t>;

  dml::Expression Weight(const DmlPtr& w, Sizes sizes) {
    return AddTensor(w, DmlBindingRef::Kind::kWeight, std::move(sizes), {});
  }
  // UINT32 weight tensor (policy-map gather indices): same binding
  // mechanics, different element type.
  dml::Expression WeightU32(const DmlPtr& w, Sizes sizes) {
    const uint64_t bytes = std::accumulate(
        sizes.begin(), sizes.end(), uint64_t{4},
        [](uint64_t a, uint32_t b) { return a * b; });
    dml::TensorDesc desc(DML_TENSOR_DATA_TYPE_UINT32, sizes);
    dml::Expression e = dml::InputTensor(graph_, next_input_++, desc);
    bindings_.push_back({DmlBindingRef::Kind::kWeight, w, bytes});
    return e;
  }
  // Strided weight view (e.g. zero-stride batch broadcast of the shared
  // smolgen table); strides are in elements.
  dml::Expression WeightStrided(const DmlPtr& w, Sizes sizes, Sizes strides) {
    return AddTensor(w, DmlBindingRef::Kind::kWeight, std::move(sizes),
                     std::move(strides));
  }
  // Per-channel weight broadcast to [1, 1, rows, C] via a zero row stride:
  // DirectML's base elementwise operators take no broadcast shapes -- the
  // idiom (same one the ONNX Runtime DML EP uses) is matching shapes with
  // stride-0 replication.
  dml::Expression WeightChannel(const DmlPtr& w, uint32_t rows, uint32_t C) {
    return WeightStrided(w, {1, 1, rows, C}, {0, 0, 0, 1});
  }
  dml::Expression Input(Sizes sizes, Sizes strides = {}) {
    return AddTensor({}, DmlBindingRef::Kind::kInput, std::move(sizes),
                     std::move(strides));
  }
  dml::Expression Input2(Sizes sizes, Sizes strides = {}) {
    return AddTensor({}, DmlBindingRef::Kind::kInput2, std::move(sizes),
                     std::move(strides));
  }
  dml::Expression Scratch(Sizes sizes, Sizes strides = {}) {
    return AddTensor({}, DmlBindingRef::Kind::kScratch, std::move(sizes),
                     std::move(strides));
  }
  dml::Expression Extra(Sizes sizes, Sizes strides = {}) {
    return AddTensor({}, DmlBindingRef::Kind::kExtra, std::move(sizes),
                     std::move(strides));
  }

  // Strided dense-memory view of an intermediate expression (the
  // no-transpose-operator workaround for the policy scores GEMM, whose
  // [N,1] batch compiles: [N,H>1]-batched strided GEMMs do not -- see
  // mha_transpose.hlsl). An empty stride list means a dense reshape
  // (NullOpt) -- passing an engaged-but-empty stride array leaves
  // DimensionCount=4 with a garbage/empty stride pointer, which is
  // undefined behavior.
  static dml::Expression ReinterpretView(dml::Expression e, Sizes sizes,
                                         Sizes strides = {}) {
    dml::TensorDimensions dims(sizes.begin(), sizes.end());
    if (strides.empty()) {
      return dml::Reinterpret(e, e.Impl()->GetOutputDesc().dataType, dims,
                              dml::NullOpt);
    }
    dml::TensorStrides ts(strides.begin(), strides.end());
    return dml::Reinterpret(e, e.Impl()->GetOutputDesc().dataType, dims, ts);
  }

  DmlCompiledOp Compile(std::vector<dml::Expression> outputs,
                        std::vector<uint64_t> output_bytes) {
    DmlCompiledOp out;
    out.op = graph_.Compile(ctx_.meta_commands()
                                ? DML_EXECUTION_FLAG_NONE
                                : DML_EXECUTION_FLAG_DISABLE_META_COMMANDS,
                            outputs);
    DML_BINDING_PROPERTIES props = out.op->GetBindingProperties();
    if (props.PersistentResourceSize != 0) {
      throw Exception(
          "directml backend: unexpectedly received a persistent-resource "
          "operator (no DML_TENSOR_FLAG_OWNED_BY_DML tensors exist here)");
    }
    out.transient_bytes = props.TemporaryResourceSize;
    out.bindings = std::move(bindings_);
    out.output_bytes = std::move(output_bytes);
    // Create the op's binding table NOW, in the build phase: this driver
    // fails DML object creation with bogus errors once dispatches have
    // been recorded (see docs/directml-handoff.md section 3), so every
    // table must exist before the first RecordDispatch.
    ctx_.GetOrCreateBindingTable(out.op.Get());
    return out;
  }

 private:
  dml::TensorDesc MakeDesc(const Sizes& sizes, const Sizes& strides) {
    const uint64_t elements = std::accumulate(
        sizes.begin(), sizes.end(), uint64_t{1},
        [](uint64_t a, uint32_t b) { return a * b; });
    if (strides.empty()) {
      return dml::TensorDesc(DmlTensorType<DataType>(), sizes);
    }
    uint64_t span = 1;
    for (size_t i = 0; i < sizes.size(); ++i) {
      span += (uint64_t(sizes[i]) - 1) * strides[i];
    }
    dml::TensorDesc desc;
    desc.dataType = DmlTensorType<DataType>();
    desc.flags = DML_TENSOR_FLAG_NONE;
    desc.sizes.assign(sizes.begin(), sizes.end());
    desc.strides.emplace(strides.begin(), strides.end());
    desc.totalTensorSizeInBytes = span * sizeof(DataType);
    desc.guaranteedBaseOffsetAlignment = 0;
    return desc;
  }

  dml::Expression AddTensor(const DmlPtr& w, DmlBindingRef::Kind kind,
                            Sizes sizes, Sizes strides) {
    const uint64_t bytes = [&] {
      uint64_t span = 1;
      for (size_t i = 0; i < sizes.size(); ++i) {
        span += (uint64_t(sizes[i]) - 1) * (strides.empty() ? sizes[i] : strides[i]);
      }
      return span * sizeof(DataType);
    }();
    dml::Expression e =
        dml::InputTensor(graph_, next_input_++, MakeDesc(sizes, strides));
    bindings_.push_back({kind, w, bytes});
    return e;
  }

  dml::Graph graph_;
  DmlDeviceContext& ctx_;
  std::vector<DmlBindingRef> bindings_;
  uint32_t next_input_ = 0;
};

// Activation as a dml expression. Mish has no single DirectML op and is
// composed exactly as the CUDA/SYCL activate() defines it:
// x * tanh(softplus(x)). Swish likewise: x * sigmoid(x). RELU_2 is
// relu(x)^2.
template <typename DataType>
dml::Expression ActivationExpr(ActivationFunction act, dml::Expression x) {
  switch (act) {
    case ACTIVATION_NONE:
    case ACTIVATION_DEFAULT:
      return x;
    case ACTIVATION_RELU:
      return dml::ActivationRelu(x);
    case ACTIVATION_MISH:
      return x * dml::ActivationTanh(dml::ActivationSoftplus(x));
    case ACTIVATION_TANH:
      return dml::ActivationTanh(x);
    case ACTIVATION_SIGMOID:
      return dml::ActivationSigmoid(x);
    case ACTIVATION_SWISH:
      return x * dml::ActivationSigmoid(x);
    case ACTIVATION_RELU_2:
      return dml::ActivationRelu(x) * dml::ActivationRelu(x);
    default:
      throw Exception(
          "directml backend: activation not supported in graphs");
  }
}

// Layer normalization, mirroring the SYCL layer_norm_kernel exactly:
//   t = act(input + bias) * alpha + skip
//   y = gamma * (t - mean) / sqrt(variance + eps) + beta
// where mean/variance are over the last (channel) dimension and bias/skip
// may be absent (the bias is the previous gemm's, fused here as in the CUDA
// backend's LayerNorm kernel). All tensors are [1, 1, T, C] except the
// per-channel ones which are [1, 1, 1, C].
template <typename DataType>
dml::Expression LayerNormExpr(dml::Expression input,
                              dml::Expression* bias, dml::Expression* skip,
                              dml::Expression& gammas, dml::Expression& betas,
                              float alpha, float eps,
                              ActivationFunction act) {
  dml::Expression t = input;
  if (bias) t = t + *bias;
  t = ActivationExpr<DataType>(act, t);
  if (alpha != 1.0f) t = t * alpha;
  if (skip) t = t + *skip;

  const uint32_t ln_axes[] = {3};
  dml::Expression mean =
      dml::Reduce(t, DML_REDUCE_FUNCTION_AVERAGE,
                  dml::Span<const uint32_t>(ln_axes, 1));
  const auto sizes = t.Impl()->GetOutputDesc().sizes;
  dml::Expression mean_b =
      GraphFactory<DataType>::ReinterpretView(mean, sizes, {0, 0, 1, 0});
  dml::Expression centered = t - mean_b;
  dml::Expression var =
      dml::Reduce(centered * centered, DML_REDUCE_FUNCTION_AVERAGE,
                  dml::Span<const uint32_t>(ln_axes, 1));
  dml::Expression inv_b = GraphFactory<DataType>::ReinterpretView(
      dml::Recip(dml::Sqrt(var + eps)), sizes, {0, 0, 1, 0});
  return centered * inv_b * gammas + betas;
}

// KDA output RMS norm: mixed * gammas / sqrt(mean(mixed^2) + eps), the
// applyKdaOutputGate-preceding norm from sycl EvalKda (which normalizes
// before gating on purpose -- they do not commute).
template <typename DataType>
dml::Expression RmsNormExpr(dml::Expression mixed, dml::Expression& gammas,
                            float eps) {
  const uint32_t rms_axes[] = {3};
  dml::Expression mean_sq =
      dml::Reduce(mixed * mixed, DML_REDUCE_FUNCTION_AVERAGE,
                  dml::Span<const uint32_t>(rms_axes, 1));
  const auto sizes = mixed.Impl()->GetOutputDesc().sizes;
  dml::Expression inv_b = GraphFactory<DataType>::ReinterpretView(
      dml::Recip(dml::Sqrt(mean_sq + eps)), sizes, {0, 0, 1, 0});
  return mixed * inv_b * gammas;
}

// Dispatches a compiled op with the layer's activation pointers substituted
// into the recorded binding order. Mirrors calling a CUDA kernel launcher
// with (output, input, input2, scratch).
template <typename DataType>
void DispatchOp(DmlExecScope& scope, DmlCompiledOp& op, DmlPtr input,
                DmlPtr input2, DmlPtr scratch, std::vector<DmlPtr> outputs,
                DmlPtr extra = {}) {
  std::vector<DmlPtr> ptrs;
  ptrs.reserve(op.bindings.size());
  for (const auto& b : op.bindings) {
    switch (b.kind) {
      case DmlBindingRef::Kind::kWeight:
        ptrs.push_back(b.weight);
        break;
      case DmlBindingRef::Kind::kInput:
        ptrs.push_back(input);
        break;
      case DmlBindingRef::Kind::kInput2:
        ptrs.push_back(input2);
        break;
      case DmlBindingRef::Kind::kScratch:
        ptrs.push_back(scratch);
        break;
      case DmlBindingRef::Kind::kExtra:
        ptrs.push_back(extra);
        break;
    }
  }
  DmlPtr transient;
  if (op.transient_bytes) transient = scope.TakeTransient(op.transient_bytes);
  scope.ctx().DispatchOperator(scope.list(), op.op.Get(), ptrs, op.bindings,
                               outputs, op.output_bytes, transient,
                               op.transient_bytes);
}

}  // namespace

// ===========================================================================
// DmlDeviceContext / DmlHalf / DmlWeightUploader plumbing
// ===========================================================================

DmlHalf::DmlHalf(float f) : bits(F32toF16Bits(f)) {}
DmlHalf::operator float() const { return F16BitsToF32(bits); }

// Implemented in network_directml.cc (device bring-up lives there, like the
// CUDA backend's device discovery in network_cuda.cc).
IDMLBindingTable* DmlDeviceContext::GetOrCreateBindingTable(
    IDMLCompiledOperator* op) {
  auto it = tables_.find(op);
  if (it != tables_.end()) return it->second.Get();

  // The table must cover the dispatchable's RequiredDescriptorCount, which
  // includes DML-internal intermediates (e.g. the KDA projection graph
  // needs 105). A smaller table than required fails E_INVALIDARG, and this
  // driver reports some undersized-table failures with the misleading
  // DXGI_ERROR_DEVICE_REMOVED code instead.
  const uint32_t slots =
      std::max<uint32_t>(op->GetBindingProperties().RequiredDescriptorCount,
                         1u);
  D3D12_CPU_DESCRIPTOR_HANDLE cpu = descriptors_.TakeCpu(slots);
  DML_BINDING_TABLE_DESC table_desc = {};
  table_desc.Dispatchable = op;
  table_desc.CPUDescriptorHandle = cpu;
  table_desc.GPUDescriptorHandle = descriptors_.CpuToGpu(cpu);
  table_desc.SizeInDescriptors = slots;
  ComPtr<IDMLBindingTable> table;
  ReportDmlErrors(
      dml_device_->CreateBindingTable(&table_desc, IID_PPV_ARGS(&table)),
      "CreateBindingTable");
  IDMLBindingTable* raw = table.Get();
  tables_.emplace(op, std::move(table));
  return raw;
}

void DmlDeviceContext::DispatchOperator(
    ID3D12GraphicsCommandList* list, IDMLCompiledOperator* op,
    const std::vector<DmlPtr>& inputs, const std::vector<DmlBindingRef>& meta,
    const std::vector<DmlPtr>& outputs,
    const std::vector<uint64_t>& output_bytes, DmlPtr transient,
    uint64_t transient_bytes) {
  IDMLBindingTable* table = GetOrCreateBindingTable(op);  // created at build

  std::vector<DML_BUFFER_BINDING> buffer_bindings(inputs.size());
  std::vector<DML_BINDING_DESC> binding_descs(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    buffer_bindings[i].Buffer = inputs[i].res;
    buffer_bindings[i].Offset = inputs[i].offset;
    buffer_bindings[i].SizeInBytes = meta[i].bytes;
    binding_descs[i].Type = DML_BINDING_TYPE_BUFFER;
    binding_descs[i].Desc = &buffer_bindings[i];
  }
  if (!binding_descs.empty()) {
    table->BindInputs(static_cast<UINT>(binding_descs.size()),
                      binding_descs.data());
  }

  if (transient_bytes) {
    DML_BUFFER_BINDING tb{transient.res, transient.offset, transient_bytes};
    DML_BINDING_DESC td{DML_BINDING_TYPE_BUFFER, &tb};
    table->BindTemporaryResource(&td);
  }

  std::vector<DML_BUFFER_BINDING> out_bindings(outputs.size());
  std::vector<DML_BINDING_DESC> out_descs(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    out_bindings[i].Buffer = outputs[i].res;
    out_bindings[i].Offset = outputs[i].offset;
    out_bindings[i].SizeInBytes = output_bytes[i];
    out_descs[i].Type = DML_BINDING_TYPE_BUFFER;
    out_descs[i].Desc = &out_bindings[i];
  }
  table->BindOutputs(static_cast<UINT>(out_descs.size()), out_descs.data());

  ID3D12DescriptorHeap* heap = descriptors_.heap();
  list->SetDescriptorHeaps(1, &heap);
  recorder_->RecordDispatch(list, op, table);

  // A global UAV barrier between dispatches: layers are strictly sequential
  // so this is correct, if conservative.
  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  list->ResourceBarrier(1, &barrier);
}

void DmlDeviceContext::WaitForFence(ID3D12Fence* fence, uint64_t value) {
  if (fence->GetCompletedValue() >= value) return;
  ReportD3DErrors(fence->SetEventOnCompletion(value, fence_event_),
                  "SetEventOnCompletion");
  WaitForSingleObject(fence_event_, INFINITE);
}

// ===========================================================================
// KdaRecurrenceLayer -- unchanged from the original stub, except Record()
// now takes byte offsets (DmlPtr) so it can bind sub-allocations of the
// shared arenas.
// ===========================================================================

namespace {

// Layout must match kda_recurrence.hlsl's KdaRecurrenceConstants cbuffer
// field-for-field: same order, same types, no gaps. Root constants are
// copied as raw 32-bit words, so any mismatch here silently scrambles the
// shader's inputs rather than failing to compile.
struct KdaShaderConstants {
  uint32_t N;
  uint32_t heads;
  uint32_t key_dim;
  uint32_t value_dim;
  uint32_t direction_count;
  uint32_t use_fused_qkv;
  uint32_t qkv_stride;
  float log_decay_floor;
  int32_t directions[16];
};
static_assert(sizeof(KdaShaderConstants) == 96,
              "KdaShaderConstants must match the HLSL cbuffer's 96-byte "
              "layout (24 x 32-bit words) -- see kda_recurrence.hlsl");

constexpr UINT kKdaNum32BitConstants = sizeof(KdaShaderConstants) / 4;
constexpr UINT kKdaRootParamConstants = 0;
constexpr UINT kKdaRootParamSrvBase = 1;
constexpr UINT kKdaRootParamUav = 9;

struct PreprocessConstants {
  uint32_t mode;
  uint32_t input_size;
  uint32_t encoding_size;
  uint32_t total_channels;
  uint32_t enc_batch_stride;
  uint32_t pad0;
  uint32_t pad1;
  uint32_t pad2;
};
static_assert(sizeof(PreprocessConstants) % 4 == 0);

struct PolicyFinalizeConstants {
  uint32_t key_width;
  uint32_t pad0;
  uint32_t pad1;
  uint32_t pad2;
};

struct TransposeConstants {
  uint32_t mode;
  uint32_t batch_size;
  uint32_t heads;
  uint32_t head_dim;
};
static_assert(sizeof(TransposeConstants) % 4 == 0);

}  // namespace

KdaRecurrenceLayer::KdaRecurrenceLayer(ID3D12Device* device, bool fp16,
                                       uint32_t key_dim, uint32_t value_dim)
    : device_(device), fp16_(fp16), key_dim_(key_dim), value_dim_(value_dim) {
  if (key_dim_ == 0 || value_dim_ == 0) {
    throw Exception(
        "directml backend: KDA key_dim/value_dim must both be non-zero.");
  }
  // D3D12 caps a thread group at 1024 threads, and one lane per value
  // dimension is the shader's whole structure.
  if (value_dim_ > 1024) {
    throw Exception(
        "directml backend: KDA value_dim exceeds the 1024-thread group "
        "limit.");
  }
  D3D12_ROOT_PARAMETER params[10] = {};

  params[kKdaRootParamConstants].ParameterType =
      D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  params[kKdaRootParamConstants].Constants.ShaderRegister = 0;
  params[kKdaRootParamConstants].Constants.RegisterSpace = 0;
  params[kKdaRootParamConstants].Constants.Num32BitValues =
      kKdaNum32BitConstants;
  params[kKdaRootParamConstants].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

  for (UINT i = 0; i < 8; ++i) {
    auto& p = params[kKdaRootParamSrvBase + i];
    p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    p.Descriptor.ShaderRegister = i;
    p.Descriptor.RegisterSpace = 0;
    p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }

  params[kKdaRootParamUav].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
  params[kKdaRootParamUav].Descriptor.ShaderRegister = 0;
  params[kKdaRootParamUav].Descriptor.RegisterSpace = 0;
  params[kKdaRootParamUav].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

  D3D12_ROOT_SIGNATURE_DESC root_desc = {};
  root_desc.NumParameters = 10;
  root_desc.pParameters = params;

  ComPtr<ID3DBlob> signature_blob;
  ComPtr<ID3DBlob> error_blob;
  HRESULT hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &signature_blob, &error_blob);
  if (FAILED(hr)) {
    std::string message = "Failed to serialize KdaRecurrence root signature: ";
    if (error_blob) {
      message.append(static_cast<const char*>(error_blob->GetBufferPointer()),
                     error_blob->GetBufferSize());
    }
    throw Exception(message);
  }
  ReportD3DErrors(device_->CreateRootSignature(0, signature_blob->GetBufferPointer(),
                                               signature_blob->GetBufferSize(),
                                               IID_PPV_ARGS(&root_signature_)),
                  "CreateRootSignature (kda)");

  // Specialised, not parameterised: with key_dim only known at runtime the
  // scan's four inner loops cannot be unrolled and `state` spills to scratch
  // memory. See the header comment in shaders/kda_recurrence.hlsl.
  ComPtr<ID3DBlob> shader_blob = CompileHlsl(
      kKdaRecurrenceShaderSource, sizeof(kKdaRecurrenceShaderSource) - 1,
      "kda_recurrence.hlsl", "KdaRecurrence", fp16_,
      {{"KDA_KEY_DIM", std::to_string(key_dim_)},
       {"KDA_VALUE_DIM", std::to_string(value_dim_)}});
  pso_ = CreateComputePso(device_.Get(), root_signature_.Get(),
                         shader_blob.Get());
}

void KdaRecurrenceLayer::Record(ID3D12GraphicsCommandList* command_list,
                                const Params& params, DmlPtr qkv, DmlPtr q,
                                DmlPtr k, DmlPtr v, DmlPtr raw_decay,
                                DmlPtr dt_bias, DmlPtr a_log, DmlPtr beta,
                                DmlPtr mixed_out) {
  const bool fused = params.use_fused_qkv;
  assert(fused ? !!qkv : (!!q && !!k && !!v));
  if (params.key_dim != key_dim_ || params.value_dim != value_dim_) {
    // The PSO bakes the geometry in, so a mismatch would silently read and
    // write the wrong slices rather than fail.
    throw Exception(
        "directml backend: KDA recurrence dispatched with a geometry other "
        "than the one its shader was compiled for.");
  }

  KdaShaderConstants constants{};
  constants.N = params.batch_size;
  constants.heads = params.heads;
  constants.key_dim = params.key_dim;
  constants.value_dim = params.value_dim;
  constants.direction_count = params.direction_count;
  constants.use_fused_qkv = fused ? 1u : 0u;
  constants.qkv_stride = params.qkv_stride;
  constants.log_decay_floor = params.log_decay_floor;
  std::memcpy(constants.directions, params.directions.data(),
              sizeof(constants.directions));

  // Root SRV/UAV bind by GPU virtual address; arena sub-allocations add
  // their byte offset to the resource base.
  const DmlPtr q_slot = fused ? qkv : q;
  const DmlPtr k_slot = fused ? qkv : k;
  const DmlPtr v_slot = fused ? qkv : v;
  DmlPtr slots[8] = {q_slot, q_slot, k_slot, v_slot,
                     raw_decay, dt_bias, a_log, beta};

  command_list->SetComputeRootSignature(root_signature_.Get());
  command_list->SetPipelineState(pso_.Get());
  command_list->SetComputeRoot32BitConstants(kKdaRootParamConstants,
                                             kKdaNum32BitConstants, &constants,
                                             0);
  for (UINT i = 0; i < 8; ++i) {
    command_list->SetComputeRootShaderResourceView(
        kKdaRootParamSrvBase + i, slots[i].GpuVA());
  }
  command_list->SetComputeRootUnorderedAccessView(kKdaRootParamUav,
                                                  mixed_out.GpuVA());

  const UINT group_count = params.batch_size * params.heads;
  command_list->Dispatch(group_count, 1, 1);

  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  command_list->ResourceBarrier(1, &barrier);
}

MhaTransposeLayer::MhaTransposeLayer(ID3D12Device* device, bool fp16)
    : device_(device), fp16_(fp16) {
  ComPtr<ID3DBlob> shader =
      CompileHlsl(kMhaTransposeShaderSource,
                  sizeof(kMhaTransposeShaderSource) - 1,
                  "mha_transpose.hlsl", "MhaTranspose", fp16_);
  root_signature_ = CreateShaderRootSignature(
      device_.Get(), sizeof(TransposeConstants) / 4, 1, 1);
  pso_ = CreateComputePso(device_.Get(), root_signature_.Get(), shader.Get());
}

void MhaTransposeLayer::Record(ID3D12GraphicsCommandList* command_list,
                               const Params& params, DmlPtr input,
                               DmlPtr output) {
  TransposeConstants constants{};
  constants.mode = params.mode;
  constants.batch_size = params.batch_size;
  constants.heads = params.heads;
  constants.head_dim = params.head_dim;

  command_list->SetComputeRootSignature(root_signature_.Get());
  command_list->SetPipelineState(pso_.Get());
  command_list->SetComputeRoot32BitConstants(0, sizeof(TransposeConstants) / 4,
                                             &constants, 0);
  command_list->SetComputeRootShaderResourceView(1, input.GpuVA());
  command_list->SetComputeRootUnorderedAccessView(2, output.GpuVA());
  const uint64_t total =
      (uint64_t)params.batch_size * params.heads * 64 * params.head_dim;
  command_list->Dispatch(static_cast<UINT>((total + 63) / 64), 1, 1);

  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  command_list->ResourceBarrier(1, &barrier);
}

struct SmolgenBiasConstants {
  uint32_t batch;
  uint32_t heads;
  uint32_t gen;
  uint32_t pad0;
};

SmolgenBiasLayer::SmolgenBiasLayer(ID3D12Device* device, bool fp16)
    : device_(device), fp16_(fp16) {
  ComPtr<ID3DBlob> shader =
      CompileHlsl(kSmolgenBiasShaderSource, sizeof(kSmolgenBiasShaderSource) - 1,
                  "smolgen_bias.hlsl", "SmolgenBias", fp16_);
  root_signature_ = CreateShaderRootSignature(
      device_.Get(), sizeof(SmolgenBiasConstants) / 4, 2, 1);
  pso_ = CreateComputePso(device_.Get(), root_signature_.Get(), shader.Get());
}

void SmolgenBiasLayer::Record(ID3D12GraphicsCommandList* command_list,
                              const Params& params, DmlPtr table, DmlPtr d2,
                              DmlPtr bias_out) {
  SmolgenBiasConstants constants{};
  constants.batch = params.batch;
  constants.heads = params.heads;
  constants.gen = params.gen;

  command_list->SetComputeRootSignature(root_signature_.Get());
  command_list->SetPipelineState(pso_.Get());
  command_list->SetComputeRoot32BitConstants(0,
                                             sizeof(SmolgenBiasConstants) / 4,
                                             &constants, 0);
  command_list->SetComputeRootShaderResourceView(1, table.GpuVA());
  command_list->SetComputeRootShaderResourceView(2, d2.GpuVA());
  command_list->SetComputeRootUnorderedAccessView(3, bias_out.GpuVA());
  const uint64_t total =
      (uint64_t)params.batch * params.heads * 4096u;
  command_list->Dispatch(static_cast<UINT>((total + 63) / 64), 1, 1);

  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  command_list->ResourceBarrier(1, &barrier);
}

struct LayerNormConstants {
  uint32_t rows;
  uint32_t channels;
  uint32_t flags;
  uint32_t activation;
  float alpha;
  float eps;
  uint32_t pad0;
  uint32_t pad1;
};

LayerNormLayer::LayerNormLayer(ID3D12Device* device, bool fp16)
    : device_(device), fp16_(fp16) {
  ComPtr<ID3DBlob> shader =
      CompileHlsl(kLayerNormShaderSource, sizeof(kLayerNormShaderSource) - 1,
                  "layer_norm.hlsl", "LayerNorm", fp16_);
  root_signature_ = CreateShaderRootSignature(
      device_.Get(), sizeof(LayerNormConstants) / 4, 5, 1);
  pso_ = CreateComputePso(device_.Get(), root_signature_.Get(), shader.Get());
}

void LayerNormLayer::Record(ID3D12GraphicsCommandList* command_list,
                            const Params& params, DmlPtr input, DmlPtr bias,
                            DmlPtr skip, DmlPtr gammas, DmlPtr betas,
                            DmlPtr output) {
  LayerNormConstants constants{};
  constants.rows = params.rows;
  constants.channels = params.channels;
  constants.flags = (params.has_bias ? 1u : 0u) | (params.has_skip ? 2u : 0u);
  constants.activation = static_cast<uint32_t>(params.act);
  constants.alpha = params.alpha;
  constants.eps = params.eps;

  // Every root SRV must point at a real allocation even when the shader
  // never reads it: a zero GPU VA in a root descriptor removes the device on
  // this driver (the same failure the PE_DENSE preprocess kernel hit).
  const DmlPtr bias_srv = params.has_bias ? bias : input;
  const DmlPtr skip_srv = params.has_skip ? skip : input;

  command_list->SetComputeRootSignature(root_signature_.Get());
  command_list->SetPipelineState(pso_.Get());
  command_list->SetComputeRoot32BitConstants(0, sizeof(LayerNormConstants) / 4,
                                             &constants, 0);
  command_list->SetComputeRootShaderResourceView(1, input.GpuVA());
  command_list->SetComputeRootShaderResourceView(2, bias_srv.GpuVA());
  command_list->SetComputeRootShaderResourceView(3, skip_srv.GpuVA());
  command_list->SetComputeRootShaderResourceView(4, gammas.GpuVA());
  command_list->SetComputeRootShaderResourceView(5, betas.GpuVA());
  command_list->SetComputeRootUnorderedAccessView(6, output.GpuVA());
  // One thread group per token row (LN_GROUP_SIZE threads cooperate on the
  // channel reduction), so the group count is the row count.
  command_list->Dispatch(static_cast<UINT>(params.rows), 1, 1);

  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  command_list->ResourceBarrier(1, &barrier);
}

struct KdaLocalConvConstants {
  uint32_t tokens;
  uint32_t emb;
  uint32_t pad0;
  uint32_t pad1;
};

KdaLocalConvLayer::KdaLocalConvLayer(ID3D12Device* device, bool fp16)
    : device_(device), fp16_(fp16) {
  ComPtr<ID3DBlob> shader =
      CompileHlsl(kKdaLocalConvShaderSource,
                  sizeof(kKdaLocalConvShaderSource) - 1, "kda_local_conv.hlsl",
                  "KdaLocalConv", fp16_);
  root_signature_ = CreateShaderRootSignature(
      device_.Get(), sizeof(KdaLocalConvConstants) / 4, 3, 1);
  pso_ = CreateComputePso(device_.Get(), root_signature_.Get(), shader.Get());
}

void KdaLocalConvLayer::Record(ID3D12GraphicsCommandList* command_list,
                               const Params& params, DmlPtr input,
                               DmlPtr weights, DmlPtr bias, DmlPtr output) {
  KdaLocalConvConstants constants{};
  constants.tokens = params.tokens;
  constants.emb = params.emb;

  command_list->SetComputeRootSignature(root_signature_.Get());
  command_list->SetPipelineState(pso_.Get());
  command_list->SetComputeRoot32BitConstants(0,
                                             sizeof(KdaLocalConvConstants) / 4,
                                             &constants, 0);
  command_list->SetComputeRootShaderResourceView(1, input.GpuVA());
  command_list->SetComputeRootShaderResourceView(2, weights.GpuVA());
  command_list->SetComputeRootShaderResourceView(3, bias.GpuVA());
  command_list->SetComputeRootUnorderedAccessView(4, output.GpuVA());
  const uint64_t total = (uint64_t)params.tokens * params.emb;
  command_list->Dispatch(static_cast<UINT>((total + 63) / 64), 1, 1);

  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  command_list->ResourceBarrier(1, &barrier);
}

// ===========================================================================
// FCLayer / EmbeddingLayer / PolicyMapLayer
// ===========================================================================

template <typename DataType>
static DmlCompiledOp BuildGemmLayerOp(
    DmlDeviceContext& ctx, uint32_t N, uint32_t tokens, uint32_t num_inputs,
    uint32_t num_outputs, bool use_bias, ActivationFunction act,
    const DmlPtr& weights, const DmlPtr& biases) {
  GraphFactory<DataType> g(ctx);
  auto x = g.Input({1, 1, tokens, num_inputs});
  auto w = g.Weight(weights, {1, 1, num_outputs, num_inputs});
  dml::Expression y = dml::Gemm(x, w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                                DML_MATRIX_TRANSFORM_TRANSPOSE);
  if (use_bias) {
    auto b = g.WeightChannel(biases, tokens, num_outputs);
    y = y + b;
  }
  y = ActivationExpr<DataType>(act, y);
  return g.Compile({y}, {(uint64_t)tokens * num_outputs * sizeof(DataType)});
}

template <typename DataType>
FCLayer<DataType>::FCLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           bool bias, ActivationFunction activation,
                           const std::vector<float>& weights,
                           const std::vector<float>& biases,
                           DmlWeightUploader& uploader)
    : BaseLayer<DataType>(C, H, W, ip),
      use_bias_(bias),
      act_(activation) {
  weights_ = uploader.Add(weights);
  if (use_bias_) biases_ = uploader.Add(biases);
}

template <typename DataType>
void FCLayer<DataType>::EnsureCompiled(int N, DmlExecScope& scope) {
  if (compiled_.count(N)) return;
  const int num_inputs = this->input_->GetC() * this->input_->GetH() *
                         this->input_->GetW();
  compiled_.emplace(N, BuildGemmLayerOp<DataType>(
                           scope.ctx(), N, N, num_inputs, C, use_bias_, act_,
                           weights_, biases_));
}

template <typename DataType>
void FCLayer<DataType>::Eval(int N, DmlPtr output, DmlPtr input, DmlPtr input2,
                             DmlPtr scratch, size_t scratch_size,
                             DmlExecScope& scope) {
  if (!compiled_.count(N)) EnsureCompiled(N, scope);
  DispatchOp<DataType>(scope, compiled_.find(N)->second, input, input2, scratch,
                       {output});
}

template <typename DataType>
EmbeddingLayer<DataType>::EmbeddingLayer(BaseLayer<DataType>* ip, int C, int H,
                                         int W, bool bias,
                                         ActivationFunction activation,
                                         const std::vector<float>& weights,
                                         const std::vector<float>& biases,
                                         DmlWeightUploader& uploader)
    : BaseLayer<DataType>(C, H, W, ip),
      use_bias_(bias),
      act_(activation) {
  weights_ = uploader.Add(weights);
  if (use_bias_) biases_ = uploader.Add(biases);
}

template <typename DataType>
void EmbeddingLayer<DataType>::EnsureCompiled(int N, DmlExecScope& scope) {
  if (compiled_.count(N)) return;
  // Token-level GEMM: [N*64, K] -> [N*64, M], no H/W flattening.
  const int num_inputs = this->input_->GetC();
  const uint32_t tokens = N * 64;
  compiled_.emplace(N, BuildGemmLayerOp<DataType>(
                           scope.ctx(), N, tokens, num_inputs, C, use_bias_,
                           act_, weights_, biases_));
}

template <typename DataType>
void EmbeddingLayer<DataType>::Eval(int N, DmlPtr output, DmlPtr input,
                                    DmlPtr input2, DmlPtr scratch,
                                    size_t scratch_size, DmlExecScope& scope) {
  if (!compiled_.count(N)) EnsureCompiled(N, scope);
  DispatchOp<DataType>(scope, compiled_.find(N)->second, input, input2, scratch,
                       {output});
}

template <typename DataType>
PolicyMapLayer<DataType>::PolicyMapLayer(BaseLayer<DataType>* ip, int usedSize,
                                         bool attention, const short* cpuWeight,
                                         DmlWeightUploader& uploader)
    : BaseLayer<DataType>(kNumOutputPolicy, 1, 1, ip), used_size_(usedSize) {
  // The CUDA/BLAS PolicyMap is a scatter: policy[map[i]] = attn[i] for each
  // attention row i in [0, usedSize). DirectML Gather is the inverse
  // direction (out[j] = in[idx[j]]), so build the inverse table: for each
  // policy index j, the attention row i with map[i] == j. The table is
  // indexed by attention row (length usedSize == 4288), NOT by policy
  // index -- reading only the first 1858 entries drops the promotion
  // region entirely.
  indices_host_.assign(kNumOutputPolicy, 0);
  int covered = 0;
  for (int i = 0; i < usedSize; ++i) {
    const int j = cpuWeight[i];
    if (j >= 0 && j < kNumOutputPolicy) {
      indices_host_[static_cast<size_t>(j)] = static_cast<uint32_t>(i);
      ++covered;
    }
  }
  if (covered != kNumOutputPolicy) {
    CERR << "directml PolicyMap: inverse map covers " << covered << "/"
         << kNumOutputPolicy << " policy indices (map may be incomplete).";
  }
  indices_ = uploader.AddRaw(indices_host_.data(),
                             indices_host_.size() * sizeof(uint32_t));
}

template <typename DataType>
void PolicyMapLayer<DataType>::EnsureCompiled(int N, DmlExecScope& scope) {
  if (compiled_.count(N)) return;
  GraphFactory<DataType> g(scope.ctx());
  auto x = g.Input({1, 1, static_cast<uint32_t>(N),
                    static_cast<uint32_t>(used_size_)});
  auto indices = g.WeightU32(
      indices_, {1, 1, 1, static_cast<uint32_t>(kNumOutputPolicy)});
  dml::Expression y = dml::Gather(x, indices, 3, 1);
  compiled_.emplace(
      N, g.Compile({y}, {(uint64_t)N * kNumOutputPolicy * sizeof(DataType)}));
}

template <typename DataType>
void PolicyMapLayer<DataType>::Eval(int N, DmlPtr output, DmlPtr input,
                                     DmlPtr input2, DmlPtr scratch,
                                     size_t scratch_size, DmlExecScope& scope) {
  if (!compiled_.count(N)) EnsureCompiled(N, scope);
  DispatchOp<DataType>(scope, compiled_.find(N)->second, input, input2, scratch,
                       {output});
}

// ===========================================================================
// AttentionBody
// ===========================================================================

template <typename DataType>
AttentionBody<DataType>::AttentionBody(
    const MultiHeadWeights& weights, DmlWeightUploader& uploader,
    Activations activations, int num_res_blocks, int input_c,
    int max_batch_size, bool is_pe_dense_embedding,
    const std::vector<int>& kda_directions, DmlDeviceContext& ctx, bool fp16)
    : BaseLayer<DataType>(static_cast<int>(weights.ip_emb_b.size()), 8, 8,
                          nullptr),
      embedding_op_size_(static_cast<int>(weights.ip_emb_b.size())),
      input_c_(input_c),
      is_pe_dense_embedding_(is_pe_dense_embedding),
      has_gating_(!weights.ip_mult_gate.empty() && !weights.ip_add_gate.empty()),
      activations_(activations),
      num_resi_blocks_(num_res_blocks) {
  if (num_resi_blocks_ > 0) {
    throw Exception(
        "The directml backend does not support attention nets with residual "
        "conv blocks yet (embedding -> encoder stacks only).");
  }

  ip_emb_w_ = uploader.Add(weights.ip_emb_w);
  ip_emb_b_ = uploader.Add(weights.ip_emb_b);

  if (is_pe_dense_embedding_) {
    ip_emb_pre_w_ = uploader.Add(weights.ip_emb_preproc_w);
    ip_emb_pre_b_ = uploader.Add(weights.ip_emb_preproc_b);
    ip_emb_ln_g_ = uploader.Add(weights.ip_emb_ln_gammas);
    ip_emb_ln_b_ = uploader.Add(weights.ip_emb_ln_betas);
    ip_emb_ffn_d1_w_ = uploader.Add(weights.ip_emb_ffn.dense1_w);
    ip_emb_ffn_d1_b_ = uploader.Add(weights.ip_emb_ffn.dense1_b);
    ip_emb_ffn_d2_w_ = uploader.Add(weights.ip_emb_ffn.dense2_w);
    ip_emb_ffn_d2_b_ = uploader.Add(weights.ip_emb_ffn.dense2_b);
    ip_emb_ffn_ln_g_ = uploader.Add(weights.ip_emb_ffn_ln_gammas);
    ip_emb_ffn_ln_b_ = uploader.Add(weights.ip_emb_ffn_ln_betas);
    embedding_dense_size_ =
        static_cast<int>(weights.ip_emb_preproc_b.size()) / 64;
    embedding_ffn_size_ = static_cast<int>(weights.ip_emb_ffn.dense2_b.size());
    embedding_ffn_dff_ = static_cast<int>(weights.ip_emb_ffn.dense1_b.size());
  } else {
    pos_encoding_host_.assign(&kPosEncoding[0][0],
                              &kPosEncoding[0][0] + 64 * kNumPosEncodingChannels);
    pos_encoding_ = uploader.Add(pos_encoding_host_);
    embedding_dense_size_ = 0;
  }

  if (has_gating_) {
    ip_mult_gate_ = uploader.Add(weights.ip_mult_gate);
    ip_add_gate_ = uploader.Add(weights.ip_add_gate);
  }

  // The preprocess compute shader: one PSO, compiled once.
  ComPtr<ID3DBlob> shader =
      CompileHlsl(kAttentionPreprocessShaderSource,
                  sizeof(kAttentionPreprocessShaderSource) - 1,
                  "attention_preprocess.hlsl", "AttentionPreprocess", fp16);
  preprocess_root_signature_ =
      CreateShaderRootSignature(ctx.device(), sizeof(PreprocessConstants) / 4,
                                2, 1);
  preprocess_pso_ =
      CreateComputePso(ctx.device(), preprocess_root_signature_.Get(),
                       shader.Get());

  const int num_encoders = static_cast<int>(weights.encoder.size());
  const float alpha = static_cast<float>(pow(2.0 * num_encoders, -0.25));
  DmlPtr smolgen_global;
  int smolgen_global_size = 0;
  if (weights.has_smolgen) {
    smolgen_global = uploader.Add(weights.smolgen_w);
    smolgen_global_size = 64 * 64;
  }
  for (const auto& enc : weights.encoder) {
    encoder_weights_.emplace_back(new EncoderBlock<DataType>(
        enc, uploader, weights.encoder_head_count, embedding_op_size_, alpha,
        smolgen_global, smolgen_global_size, max_batch_size,
        activations.smolgen_activation, activations.ffn_activation,
        is_pe_dense_embedding_ ? 1e-3f : 1e-6f, kda_directions, ctx, fp16));
  }
}

template <typename DataType>
void AttentionBody<DataType>::EnsureCompiled(int N, DmlExecScope& scope) {
  const uint32_t tokens = N * 64;
  const int input_size = kNumInputPlanes;
  const int body_input_c = is_pe_dense_embedding_
                               ? input_size + embedding_dense_size_
                               : input_size + kNumPosEncodingChannels;
  if (is_pe_dense_embedding_) {
    if (!pre_compiled_.count(N)) {
      GraphFactory<DataType> g(scope.ctx());
      auto x = g.Input({1, 1, static_cast<uint32_t>(N), 64 * 12});
      auto w = g.Weight(ip_emb_pre_w_,
                        {1, 1, static_cast<uint32_t>(64 * embedding_dense_size_),
                         64 * 12});
      auto b = g.WeightChannel(ip_emb_pre_b_, static_cast<uint32_t>(N),
                               static_cast<uint32_t>(64 * embedding_dense_size_));
      dml::Expression y = dml::Gemm(x, w, dml::NullOpt,
                                    DML_MATRIX_TRANSFORM_NONE,
                                    DML_MATRIX_TRANSFORM_TRANSPOSE) + b;
      pre_compiled_.emplace(
          N, g.Compile({y}, {(uint64_t)N * 64 * embedding_dense_size_ *
                                 sizeof(DataType)}));
    }
    if (!compiled_.count(N)) {
      GraphFactory<DataType> g(scope.ctx());
      auto x = g.Input({1, 1, tokens, static_cast<uint32_t>(body_input_c)});
      auto w = g.Weight(ip_emb_w_, {1, 1, static_cast<uint32_t>(embedding_op_size_),
                                    static_cast<uint32_t>(body_input_c)});
      auto b = g.WeightChannel(ip_emb_b_, tokens, static_cast<uint32_t>(embedding_op_size_));
      auto ln_g = g.WeightChannel(ip_emb_ln_g_, tokens, static_cast<uint32_t>(embedding_op_size_));
      auto ln_b = g.WeightChannel(ip_emb_ln_b_, tokens, static_cast<uint32_t>(embedding_op_size_));
      dml::Expression emb =
          dml::Gemm(x, w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                    DML_MATRIX_TRANSFORM_TRANSPOSE);
      dml::Expression bias_b = b;
      // LayerNorm with the previous gemm's bias fused: activation is the
      // net's default (RELU/MISH), matching the SYCL embedding LN.
      dml::Expression pre_ln = LayerNormExpr<DataType>(emb, &bias_b, nullptr, ln_g, ln_b, 1.0f, 1e-3f,
          activations_.default_activation);
      if (has_gating_) {
        auto mult = g.WeightChannel(ip_mult_gate_, tokens, static_cast<uint32_t>(embedding_op_size_));
        auto add = g.WeightChannel(ip_add_gate_, tokens, static_cast<uint32_t>(embedding_op_size_));
        pre_ln = pre_ln * mult + add;
      }
      auto ffn1_w = g.Weight(
          ip_emb_ffn_d1_w_,
          {1, 1, static_cast<uint32_t>(embedding_ffn_dff_),
           static_cast<uint32_t>(embedding_op_size_)});
      auto ffn1_b = g.WeightChannel(ip_emb_ffn_d1_b_, tokens, static_cast<uint32_t>(embedding_ffn_dff_));
      auto ffn2_w = g.Weight(
          ip_emb_ffn_d2_w_,
          {1, 1, static_cast<uint32_t>(embedding_op_size_),
           static_cast<uint32_t>(embedding_ffn_dff_)});
      auto ffn2_b = g.WeightChannel(ip_emb_ffn_d2_b_, tokens, static_cast<uint32_t>(embedding_op_size_));
      auto ffn_ln_g = g.WeightChannel(ip_emb_ffn_ln_g_, tokens, static_cast<uint32_t>(embedding_op_size_));
      auto ffn_ln_b = g.WeightChannel(ip_emb_ffn_ln_b_, tokens, static_cast<uint32_t>(embedding_op_size_));
      dml::Expression ffn1 =
          dml::Gemm(pre_ln, ffn1_w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                    DML_MATRIX_TRANSFORM_TRANSPOSE) + ffn1_b;
      ffn1 = ActivationExpr<DataType>(activations_.ffn_activation, ffn1);
      dml::Expression ffn2 =
          dml::Gemm(ffn1, ffn2_w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                    DML_MATRIX_TRANSFORM_TRANSPOSE);
      const float alpha =
          static_cast<float>(pow(2.0 * encoder_weights_.size(), -0.25));
      dml::Expression out = LayerNormExpr<DataType>(ffn2, &ffn2_b, &pre_ln, ffn_ln_g, ffn_ln_b, alpha, 1e-3f,
          ACTIVATION_NONE);
      compiled_.emplace(
          N, g.Compile({out}, {(uint64_t)tokens * embedding_op_size_ *
                                   sizeof(DataType)}));
    }
  } else {
    if (!compiled_.count(N)) {
      GraphFactory<DataType> g(scope.ctx());
      auto x = g.Input({1, 1, tokens, static_cast<uint32_t>(body_input_c)});
      auto w = g.Weight(ip_emb_w_, {1, 1, static_cast<uint32_t>(embedding_op_size_),
                                    static_cast<uint32_t>(body_input_c)});
      auto b = g.WeightChannel(ip_emb_b_, tokens, static_cast<uint32_t>(embedding_op_size_));
      dml::Expression g0 =
          dml::Gemm(x, w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                    DML_MATRIX_TRANSFORM_TRANSPOSE);
      dml::Expression y = g0 + b;
      y = ActivationExpr<DataType>(activations_.default_activation, y);
      if (has_gating_) {
        auto mult = g.WeightChannel(ip_mult_gate_, tokens, static_cast<uint32_t>(embedding_op_size_));
        auto add = g.WeightChannel(ip_add_gate_, tokens, static_cast<uint32_t>(embedding_op_size_));
        y = y * mult + add;
      }
      compiled_.emplace(
          N, g.Compile({y}, {(uint64_t)tokens * embedding_op_size_ *
                                 sizeof(DataType)}));
    }
  }
  for (const auto& enc : encoder_weights_) {
    enc->EnsureCompiled(N, scope);
  }
}

template <typename DataType>
void AttentionBody<DataType>::Eval(int N, DmlPtr output, DmlPtr input,
                                   DmlPtr input2, DmlPtr scratch,
                                   size_t scratch_size, DmlExecScope& scope) {
  ID3D12GraphicsCommandList* list = scope.list();
  DmlPtr buffer1 = input2;
  DmlPtr buffer2 = input2 + AlignUp(scratch_size / 2);
  // The fused LayerNorms need two [tokens, C] temporaries that outlive the
  // graph dispatches around them. They live in the scratch arena's second
  // half, which nothing else uses: every Eval is handed `scratch` with
  // `scratch_size` bytes, and the arena is allocated at twice that (see
  // network_directml.cc, where scratch_elems is held to >= 2 * emb).
  DmlPtr ln_scratch = scratch + AlignUp(scratch_size);
  const uint32_t tokens = N * 64;
  const int input_size = kNumInputPlanes;

  auto record_preprocess = [&](uint32_t mode, uint32_t encoding_size,
                               uint32_t total_channels,
                               uint32_t enc_batch_stride, DmlPtr in,
                               DmlPtr encoding, DmlPtr out) {
    PreprocessConstants constants{};
    constants.mode = mode;
    constants.input_size = input_size;
    constants.encoding_size = encoding_size;
    constants.total_channels = total_channels;
    constants.enc_batch_stride = enc_batch_stride;
    list->SetComputeRootSignature(preprocess_root_signature_.Get());
    list->SetPipelineState(preprocess_pso_.Get());
    list->SetComputeRoot32BitConstants(0, sizeof(PreprocessConstants) / 4,
                                       &constants, 0);
    list->SetComputeRootShaderResourceView(1, in.GpuVA());
    // Mode 2 never reads the encoding SRV; bind a valid address anyway (a
    // null DmlPtr's GpuVA() would dereference null -- PE_DENSE nets have no
    // positional-encoding table uploaded).
    const DmlPtr encoding_slot =
        encoding ? encoding : in;
    list->SetComputeRootShaderResourceView(2, encoding_slot.GpuVA());
    list->SetComputeRootUnorderedAccessView(3, out.GpuVA());
    const UINT groups =
        mode == 2 ? static_cast<UINT>(N * 64) : static_cast<UINT>(N * 64);
    list->Dispatch(groups, 1, 1);
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    list->ResourceBarrier(1, &barrier);
  };

  const int body_input_c = is_pe_dense_embedding_
                               ? input_size + embedding_dense_size_
                               : input_size + kNumPosEncodingChannels;

  if (!is_pe_dense_embedding_) {
    record_preprocess(0, kNumPosEncodingChannels,
                      input_size + kNumPosEncodingChannels, 0, input,
                      pos_encoding_, scratch);

    auto it = compiled_.find(N);
    if (it == compiled_.end()) {
      GraphFactory<DataType> g(scope.ctx());
      auto x = g.Input({1, 1, tokens, static_cast<uint32_t>(body_input_c)});
      auto w = g.Weight(ip_emb_w_, {1, 1, static_cast<uint32_t>(embedding_op_size_),
                                    static_cast<uint32_t>(body_input_c)});
      auto b = g.WeightChannel(ip_emb_b_, tokens, static_cast<uint32_t>(embedding_op_size_));
      dml::Expression g0 =
          dml::Gemm(x, w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                    DML_MATRIX_TRANSFORM_TRANSPOSE);
      dml::Expression y = g0 + b;
      y = ActivationExpr<DataType>(activations_.default_activation, y);
      if (has_gating_) {
        auto mult = g.WeightChannel(ip_mult_gate_, tokens, static_cast<uint32_t>(embedding_op_size_));
        auto add = g.WeightChannel(ip_add_gate_, tokens, static_cast<uint32_t>(embedding_op_size_));
        y = y * mult + add;
      }
      it = compiled_.emplace(
               N, g.Compile({y}, {(uint64_t)tokens * embedding_op_size_ *
                                      sizeof(DataType)}))
               .first;
    }
    DispatchOp<DataType>(scope, it->second, scratch, buffer2, buffer2,
                         {output});
  } else {
    // PE_DENSE: pos_info slice gemm, concat, embedding gemm + LN, gating,
    // embedding FFN, LN -- one graph per N, with the preprocess kernels in
    // between (the concat kernel needs the gemm output).
    DmlPtr pos_info = scratch;  // [N, 64*12]
    DmlPtr pre_out = buffer1;   // [N, 64*dense]
    DmlPtr nhwc = scratch;      // reused: [N, 64, 112+dense]
    record_preprocess(2, 12, 768, 0, input, pos_encoding_, pos_info);

    {
      auto it = pre_compiled_.find(N);
      if (it == pre_compiled_.end()) {
        GraphFactory<DataType> g(scope.ctx());
        auto x = g.Input({1, 1, static_cast<uint32_t>(N), 64 * 12});
        auto w = g.Weight(ip_emb_pre_w_,
                          {1, 1, static_cast<uint32_t>(64 * embedding_dense_size_),
                           64 * 12});
        auto b = g.WeightChannel(ip_emb_pre_b_, static_cast<uint32_t>(N),
                         static_cast<uint32_t>(64 * embedding_dense_size_));
        dml::Expression y = dml::Gemm(x, w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                                      DML_MATRIX_TRANSFORM_TRANSPOSE) + b;
        it = pre_compiled_.emplace(
                 N, g.Compile({y}, {(uint64_t)N * 64 * embedding_dense_size_ *
                                        sizeof(DataType)}))
                 .first;
      }
      DispatchOp<DataType>(scope, it->second, pos_info, buffer2, buffer2,
                           {pre_out});
    }

    record_preprocess(1, static_cast<uint32_t>(embedding_dense_size_),
                      static_cast<uint32_t>(body_input_c),
                      static_cast<uint32_t>(64 * embedding_dense_size_), input,
                      pre_out, nhwc);

    {
      auto it = compiled_.find(N);
      if (it == compiled_.end()) {
        GraphFactory<DataType> g(scope.ctx());
        auto x = g.Input({1, 1, tokens, static_cast<uint32_t>(body_input_c)});
        auto w = g.Weight(ip_emb_w_, {1, 1, static_cast<uint32_t>(embedding_op_size_),
                                      static_cast<uint32_t>(body_input_c)});
        auto b = g.WeightChannel(ip_emb_b_, tokens, static_cast<uint32_t>(embedding_op_size_));
        auto ln_g = g.WeightChannel(ip_emb_ln_g_, tokens, static_cast<uint32_t>(embedding_op_size_));
        auto ln_b = g.WeightChannel(ip_emb_ln_b_, tokens, static_cast<uint32_t>(embedding_op_size_));
        dml::Expression emb =
            dml::Gemm(x, w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                      DML_MATRIX_TRANSFORM_TRANSPOSE);
        dml::Expression bias_b = b;
        // LayerNorm with the previous gemm's bias fused: activation is the
        // net's default (RELU/MISH), matching the SYCL embedding LN.
        dml::Expression pre_ln = LayerNormExpr<DataType>(emb, &bias_b, nullptr, ln_g, ln_b, 1.0f, 1e-3f,
            activations_.default_activation);
        if (has_gating_) {
          auto mult = g.WeightChannel(ip_mult_gate_, tokens, static_cast<uint32_t>(embedding_op_size_));
          auto add = g.WeightChannel(ip_add_gate_, tokens, static_cast<uint32_t>(embedding_op_size_));
          pre_ln = pre_ln * mult + add;
        }
        auto ffn1_w = g.Weight(
            ip_emb_ffn_d1_w_,
            {1, 1, static_cast<uint32_t>(embedding_ffn_dff_),
             static_cast<uint32_t>(embedding_op_size_)});
        auto ffn1_b = g.WeightChannel(ip_emb_ffn_d1_b_, tokens, static_cast<uint32_t>(embedding_ffn_dff_));
        auto ffn2_w = g.Weight(
            ip_emb_ffn_d2_w_,
            {1, 1, static_cast<uint32_t>(embedding_op_size_),
             static_cast<uint32_t>(embedding_ffn_dff_)});
        auto ffn2_b = g.WeightChannel(ip_emb_ffn_d2_b_, tokens, static_cast<uint32_t>(embedding_op_size_));
        auto ffn_ln_g = g.WeightChannel(ip_emb_ffn_ln_g_, tokens, static_cast<uint32_t>(embedding_op_size_));
        auto ffn_ln_b = g.WeightChannel(ip_emb_ffn_ln_b_, tokens, static_cast<uint32_t>(embedding_op_size_));
        dml::Expression ffn1 =
            dml::Gemm(pre_ln, ffn1_w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                      DML_MATRIX_TRANSFORM_TRANSPOSE) + ffn1_b;
        ffn1 = ActivationExpr<DataType>(activations_.ffn_activation, ffn1);
        dml::Expression ffn2 =
            dml::Gemm(ffn1, ffn2_w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                      DML_MATRIX_TRANSFORM_TRANSPOSE);
        const float alpha =
            static_cast<float>(pow(2.0 * encoder_weights_.size(), -0.25));
        dml::Expression out = LayerNormExpr<DataType>(ffn2, &ffn2_b, &pre_ln, ffn_ln_g, ffn_ln_b, alpha, 1e-3f,
            ACTIVATION_NONE);
        it = compiled_.emplace(
            N, g.Compile({out}, {(uint64_t)tokens * embedding_op_size_ *
                                     sizeof(DataType)})).first;
      }
      DispatchOp<DataType>(scope, it->second, nhwc, buffer2, buffer2,
                           {output});
    }
  }

  for (const auto& enc : encoder_weights_) {
    enc->Eval(N, output, scratch, buffer1, buffer2, ln_scratch, scope);
  }
}

// ===========================================================================
// EncoderBlock (MHA + KDA)
// ===========================================================================

template <typename DataType>
EncoderBlock<DataType>::EncoderBlock(
    const MultiHeadWeights::EncoderLayer& cpu_weights,
    DmlWeightUploader& uploader, int heads, int size, float alpha,
    DmlPtr smolgen_global, int smolgen_global_size, int max_batch_size,
    ActivationFunction smolgen_act, ActivationFunction ffn_act,
    float default_eps, const std::vector<int>& kda_directions,
    DmlDeviceContext& ctx, bool fp16)
    : embedding_op_size_(size),
      encoder_heads_(heads),
      alpha_(alpha),
      default_eps_(default_eps),
      has_smolgen_(cpu_weights.mha.has_smolgen && !cpu_weights.is_kda),
      smolgen_activation_(smolgen_act),
      ffn_activation_(ffn_act),
      max_batch_size_(max_batch_size),
      ctx_(ctx),
      fp16_(fp16) {
  const auto& mha = cpu_weights.mha;
  const auto& ffn = cpu_weights.ffn;
  const auto& kda = cpu_weights.kda;
  is_kda_ = cpu_weights.is_kda;

  ffn_dense1_size_ = static_cast<int>(ffn.dense1_w.size()) / size;
  mha_q_size_ = size > 0 && !mha.q_w.empty()
                    ? static_cast<int>(mha.q_w.size()) / size
                    : 0;

  mha_q_w_ = uploader.Add(mha.q_w);
  mha_q_b_ = uploader.Add(mha.q_b);
  mha_k_w_ = uploader.Add(mha.k_w);
  mha_k_b_ = uploader.Add(mha.k_b);
  mha_v_w_ = uploader.Add(mha.v_w);
  mha_v_b_ = uploader.Add(mha.v_b);
  mha_dense_w_ = uploader.Add(mha.dense_w);
  mha_dense_b_ = uploader.Add(mha.dense_b);
  ln1_gammas_ = uploader.Add(cpu_weights.ln1_gammas);
  ln1_betas_ = uploader.Add(cpu_weights.ln1_betas);
  ffn_dense1_w_ = uploader.Add(ffn.dense1_w);
  ffn_dense1_b_ = uploader.Add(ffn.dense1_b);
  ffn_dense2_w_ = uploader.Add(ffn.dense2_w);
  ffn_dense2_b_ = uploader.Add(ffn.dense2_b);
  ln2_gammas_ = uploader.Add(cpu_weights.ln2_gammas);
  ln2_betas_ = uploader.Add(cpu_weights.ln2_betas);

  if (has_smolgen_) {
    const auto& smol = mha.smolgen;
    smol_compress_ = uploader.Add(smol.compress);
    smol_dense1_w_ = uploader.Add(smol.dense1_w);
    smol_dense1_b_ = uploader.Add(smol.dense1_b);
    smol_dense2_w_ = uploader.Add(smol.dense2_w);
    smol_dense2_b_ = uploader.Add(smol.dense2_b);
    smol_ln1_gammas_ = uploader.Add(smol.ln1_gammas);
    smol_ln1_betas_ = uploader.Add(smol.ln1_betas);
    smol_ln2_gammas_ = uploader.Add(smol.ln2_gammas);
    smol_ln2_betas_ = uploader.Add(smol.ln2_betas);
    smolgen_global_ = smolgen_global;
    smolgen_global_size_ = smolgen_global_size;
    smol_compress_size_ = static_cast<int>(smol.compress.size()) / size;
    smol_dense_1_size_ = static_cast<int>(smol.dense1_b.size());
    smol_dense_2_size_ = static_cast<int>(smol.dense2_b.size());
  }

  if (is_kda_) {
    kda_q_w_ = uploader.Add(kda.q_w);
    kda_q_b_ = uploader.Add(kda.q_b);
    kda_k_w_ = uploader.Add(kda.k_w);
    kda_k_b_ = uploader.Add(kda.k_b);
    kda_v_w_ = uploader.Add(kda.v_w);
    kda_v_b_ = uploader.Add(kda.v_b);
    kda_decay_a_w_ = uploader.Add(kda.decay_a_w);
    kda_decay_a_b_ = uploader.Add(kda.decay_a_b);
    kda_decay_b_w_ = uploader.Add(kda.decay_b_w);
    kda_decay_b_b_ = uploader.Add(kda.decay_b_b);
    kda_beta_w_ = uploader.Add(kda.beta_w);
    kda_beta_b_ = uploader.Add(kda.beta_b);
    kda_a_log_ = uploader.Add(kda.a_log);
    kda_dt_bias_ = uploader.Add(kda.dt_bias);
    kda_gate_a_w_ = uploader.Add(kda.gate_a_w);
    kda_gate_a_b_ = uploader.Add(kda.gate_a_b);
    kda_gate_b_w_ = uploader.Add(kda.gate_b_w);
    kda_gate_b_b_ = uploader.Add(kda.gate_b_b);
    kda_out_norm_gammas_ = uploader.Add(kda.out_norm_gammas);
    kda_dense_w_ = uploader.Add(kda.dense_w);
    kda_dense_b_ = uploader.Add(kda.dense_b);
    kda_local_conv_w_ = uploader.Add(kda.local_conv_w);
    kda_local_conv_b_ = uploader.Add(kda.local_conv_b);

    kda_key_dim_ = kda.key_dim;
    kda_value_dim_ = kda.value_dim;
    kda_gate_rank_ = kda.gate_rank;
    kda_rms_norm_epsilon_ = kda.rms_norm_epsilon;
    kda_output_gate_ = kda.output_gate;
    kda_output_rms_norm_ = kda.output_rms_norm;
    kda_local_conv_ = kda.local_conv;
    kda_qkv_silu_ = kda.qkv_silu;
    kda_direction_count_ = std::min<int>(16, (int)kda_directions.size());
    for (int i = 0; i < 16; ++i) {
      kda_directions_[i] = i < kda_direction_count_
                               ? kda_directions[i]
                               : kda_directions_[std::max(0, i - 1)];
    }
    kda_recurrence_ = std::make_unique<KdaRecurrenceLayer>(
        ctx.device(), fp16, kda_key_dim_, kda_value_dim_);
    if (kda_local_conv_) {
      kda_local_conv_layer_ =
          std::make_unique<KdaLocalConvLayer>(ctx.device(), fp16);
    }
  } else {
    // MHA head transpose (see mha_transpose.hlsl).
    mha_transpose_ = std::make_unique<MhaTransposeLayer>(ctx.device(), fp16);
    if (has_smolgen_) {
      smolgen_bias_ = std::make_unique<SmolgenBiasLayer>(ctx.device(), fp16);
    }
  }
  // Both block kinds have the same two LayerNorms in their tail.
  layer_norm_ = std::make_unique<LayerNormLayer>(ctx.device(), fp16);
}

template <typename DataType>
void EncoderBlock<DataType>::Eval(int N, DmlPtr in_out_tensor, DmlPtr scratch,
                                  DmlPtr buffer1, DmlPtr buffer2,
                                  DmlPtr ln_scratch, DmlExecScope& scope) {
  if (is_kda_) {
    EvalKda(N, in_out_tensor, scratch, buffer1, buffer2, ln_scratch, scope);
  } else {
    EvalMha(N, in_out_tensor, scratch, buffer1, buffer2, ln_scratch, scope);
  }
}

template <typename DataType>
void EncoderBlock<DataType>::EnsureCompiled(int N, DmlExecScope& scope) {
  // NOTE: the projection/attention build blocks below mirror the lazy
  // fallback blocks inside EvalKda/EvalMha (which stay as dead-man's fuses)
  // -- keep those in sync. The LN/FFN tails no longer can drift: both paths
  // call BuildKdaTails/BuildMhaTails.
  if (is_kda_) {
    const uint32_t tokens = N * 64;
    const uint32_t max_tokens = max_batch_size_ * 64;
    const uint32_t KD = encoder_heads_ * kda_key_dim_;
    const uint32_t VD = encoder_heads_ * kda_value_dim_;
    const uint32_t gr = kda_gate_rank_;
    const uint32_t emb = embedding_op_size_;
    const size_t elem = sizeof(DataType);
    if (!kda_proj_compiled_.count(N)) {
      GraphFactory<DataType> g(scope.ctx());
      auto x = g.Input({1, 1, tokens, emb});
      auto gemm_bias =
          [&](const DmlPtr& w, const DmlPtr& b, uint32_t out_c,
              ActivationFunction act) {
            auto we = g.Weight(w, {1, 1, out_c, emb});
            auto be = g.WeightChannel(b, tokens, out_c);
            dml::Expression y =
                dml::Gemm(x, we, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                          DML_MATRIX_TRANSFORM_TRANSPOSE) + be;
            return ActivationExpr<DataType>(act, y);
          };
      ActivationFunction silu_or_none =
          kda_qkv_silu_ ? ACTIVATION_SWISH : ACTIVATION_NONE;
      dml::Expression qe = gemm_bias(kda_q_w_, kda_q_b_, KD, silu_or_none);
      dml::Expression ke = gemm_bias(kda_k_w_, kda_k_b_, KD, silu_or_none);
      dml::Expression ve = gemm_bias(kda_v_w_, kda_v_b_, VD, silu_or_none);
      dml::Expression decay_hidden =
          gemm_bias(kda_decay_a_w_, kda_decay_a_b_, gr, ACTIVATION_NONE);
      dml::Expression raw_decay_e = [&] {
        auto w = g.Weight(kda_decay_b_w_, {1, 1, KD, gr});
        auto b = g.WeightChannel(kda_decay_b_b_, tokens, KD);
        return dml::Gemm(decay_hidden, w, dml::NullOpt,
                         DML_MATRIX_TRANSFORM_NONE,
                         DML_MATRIX_TRANSFORM_TRANSPOSE) + b;
      }();
      std::vector<dml::Expression> outs{qe, ke, ve, raw_decay_e};
      std::vector<uint64_t> out_bytes{
          (uint64_t)tokens * KD * elem, (uint64_t)tokens * KD * elem,
          (uint64_t)tokens * VD * elem, (uint64_t)tokens * KD * elem};
      if (kda_output_gate_) {
        dml::Expression gate_hidden_e =
            gemm_bias(kda_gate_a_w_, kda_gate_a_b_, gr, ACTIVATION_NONE);
        dml::Expression gate_e = [&] {
          auto w = g.Weight(kda_gate_b_w_, {1, 1, VD, gr});
          auto b = g.WeightChannel(kda_gate_b_b_, tokens, VD);
          return dml::Gemm(gate_hidden_e, w, dml::NullOpt,
                           DML_MATRIX_TRANSFORM_NONE,
                           DML_MATRIX_TRANSFORM_TRANSPOSE) + b;
        }();
        outs.push_back(gate_e);
        out_bytes.push_back((uint64_t)tokens * VD * elem);
      }
      dml::Expression beta_e =
          gemm_bias(kda_beta_w_, kda_beta_b_, encoder_heads_, ACTIVATION_NONE);
      outs.push_back(beta_e);
      out_bytes.push_back((uint64_t)tokens * encoder_heads_ * elem);
      kda_proj_compiled_.emplace(N, g.Compile(outs, out_bytes));
    }
    BuildKdaTails(N, scope);
    return;
  }

  // MHA block.
  const uint32_t H = encoder_heads_;
  const uint32_t d_model = mha_q_size_;
  const uint32_t D = d_model / H;
  const uint32_t tokens = N * 64;
  const uint32_t B = (uint32_t)N * H;
  const size_t elem = sizeof(DataType);
  if (!mha_qkv_compiled_.count(N)) {
    GraphFactory<DataType> g(scope.ctx());
    auto x = g.Input({1, 1, tokens, d_model});
    auto gemm_bias = [&](const DmlPtr& w, const DmlPtr& b, uint32_t out_c) {
      auto we = g.Weight(w, {1, 1, out_c, (uint32_t)embedding_op_size_});
      auto be = g.WeightChannel(b, tokens, out_c);
      return dml::Gemm(x, we, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                       DML_MATRIX_TRANSFORM_TRANSPOSE) + be;
    };
    dml::Expression qe = gemm_bias(mha_q_w_, mha_q_b_, d_model);
    dml::Expression ke = gemm_bias(mha_k_w_, mha_k_b_, d_model);
    dml::Expression ve = gemm_bias(mha_v_w_, mha_v_b_, d_model);
    mha_qkv_compiled_.emplace(
        N, g.Compile({qe, ke, ve},
                     {(uint64_t)tokens * d_model * elem,
                      (uint64_t)tokens * d_model * elem,
                      (uint64_t)tokens * d_model * elem}));
  }
  if (!mha_attn_compiled_.count(N)) {
    GraphFactory<DataType> g(scope.ctx());
    auto qt_in = g.Input({B, 1, 64, D});
    auto kt_in = g.Input2({B, 1, 64, D});
    auto vt_in = g.Scratch({B, 1, 64, D});
    if (has_smolgen_) {
      // Smolgen MLP in three stage graphs with no mid-graph reshapes (this
      // driver fails CompileGraph on reshape-carrying graphs): M1 compress
      // GEMM; M2 dense1+LN (input reinterpreted [N*64,hid] -> [N,64*hid],
      // legal because input descs are memory-only); M3 dense2+LN. The
      // SmolgenBiasLayer kernel then makes the [N,H,64,64] bias.
      const uint32_t gen = smol_dense_2_size_ / H;
      {
      GraphFactory<DataType> g1(scope.ctx());
      auto x1 = g1.Extra({1, 1, tokens, d_model});
      auto compress1 =
          g1.Weight(smol_compress_,
                    {1, 1, (uint32_t)smol_compress_size_, d_model});
      dml::Expression comp1 =
          dml::Gemm(x1, compress1, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                    DML_MATRIX_TRANSFORM_TRANSPOSE);
      mha_mlp_compiled_.emplace(
          N, g1.Compile({comp1},
                        {(uint64_t)N * 64 * smol_compress_size_ *
                         sizeof(DataType)}));
      }
      {
      GraphFactory<DataType> g2(scope.ctx());
      auto comp2 = g2.Input({1, 1, (uint32_t)N,
                             (uint32_t)(64 * smol_compress_size_)});
      auto d1w = g2.Weight(smol_dense1_w_,
                           {1, 1, (uint32_t)smol_dense_1_size_,
                            (uint32_t)(64 * smol_compress_size_)});
      auto d1 = dml::Gemm(comp2, d1w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                          DML_MATRIX_TRANSFORM_TRANSPOSE);
      auto d1b = g2.WeightChannel(smol_dense1_b_, (uint32_t)N,
                                  (uint32_t)smol_dense_1_size_);
      auto sln1g = g2.WeightChannel(smol_ln1_gammas_, (uint32_t)N,
                                    (uint32_t)smol_dense_1_size_);
      auto sln1b = g2.WeightChannel(smol_ln1_betas_, (uint32_t)N,
                                    (uint32_t)smol_dense_1_size_);
      auto dense1 = d1 + d1b;
      const uint32_t ln_axes1[] = {3};
      const dml::Span<const uint32_t> ln_axis1(ln_axes1, 1);
      const auto sz1 = dense1.Impl()->GetOutputDesc().sizes;
      auto mean1 = GraphFactory<DataType>::ReinterpretView(
          dml::Reduce(dense1, DML_REDUCE_FUNCTION_AVERAGE, ln_axis1), sz1,
          {0, 0, 1, 0});
      auto centered1 = dense1 - mean1;
      auto var1 = GraphFactory<DataType>::ReinterpretView(
          dml::Reduce(centered1 * centered1, DML_REDUCE_FUNCTION_AVERAGE,
                      ln_axis1),
          sz1, {0, 0, 1, 0});
      auto inv1 = GraphFactory<DataType>::ReinterpretView(
          dml::Recip(dml::Sqrt(var1 + 1e-3f)), sz1, {0, 0, 1, 0});
      auto out1 = centered1 * inv1 * sln1g + sln1b;
      out1 = ActivationExpr<DataType>(smolgen_activation_, out1);
      mha_mlp2_compiled_.emplace(
          N, g2.Compile({out1},
                        {(uint64_t)N * smol_dense_1_size_ * sizeof(DataType)}));
      }
      {
      GraphFactory<DataType> g3(scope.ctx());
      auto d1_in = g3.Input({1, 1, (uint32_t)N,
                             (uint32_t)smol_dense_1_size_});
      auto d2w = g3.Weight(smol_dense2_w_,
                           {1, 1, (uint32_t)smol_dense_2_size_,
                            (uint32_t)smol_dense_1_size_});
      auto d2 = dml::Gemm(d1_in, d2w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                          DML_MATRIX_TRANSFORM_TRANSPOSE);
      auto d2b = g3.WeightChannel(smol_dense2_b_, (uint32_t)N,
                                  (uint32_t)smol_dense_2_size_);
      auto sln2g = g3.WeightChannel(smol_ln2_gammas_, (uint32_t)N,
                                    (uint32_t)smol_dense_2_size_);
      auto sln2b = g3.WeightChannel(smol_ln2_betas_, (uint32_t)N,
                                    (uint32_t)smol_dense_2_size_);
      auto dense2 = d2 + d2b;
      const uint32_t ln_axes2[] = {3};
      const dml::Span<const uint32_t> ln_axis2(ln_axes2, 1);
      const auto sz2 = dense2.Impl()->GetOutputDesc().sizes;
      auto mean2 = GraphFactory<DataType>::ReinterpretView(
          dml::Reduce(dense2, DML_REDUCE_FUNCTION_AVERAGE, ln_axis2), sz2,
          {0, 0, 1, 0});
      auto centered2 = dense2 - mean2;
      auto var2 = GraphFactory<DataType>::ReinterpretView(
          dml::Reduce(centered2 * centered2, DML_REDUCE_FUNCTION_AVERAGE,
                      ln_axis2),
          sz2, {0, 0, 1, 0});
      auto inv2 = GraphFactory<DataType>::ReinterpretView(
          dml::Recip(dml::Sqrt(var2 + 1e-3f)), sz2, {0, 0, 1, 0});
      auto out2 = centered2 * inv2 * sln2g + sln2b;
      mha_mlp3_compiled_.emplace(
          N, g3.Compile({out2},
                        {(uint64_t)N * smol_dense_2_size_ * sizeof(DataType)}));
      }
    }
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
    dml::Expression logits = dml::Gemm(
        qt_in, kt_in, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
        DML_MATRIX_TRANSFORM_TRANSPOSE, softmax_scale);

    // Only smolgen blocks have a per-(batch, head) attention bias. Declaring
    // the Extra input unconditionally made every non-smolgen MHA block add a
    // tensor that EvalMha binds as a default-constructed DmlPtr -- a
    // DML_BUFFER_BINDING with a null resource -- so the logits picked up
    // whatever that resolved to. That is the ~2e-2 drift the MhaMlh and
    // KdaMha parity nets showed: both build their MHA encoder without
    // smolgen weights, while every real net that reaches this path has them.
    if (has_smolgen_) {
      auto bias_in = g.Extra({(uint32_t)B, 1, 64, 64});
      logits = logits + bias_in;
    }
    const uint32_t softmax_axes[] = {3};
    const dml::Span<const uint32_t> softmax_axis(softmax_axes, 1);
    const auto sm_sizes = logits.Impl()->GetOutputDesc().sizes;
    const std::vector<uint32_t> sm_bcast{0, 0, 0, 1};
    dml::Expression max_b = GraphFactory<DataType>::ReinterpretView(
        dml::Reduce(logits, DML_REDUCE_FUNCTION_MAX, softmax_axis), sm_sizes,
        sm_bcast);
    dml::Expression shifted = logits - max_b;
    dml::Expression exp_shifted = dml::Exp(shifted);
    dml::Expression sum_b = GraphFactory<DataType>::ReinterpretView(
        dml::Reduce(exp_shifted, DML_REDUCE_FUNCTION_SUM, softmax_axis),
        sm_sizes, sm_bcast);
    dml::Expression attn = exp_shifted * dml::Recip(sum_b);
    dml::Expression context =
        dml::Gemm(attn, vt_in, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_NONE);
    mha_attn_compiled_.emplace(N, g.Compile({context},
                                            {(uint64_t)B * 64 * D * elem}));
  }
  BuildMhaTails(N, scope);
}

// The LN/FFN tail of an encoder block, split at its two LayerNorms.
//
// Each half is a pure GEMM graph (DirectML's meta-commands are good at
// those); the LayerNorms between and after them are single fused HLSL
// dispatches. The composed dml::Graph form these replace spent ~10 nodes per
// LayerNorm, each node a separate dispatch streaming the whole
// [tokens, channels] tensor, which dominated eval time on this hardware.
//
// Idempotent, and the only place these graphs are built -- both the
// two-phase EnsureCompiled pre-pass and the Eval paths call it, so they
// cannot fall out of step.
template <typename DataType>
void EncoderBlock<DataType>::BuildKdaTails(int N, DmlExecScope& scope) {
  const uint32_t tokens = N * 64;
  const uint32_t VD = encoder_heads_ * kda_value_dim_;
  const uint32_t emb = embedding_op_size_;
  const size_t elem = sizeof(DataType);

  // 1: KDA output norm + gate + dense projection -> raw gemm. The dense
  // bias, the alpha scale and the block-input skip all belong to the fused
  // LayerNorm that follows, so they are absent here.
  if (!kda_tail1_compiled_.count(N)) {
    GraphFactory<DataType> g(scope.ctx());
    auto mixed_in = g.Input({1, 1, tokens, VD});
    dml::Expression normed = mixed_in;
    if (kda_output_rms_norm_) {
      auto gammas = g.WeightChannel(kda_out_norm_gammas_, tokens, VD);
      normed = RmsNormExpr<DataType>(normed, gammas, kda_rms_norm_epsilon_);
    }
    if (kda_output_gate_) {
      // Gate applies after the RMS norm on purpose (they do not commute).
      auto gate_in = g.Input2({1, 1, tokens, VD});
      normed = normed * dml::ActivationSigmoid(gate_in);
    }
    auto dw = g.Weight(kda_dense_w_, {1, 1, emb, VD});
    dml::Expression dense =
        dml::Gemm(normed, dw, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE);
    kda_tail1_compiled_.emplace(
        N, g.Compile({dense}, {(uint64_t)tokens * emb * elem}));
  }

  // 2: the FFN between the two LayerNorms. ffn_dense2's bias is left to the
  // second fused LayerNorm.
  if (!kda_tail2_compiled_.count(N)) {
    GraphFactory<DataType> g(scope.ctx());
    auto ln1 = g.Input({1, 1, tokens, emb});
    auto f1w = g.Weight(ffn_dense1_w_, {1, 1, (uint32_t)ffn_dense1_size_, emb});
    auto f1b =
        g.WeightChannel(ffn_dense1_b_, tokens, (uint32_t)ffn_dense1_size_);
    dml::Expression ffn1 =
        dml::Gemm(ln1, f1w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE) + f1b;
    ffn1 = ActivationExpr<DataType>(ffn_activation_, ffn1);
    auto f2w = g.Weight(ffn_dense2_w_, {1, 1, emb, (uint32_t)ffn_dense1_size_});
    dml::Expression ffn2 =
        dml::Gemm(ffn1, f2w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE);
    kda_tail2_compiled_.emplace(
        N, g.Compile({ffn2}, {(uint64_t)tokens * emb * elem}));
  }
}

template <typename DataType>
void EncoderBlock<DataType>::BuildMhaTails(int N, DmlExecScope& scope) {
  const uint32_t tokens = N * 64;
  const uint32_t d_model = mha_q_size_;
  const uint32_t emb = embedding_op_size_;
  const size_t elem = sizeof(DataType);

  // 1: the merged-heads dense projection -> raw gemm.
  if (!mha_tail1_compiled_.count(N)) {
    GraphFactory<DataType> g(scope.ctx());
    auto mg = g.Input({1, 1, tokens, d_model});
    auto dw = g.Weight(mha_dense_w_, {1, 1, emb, d_model});
    dml::Expression dense =
        dml::Gemm(mg, dw, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE);
    mha_tail1_compiled_.emplace(
        N, g.Compile({dense}, {(uint64_t)tokens * emb * elem}));
  }

  // 2: the FFN between the two LayerNorms.
  //
  // NOTE: biases here MUST be WeightChannel (strided matching-sizes), not
  // dense size-1 [1,1,1,C]: this runtime rejects size-1 broadcast in
  // elementwise Add (E_INVALIDARG at CreateOperator).
  if (!mha_tail2_compiled_.count(N)) {
    GraphFactory<DataType> g(scope.ctx());
    auto ln1 = g.Input({1, 1, tokens, emb});
    auto f1w = g.Weight(ffn_dense1_w_, {1, 1, (uint32_t)ffn_dense1_size_, emb});
    auto f1b =
        g.WeightChannel(ffn_dense1_b_, tokens, (uint32_t)ffn_dense1_size_);
    dml::Expression ffn1 =
        dml::Gemm(ln1, f1w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE) + f1b;
    ffn1 = ActivationExpr<DataType>(ffn_activation_, ffn1);
    auto f2w = g.Weight(ffn_dense2_w_, {1, 1, emb, (uint32_t)ffn_dense1_size_});
    dml::Expression ffn2 =
        dml::Gemm(ffn1, f2w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE);
    mha_tail2_compiled_.emplace(
        N, g.Compile({ffn2}, {(uint64_t)tokens * emb * elem}));
  }
}

// MHA evaluation with dense [N*H,1]-batch attention (see mha_transpose.hlsl
// for why the head interleave cannot stay a strided GEMM input on this
// driver). Four DML graphs (qkv projections, attention, dense projection,
// FFN) with two HLSL transposes and two fused LayerNorms between them:
//
//   in_out [T,H*D] --qkv--> scratch q/k/v --split--> buffer1 qt/kt/vt [B,64,D]
//     --attn--> buffer1 ctx [B,64,D] --merge--> buffer1 merged [T,H*D]
//     --tail--> in_out (with the block-input skip, like the SYCL block).
//
// B = N*H. S below = max_tokens*d_model*elem; buffer1 holds qt/kt/vt/ctx/
// merged (5*S). Scratch holds q/k/v (3*d_model/token, already covered by the
// scratch-size estimate's MHA branch).
template <typename DataType>
void EncoderBlock<DataType>::EvalMha(int N, DmlPtr in_out_tensor,
                                     DmlPtr scratch, DmlPtr buffer1,
                                     DmlPtr buffer2, DmlPtr ln_scratch,
                                     DmlExecScope& scope) {
  const uint32_t H = encoder_heads_;
  const uint32_t d_model = mha_q_size_;
  if (H == 0 || d_model == 0 || d_model % H != 0) {
    throw Exception(
        "directml backend: malformed MHA head geometry in this net.");
  }
  const uint32_t D = d_model / H;
  const uint32_t tokens = N * 64;
  const uint32_t B = (uint32_t)N * H;
  const size_t elem = sizeof(DataType);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));

  const uint64_t max_tokens = (uint64_t)max_batch_size_ * 64;
  const uint64_t S = max_tokens * d_model * elem;
  DmlPtr q = scratch;
  DmlPtr k = q + S;
  DmlPtr v = k + S;
  DmlPtr qt = buffer1;
  DmlPtr kt = buffer1 + S;
  DmlPtr vt = buffer1 + 2 * S;
  DmlPtr ctxb = buffer1 + 3 * S;
  DmlPtr merged = buffer1 + 4 * S;

  // 1. QKV projections, dense [T, H*D] into scratch.
  {
    auto it = mha_qkv_compiled_.find(N);
    if (it == mha_qkv_compiled_.end()) {
      GraphFactory<DataType> g(scope.ctx());
      auto x = g.Input({1, 1, tokens, d_model});
      auto gemm_bias = [&](const DmlPtr& w, const DmlPtr& b, uint32_t out_c) {
        auto we = g.Weight(w, {1, 1, out_c, (uint32_t)embedding_op_size_});
        auto be = g.WeightChannel(b, tokens, out_c);
        return dml::Gemm(x, we, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                         DML_MATRIX_TRANSFORM_TRANSPOSE) +
               be;
      };
      dml::Expression qe =
          gemm_bias(mha_q_w_, mha_q_b_, d_model);
      dml::Expression ke =
          gemm_bias(mha_k_w_, mha_k_b_, d_model);
      dml::Expression ve =
          gemm_bias(mha_v_w_, mha_v_b_, d_model);
      it = mha_qkv_compiled_
               .emplace(N, g.Compile({qe, ke, ve},
                                     {(uint64_t)tokens * d_model * elem,
                                      (uint64_t)tokens * d_model * elem,
                                      (uint64_t)tokens * d_model * elem}))
               .first;
    }
    DispatchOp<DataType>(scope, it->second, in_out_tensor, buffer1, buffer2,
                         {q, k, v});
  }

  // 2. Head-split transpose (HLSL): [T,H*D] -> [B,64,D] dense each.
  {
    MhaTransposeLayer::Params params{};
    params.batch_size = N;
    params.heads = H;
    params.head_dim = D;
    params.mode = 0;
    mha_transpose_->Record(scope.list(), params, q, qt);
    mha_transpose_->Record(scope.list(), params, k, kt);
    mha_transpose_->Record(scope.list(), params, v, vt);
  }

  // 3. Attention over dense [B,1,64,D]: scores, softmax, context.
  {
    auto it = mha_attn_compiled_.find(N);
    if (it == mha_attn_compiled_.end()) {
      throw Exception(
          "directml backend: MHA attention graph was not compiled; "
          "EnsureCompiled must run before Eval.");
    }
    DmlPtr bias;
    if (has_smolgen_) {
      // Smolgen MLP stage graphs (compress / dense1+LN / dense2+LN), then
      // the bias HLSL kernel: all intermediates are per-batch, so they come
      // from the transient arena at the real batch size.
      auto m1 = mha_mlp_compiled_.find(N);
      auto m2 = mha_mlp2_compiled_.find(N);
      auto m3 = mha_mlp3_compiled_.find(N);
      if (m1 == mha_mlp_compiled_.end() || m2 == mha_mlp2_compiled_.end() ||
          m3 == mha_mlp3_compiled_.end()) {
        throw Exception(
            "directml backend: smolgen MLP graphs were not compiled; "
            "EnsureCompiled must run before Eval.");
      }
      const uint32_t gen = smol_dense_2_size_ / H;
      auto compressed =
          scope.TakeTransient((uint64_t)N * 64 * smol_compress_size_ * elem);
      auto d1 = scope.TakeTransient((uint64_t)N * smol_dense_1_size_ * elem);
      auto d2 = scope.TakeTransient((uint64_t)N * smol_dense_2_size_ * elem);
      bias = scope.TakeTransient((uint64_t)N * H * smolgen_global_size_ * elem);
      DispatchOp<DataType>(scope, m1->second, in_out_tensor, DmlPtr(),
                           DmlPtr(), {compressed});
      DispatchOp<DataType>(scope, m2->second, compressed, DmlPtr(), DmlPtr(),
                           {d1});
      DispatchOp<DataType>(scope, m3->second, d1, DmlPtr(), DmlPtr(), {d2});
      SmolgenBiasLayer::Params sp{};
      sp.batch = N;
      sp.heads = H;
      sp.gen = gen;
      smolgen_bias_->Record(scope.list(), sp, smolgen_global_, d2, bias);
    }
    DispatchOp<DataType>(scope, it->second, qt, kt, vt, {ctxb}, bias);
  }

  // 4. Merge back to token-major [T, H*D].
  {
    MhaTransposeLayer::Params params{};
    params.batch_size = N;
    params.heads = H;
    params.head_dim = D;
    params.mode = 1;
    mha_transpose_->Record(scope.list(), params, ctxb, merged);
  }

  // 5. Dense projection, LN1, FFN, LN2: two GEMM graphs with a fused
  // LayerNorm dispatch after each. `dense` is dead once LN1 has consumed it,
  // so the FFN's output reuses its buffer -- two [tokens, emb] temporaries
  // in total, which is what the LN scratch region is sized for.
  {
    BuildMhaTails(N, scope);
    const uint32_t emb = (uint32_t)embedding_op_size_;
    const DmlPtr dense = ln_scratch;
    const DmlPtr ln1 = ln_scratch + max_tokens * emb * elem;
    const DmlPtr ffn2 = dense;

    DispatchOp<DataType>(scope, mha_tail1_compiled_.at(N), merged, buffer2,
                         buffer2, {dense});

    LayerNormLayer::Params ln{};
    ln.rows = tokens;
    ln.channels = emb;
    ln.has_bias = true;
    ln.has_skip = true;
    ln.act = ACTIVATION_NONE;
    ln.alpha = alpha_;
    ln.eps = default_eps_;
    // LN1's skip is the block input, which in_out_tensor still holds.
    layer_norm_->Record(scope.list(), ln, dense, mha_dense_b_, in_out_tensor,
                        ln1_gammas_, ln1_betas_, ln1);

    DispatchOp<DataType>(scope, mha_tail2_compiled_.at(N), ln1, buffer2,
                         buffer2, {ffn2});

    layer_norm_->Record(scope.list(), ln, ffn2, ffn_dense2_b_, ln1,
                        ln2_gammas_, ln2_betas_, in_out_tensor);
  }
}

template <typename DataType>
void EncoderBlock<DataType>::EvalKda(int N, DmlPtr in_out_tensor,
                                     DmlPtr scratch, DmlPtr buffer1,
                                     DmlPtr buffer2, DmlPtr ln_scratch,
                                     DmlExecScope& scope) {
  constexpr float kKdaLogDecayFloor = -10.0f;
  const uint32_t tokens = N * 64;
  const uint32_t max_tokens = max_batch_size_ * 64;
  const uint32_t KD = encoder_heads_ * kda_key_dim_;
  const uint32_t VD = encoder_heads_ * kda_value_dim_;
  const uint32_t gr = kda_gate_rank_;
  const uint32_t emb = embedding_op_size_;
  const size_t elem = sizeof(DataType);

  // Scratch layout mirrors sycl EvalKda's, in bytes.
  DmlPtr q = scratch;
  DmlPtr k = q + max_tokens * KD * elem;
  DmlPtr v = k + max_tokens * KD * elem;
  DmlPtr gate_hidden = scratch + max_tokens * (2 * KD + VD) * elem;
  DmlPtr proj_input = scratch + max_tokens * (2 * KD + VD + gr) * elem;
  DmlPtr proj_in = in_out_tensor;
  DmlPtr raw_decay = buffer2;
  DmlPtr gate = buffer2 + max_tokens * KD * elem;
  DmlPtr beta = buffer1 + max_tokens * VD * elem;
  DmlPtr mixed = buffer1;


  {
    if (kda_local_conv_) {
      // 3x3 depthwise board conv + residual as an HLSL kernel (see
      // shaders/kda_local_conv.hlsl); the projection graph consumes
      // proj_input.
      KdaLocalConvLayer::Params params{};
      params.tokens = tokens;
      params.emb = emb;
      kda_local_conv_layer_->Record(scope.list(), params, in_out_tensor,
                              kda_local_conv_w_, kda_local_conv_b_,
                              proj_input);
      proj_in = proj_input;
    }
    auto it = kda_proj_compiled_.find(N);
    if (it == kda_proj_compiled_.end()) {
      GraphFactory<DataType> g(scope.ctx());
      auto x = g.Input({1, 1, tokens, emb});
      auto gemm_bias =
          [&](const DmlPtr& w, const DmlPtr& b, uint32_t out_c,
              ActivationFunction act) {
            auto we = g.Weight(w, {1, 1, out_c, emb});
            auto be = g.WeightChannel(b, tokens, out_c);
            dml::Expression y =
                dml::Gemm(x, we, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                          DML_MATRIX_TRANSFORM_TRANSPOSE) + be;
            return ActivationExpr<DataType>(act, y);
          };
      ActivationFunction silu_or_none =
          kda_qkv_silu_ ? ACTIVATION_SWISH : ACTIVATION_NONE;
      dml::Expression qe = gemm_bias(kda_q_w_, kda_q_b_, KD, silu_or_none);
      dml::Expression ke = gemm_bias(kda_k_w_, kda_k_b_, KD, silu_or_none);
      dml::Expression ve = gemm_bias(kda_v_w_, kda_v_b_, VD, silu_or_none);
      dml::Expression decay_hidden =
          gemm_bias(kda_decay_a_w_, kda_decay_a_b_, gr, ACTIVATION_NONE);
      dml::Expression raw_decay_e = [&] {
        auto w = g.Weight(kda_decay_b_w_, {1, 1, KD, gr});
        auto b = g.WeightChannel(kda_decay_b_b_, tokens, KD);
        return dml::Gemm(decay_hidden, w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                         DML_MATRIX_TRANSFORM_TRANSPOSE) + b;
      }();
      std::vector<dml::Expression> outs{qe, ke, ve, raw_decay_e};
      std::vector<uint64_t> out_bytes{
          (uint64_t)tokens * KD * elem, (uint64_t)tokens * KD * elem,
          (uint64_t)tokens * VD * elem, (uint64_t)tokens * KD * elem};
      if (kda_output_gate_) {
        dml::Expression gate_hidden_e =
            gemm_bias(kda_gate_a_w_, kda_gate_a_b_, gr, ACTIVATION_NONE);
        dml::Expression gate_e = [&] {
          auto w = g.Weight(kda_gate_b_w_, {1, 1, VD, gr});
          auto b = g.WeightChannel(kda_gate_b_b_, tokens, VD);
          return dml::Gemm(gate_hidden_e, w, dml::NullOpt,
                           DML_MATRIX_TRANSFORM_NONE,
                           DML_MATRIX_TRANSFORM_TRANSPOSE) + b;
        }();
        outs.push_back(gate_e);
        out_bytes.push_back((uint64_t)tokens * VD * elem);
      }
      dml::Expression beta_e =
          gemm_bias(kda_beta_w_, kda_beta_b_, encoder_heads_,
                    ACTIVATION_NONE);
      outs.push_back(beta_e);
      out_bytes.push_back((uint64_t)tokens * encoder_heads_ * elem);
      it = kda_proj_compiled_.emplace(N, g.Compile(outs, out_bytes)).first;
    }
    // The proj graph's outputs land in scratch regions; the recurrence
    // consumes them next. Output buffers: q, k, v, raw_decay, [gate], beta.
    std::vector<DmlPtr> outs{q, k, v, raw_decay};
    if (kda_output_gate_) outs.push_back(gate);
    outs.push_back(beta);
    DispatchOp<DataType>(scope, it->second, proj_in, buffer1, buffer2, outs);
  }

  {
    KdaRecurrenceLayer::Params params{};
    params.batch_size = N;
    params.heads = encoder_heads_;
    params.key_dim = kda_key_dim_;
    params.value_dim = kda_value_dim_;
    params.direction_count = kda_direction_count_;
    params.directions = kda_directions_;
    params.use_fused_qkv = false;
    params.qkv_stride = 2 * KD + VD;
    params.log_decay_floor = kKdaLogDecayFloor;
    params.fp16 = fp16_;
    kda_recurrence_->Record(scope.list(), params, DmlPtr(), q, k, v,
                            raw_decay, kda_dt_bias_, kda_a_log_, beta, mixed);
  }

  // Output norm + gate + dense projection, LN1, FFN, LN2: two GEMM graphs
  // with a fused LayerNorm dispatch after each (see BuildKdaTails). `dense`
  // dies at LN1, so the FFN output reuses its buffer.
  {
    BuildKdaTails(N, scope);
    const DmlPtr dense = ln_scratch;
    const DmlPtr ln1 = ln_scratch + (uint64_t)max_tokens * emb * elem;
    const DmlPtr ffn2 = dense;

    DispatchOp<DataType>(scope, kda_tail1_compiled_.at(N), mixed, gate,
                         buffer2, {dense});

    LayerNormLayer::Params ln{};
    ln.rows = tokens;
    ln.channels = emb;
    ln.has_bias = true;
    ln.has_skip = true;
    ln.act = ACTIVATION_NONE;
    ln.alpha = alpha_;
    ln.eps = default_eps_;
    // LN1's skip is the block input, still in in_out_tensor at this point.
    layer_norm_->Record(scope.list(), ln, dense, kda_dense_b_, in_out_tensor,
                        ln1_gammas_, ln1_betas_, ln1);

    DispatchOp<DataType>(scope, kda_tail2_compiled_.at(N), ln1, buffer2,
                         buffer2, {ffn2});

    layer_norm_->Record(scope.list(), ln, ffn2, ffn_dense2_b_, ln1,
                        ln2_gammas_, ln2_betas_, in_out_tensor);
  }
}

// ===========================================================================
// AttentionPolicyHead
// ===========================================================================

template <typename DataType>
AttentionPolicyHead<DataType>::AttentionPolicyHead(
    BaseLayer<DataType>* ip, const MultiHeadWeights::PolicyHead& weights,
    DmlWeightUploader& uploader, bool attention_body, ActivationFunction act,
    int max_batch_size, DmlDeviceContext& ctx, bool fp16,
    const std::vector<int>& kda_directions)
    : BaseLayer<DataType>(64 * 64 + 8 * 24, 1, 1, ip),
      embedding_op_size_(static_cast<int>(weights.ip_pol_b.size())),
      policy_d_model_(static_cast<int>(weights.ip2_pol_b.size())),
      attention_body_(attention_body),
      // Old networks without attention body (e.g. T79) use hardcoded SELU
      // activations -- same as the SYCL head.
      act_(attention_body ? act : ACTIVATION_SELU) {
  ip_pol_w_ = uploader.Add(weights.ip_pol_w);
  ip_pol_b_ = uploader.Add(weights.ip_pol_b);
  ip2_pol_w_ = uploader.Add(weights.ip2_pol_w);
  ip2_pol_b_ = uploader.Add(weights.ip2_pol_b);
  ip3_pol_w_ = uploader.Add(weights.ip3_pol_w);
  ip3_pol_b_ = uploader.Add(weights.ip3_pol_b);
  ip4_pol_w_ = uploader.Add(weights.ip4_pol_w);
  encoder_heads_ = weights.pol_encoder_head_count;

  ComPtr<ID3DBlob> shader =
      CompileHlsl(kPolicyFinalizeShaderSource,
                  sizeof(kPolicyFinalizeShaderSource) - 1,
                  "policy_finalize.hlsl", "PolicyFinalize", fp16);
  finalize_root_signature_ = CreateShaderRootSignature(
      ctx.device(), sizeof(PolicyFinalizeConstants) / 4, 3, 1);
  finalize_pso_ = CreateComputePso(ctx.device(), finalize_root_signature_.Get(),
                                   shader.Get());

  // Alpha 1.0 and eps 1e-6 for the policy encoders, matching the SYCL head
  // (which notes they may change); smolgen is not implemented in policy
  // encoder heads there either.
  for (const auto& enc : weights.pol_encoder) {
    if (enc.mha.has_smolgen) {
      throw Exception(
          "directml backend: smolgen in policy-encoder heads is not "
          "supported (matches the SYCL backend).");
    }
    encoder_weights_.emplace_back(new EncoderBlock<DataType>(
        enc, uploader, encoder_heads_, embedding_op_size_, 1.0f, DmlPtr(), 0,
        max_batch_size, ACTIVATION_SWISH, act_, 1e-6f, kda_directions, ctx,
        fp16));
  }
}

template <typename DataType>
void AttentionPolicyHead<DataType>::EnsureCompiled(int N, DmlExecScope& scope) {
  const uint32_t tokens = N * 64;
  const size_t elem = sizeof(DataType);
  if (!compiled_.count(N)) {
    GraphFactory<DataType> g(scope.ctx());
    const int input_c = this->input_->GetC();
    auto x = g.Input({1, 1, tokens, (uint32_t)input_c});
    auto w = g.Weight(ip_pol_w_,
                      {1, 1, (uint32_t)embedding_op_size_, (uint32_t)input_c});
    auto b = g.WeightChannel(ip_pol_b_, tokens, (uint32_t)embedding_op_size_);
    dml::Expression y =
        dml::Gemm(x, w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE) + b;
    y = ActivationExpr<DataType>(act_, y);
    compiled_.emplace(
        N, g.Compile({y}, {(uint64_t)tokens * embedding_op_size_ * elem}));
  }
  if (!compiled_.count(-N - 1)) {
    GraphFactory<DataType> g(scope.ctx());
    auto x = g.Input({1, 1, tokens, (uint32_t)embedding_op_size_});
    auto wqe =
        g.Weight(ip2_pol_w_, {1, 1, (uint32_t)policy_d_model_,
                              (uint32_t)embedding_op_size_});
    auto qbe = g.WeightChannel(ip2_pol_b_, tokens, (uint32_t)policy_d_model_);
    auto wke =
        g.Weight(ip3_pol_w_, {1, 1, (uint32_t)policy_d_model_,
                              (uint32_t)embedding_op_size_});
    auto kbe = g.WeightChannel(ip3_pol_b_, tokens, (uint32_t)policy_d_model_);
    dml::Expression wq_e =
        dml::Gemm(x, wqe, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE) + qbe;
    dml::Expression wk_e =
        dml::Gemm(x, wke, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE) + kbe;
    // scores[n] = wq[n] @ wk[n]^T / sqrt(d), i.e. scores[i,j] =
    // Q_i.K_j, matching the CUDA backend (tf.matmul(queries, keys,
    // transpose_b=True)) and the BLAS backend (whose Eigen col-major
    // maps land the same Q_i.K_j in row-major (i,j); dot products
    // commute, so K_j.Q_i there is the same value). Batch over N with
    // strided views of the dense [T, d] buffers ([N,1,64,d] per batch).
    dml::Expression logits = dml::Gemm(
        GraphFactory<DataType>::ReinterpretView(wq_e, {(uint32_t)N, 1, 64,
                                (uint32_t)policy_d_model_},
                         {64u * policy_d_model_, policy_d_model_, policy_d_model_, 1u}),
        GraphFactory<DataType>::ReinterpretView(wk_e, {(uint32_t)N, 1, 64,
                                (uint32_t)policy_d_model_},
                         {64u * policy_d_model_, policy_d_model_, policy_d_model_, 1u}),
        dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE,
        1.0f / std::sqrt((float)policy_d_model_));
    compiled_.emplace(
        -N - 1,
        g.Compile({wq_e, wk_e, logits},
                  {(uint64_t)tokens * policy_d_model_ * elem,
                   (uint64_t)tokens * policy_d_model_ * elem,
                   (uint64_t)N * 4096 * elem}));
  }
  for (const auto& enc : encoder_weights_) {
    enc->EnsureCompiled(N, scope);
  }
}

template <typename DataType>
void AttentionPolicyHead<DataType>::Eval(int N, DmlPtr output, DmlPtr input,
                                         DmlPtr input2, DmlPtr scratch,
                                         size_t scratch_size,
                                         DmlExecScope& scope) {
  ID3D12GraphicsCommandList* list = scope.list();
  const uint32_t tokens = N * 64;
  const size_t elem = sizeof(DataType);
  DmlPtr buffer1 = output + AlignUp(scratch_size / 2);
  DmlPtr buffer2 = input2 + AlignUp(scratch_size / 2);
  // The fused LayerNorms need two [tokens, C] temporaries that outlive the
  // graph dispatches around them. They live in the scratch arena's second
  // half, which nothing else uses: every Eval is handed `scratch` with
  // `scratch_size` bytes, and the arena is allocated at twice that (see
  // network_directml.cc, where scratch_elems is held to >= 2 * emb).
  DmlPtr ln_scratch = scratch + AlignUp(scratch_size);

  DmlPtr embedding = input2;  // policy embedding + encoders run here
  {
    auto it = compiled_.find(N);
    DispatchOp<DataType>(scope, it->second, input, buffer2, buffer2,
                         {embedding});
  }

  for (const auto& enc : encoder_weights_) {
    enc->Eval(N, embedding, scratch, buffer1, buffer2, ln_scratch, scope);
  }

  DmlPtr wq = scratch;
  DmlPtr wk = scratch + (uint64_t)tokens * policy_d_model_ * elem;
  DmlPtr scores = scratch + 2 * (uint64_t)tokens * policy_d_model_ * elem;

  {
    auto it = compiled_.find(-N - 1);  // separate slot for the wqk/scores graph
    DispatchOp<DataType>(scope, it->second, embedding, buffer1, buffer2,
                         {wq, wk, scores});
  }

  {
    PolicyFinalizeConstants constants{};
    constants.key_width = policy_d_model_;
    list->SetComputeRootSignature(finalize_root_signature_.Get());
    list->SetPipelineState(finalize_pso_.Get());
    list->SetComputeRoot32BitConstants(0, sizeof(PolicyFinalizeConstants) / 4,
                                       &constants, 0);
    list->SetComputeRootShaderResourceView(1, scores.GpuVA());
    list->SetComputeRootShaderResourceView(2, wk.GpuVA());
    list->SetComputeRootShaderResourceView(3, ip4_pol_w_.GpuVA());
    list->SetComputeRootUnorderedAccessView(4, output.GpuVA());
    list->Dispatch(N, 1, 1);
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    list->ResourceBarrier(1, &barrier);
  }
}

// ===========================================================================
// ValueHead
// ===========================================================================

template <typename DataType>
ValueHead<DataType>::ValueHead(BaseLayer<DataType>* ip,
                               const MultiHeadWeights::ValueHead& weights,
                               DmlWeightUploader& uploader, bool wdl,
                               ActivationFunction act)
    : BaseLayer<DataType>(wdl ? 3 : 1, 1, 1, ip),
      value_hidden_size_(static_cast<int>(weights.ip1_val_b.size())),
      wdl_(wdl),
      act_(act) {
  embedding_size_ = static_cast<int>(weights.ip_val_b.size());
  ip_val_w_ = uploader.Add(weights.ip_val_w);
  ip_val_b_ = uploader.Add(weights.ip_val_b);
  ip1_val_w_ = uploader.Add(weights.ip1_val_w);
  ip1_val_b_ = uploader.Add(weights.ip1_val_b);
  ip2_val_w_ = uploader.Add(weights.ip2_val_w);
  ip2_val_b_ = uploader.Add(weights.ip2_val_b);
}

template <typename DataType>
void ValueHead<DataType>::EnsureCompiled(int N, DmlExecScope& scope) {
  if (compiled_.count(N)) return;
  {
    const uint32_t tokens = N * 64;
    const size_t elem = sizeof(DataType);
    GraphFactory<DataType> g(scope.ctx());
    auto x = g.Input({1, 1, tokens, (uint32_t)this->input_->GetC()});
    auto w0 = g.Weight(ip_val_w_,
                       {1, 1, (uint32_t)embedding_size_,
                        (uint32_t)this->input_->GetC()});
    auto b0 = g.WeightChannel(ip_val_b_, tokens, (uint32_t)embedding_size_);
    dml::Expression emb =
        dml::Gemm(x, w0, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE) + b0;
    emb = ActivationExpr<DataType>(act_, emb);
    // Flatten [N, embedding*64] for the hidden gemm.
    dml::Expression flat =
        GraphFactory<DataType>::ReinterpretView(emb, {1, 1, (uint32_t)N,
                               (uint32_t)(64 * embedding_size_)});
    auto w1 = g.Weight(ip1_val_w_,
                       {1, 1, (uint32_t)value_hidden_size_,
                        (uint32_t)(64 * embedding_size_)});
    auto b1 = g.WeightChannel(ip1_val_b_, (uint32_t)N, (uint32_t)value_hidden_size_);
    dml::Expression hidden =
        dml::Gemm(flat, w1, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE) + b1;
    hidden = ActivationExpr<DataType>(act_, hidden);
    auto w2 = g.Weight(ip2_val_w_,
                       {1, 1, wdl_ ? 3u : 1u, (uint32_t)value_hidden_size_});
    auto b2 = g.WeightChannel(ip2_val_b_, (uint32_t)N, wdl_ ? 3u : 1u);
    dml::Expression y =
        dml::Gemm(hidden, w2, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE) + b2;
    if (!wdl_) y = dml::ActivationTanh(y);
    compiled_.emplace(
        N, g.Compile({y}, {(uint64_t)N * (wdl_ ? 3 : 1) * elem}));
  }
}

template <typename DataType>
void ValueHead<DataType>::Eval(int N, DmlPtr output, DmlPtr input, DmlPtr input2,
                               DmlPtr scratch, size_t scratch_size,
                               DmlExecScope& scope) {
  if (!compiled_.count(N)) EnsureCompiled(N, scope);
  DispatchOp<DataType>(scope, compiled_.find(N)->second, input, input2, scratch,
                       {output});
}

// ===========================================================================
// Explicit template instantiations (cuda/layers.cc convention).
// ===========================================================================
template class FCLayer<float>;
template class FCLayer<DmlHalf>;
template class EmbeddingLayer<float>;
template class EmbeddingLayer<DmlHalf>;
template class PolicyMapLayer<float>;
template class PolicyMapLayer<DmlHalf>;
template class AttentionBody<float>;
template class AttentionBody<DmlHalf>;
template class EncoderBlock<float>;
template class EncoderBlock<DmlHalf>;
template class AttentionPolicyHead<float>;
template class AttentionPolicyHead<DmlHalf>;
template class ValueHead<float>;
template class ValueHead<DmlHalf>;

}  // namespace directml_backend
}  // namespace lczero
