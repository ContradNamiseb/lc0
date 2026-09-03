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

#include <d3dcompiler.h>

#include <cassert>
#include <cstring>

#include "neural/backends/directml/kda_recurrence_shader_source.h"
#include "utils/exception.h"

#pragma comment(lib, "d3dcompiler.lib")

namespace lczero {
namespace directml_backend {

namespace {

// Layout must match kda_recurrence.hlsl's KdaRecurrenceConstants cbuffer
// field-for-field: same order, same types, no gaps. Root constants are
// copied as raw 32-bit words, so any mismatch here silently scrambles the
// shader's inputs rather than failing to compile.
struct ShaderConstants {
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
static_assert(sizeof(ShaderConstants) == 96,
              "ShaderConstants must match the HLSL cbuffer's 96-byte layout "
              "(24 x 32-bit words) -- see kda_recurrence.hlsl");
static_assert(sizeof(ShaderConstants) % 4 == 0, "must be a whole number of 32-bit words");

constexpr UINT kNum32BitConstants = sizeof(ShaderConstants) / 4;

// Root parameter indices, fixed by construction order in the root
// signature built below -- Record() must agree with these.
constexpr UINT kRootParamConstants = 0;
constexpr UINT kRootParamSrvBase = 1;  // qkv, q, k, v, raw_decay, dt_bias, a_log, beta
constexpr UINT kRootParamUav = 9;      // mixed

ComPtr<ID3DBlob> CompileShader(bool fp16) {
  D3D_SHADER_MACRO macros[] = {
      {"INPUT_TYPE", fp16 ? "half" : "float"},
      {nullptr, nullptr},
  };

  UINT compile_flags = D3DCOMPILE_OPTIMIZATION_LEVEL3;
#ifndef NDEBUG
  compile_flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
  // NOTE: D3DCompile is the legacy FXC compiler, capped at shader model
  // 5.1, where `half` is only a min-precision hint (min16float) -- the
  // driver is free to still execute it at 32-bit. Genuine always-16-bit
  // storage needs DXC with target cs_6_2+ and -enable-16bit-types, which
  // this backend does not yet use. fp16=true here is therefore a real but
  // unverified-magnitude bandwidth optimization, not a guaranteed one --
  // unlike the SYCL kernel's T=sycl::half, which is a true 16-bit type.

  ComPtr<ID3DBlob> shader_blob;
  ComPtr<ID3DBlob> error_blob;
  HRESULT hr = D3DCompile(
      kKdaRecurrenceShaderSource, sizeof(kKdaRecurrenceShaderSource) - 1,
      "kda_recurrence.hlsl", macros, D3D_COMPILE_STANDARD_FILE_INCLUDE,
      "KdaRecurrence", "cs_5_1", compile_flags, 0, &shader_blob, &error_blob);

  if (FAILED(hr)) {
    std::string message = "Failed to compile kda_recurrence.hlsl (fp16=";
    message += fp16 ? "true" : "false";
    message += "): ";
    if (error_blob) {
      message.append(static_cast<const char*>(error_blob->GetBufferPointer()),
                     error_blob->GetBufferSize());
    } else {
      message += "HRESULT 0x" + std::to_string(static_cast<uint32_t>(hr));
    }
    throw Exception(message);
  }
  return shader_blob;
}

}  // namespace

KdaRecurrenceLayer::KdaRecurrenceLayer(ID3D12Device* device, bool fp16)
    : device_(device), fp16_(fp16) {
  D3D12_ROOT_PARAMETER params[10] = {};

  params[kRootParamConstants].ParameterType =
      D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  params[kRootParamConstants].Constants.ShaderRegister = 0;
  params[kRootParamConstants].Constants.RegisterSpace = 0;
  params[kRootParamConstants].Constants.Num32BitValues = kNum32BitConstants;
  params[kRootParamConstants].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

  for (UINT i = 0; i < 8; ++i) {
    auto& p = params[kRootParamSrvBase + i];
    p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    p.Descriptor.ShaderRegister = i;
    p.Descriptor.RegisterSpace = 0;
    p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }

  params[kRootParamUav].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
  params[kRootParamUav].Descriptor.ShaderRegister = 0;
  params[kRootParamUav].Descriptor.RegisterSpace = 0;
  params[kRootParamUav].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

  D3D12_ROOT_SIGNATURE_DESC root_desc = {};
  root_desc.NumParameters = 10;
  root_desc.pParameters = params;
  root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

  ComPtr<ID3DBlob> signature_blob;
  ComPtr<ID3DBlob> error_blob;
  HRESULT hr = D3D12SerializeRootSignature(
      &root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature_blob, &error_blob);
  if (FAILED(hr)) {
    std::string message = "Failed to serialize KdaRecurrence root signature: ";
    if (error_blob) {
      message.append(static_cast<const char*>(error_blob->GetBufferPointer()),
                     error_blob->GetBufferSize());
    }
    throw Exception(message);
  }

  hr = device_->CreateRootSignature(0, signature_blob->GetBufferPointer(),
                                    signature_blob->GetBufferSize(),
                                    IID_PPV_ARGS(&root_signature_));
  if (FAILED(hr)) throw Exception("Failed to create KdaRecurrence root signature");

  ComPtr<ID3DBlob> shader_blob = CompileShader(fp16);

  D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
  pso_desc.pRootSignature = root_signature_.Get();
  pso_desc.CS = {shader_blob->GetBufferPointer(), shader_blob->GetBufferSize()};

  hr = device_->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso_));
  if (FAILED(hr)) throw Exception("Failed to create KdaRecurrence PSO");
}

void KdaRecurrenceLayer::Record(ID3D12GraphicsCommandList* command_list,
                                const Params& params, ID3D12Resource* qkv,
                                ID3D12Resource* q, ID3D12Resource* k,
                                ID3D12Resource* v, ID3D12Resource* raw_decay,
                                ID3D12Resource* dt_bias, ID3D12Resource* a_log,
                                ID3D12Resource* beta,
                                ID3D12Resource* mixed_out) {
  const bool fused = params.use_fused_qkv;
  assert(fused ? (qkv != nullptr) : (q != nullptr && k != nullptr && v != nullptr));
  assert(params.key_dim <= 32 && params.value_dim <= 32 &&
        "KdaRecurrenceLayer's shader is compiled for BLOCK_SIZE=32; a "
        "larger key_dim/value_dim needs a second shader variant, the same "
        "way dx/shaders/SE.hlsl has one per channel count.");

  ShaderConstants constants{};
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

  command_list->SetComputeRootSignature(root_signature_.Get());
  command_list->SetPipelineState(pso_.Get());

  command_list->SetComputeRoot32BitConstants(
      kRootParamConstants, kNum32BitConstants, &constants, 0);

  // Structured-buffer root SRVs bind by raw GPU virtual address, and the
  // root signature always declares all four of qkv/q/k/v regardless of
  // which mode is active -- the unused three in either mode still need
  // *some* valid VA bound (a null one would fault), so the used buffer(s)
  // fill in for the unused slots rather than leaving them null.
  ID3D12Resource* const q_slot = fused ? qkv : q;
  ID3D12Resource* const k_slot = fused ? qkv : k;
  ID3D12Resource* const v_slot = fused ? qkv : v;
  command_list->SetComputeRootShaderResourceView(
      kRootParamSrvBase + 0, q_slot->GetGPUVirtualAddress());
  command_list->SetComputeRootShaderResourceView(
      kRootParamSrvBase + 1, q_slot->GetGPUVirtualAddress());
  command_list->SetComputeRootShaderResourceView(
      kRootParamSrvBase + 2, k_slot->GetGPUVirtualAddress());
  command_list->SetComputeRootShaderResourceView(
      kRootParamSrvBase + 3, v_slot->GetGPUVirtualAddress());
  command_list->SetComputeRootShaderResourceView(
      kRootParamSrvBase + 4, raw_decay->GetGPUVirtualAddress());
  command_list->SetComputeRootShaderResourceView(
      kRootParamSrvBase + 5, dt_bias->GetGPUVirtualAddress());
  command_list->SetComputeRootShaderResourceView(
      kRootParamSrvBase + 6, a_log->GetGPUVirtualAddress());
  command_list->SetComputeRootShaderResourceView(
      kRootParamSrvBase + 7, beta->GetGPUVirtualAddress());
  command_list->SetComputeRootUnorderedAccessView(
      kRootParamUav, mixed_out->GetGPUVirtualAddress());

  // One thread group per (batch, head); BLOCK_SIZE=32 threads each.
  const UINT group_count = params.batch_size * params.heads;
  command_list->Dispatch(group_count, 1, 1);
}

}  // namespace directml_backend
}  // namespace lczero
