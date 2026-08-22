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

// Mirrors sycl/layers.h: one class per layer type, each owning its compiled
// pipeline state and exposing a single dispatch entry point that appends
// work to a caller-supplied command list -- the DirectML/D3D12 analog of
// SYCL's BaseLayer<DataType>::Eval(N, output, input, ...). Layers built from
// DirectMLX operators are compiled once into a dml::CompiledOperator at
// construction the same way KdaRecurrenceLayer compiles its HLSL PSO once;
// neither allocates or recompiles per call.
//
// STATUS as of this writing: only KdaRecurrenceLayer is implemented and
// tested (see src/neural/backends/directml/test_kda_recurrence.cc). The
// remaining layer types below are declared so the intended structure is
// visible and the orchestration file (network_directml.cc) has something
// concrete to call, but their bodies are not yet written -- see the
// "directml native backend" finding in the agent-memory-bank for the current
// state and why (KDA's recurrence has no DirectML graph primitive, so it had
// to come first and does not generalize to unblock the rest for free).

#include <array>
#include <cstdint>
#include <vector>

#include <d3d12.h>
#include <wrl/client.h>

namespace lczero {
namespace directml_backend {

using Microsoft::WRL::ComPtr;

// One compiled compute PSO + root signature for the KDA recurrence shader
// (src/neural/backends/directml/shaders/kda_recurrence.hlsl), plus the one
// constant buffer it needs. Construction compiles and creates the pipeline;
// Dispatch() only records commands, no allocation.
class KdaRecurrenceLayer {
 public:
  struct Params {
    uint32_t batch_size;
    uint32_t heads;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t direction_count;
    // directions[i] in {1..8}; must be exactly 16 entries to match the
    // shader's cbuffer layout (uint4[4]) even when direction_count < 16.
    std::array<int32_t, 16> directions;
    bool use_fused_qkv;
    uint32_t qkv_stride;
    float log_decay_floor;
    bool fp16;  // selects INPUT_TYPE in the shader compile.
  };

  KdaRecurrenceLayer(ID3D12Device* device, bool fp16);

  // Binds the given buffers (qkv may be null if use_fused_qkv is false, in
  // which case q/k/v must all be set, and vice versa), uploads Params via an
  // internal upload buffer, and records one Dispatch call. The caller owns
  // command_list state (open, in a recordable state) and any resource
  // barriers needed before/after -- this only appends the compute work.
  void Record(ID3D12GraphicsCommandList* command_list, const Params& params,
              ID3D12Resource* qkv, ID3D12Resource* q, ID3D12Resource* k,
              ID3D12Resource* v, ID3D12Resource* raw_decay,
              ID3D12Resource* dt_bias, ID3D12Resource* a_log,
              ID3D12Resource* beta, ID3D12Resource* mixed_out);

 private:
  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12RootSignature> root_signature_;
  ComPtr<ID3D12PipelineState> pso_;
  bool fp16_;
};

// TODO(directml backend): EmbeddingLayer, MhaBlockLayer, SmolgenLayer,
// PolicyHeadLayer, ValueHeadLayer, MovesLeftHeadLayer -- each building a
// dml::Graph fragment and exposing a Compile()/Record() pair analogous to
// KdaRecurrenceLayer's constructor/Record() split. Not started: see the
// agent-memory-bank note on this backend's status for why KDA came first.

}  // namespace directml_backend
}  // namespace lczero
