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

// Generated verbatim from shaders/kda_recurrence.hlsl by concatenating its
// exact bytes into a raw string literal -- do not hand-edit; edit the .hlsl
// file and regenerate (cat header + hlsl + footer). Embedding the source
// (rather than reading the .hlsl file off disk at runtime, or wiring an
// offline dxc/fxc compile step into meson.build the way dx/shaders/ does)
// keeps this backend's PSO creation self-contained for both dev and
// installed builds; runtime D3DCompile cost is paid once at network load,
// not per inference.

namespace lczero {
namespace directml_backend {

inline constexpr char kKdaRecurrenceShaderSource[] = R"HLSL(
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

// DirectML has no loop/recurrence primitive (no Scan/TensorIterator
// equivalent in DirectMLX's operator set), so the KDA recurrence -- unlike
// every other layer in this backend -- cannot be expressed as a compiled
// dml::Graph. It is instead a hand-written HLSL compute shader dispatched
// directly through D3D12, exactly the way the dx backend already dispatches
// its own custom Winograd/SE kernels (see src/neural/backends/dx/shaders/).
//
// This is a line-for-line port of the SYCL reference kernel
// (kdaRecurrenceValueParallel in src/neural/backends/sycl/common_kernels.dp.cpp,
// fixed to accumulate in float regardless of storage type -- see that file's
// comments for why FP16 belongs only at the buffer boundary, not in the
// 64-step sequential scan itself). Any change to the math here needs the
// matching change made there too, and vice versa -- they are meant to
// produce bit-for-bit-modulo-rounding-identical output.
//
// One thread group per (batch, head); exactly KDA_VALUE_DIM threads per
// group, one per value-dimension lane.
//
// KDA_KEY_DIM and KDA_VALUE_DIM are #defines supplied at PSO creation, not
// cbuffer fields, and that matters for far more than numthreads. Every loop
// in the scan below is bounded by the key dimension, and `state` is a
// per-lane array indexed by those loops. With a runtime bound FXC cannot
// unroll them and must place `state` in an indexable temp -- scratch memory,
// i.e. off-chip -- so each of the four inner loops per token becomes a
// memory round trip instead of register traffic. With compile-time bounds
// the loops unroll and `state` stays in registers. This kernel is ~93% of
// the whole network eval on Iris Xe (memory-bank note 2475), so that
// difference is the difference for the backend as a whole.
//
// One PSO is compiled per (key_dim, value_dim) the loaded net uses;
// KdaRecurrenceLayer takes the geometry in its constructor.
#ifndef KDA_KEY_DIM
#define KDA_KEY_DIM 32
#endif
#ifndef KDA_VALUE_DIM
#define KDA_VALUE_DIM 32
#endif

cbuffer KdaRecurrenceConstants : register(b0) {
  uint N;                 // batch size
  uint heads;              // total head count (across all directions)
  uint key_dim;            // informational: KDA_KEY_DIM is the real bound
  uint value_dim;          // informational: KDA_VALUE_DIM is the real bound
  uint direction_count;    // number of distinct scan directions (1 or 8)
  uint use_fused_qkv;      // 1 if reading from the fused qkv buffer
  uint qkv_stride;         // stride between tokens in the fused qkv buffer
  float log_decay_floor;   // KDA_LOG_DECAY_FLOOR, clamps softplus decay
  // Packed as 4x uint4 so the 16-entry array lands contiguously (no
  // per-element padding) -- a straight 64-byte memcpy from the C++ side's
  // int32_t[16] direction table fills this correctly.
  uint4 directions[4];
};

// Bound as typed buffers so INPUT_TYPE (float or half, set at PSO creation
// via a #define) controls the storage width without a second copy of this
// file -- the same "fp16 only at the memory boundary" split the fixed SYCL
// kernel uses.
#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

// Structured (not typed/formatted) buffers: root-descriptor binding in
// D3D12 only supports raw or structured buffer views, not Buffer<T>/
// RWBuffer<T>'s formatted views, and structured buffers skip needing a
// descriptor heap entirely for a shader with this few resources.
StructuredBuffer<INPUT_TYPE> qkv           : register(t0);
StructuredBuffer<INPUT_TYPE> q_in          : register(t1);
StructuredBuffer<INPUT_TYPE> k_in          : register(t2);
StructuredBuffer<INPUT_TYPE> v_in          : register(t3);
StructuredBuffer<INPUT_TYPE> raw_decay     : register(t4);
StructuredBuffer<INPUT_TYPE> dt_bias       : register(t5);
StructuredBuffer<INPUT_TYPE> a_log         : register(t6);
StructuredBuffer<INPUT_TYPE> beta          : register(t7);
// The 16 x 64 square traversal order, uploaded from the single definition in
// neural/kda_directions.h rather than transcribed here. Transcribing it is
// exactly the hazard that header warns about: a divergence between the C++
// table and a GPU copy is not a compile error, it is silently wrong chess.
// It also used to be a branch chain that only covered directions 1-8, so the
// eight serpentine directions (9-16) fell through to plain rank order and
// produced quietly wrong output for any net trained with them.
StructuredBuffer<uint> direction_order      : register(t8);
RWStructuredBuffer<INPUT_TYPE> mixed       : register(u0);

groupshared float p_q[KDA_KEY_DIM];
groupshared float p_k[KDA_KEY_DIM];
groupshared float p_decay[KDA_KEY_DIM];

[numthreads(KDA_VALUE_DIM, 1, 1)]
void KdaRecurrence(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
  // The group is exactly KDA_VALUE_DIM threads wide, so every lane owns one
  // value dimension and there are no idle lanes to guard -- which is also
  // why the two GroupMemoryBarrierWithGroupSync() calls below are reached
  // uniformly with no early-return hazard (HLSL error X4026). This is the
  // same shape as the SYCL kernel, which dispatches exactly value_dim
  // work-items.
  const uint local_id = tid.x;

  const uint global_batch_head = gid.x;
  const uint batch = global_batch_head / heads;
  const uint head = global_batch_head % heads;

  const uint direction_index = head / (heads / direction_count);
  const uint direction = directions[direction_index / 4][direction_index % 4];
  // Clamped like neural/kda_directions.h's KdaSquareForToken; the range is
  // validated at network load, so this never actually clamps.
  const uint direction_base = (min(max(direction, 1u), 16u) - 1u) * 64u;

  const float scale = 1.0 / sqrt((float)KDA_KEY_DIM);
  const float decay_scale = exp((float)a_log[head]);
  const uint key_depth = heads * KDA_KEY_DIM;
  const uint value_depth = heads * KDA_VALUE_DIM;

  float state[KDA_KEY_DIM];
  [unroll]
  for (uint i = 0; i < KDA_KEY_DIM; ++i) {
    state[i] = 0.0;
  }

  for (uint token = 0; token < 64; ++token) {
    const uint square = direction_order[direction_base + token];

    const uint token_idx = batch * 64 + square;
    uint q_off, k_off, v_off;
    if (use_fused_qkv != 0) {
      q_off = token_idx * qkv_stride + head * KDA_KEY_DIM;
      k_off = token_idx * qkv_stride + key_depth + head * KDA_KEY_DIM;
      v_off = token_idx * qkv_stride + 2 * key_depth + head * KDA_VALUE_DIM;
    } else {
      q_off = token_idx * key_depth + head * KDA_KEY_DIM;
      k_off = token_idx * key_depth + head * KDA_KEY_DIM;
      v_off = token_idx * value_depth + head * KDA_VALUE_DIM;
    }

    const uint raw_decay_offset = token_idx * key_depth + head * KDA_KEY_DIM;
    const uint value_offset = token_idx * value_depth + head * KDA_VALUE_DIM;

    // Strided write into groupshared memory: mirrors the SYCL kernel's
    // "value lanes covering the key entries" loop.
    [unroll]
    for (uint i = local_id; i < KDA_KEY_DIM; i += KDA_VALUE_DIM) {
      p_q[i] = (float)(use_fused_qkv != 0 ? qkv[q_off + i] : q_in[q_off + i]);
      p_k[i] = (float)(use_fused_qkv != 0 ? qkv[k_off + i] : k_in[k_off + i]);

      const float decay_input =
          (float)raw_decay[raw_decay_offset + i] +
          (float)dt_bias[head * KDA_KEY_DIM + i];
      const float softplus =
          max(decay_input, 0.0) + log(1.0 + exp(-abs(decay_input)));
      const float log_decay = max(-decay_scale * softplus, log_decay_floor);
      p_decay[i] = exp(log_decay);
    }

    GroupMemoryBarrierWithGroupSync();

    float q_norm_sq = 0.0;
    float k_norm_sq = 0.0;
    [unroll]
    for (uint key = 0; key < KDA_KEY_DIM; ++key) {
      q_norm_sq += p_q[key] * p_q[key];
      k_norm_sq += p_k[key] * p_k[key];
    }
    const float q_norm = 1.0 / sqrt(max(q_norm_sq, 1.0e-12));
    const float k_norm = 1.0 / sqrt(max(k_norm_sq, 1.0e-12));

    [unroll]
    for (uint key2 = 0; key2 < KDA_KEY_DIM; ++key2) {
      state[key2] *= p_decay[key2];
    }

    const float beta_value = (float)beta[token_idx * heads + head];
    const float update_rate = 1.0 / (1.0 + exp(-beta_value));

    float prediction = 0.0;
    [unroll]
    for (uint key3 = 0; key3 < KDA_KEY_DIM; ++key3) {
      prediction += p_k[key3] * k_norm * state[key3];
    }

    const float v_value =
        (float)(use_fused_qkv != 0 ? qkv[v_off + local_id]
                                   : v_in[v_off + local_id]);
    const float delta = update_rate * (v_value - prediction);

    float output = 0.0;
    [unroll]
    for (uint key4 = 0; key4 < KDA_KEY_DIM; ++key4) {
      state[key4] += p_k[key4] * k_norm * delta;
      output += p_q[key4] * q_norm * scale * state[key4];
    }

    mixed[value_offset + local_id] = (INPUT_TYPE)output;

    // Same barrier, same reason as the SYCL kernel: p_q/p_k/p_decay are read
    // by every thread above but only rewritten by the staging lanes at
    // the top of the next iteration -- without this, a fast thread can
    // loop back and clobber them before a slow thread finishes reading.
    GroupMemoryBarrierWithGroupSync();
  }
}
)HLSL";

}  // namespace directml_backend
}  // namespace lczero
