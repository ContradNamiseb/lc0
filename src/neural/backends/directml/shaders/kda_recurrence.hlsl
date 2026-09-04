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

// 64-square traversal orders for the two diagonal scan directions. Board
// index is rank*8+file; these must stay byte-for-byte identical to
// kKdaDiagForward/kKdaDiagReverse/kKdaAntiDiagForward/kKdaAntiDiagReverse in
// sycl/common_kernels.dp.cpp.
static const uint kKdaDiagForward[64] = {
    7,  6,  15, 5,  14, 23, 4,  13, 22, 31, 3,  12, 21, 30, 39, 2,
    11, 20, 29, 38, 47, 1,  10, 19, 28, 37, 46, 55, 0,  9,  18, 27,
    36, 45, 54, 63, 8,  17, 26, 35, 44, 53, 62, 16, 25, 34, 43, 52,
    61, 24, 33, 42, 51, 60, 32, 41, 50, 59, 40, 49, 58, 48, 57, 56};
static const uint kKdaDiagReverse[64] = {
    56, 57, 48, 58, 49, 40, 59, 50, 41, 32, 60, 51, 42, 33, 24, 61,
    52, 43, 34, 25, 16, 62, 53, 44, 35, 26, 17, 8,  63, 54, 45, 36,
    27, 18, 9,  0,  55, 46, 37, 28, 19, 10, 1,  47, 38, 29, 20, 11,
    2,  39, 30, 21, 12, 3,  31, 22, 13, 4,  23, 14, 5,  15, 6,  7};
static const uint kKdaAntiDiagForward[64] = {
    0,  1,  8,  2,  9,  16, 3,  10, 17, 24, 4,  11, 18, 25, 32, 5,
    12, 19, 26, 33, 40, 6,  13, 20, 27, 34, 41, 48, 7,  14, 21, 28,
    35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 23, 30, 37, 44, 51,
    58, 31, 38, 45, 52, 59, 39, 46, 53, 60, 47, 54, 61, 55, 62, 63};
static const uint kKdaAntiDiagReverse[64] = {
    63, 62, 55, 61, 54, 47, 60, 53, 46, 39, 59, 52, 45, 38, 31, 58,
    51, 44, 37, 30, 23, 57, 50, 43, 36, 29, 22, 15, 56, 49, 42, 35,
    28, 21, 14, 7,  48, 41, 34, 27, 20, 13, 6,  40, 33, 26, 19, 12,
    5,  32, 25, 18, 11, 4,  24, 17, 10, 3,  16, 9,  2,  8,  1,  0};

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
    uint square = token;
    if (direction == 2) {
      square = 63 - token;
    } else if (direction == 3) {
      square = (token % 8) * 8 + token / 8;
    } else if (direction == 4) {
      uint reverse = 63 - token;
      square = (reverse % 8) * 8 + reverse / 8;
    } else if (direction == 5) {
      square = kKdaDiagForward[token];
    } else if (direction == 6) {
      square = kKdaDiagReverse[token];
    } else if (direction == 7) {
      square = kKdaAntiDiagForward[token];
    } else if (direction == 8) {
      square = kKdaAntiDiagReverse[token];
    }

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
