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

namespace lczero {
namespace openvino_backend {

// GPU implementation of SEResidualOp, loaded through OpenVINO's GPU
// "SimpleGPU" custom-layer mechanism, same pattern as
// kda_scan_kernel_source.h (a CONFIG_FILE XML + this kernel source, JIT'd
// per net since CHANNELS_/SE_FILTERS_ vary per model and are baked in as
// compiler -D flags).
//
// Work decomposition: one work-group per batch sample, 64 work-items per
// group, each lane owning one board PIXEL across all channels (looping over
// the channel dimension). Every global access is then coalesced -- for a
// fixed channel, the 64 lanes touch 64 consecutive floats.
//
// The previous decomposition (one lane per channel, i.e. local=[1,C,1]) was
// measured ~16% SLOWER than the native ops it replaced: with lanes strided
// 64 floats apart, each memory transaction served a single lane (~64x
// waste) on the three bulk data movements (avgpool read, apply read, apply
// write), and only SE_FILTERS_ of CHANNELS_ lanes were active during FC1.
// The bulk data movement dominates -- SE math is ~100 KFLOP per sample,
// trivially bandwidth-bound at iGPU scale.
//
// Phases (4 barriers total):
//   1. avgpool: coalesced read, per-subgroup reduction into
//      sg_partials[subgroup][channel] (no barrier in the loop), one
//      barrier, then each lane finalizes a few channels' averages.
//   2. FC1: lane j accumulates h1[j] = act(b1[j] + sum_c avg[c]*w1[c][j]);
//      w1 reads are coalesced across lanes.
//   3. FC2: lane c (cycling c += 64) computes gamma[c]/beta[c] from h1;
//      w2 reads coalesced. Sigmoid applied to gamma here.
//   4. apply: out[c][pix] = act(gamma[c]*x[c][pix] + beta[c] + skip[c][pix]),
//      coalesced read of se_input (second read; first was phase 1) and
//      skip_input, coalesced write.
//
// Dispatched with global=[N, 64, 1], local=[1, 64, 1] (BFYX output:
// B=N, F=CHANNELS_, Y=8, X=8).
inline constexpr const char kSEResidualKernelSource[] = R"CLC(
#ifdef FP16_SUPPORTED
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

#ifndef DTYPE
#define DTYPE float
#endif

inline float Activate(float x) {
#ifdef MISH_ACTIVATION
  float softplus = (x > 0.0f ? x : 0.0f) + log1p(exp(-fabs(x)));
  return x * tanh(softplus);
#else
  return x > 0.0f ? x : 0.0f;
#endif
}

__kernel void se_residual_kernel(
    __global const DTYPE* se_input,    // [N,C,8,8]
    __global const DTYPE* skip_input,  // [N,C,8,8]
    __global const DTYPE* w1,          // [C,SE_FILTERS_]
    __global const DTYPE* b1,          // [SE_FILTERS_]
    __global const DTYPE* w2,          // [SE_FILTERS_,2*C]
    __global const DTYPE* b2,          // [2*C]
    __global DTYPE* out) {             // [N,C,8,8]
  const int n = get_global_id(0);
  const int lane = get_global_id(1);   // pixel index, 0..63

  __local float avg[CHANNELS_];
  __local float h1[SE_FILTERS_];
  __local float gamma[CHANNELS_];
  __local float beta[CHANNELS_];
  // Per-subgroup avgpool partials. 64 lanes means at most 8 subgroups even
  // at SIMD8, and SLM size bounds how many work-groups are co-resident per
  // subslice, so do not overprovision. network_openvino.cc bounds CHANNELS_
  // so this fits in local memory.
  __local float sg_partials[8 * CHANNELS_];

  const int sample_base = n * CHANNELS_ * 64;

  // Phase 1: global average pool, coalesced.
#ifdef cl_intel_subgroups
  const int sg_id = get_sub_group_id();
  const int sg_lane = get_sub_group_local_id();
  const int sg_size = get_sub_group_size();
  for (int c = 0; c < CHANNELS_; ++c) {
    float s = (float)se_input[sample_base + c * 64 + lane];
    // Butterfly reduction within the subgroup. Use the intel_-prefixed
    // name: sub_group_shuffle_xor is cl_khr_subgroup_shuffle (OpenCL C
    // 2.0+), but OpenVINO's SimpleGPU path compiles custom layers as CL
    // 1.2, where only the cl_intel_subgroups declarations are guaranteed.
    for (int offset = sg_size / 2; offset > 0; offset >>= 1) {
      s += intel_sub_group_shuffle_xor(s, offset);
    }
    if (sg_lane == 0) sg_partials[sg_id * CHANNELS_ + c] = s;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int num_sg = get_num_sub_groups();
  for (int c = lane; c < CHANNELS_; c += 64) {
    float a = 0.0f;
    for (int g = 0; g < num_sg; ++g) a += sg_partials[g * CHANNELS_ + c];
    avg[c] = a * (1.0f / 64);
  }
#else
  // Fallback without subgroup intrinsics: lane 0 reduces serially. Slow
  // (two barriers per channel) but only hit where cl_intel_subgroups is
  // missing, which the Intel GPU plugin targets never are.
  __local float red[64];
  for (int c = 0; c < CHANNELS_; ++c) {
    red[lane] = (float)se_input[sample_base + c * 64 + lane];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) {
      float s = 0.0f;
      for (int i = 0; i < 64; ++i) s += red[i];
      avg[c] = s * (1.0f / 64);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#endif
  barrier(CLK_LOCAL_MEM_FENCE);

  // Phase 2: FC1 + bias + activation. Coalesced w1 reads across lanes.
  for (int j = lane; j < SE_FILTERS_; j += 64) {
    float acc = (float)b1[j];
    for (int c = 0; c < CHANNELS_; ++c) {
      acc += avg[c] * (float)w1[c * SE_FILTERS_ + j];
    }
    h1[j] = Activate(acc);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Phase 3: FC2 + bias, sigmoid on gamma. Coalesced w2 reads.
  for (int c = lane; c < CHANNELS_; c += 64) {
    float g = (float)b2[c];
    float b = (float)b2[CHANNELS_ + c];
    for (int j = 0; j < SE_FILTERS_; ++j) {
      const float hj = h1[j];
      g += hj * (float)w2[j * 2 * CHANNELS_ + c];
      b += hj * (float)w2[j * 2 * CHANNELS_ + CHANNELS_ + c];
    }
    gamma[c] = 1.0f / (1.0f + exp(-g));
    beta[c] = b;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Phase 4: scale-shift, residual add, activation. All coalesced.
  for (int c = 0; c < CHANNELS_; ++c) {
    const int idx = sample_base + c * 64 + lane;
    const float x = gamma[c] * (float)se_input[idx] + beta[c] +
                    (float)skip_input[idx];
    out[idx] = (DTYPE)Activate(x);
  }
}
)CLC";

}  // namespace openvino_backend
}  // namespace lczero
