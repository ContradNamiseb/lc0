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
// One work-group per batch sample; CHANNELS_ work-items per group, each
// lane owning one output channel end-to-end (avgpool -> its share of FC1 ->
// its own FC2 outputs -> the final per-pixel apply over all 64 board
// squares). Dispatched with global=[N, CHANNELS_, 1], local=[1, CHANNELS_,
// 1] (BFYX: B=N, F=CHANNELS_, Y=8, X=8).
inline constexpr const char kSEResidualKernelSource[] = R"CLC(
#ifdef FP16_SUPPORTED
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
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
  const int c = get_global_id(1);

  __local float avg[CHANNELS_];
  __local float h1[SE_FILTERS_];

  const int plane = n * CHANNELS_ * 64 + c * 64;

  float sum = 0.0f;
  for (int i = 0; i < 64; ++i) sum += (float)se_input[plane + i];
  avg[c] = sum / 64.0f;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int j = c; j < SE_FILTERS_; j += CHANNELS_) {
    float acc = (float)b1[j];
    for (int ci = 0; ci < CHANNELS_; ++ci) {
      acc += avg[ci] * (float)w1[ci * SE_FILTERS_ + j];
    }
    h1[j] = Activate(acc);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  float gamma_pre = (float)b2[c];
  float beta = (float)b2[CHANNELS_ + c];
  for (int j = 0; j < SE_FILTERS_; ++j) {
    const float hj = h1[j];
    gamma_pre += hj * (float)w2[j * 2 * CHANNELS_ + c];
    beta += hj * (float)w2[j * 2 * CHANNELS_ + CHANNELS_ + c];
  }
  const float gamma = 1.0f / (1.0f + exp(-gamma_pre));

  for (int i = 0; i < 64; ++i) {
    const float x = gamma * (float)se_input[plane + i] + beta +
                    (float)skip_input[plane + i];
    out[plane + i] = (DTYPE)Activate(x);
  }
}
)CLC";

}  // namespace openvino_backend
}  // namespace lczero
