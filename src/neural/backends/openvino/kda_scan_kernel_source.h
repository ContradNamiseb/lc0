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

// GPU implementation of KdaScanOp, loaded through OpenVINO's GPU
// "SimpleGPU" custom-layer mechanism (a CONFIG_FILE XML + this kernel
// source, generated at runtime by network_openvino.cc since HEADS_/
// KEY_DIM_/VALUE_DIM_ vary per net and are baked in as compiler -D flags).
//
// Embedded as a string rather than shipped as a loose .cl file next to the
// binary because lc0 has no install-time data-file mechanism to reliably
// find it at -- the custom-layer XML's <Source filename=.../> still needs
// an actual file on disk, so network_openvino.cc writes this string out to
// a temp file alongside the generated XML.
//
// Structurally mirrors sycl/common_kernels.dp.cpp's
// kdaRecurrenceValueParallel and must stay bit-for-bit equivalent to it and
// to KdaScanOp::evaluate() (kda_scan_op.cc) -- one work-group per (batch,
// head), VALUE_DIM_ work-items per group, each owning one lane of the
// recurrent state matrix (state[key][lane]) across the 64-step sequential
// scan. Dispatched with global=[N*HEADS_, VALUE_DIM_, 1],
// local=[1, VALUE_DIM_, 1] -- get_global_id(0) is decoded into (n, head)
// here; the decoding just needs to be self-consistent, not match any
// particular host-side enumeration order.
inline constexpr const char kKdaScanKernelSource[] = R"CLC(
__kernel void kda_scan_kernel(__global const float* q,          // [N,64,H,K]
                              __global const float* k,          // [N,64,H,K]
                              __global const float* v,          // [N,64,H,V]
                              __global const float* raw_decay,  // [N,64,H,K]
                              __global const float* beta,       // [N,64,H,1]
                              __global const float* dt_bias,        // [H,K]
                              __global const float* neg_decay_scale,  // [H]
                              __global float* mixed) {          // [N,64,H,V]
  const int gid0 = get_global_id(0);
  const int lane = get_global_id(1);
  const int n = gid0 / HEADS_;
  const int head = gid0 % HEADS_;

  __local float p_q[KEY_DIM_];
  __local float p_k[KEY_DIM_];
  __local float p_decay[KEY_DIM_];

  float state[KEY_DIM_];
  for (int i = 0; i < KEY_DIM_; ++i) state[i] = 0.0f;

  const float scale = 1.0f / sqrt((float)KEY_DIM_);
  const float neg_scale_h = neg_decay_scale[head];

  for (int t = 0; t < 64; ++t) {
    const int qk_base = ((n * 64 + t) * HEADS_ + head) * KEY_DIM_;
    const int v_base = ((n * 64 + t) * HEADS_ + head) * VALUE_DIM_;
    const int beta_idx = (n * 64 + t) * HEADS_ + head;

    for (int i = lane; i < KEY_DIM_; i += VALUE_DIM_) {
      p_q[i] = q[qk_base + i];
      p_k[i] = k[qk_base + i];

      const float decay_input = raw_decay[qk_base + i] + dt_bias[head * KEY_DIM_ + i];
      const float softplus = (decay_input > 0.0f ? decay_input : 0.0f) +
                             log1p(exp(-fabs(decay_input)));
      float log_decay = neg_scale_h * softplus;
      log_decay = log_decay > -10.0f ? log_decay : -10.0f;
      p_decay[i] = exp(log_decay);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
    for (int key = 0; key < KEY_DIM_; ++key) {
      q_norm_sq += p_q[key] * p_q[key];
      k_norm_sq += p_k[key] * p_k[key];
    }
    const float q_norm = 1.0f / sqrt(q_norm_sq > 1e-12f ? q_norm_sq : 1e-12f);
    const float k_norm = 1.0f / sqrt(k_norm_sq > 1e-12f ? k_norm_sq : 1e-12f);

    for (int key = 0; key < KEY_DIM_; ++key) state[key] *= p_decay[key];

    const float beta_value = beta[beta_idx];
    const float update_rate = 1.0f / (1.0f + exp(-beta_value));

    float prediction = 0.0f;
    for (int key = 0; key < KEY_DIM_; ++key) {
      prediction += p_k[key] * k_norm * state[key];
    }

    const float delta = update_rate * (v[v_base + lane] - prediction);

    float out = 0.0f;
    for (int key = 0; key < KEY_DIM_; ++key) {
      state[key] += p_k[key] * k_norm * delta;
      out += p_q[key] * q_norm * scale * state[key];
    }

    mixed[v_base + lane] = out;

    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
)CLC";

}  // namespace openvino_backend
}  // namespace lczero
