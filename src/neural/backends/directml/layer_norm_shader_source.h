#pragma once

// Generated verbatim from shaders/layer_norm.hlsl by concatenating its exact
// bytes into a raw string literal -- do not hand-edit; edit the .hlsl file
// and regenerate. Compiled at runtime with D3DCompile, the same way
// smolgen_bias_shader_source.h is.

namespace lczero {
namespace directml_backend {

inline constexpr char kLayerNormShaderSource[] = R"HLSL(
// Fused layer normalization for the DirectML backend, a direct port of the
// SYCL backend's layer_norm_kernel (common_kernels.dp.cpp):
//
//   t = act(input + bias) * alpha + skip
//   y = gamma * (t - mean) / sqrt(var + eps) + beta
//
// with mean/var taken over the channel dimension of each [row, channels]
// token row (population variance, matching DML's REDUCE_FUNCTION_AVERAGE).
//
// Why this is hand-written HLSL rather than a dml::Graph subexpression: the
// composed form needs ~10 DML nodes (bias add, activation, alpha scale, skip
// add, two Reduces, two broadcast views, Sqrt, Recip, two Mul, Add), and
// DirectML executes every node as its own dispatch with a barrier between,
// each one streaming the whole [tokens, channels] tensor through memory. One
// encoder's two LayerNorms therefore cost ~20 dispatches and ~16 full-tensor
// round trips for what is a single pass of arithmetic. This kernel does it in
// one dispatch and (for the common case) one read of the input.
//
// One thread group per token row; the row is cached in groupshared memory so
// passes 2 and 3 do not re-read global memory. Rows wider than
// LN_MAX_CHANNELS fall back to recomputing t from global memory, which is the
// same arithmetic -- the branch is uniform across the group.

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

#define LN_GROUP_SIZE 64
// 2048 floats = 8KB of the 64KB shared local memory, so eight groups per
// subslice still fit. Embedding widths in shipped nets are 128..1024.
#define LN_MAX_CHANNELS 2048

#define LN_FLAG_HAS_BIAS 1u
#define LN_FLAG_HAS_SKIP 2u

// Mirrors lczero::ActivationFunction (neural/tables/activation_function.h).
#define LN_ACT_DEFAULT 0u
#define LN_ACT_MISH    1u
#define LN_ACT_RELU    2u
#define LN_ACT_NONE    3u
#define LN_ACT_TANH    4u
#define LN_ACT_SIGMOID 5u
#define LN_ACT_SELU    6u
#define LN_ACT_SWISH   7u
#define LN_ACT_RELU_2  8u

cbuffer LayerNormConstants : register(b0) {
  uint rows;        // token count, N * 64
  uint channels;    // C, normalized over
  uint flags;       // LN_FLAG_*
  uint activation;  // LN_ACT_*
  float alpha;      // skip-connection scale
  float eps;        // variance epsilon
  uint pad0;
  uint pad1;
};

StructuredBuffer<INPUT_TYPE> input_buf : register(t0);
// bias/skip are always bound to a valid resource; when the corresponding
// LN_FLAG_ bit is clear they are aliases of input_buf and never read. A null
// root SRV removes the device on this driver (see the PE_DENSE preprocess
// crash in docs/directml-handoff.md), so absence is a flag, not a null.
StructuredBuffer<INPUT_TYPE> bias_buf : register(t1);
StructuredBuffer<INPUT_TYPE> skip_buf : register(t2);
StructuredBuffer<INPUT_TYPE> gamma_buf : register(t3);
StructuredBuffer<INPUT_TYPE> beta_buf : register(t4);
RWStructuredBuffer<INPUT_TYPE> output_buf : register(u0);

groupshared float ln_row[LN_MAX_CHANNELS];
groupshared float ln_partial[LN_GROUP_SIZE];

float ln_activate(float x, uint act) {
  switch (act) {
    case LN_ACT_RELU:
      return max(x, 0.0f);
    case LN_ACT_MISH: {
      // x * tanh(softplus(x)), the composition activate() uses; the
      // large-x guard keeps exp() from overflowing where softplus is
      // already linear.
      float sp = x > 20.0f ? x : log(1.0f + exp(x));
      return x * tanh(sp);
    }
    case LN_ACT_TANH:
      return tanh(x);
    case LN_ACT_SIGMOID:
      return 1.0f / (1.0f + exp(-x));
    case LN_ACT_SELU: {
      const float kAlpha = 1.67326324f;
      const float kScale = 1.05070098f;
      return x > 0.0f ? kScale * x : kScale * kAlpha * (exp(x) - 1.0f);
    }
    case LN_ACT_SWISH:
      return x / (1.0f + exp(-x));
    case LN_ACT_RELU_2: {
      float r = max(x, 0.0f);
      return r * r;
    }
    default:  // LN_ACT_NONE, LN_ACT_DEFAULT
      return x;
  }
}

// t as defined above, recomputed from global memory (the > LN_MAX_CHANNELS
// path).
float ln_compute(uint base, uint c) {
  float v = (float)input_buf[base + c];
  if ((flags & LN_FLAG_HAS_BIAS) != 0u) v += (float)bias_buf[c];
  v = ln_activate(v, activation) * alpha;
  if ((flags & LN_FLAG_HAS_SKIP) != 0u) v += (float)skip_buf[base + c];
  return v;
}

float ln_group_sum(float v, uint lane) {
  ln_partial[lane] = v;
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint s = LN_GROUP_SIZE / 2u; s > 0u; s >>= 1u) {
    if (lane < s) ln_partial[lane] += ln_partial[lane + s];
    GroupMemoryBarrierWithGroupSync();
  }
  float total = ln_partial[0];
  // The caller reuses ln_partial for the next reduction, so every lane must
  // be done reading before any lane writes again.
  GroupMemoryBarrierWithGroupSync();
  return total;
}

[numthreads(LN_GROUP_SIZE, 1, 1)]
void LayerNorm(uint3 gtid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
  const uint row = gtid.x;
  // Whole-group condition, so the barriers below are still uniformly
  // reached by every thread that survives.
  if (row >= rows) return;

  const uint lane = tid.x;
  const uint base = row * channels;
  const bool cached = channels <= LN_MAX_CHANNELS;

  // 1. mean of t
  float s = 0.0f;
  for (uint c = lane; c < channels; c += LN_GROUP_SIZE) {
    float v = ln_compute(base, c);
    if (cached) ln_row[c] = v;
    s += v;
  }
  if (cached) GroupMemoryBarrierWithGroupSync();
  s = ln_group_sum(s, lane);
  const float mean = s / (float)channels;

  // 2. variance of t
  float sq = 0.0f;
  for (uint c2 = lane; c2 < channels; c2 += LN_GROUP_SIZE) {
    float d = (cached ? ln_row[c2] : ln_compute(base, c2)) - mean;
    sq += d * d;
  }
  sq = ln_group_sum(sq, lane);
  // 1/sqrt rather than rsqrt(): rsqrt is an approximation on some drivers and
  // this backend is held to a 2e-4 parity bar against BLAS.
  const float inv_std = 1.0f / sqrt(sq / (float)channels + eps);

  // 3. normalize, scale, shift
  for (uint c3 = lane; c3 < channels; c3 += LN_GROUP_SIZE) {
    float t = cached ? ln_row[c3] : ln_compute(base, c3);
    float y = (t - mean) * inv_std * (float)gamma_buf[c3] + (float)beta_buf[c3];
    output_buf[base + c3] = (INPUT_TYPE)y;
  }
}
)HLSL";

}  // namespace directml_backend
}  // namespace lczero
