#pragma once

// Generated verbatim from shaders/smolgen_bias.hlsl by concatenating its
// exact bytes into a raw string literal -- do not hand-edit; edit the .hlsl
// file and regenerate. Compiled at runtime with D3DCompile, the same way
// kda_recurrence_shader_source.h is.

namespace lczero {
namespace directml_backend {

inline constexpr char kSmolgenBiasShaderSource[] = R"HLSL(
// Smolgen generated-bias matmul for the DirectML backend's MHA path.
//
// Computes bias[n, h, i] = sum_g table[h, i, g] * d2[n, h, g] where:
//   table -- the shared smolgen weight table, [H, 4096, gen] (natural);
//   d2    -- the smolgen MLP output for this batch, [N, H*gen] (natural,
//            per-head columns h*gen .. (h+1)*gen-1);
//   bias  -- the per-(batch, head) attention bias, [N, H*4096] (viewed as
//            [N, H, 64, 64] and added to the attention logits before
//            softmax).
//
// This runs as hand-written HLSL (like mha_transpose.hlsl) because the
// equivalent DML graph needs a 5-D broadcast GEMM operand that this driver
// rejects. One thread per output element; gen is a small inner loop.

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

cbuffer SmolgenBiasConstants : register(b0) {
  uint batch;  // N
  uint heads;  // H
  uint gen;    // per-head generated-vector width
};

StructuredBuffer<INPUT_TYPE> table_buf : register(t0);
StructuredBuffer<INPUT_TYPE> d2_buf : register(t1);
RWStructuredBuffer<INPUT_TYPE> bias_buf : register(u0);

[numthreads(64, 1, 1)]
void SmolgenBias(uint3 tid : SV_GroupThreadID, uint3 gtid : SV_GroupID) {
  const uint total = batch * heads * 4096u;
  const uint flat = gtid.x * 64u + tid.x;
  if (flat >= total) return;

  const uint i = flat % 4096u;
  const uint h = (flat / 4096u) % heads;
  const uint n = flat / (4096u * heads);

  // The smolgen weight table is ONE SHARED [4096, gen] matrix (the SYCL
  // backend's smol_global gemm passes the same A for every (n, h) batch
  // entry) -- head specificity comes from d2's per-head columns, not from
  // the table.
  const uint trow = i * gen;
  const uint drow = n * heads * gen + h * gen;
  float s = 0.0f;
  for (uint g = 0; g < gen; ++g) {
    s += (float)table_buf[trow + g] * (float)d2_buf[drow + g];
  }
  bias_buf[flat] = (INPUT_TYPE)s;
}
)HLSL";

}  // namespace directml_backend
}  // namespace lczero
