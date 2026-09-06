#pragma once

// Generated verbatim from shaders/policy_finalize.hlsl by concatenating its
// exact bytes into a raw string literal -- do not hand-edit; edit the .hlsl
// file and regenerate. Compiled at runtime with D3DCompile, the same way
// kda_recurrence_shader_source.h is.

namespace lczero {
namespace directml_backend {

inline constexpr char kPolicyFinalizeShaderSource[] = R"HLSL(
// Attention policy finalize: interleaves the 64x64 attention scores with the
// 8x24 promotion logits into the [N, 64*64 + 8*24] rows the attention policy
// map indexes into. Ported from the SYCL backend's promotion_logits_kernel
// (sycl/common_kernels.dp.cpp); the promotion math must stay identical.
//
// One thread group per sample. 4288 elements per sample: the first 4096 copy
// the scores, the remaining 192 (= 8 rows x 24) are promotion logits:
//
//   output[n, 4096 + t] = scores[n, (48 + y) * 64 + (56 + w)]
//                          + dot(keys[n, 56 + w, :], ppo[c, :])
//                          + dot(keys[n, 56 + w, :], ppo[3, :])
//   with t = y * 24 + x, w = x / 3, c = x % 3.
//
// The knight's row (ppo row 3) is folded into the three underpromotions, as
// the trainer does. keys is the wk output of the wqk gemm: [N*64, C] with
// only rows 56..63 of each 64-row block used. ppo is ip4_pol_w: [4, C].

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

cbuffer PolicyFinalizeConstants : register(b0) {
  uint key_width;  // C (policy d_model)
};

StructuredBuffer<INPUT_TYPE> scores_buf : register(t0);
StructuredBuffer<INPUT_TYPE> keys_buf : register(t1);
StructuredBuffer<INPUT_TYPE> ppo_buf : register(t2);
RWStructuredBuffer<INPUT_TYPE> output_buf : register(u0);

#define PROMO_BASE 4096
#define ROW_STRIDE 4288  // 64 * 64 + 8 * 24

[numthreads(64, 1, 1)]
void PolicyFinalize(uint3 tid : SV_GroupThreadID, uint3 gtid : SV_GroupID) {
  const uint n = gtid.x;
  const uint total = PROMO_BASE + 192;
  for (uint flat = tid.x; flat < total; flat += 64) {
    if (flat < PROMO_BASE) {
      output_buf[n * ROW_STRIDE + flat] = scores_buf[n * 4096 + flat];
      continue;
    }
    const uint t = flat - PROMO_BASE;  // 0..191
    const uint y = t / 24;             // promotion "from" rank row 0..7
    const uint x = t % 24;             // file * 3 + piece
    const uint w = x / 3;
    const uint c = x % 3;

    // Two dot products of length C: ppo row c and the knight row 3.
    float s = 0.0f;
    float s_knight = 0.0f;
    const uint key_row = (n * 64 + 56 + w) * key_width;
    for (uint i = 0; i < key_width; ++i) {
      const float k = (float)keys_buf[key_row + i];
      s += k * (float)ppo_buf[c * key_width + i];
      s_knight += k * (float)ppo_buf[3 * key_width + i];
    }

    const float n_promo_logit =
        (float)scores_buf[n * 4096 + (48 + y) * 64 + (56 + w)];
    output_buf[n * ROW_STRIDE + PROMO_BASE + t] =
        (INPUT_TYPE)(n_promo_logit + s + s_knight);
  }
}
)HLSL";

}  // namespace directml_backend
}  // namespace lczero
