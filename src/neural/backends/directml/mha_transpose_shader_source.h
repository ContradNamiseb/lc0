#pragma once

// Generated verbatim from shaders/mha_transpose.hlsl by concatenating its
// exact bytes into a raw string literal -- do not hand-edit; edit the .hlsl
// file and regenerate. Compiled at runtime with D3DCompile, the same way
// kda_recurrence_shader_source.h is.

namespace lczero {
namespace directml_backend {

inline constexpr char kMhaTransposeShaderSource[] = R"HLSL(
// Head-split/merge transpose for the DirectML backend's MHA path.
//
// DirectML has no transpose operator, and strided ReinterpretView inputs to
// batched GEMMs with H>1 fail to create on this driver (Intel Iris Xe,
// system DirectML 1.x: m_device->CreateOperator returns E_INVALIDARG for a
// [N,H]-batched GEMM with strided inputs, while the identical [N,1]-batched
// form compiles). So head (de)interleaving runs here as a plain copy kernel,
// and all MHA GEMMs stay dense with batch [N*H, 1] (the shape already proven
// by the policy head's scores GEMM).
//
// mode 0 (split): out[(n*H+h)*64*D + s*D + d] = in[(n*64+s)*HD + h*D + d]
//   in:  [T, H*D] token-major dense (a q/k/v projection);
//   out: [B=N*H, 64, D] dense batch-major.
// mode 1 (merge): out[(n*64+s)*HD + h*D + d] = in[(n*H+h)*64*D + s*D + d]
//   in:  [B, 64, D] dense (attention context);
//   out: [T, H*D] token-major dense (dense-projection input).
//
// One thread per output element; total = N*H*64*D in both modes.

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

cbuffer TransposeConstants : register(b0) {
  uint mode;
  uint batch_size;  // N
  uint heads;       // H
  uint head_dim;    // D
};

StructuredBuffer<INPUT_TYPE> in_buf : register(t0);
RWStructuredBuffer<INPUT_TYPE> out_buf : register(u0);

[numthreads(64, 1, 1)]
void MhaTranspose(uint3 tid : SV_GroupThreadID,
                  uint3 gtid : SV_GroupID) {
  const uint total = batch_size * heads * 64u * head_dim;
  const uint groups = (total + 63u) / 64u;
  if (gtid.x >= groups) return;
  const uint flat = gtid.x * 64u + tid.x;
  if (flat >= total) return;

  if (mode == 0) {
    // out flat = ((n*H + h)*64 + s)*D + d.
    const uint d = flat % head_dim;
    const uint s = (flat / head_dim) % 64u;
    const uint h = (flat / (head_dim * 64u)) % heads;
    const uint n = flat / (head_dim * 64u * heads);
    out_buf[flat] = in_buf[(n * 64u + s) * (heads * head_dim) +
                           h * head_dim + d];
  } else {
    // out flat = token-major (t, hd): t = n*64+s, hd = h*D+d.
    const uint HD = heads * head_dim;
    const uint hd = flat % HD;
    const uint t = flat / HD;
    const uint h = hd / head_dim;
    const uint d = hd % head_dim;
    const uint n = t / 64u;
    const uint s = t % 64u;
    out_buf[flat] = in_buf[((n * heads + h) * 64u + s) * head_dim + d];
  }
}
)HLSL";

}  // namespace directml_backend
}  // namespace lczero
