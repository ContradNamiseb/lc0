// KDA local 3x3 depthwise board convolution + residual for the DirectML
// backend. Port of the SYCL applyKdaLocalDepthwiseConv: output[token, c] =
// input[token, c] + bias[c] + sum_{kr,kf} w[c, kr*3+kf] *
// input[neighbour_token, c], where the neighbour is the square offset by
// (kr-1, kf-1) on the 8x8 board, clipped at the board edge (zero padding).
//
// Runs as hand-written HLSL: the DML-graph form needs a strided NHWC-style
// view as a grouped-convolution input, which this driver rejects.

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

cbuffer KdaLocalConvConstants : register(b0) {
  uint tokens;  // N * 64
  uint emb;     // embedding width (channels)
};

StructuredBuffer<INPUT_TYPE> input_buf : register(t0);
StructuredBuffer<INPUT_TYPE> weight_buf : register(t1);  // [emb, 9]
StructuredBuffer<INPUT_TYPE> bias_buf : register(t2);    // [emb]
RWStructuredBuffer<INPUT_TYPE> output_buf : register(u0);

[numthreads(64, 1, 1)]
void KdaLocalConv(uint3 tid : SV_GroupThreadID, uint3 gtid : SV_GroupID) {
  const uint total = tokens * emb;
  const uint flat = gtid.x * 64u + tid.x;
  if (flat >= total) return;

  const uint c = flat % emb;
  const uint token = flat / emb;
  const uint batch_base = (token / 64u) * 64u;
  const uint square = token - batch_base;
  const uint rank = square / 8u;
  const uint file = square % 8u;

  float sum = (float)bias_buf[c];
  for (uint tap = 0; tap < 9u; ++tap) {
    const uint kr = tap / 3u;
    const uint kf = tap % 3u;
    const int nr = (int)rank + (int)kr - 1;
    const int nf = (int)file + (int)kf - 1;
    if (nr < 0 || nr > 7 || nf < 0 || nf > 7) continue;
    const uint neighbour = batch_base + (uint)nr * 8u + (uint)nf;
    sum += (float)weight_buf[c * 9u + tap] *
           (float)input_buf[neighbour * emb + c];
  }
  output_buf[flat] =
      (INPUT_TYPE)((float)input_buf[flat] + sum);
}
