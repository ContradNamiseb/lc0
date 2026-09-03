// Input preprocessing for the attention body: NHWC conversion plus
// positional-encoding concat, ported line-for-line from the SYCL backend's
// preprocess_for_attention_body_kernel (sycl/common_kernels.dp.cpp) so the
// two backends agree on layout down to the element.
//
// One thread group covers one (sample, square) pair; group id = n * 64 + sq.
// Channels are walked by the group's threads in strides of 64.
//
// The input planes buffer is always float32 (the host-side expansion writes
// floats); everything downstream of this kernel is INPUT_TYPE, so encoding
// and output buffers use INPUT_TYPE.
//
// Modes (root constant "mode"):
//   0 (PE_MAP):   out[n, sq, c] = c < input_size
//                          ? in_nchw[n, c, sq]
//                          : encoding[sq * encoding_size + (c - input_size)]
//                 -- the classic 112-plane + 64-channel positional encoding.
//   1 (PE_DENSE):  out[n, sq, c] = c < input_size
//                          ? in_nchw[n, c, sq]
//                          : pre_gemm[n * enc_batch_stride + sq * encoding_size
//                                     + (c - input_size)]
//                 -- concat of the dense-embedding preprocess gemm output,
//                 laid out [N, 64, dense_size].
//   2 (DENSE_POS_INFO): pos_info[n, sq * 12 + c] = in_nchw[n, c, sq] for
//                 c < 12 -- the flattened 12-channel slice the preprocess
//                 gemm consumes. encoding_size must be 12 here and the
//                 output width per sample is 64 * 12.

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

cbuffer PreprocessConstants : register(b0) {
  uint mode;
  uint input_size;
  uint encoding_size;
  uint total_channels;    // input_size + encoding_size (modes 0/1)
  uint enc_batch_stride;  // only used by mode 1
};

StructuredBuffer<float> input_buf : register(t0);
StructuredBuffer<INPUT_TYPE> encoding_buf : register(t1);
RWStructuredBuffer<INPUT_TYPE> output_buf : register(u0);

[numthreads(64, 1, 1)]
void AttentionPreprocess(uint3 tid : SV_GroupThreadID,
                         uint3 gtid : SV_GroupID) {
  const uint n = gtid.x / 64;
  const uint sq = gtid.x % 64;

  if (mode == 2) {
    // 12 channels per square; threads 0..11 do the work.
    for (uint c = tid.x; c < 12; c += 64) {
      output_buf[n * 768 + sq * 12 + c] =
          (INPUT_TYPE)input_buf[n * input_size * 64 + c * 64 + sq];
    }
    return;
  }

  for (uint c = tid.x; c < total_channels; c += 64) {
    INPUT_TYPE value;
    if (c < input_size) {
      value = (INPUT_TYPE)input_buf[n * input_size * 64 + c * 64 + sq];
    } else {
      const uint ch = c - input_size;
      if (mode == 1) {
        value = encoding_buf[n * enc_batch_stride + sq * encoding_size + ch];
      } else {
        value = encoding_buf[64 * sq + ch];
      }
    }
    output_buf[n * 64 * total_channels + sq * total_channels + c] = value;
  }
}
