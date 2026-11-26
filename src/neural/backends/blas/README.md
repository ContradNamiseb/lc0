The files in this directory comprise the BLAS backend of Lc0.

## Structure

As in the CUDA and SYCL backends, the network is a list of layers built once
from the weights (`layers.h`): `ConvLayer`, `ResidualBlock`, `AttentionBody`
with its `EncoderBlock`s, `AttentionPolicyHead`, `PolicyMapLayer`,
`ValueHead`, `DenseLayer` and `EmbeddingLayer`. A computation runs them in
order over three tensors and a workspace that the network pools and reuses.

## INT8 nets

Nets whose layers are stored as W8A8 INT8 (encoding `INT8` in `net.proto`)
run those layers in INT8: the residual blocks' 3x3 convolutions, and the Q, K,
V, output and feed-forward projections of the attention body.

On x86-64 CPUs with AVX-512 VNNI or AVX-VNNI, custom runtime-dispatched VNNI
intrinsic kernels (`int8_vnni_avx512.cc`, `int8_vnni_avx.cc`) execute these
layers for both the `blas` and `eigen` backends without requiring MKL:
- Linear layers pre-pack weights into 64-byte aligned panels and fuse
  dequantization, bias, and activation in registers.
- Residual towers convert once to NHWC and run a direct 9-tap 10x10 padded
  L1-resident convolution, fusing `conv1` epilogue directly into `conv2`'s
  quantized input pad and fusing global average pooling into `conv2`.
- Multi-threading (`threads > 1`) parallelizes whole layers across boards and
  row tiles via a lightweight spin-then-park thread pool.

When VNNI is unavailable, MKL builds fall back to `cblas_gemm_s8u8s32`.
Use `--backend-opts=int8_kernel=auto|vnni|avx512|avx|mkl` to select a kernel
explicitly, or `--backend-opts=int8=false` to run the dequantized FP32 weights.

## License

Leela Chess is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Leela Chess is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

**The source files of this directory are not covered by any additional
permission.**


