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
  dequantization, bias, activation, and optional downstream requantization
  (e.g., `FFN1 + ReLU² + Quantize`) in registers.
- Transformer encoders fuse `LayerNorm + Skip + Quantize` in a single
  register-resident pass, run register-resident `Softmax64`, and use dynamic
  W8A8 (`DynamicInt8Weights`) for `Smolgen`, `ip_emb_ffn`, and
  `AttentionPolicyHead` (with parallel `Q × K^T`).
- Residual towers run SmoothQuant-109 INT8 direct-NHWC `in_conv` (scaling
  `rule50` plane 109 on non-hectoplies nets), direct 9-tap 10x10 padded
  L1-resident convolutions (`8×32` and `8×48` leaf microkernels), fused
  `conv1` → `conv2` requantization and global average pooling, fast reciprocal
  Mish/Swish/Sigmoid, and dynamic W8A8 direct-NHWC 3×3 conv policy heads
  (`DynamicConv3x3Int8`).
- Multi-threading (`threads > 1`) parallelizes whole layers across boards and
  row tiles via a lightweight spin-then-park thread pool.

When VNNI is unavailable, MKL builds fall back to `cblas_gemm_s8u8s32`.
Use `--backend-opts=int8_kernel=auto|vnni|avx512|avx|mkl` to select a kernel
explicitly, or `--backend-opts=int8=false` to run the dequantized FP32 weights.

## Benchmarks and match results

Measured on an Intel Core i5-1135G7 (4C/8T, AVX-512 VNNI) using the quantized
nets from the [`quantized-nets`](https://github.com/ContradNamiseb/lc0/tree/quantized-nets)
branch (`backendbench --backend=blas --backend-opts="threads=4"
--start-batch-size=16 --max-batch-size=16`):

| Net | `097a16be` VNNI (`t=4, B=16`) | `7b1cc65e` VNNI (`t=4, B=16`) | Gain over `097a16be` |
| --- | ---: | ---: | ---: |
| T1 (`256×10`) | 314 nps (50.95 ms) | 490 nps (32.65 ms) | +56.1% |
| T79 (`15×192`, Mish) | 539 nps (29.70 ms) | 610 nps (26.22 ms) | +13.2% |
| T74 (`10×128`, ReLU) | 1,394 nps (11.48 ms) | 1,838 nps (8.70 ms, peak 2,849) | +31.8% |

Four 100-game FastChess matches (50 opening pairs, swapped colors,
`UHO_2024_8mvs_+085_+094.epd`, `concurrency=1`, `0` forfeits) comparing
`T74_blas_int8` (`7b1cc65e`, `t74-744021-sycl-int8.pb.gz`, `Backend=blas`,
`threads=8`, `Threads=1`, `MinibatchSize=16`) against clean `upstream/master`
(`fd41ee6b`, ISPC + MKL AVX-512, `744021.pb.gz` and `791556.pb.gz`,
`Backend=blas`, `threads=4`, `Threads=1`, `MinibatchSize=32`; resign
adjudication `500` cp on the first match and `900` cp on the other three):

| Match (100 games each) | Control | W / L / D (`T74_blas_int8`) | Score | Elo (±95%) | Ptnml (0–2) | Median NPS (INT8 vs FP32) |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `T74_blas_int8` vs `T74_master_fp32` | `30+0.5` | 80 / 0 / 20 | 90.0% | +381.70 ± 73.89 | `[0, 0, 1, 18, 31]` | 2,046 vs 419 (4.88×) |
| `T74_blas_int8` vs `T74_master_fp32` | 1,000 nodes | 31 / 25 / 44 | 53.0% | +20.87 ± 34.48 | `[0, 9, 27, 13, 1]` | 2,147 vs 484 (4.44×) |
| `T74_blas_int8` vs `T79_master_fp32` | `30+0.5` | 81 / 0 / 19 | 90.5% | +391.57 ± 70.75 | `[0, 0, 0, 19, 31]` | 2,075 vs 138 (15.0×) |
| `T74_blas_int8` vs `T79_master_fp32` | 1,000 nodes | 5 / 51 / 44 | 27.0% | −172.78 ± 44.16 | `[9, 30, 9, 2, 0]` | 2,196 vs 175 (12.5×) |

At equal 1,000 nodes/move, `T74_blas_int8` is within noise of `T74_master_fp32`
(+20.9 ± 34.5 Elo, with the slight positive edge likely aided by
`MinibatchSize=16` vs `32`), confirming negligible quantization Elo loss per
node while converting the 4.9× speedup into +381.7 Elo under `30+0.5`. Against
`T79` FP32, `T74_blas_int8`'s 15× NPS advantage overcomes a −172.8 Elo per-node
deficit to win by +391.6 Elo (81W–0L–19D) under `30+0.5`.

Against an all-INT8 ONNX T74 (`744204_int8_5.pb.gz`, `onnx-cpu`), `T74_blas_int8`
(pre-`7b1cc65e`) won an 18-game `30+0.5` match 6W–2L–10D (~+79 Elo, 0 lost
pairs). On 12 test positions vs FP32 `744021`, `T74_blas_int8` (`7b1cc65e`)
achieves `mean|dQ| = 0.0121`, `max|dQ| = 0.0434`, `meanKL(P) = 0.0184`, and
12/12 legal Top-1 agreement, outperforming a rebuilt SmoothQuant-109 static ONNX
model of matching accuracy (`mean|dQ| = 0.0114`, `max|dQ| = 0.0468`) at 1,838 vs
1,619 nps (`t=4, B=16`).

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


