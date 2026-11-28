The files in this directory comprise the BLAS backend of Lc0.

## Structure

As in the CUDA and SYCL backends, the network is a list of layers built once
from the weights (`layers.h`): `ConvLayer`, `ResidualBlock`, `AttentionBody`
with its `EncoderBlock`s, `AttentionPolicyHead`, `PolicyMapLayer`,
`ValueHead`, `DenseLayer` and `EmbeddingLayer`. A computation runs them in
order over three tensors and a workspace that the network pools and reuses.

## BLAS INT8 (`feature/blas-int8` / `experiment/t79-ablations`)

This backend adds W8A8 INT8 CPU inference (encoding `INT8` in `net.proto`) to
both the BLAS and Eigen backends for quantized nets from the
[`quantized-nets`](https://github.com/ContradNamiseb/lc0/tree/quantized-nets)
branch (`t1-256x10-sycl-int8.pb.gz`, `t74-744021-sycl-int8.pb.gz`,
`t79-791556-sycl-int8.pb.gz`).

On x86-64 CPUs with AVX-512 VNNI or AVX-VNNI, custom runtime-dispatched VNNI
intrinsic kernels (`int8_vnni_avx512.cc`, `int8_vnni_avx.cc`) execute quantized
layers for both `blas` and `eigen` backends without requiring MKL:

- Backend selection in UCI / `--backend`: `blas-auto` (default, auto-selects
  `blas-int8` for INT8 nets and `blas-fp32` for FP32 nets or non-INT8 hosts),
  `blas-int8` (permissive: warns and falls back to FP32 if the net/host lacks
  INT8), `blas-fp32` (forces dequantized FP32), and `blas` (alias for
  `blas-auto`), plus matching `eigen-auto`, `eigen-int8`, `eigen-fp32`, and
  `eigen` options. `--backend-opts=int8=false` also remains accepted and forces
  dequantized FP32 execution.
- Linear layers pre-pack weights into 64-byte aligned panels and fuse
  dequantization, bias, activation, and optional downstream requantization
  (e.g., `FFN1 + ReLU² + Quantize`) in registers, while pre-transposed
  multi-threaded AVX-512 FP32 `GemmFp32Avx512` accelerates `ValueHead`,
  `MovesLeft`, and classical/attention policy tails without MKL OpenMP
  spin-wait contention (`MklLocalThreadsGuard` pins MKL local calls to 1
  thread inside `BlasNetwork::Forward()` when `Int8ThreadPool` is active).
- Transformer encoders fuse `LayerNorm + Skip + Quantize` (including dual
  static QKV + dynamic `smol_compress` quantization on `ln2`) in one register
  pass, fuse `Smolgen` `LayerNorm + DynamicQuantize` (`smol_ln1`/`smol_ln2`),
  shard dynamic `smolgen_global` per head (`ForwardPreparedRange`), fuse
  L1d-resident `depth=32` `Q·K^T + smolgen -> Softmax64 -> Attn·V`
  (`MhaAttentionHead32Avx512`, which also runs on AVX-512 FP32 paths within
  `~4e-6` of MKL SGEMM) and parallel input prep/gating, and use vectorized
  `SELU` + `192d` quant + shared `Q`/`K` input + AVX-512 promotion offsets in
  `AttentionPolicyHead`.
- Residual towers run fused `NCHW->NHWC` + SmoothQuant-109 INT8 `input_conv`
  (`conv_nr=48` on 192 channels), direct 9-tap 10x10 padded L1-resident
  convolutions (`8x32` and `8x48` leaf microkernels), fused `conv1 -> conv2`
  requantization and global average pooling, IEEE-754 bit-shift `Exp` +
  interleaved `Mish3x` + pre-transposed SE `ComputeSeFcAvx512`, dynamic W8A8
  direct-NHWC 3x3 conv policy heads (`DynamicConv3x3Int8`), and end-to-end NHWC
  head execution avoiding `NhwcToNchw` roundtrips.
- Multi-threading (`threads > 1`) parallelizes whole layers across boards,
  heads, and row tiles via a spin-then-park thread pool (`Int8ThreadPool`).

When VNNI is unavailable, MKL builds fall back to `cblas_gemm_s8u8s32` (while
`eigen-int8` requires VNNI). Select `--backend=blas-fp32` (or
`--backend-opts=int8=false`) to run dequantized FP32 weights, or use
`--backend-opts=int8_kernel=auto|vnni|avx512|avx|mkl` to override the INT8
kernel explicitly.

### Building `blas-int8`

On `x86_64`, enabling `-Dblas=true` (the default) automatically probes compiler
support for AVX-512 VNNI (`-DLC0_HAS_AVX512_VNNI`) and AVX-VNNI
(`-DLC0_HAS_AVX_VNNI`). Only those two translation units are compiled with VNNI
flags; dispatch happens at runtime via `cpuid` and `xgetbv`, so one binary runs
safely on both VNNI and non-VNNI CPUs.

Installing [ISPC](https://ispc.github.io/) (`-Dispc=true`, on by default when
`ispc` is on `PATH`) is recommended for fast FP32 Winograd, LayerNorm, and
activation helpers. Pass `-Dispc_native_only=false` if distributing to other
x86-64 microarchitectures.

#### 1. Windows (MSVC + Intel oneAPI MKL + ISPC -- recommended)

From an x64 Native Tools Command Prompt (or after `vcvars64.bat`) with `ispc`
and oneAPI on `PATH`:

```cmd
meson setup build-blas ^
  -Dbuildtype=release -Db_lto=false -Ddefault_library=static -Db_vscrt=md ^
  -Dblas=true -Dmkl=true -Dopenblas=false -Ddnnl=false -Donednn=false ^
  -Ddx=false -Dcudnn=false -Dplain_cuda=false -Dopencl=false -Dsycl=off ^
  -Dgtest=true ^
  "-Dmkl_include=C:\Program Files (x86)\Intel\oneAPI\mkl\latest\include" ^
  "-Dmkl_libdirs=C:\Program Files (x86)\Intel\oneAPI\mkl\latest\lib"
ninja -C build-blas lc0.exe
```

To run `lc0.exe` standalone without oneAPI on `PATH`, place the MKL runtime DLLs
(`mkl_rt.*.dll`, `mkl_core.*.dll`, `mkl_intel_thread.*.dll`, `mkl_avx512.*.dll`,
`mkl_avx2.*.dll`, `mkl_def.*.dll`) and `libiomp5md.dll` next to `lc0.exe`.

#### 2. Standalone VNNI without MKL/OpenBLAS (`eigen` backend)

```shell
meson setup build-eigen-int8 \
  -Dbuildtype=release -Dblas=true \
  -Dmkl=false -Dopenblas=false -Ddnnl=false -Daccelerate=false \
  -Dplain_cuda=false -Dcudnn=false -Dopencl=false -Ddx=false -Dsycl=off -Donednn=false
ninja -C build-eigen-int8 lc0
```

Run with `--backend=eigen`.

#### 3. Linux (GCC/Clang + OpenBLAS or Intel oneAPI MKL + ISPC)

With OpenBLAS (`libopenblas-dev` + `ispc`):

```shell
meson setup build-blas \
  -Dbuildtype=release -Dblas=true -Dopenblas=true -Dmkl=false -Ddnnl=false \
  -Dplain_cuda=false -Dcudnn=false -Dopencl=false -Dsycl=off -Donednn=false
ninja -C build-blas lc0
```

With Intel oneAPI MKL:

```shell
meson setup build-blas \
  -Dbuildtype=release -Dblas=true -Dmkl=true -Dopenblas=false -Ddnnl=false \
  -Dplain_cuda=false -Dcudnn=false -Dopencl=false -Dsycl=off -Donednn=false \
  -Dmkl_include=/opt/intel/oneapi/mkl/latest/include \
  -Dmkl_libdirs=/opt/intel/oneapi/mkl/latest/lib
ninja -C build-blas lc0
```

#### Verifying

```shell
ninja -C build-blas blas_int8_vnni_test
meson test -C build-blas BlasInt8VnniTest
```

### Benchmarks, parity, and match results

All measurements below were taken on an Intel Core i5-1135G7 (4C/8T Tiger Lake,
AVX-512 VNNI). Because sustained multi-thread runs on a 15W laptop throttle from
PL2 to PL1, alternating 3-round medians and matched A/B benchmarks are the most
reliable comparisons.

#### Independent alternating 3-round medians (`master` FP32 `fd41ee6b` -> `097a16be` -> `8cbdce7d` -> `1b3540e2`)

| Workload | `master` FP32 | `097a16be` | `8cbdce7d` | `1b3540e2` INT8 | `1b3540e2` `blas-fp32` |
| --- | ---: | ---: | ---: | ---: | ---: |
| T74 `backendbench t=1 B=16` | 337 | 784 | 903 | **926** | 341 |
| T74 `backendbench t=4 B=16` | 439 | 1,056 | 1,330 | **1,460** | 466 |
| T74 search (`--threads=2 B=7 backend t=4`) | 367 | 981 | 1,218 | **1,445** | 372 |
| T79 `backendbench t=1 B=16` | 111 | 263 | 279 | **291** | 112 |
| T79 `backendbench t=4 B=16` | 165 | 397 | 416 | **554** | 168 |
| T79 search (`--threads=2 B=7 backend t=4`) | 119 | 320 | 352 | **376** | 115 |
| T1 `backendbench t=1 B=16` | 67 | 145 | 219 | **224** | 67 |
| T1 `backendbench t=4 B=16` | 107 | 229 | 326 | **462** | 126 |
| T1 search (`--threads=2 B=7 backend t=4`) | 80 | 161 | 224 | **274** | 85 |

*(A `6x64` VNNI GEMM microkernel ablation for T1 regressed due to register
spills on Tiger Lake's single VNNI port -- `t=1 B=16` dropped `253-260` ->
`239` median nps -- and was reverted.)*

#### Parity & accuracy vs. `master` FP32

- **FP32 path check (`blas-fp32` vs. `master`):** T74 is exact (`0`); T79 max
  policy diff is `2.4e-7`; T1 max diff is `3.0e-6` policy / `4.1e-6` value due
  to the fused `depth=32` AVX-512 MHA kernel (`MhaAttentionHead32Avx512`) also
  running on `blas-fp32`.
- **INT8 thread consistency:** `threads=4` vs. `threads=1` error is `0` on
  T74, T79, and T1.
- **Accuracy across 300 held-out positions (`097a16be` -> `1b3540e2` vs.
  `master` FP32):** load-time/dynamic quantization of boundary/head layers
  (`input_conv`, `Smolgen`, policy heads) trades a small accuracy cost relative
  to trunk-only quantization:
  - T74: `mean|dV| 0.0116 -> 0.0134` (`+16%`), Top-1 `95.3% -> 94.7%`, KL
    `0.0019 -> 0.0022`
  - T79: `mean|dV| 0.0113 -> 0.0120` (`+6%`), Top-1 `96.7% -> 96.0%`
  - T1: `mean|dV| 0.0154 -> 0.0157` (`+2%`), Top-1 `91.0% -> 91.3%`,
    `max|dV| 0.131 -> 0.201`

#### FastChess match results (`T74_blas_int8` @ `7b1cc65e` vs. `upstream/master` @ `fd41ee6b`)

Four 100-game FastChess matches (50 opening pairs, swapped colors,
`UHO_2024_8mvs_+085_+094.epd`, `concurrency=1`, `0` forfeits). Engine settings:
`T74_blas_int8` (`t74-744021-sycl-int8.pb.gz`, `Backend=blas`, `threads=8`,
`Threads=1`, `MinibatchSize=16`) vs. `upstream/master` (`744021.pb.gz` and
`791556.pb.gz`, `Backend=blas`, `threads=4`, `Threads=1`, `MinibatchSize=32`).
Resign adjudication was `500` cp on the first match and `900` cp on the other
three:

| Match (100 games each) | Control | W / L / D (`T74_blas_int8`) | Score | Elo (±95%) | Ptnml (0-2) | Median NPS (INT8 vs FP32) |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `T74_blas_int8` vs `T74_master_fp32` | `30+0.5` | 80 / 0 / 20 | 90.0% | +381.70 ± 73.89 | `[0, 0, 1, 18, 31]` | 2,046 vs 419 (4.88x)* |
| `T74_blas_int8` vs `T74_master_fp32` | 1,000 nodes | 31 / 25 / 44 | 53.0% | +20.87 ± 34.48 | `[0, 9, 27, 13, 1]` | 2,147 vs 484 (4.44x)* |
| `T74_blas_int8` vs `T79_master_fp32` | `30+0.5` | 81 / 0 / 19 | 90.5% | +391.57 ± 70.75 | `[0, 0, 0, 19, 31]` | 2,075 vs 138 (15.0x)* |
| `T74_blas_int8` vs `T79_master_fp32` | 1,000 nodes | 5 / 51 / 44 | 27.0% | -172.78 ± 44.16 | `[9, 30, 9, 2, 0]` | 2,196 vs 175 (12.5x)* |

*\*Note: the `4.88x` (`4.44x` at 1k nodes) ratio compares each engine at its
match settings (`MinibatchSize=16, threads=8` vs `MinibatchSize=32, threads=4`);
under matched search settings (`MinibatchSize=7, --threads=2, backend threads=4`),
`7b1cc65e`/`8cbdce7d` T74 search is `3.3x` `master` and `1b3540e2` is `3.9x`.
The equal-node (`1,000` nodes) match also compares `MinibatchSize=16` vs `32`, so
`+20.9 ± 34.5` Elo is within noise with that batch-size difference.*

Against an earlier all-INT8 ONNX T74 (`744204_int8_5.pb.gz`, `onnx-cpu`),
pre-`7b1cc65e` `T74_blas_int8` scored `6W-2L-10D` across 18 games at `30+0.5`.

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
