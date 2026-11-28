# Quantized lc0 nets (W8A8 INT8)

INT8 versions of three lc0 nets, stored in lc0's own `.pb.gz` format, for the
SYCL backend on Intel GPUs. Weights are 8-bit integers and the inputs of each
quantized layer are 8-bit too (W8A8), so the heavy matrix work runs on the
GPU's integer dot-product units (DP4A) instead of in FP16. The nets were not
retrained: they come from post-training quantization, calibrated on 1,024
positions from real lc0 games.

**Engine:** these files need the SYCL INT8 build of lc0 from the
[`feature/sycl-int8`](https://github.com/ContradNamiseb/lc0/tree/feature/sycl-int8)
branch. Stock lc0 can't read them, because they use a new weight encoding
(`INT8 = 5`) and new fields in `net.proto`.

## The nets

| File | Made from | Architecture | Size |
| --- | --- | --- | --- |
| [`nets/t1-256x10-sycl-int8.pb.gz`](nets/t1-256x10-sycl-int8.pb.gz) | `t1-256x10-distilled-swa-2432500` | transformer, 10 encoders × 256 | 37.1 → 29.5 MB |
| [`nets/t74-744021-sycl-int8.pb.gz`](nets/t74-744021-sycl-int8.pb.gz) | `744021` (T74) | SE-ResNet, 10 blocks × 128, ReLU | 6.4 → 3.5 MB |
| [`nets/t79-791556-sycl-int8.pb.gz`](nets/t79-791556-sycl-int8.pb.gz) | `791556` (T79) | SE-ResNet, 15 blocks × 192, Mish | 18.6 → 9.1 MB |

[`nets/SHA256SUMS`](nets/SHA256SUMS) lists the checksums of these files and of
the source nets they were made from.

Only the trunk is quantized: the transformer's attention and feed-forward
projections, or the ResNets' residual 3×3 convolutions. These carry most of the
compute. The input layer, attention scores, smolgen, squeeze-and-excitation,
layer norms and all output heads stay in floating point.

## Running them

Build `feature/sycl-int8` with the SYCL Level Zero backend and oneDNN (oneDNN
is needed for the two ResNets; see [the docs](docs/quantization.md#building-the-engine)),
then:

```
lc0 --weights=nets/t74-744021-sycl-int8.pb.gz --backend=sycl-int8
```

The log line `Using SYCL W8A8 INT8 ...` confirms the INT8 path is active. The
same files also run in FP16 on that build, using their dequantized weights:

```
lc0 --weights=nets/t74-744021-sycl-int8.pb.gz --backend=sycl-fp16
```

## Speed and accuracy

Measured on an Intel Core i5-1135G7 with Iris Xe graphics (80 EUs), comparing
`sycl-int8` on these nets with `sycl-fp16` on the original nets, both on the
`feature/sycl-int8` build. Nodes per second in `backendbench`, by batch size:

| Batch | T1 | T74 | T79 |
| --- | --- | --- | --- |
| 16 | 390 → 510 (+31%) | 2,653 → 3,692 (+39%) | 1,184 → 1,375 (+16%) |
| 30 | 417 → 512 (+23%) | 4,486 → 4,783 (+7%) | 1,669 → 1,408 (−16%) |
| 80 | 371 → 482 (+30%) | 4,799 → 4,890 (+2%) | 1,655 → 1,434 (−13%) |
| 160 | 358 → 477 (+33%) | 4,646 → 5,137 (+11%) | 1,750 → 1,598 (−9%) |
| 240 | 409 → 491 (+20%) | 4,000 → 5,199 (+30%) | 1,597 → 1,442 (−10%) |

Each figure is the median of three rounds of 100 batches, from 2026-11-25. The
FP16 and INT8 runs alternated, so the two numbers in one cell are fair to
compare. Different batch sizes were measured at different times, and the
laptop's speed varies with its temperature, so comparing across rows is less
reliable: T74's three rounds at batch 160 alone gave +6% to +26%. An earlier
run on 2026-11-23 measured +20%, +26% and −7% at batch 160, and +36%, +15% and
−3% in real search at minibatch 160.

The ResNets' INT8 convolutions round every batch above 16 up to a multiple of
32, so batch 30 runs as 32, batch 80 as 96 and batch 240 as 256, with the extra
entries wasted. The transformer (T1) has no such rounding.

T1 and T74 are faster in INT8 at every batch size, T74 by the most at batch 16.
T79 is faster only at batch 16, and 9% to 16% slower from batch 30 up: on this
GPU, the FP16 Winograd convolutions it replaces are about as fast as INT8, and
its Mish activations need extra zero-point work.

The quantized models pick the same best move as the original FP32 nets on
93.0% (T1), 96.5% (T74) and 95.7% (T79) of 256 positions not used for
calibration. That check ran in OpenVINO on the GPU, using exactly the same int8
weights and scales as the SYCL nets. No game match has been played on the GPU
with the SYCL INT8 nets yet (see the CPU FastChess match results below).

### On the CPU

The [`feature/blas-int8`](https://github.com/ContradNamiseb/lc0/tree/feature/blas-int8)
branch runs the same files in INT8 on the CPU with `--backend=blas` or
`--backend=eigen`. On x86-64 CPUs with AVX-512 VNNI or AVX-VNNI it uses custom
runtime-dispatched VNNI kernels by default (no MKL required), and falls back to
MKL's `cblas_gemm_s8u8s32` when VNNI is unavailable (selectable with
`--backend-opts=int8_kernel=auto|vnni|avx512|avx|mkl`).

**AVX-512 VNNI (`int8_kernel=auto` / `avx512`):** pre-packed weight panels,
fused epilogues, direct 9-tap NHWC residual-tower convolutions with fused SE
pooling, and whole-layer threading. On the same laptop's i5-1135G7 CPU,
`backendbench` comparing `master` (`fd41ee6b`) FP32 on the original nets against
`feature/blas-int8` (`097a16be`) AVX-512 VNNI INT8 on these files, medians of
three alternating rounds:

| Batch | T1 | T74 | T79 |
| --- | --- | --- | --- |
| 7 | 67 → 144 (+115%) | 367 → 807 (+120%) | 123 → 270 (+120%) |
| 16 | 71 → 151 (+113%) | 347 → 794 (+129%) | 119 → 277 (+133%) |
| 64 | 71 → 150 (+111%) | 322 → 775 (+141%) | 116 → 269 (+132%) |
| 256 | 71 → 147 (+107%) | 323 → 778 (+141%) | 112 → 269 (+140%) |
| 7, `threads=4` | 95 → 188 (+98%) | 396 → 1,054 (+166%) | 158 → 395 (+150%) |
| 16, `threads=4` | 100 → 215 (+115%) | 395 → 1,005 (+154%) | 148 → 379 (+156%) |
| 64, `threads=4` | 109 → 215 (+97%) | 378 → 1,035 (+174%) | 145 → 383 (+164%) |

In real search (`lc0 benchmark --threads=2 --minibatch-size=7 --backend-opts=threads=4`,
3 positions, 4,000 / 1,500 / 1,000 nodes, median of three alternating rounds),
AVX-512 VNNI speeds up `master` FP32 from 86 → 165 nps (+92%) on T1,
412 → 1,087 nps (+164%) on T74, and 129 → 343 nps (+166%) on T79. On 300
held-out positions against `master` FP32, AVX-512 VNNI picks the same best move
on 91.0% (T1), 95.3% (T74), and 96.7% (T79).

**Latest AVX-512 VNNI optimizations (`df0089da` + `7b1cc65e`):** fused
register-resident `LayerNorm + Skip + Quantize`, fused `FFN1 + ReLU² + Quantize`,
register-resident `Softmax64`, dynamic W8A8 for T1 `Smolgen` / `ip_emb_ffn` /
`AttentionPolicyHead` (with parallel `Q × K^T`), SmoothQuant-109 INT8 direct-NHWC
`in_conv`, 9-tap fused `8×32` and `8×48` (`C=192`) leaf VNNI microkernels,
fast reciprocal Mish/Swish/Sigmoid, and dynamic W8A8 direct-NHWC 3×3 conv policy
head (`policy1` + `policy`). On the same i5-1135G7 (`backendbench`, `threads=4`,
batch 16), `7b1cc65e` raises mean throughput over `097a16be` VNNI to:

| Net | `097a16be` VNNI (`t=4, B=16`) | `7b1cc65e` VNNI (`t=4, B=16`) | Gain over `097a16be` |
| --- | ---: | ---: | ---: |
| T1 | 314 nps (50.95 ms) | 490 nps (32.65 ms) | +56.1% |
| T79 | 539 nps (29.70 ms) | 610 nps (26.22 ms) | +13.2% |
| T74 | 1,394 nps (11.48 ms) | 1,838 nps (8.70 ms, peak 2,849) | +31.8% |

**Further AVX-512 VNNI optimizations (`experiment/t79-ablations` over `8cbdce7d`):** eliminated MKL OpenMP spin-wait contention against `Int8ThreadPool` (`mkl_set_num_threads(1)` + local guard), vectorized `SELU` + `192d` dynamic quantization and reused quantized `flow` across attention policy `Q`/`K`, replaced `Exp` with integer bit-shifting + `Mish3x` + pre-transposed SE FC1/FC2 weights, fused `NCHW→NHWC` + `plane109` into `conv_nr=48` `input_conv`, and added multi-threaded AVX-512 FP32 `GemmFp32Avx512` to keep `ValueHead` / `MovesLeft` / classical policy on NHWC without a `NhwcToNchw` roundtrip. Across 3 alternating A/B rounds on the same i5-1135G7 (`backendbench`, `threads=4`, batch 16):

| Net | `8cbdce7d` (`t=4, B=16` mean / med) | Optimized (`t=4, B=16` mean / med / max) | Gain (mean / med) |
| --- | ---: | ---: | ---: |
| T79 | 549 / 532 nps | 848 / 960 / 1,126 nps | +54.5% / +80.6% |
| T74 | 1,653 / 1,545 nps | 2,326 / 2,547 / 3,739 nps | +40.7% / +64.9% |
| T1 | 435 / 425 nps | 617 / 679 / 753 nps | +41.8% / +59.6% |

On top of those T79-focused changes, **T1 Transformer & Smolgen ablations (`A + B + C1`)** in `experiment/t79-ablations` parallelized `ip_emb` / PE prep / input gating / `ValueHead` / `MovesLeft` / policy promotion (**A**), fused L1d-resident `depth=32` `Q·K^T + smolgen -> Softmax64 -> Attn·V` (`MhaAttentionHead32Avx512`, also active on `blas-fp32` within `4.1e-6` of MKL SGEMM) with head-sharded `smolgen_global` (`ForwardPreparedRange`) (**B**), and fused `smol_ln1`/`smol_ln2` + dual `ln2` static/dynamic quantization (**C1**) — while discarding a `6×64` VNNI GEMM microkernel (**C2**) that regressed on Tiger Lake (`t=1 B=16`: `239` vs `253–260` median nps).

In independent 3-round alternating benchmarks (`master` FP32 `fd41ee6b` → `097a16be` → `8cbdce7d` → `1b3540e2` INT8, medians):
- **`backendbench t=1, B=16`:** T74 `337 → 784 → 903 → 926`, T79 `111 → 263 → 279 → 291`, T1 `67 → 145 → 219 → 224` nps.
- **`backendbench t=4, B=16`:** T74 `439 → 1,056 → 1,330 → 1,460`, T79 `165 → 397 → 416 → 554`, T1 `107 → 229 → 326 → 462` nps (`+10%` / `+33%` / `+42%` over `8cbdce7d`).
- **Live search (`--threads=2`, `--minibatch-size=7`, `backend threads=4`, nodes `4000/1500/1000`):** T74 `367 → 981 → 1,218 → 1,445` (`3.9×` `master`), T79 `119 → 320 → 352 → 376` (`3.2×` `master`), T1 `80 → 161 → 224 → 274` nps (`3.4×` `master`).

Accuracy across 300 held-out positions (`097a16be` → `1b3540e2` vs. `master` FP32) shows that load-time and dynamic quantization of the boundary/head layers trades a small accuracy cost relative to trunk-only quantization: T74 `mean|dV| 0.0116 → 0.0134` (`+16%`, Top-1 `95.3% → 94.7%`, KL `0.0019 → 0.0022`), T79 `0.0113 → 0.0120` (`+6%`, Top-1 `96.7% → 96.0%`), and T1 `0.0154 → 0.0157` (`+2%`, Top-1 `91.0% → 91.3%`, `max|dV| 0.131 → 0.201`). On a separate 200-position check vs FP32 `t1-256x10-distilled-swa-2432500.pb.gz`, `1b3540e2` scored `193/200` (`96.5%`) Top-1 (`mean|dQ|=0.0193`), and `threads=4` vs `threads=1` INT8 parity is exact (`0` diff on T74/T79/T1).

#### Playing strength on CPU (FastChess matches)

Four 100-game (50 opening pairs, swapped colors, `UHO_2024_8mvs_+085_+094.epd`,
`concurrency=1`, `0` forfeits) FastChess matches on the same i5-1135G7 (4C/8T)
comparing `T74_blas_int8` (`feature/blas-int8` @ `7b1cc65e`,
`t74-744021-sycl-int8.pb.gz`, `Backend=blas`, `threads=8`, `Threads=1`,
`MinibatchSize=16`) against clean `upstream/master` (`fd41ee6b`, ISPC + MKL
AVX-512, `744021.pb.gz` and `791556.pb.gz`, `Backend=blas`, `threads=4`,
`Threads=1`, `MinibatchSize=32`, match-tuned settings for each engine; resign
adjudication `500` cp on the first match and `900` cp on the other three):

| Match (100 games each) | Control | W / L / D (`T74_blas_int8`) | Score | Elo (±95%) | Ptnml (0–2) | Median NPS (INT8 vs FP32) |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `T74_blas_int8` vs `T74_master_fp32` | `30+0.5` | 80 / 0 / 20 | 90.0% | +381.70 ± 73.89 | `[0, 0, 1, 18, 31]` | 2,046 vs 419 (4.88×)* |
| `T74_blas_int8` vs `T74_master_fp32` | 1,000 nodes | 31 / 25 / 44 | 53.0% | +20.87 ± 34.48 | `[0, 9, 27, 13, 1]` | 2,147 vs 484 (4.44×)* |
| `T74_blas_int8` vs `T79_master_fp32` | `30+0.5` | 81 / 0 / 19 | 90.5% | +391.57 ± 70.75 | `[0, 0, 0, 19, 31]` | 2,075 vs 138 (15.0×)* |
| `T74_blas_int8` vs `T79_master_fp32` | 1,000 nodes | 5 / 51 / 44 | 27.0% | −172.78 ± 44.16 | `[9, 30, 9, 2, 0]` | 2,196 vs 175 (12.5×)* |

*\*Match-setting vs matched-search NPS & batch-size note:* at 1,000 nodes/move, `T74_blas_int8` ran at `MinibatchSize=16` vs `32` for
`T74_master_fp32`, so that per-node comparison is confounded and `+20.9 ± 34.5`
Elo is within noise either way. Likewise, the `4.88×` (`4.44×` at 1k nodes) NPS
ratio reflects those match-tuned settings (`threads=8, MinibatchSize=16` vs
`threads=4, MinibatchSize=32`); under matched search settings (`--threads=2`,
`--minibatch-size=7`, `backend threads=4`), `7b1cc65e`/`8cbdce7d` T74 search is
`3.3×` `master` and `1b3540e2` is `3.9×`. Under `30+0.5`, that speed advantage
converts into +381.7 Elo (80W–0L–20D). Against the larger `T79` FP32 net, `T79`
is +172.8 Elo stronger per node, but `T74_blas_int8`'s match-setting NPS edge
swings the `30+0.5` match to +391.6 Elo (81W–0L–19D).

Against an earlier all-INT8 ONNX T74 (`744204_int8_5.pb.gz`, `onnx-cpu`),
`T74_blas_int8` (even before the `7b1cc65e` conv policy head quantization) scored
`6W–2L–10D` across 18 games at `30+0.5` thanks to keeping SE, residual adds, and
the value head in FP32 and scaling `rule50` plane 109 (SmoothQuant-109): on 12
test positions vs FP32 `744021`, `T74_blas_int8` (`7b1cc65e`) has `mean|dQ| =
0.0121`, `max|dQ| = 0.0434`, `meanKL(P) = 0.0184`, and 12/12 legal Top-1
agreement, whereas a static ONNX model rebuilt with SmoothQuant-109 and FP32
SE/residual-add/value reaches matching accuracy (`mean|dQ| = 0.0114`, `max|dQ| =
0.0468`) at 1,619 nps (`onnx-cpu`, `t=4, B=16`) versus 1,838 nps for `blas-int8`.

**MKL (`int8_kernel=mkl`, `cblas_gemm_s8u8s32`):** earlier baseline using MKL's
integer matrix multiply with one MKL thread (the blas default), FP32 on the
original nets against INT8 on these files, medians of three alternating rounds:

| Batch | T1 | T74 | T79 |
| --- | --- | --- | --- |
| 16 | 69 → 105 (+52%) | 346 → 416 (+20%) | 116 → 162 (+40%) |
| 64 | 69 → 104 (+51%) | 316 → 403 (+28%) | 112 → 156 (+39%) |
| 256 | 69 → 103 (+49%) | 307 → 384 (+25%) | 109 → 153 (+40%) |
| 64, `threads=4` | 107 → 107 (0%) | 438 → 479 (+9%) | 168 → 193 (+15%) |

With more MKL threads the MKL-path gain shrinks, because quantizing the inputs
and `im2col` run on one thread. On 300 held-out positions, MKL INT8 picks the
same best move as the original FP32 net on 91.3% (T1), 95.3% (T74) and 96.0%
(T79).

## How they were made

Two steps, both scripts in [`scripts/`](scripts):

1. [`quantize_lc0_int8.py`](scripts/quantize_lc0_int8.py) exports the net to
   ONNX with `lc0 leela2onnx`, calibrates it on the games in
   [`calibration/`](calibration), and writes a quantized ONNX model
   (QuantizeLinear/DequantizeLinear pairs) plus an OpenVINO IR. Calibration
   uses MSE-optimal clipping, per-channel weight scales, SmoothQuant folded into
   the transformer's feed-forward layers, and fast bias correction.
2. [`pack_lc0_sycl_int8.py`](scripts/pack_lc0_sycl_int8.py) copies the int8
   weights, scales, zero points and corrected biases from that ONNX model into
   the original `.pb.gz`. Every layer that isn't quantized stays byte-for-byte
   the same.

[`docs/quantization.md`](docs/quantization.md) explains each step, the number
format, what is and isn't quantized, the tools and versions used, and the
commands to rebuild the nets. [`docs/runtime-vs-offline-quantization.md`](docs/runtime-vs-offline-quantization.md)
documents why boundary and head layers (`input_conv`, `Smolgen`, `AttentionPolicy`,
`ConvPolicy`) stay unquantized on disk and are quantized at load time / runtime in
`blas-int8`, along with offline-vs-runtime ablation measurements on 256 held-out
positions. [`logs/quantize_log.txt`](logs/quantize_log.txt) is the log of the
calibration run that produced the trunk nets.

## Related branches

- [`feature/sycl-int8`](https://github.com/ContradNamiseb/lc0/tree/feature/sycl-int8):
  the SYCL INT8 engine that runs these nets.
- [`feature/blas-int8`](https://github.com/ContradNamiseb/lc0/tree/feature/blas-int8):
  the blas backend rebuilt from layer classes, with INT8 on the CPU.
- [`feature/openvino-int8-static`](https://github.com/ContradNamiseb/lc0/tree/feature/openvino-int8-static):
  the OpenVINO backend for the INT8 IR from step 1; the quantize script comes
  from there.
