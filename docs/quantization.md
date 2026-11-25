# How the nets were quantized

This page explains how the three nets in [`nets/`](../nets) were made, what
each step does, and how to rebuild them. The work was done on 2026-11-23 on an
Intel Core i5-1135G7 laptop with Iris Xe graphics (80 EUs).

## Overview

```
original .pb.gz ──lc0 leela2onnx──> FP32 ONNX
                                       │  quantize_lc0_int8.py
                                       │  (calibration on 1,024 game positions)
                                       ▼
                          quantized ONNX (QDQ) + OpenVINO IR
                                       │  pack_lc0_sycl_int8.py
original .pb.gz ──────────────────────►│
                                       ▼
                        INT8 .pb.gz  ──>  lc0 --backend=sycl-int8
```

The method is post-training quantization with static scales: no retraining,
and every scale and zero point is fixed ahead of time from calibration data.
The quantization itself happens in ONNX, where it can be measured against the
FP32 model. The packing step then moves the result into lc0's own weight file,
so the SYCL backend can read it directly.

## Number format (W8A8)

For every quantized layer:

- **Weights:** signed 8-bit integers, symmetric (zero point 0), with one scale
  per output channel. Each channel's scale is searched for the lowest mean
  squared error rather than taken from the largest weight, so a few outliers
  don't waste the range.
- **Activations (the layer's input):** unsigned 8-bit integers with one scale
  and one zero point per tensor (asymmetric, `0..255`). The clipping range is
  also chosen by an MSE search over percentiles of the calibration data, with
  0.0 always exactly representable. In the ReLU nets every input is
  non-negative, so the zero point comes out as 0. T79's Mish activations go
  slightly negative, so its zero points are between 5 and 18.
- **Arithmetic:** int8 × uint8 products are summed in int32, then multiplied
  back by the two scales and the bias is added in floating point.

## What is quantized

| Part | T1 (transformer) | T74 / T79 (SE-ResNet) |
| --- | --- | --- |
| Quantized to INT8 | Q, K, V and output projections, and both feed-forward layers, in all 10 encoders (60 layers) | both 3×3 convolutions of every residual block (20 layers in T74, 30 in T79) |
| Kept in floating point | input embedding and gating, attention scores (QKᵀ, softmax, AV), smolgen, layer norms, residual adds | input convolution, squeeze-and-excitation, residual adds |
| Output heads | policy, value (WDL) and moves-left heads stay in floating point | same |

The quantized layers carry well over 85% of each net's multiply-adds. The
parts left in floating point are small but sensitive. Quantizing them too
(squeeze-and-excitation, smolgen) was tried as the "naive" variant in the log,
and it was less accurate.

## Calibration data

The calibration positions come from two PGN files in
[`calibration/`](../calibration), both games played by lc0 on the same laptop:

| File | Games | Share used |
| --- | --- | --- |
| `sprt_dagfix_vs12a3852_local.pgn` | 474 games of a 3-minute SPRT between two lc0 builds, UHO openings | 80% |
| `endgame_935532_vs_825532.pgn` | 41 endgame games between two nets | 20% |

From each file the script reads the first 400 games of 16 moves or more, and
splits them by game: the first 75% for calibration, the last 25% for
evaluation, so no game contributes to both. It samples plies from across each
game, the opening and the endgame included, and keeps the 8 previous
positions. It then encodes each one exactly as lc0 does (the classical
112-plane input with history). Before calibrating, it checks this encoding
against lc0 itself. The original run checked one position and the WDL output
matched to 0.0011. The script here checks three positions (white and black to
move, a rule-50 count, no castling, a repetition), all within 0.002.

The result is 1,024 calibration positions and 256 held-out evaluation
positions (819 + 205 and 204 + 52 from the two files). Sampling uses a fixed
seed, so the same files give the same positions.

## Calibration steps

[`scripts/quantize_lc0_int8.py`](../scripts/quantize_lc0_int8.py) does this
for each net:

1. **Export.** `lc0 leela2onnx --onnx-opset=17 --onnx-data-type=f32` writes the
   FP32 ONNX model. The lc0 used was `v0.33.0-dev+git.91c9d8b`.
2. **Choose the layers** listed above, from the ONNX node names.
3. **SmoothQuant, T1 only.** T1's feed-forward block uses a squared ReLU, which
   makes the input of the second feed-forward layer extremely uneven between
   channels (kurtosis up to about 500). The script moves part of that
   unevenness into the weights (α = 0.5). It divides the first layer's weights
   and bias by √s per channel and multiplies the second layer's weights by s.
   Because `relu(x/√s)² = relu(x)²/s`, the FP32 net computes exactly the same
   thing, with no extra operations at run time. The same trick hurt the two
   ResNets, so it is used only on T1.
4. **Activation ranges.** The calibration positions are run through the model.
   For every quantized layer's input, the clipping range is picked by an MSE
   search over percentiles of about 260,000 sampled values.
5. **Weight scales.** Per-channel symmetric int8, with MSE-searched clipping.
6. **Insert QDQ nodes.** Each quantized layer gets QuantizeLinear and
   DequantizeLinear on its input, and its weights are stored as int8 with
   DequantizeLinear. That is the standard ONNX way to describe a statically
   quantized model.
7. **Fast bias correction.** Rounding shifts the average output of each
   quantized layer slightly, and in a deep net these shifts add up. Going layer
   by layer from the input, the script measures each output channel's mean on
   256 calibration positions, in the FP32 model and in the quantized model so
   far. It then adds the difference to that layer's bias, clamped to two
   standard deviations of the channel. It's the same idea as NNCF's
   FastBiasCorrection, but written directly in the script; NNCF itself was not
   used. It costs nothing at run time.
8. **Evaluate and save.** The script compares variants on the 256 held-out
   positions against the FP32 model. It keeps the more accurate of
   "MSE + bias correction" and "MSE only" (all three nets picked the first),
   and saves it as `<net>_int8.onnx` and as OpenVINO IR (`.xml`/`.bin`).

## Accuracy

From [`logs/quantize_log.txt`](../logs/quantize_log.txt): the chosen quantized
model compared with the FP32 net on the 256 held-out positions, run in OpenVINO
on the Iris Xe GPU.

| Net | Same best move | Policy KL | Mean \|ΔQ\| | Max \|ΔQ\| |
| --- | --- | --- | --- | --- |
| T1 | 93.0% | 0.0055 | 0.014 | 0.089 |
| T74 | 96.5% | 0.0030 | 0.009 | 0.199 |
| T79 | 95.7% | 0.0029 | 0.011 | 0.226 |

The log also shows each step's effect. For example, T74's mean |ΔQ| goes from
0.0133 with plain min/max ranges to 0.0099 with MSE ranges, and to 0.0091 after
bias correction.

On a 4-position check with the SYCL INT8 engine, all three nets chose the same
move as their quantized ONNX model. The largest |ΔQ| was 0.0029 for T1, 0.0102
for T74 and 0.0053 for T79. Against the original FP16 net, T79 picked a
different first move from the starting position (`d2d4` instead of `e2e4`).

## Packing into lc0's `.pb.gz`

[`scripts/pack_lc0_sycl_int8.py`](../scripts/pack_lc0_sycl_int8.py) reads the
int8 weights, per-channel scales, activation scales, zero points and corrected
biases from the quantized ONNX model. It writes them into a copy of the original
`.pb.gz`. It edits the protobuf directly (no `protoc` needed) and leaves every
other layer byte-for-byte unchanged. The new `net.proto` fields, defined on
`feature/sycl-int8`, are:

- `Weights.Layer.Encoding.INT8 = 5`: raw signed bytes, one per weight.
- `Weights.MHA` fields 13–20: per-channel weight scales for Q, K, V and the
  output projection, and the activation scale and zero point of the QKV input
  and the output-projection input.
- `Weights.FFN` fields 5–10: the same for the two feed-forward layers.
- `Weights.ConvBlock` fields 7–9: per-channel weight scales, and the input
  activation scale and zero point.

The file-level `weights_encoding` stays `LINEAR16`, because only some layers
are INT8, so `lc0 describenet` reports these nets exactly like their FP
originals. Stock lc0 can run `describenet` on them, but stops with
`Unknown layer encoding Encoding(5)` when it loads them for search.

In the quantized convolution blocks, batch norm is already folded into the
weights and biases, so the batch-norm fields are left empty. The loader
rejects an INT8 block that still has them, so batch norm can't be applied
twice.

All three nets in `nets/` were checked: running the pack script from
`feature/sycl-int8` on the source nets and the quantized ONNX models gives
byte-identical weights. The `.pb.gz` files differ only in gzip metadata.

## Running on SYCL

On `feature/sycl-int8`, `--backend=sycl-int8` picks the engine by net type:

- **T1 (transformer):** the projections run as oneMKL `gemm_bias`
  (int8 weights × uint8 activations → int32), which reached about 5 TOPS on the
  Iris Xe. Small SYCL kernels quantize the inputs and convert the results back
  to FP16. Attention, smolgen and layer norms stay FP16.
- **T74 / T79 (SE-ResNet):** each residual block runs as two oneDNN direct INT8
  convolutions. The input is reordered once into oneDNN's blocked layout, and
  the second convolution writes FP16 straight back. A fused FP16 kernel then
  does squeeze-and-excitation, the skip add and the activation. Weights are
  packed once at load time. Batch sizes are rounded up to a few fixed sizes
  (1, 4, 8, 16, then multiples of 32) that are compiled ahead of time, so
  oneDNN doesn't compile a new kernel during search.

`--backend=sycl-fp16 --backend-opts=int8=false` runs the same file in FP16
from dequantized weights. Multi-stream is switched off while INT8 is active.

### Building the engine

The ResNet path needs oneDNN. The nets were measured with a build configured
like this (Windows, Visual Studio 2026, Intel oneAPI 2026.1, `CC=cl`,
`CXX=icx`):

```
meson setup build --buildtype release -Db_lto=false -Dsycl=l0 -Dblas=true -Dmkl=false -Ddnnl=false -Donednn=false -Ddefault_library=static -Db_vscrt=md -Ddnnl_dir="C:\Program Files (x86)\Intel\oneAPI\dnnl\latest" -Dmkl_include=... -Dmkl_libdirs=... -Dopencl_include=... -Dopencl_libdirs=...
```

`meson setup` prints `Enabling oneDNN SYCL INT8 residual convolutions` when
oneDNN is found. Without it, the transformer still runs in INT8, but the
ResNets fall back to FP16.

## Tools used

- **lc0** `leela2onnx` (`v0.33.0-dev+git.91c9d8b`), to export the nets to ONNX.
- **Python 3.11.15** with:
  - **ONNX 1.14.1** and **ONNX Runtime 1.29.0**, for calibration, editing the
    QDQ graph, and accuracy checks on the CPU;
  - **NumPy 1.26.4**;
  - **python-chess 1.11.2**, for reading the PGNs and building the move
    histories;
  - **protobuf 4.25.9**.
- **OpenVINO 2026.3.0**, to write the IR and to check accuracy on the GPU and
  the CPU.
- **Intel oneAPI 2026.1**: the `icx` SYCL compiler, Level Zero, **oneMKL**
  (INT8 GEMM for the transformer) and **oneDNN** (INT8 convolutions for the
  ResNets).

## Rebuilding the nets

Requirements:

- Python 3.11 with the packages above;
- an lc0 binary with `leela2onnx`;
- lc0's `src/neural/encoder.cc`, which the script reads lc0's policy move
  table from;
- an Intel GPU for the OpenVINO accuracy check.

The run is deterministic: sampling and calibration use fixed seeds. With the
versions above, step 1 for T74 rewrites a byte-identical
`t74_744021_int8.onnx` (checked 2026-11-25; the SHA-256 is in
`nets/SHA256SUMS`). A different CPU or ONNX Runtime version can change the
multi-threaded sums in bias correction by a rounding step, so expect tiny
differences elsewhere.

Step 1, once per net. Use `--arch transformer` for T1 and `--arch resnet` for
T74 and T79; the tags `t1_256x10`, `t74_744021` and `t79_791556` reproduce the
original file names:

```
python scripts/quantize_lc0_int8.py --lc0 <lc0.exe> --weights 744021.pb.gz --arch resnet --tag t74_744021 --pgns calibration/sprt_dagfix_vs12a3852_local.pgn calibration/endgame_935532_vs_825532.pgn --pgn-fractions 0.8 0.2 --encoder-cc <lc0 checkout>/src/neural/encoder.cc --out-dir out
```

Step 2:

```
python scripts/pack_lc0_sycl_int8.py --src-pb 744021.pb.gz --qdq-onnx out/t74_744021_int8.onnx --out-pb t74-744021-sycl-int8.pb.gz
```

About the scripts:

- `pack_lc0_sycl_int8.py` is the same file as on `feature/sycl-int8`.
- `quantize_lc0_int8.py` is the file from `feature/openvino-int8-static`
  with two options added:
  - `--pgn-fractions` sets how many positions come from each PGN. The
    original run took 80/20 from the two files, while that branch's version
    always splits evenly.
  - `--encoder-cc` points to the move table when the script isn't inside an
    lc0 checkout.
- The original run used a working copy that took the move table from
  another local tool (the same 1,858 moves in the same order) and had the
  80/20 split built in. Its output also lacks the `[0] OV GPU FP16 Baseline`
  line that the branch version prints.

## Limitations

- Measured on one laptop GPU (Iris Xe, 80 EUs) under Windows only.
- No game match has been played with the SYCL INT8 nets. The OpenVINO INT8
  version of T74 beat OpenVINO FP16 by about +150 Elo in a 60-game match at
  10 s + 0.1 s, but that win comes from the speed gain and a different backend.
- T79 is faster than FP16 only at batch 16 on this GPU (+16%), and 9% to
  16% slower from batch 30 up.
- Only the classical 112-plane input format is supported by the calibration
  script.
