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
lc0 --weights=nets/t74-744021-sycl-int8.pb.gz --backend=sycl-fp16 --backend-opts=int8=false
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
weights and scales as the SYCL nets. No game match has been played with the
SYCL INT8 nets yet, so their playing strength is unmeasured.

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
commands to rebuild the nets. [`logs/quantize_log.txt`](logs/quantize_log.txt)
is the log of the calibration run that produced them.

## Related branches

- [`feature/sycl-int8`](https://github.com/ContradNamiseb/lc0/tree/feature/sycl-int8):
  the SYCL INT8 engine that runs these nets.
- [`feature/openvino-int8-static`](https://github.com/ContradNamiseb/lc0/tree/feature/openvino-int8-static):
  the OpenVINO backend for the INT8 IR from step 1; the quantize script comes
  from there.
