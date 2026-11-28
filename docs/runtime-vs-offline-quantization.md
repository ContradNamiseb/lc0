# Why Non-Trunk Layers Stay Unquantized on Disk (Offline vs. Runtime INT8)

Inside [`nets/`](../nets), only the heavy trunk blocks (residual `ConvBlock`s on T74/T79, and `MHA`/`FFN` projections inside Transformer encoders on T1) are stored pre-quantized as `INT8` in the `.pb.gz` files. The remaining layers are kept unquantized on disk, and on `feature/blas-int8` AVX-512 VNNI CPU builds the most compute-heavy of them are quantized at load time / runtime instead.

See [`quantization.md`](quantization.md) for how the offline trunk calibration pipeline itself works.

## Storage & Runtime Execution Tiers

| Architecture | Pre-quantized Offline in `.pb.gz` (Static W8A8) | Quantized at Load Time + Runtime in `blas-int8` | Always Pure FP32 / FP16 |
| --- | --- | --- | --- |
| **T79** (`791556`, 15×192 Mish SE-ResNet) | 30 residual 3×3 convs (`conv1`, `conv2`) | `input` 3×3 conv (**SmoothQuant-109**); Attention Policy `ip_pol`, `ip2_pol` (`Q`), `ip3_pol` (`K`) (**Dynamic W8A8**) | SE units, `Q·K^T`, promotion, Value head, MovesLeft head |
| **T74** (`744021`, 10×128 ReLU SE-ResNet) | 20 residual 3×3 convs (`conv1`, `conv2`) | `input` 3×3 conv (**SmoothQuant-109**); `policy1`, `policy` 3×3 convs (**Dynamic W8A8**) | SE units, Value head, MovesLeft head |
| **T1** (`t1-256x10`, 10×256 Transformer) | 60 encoder `MHA` + `FFN` projections | `ip_emb` preproc/FFN/gates, **Smolgen** (40 projections), Attention Policy `ip_pol`/`Q`/`K` (**Dynamic W8A8**) | `Q·K^T`, `attn·V`, LayerNorms, promotion, Value head, MovesLeft head |

*(Note: when loading any INT8 layer from `.pb.gz`, `network_legacy.cc` also populates an in-memory `DequantizeInt8RowMajor` float copy so FP32/GPU backends can still run the file unchanged.)*

---

## Four Reasons Those Layers Stay Unquantized on Disk

### 1. Accuracy: Dynamic Quantization & SmoothQuant-109 Protect Sensitive Layers

- **Why the trunk uses static offline calibration:** Each residual `ConvBlock` or encoder projection uses a single global `(act_scale, act_zp)` calibrated offline across 1,024 game positions. That eliminates min/max scanning during inference, and residual skip connections tolerate small per-layer quantization noise.
- **Why policy heads and Smolgen benefit from dynamic activation quantization:**
  - `DynamicInt8Weights` computes a fresh per-row `(min, max)` on every square/row at runtime.
  - `DynamicConv3x3Int8` computes a fresh per-board `(min, max)` on every board at runtime.
  - Because each position gets the full `0..255` unsigned range with zero outlier clipping, sensitive computations like T1's attention-bias generator (`Smolgen`) and `T74`'s convolutional policy head avoid static range distortion while adding minimal min/max overhead.
- **Why the input 3×3 conv (`112 → 192` / `128`) is special (SmoothQuant-109):**
  - 111 of the 112 classical input planes lie in `[0, 1]` (binary piece planes, castling rights, side-to-move, repetition flags, and plane 111 all-ones), **except plane 109** (the 50-move rule halfmove counter), which spans `0..100` on `INPUT_CLASSICAL_112_PLANE` inputs.
  - Naive static offline calibration stretches the shared input activation scale to accommodate `rule50`, crushing the binary piece planes into tiny integer steps (`0`, `2`, `3`).
  - Instead, `blas-int8` applies **SmoothQuant-109 at load time** (`layers.cc` in `ConvLayer`): it multiplies plane 109's weights by `100` before per-channel symmetric `int8` weight quantization and scales plane 109 activations down by `0.01` (`1/255` scale, `zp=0`) at inference time. Every binary plane then maps **exact `0 → 0` and `1 → 255`** with zero activation error.

### 2. Format History: Built Originally as "Trunk-Only INT8" for GPU

The `*-sycl-int8.pb.gz` weights were designed around a **trunk-only** PTQ strategy targeting Intel GPU DP4A (`feature/sycl-int8`, commit `b69a3302`) and OpenVINO: quantize the trunk (`>85%` of FLOPs) while keeping input convs, SE units, Smolgen, and heads in native FP16/FP32. On a GPU, running those small boundary layers in FP16 has negligible cost; on CPU (`feature/blas-int8`), Winograd FP32 policy convs and dense policy/Smolgen projections were significant bottlenecks, so `blas-int8` added load-time weight quantization and dynamic activation quantization later (commits `df0089da` and `7b1cc65e`) specifically for CPU speed.

### 3. Protobuf Schema (`proto/net.proto`) Has No Policy-Head INT8 Fields

Commit `b69a3302` added `Weights::Layer::Encoding::INT8 = 5` and static calibration slots (`w_scales`, `act_scale`, `act_zp`) only to `ConvBlock`, `MHA`, and `FFN`. `PolicyHead` dense weights (`ip_pol_w`, `ip2_pol_w`, `ip3_pol_w`), `Smolgen`, and `ValueHead` matrices are stored as plain `Layer` messages with no protobuf fields for offline per-channel weight scales or activation zero-points.

### 4. Compatibility with Existing `quantized-nets`

Load-time weight quantization plus dynamic activation quantization allows `blas-int8` to accelerate every existing `*-sycl-int8.pb.gz` file out of the box—requiring **zero schema changes, zero re-calibration, and zero file regeneration**—while leaving full-precision weights intact on disk for GPU/FP16 backends that prefer running heads and Smolgen in FP16.

---

## Empirical Validation: Offline-Static vs. Runtime-Dynamic (256 Held-Out Positions)

To test rather than assume the accuracy effect, we calibrated offline static QDQ (`MSE` clipping + `Sequential Fast Bias Correction` using `quantize_lc0_int8.py` on the same 1,024 calibration / 256 held-out real-game split, `50` positions with `rule50 >= 10`) on the non-trunk layers and compared directly against FP32 ONNX baselines and live `lc0.exe --backend=blas` (`blas-int8`, `PolicyTemperature=1.0`, `NNCacheSize=0`, `threads=4`):

### T79 (`791556`, 15×192 Mish SE-ResNet + 192d Attention Policy)

| Variant | Layers INT8 | Top-1% | Policy KL | `Q_MAE` | `Q_Max` | Top-1% (`rule50 >= 10`, `n=50`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trunk-only Offline (input/head FP32) | 30 | 96.5% | 0.00285 | 0.01036 | 0.25041 | 94.0% (`Q_MAE=0.00561`) |
| **Live `blas-int8` (Runtime SQ109 In + Dyn AttnPol)** | **30 + 4** | **95.3%** | **0.00345** | **0.01209** | **0.26100** | **94.0% (`Q_MAE=0.00824`)** |
| Trunk + Offline `/inputconv` (raw `rule50`) | 31 | **73.0%** | **0.13533** | **0.10017** | **0.86932** | **64.0% (`Q_MAE=0.08252`)** |
| Trunk + Offline `/inputconv` (SQ109 folded) | 31 | 92.6% | 0.00381 | 0.01291 | 0.25369 | 96.0% (`Q_MAE=0.00649`) |
| Trunk + Offline `AttnPolicy` (`dense1`/`Q`/`K`) | 33 | 95.3% | 0.00340 | 0.01036 | 0.25041 | — |
| Trunk + Offline `InputConv(SQ109)` + `AttnPolicy` | 34 | 91.8% | 0.00434 | 0.01291 | 0.25369 | 94.0% (`Q_MAE=0.00649`) |

### T74 (`744021`, 10×128 ReLU SE-ResNet + Conv Policy)

| Variant | Layers INT8 | Top-1% | Policy KL | `Q_MAE` | `Q_Max` | Top-1% (`rule50 >= 10`, `n=50`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trunk-only Offline (input/head FP32) | 20 | 95.7% | 0.00308 | 0.00912 | 0.19089 | 98.0% (`Q_MAE=0.00932`) |
| **Live `blas-int8` (Runtime SQ109 In + Dyn ConvPol)** | **20 + 3** | **96.1%** | **0.00386** | **0.01025** | **0.21371** | **98.0% (`Q_MAE=0.00932`)** |
| Trunk + Offline `/inputconv` (raw `rule50`) | 21 | **53.5%** | **0.48395** | **0.39797** | **1.99878** | **56.0% (`Q_MAE=0.27262`)** |
| Trunk + Offline `/inputconv` (SQ109 folded) | 21 | 95.7% | 0.00368 | 0.01212 | 0.18367 | 98.0% (`Q_MAE=0.01225`) |
| Trunk + Offline `ConvPolicy` (`policy1` + `policy`) | 22 | 94.9% | 0.00371 | 0.00912 | 0.19089 | — |
| Trunk + Offline `InputConv(SQ109)` + `ConvPolicy` | 23 | 95.7% | 0.00416 | 0.01212 | 0.18367 | 98.0% (`Q_MAE=0.01225`) |

### T1 (`t1-256x10`, 10×256 Transformer + Smolgen + 256d Attention Policy)

| Variant | Layers INT8 | Top-1% | Policy KL | `Q_MAE` | `Q_Max` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trunk-only Offline (`+ZeroOpSQ`) | 60 | 93.0% | 0.00566 | 0.01446 | 0.13928 |
| **Live `blas-int8` (Trunk + Dyn Smolgen + Dyn AttnPol)** | **60 + 43** | **91.0%** | **0.00666** | **0.01457** | **0.21145** |
| Trunk + Offline `AttnPolicy` (`dense1`/`Q`/`K`) | 63 | 94.1% | 0.00605 | 0.01446 | 0.13928 |
| Trunk + Offline `Smolgen` (40 layers) | 100 | 92.2% | 0.00655 | **0.01752** | 0.18177 |
| Trunk + Offline `Smolgen` + `AttnPolicy` | 103 | 91.8% | 0.00697 | **0.01752** | 0.18177 |

### Takeaways from Measurement
1. **InputConv requires SmoothQuant-109, and runtime SQ109 is cleaner than offline SQ109:** Without SQ109, offline static `/inputconv` collapses accuracy to `73.0%` (T79) and `53.5%` (T74). Even with SQ109 folded offline, runtime SQ109 preserves higher Top-1 on T79 (`95.3%` vs `92.6%`) and lower `Q_MAE` on both T79 (`0.01209` vs `0.01291`) and T74 (`0.01025` vs `0.01212`).
2. **T1 Smolgen needs runtime dynamic quantization for Value accuracy:** Offline static Smolgen increases Value `Q_MAE` by `+21%` (`0.01446 → 0.01752`), while `blas-int8` dynamic per-row Smolgen stays at `0.01457`.
3. **AttentionPolicy tolerates offline static quantization cleanly:** With MSE clipping + Fast Bias Correction, offline `AttnPolicy` matches live runtime quantization on T79 (`95.3%`) and slightly edges it out on T1 (`94.1%` vs `91.0%`), whereas `T74`'s convolutional `ConvPolicy` favors runtime dynamic quantization (`96.1%` vs `94.9%`).

---

## Decision: Keep vs. Discard

- **Keep:**
  - The trunk-only `.pb.gz` weights in [`nets/`](../nets) (`t79-791556-sycl-int8.pb.gz`, `t74-744021-sycl-int8.pb.gz`, `t1-256x10-sycl-int8.pb.gz`) and their verified copy in `nets-backup-trunk-only/`.
  - Runtime SmoothQuant-109 for `input_conv` (`T74`/`T79`) and runtime dynamic W8A8 (`Smolgen`, `AttentionPolicy`, `ConvPolicy`) in `feature/blas-int8`.
  - The verified `experiment/t79-ablations` inference optimizations (MKL OpenMP contention fix, vectorized `SELU` + 192d Q/K quantization reuse, integer-exponent `Exp`/`Mish3x` + pre-transposed SE, fused `InputConv` `nr=48`, and NHWC AVX-512 FP32 heads).
- **Discard:**
  - Experimental offline non-trunk quantized `.onnx` and `.pb.gz` files (`nets-exp-offline/`), because offline static `/inputconv` collapses accuracy without SQ109 and still trails runtime SQ109 with it, offline T1 `Smolgen` degrades Value `Q_MAE` by `+21%`, and `net.proto` has no static policy/Smolgen scale fields anyway.
