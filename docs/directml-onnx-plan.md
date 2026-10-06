# Plan: ONNX-converter-driven DirectML backend

**Date:** 2026-10-06. **Base branch:** `feature/directml-dmlx-ops` (e318d55d,
the current DirectML mainline). **Context:** memory-bank handoffs #53-55,
notes 2518/2527, `docs/directml-handoff.md`.

## 1. Goal

Give DirectML the OpenVINO execution model:

```
WeightsFile --ConvertWeightsToOnnx()--> in-memory ONNX --graph executor on DML--> evals
```

so new trainer features (smolgen variants, head changes, new mixer knobs)
need **no backend code changes**: the converter already linearizes smolgen,
kda local_conv, PE_DENSE, and MHA into primitives — OpenVINO runs all of
them with zero per-feature C++.

Secondary motivation: the native backend's remaining gap to SYCL is ~1.34x
(note 2518, down from ~1.5x). A vendor graph compiler (Route A) or our own
cross-layer fusion (Route B) attacks the per-op overhead that hand-composed
DML chains still pay.

## 2. Current state (verified in tree)

| Piece | Status |
|---|---|
| Native `directml` backend | Mature on this lineage: batch ladder (single definition, densified, batch>max refuses instead of poisoning the driver), shader-blob caching (startup 17s→6s), native DirectMLX SOFTMAX1 in MHA (+8.6% per-eval, note 2518), FP16 correctness landed (FP32 recurrence + dml::Cast boundaries, note 2527), **85/85 parity suite green**. Gap to SYCL: **1.336x** (was ~1.5x). |
| `ConvertWeightsToOnnx` | Done; emits opset-17 ONNX incl. the KDA recurrence as a **Scan** body, fused qkv, serpentine directions, qkv_silu, fp16 option. OpenVINO consumes it in-memory (`network_openvino.cc:714-724`). |
| ORT `onnx` backend with **DML EP branch** | Done: `OnnxProvider::DML` (`network_onnx.cc:657-660`, `AppendExecutionProvider("DML")`), `USE_DML` guards, meson probes `dml_provider_factory.h` (`meson.build:668`). It calls `ConvertWeightsToOnnx` itself when the net has no embedded ONNX (`network_onnx.cc:980`). |
| Converter-driven DirectML | **Does not exist on any branch** (grepped `ConvertWeightsToOnnx` across the directml backends). |
| Known ONNX/KDA gap (handoff #52) | onnx-\* backends **unverified with KDA nets**: the Scan body relies on outer-scope initializers; onnx2leela cannot round-trip KDA. |

## 3. Route A — ORT DirectML EP (days; ship first)

The literal OpenVINO analogy: a vendor runtime consumes converter output and
does graph optimization/fusion for us.

**Phase A0 — build & environment (0.5 day)**
- Fetch the ORT-with-DML nuget (`Microsoft.ML.OnnxRuntime.DirectML`), wire
  `dml_provider_factory.h`/lib paths into the meson probe (same flow as the
  DirectML redist DLL), build `onnx-dml`.
- Copy `onnxruntime.dll` + `DirectML.dll` beside the exe.

**Phase A1 — classical-net verification (0.5 day)**
- Run the parity pattern with `backend=onnx-dml` on a classical net
  (T79-class) first: validates the EP wiring independent of KDA.

**Phase A2 — KDA Scan verification (1-3 days, the real work)**
- Run the 85-test parity suite on `onnx-dml`. Expected failure: ORT has no
  DML implementation of Scan (CPU-only) → the graph partitions at every Scan;
  also the **outer-scope initializers** the Scan body references may be
  rejected by ORT's partitioner.
- Contingency (converter-side, small): make `Converter::Scan` inline the
  outer-scope initializers it references into the body subgraph as
  body-local initializers (they are constants; duplication is bounded by
  body size). Flag it behind `WeightsToOnnxConverterOptions` so OpenVINO's
  path is untouched.
- Perf note to record: Scan-on-CPU means D2H/D3H per KDA layer per eval.
  On iGPU (shared memory) this is cheap-ish; measure before judging.

**Phase A3 — benchmarks + fp16 (0.5 day)**
- All real nets (`kda-native-935532`, `kda-t1-*`, …) via `lc0 bench`, vs the
  SYCL and native-directml numbers. Then flip
  `converter_options.data_type` to Float16 (ORT-DML is fp16-native) and
  re-run the parity suite + bench.

**Route A risks:** ORT DLL heft (~15-50 MB); Scan CPU-fallback perf; ORT
version/ABI churn; license is MIT (fine).

## 4. Route B — in-house ONNX→DirectMLX compiler (`dml-onnx`, ~2 weeks; strategic)

A translator that walks the converter's ONNX and builds DML graphs, reusing
100% of the existing execution infrastructure (DmlDeviceContext, arenas,
cached binding tables, batch ladder, DmlExecScope, InputsOutputs, the six
HLSL kernels).

**Key simplification:** the converter's op vocabulary is *closed* (our own
emitter) — Phase B0 inventories it precisely from converter.cc (Gemm /
Add / Mul / Conv / Softmax / Reduce\* / Concat / Gather / Slice / Transpose /
Reshape / activations + Scan). Every non-Scan op already has a working
DirectMLX equivalent inside layers.cc today; refactor them from layer
methods into free `BuildXxx(GraphFactory&, ...)` functions. **Exactly one
special case remains: Scan → `KdaRecurrenceLayer`** (+ `KdaLocalConvLayer`
if the grouped dml::Convolution form still trips this driver). Smolgen /
PE_DENSE / MHA need nothing — the converter already expressed them as
primitives.

**The real engineering: the graph partitioner (Phase B1).** This driver's
rules (docs/directml-handoff.md §4) forbid mid-graph reshapes, size-1
broadcasts, and strided batched-GEMM operands. The partitioner walks the
ONNX topology and cuts it into DML-legal segments at Reshape / Transpose /
broadcast boundaries; segment outputs land in arena slots (the ladder
rounds batches); segments are stitched by dispatch order exactly like
EvalKda stitches proj→recurrence→tail today. That stitching logic already
exists in layers.cc — the partitioner generalizes it.

**Phases:** B0 op inventory + converter-emitter contract test (0.5d) →
B1 partitioner (3-5d) → B2 translators (2-3d, mostly refactor) →
B3 Scan substitution (1d) → B4 parity + bench (1-2d).

**Payoff:** no ORT dependency; delete the per-layer sync burden; selective
HLSL fusion becomes trivial (swap any segment for a kernel — ORT can never
use `KdaRecurrenceLayer`).

## 5. Decision gate

1. Build Route A, measure (it is mostly verification work; days not weeks).
2. If ORT-DML (even with Scan on CPU) beats the native backend → ship A as
   `onnx-dml`, keep B as the perf/dependency follow-up.
3. If ORT's KDA path is broken beyond the converter contingency → B
   directly, with A's measurements as the ORT ceiling reference.
4. Independent of both: if the 1.336x native gap is acceptable and
   maintenance is the only pain, Route A alone (a thin, already-existing EP)
   is the cheapest maintenance win.

## 6. Shared harness (already built)

- Parity: the 85-test suite accepts any registered backend string — add the
  new backend name to its matrix.
- Benchmarks: `lc0 bench --weights=<real net> --backend=<name>` on the real
  nets; compare against the SYCL and native numbers from notes 2518/2527.
- Driver rules + timing data: `docs/directml-handoff.md` §4.

## 7. Risks

| Risk | Mitigation |
|---|---|
| ORT rejects outer-scope initializers in Scan body | Converter contingency inlines them (A2) |
| Scan CPU fallback slow | Measure; iGPU shared memory may make it moot; B eliminates it |
| Partitioner misses a driver rule | Reuse §4 rules as build-time assertions; the 85-test suite catches the rest |
| Two DirectML backends confuse users | Native `directml` stays as the precision reference; document `onnx-dml` vs `dml-onnx` naming |
| Plan premises drift (this doc was first drafted against the fa60b05-era state) | Re-verify §2 against the checked-out tree before starting any phase |

## 8. Provenance

First drafted 2026-09-04 against `feature/directml-backend` @ fa60b05
(4/6 parity, 27-160 nps, three-stage smolgen). Rewritten 2026-10-06 against
`feature/directml-dmlx-ops` @ e318d55d after the backend matured
independently (85/85 parity, 1.336x gap, FP16 correctness, shader caching);
Route A/B analysis unchanged, current-state section and motivation updated.
