# Plan: ONNX-converter-driven DirectML backend

**Date:** 2026-10-06. **Repo:** lc0 fork, branch `feature/directml-backend`.
**Context:** handoffs #53-55 (memory bank), `docs/directml-handoff.md`.

## 1. Goal

Replace the hand-built per-layer DirectML dispatch (layers.cc: AttentionBody /
EncoderBlock / AttentionPolicyHead / ValueHead, ~3k lines that must be kept in
sync with the trainer) with the OpenVINO flow:

```
WeightsFile --ConvertWeightsToOnnx()--> in-memory ONNX --graph executor on DML--> evals
```

so that new trainer features (smolgen variants, head changes, new mixer
knobs) need **no backend code changes** — the converter already linearizes
smolgen, kda local_conv, PE_DENSE, and MHA into primitives; OpenVINO runs all
of them with zero per-feature C++.

## 2. Current state (verified in tree)

| Piece | Status |
|---|---|
| `ConvertWeightsToOnnx(weights, WeightsToOnnxConverterOptions)` | Done; emits opset-17 ONNX incl. the KDA recurrence as a **Scan** body, fused qkv, serpentine directions, qkv_silu, fp16 option. OpenVINO consumes it in-memory (`network_openvino.cc:714-724`). |
| ORT `onnx` backend with **DML EP branch** | Done: `OnnxProvider::DML` (`network_onnx.cc:657-660`), `USE_DML` guards, meson probes `dml_provider_factory.h` (`meson.build:668`). It calls `ConvertWeightsToOnnx` itself when the net has no embedded ONNX (`network_onnx.cc:980`). |
| Native `directml` backend | Feature-complete, hand-built per layer + 6 HLSL kernels. Parity 4/6 at 2e-4; real-net benches 27-160 nps vs SYCL 299-984 (`fa60b05`). |
| Known gap (handoff #52) | onnx-\* backends **unverified with KDA nets**: the Scan body relies on outer-scope initializers; onnx2leela cannot round-trip KDA. |

## 3. Route A — ORT DirectML EP (days; ship first)

The literal OpenVINO analogy: vendor runtime consumes converter output and
does graph optimization/fusion for us.

**Phase A0 — build & environment (0.5 day)**
- Fetch ORT-with-DML nuget (`Microsoft.ML.OnnxRuntime.DirectML`), wire
  `dml_provider_factory.h`/lib paths into the meson probe (same flow as the
  DirectML redist DLL we placed beside the exe), build `onnx-dml`.
- Copy `onnxruntime.dll` + `DirectML.dll` beside `build-dml/lc0.exe`.

**Phase A1 — classical-net verification (0.5 day)**
- Run `kda_parity_test` pattern with `backend=onnx-dml` on a classical net
  (T79-class) first: validates the EP wiring independent of KDA.

**Phase A2 — KDA Scan verification (1-3 days, the real work)**
- Run all 6 synthetic parity nets on `onnx-dml`. Expected failure: ORT has
  no DML implementation of Scan (CPU-only) → graph partition splits at every
  Scan; also the **outer-scope initializers** the Scan body references may be
  rejected by ORT's partitioner.
- Contingency (converter-side, small): make `Converter::Scan` inline the
  outer-scope initializers it references into the body subgraph as
  body-local initializers (they are constants; duplication is bounded by
  body size). Flag it behind `WeightsToOnnxConverterOptions` so OpenVINO's
  path is untouched.
- Perf note to record: Scan-on-CPU means D2H/D3H per KDA layer per eval.
  On iGPU (shared memory) this is cheap-ish; measure before judging.

**Phase A3 — benchmarks + fp16 (0.5 day)**
- All 4 real nets, `bench`, vs the table in handoff #55. Then flip
  `converter_options.data_type` to Float16 (ORT-DML is fp16-native) and
  re-run parity + bench.

**Route A risks:** ORT DLL heft (~15-50 MB); Scan CPU-fallback perf; ORT
version/ABI churn; licensing size acceptable (MIT).

## 4. Route B — in-house ONNX→DirectMLX compiler (`dml-onnx`, ~2 weeks; strategic)

Write a translator that walks the converter's ONNX and builds DML graphs,
reusing 100% of the existing execution infrastructure (DmlDeviceContext,
arenas, cached binding tables, batch ladder, DmlExecScope, InputsOutputs).

**Key simplification:** the converter's op vocabulary is *closed* (it is our
own emitter) — Phase B0 inventories it precisely from converter.cc
(Gemm/Add/Mul/Conv/Softmax/Reduce\*/Concat/Gather/Slice/Transpose/Reshape/
activations + Scan). Every non-Scan op already has working DirectMLX
equivalents inside layers.cc today; they get refactored from layer methods
into free `BuildXxx(GraphFactory&, ...)` functions. **Exactly one special
case remains: Scan → the existing `KdaRecurrenceLayer`** (+ KdaLocalConvLayer
if the conv stays a kernel; try dml::Convolution group=emb first, it is
standard DML). Smolgen/PE_DENSE/MHA need nothing — the converter already
expressed them as primitives (that is why OpenVINO needs no smolgen code).

**The real engineering: the graph partitioner (Phase B1).** This driver's
rules (docs/directml-handoff.md §4) forbid mid-graph reshapes, size-1
broadcasts, and strided batched-GEMM operands. The partitioner walks the ONNX
topology and cuts it into DML-legal segments at Reshape/Transpose/broadcast
boundaries; segment outputs land in arena slots (the ladder rounds batches);
segments are stitched by dispatch order exactly like EvalKda stitches
proj→recurrence→tail today. That stitching logic already exists three times
in layers.cc — the partitioner generalizes it.

**Phases:** B0 op inventory + converter-emitter contract test (0.5d) →
B1 partitioner (3-5d) → B2 translators (2-3d, mostly refactor) →
B3 Scan substitution (1d) → B4 parity + bench (1-2d).

**Payoff:** no ORT dependency; delete the per-layer sync burden; selective
HLSL fusion becomes trivial (swap any segment for a kernel — ORT can never
use KdaRecurrenceLayer).

## 5. Decision gate

1. Build Route A, measure.
2. If ORT-DML (even with Scan on CPU) beats the native backend's 27-160 nps
   → ship A as `onnx-dml`, keep B as the perf/dependency follow-up.
3. If ORT's KDA path is broken beyond the converter contingency → B directly,
   with A's measurements as the ORT ceiling reference.

## 6. Shared harness (already built)

- Parity: extend `test_kda_parity_directml.cc`'s `CompareBackends` to the new
  backend names (it takes any registered backend string).
- Benchmarks: same `lc0 bench --weights=<real net> --backend=<name>` flow on
  the 4 real nets in `C:\Users\Contrad\Documents\lc0-Engine\sycl\`.
- Driver rules + timing data: `docs/directml-handoff.md` §4, handoff #55.

## 7. Risks

| Risk | Mitigation |
|---|---|
| ORT rejects outer-scope initializers in Scan body | Converter contingency inlines them (A2) |
| Scan CPU fallback slow | Measure; iGPU shared memory may make it moot; B eliminates it |
| Partitioner misses a driver rule | Reuse §4 rules as assertions at build time; parity suite catches the rest |
| Two DirectML backends confuse users | Native `directml` stays as fallback/precision reference; document `onnx-dml` vs `dml-onnx` naming |
