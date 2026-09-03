# DirectML Backend Correctness — Debug Handoff

**Date:** 2026-09-03/04. **Author of this session:** Muse Spark (via OpenCode).
**Repo:** `lc0-training/official-training-branch/libs/lc0`, branch `feature/directml-backend`.
**Machine:** Windows, Intel Iris Xe Graphics (integrated), system DirectML 1.x, oneAPI icx toolchain.
**Build:** `cmd /c build-dml.cmd` (from repo root; uses `build-dml/` meson dir).
**Test binary:** `build-dml/kda_parity_test_directml.exe` (gtest).
**Tolerance (SYCL bar):** `kTol = 2e-4` on Q, D, and worst policy diff over 1858 moves.

## 1. Goal and current status

Get `DirectMlKdaParity.*` (whole-network DirectML vs BLAS outputs on synthetic
KDA-hybrid nets) to tolerance. The BLAS reference already contains the
`qkv_silu` fix (ported from `sycl-openvino-fix`, `network_blas.cc:418`).

Latest measured status (all deterministic, bit-identical across runs):

| Test | Q | D | Policy worst | Verdict |
|---|---|---|---|---|
| KdaHybridNet (KDA enc, WDL, no MLH) | 0.0029 vs 0.0014 | 0.333 vs 0.326 | **3.2e-4, move 1794, all 18 bad moves in promo region 1793–1857** | FAIL (close) |
| KdaMhaNet (KDA+MHA+MLH) | — | — | — | **ERROR: exception at load/Eval, see §3** |
| NoEncoderNet (no encoders, WDL, no MLH) | -0.027 vs 0.015 (diff 0.043!) | 0.3278 vs 0.3284 (6e-4) | **total_bad=0 — PASSES** | FAIL on Q only |

So: **policy is FIXED on NoEncoder; value Q is still off (0.043); hybrid has
small residual drift (promo 3.2e-4, Q 0.0015, D 0.007); MHA+MLH nets cannot
even build their graphs (hard blocker, §3).**

## 2. Bugs already found and FIXED (all in tree — do not regress)

1. **Missing input upload** (`network_directml.cc`, forwardEval ~line 590).
   `forwardEval` expanded planes into `io->input_mapped_` (upload heap) but
   never copied them to `tensor_mem[0]`; the body read zero-init'd memory.
   Fix: transition `tensor_arena_` UAV→COPY_DEST, `CopyBufferRegion` from
   `input_upload_`, transition back — all recorded before body Eval.
2. **Zero-init staging lifetime hung the GPU** (`network_directml.cc` ~437).
   The zero-fill block created the 16MB `zeros` staging buffer *inside* the
   per-arena loop, releasing it before Execute → GPU read freed memory →
   `DXGI_ERROR_DEVICE_HUNG` (0x887a0006), fence stuck at UINT64_MAX, all
   outputs zero. Fix: one shared `zeros` buffer outliving Execute+Wait.
   Diagnostic used: `GetDeviceRemovedReason()` + fence value print.
3. **PolicyMap was scatter→gather inverted AND truncated** (`layers.cc`
   PolicyMapLayer ctor ~839). Old code gathered `policy[j] = attn[map[j]]`
   for j<1858 (wrong direction, drops promotion region). Fix: build the
   *inverse* table (`inv[j] = i` where `map[i]==j`) over all 4288 rows.
   Verified `kAttnPolicyMap` is bijective (1858 valid, 0 dups, 0 uncovered —
   keep that property if the map ever changes).
4. **Dangling uploader pointers (THE flakiness)** (`layers.h`/`layers.cc`).
   `DmlWeightUploader::Add/AddRaw` borrowed caller pointers; flush happens
   after all layers are built, so ctor-locals (`pos_encoding` 16KB,
   policy-map `inverse` 7KB) were freed → nondeterministic garbage weights
   (run-to-run flips between sane/e36/NaN). Fixes (both, defense in depth):
   - `Pending` now OWNS a `std::vector<uint8_t>` copy (`layers.h:100`).
   - `indices_host_` / `pos_encoding_host_` members keep sources alive.
   After this fix all runs are bit-identical. **Lesson: any future
   nondeterminism across fresh processes = dangling host pointer or
   uninitialized GPU read, not driver flakiness.**
5. **Scores orientation — original code was RIGHT.** Com initial analysis
   claimed BLAS/CUDA compute KᵀQ and "fixed" DML to `Gemm(wk,wq)`. Empirical
   test (both orientations) proved this wrong: `Gemm(wq,wk)` (Q·Kᵀ) gives
   `total_bad=0`, the other diverges. Reason: dot products commute, so
   BLAS's K_j·Q_i == Q_i·K_j. The comment in `layers.cc` (scores section)
   now records this. **Do not "fix" the orientation again.**
6. **Value graph output never lands in io buffers** (`network_directml.cc`
   ~649). The value graph computes correctly (proven by outputting to
   scratch: sane logits) but writes to `value_gpu_`/`policy_gpu_` are
   silently dropped (no error); PolicyMap writes to the same buffers fine.
   Workaround shipped: value graph outputs to `scratch+0`, then an explicit
   `CopyBufferRegion` to `value_gpu_`. Comment + TODO in code. (Prime
   suspect, unproven: interaction with the graph's temporary resource;
   only value uses temp among the no-encoder graphs. Re-test direct io
   output on newer drivers/discrete GPUs.)
7. **MHA via HLSL transpose + dense [N·H,1]-batch attention** (new files
   `shaders/mha_transpose.hlsl`, `mha_transpose_shader_source.h`,
   `MhaTransposeLayer` in `layers.h/.cc`, `EvalMha` rewrite with
   `mha_qkv/attn/tail_compiled_` maps). Reason: `[N,H]`-batched GEMMs with
   strided `ReinterpretView` inputs throw `E_INVALIDARG` at `CreateOperator`
   on this driver, while identical `[N,1]`-batched forms compile. The old
   `HeadViewStrides` helper is deleted. MHA qkv/attn/tail graphs now compile
   (verified in GRAPH # log). Transpose buffers live in `buffer1`
   (qt/kt/vt/ctx/merged = 5×S, S=max_tokens·d·elem). `transient_arena_` was
   bumped 64→256MB for MHA attention temporaries at large batches.
8. **MHA tail f1b/f2b must be WeightChannel** (`layers.cc` ~1407 NOTE).
   The old MHA code used dense size-1 `[1,1,1,C]` bias inputs to Add, which
   this runtime rejects. All biases everywhere must be strided
   matching-sizes (`WeightChannel`) or dense same-shape — **never size-1
   broadcast** (see §4 rule 1).
9. **`ReinterpretView` with empty strides is UB** (`layers.cc` helper now
   maps empty→`dml::NullOpt` dense). An engaged-but-empty stride array
   leaves DimensionCount=4 with a garbage stride pointer.
10. **Misc correctness:** `DmlWeightUploader` copies (see 4);
    `transient_arena_` 256MB (see 7).

## 3. CLOSED (2026-09-04, ZCode session): MLH-embed build/record throw

**Root cause (empirically established): this driver fails DML object
creation with misleading error codes once dispatches have been recorded on
the device** -- CreateBindingTable after ~7 recorded dispatches returns
bogus DXGI_ERROR_DEVICE_REMOVED (GetDeviceRemovedReason() is S_OK!), and
CreateOperator after records returns bogus E_INVALIDARG. museSpark's
"6th build" pattern and the earlier device-removed symptoms were all this
one driver defect; the interleaved-build hypothesis was close but the
trigger is *recording*, not building.

**Fix shipped:** (a) two-phase compile -- every layer got EnsureCompiled,
forwardEval compiles all graphs for the batch before recording anything;
(b) binding tables are created at BUILD time (GraphFactory::Compile calls
GetOrCreateBindingTable) and cached per compiled operator in
DmlDeviceContext::tables_, rebound per dispatch, never re-created. The
descriptor pool is now permanent (no per-batch reset) since cached tables
hold their slots.

Also fixed en route: binding tables sized exactly to RequiredDescriptorCount
(the KDA projection graph needs 105; a fixed 64-slot table failed
E_INVALIDARG), and the MHA scratch estimate (EvalMha keeps 5*d_model/token
of head-transpose buffers in the buffer1 slot; the 3*d_model estimate
under-reserved the arena).

Status after fix: ALL nets run end to end (KDA, KDA+MLH, MHA+MLH,
KDA+MHA+MLH, NoEncoder). Remaining gaps are purely numeric -- see section 6.

---

## 3. ORIGINAL BLOCKER TEXT (historical): MLH-embed `CreateOperator` throw (E_INVALIDARG)

**Symptom:** any net with a moves-left head (`MOVES_LEFT_V1`: KdaMhaNet)
throws `m_device->CreateOperator(&opDesc, ...)` during first-Eval graph
builds, in MLH-embed (`BuildGemmLayerOp`, N=4 tok=256 in=32 out=4 act=2/RELU
bias=1 — plain Gemm + strided-bias Add + ReLU). NoEncoder/Hybrid (no MLH)
build fine.

**What is established (all verified empirically, do not re-litigate):**
- The descs are valid: x `[1,1,256,32]` dense total 32768, w `[1,1,4,32]`
  dense total 512, align 0 dtype 1 flags 0 (dumped). The *identical* pattern
  compiles first-try in isolation, after 10 dummy builds, after LN/reduce/
  Gather/strided/value-pattern/big-temp builds, and after 10 *bound*
  dispatches.
- It is NOT: graph count (12 identical Gathers compile), retention (10
  retained + build compiles), transient (retained temp-graphs fine),
  strided biases (sweep compiles C=1..128), small M (M=1..32 compile),
  K=32 (compiles), activation (RELU=2 verified, not exotic), weight values
  (validation is shape-only), device creation (same setup everywhere),
  MLH-intrinsic (first-try compiles), KDA/MHA-specific (throws with NO
  encoders too: body,pembed,wqk,map,value then MLH throws).
- Bisect-by-skipping (TEMP, since removed): skipping value didn't help;
  skipping map+value moved the throw from MLH-embed to MLH-fc2 (i.e. the
  *6th build* throws regardless of which graphs — once it was MLH-embed as
  6th, once MLH-fc2 as 6th). With policy-head also skipped: MLH-embed+fc1
  built, fc2 threw. **Pattern: roughly the 6th backend build throws, but
  sweep builds of 10-12 succeed.** The difference between backend builds
  and sweep builds that remains untested: backend graphs are *dispatched*
  (bound+recorded) interleaved with builds; the closest sweep test
  (diverse build+record ×5) hit an ordering artifact and needs a rerun
  (see §5.1).
- A companion observation: `DmlSmoke.DenseGemmStableAcrossDispatches`
  (10 identical dispatches + later build) compiles fine, so same-shape
  record→build is clean.

**Primary hypothesis (untested): interleaved *diverse-shape* build+record
poisons later CreateOperator calls on this driver** (binding-table or
recorder state leaking across dispatchables?). The backend interleaves
(build N, record N, build N+1…); all passing sweeps either never record, or
record identical graphs.

**Suggested next steps (ordered):**
1. Reproduce minimally: extend the diverse build+record sweep (in
   `DmlSmoke.DenseGemmStableAcrossDispatches`, still present in the test
   file) to record 5 *different-shaped* dispatches then build MLH-embed.
   If it throws → driver quirk confirmed.
2. If confirmed, restructure to two-phase: compile ALL graphs up front
   (before recording any dispatch), then record. Cleanest: give each layer
   a `Compile(N, ctx)` split from Eval, call them all at forwardEval top
   (or first-Eval), keep Eval dispatch-only. This is also a latency win
   (no lazy-compile stalls mid-search).
3. Alternative (smaller): fuse sequential graphs (MLH embed+fc1+fc2 are one
   chain; policy-embed+wqk are one chain) to reduce build count — fragile
   if the trigger isn't count, prefer (2).
4. Note: `FIND: FCLayer eval` CERR (uncommitted worktree change) suggests
   someone already started tracing the MLH path — coordinate before
   duplicating.

## 4. Driver rules learned (Iris Xe, system DirectML 1.x) — FOLLOW THESE

1. **No size-1 broadcast in elementwise ops.** Biases must be strided
   matching-sizes (`WeightChannel`, e.g. `[1,1,256,64]` strides
   `{0,0,0,1}`), never dense `[1,1,1,C]`. The latter throws E_INVALIDARG.
   (Proven: MHA tail f1b/f2b; sweep `StridedAddChannelSweep` + every
   backend Add uses the strided form.)
2. **No implicit broadcast at all.** Every elementwise input must be
   explicitly reshaped to matching sizes via `ReinterpretView` (see the
   softmax `sm_bcast`, LayerNorm `mean_b`). Raw Reduce outputs
   (`[N,H,64,1]`) subtracted directly throw.
3. **No strided inputs to `[N,H>1]`-batched GEMMs.** Use the HLSL transpose
   (`mha_transpose.hlsl`) + dense `[N·H,1]`-batch GEMMs (`[N,1]`-batch is
   proven: policy scores). Same likely applies to other ops; prefer dense.
4. **NullOpt (omitted) strides = dense.** Never pass engaged-empty stride
   arrays (UB).
5. **Uploader owns bytes** (`Pending::owned`). Never borrow caller memory
   across the build→flush gap (caused day-long nondeterminism hunt, §2.4).
6. **Staging buffers must outlive Execute+Wait** (see §2.2 hang).
7. **Graph outputs to io buffers are suspect for temp-using graphs.**
   Value (only temp-user among simple graphs) needs scratch+copy
   (`network_directml.cc` TODO). Prefer arena outputs + explicit copies for
   small heads until proven otherwise on other drivers.

## 5. TEMP/leftover state in tree (clean up or finish)

- `layers.cc` uncommitted diff adds `CERR << "FIND: FCLayer eval"` (someone's
  live trace) and *removes* several TEMP markers (`MAP building`, `MHA tail
  * done`, `MHABUILD tail compiled`). The GRAPH-# log in
  `GraphFactory::Compile`, `MLHBUILD` markers in `BuildGemmLayerOp`, and all
  `DMLVALS`/`dist[`/`promo-bad` prints in the test were already removed.
  Decide with the team what diagnostics to keep (recommend: keep NONE in
  final, keep the `NOTE:`/`TODO:` comments).
- `test_kda_parity_directml.cc` still contains all `DmlSmoke.*` scaffolding
  (`DenseGemmStableAcrossDispatches` + diverse-record block,
  `StridedAddChannelSweep`, `GemmOutputSweep`, `MlhEmbed*` family,
  `GatherBuildLoop`, `PolicyMapCoversAll`) — useful, keep until §3 is
  closed, then trim to the two essential tests
  (`DenseGemmStableAcrossDispatches`, `PolicyMapCoversAll`).
  5.1. **Unfinished micro-experiment:** the diverse build+record block
  inside `DenseGemmStableAcrossDispatches` needs `bar()`-style per-shape
  error prints (currently a single shared block; last run threw inside it
  but the exact shape wasn't isolated — re-add the try/catch markers from
  git history if needed).
- `RunNetwork` in the test does NOT set `max_batch` (an earlier
  `max_batch=8` experiment was reverted; default 256 is correct for now —
  note the 819MB tensor arena this implies; see §6).
- `MakeNoEncoderNet` is back to `MOVES_LEFT_NONE`/no-MLH (my MLH bisect
  edit was reverted); `MakeKdaMhaNet` has KDA+MHA+MLH (restored).
- `meson.build`/`meson_options.txt` (tracked, modified): directml backend
  wiring + gtest. `src/neural/onnx/converter.cc` (tracked, modified): KDA
  ONNX export (serpentine/qkv_silu/scan) from the parallel line — do not
  touch unless the backend needs new export semantics.

## 6. After §3 is unblocked — remaining math gaps (all deterministic)

1. **Value Q off by 0.043 on NoEncoder** (D is fine at 6e-4): value logits
   systematically off ~0.1. Suspects, in order: final `ip2_val` bias
   handling (`WeightChannel` C=3 — verify binding bytes/offset!),
   `flat` reshape orientation ([1,1,4,2048] row-major = token-major rows —
   verify against BLAS `head_buffer` layout), ReLU placement (matches
   BLAS: emb-ReLU, hidden-ReLU, y-none — verified by reading, re-verify).
   Approach that worked before: CPU reference in-test (as done for body
   flow, which matched to 1e-6 — see git history for the CPUFLOW/FLOW
   snippet) extended stage by stage through the value head.
2. **Hybrid promo region 3.2e-4 (moves 1793–1857, 18 moves)** while the
   64×64 region passes: suspects are the finalize dots (wk rows 56–63 ×
   ppo) or `ip4_pol_w` upload/binding. The HLSL matches BLAS line-for-line
   (checked); next step is snapshotting `scores`/`wk`/`ppo` vs CPU (the
   snapshot plumbing was removed — re-add minimally, one buffer at a time,
   and *remove it the same day*: snapshot transitions were briefly
   suspected of perturbing driver state (unproven, but keep the window
   small).
3. **Hybrid Q/D drift (0.0015/0.007)**: likely downstream of (1) (same value
   head) + KDA recurrence precision. Re-measure after (1).
4. **KDA parity hardening** (from `docs/museSpark-findings.md`, still valid):
   parity test covers one config only; add `local_conv=true`,
   `qkv_silu=false`, serpentine dirs 9–16, heads=16, gate/norm-off cases.
5. **Perf follow-ups (not correctness):** 819MB tensor arena at
   max_batch=256 (three slots of `max_tokens·max(emb,4168)`); transient
   256MB committed; MLH fc layers go through generic per-N graphs.
   Fine for now.

## 7. Repro commands

```powershell
# from C:\Users\Contrad\Documents\Code\repos\lc0-training\official-training-branch\libs\lc0
cmd /c build-dml.cmd                      # configure (first time) + build
.\build-dml\kda_parity_test_directml.exe --gtest_filter=DirectMlKdaParity.MatchesBlasOnNoEncoderNet
.\build-dml\kda_parity_test_directml.exe --gtest_filter=DirectMlKdaParity.MatchesBlasOnKdaHybridNet
.\build-dml\kda_parity_test_directml.exe --gtest_filter=DirectMlKdaParity.MatchesBlasOnKdaMhaNet
.\build-dml\kda_parity_test_directml.exe --gtest_filter=DmlSmoke.*
.\build-dml\kda_recurrence_test_directml.exe   # HLSL recurrence vs CPU (passes, unaffected)
```

## 8. Key files (all paths under `src/neural/backends/directml/`)

- `network_directml.cc` — orchestration: upload fix (~590), zero-init
  (~437), value scratch+copy (~649), transient 256MB (~379), MLH layers.
- `layers.cc` — all layers/graphs: `GraphFactory`+`WeightChannel`,
  `ReinterpretView`, `ActivationExpr`/`LayerNormExpr`/`RmsNormExpr`,
  `KdaRecurrenceLayer`, `MhaTransposeLayer`, `AttentionBody`,
  `EncoderBlock::{EvalKda,EvalMha}`, `AttentionPolicyHead` (scores+finalize),
  `PolicyMapLayer` (inverse map), `ValueHead`, `FCLayer`/`EmbeddingLayer`
  (`BuildGemmLayerOp` — MLH throw site), `DispatchOperator`.
- `layers.h` — `DmlWeightUploader` (owning), `MhaTransposeLayer`,
  `mha_{qkv,attn,tail}_compiled_` maps, `indices_host_`/`pos_encoding_host_`.
- `dml_common.h` — arenas, descriptor pool, `DmlDeviceContext`
  (+`DispatchOperator`, fences), `DmlExecScope`.
- `test_kda_parity_directml.cc` — nets, `CompareBackends` (kTol=2e-4),
  smoke scaffolding.
- `shaders/*.hlsl` + `*_shader_source.h` — keep pairs byte-identical.

## 9. Dead ends already ruled out (do not retry without new evidence)

Input upload path (fixed), GPU hang (fixed, lifetime), scores orientation
(original Q·Kᵀ correct), policymap direction (inverse correct),
Reinterpret-empty-strides (fixed), transient size (256MB),
graph/input/output *counts* (match), binding order (matches creation order),
strided biases per se (compile C=1..128), small-M GEMM (M=1..32 compile),
K/C sizes (all compile standalone), activation values (RELU=2 verified),
retention/count/retention-of-temp/plain-builds/Gather-builds (all compile
after), descriptor-pool overflow (bounds-checked, 320/8192 used),
fence waits (sound), command-list reuse (sound), weight offsets (verified
prefix + audit), bit→square mapping (matches BLAS `EncodePlanes`),
pos-encoding dims (176 both sides), Eigen row/col-major (re-derived twice;
empirics win).
