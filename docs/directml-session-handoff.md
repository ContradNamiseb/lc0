# DirectML Backend — F1-F9 Audit Closeout & SYCL-Speed Campaign Summary

**Date:** 2026-09-07. **Author of this session:** Claude Opus (Claude Code).
Primary reviewer for most of this session: gemini-antigravity (agora thread 19
#588 requested this document). gemini-antigravity then exhausted its weekly
quota mid-session; muse-spark took over as interim primary reviewer at the
user's direct instruction (#600-#606) for everything from the GPU-Based
Validation re-run onward, including the true MetaCommand root-cause fix in
section 2 below. codex-sol reviewed throughout as secondary (#587, #589,
#591, #601).
**Repo:** `lc0-training/official-training-branch/libs/lc0`, branch
`feature/directml-sycl-speed`.
**Machine:** Windows 11, Intel(R) Iris(R) Xe Graphics (integrated),
system DirectML, MSVC/clang-cl toolchain via `vcvars64.bat`.
**Build:** `build-dml.cmd` from repo root (uses `build-dml/` meson dir).
**Test binaries:** `build-dml\kda_parity_test_directml.exe`,
`build-dml\kda_recurrence_test_directml.exe` (both gtest).

This document is the closeout summary for the codex-sol-originated F1-F9
correctness/robustness audit (agora thread 19 #560), the related SYCL-speed
work, and the MetaCommand-lifecycle investigation that grew out of it. It is
a different document from [`docs/directml-handoff.md`](directml-handoff.md),
which is the 2026-09-03/04 correctness-bring-up log (Muse Spark / ZCode
sessions) for getting `DirectMlKdaParity.*` from failing to passing in the
first place. That doc's "current status" table is now stale -- every net it
lists as FAIL now passes (60/60 in the current parity suite) -- but its
driver-quirk rules (§4) and dead-ends list (§9) remain valid engineering
knowledge and are not repeated here.

## 1. Audit package: what shipped, in commit order

All on `feature/directml-sycl-speed`, all with a dual-hash manifest
(raw sha256 + `git hash-object`) posted to agora thread 19 before commit,
per this project's standing review protocol:

| Commit | Finding | Fix |
|---|---|---|
| `fb3884d` | Phase 3 Step 1 | `min_batch_size_` default `min(4,max)` → `min(1,max)` |
| `8623963` | F5 (codex-sol) | `AssertFiniteOutputs` oracle hardening -- test harness previously let a stray NaN through a worst-diff/argmax search as a false pass |
| `1fd5d99` | F1 remainder (codex-sol) | D3D12 resource-state barriers on the value/moves-left head copy source (`scratch_arena_`), previously never transitioned |
| `1251bf4` | tooling | `LC0_DML_DEBUG_LAYER=1` opt-in D3D12 debug layer + `ID3D12InfoQueue1::RegisterMessageCallback` routing validation messages to `CERR` (invisible otherwise without an attached debugger) |
| `a7317c5` | F7 + F9 (codex-sol) | F7: policy-slot sizing used `max_tokens` instead of `max_batch_size_` (64x overallocation, ~792MB across 3 rotating tensor slots). F9: `WaitForFence` treated the `UINT64_MAX` device-removed sentinel as "done," reading outputs from a computation that never finished |
| `1b32741` | F6 + F8 (codex-sol) | F6: `F32toF16Bits` used a pre-truncated 10-bit mantissa in its Inf/NaN and subnormal-rounding paths, silently turning some NaN payloads into Infinity and losing the sticky bit at the subnormal round boundary. F8: `ComputeBlocking()` on an empty batch reached `forwardEval` with an empty `planes_`, underflowing a `size_t` index |
| `63518de` | test coverage | In-tree `DirectMlRegressionCoverage.{EmptyBatchNoCrash,Fp16ConverterEdgeCases}`, satisfying codex-sol's #578 condition that F6/F8 not rest on an out-of-tree bite-test alone |
| `5bc77f8` | user-directed | Fixed a missing semicolon and documented redundancy in a user-added defensive empty-batch guard (harmless, kept per the user's explicit request) |
| `dcf6cba` | tooling | `LC0_DML_GBV=1` opt-in GPU-Based Validation (a second, independent opt-in nested inside `LC0_DML_DEBUG_LAYER` -- GBV's shader-instrumentation overhead, measured ~78x on this suite, must never turn on just because the cheap message-callback plumbing was wanted) |
| `d0695cb` | root-cause fix | `IDMLOperatorInitializer` -- see section 2, this is the actual fix, not a workaround |

F2 (buffer SRV/UAV binding pattern without explicit barriers) went through
three dispositions this session -- cleared, reopened, re-cleared -- entirely
because the tooling used to check it improved partway through. See section 2
for the full story; the short version is that the final GBV-based re-clearance
is the one that stands.

F3/F4/F11/F12 from codex-sol's original #560 review are explicitly
out of scope for this checkout (a different repository/checkout, per
gemini's scoping).

**Current suite state:** 60/60 `DirectMlKdaParity` (2 env-gated real-net
tests skip by design) + 2/2 `DirectMlRegressionCoverage` + 4/4
`KdaRecurrence` (1 env-gated test skips by design). Parity suite wall-clock
dropped from ~85s to ~56-58s after F7's VRAM reduction (less
allocation/zeroing overhead across 88 test fixtures); the `IDMLOperatorInitializer`
fix (`d0695cb`) added a small, roughly-constant per-test-binary startup cost
(one extra Reset/record/Close/Execute/Signal/Wait round trip at load) that
was not separately isolated and measured against the F7 baseline -- flagging
that as unmeasured rather than claiming it is negligible.

## 2. MetaCommand lifecycle: true root cause, the fix, and F2's final disposition

This section replaces an earlier draft of this same file (never committed)
that recommended `--backend-opts=meta_commands=false` as a workaround for
Intel Iris Xe. That workaround framing turned out to be treating a symptom;
the paragraphs below are the corrected version, in the order the
investigation actually happened.

### 2.1 First observation (agora #584/#585)

The `LC0_DML_DEBUG_LAYER` tooling (commit `1251bf4`), built to check F2,
also surfaced an unrelated finding: DirectML's own internal MetaCommand
lifecycle was producing D3D12 validation messages --
`ID3D12CommandQueue1::ExecuteCommandLists: The MetaCommand (...) was used in
ExecuteMetaCommand before calling InitializeMetaCommand with the same
MetaCommand` (ERROR severity) and `ID3D12GraphicsCommandList::CreateMetaCommand:
MetaCommand parameters are not supported by the current system configuration.`
(MESSAGE severity, a separate, unrelated notice about parameter support, not
initialization ordering -- it is unaffected by everything in this section).
At the time, `--backend-opts=meta_commands=false` was measured to eliminate
the ERROR-severity messages with no measurable throughput cost on this
hardware (batch 16: 547,580 vs 596,593 nps; batch 32: 617,601 vs 582,596 nps
-- both deltas inside this machine's ~12% run-to-run noise floor), and was
provisionally recommended as an Iris-Xe-specific default (#588, concurred
#589), with cross-vendor validation explicitly left open.

### 2.2 The real root cause (codex-sol #591)

codex-sol's independent review found the actual explanation by reading the
DirectML API contract directly: every compiled operator must be initialized
exactly once, via its own `IDMLOperatorInitializer`, before its first real
dispatch -- independent of `PersistentResourceSize`. This backend never did
that anywhere. DirectML's own internal MetaCommand selection (used for
GEMMs/convolutions when `meta_commands=true`, the default) performs its own
`InitializeMetaCommand` call as part of that missing initialization step --
which is exactly why skipping it produced the exact ERROR text above.
`meta_commands=false` had only ever been routing around this gap (its
fallback compute-shader path doesn't register with the MetaCommand tracking
runtime at all), not fixing it. codex-sol also flagged that the debug-layer
tooling only ever called `EnableDebugLayer()`, which validates API
parameters and CPU-tracked resource state -- not GPU-timeline hazards -- so
F2's earlier "zero validation errors" clearance (ratified #573) had only ever
had CPU-layer coverage. Both points were independently verified against
source before being accepted (#593), and gemini ratified the root-cause
reading in #594.

### 2.3 GPU-Based Validation reopens F2 (agora #597/#598/#599)

`LC0_DML_GBV=1` tooling was added (`dcf6cba`) to get real GPU-timeline
validation coverage. A full-suite run (60/60 parity + 2/2 regression,
`LC0_DML_DEBUG_LAYER=1 LC0_DML_GBV=1`, ~78x overhead, 75.6 minutes wall time)
produced 261 ERROR-severity messages, split into two different things: 199
were the already-known MetaCommand-lifecycle error scaled to the full suite,
and **62 were new**: `GPU-BASED VALIDATION: Dispatch, Incompatible resource
state` -- a root-parameter SRV read hitting a resource GBV's shadow tracker
believed was in `UNORDERED_ACCESS` state. This was new information the
CPU-layer-only validation had never surfaced, so F2 was reopened rather than
re-cleared on the spot.

### 2.4 Dispatch-site correlation (agora #601)

Rather than accept 62 errors as 62 unrelated sites, they were traced
precisely: temporary PSO-identity tags were added at every compute-shader
PSO's creation site, three single-test GBV runs were done across different
net shapes (`MatchesBlasOnKdaMlhNet`, `MatchesBlasOnLocalConvNet`,
`MatchesBlasOnKdaMhaNet`), and each run's error's `Pipeline State:` pointer
was matched against that same run's own PSO tags (valid because pointers are
only compared within one process). **All three landed on the exact same
site**: `AttentionBody::Eval`'s `record_preprocess` lambda, its first
invocation, reading the network's raw input-planes tensor via a root SRV.
Traced further into `network_directml.cc`'s `forwardEval`: immediately
before that dispatch, the input-planes tensor is explicitly transitioned
`UNORDERED_ACCESS -> COPY_DEST`, copied into, then transitioned back to
`UNORDERED_ACCESS` -- and the very next GPU-timeline operation binds that
same resource as a root SRV, with no further transition. An explicit
transition barrier is present; it just leaves the resource in a state a
naive SRV/UAV state machine would call incompatible with the next access.
Whether that is a real hazard was read as turning on the buffer-specific
D3D12 carve-out this session had already found real evidence for (the debug
layer's own `CreateCommittedResource` notice, "Buffers are effectively
created in state COMMON," implies buffer resource-state transitions are
host-side bookkeeping with no real hardware effect) -- consistent with, not
proven by, this exact pattern producing bit-identical BLAS-matching output
across the entire suite and every real net, all session. The temporary
diagnostic tags were reverted (not committed) once the correlation was done.
muse-spark (interim primary, #602) accepted this reading. codex-sol has not
posted since #591 (timed out per #595); no challenge from it is recorded,
but that is an absence, not an acceptance -- correction directed by
muse-spark in #609 against an earlier draft of this paragraph that overstated
codex-sol's position. The 62 were explained, not proven safe beyond doubt.

### 2.5 The fix (commit `d0695cb`)

`DmlDeviceContext::InitializeCompiledOperators` batches every compiled
operator this backend has ever created (tracked in a new `all_ops_` vector)
into one `IDMLDevice::CreateOperatorInitializer` call, binds a
`DML_BINDING_TYPE_NONE` output per operator (none have a persistent
resource -- checked, and would throw if that invariant ever broke), binds
the initializer's own temporary-resource requirement if nonzero, and records
the dispatch. Called once, right after the existing load-time batch-ladder
`EnsureCompiled` loop -- the one point guaranteed to run after every operator
across every valid batch size has been compiled and before any real eval,
matching this backend's existing two-phase-compile discipline
(`docs/directml-handoff.md` section 3: nothing may be compiled after the
first batch runs). Caught and fixed one bug in the draft before it was ever
built: the temporary-resource buffer was originally a local `ComPtr`, which
would have freed it the instant the function returned, while the caller's
subsequent Execute+Wait still had a queued GPU read pending against it --
`docs/directml-handoff.md` section 2.2's exact device-hang mechanism, for a
different buffer. Fixed by storing it in a context member instead.

### 2.6 Final verification (agora #604/#606)

Full GBV re-run, same suite, same protocol as 2.3, on `d0695cb`:

| Severity | Before (`5bc77f8`) | After (`d0695cb`) | Delta |
|---|---|---|---|
| ERROR | 261 | 62 | **-199**, exactly the known lifecycle-error count |
| MESSAGE | 35954 | 35954 | unchanged (parameter-support notice, unrelated to init ordering) |
| WARNING | 641 | 641 | unchanged (benign buffer-COMMON-creation notices) |
| CORRUPTION | 0 | 0 | unchanged |

Confirmed by direct count: zero `ExecuteMetaCommand before calling
InitializeMetaCommand` messages remain; all 62 remaining ERRORs are the
already-explained `Incompatible resource state` pattern from 2.4, at the
same single dispatch site. Functional: 60/60 parity + 2/2 regression + 4/4
recurrence all pass, zero output change (this only changes initialization
lifecycle, not any computed value).

**F2 disposition: re-cleared, on GBV coverage this time, not CPU-layer-only
coverage.** The 62 SRV/UAV notices are read as GBV's shadow resource-state
tracker not modeling the buffer-specific COMMON-state carve-out, not a
functional defect -- see 2.4 for exactly how confident that reading is (not
proven beyond doubt, but consistent with everything measured).

**The `meta_commands=false` recommendation from 2.1 is now superseded, not
merely optional.** With the real root cause fixed, `meta_commands=true` (the
default) now produces zero MetaCommand-lifecycle errors on its own -- there
is no longer a known reason to prefer `false` on Iris Xe specifically. This
has **not** been re-verified with a fresh throughput/error A/B against the
post-fix tree; that re-verification, and the original cross-vendor question,
are both listed as open items below rather than assumed.

## 3. Roofline analysis (gemini #588) -- analytical estimate, not a measured ceiling

Per codex-sol's #589 condition, the following is gemini-antigravity's
analytical estimate from published Iris Xe specs and one measured stage
duration, **not an independently measured hardware ceiling, and not
verified by claude-opus against a vendor profiler**:

- Intel Iris Xe Graphics (96 EUs @ ~1.4GHz): ~2.15 TFLOPs theoretical peak
  FP32 compute, ~40-68 GB/s theoretical peak memory bandwidth (shared
  LPDDR4x/DDR4).
- On `kda-native-935532` at batch 32 (2048 tokens, 3 KDA blocks): compute
  per eval for the KDA projections estimated at 3x1.24 = 3.72 GFLOPs;
  measured `kda_proj` stage duration 16.99ms implies ~219 GFLOPs achieved,
  ~10.2% of theoretical peak.
- Interpretation offered: small non-square GEMMs (M=2048, K=128,
  N=32...512) with frequent threadgroup synchronization and low arithmetic
  intensity explain the low utilization; three independent fusion attempts
  this session (QKV concatenation, intermediate stride packing, kda_tail
  graph fusion) each measured a null result (0.0%, 0.0%, -0.46%).

Treat the "~10% of theoretical peak" figure and any "architectural limit"
language as a hypothesis consistent with the measured null-fusion results
on this specific net and hardware, not a proven universal ceiling --
peak FLOPs/bandwidth figures are vendor-published theoreticals, and a
different net shape or driver could change the achievable fraction.

## 4. SYCL vs DirectML comparison (as reported by gemini #588)

The following comparison figures are as stated by gemini-antigravity in
#588. The DirectML-side batch-1 figure (156-174 nps) is consistent with
this session's own Phase 3 Step 1 measurement of the `min_batch_size_`
default change (~156-171 nps observed against an ~86 nps prior default).
The SYCL-side figures were not independently re-measured by claude-opus in
this session -- flagging that provenance gap rather than presenting them
as freshly verified:

- Batch 1: SYCL Level Zero ~117 nps vs DirectML (commit `fb3884d`) 156-174 nps.
- Batch 16: DirectML ~580-595 nps vs SYCL ~580-600 nps (parity within noise).
- Batch 32: DirectML ~590-610 nps vs SYCL ~600-640 nps (parity within noise).

## 5. Repro commands

```powershell
# from C:\Users\Contrad\Documents\Code\repos\lc0-training\official-training-branch\libs\lc0
cmd /c build-dml.cmd
.\build-dml\kda_parity_test_directml.exe --gtest_filter="DirectMlKdaParity.*:DirectMlRegressionCoverage.*"
.\build-dml\kda_recurrence_test_directml.exe

# Full-suite GBV verification (section 2.3/2.6; ~75min, expect ERROR=62,
# CORRUPTION=0, zero "ExecuteMetaCommand before calling InitializeMetaCommand"):
$env:LC0_DML_DEBUG_LAYER = "1"
$env:LC0_DML_GBV = "1"
.\build-dml\kda_parity_test_directml.exe --gtest_filter="DirectMlKdaParity.*:DirectMlRegressionCoverage.*" > gbv_run.log 2>&1
Remove-Item Env:\LC0_DML_DEBUG_LAYER
Remove-Item Env:\LC0_DML_GBV
Select-String -Path gbv_run.log -Pattern "D3D12 debug layer \[1\]:" | Measure-Object   # ERROR count
Select-String -Path gbv_run.log -Pattern "ExecuteMetaCommand before calling InitializeMetaCommand"  # must be empty

# MetaCommand A/B (net path is this session's export location; superseded
# per 2.6, kept here only as the repro recipe if the A/B is ever redone):
$net = "C:\Users\Contrad\Documents\runpod-results\networks-directml\kda-native-935532.pb.gz"
.\build-dml\lc0.exe backendbench --backend=directml --weights="$net" --batches=100 --start-batch-size=16 --max-batch-size=16
.\build-dml\lc0.exe backendbench --backend=directml --backend-opts=meta_commands=false --weights="$net" --batches=100 --start-batch-size=16 --max-batch-size=16
```

## 6. Open items

1. **Re-verify the `meta_commands=false` question against the post-fix tree**
   (section 2.6) -- the original recommendation was superseded by the real
   fix, but not re-measured. A fresh throughput/error A/B on `d0695cb` (or
   later) would confirm whether `meta_commands=false` still has any reason
   to exist as a per-hardware override, and cross-vendor/cross-driver
   validation remains untested regardless (only ever run on this one Iris Xe
   / driver combination).
2. The roofline "~10% of peak" reading (§3) is an estimate for this
   net/hardware pair, not re-derived for other net shapes.
3. The `IDMLOperatorInitializer` fix's own load-time cost was not isolated
   and measured (§1) -- only the full-suite wall-clock delta is known, and
   the suite constructs a fresh network (and pays this cost) per test.
4. F3/F4/F11/F12 from codex-sol's original #560 review remain out of
   scope for this checkout.
