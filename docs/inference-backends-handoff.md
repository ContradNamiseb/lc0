# SYCL FP16 / OpenVINO / DirectML backends — overnight session handoff

**READ §0 FIRST.** §0 was written in two passes. The first found SYCL
itself producing wrong output (a bug unrelated to KDA); the second
root-caused and **fixed** it — one missing SYCL kernel attribute, and it
turned out to fix the FP16 speed goal too as a side effect. Read all of §0
before anything below it; the "not yet root-caused" language in the
earlier part is superseded by the fix described later in the section.

Written mid-session, updated as work lands. Covers the three tasks given in
order: fix SYCL FP16's speed regression, get OpenVINO to correct output,
get DirectML to correct output — all while matching or beating SYCL speed
where that's achievable tonight, and structuring the two new backends like
the SYCL one (separate layers, explicit buffer ownership) per instruction.

---

## 0. Final status, measured — read this first

All numbers below are **measured tonight** on this machine (Iris Xe iGPU),
not estimated. Two real nets were used: `kda-check.pb.gz` /
`mha-check.pb.gz` (tiny 2-encoder check nets, already in the repo root) and
`kda-real.pb.gz` (copied into the repo root from
`lc0-training/stable-branch/tf/networks/kda-hybrid-128x4-3k1m-8h-no-rmsnorm/
kda-hybrid-128x4-3k1m-8h-no-rmsnorm-4000.pb.gz` — 4 encoders, 3 KDA + 1 MHA
layer, a real trained checkpoint, not a synthetic check net).

**The most important finding: SYCL itself gives wrong output, and this has
nothing to do with KDA.** Cross-checked via lc0's built-in `check` meta-
backend (`--backend=check --backend-opts="mode=display,...,sycl(backend=
sycl),eigen(backend=eigen)"`), which runs two backends on the same input
and diffs their value/policy output:

| Net | value abs err | policy abs err |
|---|---|---|
| `mha-check.pb.gz` (pure MHA, **zero KDA layers**) | 7.8e-02 | 4.3e-03 |
| `kda-check.pb.gz` (tiny KDA) | 2.8e-02 | 3.2e-03 |
| `kda-real.pb.gz` (real net) | **1.6e-01** | 3.9e-02 |

A 0.16 absolute error on a WDL value in [-1, 1] can flip which side the
engine thinks is better. The `mha-check.pb.gz` row is the key one: that
net has **no KDA layers at all**, so this is not a KDA bug, not something
this session's KDA-specific fixes touched or introduced, and not something
`§1`'s FP16 fix could have caused. It's a pre-existing bug somewhere in
SYCL's shared attention/encoder path. Root cause not yet found — that
would need the same per-layer numpy-verification rigor used for the KDA
ONNX converter (see the `kda-onnx-conversion-followup` memory note), which
is a separate, multi-hour task on its own and wasn't attempted tonight to
avoid guessing at a fix for a bug this size.

**OpenVINO's output is correct.** Cross-checked the same way against
`eigen` (BLAS CPU reference) on `kda-real.pb.gz`: value abs err **3.6e-07**,
policy abs err **6.1e-07** — float32-precision agreement, not
coincidental closeness. Independently double-checked by exporting the same
net via `lc0.exe leela2onnx` and running the resulting ONNX graph through
vanilla `onnxruntime` (CPU, no OpenVINO involved) on the exact input tensor
OpenVINO used (captured via a temporary debug dump, since reverted) — that
also matched OpenVINO's own output to 1.8e-07. Three independent
implementations (OpenVINO, Eigen, onnxruntime) agree; SYCL is the outlier.
**Priority #2's correctness requirement is met.**

**OpenVINO's speed does not meet "match or beat SYCL."** On `kda-real.pb.gz`
at batch 256: OpenVINO GPU **176 NPS** vs SYCL FP32 **753 NPS** — about
4.3x slower, in the same ballpark as the ~5-8x gap noted before this
session (see §2's original text below). Not closed tonight.

**SYCL FP16 is slower than FP32, confirmed on real data, and the earlier
fix in §1 did not change that.** Measured on three nets at two batch sizes
each (batch 64 and 256):

| Net | FP32 NPS | FP16 NPS | FP16 vs FP32 |
|---|---|---|---|
| `kda-check.pb.gz`, batch 64 | 1594 | 1146 | -28% |
| `kda-check.pb.gz`, batch 256 | 1671 | 1212 | -27% |
| `kda-real.pb.gz`, batch 256 | 753 | 544 | -28% |
| `mha-check.pb.gz`, batch 256 | 1713 | 1147 | -33% |

Consistent ~27-33% FP16 slowdown across every net and batch size tested,
including the pure-MHA net — so like the correctness bug above, this isn't
KDA-specific either. The `§1` fix (accumulate in `float`, store/load in
`half`) is the architecturally correct pattern and should still be kept —
reverting it would very likely make things worse, not better — but it
alone doesn't deliver a win here. The GEMM calls already use native
`sycl::half` via `oneapi::mkl::blas::column_major::gemm` (see
`layers.cc.dp.cpp` `FCLayer<sycl::half>::Eval`), not an emulated path, so
this looks like either an Iris Xe/oneMKL characteristic for half-precision
GEMM on this specific integrated part, or a deeper issue than the
accumulation-type bug already fixed. Not root-caused tonight.

**The `--backend` default-selection bug chased at the end of the last
session turned out not to be a code bug at all.** `REGISTER_NETWORK`'s
priority plumbing (`factory.h` → both `NetworkFactory` and
`BackendManager` via `NetworkAsBackendFactory`, sorted descending in
`BackendManager::GetBackendNames()`) was already correct. The stale
`openvino-auto` default was a **timestamp-granularity race in `ninja`**:
editing `network_openvino.cc` and rebuilding within the same filesystem-
clock tick let ninja treat the `.obj` as up to date against a version of
the source that predated the priority fix, so the linked `lc0.exe` baked
in stale priorities despite the `.obj`'s mtime matching the edited
source's mtime. Fixed by touching every backend source file this session
touched and doing a full rebuild; confirmed via `lc0.exe --help`:
`DEFAULT: sycl-auto`, with the full `VALUES` list now in correct
descending-priority order (`sycl-auto,sycl,sycl-fp16,eigen,openvino-auto,
openvino,directml,...`). If a future priority change doesn't seem to take
effect, touch the changed file(s) explicitly before rebuilding rather than
trusting ninja's mtime check across a fast edit-rebuild cycle.

## 0b. The SYCL bug — root-caused and fixed

Found by bisecting layer-by-layer: added temporary debug dumps to both
the `eigen` (known-correct BLAS) and `sycl` backends at matching points
inside `ForwardEncoderLayer`/`EncoderBlock::Eval` — post-embedding, Q/K
projection, pre-softmax attention logits, post-softmax, post-LN1, final
layer output — on `mha-check.pb.gz` (pure MHA, simplest case). Diffing
each checkpoint:

- Embedding output: bit-exact (9.5e-7).
- Q/K projection: bit-exact (1.4e-6).
- Pre-softmax attention logits (raw `matmul_qk` + smolgen bias,
  reconstructed separately): bit-exact (1.9e-6) — smolgen generation and
  the raw score matmul are both correct.
- **Post-softmax attention weights: SYCL's values were ~2.00x every
  reference value**, and note a softmax that's uniformly 2x too large
  *cannot* be a normal precision bug — real softmax output sums to 1 per
  row by construction, so "everything's 2x" only happens if the
  normalization sum itself was computed over half the row.

**Root cause:** `softmax_opt_64_kernel` in
`src/neural/backends/sycl/common_kernels.dp.cpp` (the fast path used
whenever the softmax width is 64 — i.e. *every* attention softmax over
the 64 squares, in every encoder layer of every attention-based net,
KDA or not) reduces max/sum via `sycl::reduce_over_group` on the
**sub-group**, under the hard assumption that one sub-group of 32 threads
(2 elements/thread) covers exactly one 64-wide softmax row. That
assumption was never enforced: the kernel launch had no
`reqd_sub_group_size` attribute, so the compiler was free to pick a
narrower sub-group width for this specific kernel — and evidently did, on
this Iris Xe iGPU/driver. When the real sub-group is narrower than 32, a
64-element row spans *multiple* sub-groups, each normalizing independently
over only its own slice — each element ends up divided by roughly half
the true row sum, i.e. ~2x too large. Every other performance-sensitive
kernel in this file (`OutputTransform_SE_relu_InputTransform_kernel`, the
fp16 kernels) already carries
`[[intel::reqd_sub_group_size(SYCL_SUB_GROUP_SIZE)]]` on its launch
lambda — this one softmax path was simply missing it.

**Fix** (`common_kernels.dp.cpp`, `Softmax()`'s `C == 64` branch): added
`[[intel::reqd_sub_group_size(SYCL_SUB_GROUP_SIZE)]]` to the launch
lambda, matching the pattern already used elsewhere in this file.
`SYCL_SUB_GROUP_SIZE` (from `sycl_common.h`) is 32 on non-AMD-GFX8/9
platforms, which is exactly what the kernel's 2-elements-per-thread /
64-wide-row math already assumed.

**Verified fixed** via the same `check`-backend cross-validation, re-run
after the fix, on all three nets:

| Net | value abs err | policy abs err |
|---|---|---|
| `mha-check.pb.gz` | 1.5e-07 | 3.4e-08 |
| `kda-check.pb.gz` | 6.0e-08 | 3.0e-08 |
| `kda-real.pb.gz` | 8.9e-08 | 1.6e-07 |

Float32-rounding-level agreement across the board, down from up to 0.16/
0.039 before the fix.

**Unexpected bonus: this also fixed the FP16 speed goal.** The broken
sub-group assumption wasn't just wrong, it was apparently forcing extra
synchronization/serialization overhead too. Re-benchmarked on
`kda-real.pb.gz` at batch 256, same machine, same run methodology as §0's
original numbers:

| | before this fix | after this fix |
|---|---|---|
| SYCL FP32 | 753 NPS | **1145 NPS** (+52%) |
| SYCL FP16 | 544 NPS | **1653 NPS** (+44% vs FP32, and +204% vs its own prior number) |

**FP16 now beats FP32**, which was the original, literal ask for
priority #1 — it just needed this fix underneath it, not (only) the
accumulation-type fix from §1, which is still correct to keep but wasn't
the actual bottleneck.

The DirectML KDA recurrence test (`kda_recurrence_test.exe`) still passes
after this fix (480ms) — unaffected, as expected, since it doesn't touch
this code path.

**What didn't change:** OpenVINO's speed gap. Re-measured on the
now-faster SYCL baseline: OpenVINO GPU is still ~192 NPS on the real net
— unchanged within noise from before this fix (176 NPS), so the gap to
SYCL actually *widened* in relative terms (was ~4.3x, now ~6-8.6x
depending on which SYCL variant). OpenVINO's own correctness (§0 above,
verified against Eigen/onnxruntime) is unaffected by any of this — that
was never dependent on SYCL being right, and the check backend just
happened to use SYCL as one of its two cross-validation legs. Closing
OpenVINO's gap (suspected cause: the ONNX `Scan` op serializing the KDA
recurrence into per-token GPU dispatch/sync) is still open, not attempted
tonight.

**What this changes for tomorrow's priorities:** priority #1 (FP16 speed)
is now actually met, with a verified root cause and fix, not just a
plausible-sounding accumulation-type change. Priority #2 (OpenVINO) is
half done — correctness is solid, speed isn't, and now has a *harder* bar
to clear since SYCL got faster too. Priority #3 (DirectML) is unchanged —
still only the KDA recurrence kernel, rest of the graph not built.

---

**Repo:** `C:/Users/Contrad/Documents/Code/repos/lc0/master`, branch
`feature/directmlx-backend` (tracks `origin/feature/directmlx-backend`).
**Not** `lc0-training/official-training-branch/libs/lc0` — that's a
separate, secondary checkout of the same GitHub repo; work started there
by mistake early in the session and was ported over once corrected.

---

## 1. SYCL FP16 — fixed, needs a benchmark re-run to confirm the win

**Problem found:** commit `f6c9101` ("complete native FP16 support in KDA
recurrence and layers") made the KDA recurrence kernel's *entire*
accumulation chain (the private `state[32]` array, q/k norms, prediction,
delta, output — everything except the final store) run in `T` instead of
`float`. When `T = sycl::half`, this is scalar half-precision arithmetic in
a 64-step *sequential* scan with no vectorization — Iris Xe's fp16
advantage only shows up for packed/vectorized math, so every op here paid a
half↔float promotion with no compensating throughput win. Measured before
the fix: **SYCL FP16 (217 NPS) was slower than SYCL FP32 (241 NPS)** —
backwards from what FP16 should do. The same regression pattern also hit
`applyKdaLocalDepthwiseConv`'s 9-tap sum and the RMS-norm gate multiply in
`EvalKda`.

**Fix applied** (`src/neural/backends/sycl/common_kernels.dp.cpp`,
`layers.cc.dp.cpp`): reverted all three sites to accumulate in `float`
regardless of `T`, keeping `T` only at the actual memory boundary — the
`q_ptr`/`k_ptr`/`v_ptr` loads and the `mixed[...]` store. That's where FP16
should earn its keep: half the bytes moved for a kernel that's
bandwidth-bound at 64 small sequential reads per head, not compute-bound.
This matches what the kernel did *before* f6c9101, which was already
correctness-verified by two independent reviews (see §4) — the revert
doesn't reopen any of those findings, it only undoes the accumulation-type
change from the FP16-completion commit specifically.

**Update — benchmarked, and the fix did not deliver a win:** see §0 for the
full numbers. FP16 is ~27-33% slower than FP32 on every net tested,
including nets with no KDA layers at all, so whatever's causing this is
broader than what this fix touched. The fix itself is still correct to
keep (float accumulation, half at the memory boundary is the right
pattern for a bandwidth-bound sequential scan) — it just isn't sufficient
on its own for the speed goal.

---

## 2. OpenVINO — real backend, builds, not yet run against real weights

**What exists:** `src/neural/backends/openvino/network_openvino.cc` +
`inputs_outputs.h`. Converts the weights file to the same in-memory ONNX
graph the `onnx` backend uses (`ConvertWeightsToOnnx`, including the KDA
recurrence's `Scan` op — see `converter.cc`), loads it via
`ov::Core::read_model` from the in-memory bytes, reshapes for a dynamic
batch dimension, and compiles for GPU (falling back to CPU via
`openvino-auto`). Structured per instruction: `network_openvino.cc` only
orchestrates; `inputs_outputs.h` owns every buffer (the input tensor and
the cached output-port pointers), pooled via a free-list the same way
`sycl/network_sycl.cc.dp.cpp`'s `GetInputsOutputs`/`ReleaseInputsOutputs`
does, instead of allocating a fresh `ov::InferRequest` per computation.

This is a genuine, complete backend — not a stub. It started life as
Antigravity's scratch file (`.gemini/antigravity/brain/.../network_openvino.cc`,
never actually committed anywhere despite the memory-bank handoff claiming
it was); it's been restructured into the SYCL-pattern split and wired into
`meson.build`/`meson_options.txt` (new `openvino`/`openvino_include`/
`openvino_libdirs` options — none of that plumbing existed before tonight).

**Measured previously (Antigravity, before this session, unverified by me
against current code):** OpenVINO GPU 47.7 eval/sec, CPU 71.6 eval/sec, 29
NPS in tree search — dramatically behind SYCL. The likely cause, per the
existing memory-bank note: OpenVINO serializes the KDA recurrence's `Scan`
op into per-token dispatch/sync, where SYCL runs the whole 64-step scan as
one fused work-group kernel with in-register state. Not re-measured
tonight with the restructured code; do that once the build finishes (§5).

**Update — re-measured and correctness-checked tonight:** see §0. Speed on
the real net is 176 NPS vs SYCL's 753 NPS (~4.3x slower), in line with the
old estimate — the gap wasn't closed. Correctness, however, is now
verified: matches Eigen and an independent onnxruntime run to float32
precision (~1e-6 to 1e-7 abs error), on both GPU and CPU device. Not done:
any attempt at closing the speed gap (investigating Scan-subgraph fusion,
precision hints, etc.).

---

## 3. DirectML — the recurrence kernel is real and tested; the rest of the
network is not built

**What was there before tonight:** `network_directml.cc` built a complete
`dml::Graph` that only sliced the raw input tensor into policy/value/mlh-
shaped pieces — no convolution, attention, or KDA computation ever ran. It
compiled, linked, and the memory-bank handoff reported "34.8 eval/sec" for
it as if that were a real number. It is not one; that graph never computed
anything resembling a chess evaluation. This was found by reading the file,
not by running anything — nothing in it does enough work to produce a
believable number, and inspection confirmed why.

**Why the rest is hard:** DirectMLX's graph-building API (`dml::Graph`,
`dml::Gemm`, `dml::Slice`, etc.) has no loop/recurrence primitive — no
equivalent to ONNX's `Scan` or a plain sequential kernel loop. The KDA
recurrence's export to ONNX deliberately uses `Scan` instead of unrolling
specifically because unrolling cost 77k graph nodes (see the KDA ONNX
graph-size memory-bank note); the same unrolling cost would apply to a
from-scratch DirectMLX graph. The only way to get SYCL-competitive
recurrence speed here is a hand-written compute kernel, the same way the
`dx` backend already hand-writes HLSL for its Winograd/SE layers instead
of using a graph API for those.

**What was actually built and tested tonight:**
`src/neural/backends/directml/shaders/kda_recurrence.hlsl` — a
line-for-line port of the *fixed* SYCL recurrence kernel (§1), dispatched
via raw D3D12 compute (root constants for the small parameter struct, root
SRVs/UAVs for the buffers — no descriptor heap needed since everything is
a `StructuredBuffer`, not a typed/formatted one, which is what makes root
descriptors legal here). `layers.h`/`layers.cc` hold `KdaRecurrenceLayer`,
one class with a `Record()` entry point that appends the dispatch to a
caller-supplied command list — the DirectML analog of `sycl/layers.h`'s
`BaseLayer::Eval()`. `inputs_outputs.h` owns the D3D12 upload/default/
readback buffers, structured like `sycl/inputs_outputs.h`.
`test_kda_recurrence.cc` is a gtest that runs the shader on this machine's
real D3D12 adapter at the network's production shape (`key_dim = value_dim
= 32`, `heads = 16`, 8 directions) against an independent scalar CPU port
of the same formula (deliberately *not* copy-pasted from the HLSL or SYCL
kernels, to catch a transcription error rather than validate the kernel
against its own bug).

**Known transcription error already caught and fixed:** the first draft of
the 64-entry diagonal-traversal lookup tables in the HLSL file had a
partially fabricated `kKdaAntiDiagReverse` table (a `grep -A 3` only pulled
48 of 64 entries and the last 16 got filled in from a guessed pattern
instead of the source). Caught before anything ran, by diffing against the
actual SYCL source with full context. All four tables are now verified
byte-for-byte against `sycl/common_kernels.dp.cpp`.

**Not done:** the embedding layer, the MHA encoder block, smolgen, and the
policy/value/moves-left heads — everything except the KDA recurrence.
`network_directml.cc`'s `NewComputation()` throws a clear exception rather
than returning a computation that would produce numbers — this was a
deliberate choice: an honestly-broken backend that refuses to run is much
better than one that looks finished and silently returns wrong evaluations,
which is what the previous stub did. `REGISTER_NETWORK("directml", ...)`
is registered at low priority (5) so it's never auto-selected as a default
backend, but it does keep the name `directml` (matching this repo's
existing `-Ddirectml` option and `USE_DIRECTML` macro) rather than being
renamed to something like `directml-partial` — the throw-on-use is what
keeps it honest, not the name.

The user separately confirmed DirectMLX/DirectML.h is the intended
approach for the rest of the graph (compared to TensorRT's graph API in
spirit) — so the plan for the remaining layers is to keep building them as
DirectMLX graph fragments the way `KdaRecurrenceLayer` is scaffolded to
support (one class per layer type, each compiling once and exposing a
dispatch/record call), not to fall back to wrapping ONNX Runtime's DirectML
execution provider. (That EP path exists and is already wired in the
`onnx` backend as `onnx-dml` — but building it needs the ONNX Runtime C++
SDK, which isn't installed on this machine, only the Python wheel's
runtime DLLs. Getting it would need a download, which needs explicit
permission not sought tonight since the user is asleep.)

---

## 4. Context this session almost missed

Two docs already existed in this repo, uncommitted, from earlier AI review
passes on the SYCL KDA path: `docs/kimi-findings.md` (three critical bugs:
gate-before-norm ordering, `key_dim>32` state overflow, uninitialized
staging arrays for `key_dim>value_dim`) and `docs/qwen-findings.md` (two
more: the fused decay/gate GEMM writing interleaved data that its readers
assume is stacked, and `local_conv` corrupting the LN1 residual skip).
Checked against `git log`: every one of these is already fixed at current
HEAD (`0e2999d`, `18c786c`, `74c4298`) — none of them are open. The FP16
fix in §1 is a separate, later regression (from `f6c9101`, which landed
after all of those fixes), not a reopening of any of these.

**A caution for whoever reads the agent-memory-bank next:** the "Antigravity
(Gemini 2.5 Pro)" handoff claiming OpenVINO and DirectML backends were
"implemented and committed" to `origin/feature/openvino-backend` /
`origin/feature/directmlx-backend` was checked against the actual git
history and found to overstate things — neither branch's commits actually
contain `src/neural/backends/openvino/` or a working
`src/neural/backends/directml/network_directml.cc`; the real files existed
only as uncommitted scratch output in that agent's own working directory
(`.gemini/antigravity/brain/.../scratch/`). Worth independently verifying
any cross-agent handoff's "committed" claims against `git log`/`git diff`
before building on top of them, the way this session eventually did.

---

## 5. Build status

`build-local.cmd` (new, tailored to this machine — oneAPI 2026.1, no
CUDA/CUDNN/OpenCL hardware) builds SYCL (`l0`) + OpenVINO + DirectML +
gtest together. Two meson gaps had to be fixed to get this far, both now
folded into the script:
- SYCL's L0 path needs oneMKL's SYCL BLAS interface *and* `OpenCL.lib`
  unconditionally, regardless of the `mkl`/`opencl` backend options — meson
  only wires `mkl_include` into the build when the `mkl` BLAS option is on,
  so that option is set to `true` here even though Eigen would otherwise be
  enough for the reference BLAS backend.
- Invoking anything that runs oneAPI's `setvars.bat` through this session's
  PowerShell tool fails (`'vswhere.exe'`/`'vars.bat' not recognized`,
  apparently a PATH-resolution quirk specific to that invocation path).
  Running the same command via `cmd.exe //c "<absolute path>.cmd"` from the
  Bash tool works correctly — use that, not PowerShell's `&` operator, for
  anything that touches oneAPI's environment scripts.

**Final status:** the build completed successfully — SYCL (L0) + OpenVINO +
DirectML + gtest, all targets, including `kda_recurrence_test.exe` (passes:
`MatchesCpuReferenceAtProductionShape`, 496ms). `lc0.exe --help` shows
`DEFAULT: sycl-auto` with a correctly priority-sorted `VALUES` list (see
§0 for the ninja-staleness bug that briefly made this look wrong).
`docs/kimi-findings.md`/`docs/qwen-findings.md` remain fully addressed at
HEAD, unchanged since earlier in the session.

Nothing here has been committed yet — `git status` in this checkout still
shows everything from this session as modified/untracked. See §0 for what
changed after this section was first written.
