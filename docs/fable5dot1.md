# OpenVINO backend — release-readiness review

Reviewed: `feature/openvino-backend` @ `b6ce47e` ("openvino: SE residual fused
op, merged custom-layer config, warmup"), files under
`src/neural/backends/openvino/`, plus the `meson.build`/`meson_options.txt`
wiring and the shared `neural/kda_directions.h` / `onnx/converter.cc` contract
the backend depends on.

**Not reviewed:** the uncommitted work described in the latest memory-bank
handoff (`fp16_safe` mixed precision, `fp16_safe` defaulting on for KDA nets,
QKV fusion in `converter.cc`, SiLU/serpentine work). Those live only in the
main checkout and are not on any branch yet. Several findings below assume
the committed state (`fp16` defaults to plain FP16 on GPU); re-check them once
that work lands.

## Verdict

**Not ready to release as-is.** The core inference path is sound and has been
verified against reference backends, but there are two correctness bugs that
affect real users (auto-selection priority, `min_batch=0` hang), one
robustness/security hazard in how the GPU kernel source is staged on disk,
zero automated test or CI coverage, and no user-facing documentation. None of
the blockers are large; all are fixable in a day or two.

Severity legend: **Blocker** = ship-stopping; **High** = fix before release;
**Medium** = fix soon after, or document as known limitation; **Low** =
nice-to-have / hygiene.

---

## Blockers

### B1. Backend priority is below blas/eigen, so auto-selection never picks it

`network_openvino.cc:1001-1002` registers `openvino` at 45 and `openvino-auto`
at 46, with the comment "Still above blas/eigen (49-50)". It is not:
`factory.h:97-100` sorts factories with `priority > other.priority`, i.e. a
**larger** number wins. blas (50) and eigen (49) therefore outrank openvino
(45/46). On a machine with an Intel iGPU and a blas build, `lc0` with no
`--backend` will silently run on CPU BLAS.

Fix: raise both to something in the 51-58 range (below the onnx providers at
59-65, above blas at 50), and re-word the comment. Given the measured numbers
(fp16 ~1965 nps vs blas on the same box, per handoff) it clearly should beat
the CPU fallbacks.

### B2. `min_batch=0` (or negative) hangs the process on KDA nets

`network_openvino.cc:530` accepts any `min_batch`, only clamping the upper
bound. `network_openvino.cc:768-773` then does
`for (int b = min_batch_; b < kMaxBatchSize; b *= 2)`. With `min_batch=0`,
`b` stays 0 forever and the constructor never returns (memory grows unbounded
as it pushes into `batch_buckets_`). Negative values loop forever too.

Fix: `min_batch_ = std::clamp(..., 1, kMaxBatchSize)`; optionally reject
non-positive values with an `Exception`.

### B3. No automated verification of the two hand-written kernels

`KdaScanOp::evaluate` (`kda_scan_op.cc:90-192`), the OpenCL
`kda_scan_kernel` (`kda_scan_kernel_source.h`), `SEResidualOp::evaluate`, and
`se_residual_kernel` all claim bit-for-bit parity with `converter.cc`'s Scan
body and `sycl/common_kernels.dp.cpp`'s `kdaRecurrenceValueParallel`, but
there is no gtest, no CI job, and the only evidence is manual `--backend=check`
runs recorded in the memory bank. The header comments say "if either of those
changes, this must change with it" — nothing enforces that.

Minimum for release: a gtest that runs `KdaScanOp::evaluate` on a small random
fixture against a straightforward reference recurrence (the NumPy/TF-verified
one from `docs/model-design.md` §15), for each of the 16 directions and for
`direction_count` in {1, 8}. A CI job that at least compiles the backend
(see H4) is separate but also needed.

---

## High

### H1. GPU kernel sources are staged in a fixed, shared temp path

`CustomLayerConfigDir()` (`network_openvino.cc:382-392`) always uses
`<temp>/lc0_openvino_kda_scan/`, and `WriteKdaScanGpuConfig` /
`WriteSEResidualGpuConfig` / `WriteMergedGpuConfig` overwrite fixed filenames
(`kda_scan.cl`, `se_residual.cl`, `lc0_custom_layers.xml`) with
`std::ios::trunc`. Consequences:

- **Concurrent processes race.** Two engines in a cutechess match, or
  multiple self-play workers, each truncate and rewrite the same three files
  while the other's `compile_model()` may be reading them. Same net → risk of
  reading a half-written file (build failure). Different nets → one process
  compiles the other net's `HEADS_/KEY_DIM_/VALUE_DIM_`, producing silently
  wrong output on the first infer or an out-of-bounds kernel.
- **Shared `/tmp` on Linux is a code-injection vector.** Another local user can
  pre-create the directory (`create_directories` succeeds on an existing dir)
  and own the files. The `std::ofstream` writes at `:420`, `:493`, `:401` are
  never checked for success, so a permission failure leaves the attacker's
  kernel source in place and lc0 happily loads it onto the GPU.
- Files are never cleaned up.

Fix: per-process unique directory (pid + random suffix, or
`std::filesystem::temp_directory_path()/lc0_openvino_<pid>_<rand>`), verify
`ofstream` state after each write (`if (!cl) throw ...`), and remove the
directory in `~OpenVinoNetwork`. On POSIX, `mkdir` with 0700.

### H2. Default `fp16=true` has no strength validation

`network_openvino.cc:666` defaults FP16 on for `device=GPU`. Measured
(handoff): plain fp16 policy abs error 1.4e-3 on kda-native-635532 vs 2.6e-4
for the uncommitted `fp16_safe` and ~0 for fp32. Older note 2429 measured
~1e-2 on a CNN net (needs `atol≈0.05` to pass `check`). No fixed-nodes or
match test exists for either number.

For release either (a) ship the `fp16_safe`-on-for-KDA default the handoff
describes and record the check-backend numbers in the docs, or (b) run a
fixed-nodes match (fp16 vs fp32, ≥200 games) and record the result. Do not
ship a default whose accuracy cost is only characterised on 10 positions.

### H3. Value/policy head selection is hard-coded

`network_openvino.cc:556-557` fixes `policy_head = "vanilla"` and
`value_head = "winner"`. The onnx backend exposes both as backend options
(`network_onnx.cc:948-951`). A multi-head net whose intended heads are e.g.
`optimistic`/`q` cannot be used correctly. Expose `policy_head` and
`value_head` options with the same defaults.

### H4. No CI build, no README entry, no documented requirements

- No `.github/workflows/openvino.yml`; the backend is never compiled by CI.
  A `sycl.yml`-style job installing the OpenVINO apt repo and building with
  `-Dopenvino=true -Dopenvino_include=... -Dopenvino_libdirs=...` is enough.
- `README.md` does not list the backend, its options (`device`, `fp16`,
  `min_batch`, `bucket_batches`, `warmup`, `cache_dir`, `ir_path`, `profile`,
  `se_fusion`), or the Intel GPU driver requirement.
- Minimum OpenVINO version is not stated or checked. The code uses
  `ov::hint::execution_mode` (2023.0+), `ov::compilation_num_threads`,
  opset-8 `Slice`/`Gather` matching, and the SimpleGPU CONFIG_FILE mechanism.
  Add a `#if OPENVINO_VERSION_MAJOR < 2023` `#error` or a runtime check on
  `ov::get_openvino_version()`.

### H5. Uncommitted work must land (or be dropped) before tagging

Per the handoff: `fp16_safe` + head-selection (`network_openvino.cc`), QKV
fusion (`converter.cc`), SiLU/serpentine work are all uncommitted in the main
checkout. The QKV fusion changes the exported ONNX graph that `ReplaceKdaScan`
pattern-matches on (`kda_scan_pass.cc:154-158` looks up inputs by
`/scan/q`… suffix and traces through Concat→Gather→Slice). Verify the pass
still matches post-fusion — the handoff says eigen parity was checked, so it
probably does, but it needs a commit and a re-run of `--backend=check`.

---

## Medium

### M1. `GetMiniBatchSize()` returns 1024 for every device

`network_openvino.cc:123`. For `device=CPU` this is far too large (onnx-cpu
defaults to 16) and will hurt latency badly; for a small iGPU 1024 also means
the warmup at `:839-898` JIT-compiles and runs a 1024-row inference for the
top bucket, which is real memory and time on a 96-EU part. Consider
`device == "CPU" ? 16 : 256` (or a `batch` option like onnx), and let
`kMaxBatchSize` stay 1024 as the hard cap.

### M2. Stale comments that will mislead the next maintainer

- `:209-214` says the IR path "still needs the forced-f32 precision hint" —
  the code at `:777-778` applies whatever `want_fp16` says, f16 by default.
- `:665` "2.3x-2.4x speedup" — current measurement is ~6.1x (1965 vs 321 nps).
- `:992-1000` says GPU throughput is "~5-8x slower than SYCL ... likely from
  OpenVINO serializing the KDA recurrence's ONNX Scan op". That was before
  `KdaScanOp` and the direction-table fold (notes 2432/2433); it is not the
  current state and it is the justification for the wrong priority in B1.
- `kda_scan_op.h:32-35` says the op is "direction-agnostic" and that
  converter.cc reorders q/k/v outside the Scan; since `f34ffc2` the kernel
  applies the permutation itself via `kDirectionTable` and the pass bypasses
  the graph's Gather. The header contradicts `kda_scan_kernel_source.h:29-32`.

### M3. `WriteKdaScanGpuConfig` silently defaults directions to 1..8

`network_openvino.cc:444-446`. The constructor comment at `:598-606` argues at
length that defaulting to {1..8} "would silently evaluate a net ... with the
wrong traversal order", and `converter.cc:687` throws on an empty set — yet
this function re-introduces the default. It is unreachable today (a KDA net
without directions never gets past the converter) but should `throw` for
consistency.

### M4. CPU device path for KDA nets depends on `evaluate()` in f32 only

`KdaScanOp::has_evaluate()` (`kda_scan_op.cc:78-80`) returns true only for
f32. On CPUs where the plugin picks bf16 (AMX-capable Xeons with
PERFORMANCE mode) the custom op has no implementation and `compile_model`
fails. Either force `inference_precision=f32` for CPU when a `KdaScanOp` is
present, or document `device=CPU` as f32-only.

### M5. SE fused kernel work-group size can exceed device limits

`se_residual_kernel` is dispatched with `local="1,F,1"` where `F=CHANNELS_`
(`network_openvino.cc:520`). Intel Gen9–Gen12 iGPUs cap work-group size at
256; nets with 320/384/512 filters will fail to launch. Off by default
(`se_fusion=false`, measured 16% slower) so not a blocker, but either guard it
(`channels > max_work_group_size → refuse`) or drop the option from the
release build.

### M6. Bucketing/warmup interaction with `MinibatchSize`

Buckets are `min_batch, 2·min_batch, …, 1024`. A user setting
`--minibatch-size=96` will be padded to 128 (33% wasted compute) on every
call. Document this, or derive buckets from the configured minibatch size
(needs the option plumbed through; the cuda backend does not do this either,
so acceptable as a documented limitation).

---

## Low / hygiene

- `network_openvino.cc:849-883`: the `for (const int shape : warm_shapes)`
  body is not indented; run clang-format.
- `kda_scan_kernel_source.h:56-63`: `int` indexing (`qk_base`) is fine at
  N≤1024 but add a comment on the bound, or use `size_t`/`long`.
- `AccumulateProfile` uses `std::map<std::string,…>` keyed by node name under
  a mutex on every infer when `profile=true` — fine for a diagnostic, but say
  so in the option's help text.
- `MakeOpenVinoNetworkAuto` constructs a throwaway `ov::Core` just to list
  devices; harmless but adds ~100 ms of plugin loading on startup.
- `meson.build:925` uses `cc.find_library` (C compiler) for a C++ library;
  matches the rest of the file, fine. `openvino_include` defaulting to `''`
  turns into `include_directories('')` = source root; harmless but should
  probably be `required` when `openvino=true`.
- `is_ir_model_` path (`:294-307`) allocates a fresh `ov::Tensor` per call.
  Documented as intentional; acceptable.
- The output-port lookup at `:573-583` matches by substring (`"value"` would
  also match a node named e.g. `"value_embedding"` if it were ever exposed as
  an output). Fine for the graphs converter.cc emits; add a comment that
  exactly one output per category is expected and throw on duplicates.

---

## What is in good shape

- **Correctness of the main path.** Own host buffers bound via `set_tensor`
  before `infer()` (fixes the get_tensor() fault, `:316-338`); zero-copy ROI
  view for the ONNX path with a documented reason the IR path can't use it;
  padding rows zeroed; only `batch_size_` rows read back.
- **Failure-loud pattern matching.** `ReplaceKdaScan` throws on every
  assumption it cannot verify (input naming, body constants, unconsumed final
  state, downstream Concat) instead of silently falling back — exactly right
  for a graph rewrite that would otherwise "produce wrong chess".
- **Geometry uniformity check** (`:709-758`) before baking one kernel's
  `-D` defines for every op instance.
- **Direction table** generated from `KdaSquareForToken` rather than
  transcribed (`:423-436`), and the `DIRECTIONS_LIST_` whitespace hazard
  (note 2434) is respected.
- **`cache_dir` refused when a custom layer is present** (`:800-806`) — the
  crash it prevents is well documented (note 2437).
- **Warmup is best-effort** and cannot fail startup (`:892-897`).
- **InputsOutputs lifetime** ordering is explicit and explained
  (`inputs_outputs.h:61-66`).
- Comments throughout record *measurements* with net IDs and numbers, which
  is what made this review possible.

---

## Release checklist

1. [ ] B1 — fix priorities (51/52) and comment.
2. [ ] B2 — clamp `min_batch` to `[1, kMaxBatchSize]`.
3. [ ] B3 — gtest for `KdaScanOp::evaluate` vs reference recurrence.
4. [ ] H1 — per-process temp dir, checked writes, cleanup.
5. [ ] H2 — decide fp16 default; record check-backend numbers and a match
       result in docs.
6. [ ] H3 — expose `policy_head` / `value_head` options.
7. [ ] H4 — CI job, README section, minimum OpenVINO version check.
8. [ ] H5 — commit or drop the uncommitted fp16_safe / QKV-fusion work; re-run
       `--backend=check` on a KDA net and a CNN net afterwards.
9. [ ] M2 — refresh the four stale comments.
10. [ ] Re-run the profile (`profile=true`) on the release build and confirm
        reference-kernel share is still low (note 2432 reported 52% in
        StridedSlice/Gather/Concat *before* the direction fold — verify the
        post-fold number is recorded).
