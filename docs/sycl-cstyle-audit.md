# SYCL Backend — C-style Performance Guide Audit

**Scope:** all 10 files under `src/neural/backends/sycl/` (11,705 lines), reviewed
against `cpp_guide_for_agents.md`'s checklist: hot/cold data separation,
allocation-in-loop, string identity in hot paths, container choice, runtime
branching in inner loops, ownership/RAII, debug-build cost.

**Method:** grepped the whole backend for the guide's "avoid by default" list
(`std::vector<std::vector<>>`, `std::unordered_map`/`std::map`, `std::string`/
`std::ostringstream`, `std::function`, `std::optional`, `std::shared_ptr`,
`virtual`, `new`/`malloc`), then read every hit in context plus the two hot
entry points end to end: `EncoderBlock::EvalKda` (layers.cc.dp.cpp:3077) and
`SyclNetworkComputation::ComputeBlocking` → `Network::forwardEval`
(network_sycl.cc.dp.cpp).

**Verdict:** the backend already follows the guide closely. One real
inner-loop branching violation found; everything else that looked like a hit
turned out to be cold-path code the guide explicitly carves out.

---

## Already conforms (no action needed)

- **No allocation on the hot path.** Every `sycl::malloc_device` /
  `sycl::malloc_shared` call (~35 sites) is inside a layer constructor or
  `MakeSyclNetwork` — one-time weight/workspace setup. `EvalKda`,
  `ComputeBlocking`, and every kernel in `common_kernels.dp.cpp` take
  pre-sized `scratch`/`buffer1`/`buffer2` pointers and do pure pointer
  arithmetic into them (e.g. layers.cc.dp.cpp:3096-3250). This is exactly the
  guide's "one retained workspace, reused between evaluations" pattern.
- **No STL containers in hot code.** `std::vector` appears only at
  construction (weight loading: layers.cc.dp.cpp:763,1329,1461,2006; device
  enumeration: network_sycl.cc.dp.cpp:303; the one-time `network_` layer list:
  network_sycl.cc.dp.cpp:1092). `kda_directions_` is copied from the
  constructor's `std::vector<int>` into a fixed `std::array<int, 16>` member
  once (layers.h:378) — `kdaRecurrenceValueParallel` never sees the vector.
  No `std::unordered_map`/`std::map` anywhere in the backend.
- **No strings in hot code.** Every `std::string`/`std::to_string` hit is
  either option parsing (`policy_head`, `value_head`), device-info logging, or
  an `Exception` message built at construction/validation time (e.g. the KDA
  weight-shape checks at layers.cc.dp.cpp:2519-2534, confirmed by
  `docs/kimi-findings.md` as construction-time hardening). None are on a path
  that runs per token or per batch.
- **`float state[32]`** in `kdaRecurrenceValueParallel`
  (common_kernels.dp.cpp:2224) is plain fixed-size stack storage, not
  `std::array` — matches the guide's "Small Fixed Storage" section directly.
- **Virtual dispatch is coarse, not per-element.** `BaseLayer::Eval` is
  virtual (layers.h:72), but it's called once per *layer* per batch
  (network_sycl.cc.dp.cpp:807-965, ~15-25 calls per forward pass) — the guide's
  objection is to dispatch inside per-element inner loops, which this isn't.

## Finding: runtime direction branch inside the per-token inner loop

**File:** `common_kernels.dp.cpp:2229-2246`, `kdaRecurrenceValueParallel`.

`direction` is resolved once per work-item from `head` before the token loop
starts (line 2216-2217) and is invariant for the rest of the kernel
invocation. But the loop body re-branches on it every one of the 64
iterations:

```cpp
for (int token = 0; token < 64; ++token) {
  int square = token;
  if (direction == 2) {
    square = 63 - token;
  } else if (direction == 3) {
    square = (token % 8) * 8 + token / 8;
  } else if (direction == 4) {
    ...
  } else if (direction == 5) {
    square = kKdaDiagForward[token];
  } ...  // up to direction == 8
```

This is the guide's "runtime kind branching inside inner loops" case
directly — the kind (`direction`) is known before the loop, so the branch
should be resolved once, not re-evaluated on every token. It's inconsistent
with itself, too: the four diagonal directions (5-8) already use a
precomputed `constexpr int[64]` table (`kKdaDiagForward` etc., defined at
common_kernels.dp.cpp:39-58) and pay just an array read; only the four
orthogonal directions (1-4) still compute `square` with a branch + arithmetic
every iteration.

**Applied.** Extended the existing table pattern to cover all 8 directions,
replacing the four separate diagonal-only tables with one
`constexpr int kKdaDirectionOrder[8][64]` (common_kernels.dp.cpp:48-86,
indexed `[direction - 1][token]`). The lookup row is now resolved once per
work-item before the token loop:

```cpp
const int* const square_order = kKdaDirectionOrder[direction - 1];
...
for (int token = 0; token < 64; ++token) {
  const int square = square_order[token];
  ...
```

`direction` is validated to be in `[1, 8]` at layer construction
(layers.cc.dp.cpp:2494), so the row index needs no runtime bounds check.

**Verification performed:**
- `icpx -fsycl -std=c++20 -fsyntax-only` on the edited file: 0 errors, 0 new
  warnings (only pre-existing `get_pointer`-deprecation warnings elsewhere in
  the file, unrelated to this change).
- Independent behavioral-equivalence check: re-implemented the original
  branch chain plus the four old tables in a standalone harness and compared
  against the new table for all 8 directions × 64 tokens (512 cases) — exact
  match, 0 mismatches.

**Not performed:** an on-hardware `backendbench` before/after run. This
machine has a real Intel iGPU (`Iris(R) Xe Graphics`, confirmed via
`sycl-ls`), but `build-sycl.cmd` builds this repo as a multi-backend binary
that also requires a CUDA v10.0 + cuDNN toolchain at a hardcoded path, which
this machine doesn't have — a full build here would likely fail for reasons
unrelated to this change. The change is correctness-verified but not yet
perf-measured; treat the "removes redundant per-iteration branching" framing
as the rationale, not a measured number, until it's benchmarked in an
environment that can build `backendbench` cleanly.

## Drive-by note (not a guide finding — flagging separately)

**File:** `layers.cc.dp.cpp:4185-4192`, `CublasError`.

```cpp
char message[128];
sprintf(message, "CUBLAS error: %s (%s:%d) ", CublasGetErrorString(status),
        file, line);
```

Cold error-only path, irrelevant to the guide's hot-path scope. Flagging
because `file` is `__FILE__`, which can exceed the 128-byte budget in a deep
build tree and overflow the stack buffer. Trivial fix (`snprintf` with
`sizeof(message)`) whenever this file is touched next; not urgent enough to
justify its own change.

---

## Summary

| Area | Status |
|---|---|
| Allocation in hot path | Clean |
| Containers in hot path | Clean |
| Strings in hot path | Clean |
| Small fixed storage | Clean |
| Dispatch granularity | Clean |
| Runtime branching in inner loop | **1 finding — fixed** (direction lookup table, common_kernels.dp.cpp) |
| Unrelated safety nit | `sprintf` in `CublasError`, cold path only — not fixed, flagged only |

The direction-lookup fix is applied on this branch
(`perf/sycl-cstyle-audit`) and correctness-verified (compile + exact
behavioral equivalence, see above), but **not yet perf-benchmarked on
hardware** — this machine can't build `backendbench` cleanly (see note
above). Run `build-sycl.cmd` → `backendbench` before/after this commit on a
machine that can build the full multi-backend binary before relying on this
as a measured win. The `sprintf` note is a drive-by observation, not acted
on.
