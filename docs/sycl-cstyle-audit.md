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

**Perf-measured on hardware — result: no measurable win.** Built a
SYCL-only `lc0.exe` locally (working around this machine's broken oneAPI env
propagation and adding the `mkl_sycl_blas_dll`/OpenCL deps the `sycl=l0`
meson path requires but the general `-Dmkl`/`-Dopencl` flags don't cover —
see `build-sycl-local.cmd`) and ran `backendbench -b sycl` against a real
KDA net (`kda-native-410835.pb.gz`) on the machine's Intel Iris Xe iGPU, two
full sweeps (batch sizes 1-32, 200 batches each) per side:

| | mean nps, pooled over 2 runs |
|---|---|
| Before (branch chain) | baseline |
| After (table lookup) | **-1.0% average**, ranging -14.2% to +6.4% across batch sizes |

The catch: run-to-run noise on this box is large enough to swallow the
signal. Two runs of the *identical* before-fix binary differed by a mean
12.35% (up to ~20% at some batch sizes) — bigger than the average
before/after gap, and bigger than every individual before/after delta except
the two worst outliers (batch 24: -14.2%, batch 29: -9.9%), which themselves
don't exceed what same-binary noise already produced elsewhere in the sweep.
14/32 batch sizes favored "after", 18/32 favored "before" — no consistent
direction. Likely cause: a single laptop iGPU with no thermal/background-load
isolation between ~5-minute sweeps, not a real regression or a real win.

**Verdict: keep the change for its stated reason (removes a per-iteration
branch that's redundant given `direction` is already loop-invariant, unifies
the previously-inconsistent orthogonal/diagonal code paths), but do not cite
it as a measured speedup.** It is correctness-neutral (bit-exact equivalence
confirmed above) and stylistically an improvement per the guide, not a
proven perf win on this hardware. Re-measuring on a machine with a
discrete GPU and better run-to-run isolation (dedicated benchmarking box,
more repeats, interleaved A/B ordering to cancel thermal drift) would be
needed before this could honestly be called a speedup.

**Direction-table correctness — cross-checked against the actual spec, not
just self-consistency.** Beyond compile + old-SYCL-vs-new-SYCL equivalence
(above), independently re-derived all 8 directions from two upstream
references and diffed against the committed `kKdaDirectionOrder` table,
512/512 exact matches against each:
- `KdaSquareForToken` in `src/neural/backends/blas/network_blas.cc:281`
  (the BLAS backend's reference implementation, explicitly commented "Must
  match KDA_TRAVERSALS in the trainer").
- `KDA_TRAVERSALS` in `tf/tfprocess.py:107` in the training repo (the actual
  trainer ground truth) — regenerated all 8 direction orders from its
  Python generator expressions and compared token-by-token.

So the direction semantics were already correct before this refactor, and
the refactor preserved them exactly; this was purely a shape/branching
change, not a behavior change.

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
| Runtime branching in inner loop | **1 finding — fixed, correctness-verified, perf-measured neutral** (direction lookup table, common_kernels.dp.cpp) |
| Unrelated safety nit | `sprintf` in `CublasError`, cold path only — not fixed, flagged only |

The direction-lookup fix is applied on this branch
(`perf/sycl-cstyle-audit`), correctness-verified three ways (compile,
old-vs-new equivalence, and cross-check against the BLAS reference and the
trainer's actual `KDA_TRAVERSALS` spec — 512/512 matches each), and
perf-measured on real Intel iGPU hardware. The perf result was a wash
(-1.0% average, within this machine's run-to-run noise floor) — see above
for the full before/after data. Keep the change for code quality, not as a
claimed speedup. The `sprintf` note is a drive-by observation, not acted on.
