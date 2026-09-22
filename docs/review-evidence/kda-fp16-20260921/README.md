# KDA fp16 investigation + OpenVINO default flip — 2026-09-21

Repo: lc0 master, branch sycl-openvino-fix, base commit 101d70e0.
Net: kda-native-935532.pb.gz (serpentine directions 9-16, qkv_silu on).
Hardware: Intel Iris Xe (OpenVINO GPU plugin 2026.3), SYCL Level Zero.

## Question

kda_safe=true (precision-sensitive marks on scan inputs + value head) was the
default for KDA nets on OpenVINO GPU. Measure whether it buys anything.

## Method

- Speed: `lc0.exe backendbench --backend=openvino --backend-opts=<variant>
  --weights=<net> --batches=40-50` at batch sizes 1..64, repeated.
- Accuracy: raw single-eval diffs vs blas via two probes (no search
  amplification): `openvino-review-probe.exe` (OpenVINO build) and
  `sycl-fp16-probe.cc` (this directory; built in build-sycl-review-20260917).

## Measurements (batch 8, 40-50 batches)

| config | mean nps | policy_abs (probe, raw) | note |
|---|---|---|---|
| fp16=false (pure f32) | 992-997 | 0.003 | best accuracy |
| kda_safe=true | 891-981 | 0.29 | flood + f16 tail |
| kda_safe=false (full f16) | 1289-1369 | 11.8 | broken |
| SYCL-fp16 (probe) | — | rel. err 12-16% | net-inherent, not OV-specific |

Batch sweep means: batch 1: f32 181 / safe 166 / f16 208; batch 32: f32 1215 /
safe 1199 / f16 2070. f32 >= safe at every size.

## Findings

1. kda_safe=true is strictly dominated by plain f32 on this hardware: the
   marks flood f32 backward through the whole producer chain (OpenVINO
   semantics: marking an input disables f16 conversion of the subgraph before
   it), so the trunk never actually runs f16; the unmarked f16 tail (policy
   head etc.) then only adds error. Slower AND less accurate than f32.
2. Full f16 (and SYCL-fp16) break accuracy on this trained net — 8-16%
   relative policy error with f32 recurrence internals, so the failure is in
   the fp16 trunk itself, not in backend-specific scan handling.
3. Probe's q_abs/d_abs read 0 on every config — those channels are not
   discriminating here (value head saturates); policy_abs is the signal.

## Change

`network_openvino.cc`: fp16 default is now `is_gpu_device && !has_kda` — KDA
nets run pure f32 unless the user explicitly forces fp16. kda_safe marking
machinery retained for explicit `fp16=true`. Also `kda_parity_test.cc`: three
SYCL tests (SyclEnforcesMaxBatchCapacity, SyclAsyncErrorFailsComputation,
SyclConstructorFailureUnwindsCleanly) now GTEST_SKIP instead of ASSERT-fail
when the binary was built without SYCL (they failed in OpenVINO-only builds).

## Verification after change

- Default load compiles FP32; probe review_failures 0 (was 11).
- openvino_parity_test 9/9 PASS; kda_parity_test passes in the SYCL build
  (3/3) and skips cleanly in the OpenVINO-only build.
- Explicit overrides still work: fp16=true -> kda_safe path (951 nps, marks
  applied); fp16=true,kda_safe=false -> full f16 (1289 nps).
