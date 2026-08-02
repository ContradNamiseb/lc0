# KDA Hybrid Transformer Design for LCZero

Status: design proposal

Last researched: 2026-07-29

Target training branch:
[`daniel-monroe/lczero-training:stable`](https://github.com/daniel-monroe/lczero-training/tree/stable)

Target engine: this lc0 branch, with an initial reference implementation and a
native SYCL implementation.

## 1. Purpose

This document describes how to extend the LCZero training and inference stacks
to train and run transformer networks containing Kimi Delta Attention (KDA).
The change crosses four boundaries:

1. The TensorFlow model in `lczero-training`.
2. The shared protobuf network format in `lczero-common` and lc0.
3. The lc0 weight loader and backend-independent weight representation.
4. At least one lc0 inference backend, followed by the SYCL backend.

Adding a TensorFlow KDA function alone is not sufficient. A KDA checkpoint can
be trained without an engine implementation, but it cannot be exported as a
usable native `.pb.gz` network until the protobuf schema and an lc0 backend
understand the same recurrence, dimensions, traversal order, and output gate.

## 2. Executive Decision

The recommended first network is a hybrid attention body with a repeated
three-to-one KDA-to-MHA pattern:

```text
embedding
  -> KDA -> FFN
  -> KDA -> FFN
  -> KDA -> FFN
  -> MHA -> FFN
  -> repeat
  -> existing policy, value, and moves-left heads
```

The KDA blocks should use four fixed board traversal directions, distributed
across their heads:

- rank-major forward
- rank-major reverse
- file-major forward
- file-major reverse

For 16 heads, four heads are assigned to each traversal. Their outputs are
restored to the normal square order and concatenated before the output gate and
output projection.

The following parts of the current model should not change in the first
experiment:

- input and square embedding
- DeepNorm/residual behavior
- encoder RMSNorm or LayerNorm behavior
- encoder FFN
- policy embedding
- final policy query/key operation
- promotion logits
- value heads
- moves-left head

In particular, the final policy query/key matrix must remain full 64 by 64
attention. Those logits directly represent from-square/to-square moves and are
not merely an internal token mixer.

## 3. Why Chess Needs A Modified KDA Design

KDA was designed as a causal sequence mixer. LCZero presents a different
problem:

- Every input contains exactly 64 square tokens.
- The tokens describe one complete board, not an unbounded sequence.
- All squares should be able to influence all other squares.
- A flattened square order is a representation choice, not a causal fact.
- Each inference batch contains independent positions.

A single causal scan from square 0 to square 63 would make later squares able
to use earlier squares, while earlier squares could not use later squares. It
would also make one arbitrary square ordering part of the architecture.

The four traversal groups reduce this problem. At every square, the concatenated
head output contains information from opposite scan directions and from both
rank-major and file-major views. Periodic full MHA layers then provide an
order-independent global communication step.

This is still not exactly rotation or reflection equivariant. Existing input
canonicalization and training augmentation may reduce that sensitivity, but
symmetry tests are required. If the four-direction model remains sensitive to
orientation, the next experiments should be:

1. Eight D4-related traversal orders.
2. A learned mixture across traversal groups.
3. A true two-dimensional selective scan.
4. More frequent full MHA layers.

## 4. KDA Recurrence

For batch size `B`, sequence length `T = 64`, head count `H`, key dimension
`K`, and value dimension `V`, use:

```text
q, k:     [B, T, H, K]
v:        [B, T, H, V]
g:        [B, T, H, K]  log-space decay
beta:     [B, T, H]
state S:  [B, H, K, V]
output:   [B, T, H, V]
```

For token `t`:

$$
g_t = -\exp(A_{\log})\operatorname{softplus}(f(x_t) + d_t)
$$

$$
\widetilde S_t = \exp(g_t)\odot S_{t-1}
$$

$$
e_t = v_t - \widetilde S_t^\mathsf{T} k_t
$$

$$
S_t = \widetilde S_t + \beta_t k_t e_t^\mathsf{T}
$$

$$
o_t = S_t^\mathsf{T}\left(q_t / \sqrt{K}\right)
$$

Use L2-normalized queries and keys, and calculate:

$$
\beta_t = \operatorname{sigmoid}(W_\beta x_t)
$$

Recommended gate parameter shapes are:

```text
A_log:    [H, 1]
dt_bias:  [H, K]
f_a(x):   [B, T, gate_rank]
f_b(...): [B, T, H * K]
```

The state must start at zero for every direction, layer, and evaluated chess
position. It must not persist between:

- positions in the same inference batch
- MCTS nodes
- chess moves
- calls to `NetworkComputation::ComputeBlocking()`

LCZero caches complete network evaluations. A recurrent state carried between
positions would make the result depend on evaluation order and would be
incorrect.

## 5. KDA Output Gate

The official Kimi layer applies a gated normalization to the recurrent output
before its final projection. The first LCZero implementation should use the
following explicit operation:

```text
gate = sigmoid(gate_b(gate_a(x)))
mixed = rms_norm(kda_output) * gate
output = output_projection(mixed)
```

The output gate is part of the serialized architecture. Training and inference
must agree on:

- gate activation
- RMSNorm epsilon
- whether RMSNorm has a learned scale
- placement of the gate before or after normalization
- output projection layout

Do not silently substitute the encoder's outer normalization for this internal
KDA output normalization.

## 6. Initial Model Configuration

Add the following fields under `model` in a new
`tf/configs/kda-hybrid.yaml`, based on the stable branch's `example.yaml`:

```yaml
model:
  embedding_size: 512
  policy_embedding_size: 512
  value_embedding_size: 128
  moves_left_embedding_size: 32

  encoder_layers: 12
  encoder_heads: 16
  encoder_d_model: 512
  encoder_dff: 512

  encoder_mixer_pattern: [kda, kda, kda, mha]

  kda_key_dim: 32
  kda_value_dim: 32
  kda_gate_rank: 32
  kda_directions:
    - rank_forward
    - rank_reverse
    - file_forward
    - file_reverse
  kda_output_gate: true
  kda_rms_norm_epsilon: 0.00001

  # These apply to the remaining MHA blocks.
  use_smolgen: true
  smolgen_hidden_channels: 32
  smolgen_hidden_sz: 256
  smolgen_gen_sz: 256
  smolgen_activation: swish

  encoder_rms_norm: true
  omit_qkv_biases: true
```

During training-only development, also set:

```yaml
training:
  disable_pb_checkpointing: true
```

This prevents the unmodified exporter from emitting a network that appears to
be a normal attention-body net but contains weights the engine cannot load.

The configuration parser should validate:

- `encoder_mixer_pattern` contains only `mha` and `kda`.
- `encoder_heads` is divisible by the number of KDA directions.
- `kda_key_dim`, `kda_value_dim`, and `kda_gate_rank` are positive.
- KDA projection output sizes agree with their declared dimensions.
- Smolgen is used only by MHA blocks in the first implementation.
- The complete layer pattern is deterministic after repetition.

The effective mixer for layer `i` is:

```python
mixer = pattern[i % len(pattern)]
```

Store the resolved mixer type per encoder layer in the protobuf. Do not require
the engine to reconstruct it from a global ratio.

## 7. Training Repository Setup

The stable branch requires TensorFlow 2.13 or newer and was tested with
TensorFlow 2.14. Its requirements currently select TensorFlow 2.14 with CUDA.
Linux is the expected training environment.

```bash
git clone --branch stable --recurse-submodules \
  https://github.com/daniel-monroe/lczero-training.git
cd lczero-training
git switch -c kda-hybrid

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tf/requirements.txt
./init.sh
```

The stable branch pins `libs/lczero-common` to commit
`82f6b0a5fdc8c79f527d3f9accf6b529cc8abac1`. Create a fork of
`LeelaChessZero/lczero-common`, add the KDA schema there, and update the training
repository's submodule to the KDA schema commit.

Keep these three schemas synchronized:

1. `lczero-training/libs/lczero-common/proto/net.proto`
2. the schema fork's `proto/net.proto`
3. lc0's `proto/net.proto`

## 8. TensorFlow Implementation

### 8.1 Configuration

In `tf/tfprocess.py`, read the KDA settings in `TFProcess.__init__` near the
existing encoder settings:

```python
self.encoder_mixer_pattern = self.cfg["model"].get(
    "encoder_mixer_pattern", ["mha"])
self.kda_key_dim = self.cfg["model"].get("kda_key_dim", 32)
self.kda_value_dim = self.cfg["model"].get("kda_value_dim", 32)
self.kda_gate_rank = self.cfg["model"].get("kda_gate_rank", 32)
self.kda_directions = self.cfg["model"].get(
    "kda_directions",
    ["rank_forward", "rank_reverse", "file_forward", "file_reverse"],
)
self.kda_output_gate = self.cfg["model"].get("kda_output_gate", True)
self.kda_rms_norm_epsilon = self.cfg["model"].get(
    "kda_rms_norm_epsilon", 1e-5)
```

Use `mha` as the default pattern. Existing YAML configurations and checkpoints
must continue to build the current model without modification.

### 8.2 Layer Names

Use stable, serialization-friendly names under each encoder:

```text
encoder_N/kda/wq
encoder_N/kda/wk
encoder_N/kda/wv
encoder_N/kda/decay_a
encoder_N/kda/decay_b
encoder_N/kda/beta
encoder_N/kda/a_log
encoder_N/kda/dt_bias
encoder_N/kda/gate_a
encoder_N/kda/gate_b
encoder_N/kda/out_norm
encoder_N/kda/dense
```

Avoid anonymous `Lambda` layers for trainable operations. Explicit names make
checkpoint inspection and `tf_name_to_pb_name()` deterministic.

### 8.3 Reference Recurrence

Start with an ordinary TensorFlow recurrence. It is slower than a fused kernel,
but it is easy to test and provides the numerical oracle for all optimized
implementations.

```python
def recurrent_kda(q, k, v, log_decay, beta):
    """KDA recurrence with tensors shaped [B, T, H, D]."""
    q = tf.math.l2_normalize(q, axis=-1)
    k = tf.math.l2_normalize(k, axis=-1)

    q = tf.transpose(q, [1, 0, 2, 3])
    k = tf.transpose(k, [1, 0, 2, 3])
    v = tf.transpose(v, [1, 0, 2, 3])
    log_decay = tf.transpose(log_decay, [1, 0, 2, 3])
    beta = tf.transpose(beta, [1, 0, 2])

    batch = tf.shape(q)[1]
    heads = tf.shape(q)[2]
    key_dim = tf.shape(q)[3]
    value_dim = tf.shape(v)[3]

    initial_state = tf.zeros(
        [batch, heads, key_dim, value_dim], dtype=tf.float32)
    initial_output = tf.zeros(
        [batch, heads, value_dim], dtype=tf.float32)
    scale = tf.math.rsqrt(tf.cast(key_dim, tf.float32))

    elems = tuple(tf.cast(x, tf.float32)
                  for x in (q, k, v, log_decay, beta))

    def step(carry, values):
        state, _ = carry
        q_t, k_t, v_t, decay_t, beta_t = values

        decayed = state * tf.exp(decay_t)[..., None]
        prediction = tf.einsum("bhk,bhkv->bhv", k_t, decayed)
        delta = beta_t[..., None] * (v_t - prediction)
        next_state = decayed + tf.einsum(
            "bhk,bhv->bhkv", k_t, delta)
        output = tf.einsum(
            "bhk,bhkv->bhv", q_t * scale, next_state)
        return next_state, output

    _, outputs = tf.scan(
        step,
        elems,
        initializer=(initial_state, initial_output),
        parallel_iterations=1,
    )
    outputs = tf.transpose(outputs, [1, 0, 2, 3])
    return tf.cast(outputs, v.dtype)
```

Keep state updates, exponentials, softplus, inner products, and reductions in
FP32 during the first implementation. Inputs and final outputs may use the
configured mixed-precision dtype.

### 8.4 Traversal Orders

Define each traversal as a constant permutation of the existing 64-token
order. Derive the arrays once and test them rather than duplicating index
arithmetic in every call.

For each direction:

```python
ordered = tf.gather(projected, order, axis=1)
ordered_output = recurrent_kda(...)
inverse_order = tf.argsort(order)
normal_output = tf.gather(ordered_output, inverse_order, axis=1)
```

Split the head dimension evenly across directions before running the scans,
then concatenate the normal-order outputs on the head dimension. The direction
permutations must be serialized as a known enum or fixed format version. Do not
allow training and inference to define the order independently.

### 8.5 KDA Projection And Gate

The KDA mixer should perform:

1. Dense Q, K, and V projections.
2. Low-rank decay projection.
3. Per-head beta projection.
4. Reshape into heads.
5. Generate log-space decay.
6. Run one recurrence per directional head group.
7. Restore normal square order and concatenate heads.
8. Apply gated RMSNorm.
9. Apply the output dense projection.

Pseudocode:

```python
raw_decay = decay_b(decay_a(inputs))
raw_decay = tf.reshape(raw_decay, [batch, 64, heads, key_dim])
log_decay = (
    -tf.exp(a_log)[None, None, :, :]
    * tf.nn.softplus(raw_decay + dt_bias[None, None, :, :])
)
beta = tf.sigmoid(beta_projection(inputs))

mixed = run_direction_groups(q, k, v, log_decay, beta)
gate = tf.sigmoid(gate_b(gate_a(inputs)))
mixed = rms_norm(mixed) * gate
output = output_projection(mixed)
```

Initialize `A_log` in the same useful range as the reference Kimi layer, for
example the logarithm of a uniform value in `[1, 16]`. Initialize `dt_bias` so
the initial decay is neither almost zero nor almost one. Log decay histograms
during training to detect saturated gates.

### 8.6 Encoder Dispatch

Modify `TFProcess.encoder_layer()` so only the mixer changes:

```python
if mixer_type == "mha":
    mixer_output, mixer_weights = self.mha(...)
elif mixer_type == "kda":
    mixer_output = self.kda(...)
    mixer_weights = None
else:
    raise ValueError("Unknown encoder mixer: {}".format(mixer_type))
```

The existing dropout, first residual/norm, FFN, second residual/norm, and
DeepNorm scaling should remain untouched. `construct_net()` resolves the mixer
for each layer and passes it to `encoder_layer()`.

KDA layers may append compact decay/beta diagnostics when
`return_attn_wts` is enabled, but they should not materialize a fake 64 by 64
attention matrix.

### 8.7 Short Convolution

The official Kimi layer applies a causal short convolution to Q, K, and V.
Omit it in phase one because a 1D convolution across flattened square indices
creates an artificial edge between the end of one rank and the beginning of
the next.

A later LCZero-specific experiment may add one of:

- separate rank and file convolutions
- a shared 3 by 3 board convolution before token flattening
- a depthwise 2D convolution over square embeddings

Treat that as a separate controlled experiment, not as part of validating the
core KDA recurrence.

## 9. TensorFlow Tests

Add focused tests before running a full training job:

1. Compare `tf.scan` with a small NumPy recurrence in FP32.
2. Verify every traversal is a permutation of `range(64)`.
3. Verify applying a traversal and its inverse returns the original tensor.
4. Check output shape for multiple batch sizes and head counts.
5. Check every trainable KDA variable receives a finite gradient.
6. Check zero input and zero state produce finite output.
7. Check large positive/negative raw decay values do not produce NaNs.
8. Run under both FP32 and the configured mixed-precision policy.
9. Verify old MHA-only YAML produces the previous variable names and shapes.
10. Measure sensitivity under legal board rotations/reflections.

Use a tiny smoke-test model before the production-sized model:

```yaml
model:
  embedding_size: 128
  encoder_layers: 2
  encoder_heads: 4
  encoder_d_model: 128
  encoder_dff: 128
  encoder_mixer_pattern: [kda, mha]
  kda_key_dim: 32
  kda_value_dim: 32
  kda_gate_rank: 16
```

Run at least several thousand optimizer steps and monitor:

- total and per-head gradient norms
- `A_log` and `dt_bias`
- mean and extrema of `exp(log_decay)`
- beta distribution
- recurrent state norm by layer
- policy/value loss against the MHA baseline
- positions per second

## 10. Protobuf Format

### 10.1 Compatibility Rule

KDA must have a distinct network-format marker and a new minimum lc0 version.
Do not label a KDA net as `NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT`. An old
engine could otherwise select the MHA path and interpret absent MHA tensors as
zero-sized weights.

The stable training branch's pinned schema uses legacy enum values that differ
from the current lc0 schema. Current lc0 already translates the old multihead
value. Apply KDA changes to a synchronized schema fork instead of editing only
the generated Python protobuf.

An illustrative allocation is:

```protobuf
NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT = 135;
```

The exact number must be checked against the schema used for upstreaming. Once
published, a protobuf enum value must never be reused for another format.

### 10.2 Proposed Messages

Add a KDA message alongside `Weights.MHA`:

```protobuf
message KDA {
  optional Layer q_w = 1;
  optional Layer q_b = 2;
  optional Layer k_w = 3;
  optional Layer k_b = 4;
  optional Layer v_w = 5;
  optional Layer v_b = 6;

  optional Layer decay_a_w = 7;
  optional Layer decay_a_b = 8;
  optional Layer decay_b_w = 9;
  optional Layer decay_b_b = 10;
  optional Layer beta_w = 11;
  optional Layer beta_b = 12;
  optional Layer a_log = 13;
  optional Layer dt_bias = 14;

  optional Layer gate_a_w = 15;
  optional Layer gate_a_b = 16;
  optional Layer gate_b_w = 17;
  optional Layer gate_b_b = 18;
  optional Layer out_norm_gammas = 19;
  optional Layer dense_w = 20;
  optional Layer dense_b = 21;

  optional uint32 key_dim = 22;
  optional uint32 value_dim = 23;
  optional uint32 gate_rank = 24;
  optional float rms_norm_epsilon = 25;
}
```

Extend `Weights.EncoderLayer` using unused field numbers:

```protobuf
message EncoderLayer {
  enum MixerType {
    MIXER_MHA = 0;
    MIXER_KDA = 1;
  }

  optional MHA mha = 1;
  optional Layer ln1_gammas = 2;
  optional Layer ln1_betas = 3;
  optional FFN ffn = 4;
  optional Layer ln2_gammas = 5;
  optional Layer ln2_betas = 6;

  optional MixerType mixer = 7 [default = MIXER_MHA];
  optional KDA kda = 8;
}
```

Add a network-format field that defines the traversal scheme. Prefer a named
enum over arbitrary serialized index arrays for the first version:

```protobuf
enum KdaTraversal {
  KDA_TRAVERSAL_UNKNOWN = 0;
  KDA_TRAVERSAL_RANK_FILE_BIDIRECTIONAL = 1;
}

optional KdaTraversal kda_traversal = 11;
```

Explicit dimensions are required even when a training configuration omits
biases. Inferring dimensions from bias-vector length would produce zero for
bias-free projections.

### 10.3 Versioning

Add a training exporter constant such as:

```python
LC0_MINOR_WITH_KDA = <coordinated lc0 minor version>
```

When any encoder uses KDA, the exporter should:

- select the KDA hybrid network format
- set `min_version` to the first supporting lc0 version
- set the traversal enum
- serialize mixer type for every encoder layer

An older lc0 must reject the file based on `min_version` or unknown network
format. It must not fall back to MHA.

## 11. Export And Import

In stable `tf/net.py`, add a mapping helper for KDA names:

```python
def kda_to_pb(layer, weight):
    # Map encoder_N/kda/<layer>/<weight> to kda.<protobuf field>.
    ...
```

Then extend the encoder mapping:

```python
elif base_layer.startswith("encoder"):
    encoder_block = int(base_layer.split("_")[1]) - 1
    if layers[1] == "mha":
        pb_name = "mha." + mha_to_pb(...)
    elif layers[1] == "kda":
        pb_name = "kda." + kda_to_pb(...)
    elif layers[1] == "ffn":
        pb_name = "ffn." + ffn_to_pb(...)
    else:
        pb_name = encoder_to_pb(...)
```

`fill_net_v2()` already transposes rank-two TensorFlow dense weights from
`[input, output]` to lc0's `[output, input]`. KDA dense matrices should use the
same rule. Vectors such as `A_log`, `dt_bias`, biases, and RMSNorm scale are not
transposed.

Set each encoder's `mixer` field explicitly while exporting. Weight-name
mapping alone cannot set a protobuf enum.

Regenerate Python bindings after every schema change:

```bash
./init.sh
```

Before enabling normal protobuf checkpoints, test:

1. TensorFlow checkpoint to `.pb.gz`.
2. `.pb.gz` back into a newly constructed TensorFlow model.
3. Exact tensor shape and name agreement.
4. Tensor-by-tensor round-trip error caused by LINEAR16 encoding.
5. MHA-only export remains unchanged.

## 12. LCZero Weight Loading

Update the following engine areas:

- `proto/net.proto`
- `src/neural/loader.cc`
- `src/neural/network_legacy.h`
- `src/neural/network_legacy.cc`

Add a backend-independent `BaseWeights::KDA` structure containing all KDA
weights and dimensions. Extend `BaseWeights::EncoderLayer` with:

```cpp
enum class MixerType { kMha, kKda };

MixerType mixer_type;
MHA mha;
KDA kda;
```

The loader should validate before constructing a backend:

- network format is supported
- KDA fields exist for every KDA layer
- MHA fields exist for every MHA layer
- key/value dimensions are nonzero
- head count is divisible by traversal count
- projection sizes agree with embedding and head dimensions
- output projection returns the embedding size
- traversal enum is recognized

Unsupported backends should report a direct message such as:

```text
This backend does not support KDA hybrid attention-body networks.
```

Do not allow an unsupported backend to proceed with empty MHA weights.

## 13. Reference Engine Implementation

Implement the recurrence in a simple CPU or BLAS path before optimizing SYCL.
The reference path should:

1. Compute Q, K, V, raw decay, beta, and output gate with existing dense
   operations.
2. Convert or retain recurrent state arithmetic in FP32.
3. Iterate the 64 tokens in the selected order.
4. Start each state at zero.
5. Restore normal square order.
6. Apply output RMSNorm, gate, and projection.
7. Continue through the existing residual/norm and FFN path.

This implementation is the engine-side oracle for SYCL and other accelerated
backends. It does not need to be fast enough for production search.

An ONNX export can provide another temporary parity check, but a converted
TensorFlow `scan` may become an inefficient ONNX `Loop`, and it does not replace
the native SYCL implementation.

## 14. SYCL Implementation

### 14.1 Encoder Dispatch

The current SYCL `EncoderBlock` owns MHA projections, full 64 by 64 attention,
residual/norm, FFN, and the second residual/norm. Refactor it so the common
encoder shell dispatches to one of two internal mixers:

```text
EncoderBlock
  -> MhaMixer::Eval() or KdaMixer::Eval()
  -> existing LN1/residual
  -> existing FFN
  -> existing LN2/residual
```

Do not duplicate the FFN and residual code in two complete encoder classes.

The final policy-head QK computation remains unchanged.

### 14.2 Dense Projections

Use the existing BLAS wrappers for the KDA dense projections. A later
optimization can pack projections with matching input/output dimensions into a
single strided batched GEMM, as the MHA implementation currently packs Q/K/V.

Initial intermediate buffers are:

```text
q:         [N, 64, H, K]
k:         [N, 64, H, K]
v:         [N, 64, H, V]
raw_decay: [N, 64, H, K]
beta:      [N, 64, H]
gate:      [N, 64, H, V]
mixed:     [N, 64, H, V]
state:     [N, H, K, V] in FP32
```

### 14.3 Recurrence Kernel

For the initial optimized kernel, assign one workgroup to each
`(batch, head)` pair. The workgroup:

1. Selects the traversal associated with the head.
2. Initializes its `K * V` state to zero.
3. Iterates 64 squares inside one kernel invocation.
4. Applies decay to the state.
5. Reduces over `K` for the prediction.
6. Applies beta and the delta update.
7. Reduces over `K` for output.
8. Writes output at the original square index.

For `K = V = 32`, the state contains 1024 FP32 values, or 4 KiB per
workgroup. Do not assume a 1024-thread workgroup. A 128- or 256-thread group can
process multiple state cells per work-item and use subgroup/workgroup reductions
for the two K reductions.

Keep the state in local memory when the target device permits it. Provide a
global-memory fallback for dimensions or devices that exceed local-memory
limits. Use the runtime-supported subgroup size conventions already established
by the SYCL backend.

### 14.4 Scratch Sizing

Update `getMaxAttentionBodySize()` to consider both mixer types. MHA needs the
64 by 64 logit matrix; KDA needs its projections, gates, and state. Scratch size
must be the maximum requirement across all encoder layers, not a value inferred
only from the first layer.

Because FP16 inference still uses FP32 recurrence state initially, calculate
state bytes separately from `sizeof(DataType)`.

### 14.5 Lifetime And Batching

Allocate reusable device scratch for the configured maximum batch, but zero or
overwrite every KDA state at the start of each layer evaluation. Scratch memory
may be reused across calls; its contents may not be treated as a valid initial
state.

Each batch item has an independent state. There is no cross-batch reduction or
state sharing.

### 14.6 Numerical Behavior

The first SYCL version should use:

- FP16 or FP32 projection inputs according to backend configuration
- FP32 Q/K normalization reductions
- FP32 log-decay activation
- FP32 recurrent state
- FP32 prediction and output reductions
- cast to `DataType` after recurrent output or output projection

Only reduce state precision after profiling proves that FP32 state is the
bottleneck and parity/Elo tests establish an acceptable alternative.

## 15. Cross-Implementation Parity

Create fixed random and real-position fixtures and capture intermediate tensors
from TensorFlow:

- Q, K, and V
- log decay
- beta
- output gate
- directional recurrent output
- concatenated mixer output
- projected mixer output
- post-LN1 output
- post-FFN output
- final policy and value outputs

Compare in this order:

1. NumPy recurrence against TensorFlow FP32.
2. Reference lc0 backend against TensorFlow FP32.
3. SYCL FP32 against the reference backend.
4. SYCL FP16 against SYCL FP32.
5. Full `.pb.gz` evaluation against the TensorFlow checkpoint.

Initial numerical gates:

- FP32 recurrence maximum absolute error at or below `1e-5` on small fixtures.
- FP32 full-network relative/absolute tolerances established per output.
- FP16 tolerance measured from actual policy/value outputs, not chosen only at
  the internal tensor level.
- No NaN or infinity for any legal test position.
- Results independent of batch position and prior evaluations.

The LINEAR16 protobuf quantization error should be measured separately from
backend arithmetic error.

## 16. Training Plan

### Phase A: Training-Only Proof

1. Implement the TensorFlow reference KDA.
2. Keep protobuf checkpointing disabled.
3. Train the tiny two-layer smoke model.
4. Verify gradients, stability, and loss decrease.
5. Compare MHA, KDA-only, and hybrid models at the same small scale.

### Phase B: Serialization And Reference Inference

1. Extend the shared protobuf.
2. Add exporter and importer mappings.
3. Add the new network format and minimum version.
4. Implement reference lc0 inference.
5. Pass TensorFlow-to-lc0 parity tests.

### Phase C: Native SYCL

1. Add KDA weight upload and mixer dispatch.
2. Implement an FP32 recurrence kernel.
3. Pass reference-to-SYCL parity.
4. Add mixed-precision projections.
5. Profile scratch, occupancy, local memory, and batch scaling.

### Phase D: Controlled Network Experiments

Train at least these equal-budget variants:

```text
A: all MHA baseline
B: all four-direction KDA
C: KDA, KDA, KDA, MHA
D: KDA, MHA alternating
```

Hold constant where practical:

- training data and data order
- optimizer and learning-rate schedule
- batch size/effective batch size
- training positions
- embedding width
- parameter count or measured inference budget
- policy/value/moves-left heads
- evaluation positions

Run the normal stable-branch entry point from `tf`:

```bash
./train.py \
  --cfg configs/kda-hybrid.yaml \
  --output networks/kda-hybrid.txt
```

The training pipeline resumes checkpoints found under the configured
`training.path` and network `name`. Use a new name for every architecture so a
checkpoint with incompatible variables is not restored accidentally.

## 17. Evaluation And Acceptance Criteria

KDA's primary published benefit is efficient long-context processing. LCZero
has only 64 tokens, so the asymptotic argument is not enough. For one head,
full attention spends work on 64 by 64 score/value matrices, while KDA repeatedly
updates a `K * V` state. Depending on dimensions and kernel quality, KDA may be
faster, equal, or slower.

Measure:

- training positions per second
- training memory
- batch-1 inference latency
- latency at realistic search batch sizes
- maximum sustainable batch
- engine nodes per second
- policy loss and policy accuracy
- value loss and calibration
- moves-left error
- parameter count
- actual backend FLOPs/operations where available
- Elo or SPRT against the equal-budget MHA baseline

Suggested release gates are:

1. All serialization and parity tests pass.
2. Existing MHA networks retain their prior results and throughput.
3. KDA training remains finite through a representative run.
4. No evaluation-order or batch-position dependence exists.
5. The hybrid model provides a measured quality/throughput tradeoff worth its
   additional format and backend complexity.
6. Elo claims are based on controlled games, not validation loss alone.

A reasonable first SPRT can test `H0 = 0 Elo` against `H1 = +5 Elo`, but the
exact bounds and game count should follow the project's normal testing policy.

## 18. Risks And Mitigations

### Square-order bias

Risk: the recurrence learns artifacts from one flattening order.

Mitigation: four direction groups, periodic MHA, symmetry tests, and possible
eight-direction follow-up.

### No speedup at 64 tokens

Risk: recurrent synchronization and state updates cost more than optimized MHA.

Mitigation: keep the architecture hybrid, benchmark before a large run, and
pack/fuse projections only after correctness.

### Numerical instability

Risk: repeated decay and rank-one updates under mixed precision produce NaNs or
vanishing state.

Mitigation: FP32 state/reductions, normalized Q/K, bounded initialization,
gradient clipping, and gate/state histograms.

### Format fragmentation

Risk: training, schema, and engine forks serialize incompatible fields or enum
values.

Mitigation: one schema source, pinned submodule commit, a distinct network
format, a minimum version, and round-trip tests.

### Unsupported backends

Risk: users select CUDA, Metal, or another backend that assumes every encoder
contains MHA.

Mitigation: explicit capability rejection until each backend implements KDA.
Do not silently substitute MHA.

### Policy regression

Risk: replacing the policy's final pairwise logits would remove direct move
structure.

Mitigation: replace only attention-body mixers. Preserve the final full policy
QK logits and promotion path.

## 19. Compatibility Strategy

Existing MHA networks remain compatible because:

- missing `EncoderLayer.mixer` defaults to MHA
- old network-format values are unchanged
- existing MHA protobuf fields retain their numbers
- old YAML defaults to `[mha]`
- backend dispatch selects the current MHA implementation

New KDA networks require:

- the KDA hybrid network-format enum
- the coordinated minimum lc0 version
- recognized traversal metadata
- a backend advertising KDA support

Once native export is enabled, include the architecture and traversal in net
inspection output so users can identify the format without attempting a search.

## 20. Definition Of Done

The feature is complete only when all of the following are true:

- A stable-branch TensorFlow model can train KDA and hybrid encoder stacks.
- MHA-only training remains unchanged by default.
- KDA gradients and training remain numerically stable.
- Checkpoint-to-protobuf-to-checkpoint round trips pass.
- Older lc0 versions reject KDA nets cleanly.
- A reference lc0 backend matches TensorFlow.
- SYCL matches the reference backend within defined tolerances.
- KDA state resets for every position and evaluation.
- The final policy head remains full pairwise attention.
- Existing MHA network tests still pass.
- Throughput, memory, validation quality, and Elo are measured against a fair
  MHA baseline.

## 21. Primary References

- [Kimi Linear paper, arXiv 2510.26692](https://arxiv.org/abs/2510.26692)
- [Moonshot AI Kimi Linear repository](https://github.com/MoonshotAI/Kimi-Linear)
- [Official Kimi model implementation](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py)
- [Flash Linear Attention KDA recurrence](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py)
- [Flash Linear Attention KDA kernels](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)
- [LCZero stable training branch](https://github.com/daniel-monroe/lczero-training/tree/stable)
- [Stable training example configuration](https://github.com/daniel-monroe/lczero-training/blob/stable/tf/configs/example.yaml)
