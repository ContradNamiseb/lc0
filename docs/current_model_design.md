# Current KDA Hybrid Model Architecture & Design Specification

**Status**: Active / Production Model Specification  
**Last Updated**: 2026-08-05  

**Target Repositories**:
- Training Stack: [`ContradNamiseb/lczero-training`](https://github.com/ContradNamiseb/lczero-training) (branch: `arch/kda-8-direction-scan`)
- Inference Engine: [`ContradNamiseb/lc0`](https://github.com/ContradNamiseb/lc0) (branch: `feature/kda-net-support`)

---

## 1. Overview & Purpose

This document specifies the updated architecture for Kimi Delta Attention (KDA) hybrid models in LCZero. It reflects the current implementation across both the TensorFlow training stack (`lczero-training`) and the native engine inference stack (`lc0`).

The model extends standard Transformer architectures by replacing selected Multi-Head Attention (MHA) layers in the encoder body with linear KDA recurrence blocks. To address the 2D non-causal nature of an $8 \times 8$ chess board, the current design incorporates **8 board traversal directions** and a **local $3 \times 3$ depthwise spatial convolution** prior to the linear recurrence.

---

## 2. Key Architecture Features

### 2.1 Hybrid Encoder Body Pattern
The recommended model uses a repeating KDA-to-MHA ratio (e.g., `[kda, kda, kda, mha]`):

```text
Input Board Planes (112 x 8 x 8)
  │
  ├──► Input Embedding / Preprocess Layer
  │
  ├──► Encoder Layer 1: KDA Block  + Residual + FFN
  ├──► Encoder Layer 2: KDA Block  + Residual + FFN
  ├──► Encoder Layer 3: KDA Block  + Residual + FFN
  ├──► Encoder Layer 4: MHA Block  + Residual + FFN
  │     (repeat 3:1 pattern)
  │
  └──► Policy Head (Full 64x64 Attention) & Value / Moves-Left Heads
```

- **KDA Blocks**: Function as linear recurrent mixers with $O(T)$ complexity, tracking sequence interactions across board traversals.
- **MHA Blocks**: Provide order-independent global square-to-square attention.
- **Policy Head**: Preserves full $64 \times 64$ pairwise QK attention matrix so that move logits directly represent from-square / to-square move structure.

---

## 3. Detailed Component Specification

### 3.1 Local $3 \times 3$ Depthwise Convolution (`kda_local_conv`)
- **Motivation**: Linear scans along 1D traversals require multiple steps to propagate information between spatially adjacent squares (e.g., diagonal pawn captures, knight moves, adjacent king/bishop steps). A $3 \times 3$ depthwise convolution allows each square to inspect its immediate 2D spatial neighborhood before entering the linear recurrence.
- **Implementation**:
  1. The 64 token embeddings are reshaped into an $8 \times 8 \times D_{\text{model}}$ spatial grid.
  2. A same-padded 2D depthwise convolution (`DepthwiseConv2D`, kernel size $3 \times 3$, padding `"same"`) is applied.
  3. A residual connection adds the convolution output back to the KDA input:
     $$\mathbf{x}_{\text{conv}} = \mathbf{x} + \text{DepthwiseConv2D}_{3 \times 3}(\mathbf{x})$$
  4. The result $\mathbf{x}_{\text{conv}}$ is fed into the KDA projections ($W_q, W_k, W_v, W_{\text{decay}}, W_\beta$).
- **Protobuf / Versioning**: Gated by `local_conv = true` in protobuf and `LC0_MINOR_WITH_KDA` (version 32).

---

### 3.2 8 Board Traversal Directions (`kda_directions`)

Because chess board interactions are 2-dimensional and non-causal, 64 squares are scanned using **8 fixed traversal schemes** to eliminate directional bias:

| Traversal Name | Description | Formula / Token Ordering |
| :--- | :--- | :--- |
| `rank_forward` | Rank-major scan, forward | `0, 1, 2, ..., 63` |
| `rank_reverse` | Rank-major scan, reverse | `63, 62, 61, ..., 0` |
| `file_forward` | File-major scan, forward | Transposed $8 \times 8$ grid (`file * 8 + rank`) |
| `file_reverse` | File-major scan, reverse | Transposed $8 \times 8$ grid reversed |
| `diag_forward` | Main diagonals (a1-h8 style), forward | Grouped along $r - f = c$, bottom-to-top |
| `diag_reverse` | Main diagonals (a1-h8 style), reverse | Reversed main diagonal order |
| `anti_diag_forward` | Anti-diagonals (a8-h1 style), forward | Grouped along $r + f = c$, bottom-to-top |
| `anti_diag_reverse` | Anti-diagonals (a8-h1 style), reverse | Reversed anti-diagonal order |

- **Head Partitioning**: Total heads $H$ are divided equally among the 8 directions (requires $H \pmod 8 = 0$, e.g., 16 heads = 2 heads per direction).
- **Single-Pass Recurrence Execution**:
  1. Input projections ($Q, K, V, \text{raw\_decay}, \beta$) are reshaped by head.
  2. Each head group is gathered into its direction's 64-square permutation array.
  3. All head groups are concatenated into a single tensor, executing the recurrence in **one unified scan call**.
  4. Output tokens are scattered back to standard square order using `argsort(permutation)`.
- **Protobuf / Versioning**: Enums `KDA_DIRECTION_RANK_FORWARD` (1) through `KDA_DIRECTION_ANTI_DIAG_REVERSE` (8). Version gate: `LC0_MINOR_WITH_KDA` (version 32).

---

### 3.3 KDA Recurrence & Gated Delta Rule

For batch size $B$, tokens $T=64$, heads $H$, key dimension $K$, and value dimension $V$:

$$\mathbf{q}_t, \mathbf{k}_t \in \mathbb{R}^{B \times H \times K}, \quad \mathbf{v}_t \in \mathbb{R}^{B \times H \times V}$$

1. **In-Layer L2 Normalization**:
   $$\mathbf{q}_t \leftarrow \frac{\mathbf{q}_t}{\|\mathbf{q}_t\|_2}, \quad \mathbf{k}_t \leftarrow \frac{\mathbf{k}_t}{\|\mathbf{k}_t\|_2}$$

2. **Per-Dimension Log Decay**:
   $$\text{raw\_decay} = W_{\text{decay\_b}} \left( W_{\text{decay\_a}} \mathbf{x} \right) \in \mathbb{R}^{B \times T \times H \times K}$$
   $$g_t = -\exp(a_{\log}) \cdot \text{softplus}(\text{raw\_decay}_t + \text{dt\_bias})$$
   $$g_t \leftarrow \max(g_t, \text{KDA\_LOG\_DECAY\_FLOOR}) \quad (\text{floor} = -10.0)$$

3. **Per-Head Sigmoid Beta**:
   $$\beta_t = \text{sigmoid}(W_\beta \mathbf{x}_t) \in \mathbb{R}^{B \times T \times H}$$

4. **Recurrent Gated Delta Rule**:
   $$\widetilde{S}_t = \exp(g_t) \odot S_{t-1}$$
   $$e_t = \mathbf{v}_t - \widetilde{S}_t^\top \mathbf{k}_t$$
   $$S_t = \widetilde{S}_t + \beta_t \mathbf{k}_t e_t^\top$$
   $$\mathbf{o}_t = S_t^\top \left( \frac{\mathbf{q}_t}{\sqrt{K}} \right)$$

- **Zero-State Guarantee**: Recurrent state $S_0$ is initialized to zero at the start of every position/node evaluation. State is strictly stateless across independent MCTS node evaluations and inference batch items.

---

### 3.4 Chunkwise Parallel Recurrence (Training)
To maximize GPU throughput during training (e.g., DirectML/CUDA/XPU), `lczero-training` implements a chunkwise-parallel recurrence (`KDA_CHUNK_SIZE = 16`):
- Converts 64 sequential board tokens into 4 chunks of size 16.
- Solves intra-chunk recurrence in closed form via dense matrix multiplications and exact Neumann series matrix inversion (nilpotent triangular attention matrix).
- Drops sequential step count from 64 to 4 while producing outputs identical to the sequential token-by-token recurrence up to $10^{-5}$ precision.

---

### 3.5 Gated Output Path (No Internal RMSNorm)
The current KDA layer uses a low-rank sigmoid output gate without internal RMSNorm (`output_rms_norm = false`):

$$\mathbf{g}_{\text{out}} = \text{sigmoid}\left( W_{\text{gate\_b}} \left( W_{\text{gate\_a}} \mathbf{x} \right) \right)$$
$$\mathbf{mixed} = \mathbf{o} \odot \mathbf{g}_{\text{out}}$$
$$\mathbf{output} = W_{\text{dense}} (\mathbf{mixed})$$

- Version Gate: `LC0_MINOR_WITH_KDA` (version 32).

---

## 4. Protobuf & Weight Mapping Summary

### Protobuf Message Structure (`proto/net.proto`)
```protobuf
enum KdaDirection {
  KDA_DIRECTION_UNKNOWN = 0;
  KDA_DIRECTION_RANK_FORWARD = 1;
  KDA_DIRECTION_RANK_REVERSE = 2;
  KDA_DIRECTION_FILE_FORWARD = 3;
  KDA_DIRECTION_FILE_REVERSE = 4;
  KDA_DIRECTION_DIAG_FORWARD = 5;
  KDA_DIRECTION_DIAG_REVERSE = 6;
  KDA_DIRECTION_ANTI_DIAG_FORWARD = 7;
  KDA_DIRECTION_ANTI_DIAG_REVERSE = 8;
}

message KDA {
  optional Layer q_w = 1;         optional Layer q_b = 2;
  optional Layer k_w = 3;         optional Layer k_b = 4;
  optional Layer v_w = 5;         optional Layer v_b = 6;
  optional Layer decay_a_w = 7;   optional Layer decay_a_b = 8;
  optional Layer decay_b_w = 9;   optional Layer decay_b_b = 10;
  optional Layer beta_w = 11;     optional Layer beta_b = 12;
  optional Layer a_log = 13;      optional Layer dt_bias = 14;
  optional Layer gate_a_w = 15;   optional Layer gate_a_b = 16;
  optional Layer gate_b_w = 17;   optional Layer gate_b_b = 18;
  optional Layer out_norm_gammas = 19;
  optional Layer dense_w = 20;    optional Layer dense_b = 21;
  optional uint32 key_dim = 22;   optional uint32 value_dim = 23;
  optional uint32 gate_rank = 24;
  optional float rms_norm_epsilon = 25;
  optional bool output_gate = 26 [default = true];
  optional bool output_rms_norm = 27 [default = true];
  optional Layer local_conv_w = 28;
  optional Layer local_conv_b = 29;
  optional bool local_conv = 30 [default = false];
}
```

### Version Gates
- `LC0_MINOR_WITH_KDA` (32): Required minimum LC0 version for all KDA hybrid network features (including 8-direction traversals, 3x3 depthwise local conv, and gated output without RMSNorm).

---

## 5. Configuration Example (`tf/configs/kda-hybrid.yaml`)

```yaml
model:
  embedding_size: 512
  policy_embedding_size: 512
  value_embedding_size: 128
  moves_left_embedding_size: 32

  encoder_layers: 12
  encoder_heads: 16
  encoder_d_model: 512
  encoder_dff: 768

  # Hybrid mixer pattern (3 KDA layers to 1 MHA layer)
  encoder_mixer_pattern: [kda, kda, kda, mha]

  # KDA parameters
  kda_key_dim: 32
  kda_value_dim: 32
  kda_gate_rank: 32
  kda_local_conv: true          # Enable 3x3 board depthwise pre-conv
  kda_directions:               # Full 8-direction scan
    - rank_forward
    - rank_reverse
    - file_forward
    - file_reverse
    - diag_forward
    - diag_reverse
    - anti_diag_forward
    - anti_diag_reverse
  kda_output_gate: true
```

---

## 6. Implementation Consistency Checklist

- [x] **8-Direction Scan**: Defined in `tfprocess.py` (`KDA_TRAVERSALS`), mapped in `net.py` (`set_kda_directions`), supported by `net.proto` (`KdaDirection`).
- [x] **Local 3x3 Conv**: Implemented in `tfprocess.py` (`kda` method, `kda_local_conv`), mapped in `net.py` (`set_encoder_mixer`), loaded in `lc0` (`BaseWeights::KDA::local_conv_w/b`).
- [x] **Recurrence & Floating-Point Stability**: L2 normalization on Q/K, bounded log-decay ($[-10, 0]$), sigmoid beta.
- [x] **Engine & Trainer Parity**: Both training (`KDARecurrence`) and engine backends (BLAS, SYCL) implement identical mathematical recurrence and weight structures.
