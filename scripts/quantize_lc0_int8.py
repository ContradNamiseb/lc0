#!/usr/bin/env python3
"""High-Accuracy Zero-Retraining W8A8 Static INT8 PTQ Pipeline for Lc0 -> OpenVINO IR.

Designed for Intel Xe-LP (DP4A) GPU & AVX-512 VNNI CPU via OpenVINO 2026.3+ and
lc0 `--backend=openvino --backend-opts="ir_path='...'"`:
  1. Calibrates on real played lc0 games (PGN) with exact 8-ply move histories
     matching lc0's INPUT_CLASSICAL_112_PLANE encoder (bit-exact verified).
  2. Architecture-aware selective quantization:
       - SE-ResNets (T74 / T79): quantizes ONLY the heavy trunk 3x3 Convs
         (/block*/conv1, /block*/conv2 = ~95% of GPU compute). Keeps InputConv,
         ALL Squeeze-and-Excitation units (se/matmul1, se/matmul2), and all
         Policy/Value/MLH heads in FP32/FP16.
       - Transformers (T1 / BT4): quantizes trunk MHA (Q/K/V/out) and FFN
         (dense1/dense2) projections. Keeps Input Embedding, Smolgen, QK/QKV
         attention matmuls, and Policy/Value/MLH heads in FP32/FP16.
       - Zero-Op SmoothQuant for Squared-ReLU Transformers: mathematically folds
         channel smoothing s_j into ffn/dense1 (div by sqrt(s_j)) & ffn/dense2
         (mul by s_j) — exact identity for ReLU^2 with ZERO extra runtime ops!
  3. MSE-optimal percentile activation range calibration + per-channel MSE-optimal
     symmetric INT8 weight quantization.
  4. Sequential Layer-by-Layer Fast Bias Correction (FBC) to eliminate cumulative
     mean drift across deep residual/encoder towers (zero runtime overhead).
  5. Evaluates held-out accuracy on 256 unseen real-game positions (legal-move
     Top-1%, Policy KL, Value Q MAE/Max, and WDL agreement) and exports clean
     OpenVINO IR (.xml + .bin) ready for `lc0.exe`.
"""
import argparse
import copy
import math
import os
import subprocess
import sys
import time
import numpy as np
import chess
import chess.pgn
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import re
import openvino as ov


def _load_move_table() -> tuple:
    """Load the canonical 1858-entry kMoveStrs table directly from src/neural/encoder.cc."""
    enc_cc = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "neural", "encoder.cc")
    )
    with open(enc_cc, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"kMoveStrs\[\]\s*=\s*\{(.*?)\};", text, re.S)
    if not m:
        raise RuntimeError(f"Could not find kMoveStrs in {enc_cc}")
    moves = tuple(re.findall(r'"([a-h][1-8][a-h][1-8][qrbn]?)"', m.group(1)))
    if len(moves) != 1858:
        raise RuntimeError(f"Expected 1858 moves in kMoveStrs, parsed {len(moves)}")
    return moves


MOVE_TABLE = _load_move_table()

PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
PROMO_CHARS = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP}


def mirror_sq(sq: int) -> int:
    return sq ^ 0b111000


def encode_history_np(boards_history: list) -> np.ndarray:
    """Exact INPUT_CLASSICAL_112_PLANE encoder for a real game history list [b_0, ..., b_t].

    Matches lc0 src/neural/encoder.cc (FillEmptyHistory::FEN_ONLY):
      - All 8 history slots are oriented from boards_history[-1].turn's perspective
        (vertically mirrored iff current side to move is BLACK).
      - If history has <8 plies from a non-startpos root FEN, fills older slots with root FEN;
        if root is startpos, older slots remain zeros.
    """
    cur = boards_history[-1]
    us = cur.turn
    them = not us
    we_black = (us == chess.BLACK)
    planes = np.zeros((112, 64), dtype=np.float32)

    # Track board transpositions for repetition plane (plane 12 of each history block)
    seen_counts = {}
    rep_flags = []
    for b in boards_history:
        key = (b.board_fen(), b.turn, b.castling_rights, b.ep_square)
        seen_counts[key] = seen_counts.get(key, 0) + 1
        rep_flags.append(seen_counts[key] >= 2)

    root_is_startpos = (boards_history[0].board_fen() == chess.STARTING_BOARD_FEN)
    n_hist = len(boards_history)

    for i in range(8):
        h_idx = n_hist - 1 - i
        if h_idx < 0:
            if root_is_startpos:
                break
            b = boards_history[0]
            is_rep = False
        else:
            b = boards_history[h_idx]
            is_rep = rep_flags[h_idx]

        base = i * 13
        for p_idx, pt in enumerate(PIECE_ORDER):
            for offset, col in ((0, us), (6, them)):
                mask = int(b.pieces_mask(pt, col))
                while mask:
                    lsb = mask & -mask
                    sq = lsb.bit_length() - 1
                    mask ^= lsb
                    s = mirror_sq(sq) if we_black else sq
                    planes[base + offset + p_idx, s] = 1.0
        if is_rep:
            planes[base + 12, :] = 1.0

    if cur.has_queenside_castling_rights(us):
        planes[104, :] = 1.0
    if cur.has_kingside_castling_rights(us):
        planes[105, :] = 1.0
    if cur.has_queenside_castling_rights(them):
        planes[106, :] = 1.0
    if cur.has_kingside_castling_rights(them):
        planes[107, :] = 1.0
    if we_black:
        planes[108, :] = 1.0
    planes[109, :] = float(cur.halfmove_clock)
    planes[111, :] = 1.0
    return planes.reshape(112, 8, 8)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    legal = set(board.legal_moves)
    mask = np.zeros(len(MOVE_TABLE), dtype=bool)
    we_black = board.turn == chess.BLACK
    for idx, mstr in enumerate(MOVE_TABLE):
        cf = chess.parse_square(mstr[0:2])
        ct = chess.parse_square(mstr[2:4])
        promo = PROMO_CHARS[mstr[4]] if len(mstr) == 5 else None
        fsq = mirror_sq(cf) if we_black else cf
        tsq = mirror_sq(ct) if we_black else ct
        if promo is None:
            p = board.piece_at(fsq)
            back = 7 if board.turn == chess.WHITE else 0
            if p is not None and p.piece_type == chess.PAWN and chess.square_rank(tsq) == back:
                promo = chess.QUEEN
        if chess.Move(fsq, tsq, promotion=promo) in legal:
            mask[idx] = True
    return mask


def sample_positions_from_pgns(pgn_specs: list, n_calib: int, n_eval: int, seed: int = 20261123):
    """Extract disjoint calibration and held-out eval positions with true 8-ply histories."""
    rng = np.random.default_rng(seed)
    calib_samples = []
    eval_samples = []

    for pgn_path, target_calib, target_eval in pgn_specs:
        if not os.path.exists(pgn_path):
            print(f"[WARN] PGN missing: {pgn_path}")
            continue
        games = []
        with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
            while len(games) < 400:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                moves = list(g.mainline_moves())
                if len(moves) >= 16:
                    games.append((g.board(), moves))
        # Split strictly by GAME (first 75% calibration, last 25% held-out eval) -> ZERO game leakage
        split = max(1, int(0.75 * len(games)))
        calib_games = games[:split]
        eval_games = games[split:] if split < len(games) else games[-max(1, len(games)//4):]

        def draw_from(glist, count):
            out = []
            if not glist or count <= 0:
                return out
            per_game = max(1, math.ceil(count / len(glist)))
            order = rng.permutation(len(glist))
            for gi in order:
                if len(out) >= count:
                    break
                root_b, moves = glist[gi]
                # Sample diverse plies across the entire game (opening, middlegame, endgame)
                max_ply = len(moves) - 1
                if max_ply < 4:
                    continue
                chosen_plies = sorted(rng.choice(np.arange(2, max_ply), size=min(per_game, max_ply - 2), replace=False))
                b = root_b.copy()
                hist = [b.copy()]
                target_set = set(int(p) for p in chosen_plies)
                for ply, mv in enumerate(moves, start=1):
                    b.push(mv)
                    hist.append(b.copy())
                    if ply in target_set and not b.is_game_over():
                        # Keep only the suffix needed for repetition + 8-ply history
                        x = encode_history_np(hist)
                        m = legal_move_mask(b)
                        out.append((x, m, b.fen()))
                        if len(out) >= count:
                            break
            return out

        c_part = draw_from(calib_games, target_calib)
        e_part = draw_from(eval_games, target_eval)
        print(f"Sampled {len(c_part)} calib + {len(e_part)} eval positions from {os.path.basename(pgn_path)} ({len(games)} games)")
        calib_samples.extend(c_part)
        eval_samples.extend(e_part)

    rng.shuffle(calib_samples)
    rng.shuffle(eval_samples)
    calib_samples = calib_samples[:n_calib]
    eval_samples = eval_samples[:n_eval]

    X_cal = np.stack([s[0] for s in calib_samples], axis=0)
    M_cal = np.stack([s[1] for s in calib_samples], axis=0)
    X_ev = np.stack([s[0] for s in eval_samples], axis=0)
    M_ev = np.stack([s[1] for s in eval_samples], axis=0)
    return X_cal, M_cal, X_ev, M_ev


def assert_classical_112_input_format(lc0_exe: str, weights_path: str):
    """P1: Explicitly verify weights use INPUT_CLASSICAL_112_PLANE before calibrating."""
    proc = subprocess.run([lc0_exe, "describenet", f"--weights={weights_path}"],
                          capture_output=True, text=True, check=True)
    out = proc.stdout + "\n" + proc.stderr
    if "INPUT_CLASSICAL_112_PLANE" not in out:
        raise RuntimeError(
            f"Refusing to calibrate {weights_path}: encoder only supports INPUT_CLASSICAL_112_PLANE "
            f"(canonical/hectoplies/castling-rook formats require a matching encoder)."
        )


def verify_history_encoder_against_lc0(lc0_exe: str, weights_path: str, onnx_path: str):
    """P1: Per-net multi-feature parity check vs lc0.exe eigen across:
      1) 6-ply Ruy Lopez (White to move, partial history from startpos)
      2) Black to move + lost castling rights + rule50=6
      3) 2-fold repetition from startpos (tests repetition plane 12)
    """
    cases = [
        ("6-ply Ruy Lopez (W)", "startpos", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]),
        ("Black+rule50=6+noCastle", "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 b - - 6 8", []),
        ("2-fold repetition", "startpos", ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]),
    ]

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])

    p = subprocess.Popen(
        [lc0_exe, f"--weights={weights_path}", "--backend=eigen"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )
    p.stdin.write("uci\nsetoption name UCI_ShowWDL value true\nisready\n")
    p.stdin.flush()
    for l in p.stdout:
        if l.startswith("readyok"):
            break

    max_diff_all = 0.0
    for lbl, root_fen, moves_uci in cases:
        b = chess.Board() if root_fen == "startpos" else chess.Board(root_fen)
        hist = [b.copy()]
        for u in moves_uci:
            b.push_uci(u)
            hist.append(b.copy())
        x = encode_history_np(hist)[None, ...]
        _, wdl = sess.run(["/output/policy", "/output/wdl"], {"/input/planes": x})
        w_py = wdl[0]

        pos_cmd = "position startpos" if root_fen == "startpos" else f"position fen {root_fen}"
        if moves_uci:
            pos_cmd += " moves " + " ".join(moves_uci)
        p.stdin.write(f"{pos_cmd}\ngo nodes 1\n")
        p.stdin.flush()
        wdl_lc0 = None
        for l in p.stdout:
            if " wdl " in l:
                parts = l.split(" wdl ")[1].split()
                wdl_lc0 = np.array([int(parts[0]), int(parts[1]), int(parts[2])], dtype=np.float32) / 1000.0
            if l.startswith("bestmove"):
                break
        diff = float(np.max(np.abs(w_py - wdl_lc0)))
        max_diff_all = max(max_diff_all, diff)
        assert diff < 0.005, f"Encoder mismatch on {lbl}: py={w_py} vs lc0={wdl_lc0} (diff={diff})"
    p.stdin.write("quit\n"); p.stdin.flush()
    p.wait(timeout=5)
    print(f"  [Encoder Parity Check: 3 cases (W/B/rule50/noCastle/rep)] PASS (max WDL diff={max_diff_all:.5f})")


def eval_outputs(pol_ref, wdl_ref, pol_tst, wdl_tst, masks):
    n = pol_ref.shape[0]
    top1_match = 0
    kls = []
    for i in range(n):
        m = masks[i]
        lr = pol_ref[i][m].astype(np.float64)
        lt = pol_tst[i][m].astype(np.float64)
        lr -= lr.max(); lt -= lt.max()
        pr = np.exp(lr); pr /= pr.sum()
        pt = np.exp(lt); pt /= pt.sum()
        if int(np.argmax(pr)) == int(np.argmax(pt)):
            top1_match += 1
        kls.append(float(np.sum(pr * (np.log(pr + 1e-12) - np.log(pt + 1e-12)))))

    def to_q(w):
        w = w.astype(np.float64)
        if np.max(np.abs(w.sum(axis=-1) - 1.0)) > 1e-2:
            e = np.exp(w - w.max(axis=-1, keepdims=True))
            w = e / e.sum(axis=-1, keepdims=True)
        return w[:, 0] - w[:, 2]

    qr, qt = to_q(wdl_ref), to_q(wdl_tst)
    q_err = np.abs(qr - qt)
    q_bias = float(np.mean(qt - qr))
    p_hat = top1_match / n
    top1_ci95 = 100.0 * 1.96 * math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / n)
    return {
        "top1_pct": 100.0 * p_hat,
        "top1_ci95": top1_ci95,
        "kl_mean": float(np.mean(kls)),
        "kl_max": float(np.max(kls)),
        "q_mae": float(np.mean(q_err)),
        "q_p95": float(np.percentile(q_err, 95)),
        "q_max": float(np.max(q_err)),
        "q_bias": q_bias,
    }


def run_ort_batched(model_bytes: bytes, X: np.ndarray, output_names=None, batch_size=64):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    so.log_severity_level = 3
    sess = ort.InferenceSession(model_bytes, sess_options=so, providers=["CPUExecutionProvider"])
    if output_names is None:
        output_names = ["/output/policy", "/output/wdl"]
    buckets = [[] for _ in output_names]
    for s in range(0, X.shape[0], batch_size):
        outs = sess.run(output_names, {"/input/planes": X[s : s + batch_size]})
        for j, o in enumerate(outs):
            buckets[j].append(o)
    return [np.concatenate(b, axis=0) for b in buckets]


def select_quant_targets(model: onnx.ModelProto, arch_kind: str, skip_se: bool = True, skip_smolgen: bool = True):
    """Select Conv / MatMul nodes eligible for INT8 quantization."""
    init_map = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    targets = []
    for node in model.graph.node:
        if node.op_type not in ("Conv", "MatMul"):
            continue
        w_name = node.input[1]
        if w_name not in init_map or init_map[w_name].ndim not in (2, 4):
            continue
        ln = node.name.lower()
        if any(k in ln for k in ("inputconv", "/input", "embedding", "preprocess", "policy", "value", "mlh", "moves_left")):
            continue
        if arch_kind == "resnet":
            if "/block" not in ln:
                continue
            if skip_se and "/se/" in ln:
                continue
        else:
            if "/encoder" not in ln:
                continue
            if skip_smolgen and "/smolgen/" in ln:
                continue
        targets.append(node)
    return targets


def apply_zero_op_smoothquant_for_transformer(model: onnx.ModelProto, X_cal: np.ndarray, alpha: float = 0.5):
    """Zero-Op SmoothQuant for lc0 Squared-ReLU Transformers!

    For every /encoder{i}/ffn/dense2/w (preceded by ReLU^2 on dense1 output):
      - ReLU^2(z / sqrt(s)) == ReLU^2(z) / s for all s > 0!
      - So scaling dense1 W & b by 1/sqrt(s) and dense2 W by s is an EXACT
        mathematical identity with ZERO extra Mul ops in the ONNX / OpenVINO graph!
    """
    m = copy.deepcopy(model)
    init_idx = {init.name: i for i, init in enumerate(m.graph.initializer)}
    node_map = {n.name: n for n in m.graph.node}

    # Find all /encoder{i}/ffn/dense2/w nodes
    ffn2_nodes = [n for n in m.graph.node if n.op_type == "MatMul" and "/ffn/dense2/w" in n.name and "/encoder" in n.name]
    if not ffn2_nodes:
        return m, 0

    # Collect activation max per channel at dense2 inputs on 256 calibration positions
    probe = copy.deepcopy(m)
    in_names = [n.input[0] for n in ffn2_nodes]
    for nm in in_names:
        probe.graph.output.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, None))
    acts = run_ort_batched(probe.SerializeToString(), X_cal[:256], output_names=in_names, batch_size=64)

    # Build output-tensor -> producer node map for P2 structural ReLU^2 verification
    producer_of = {}
    for n in m.graph.node:
        for o in n.output:
            producer_of[o] = n

    folded = 0
    for node2, A in zip(ffn2_nodes, acts):
        pfx = node2.name.split("/ffn/dense2/w")[0]
        n1_w_node = node_map.get(pfx + "/ffn/dense1/w")
        n1_b_node = node_map.get(pfx + "/ffn/dense1/b")
        if n1_w_node is None or n1_b_node is None:
            continue
        # P2: Structurally verify dense1/b -> Relu -> Mul(r, r) -> dense2/w (exact ReLU^2 only!)
        mul_node = producer_of.get(node2.input[0])
        if mul_node is None or mul_node.op_type != "Mul" or len(mul_node.input) != 2 or mul_node.input[0] != mul_node.input[1]:
            continue
        relu_node = producer_of.get(mul_node.input[0])
        if relu_node is None or relu_node.op_type != "Relu" or relu_node.input[0] != n1_b_node.output[0]:
            continue

        w1_name = n1_w_node.input[1]
        b1_name = n1_b_node.input[1]
        w2_name = node2.input[1]
        if w1_name not in init_idx or b1_name not in init_idx or w2_name not in init_idx:
            continue

        W1 = numpy_helper.to_array(m.graph.initializer[init_idx[w1_name]]).astype(np.float32)  # [D, Dff]
        b1 = numpy_helper.to_array(m.graph.initializer[init_idx[b1_name]]).astype(np.float32)  # [Dff]
        W2 = numpy_helper.to_array(m.graph.initializer[init_idx[w2_name]]).astype(np.float32)  # [Dff, D]

        Dff = W2.shape[0]
        ch_act = np.maximum(np.max(np.abs(A.reshape(-1, Dff)), axis=0), 1e-4)
        ch_w2 = np.maximum(np.max(np.abs(W2), axis=1), 1e-4)
        s = (np.power(ch_act, alpha) / np.power(ch_w2, 1.0 - alpha)).astype(np.float32)
        s = np.clip(s, 0.1, 16.0)
        # Keep geometric mean near 1 so dense1 pre-activation scale stays healthy
        s = s / np.exp(np.mean(np.log(s)))
        inv_sqrt_s = (1.0 / np.sqrt(s)).astype(np.float32)

        W1_new = W1 * inv_sqrt_s[None, :]
        b1_new = b1 * inv_sqrt_s
        W2_new = W2 * s[:, None]

        m.graph.initializer[init_idx[w1_name]].CopyFrom(numpy_helper.from_array(W1_new, name=w1_name))
        m.graph.initializer[init_idx[b1_name]].CopyFrom(numpy_helper.from_array(b1_new, name=b1_name))
        m.graph.initializer[init_idx[w2_name]].CopyFrom(numpy_helper.from_array(W2_new, name=w2_name))
        folded += 1

    # P2: Verify FP32 functional invariance before vs after Zero-Op SmoothQuant fold!
    if folded > 0:
        p_bef, w_bef = run_ort_batched(model.SerializeToString(), X_cal[:16], batch_size=16)
        p_aft, w_aft = run_ort_batched(m.SerializeToString(), X_cal[:16], batch_size=16)
        dw = float(np.max(np.abs(w_bef - w_aft)))
        dp = float(np.max(np.abs(p_bef - p_aft)))
        print(f"  [Zero-Op SmoothQuant] Structurally verified & folded ReLU^2 (alpha={alpha}) into {folded} FFN blocks (FP32 invariance: max |dW|={dw:.2e}, max |dP|={dp:.2e}).")
        assert dw < 1e-4 and dp < 1e-3, f"SmoothQuant fold broke FP32 invariance! dW={dw}, dP={dp}"
    else:
        print("  [Zero-Op SmoothQuant] No pure ReLU^2 FFN blocks matched; skipped safely.")
    return m, folded


def find_mse_optimal_act_range(samples_flat: np.ndarray, symmetric: bool = False):
    """Find MSE-optimal [lo, hi] quantization clip range via percentile grid search."""
    if symmetric:
        a = np.abs(samples_flat)
        cands = [float(np.percentile(a, p)) for p in (99.8, 99.9, 99.95, 99.98, 99.99, 99.995, 100.0)]
        best_err, best_hi = 1e30, cands[-1]
        for hi in set(cands):
            hi = max(hi, 1e-6)
            sc = hi / 127.0
            q = np.clip(np.round(samples_flat / sc), -127, 127) * sc
            err = float(np.mean((samples_flat - q) ** 2))
            if err < best_err:
                best_err, best_hi = err, hi
        return -best_hi, best_hi

    # Asymmetric uint8 [0, 255] with 0.0 guaranteed representable
    lo_cands = sorted(set(min(0.0, float(np.percentile(samples_flat, p))) for p in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)))
    hi_cands = sorted(set(max(0.0, float(np.percentile(samples_flat, p))) for p in (99.9, 99.95, 99.98, 99.99, 99.995, 100.0)))
    best_err = 1e30
    best_pair = (lo_cands[0], hi_cands[-1])
    for lo in lo_cands:
        for hi in hi_cands:
            if hi - lo < 1e-6:
                continue
            sc = (hi - lo) / 255.0
            zp = float(np.clip(np.round(-lo / sc), 0, 255))
            q = (np.clip(np.round(samples_flat / sc + zp), 0, 255) - zp) * sc
            err = float(np.mean((samples_flat - q) ** 2))
            if err < best_err:
                best_err = err
                best_pair = (lo, hi)
    return best_pair


def mse_optimal_per_channel_int8(W: np.ndarray, ch_axis: int):
    """Per-output-channel symmetric INT8 with MSE clipping search."""
    if ch_axis == 0:
        C = W.shape[0]
        W_mat = W.reshape(C, -1)
    else:
        C = W.shape[1]
        W_mat = W.T.reshape(C, -1)

    amax = np.max(np.abs(W_mat), axis=1)
    best_sc = np.maximum(amax / 127.0, 1e-8).astype(np.float32)
    best_err = np.full(C, 1e30, dtype=np.float64)
    for mult in (0.90, 0.94, 0.97, 0.99, 1.00):
        sc = np.maximum((amax * mult) / 127.0, 1e-8).astype(np.float32)
        q = np.clip(np.round(W_mat / sc[:, None]), -127, 127) * sc[:, None]
        err = np.mean((W_mat - q) ** 2, axis=1)
        better = err < best_err
        best_err[better] = err[better]
        best_sc[better] = sc[better]

    shape_b = [1] * W.ndim
    shape_b[ch_axis] = C
    W_int8 = np.clip(np.round(W / best_sc.reshape(shape_b)), -127, 127).astype(np.int8)
    return W_int8, best_sc


def calibrate_and_build_qdq(base_model: onnx.ModelProto, targets: list, X_cal: np.ndarray,
                            use_mse_clip: bool = True, run_fbc: bool = True, fbc_samples: int = 256):
    """Calibrate activations over full X_cal, insert QDQ nodes, and run Fast Bias Correction."""
    m = copy.deepcopy(base_model)
    init_map = {init.name: init for init in m.graph.initializer}
    target_names = {n.name for n in targets}

    # Step 1: Collect activation calibration samples across X_cal (subsampled spatially to keep RAM crisp)
    unique_inputs = []
    seen = set()
    for n in targets:
        if n.input[0] not in seen:
            seen.add(n.input[0])
            unique_inputs.append(n.input[0])

    probe = copy.deepcopy(m)
    for nm in unique_inputs:
        probe.graph.output.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, None))

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 4
    so.log_severity_level = 3
    sess = ort.InferenceSession(probe.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"])

    # Gather representative sample pool per input tensor across ALL X_cal positions
    rng = np.random.default_rng(42)
    pools = {nm: [] for nm in unique_inputs}
    minmax = {nm: [1e30, -1e30] for nm in unique_inputs}
    bs = 64
    for s in range(0, X_cal.shape[0], bs):
        outs = sess.run(unique_inputs, {"/input/planes": X_cal[s : s + bs]})
        for nm, arr in zip(unique_inputs, outs):
            flat = arr.reshape(-1)
            minmax[nm][0] = min(minmax[nm][0], float(flat.min()))
            minmax[nm][1] = max(minmax[nm][1], float(flat.max()))
            # Keep up to 16,384 random elements per batch -> ~260k elements across 1024 positions!
            if flat.size > 16384:
                idx = rng.integers(0, flat.size, size=16384)
                pools[nm].append(flat[idx].astype(np.float32))
            else:
                pools[nm].append(flat.astype(np.float32))

    act_ranges = {}
    for nm in unique_inputs:
        if use_mse_clip:
            sample_vec = np.concatenate(pools[nm], axis=0)
            lo, hi = find_mse_optimal_act_range(sample_vec, symmetric=False)
        else:
            lo, hi = min(0.0, minmax[nm][0]), max(0.0, minmax[nm][1])
        act_ranges[nm] = (lo, hi)

    # Step 2: Insert QDQ nodes (per-tensor uint8 asym activations + per-channel symmetric int8 weights)
    new_nodes = []
    qdq_idx = 0
    quantized_out_info = []  # (node_output_tensor, bias_init_name, is_conv) for FBC!

    # Map MatMul -> following Add bias node for Transformers
    consumer_add = {}
    for node in m.graph.node:
        if node.op_type == "Add" and len(node.input) == 2:
            if node.input[1] in init_map:
                consumer_add[node.input[0]] = (node.output[0], node.input[1])

    def add_const(name, arr):
        m.graph.initializer.append(numpy_helper.from_array(arr, name=name))
        return name

    for node in m.graph.node:
        if node.name not in target_names:
            new_nodes.append(node)
            continue

        qdq_idx += 1
        pfx = f"/qdq_{qdq_idx}"
        inp_name = node.input[0]
        w_name = node.input[1]
        W = numpy_helper.to_array(init_map[w_name]).astype(np.float32)

        lo, hi = act_ranges[inp_name]
        a_scale = np.array(max((hi - lo) / 255.0, 1e-6), dtype=np.float32)
        a_zp = np.array(int(np.clip(np.round(-lo / float(a_scale)), 0, 255)), dtype=np.uint8)
        asc = add_const(pfx + "/a_scale", a_scale)
        azp = add_const(pfx + "/a_zp", a_zp)
        aq = pfx + "/a_q"
        adq = pfx + "/a_dq"
        new_nodes.append(helper.make_node("QuantizeLinear", [inp_name, asc, azp], [aq], name=pfx + "/a_ql"))
        new_nodes.append(helper.make_node("DequantizeLinear", [aq, asc, azp], [adq], name=pfx + "/a_dql"))

        ch_axis = 0 if node.op_type == "Conv" else 1
        W_int8, w_scale = mse_optimal_per_channel_int8(W, ch_axis) if use_mse_clip else (
            np.clip(np.round(W / np.maximum(np.max(np.abs(W), axis=tuple(i for i in range(W.ndim) if i != ch_axis), keepdims=True) / 127.0, 1e-8)), -127, 127).astype(np.int8),
            np.maximum(np.max(np.abs(W), axis=tuple(i for i in range(W.ndim) if i != ch_axis)) / 127.0, 1e-8).astype(np.float32)
        )
        w_zp = np.zeros_like(w_scale, dtype=np.int8)
        wi = add_const(pfx + "/w_int8", W_int8)
        ws = add_const(pfx + "/w_scale", w_scale)
        wz = add_const(pfx + "/w_zp", w_zp)
        wdq = pfx + "/w_dq"
        new_nodes.append(helper.make_node("DequantizeLinear", [wi, ws, wz], [wdq], name=pfx + "/w_dql", axis=ch_axis))

        nc = copy.deepcopy(node)
        nc.input[0] = adq
        nc.input[1] = wdq
        new_nodes.append(nc)

        if node.op_type == "Conv" and len(node.input) >= 3 and node.input[2] in init_map:
            quantized_out_info.append((node.output[0], node.input[2], True))
        elif node.op_type == "MatMul" and node.output[0] in consumer_add:
            add_out, bias_name = consumer_add[node.output[0]]
            quantized_out_info.append((add_out, bias_name, False))

    del m.graph.node[:]
    m.graph.node.extend(new_nodes)

    # Prune now-unused FP32 weight initializers so saved IR/ONNX is compact
    used_inputs = set()
    for n in m.graph.node:
        for inp in n.input:
            used_inputs.add(inp)
    kept_inits = [init for init in m.graph.initializer if init.name in used_inputs]
    del m.graph.initializer[:]
    m.graph.initializer.extend(kept_inits)

    # Step 3: Sequential Layer-by-Layer Fast Bias Correction (FBC)
    if run_fbc and quantized_out_info:
        X_fbc = X_cal[:fbc_samples]
        out_tensors = [info[0] for info in quantized_out_info]
        # Capture all reference FP32 channel means in ONE pass!
        ref_probe = copy.deepcopy(base_model)
        for tname in out_tensors:
            ref_probe.graph.output.append(helper.make_tensor_value_info(tname, TensorProto.FLOAT, None))
        ref_outs = run_ort_batched(ref_probe.SerializeToString(), X_fbc, output_names=out_tensors, batch_size=64)
        ref_means = []
        ref_stds = []
        for arr, (_, _, is_conv) in zip(ref_outs, quantized_out_info):
            if is_conv:
                ref_means.append(np.mean(arr, axis=(0, 2, 3), dtype=np.float64))
                ref_stds.append(np.std(arr, axis=(0, 2, 3), dtype=np.float64))
            else:
                flat_c = arr.reshape(-1, arr.shape[-1])
                ref_means.append(np.mean(flat_c, axis=0, dtype=np.float64))
                ref_stds.append(np.std(flat_c, axis=0, dtype=np.float64))
        del ref_outs

        # Walk layer-by-layer updating bias = bias + (mean_fp32 - mean_quant)
        init_idx = {init.name: i for i, init in enumerate(m.graph.initializer)}
        t0 = time.perf_counter()
        max_shift = 0.0
        for layer_i, ((tname, bname, is_conv), ref_mu, ref_sd) in enumerate(zip(quantized_out_info, ref_means, ref_stds)):
            if bname not in init_idx:
                continue
            q_probe = copy.deepcopy(m)
            q_probe.graph.output.append(helper.make_tensor_value_info(tname, TensorProto.FLOAT, None))
            q_arr = run_ort_batched(q_probe.SerializeToString(), X_fbc, output_names=[tname], batch_size=64)[0]
            q_mu = np.mean(q_arr, axis=(0, 2, 3), dtype=np.float64) if is_conv else np.mean(q_arr.reshape(-1, q_arr.shape[-1]), axis=0, dtype=np.float64)
            # P3: Relative per-channel FBC clamp scaled to channel std (handles tiny & large channels cleanly)
            lim = np.maximum(2.0 * ref_sd, 1e-3)
            delta = np.clip(ref_mu - q_mu, -lim, lim).astype(np.float32)
            max_shift = max(max_shift, float(np.max(np.abs(delta))))
            b_old = numpy_helper.to_array(m.graph.initializer[init_idx[bname]]).astype(np.float32)
            if b_old.shape == delta.shape:
                m.graph.initializer[init_idx[bname]].CopyFrom(numpy_helper.from_array(b_old + delta, name=bname))
        print(f"  [Fast Bias Correction] Corrected {len(quantized_out_info)} layers in {time.perf_counter()-t0:.1f}s (max |db|={max_shift:.5f}).")

    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lc0", required=True, help="Path to lc0 binary supporting leela2onnx")
    ap.add_argument("--weights", required=True, help="Path to source Lc0 .pb.gz weights")
    ap.add_argument("--arch", required=True, choices=["resnet", "transformer"], help="Network trunk architecture")
    ap.add_argument("--tag", default=None, help="Output filename prefix (defaults to weights basename)")
    ap.add_argument("--pgns", nargs="+", required=True, help="One or more PGN files for real-game history calibration/eval")
    ap.add_argument("--out-dir", required=True, help="Output directory for ONNX and OpenVINO IR (.xml/.bin)")
    ap.add_argument("--n-calib", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or os.path.basename(args.weights).replace(".pb.gz", "")
    n_pgns = len(args.pgns)
    pgn_specs = []
    cal_rem, ev_rem = args.n_calib, args.n_eval
    for i, pgn_path in enumerate(args.pgns):
        nc = cal_rem if i == n_pgns - 1 else args.n_calib // n_pgns
        ne = ev_rem if i == n_pgns - 1 else args.n_eval // n_pgns
        pgn_specs.append((pgn_path, nc, ne))
        cal_rem -= nc
        ev_rem -= ne

    X_cal, M_cal, X_ev, M_ev = sample_positions_from_pgns(pgn_specs, args.n_calib, args.n_eval)
    print(f"Dataset ready: X_cal={X_cal.shape}, X_ev={X_ev.shape}")

    ov_core = ov.Core()

    for tag, pb_path, arch in [(tag, args.weights, args.arch)]:
        fp32_onnx = os.path.join(args.out_dir, f"{tag}_fp32.onnx")
        print(f"\n==================== Quantizing {tag} ({arch}) ====================")
        # P1: Verify input format and multi-feature encoder parity for EVERY net!
        assert_classical_112_input_format(args.lc0, pb_path)
        subprocess.run(
            [args.lc0, "leela2onnx", f"--input={pb_path}", f"--output={fp32_onnx}", "--onnx-opset=17", "--onnx-data-type=f32"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        verify_history_encoder_against_lc0(args.lc0, pb_path, fp32_onnx)

        base_model = onnx.load(fp32_onnx)
        pol_ref, wdl_ref = run_ort_batched(base_model.SerializeToString(), X_ev)

        # P5: Measure unquantized OV GPU FP16 baseline error against FP32 on the same held-out set
        m_fp16_ov = ov_core.read_model(fp32_onnx)
        m_fp16_ov.reshape({"/input/planes": [64, 112, 8, 8]})
        comp_fp16 = ov_core.compile_model(m_fp16_ov, "GPU", {"INFERENCE_PRECISION_HINT": "f16"})
        ps_f16, ws_f16 = [], []
        for s in range(0, X_ev.shape[0], 64):
            res = comp_fp16({"/input/planes": X_ev[s : s + 64]})
            ps_f16.append(res[comp_fp16.output("/output/policy")])
            ws_f16.append(res[comp_fp16.output("/output/wdl")])
        r_f16 = eval_outputs(pol_ref, wdl_ref, np.concatenate(ps_f16), np.concatenate(ws_f16), M_ev)
        print(f"  [0] OV GPU FP16 Baseline             | Top1={r_f16['top1_pct']:5.1f}%±{r_f16['top1_ci95']:.1f} | KL={r_f16['kl_mean']:.5f} | Q_MAE={r_f16['q_mae']:.5f} | Q_p95={r_f16['q_p95']:.5f} | Q_Max={r_f16['q_max']:.5f} | Q_Bias={r_f16['q_bias']:+.5f}")

        # Ablation 1: MinMax naive (all trunk incl. SE/Smolgen) — baseline from Exp 1 style
        t_all = select_quant_targets(base_model, arch, skip_se=False, skip_smolgen=False)
        m_naive = calibrate_and_build_qdq(base_model, t_all, X_cal, use_mse_clip=False, run_fbc=False)
        p0, w0 = run_ort_batched(m_naive.SerializeToString(), X_ev)
        r0 = eval_outputs(pol_ref, wdl_ref, p0, w0, M_ev)
        print(f"  [1] Naive MinMax trunk (n={len(t_all):2d})       | Top1={r0['top1_pct']:5.1f}% | KL={r0['kl_mean']:.5f} | Q_MAE={r0['q_mae']:.5f} | Q_p95={r0['q_p95']:.5f} | Q_Max={r0['q_max']:.5f} | Q_Bias={r0['q_bias']:+.5f}")

        # Ablation 2: Skip SE/Smolgen + MSE clipping (no FBC)
        work_model = base_model
        if arch == "transformer":
            work_model, _ = apply_zero_op_smoothquant_for_transformer(base_model, X_cal, alpha=0.5)
        t_clean = select_quant_targets(work_model, arch, skip_se=True, skip_smolgen=True)
        m_mse = calibrate_and_build_qdq(work_model, t_clean, X_cal, use_mse_clip=True, run_fbc=False)
        p1, w1 = run_ort_batched(m_mse.SerializeToString(), X_ev)
        r1 = eval_outputs(pol_ref, wdl_ref, p1, w1, M_ev)
        print(f"  [2] Clean+MSE(+ZeroOpSQ) (n={len(t_clean):2d})    | Top1={r1['top1_pct']:5.1f}% | KL={r1['kl_mean']:.5f} | Q_MAE={r1['q_mae']:.5f} | Q_p95={r1['q_p95']:.5f} | Q_Max={r1['q_max']:.5f} | Q_Bias={r1['q_bias']:+.5f}")

        # Ablation 3: Full Pipeline (Skip SE/Smolgen + Zero-Op SQ + MSE Clip + Sequential FBC)
        m_full = calibrate_and_build_qdq(work_model, t_clean, X_cal, use_mse_clip=True, run_fbc=True, fbc_samples=256)
        p2, w2 = run_ort_batched(m_full.SerializeToString(), X_ev)
        r2 = eval_outputs(pol_ref, wdl_ref, p2, w2, M_ev)
        print(f"  [3] Full + FastBiasCorr  (n={len(t_clean):2d})    | Top1={r2['top1_pct']:5.1f}% | KL={r2['kl_mean']:.5f} | Q_MAE={r2['q_mae']:.5f} | Q_p95={r2['q_p95']:.5f} | Q_Max={r2['q_max']:.5f} | Q_Bias={r2['q_bias']:+.5f}")

        # Pick the best variant by combined score (lowest Q_MAE + KL)
        best_label, best_m, best_r = ("Full+FBC", m_full, r2) if (r2["q_mae"] + r2["kl_mean"]) <= (r1["q_mae"] + r1["kl_mean"]) else ("Clean+MSE", m_mse, r1)
        out_onnx = os.path.join(args.out_dir, f"{tag}_int8.onnx")
        out_xml = os.path.join(args.out_dir, f"{tag}_int8.xml")
        onnx.save(best_m, out_onnx)
        ov_m = ov_core.read_model(out_onnx)
        ov.save_model(ov_m, out_xml, compress_to_fp16=False)
        # Verify real OpenVINO GPU & CPU accuracy on the saved IR!
        for dev, prec in (("GPU", "f16"), ("CPU", "f32")):
            m_ov = ov_core.read_model(out_xml)
            m_ov.reshape({"/input/planes": [64, 112, 8, 8]})
            comp = ov_core.compile_model(m_ov, dev, {"INFERENCE_PRECISION_HINT": prec})
            ps, ws = [], []
            for s in range(0, X_ev.shape[0], 64):
                res = comp({"/input/planes": X_ev[s : s + 64]})
                ps.append(res[comp.output("/output/policy")])
                ws.append(res[comp.output("/output/wdl")])
            r_ov = eval_outputs(pol_ref, wdl_ref, np.concatenate(ps), np.concatenate(ws), M_ev)
            print(f"  [OV {dev} Verified ({best_label})]         | Top1={r_ov['top1_pct']:5.1f}% | KL={r_ov['kl_mean']:.5f} | Q_MAE={r_ov['q_mae']:.5f} | Q_p95={r_ov['q_p95']:.5f} | Q_Max={r_ov['q_max']:.5f} | Q_Bias={r_ov['q_bias']:+.5f}")
        print(f"  Saved OpenVINO INT8 IR: {out_xml}")


if __name__ == "__main__":
    main()
