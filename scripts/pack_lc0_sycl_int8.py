#!/usr/bin/env python3
"""Pack a calibrated W8A8 Static INT8 Transformer or SE-ResNet into native Lc0 .pb.gz format for SYCL INT8.

Uses the INT8 quantization slots in proto/net.proto:
  - Weights::Layer::Encoding::INT8 = 5 (raw signed int8 bytes, 1 byte/weight)
  - MHA fields 13..20: q/k/v/dense_w_scales (FLOAT32 Layer) + qkv/dense act_scale (float) & act_zp (uint32)
  - FFN fields 5..10 : dense1/dense2_w_scales (FLOAT32 Layer) + dense1/dense2 act_scale (float) & act_zp (uint32)
  - ConvBlock fields 7..9: w_scales (FLOAT32 Layer) + act_scale (float) & act_zp (uint32)

All non-quantized layers (Input Embedding/Conv, Smolgen, SE units, LayerNorms, Policy/Value/MLH heads)
are preserved byte-for-byte from the source .pb.gz.
"""
import argparse
import gzip
import os
import struct
import sys
import numpy as np
import onnx
from onnx import numpy_helper

# ---------------- Minimal Proto2 Wire Codec (Zero external protoc needed) ----------------

def _encode_varint(v: int) -> bytes:
    if v < 0:
        v &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _decode_varint(buf: bytes, pos: int):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _iter_proto_fields(buf: bytes):
    """Yield (field_number, wire_type, raw_field_bytes, payload_bytes) for every field in buf."""
    pos = 0
    n = len(buf)
    while pos < n:
        start = pos
        tag, pos = _decode_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 7
        if wire_type == 0:  # varint
            val_start = pos
            _, pos = _decode_varint(buf, pos)
            payload = buf[val_start:pos]
        elif wire_type == 1:  # 64-bit
            payload = buf[pos:pos + 8]
            pos += 8
        elif wire_type == 2:  # length-delimited
            length, pos = _decode_varint(buf, pos)
            payload = buf[pos:pos + length]
            pos += length
        elif wire_type == 5:  # 32-bit
            payload = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported proto2 wire type {wire_type} at offset {start}")
        yield field_num, wire_type, buf[start:pos], payload


def _field_bytes(field_num: int, payload: bytes) -> bytes:
    return _encode_varint((field_num << 3) | 2) + _encode_varint(len(payload)) + payload


def _field_varint(field_num: int, val: int) -> bytes:
    return _encode_varint((field_num << 3) | 0) + _encode_varint(val)


def _field_float(field_num: int, val: float) -> bytes:
    return _encode_varint((field_num << 3) | 5) + struct.pack("<f", float(val))


def _encode_layer_int8(w_row_major_i8: np.ndarray) -> bytes:
    """Encode Weights::Layer with Encoding::INT8 (5) and raw int8 bytes."""
    assert w_row_major_i8.dtype == np.int8
    raw = np.ascontiguousarray(w_row_major_i8).tobytes()
    return _field_bytes(3, raw) + _field_varint(4, 5)  # params=3, encoding=4 (INT8=5)


def _encode_layer_f32(arr_f32: np.ndarray) -> bytes:
    """Encode Weights::Layer with Encoding::FLOAT32 (4) and raw little-endian float32 bytes."""
    raw = np.ascontiguousarray(arr_f32, dtype="<f4").tobytes()
    return _field_bytes(3, raw) + _field_varint(4, 4)  # params=3, encoding=4 (FLOAT32=4)


# ---------------- QDQ ONNX Extraction ----------------

def extract_encoder_int8_from_qdq_onnx(onnx_path: str) -> list:
    """Extract per-encoder INT8 weights (transposed to [M, K] row-major), scales, zps, and FBC biases."""
    m = onnx.load(onnx_path)
    init_map = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}
    prod_map = {out: node for node in m.graph.node for out in node.output}

    # Map MatMul output -> following bias Add initializer array
    bias_after = {}
    for node in m.graph.node:
        if node.op_type == "Add" and len(node.input) == 2 and node.input[1] in init_map:
            bias_after[node.input[0]] = init_map[node.input[1]].astype(np.float32)

    # Index quantized MatMul nodes by name
    q_matmuls = {}
    for node in m.graph.node:
        if node.op_type != "MatMul":
            continue
        p0 = prod_map.get(node.input[0])
        p1 = prod_map.get(node.input[1])
        if (p0 is not None and p0.op_type == "DequantizeLinear"
                and p1 is not None and p1.op_type == "DequantizeLinear"):
            a_scale = float(init_map[p0.input[1]])
            a_zp = int(init_map[p0.input[2]])
            w_i8_km = init_map[p1.input[0]].astype(np.int8)       # ONNX shape [K, M]
            w_scale = init_map[p1.input[1]].astype(np.float32)    # shape [M]
            w_zp = init_map[p1.input[2]]
            assert np.all(w_zp == 0), f"Non-zero weight zp in {node.name}"
            assert w_i8_km.ndim == 2 and w_scale.shape == (w_i8_km.shape[1],), (
                f"Shape mismatch on {node.name}: W={w_i8_km.shape}, sc={w_scale.shape}"
            )
            # Lc0 .pb.gz & SYCL store weights as [M=out, K=in] row-major (transpose of ONNX [K, M])!
            w_i8_mk = np.ascontiguousarray(w_i8_km.T, dtype=np.int8)
            bias = bias_after.get(node.output[0])
            assert bias is not None and bias.shape == (w_i8_mk.shape[0],), (
                f"Missing/mismatched bias after {node.name}"
            )
            q_matmuls[node.name] = {
                "w_i8": w_i8_mk,
                "w_scale": w_scale,
                "bias": bias,
                "a_scale": a_scale,
                "a_zp": a_zp,
            }

    # Group by encoder index 0, 1, ...
    enc_idx = 0
    encoders = []
    while f"/encoder{enc_idx}/mha/Q/w" in q_matmuls:
        pfx = f"/encoder{enc_idx}"
        q = q_matmuls[f"{pfx}/mha/Q/w"]
        k = q_matmuls[f"{pfx}/mha/K/w"]
        v = q_matmuls[f"{pfx}/mha/V/w"]
        d = q_matmuls[f"{pfx}/mha/out/dense/w"]
        f1 = q_matmuls[f"{pfx}/ffn/dense1/w"]
        f2 = q_matmuls[f"{pfx}/ffn/dense2/w"]

        # Q, K, V share the exact same input activation tensor -> scales & zero-points must match!
        assert abs(q["a_scale"] - k["a_scale"]) < 1e-12 and abs(q["a_scale"] - v["a_scale"]) < 1e-12, (
            f"Encoder {enc_idx}: Q/K/V activation scales differ!"
        )
        assert q["a_zp"] == k["a_zp"] == v["a_zp"], f"Encoder {enc_idx}: Q/K/V activation zps differ!"

        encoders.append({"q": q, "k": k, "v": v, "dense": d, "ffn1": f1, "ffn2": f2})
        enc_idx += 1

    if not encoders:
        raise RuntimeError(f"No quantized /encoder*/mha/Q/w layers found in {onnx_path}")
    return encoders


def extract_residual_int8_from_qdq_onnx(onnx_path: str) -> list:
    """Extract per-residual-block INT8 conv1 & conv2 weights ([O,I,3,3]), scales, zps, and FBC biases."""
    m = onnx.load(onnx_path)
    init_map = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}
    prod_map = {out: node for node in m.graph.node for out in node.output}

    q_convs = {}
    for node in m.graph.node:
        if node.op_type != "Conv":
            continue
        p0 = prod_map.get(node.input[0])
        p1 = prod_map.get(node.input[1])
        if (p0 is not None and p0.op_type == "DequantizeLinear"
                and p1 is not None and p1.op_type == "DequantizeLinear"):
            a_scale = float(init_map[p0.input[1]])
            a_zp_arr = init_map[p0.input[2]]
            assert a_zp_arr.dtype == np.uint8, f"Expected uint8 activation zp on {node.name}, got {a_zp_arr.dtype}"
            a_zp = int(a_zp_arr)
            w_i8 = np.ascontiguousarray(init_map[p1.input[0]], dtype=np.int8)   # [O, I, 3, 3] OIHW
            w_scale = np.ascontiguousarray(init_map[p1.input[1]], dtype=np.float32)
            w_zp = init_map[p1.input[2]]
            assert np.all(w_zp == 0), f"Non-zero weight zp in {node.name}"
            assert w_i8.ndim == 4 and w_i8.shape[2:] == (3, 3), f"Unexpected conv kernel shape on {node.name}: {w_i8.shape}"
            assert w_scale.shape == (w_i8.shape[0],), f"Scale mismatch on {node.name}: {w_scale.shape} vs {w_i8.shape}"
            assert len(node.input) >= 3 and node.input[2] in init_map, f"Missing folded bias on {node.name}"
            bias = np.ascontiguousarray(init_map[node.input[2]], dtype=np.float32)
            assert bias.shape == (w_i8.shape[0],), f"Bias mismatch on {node.name}: {bias.shape}"
            q_convs[node.name] = {
                "w_i8": w_i8,
                "w_scale": w_scale,
                "bias": bias,
                "a_scale": a_scale,
                "a_zp": a_zp,
            }

    blk_idx = 0
    blocks = []
    while f"/block{blk_idx}/conv1" in q_convs and f"/block{blk_idx}/conv2" in q_convs:
        blocks.append({
            "conv1": q_convs[f"/block{blk_idx}/conv1"],
            "conv2": q_convs[f"/block{blk_idx}/conv2"],
        })
        blk_idx += 1

    if not blocks:
        raise RuntimeError(f"No quantized /block*/conv{{1,2}} layers found in {onnx_path}")
    return blocks


def _encode_int8_conv_block(cq: dict) -> bytes:
    """Encode a BN-folded W8A8 INT8 Weights::ConvBlock (fields 1,2,7,8,9)."""
    out = bytearray()
    out.extend(_field_bytes(1, _encode_layer_int8(cq["w_i8"])))
    out.extend(_field_bytes(2, _encode_layer_f32(cq["bias"])))
    out.extend(_field_bytes(7, _encode_layer_f32(cq["w_scale"])))
    out.extend(_field_float(8, cq["a_scale"]))
    out.extend(_field_varint(9, cq["a_zp"]))
    return bytes(out)


def pack_residual_int8_pb_gz(src_pb_path: str, blocks_q: list, out_pb_path: str) -> None:
    """Splice INT8 Residual ConvBlock payloads into src_pb_path and write out_pb_path (.pb.gz)."""
    with gzip.open(src_pb_path, "rb") as f:
        net_bytes = f.read()

    out_net = bytearray()
    weights_payload = None
    for fnum, wtype, raw, payload in _iter_proto_fields(net_bytes):
        if fnum == 10 and wtype == 2:
            weights_payload = payload
        else:
            out_net.extend(raw)
    if weights_payload is None:
        raise RuntimeError(f"Source {src_pb_path} has no Weights field (10).")

    out_weights = bytearray()
    blk_seen = 0
    for fnum, wtype, raw, payload in _iter_proto_fields(weights_payload):
        if fnum != 2:  # Weights::residual = 2
            out_weights.extend(raw)
            continue

        if blk_seen >= len(blocks_q):
            raise RuntimeError(
                f"Source .pb.gz has more residual blocks than calibrated ONNX ({len(blocks_q)})"
            )
        bq = blocks_q[blk_seen]
        blk_seen += 1

        # Rewrite Residual: replace conv1 (1) & conv2 (2), preserve se (3) and any other fields verbatim.
        out_res = bytearray()
        for rfnum, rwtype, rraw, _ in _iter_proto_fields(payload):
            if rfnum not in (1, 2):
                out_res.extend(rraw)
        out_res.extend(_field_bytes(1, _encode_int8_conv_block(bq["conv1"])))
        out_res.extend(_field_bytes(2, _encode_int8_conv_block(bq["conv2"])))
        out_weights.extend(_field_bytes(2, bytes(out_res)))

    if blk_seen != len(blocks_q):
        raise RuntimeError(f"Residual block count mismatch: pb.gz had {blk_seen}, ONNX had {len(blocks_q)}")

    out_net.extend(_field_bytes(10, bytes(out_weights)))
    os.makedirs(os.path.dirname(os.path.abspath(out_pb_path)), exist_ok=True)
    with gzip.open(out_pb_path, "wb", compresslevel=6) as f:
        f.write(bytes(out_net))

    src_mb = os.path.getsize(src_pb_path) / 1e6
    dst_mb = os.path.getsize(out_pb_path) / 1e6
    print(f"Packed {blk_seen} INT8 residual blocks -> {out_pb_path}")
    print(f"  Size: {src_mb:.2f} MB (src .pb.gz) -> {dst_mb:.2f} MB (INT8 .pb.gz, saved {src_mb - dst_mb:.2f} MB)")


def pack_int8_pb_gz(src_pb_path: str, encoders_q: list, out_pb_path: str) -> None:
    """Splice INT8 encoder payloads into src_pb_path and write out_pb_path (.pb.gz)."""
    with gzip.open(src_pb_path, "rb") as f:
        net_bytes = f.read()

    # Parse top-level Net, locating Weights (field 10)
    out_net = bytearray()
    weights_payload = None
    for fnum, wtype, raw, payload in _iter_proto_fields(net_bytes):
        if fnum == 10 and wtype == 2:
            weights_payload = payload
        else:
            out_net.extend(raw)
    if weights_payload is None:
        raise RuntimeError(f"Source {src_pb_path} has no Weights field (10).")

    # Parse Weights, rewriting each repeated EncoderLayer (field 27) in order
    out_weights = bytearray()
    enc_seen = 0
    for fnum, wtype, raw, payload in _iter_proto_fields(weights_payload):
        if fnum != 27:
            out_weights.extend(raw)
            continue

        if enc_seen >= len(encoders_q):
            raise RuntimeError(
                f"Source .pb.gz has more encoders than calibrated ONNX ({len(encoders_q)})"
            )
        eq = encoders_q[enc_seen]
        enc_seen += 1

        # Rewrite EncoderLayer: keep ln1/ln2 (2,3,5,6) untouched; rewrite mha (1) & ffn (4)
        out_enc = bytearray()
        for efnum, ewtype, eraw, epayload in _iter_proto_fields(payload):
            if efnum == 1 and ewtype == 2:  # MHA
                # Keep smolgen (9) and any non-weight fields verbatim
                out_mha = bytearray()
                for mfnum, mwtype, mraw, mpayload in _iter_proto_fields(epayload):
                    if mfnum not in (1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20):
                        out_mha.extend(mraw)
                out_mha.extend(_field_bytes(1, _encode_layer_int8(eq["q"]["w_i8"])))
                out_mha.extend(_field_bytes(2, _encode_layer_f32(eq["q"]["bias"])))
                out_mha.extend(_field_bytes(3, _encode_layer_int8(eq["k"]["w_i8"])))
                out_mha.extend(_field_bytes(4, _encode_layer_f32(eq["k"]["bias"])))
                out_mha.extend(_field_bytes(5, _encode_layer_int8(eq["v"]["w_i8"])))
                out_mha.extend(_field_bytes(6, _encode_layer_f32(eq["v"]["bias"])))
                out_mha.extend(_field_bytes(7, _encode_layer_int8(eq["dense"]["w_i8"])))
                out_mha.extend(_field_bytes(8, _encode_layer_f32(eq["dense"]["bias"])))
                # Reserved INT8 slots 13..20
                out_mha.extend(_field_bytes(13, _encode_layer_f32(eq["q"]["w_scale"])))
                out_mha.extend(_field_bytes(14, _encode_layer_f32(eq["k"]["w_scale"])))
                out_mha.extend(_field_bytes(15, _encode_layer_f32(eq["v"]["w_scale"])))
                out_mha.extend(_field_bytes(16, _encode_layer_f32(eq["dense"]["w_scale"])))
                out_mha.extend(_field_float(17, eq["q"]["a_scale"]))
                out_mha.extend(_field_varint(18, eq["q"]["a_zp"]))
                out_mha.extend(_field_float(19, eq["dense"]["a_scale"]))
                out_mha.extend(_field_varint(20, eq["dense"]["a_zp"]))
                out_enc.extend(_field_bytes(1, bytes(out_mha)))
            elif efnum == 4 and ewtype == 2:  # FFN
                out_ffn = bytearray()
                out_ffn.extend(_field_bytes(1, _encode_layer_int8(eq["ffn1"]["w_i8"])))
                out_ffn.extend(_field_bytes(2, _encode_layer_f32(eq["ffn1"]["bias"])))
                out_ffn.extend(_field_bytes(3, _encode_layer_int8(eq["ffn2"]["w_i8"])))
                out_ffn.extend(_field_bytes(4, _encode_layer_f32(eq["ffn2"]["bias"])))
                # Reserved INT8 slots 5..10
                out_ffn.extend(_field_bytes(5, _encode_layer_f32(eq["ffn1"]["w_scale"])))
                out_ffn.extend(_field_bytes(6, _encode_layer_f32(eq["ffn2"]["w_scale"])))
                out_ffn.extend(_field_float(7, eq["ffn1"]["a_scale"]))
                out_ffn.extend(_field_varint(8, eq["ffn1"]["a_zp"]))
                out_ffn.extend(_field_float(9, eq["ffn2"]["a_scale"]))
                out_ffn.extend(_field_varint(10, eq["ffn2"]["a_zp"]))
                out_enc.extend(_field_bytes(4, bytes(out_ffn)))
            else:
                out_enc.extend(eraw)

        out_weights.extend(_field_bytes(27, bytes(out_enc)))

    if enc_seen != len(encoders_q):
        raise RuntimeError(f"Encoder count mismatch: pb.gz had {enc_seen}, ONNX had {len(encoders_q)}")

    out_net.extend(_field_bytes(10, bytes(out_weights)))
    os.makedirs(os.path.dirname(os.path.abspath(out_pb_path)), exist_ok=True)
    with gzip.open(out_pb_path, "wb", compresslevel=6) as f:
        f.write(bytes(out_net))

    src_mb = os.path.getsize(src_pb_path) / 1e6
    dst_mb = os.path.getsize(out_pb_path) / 1e6
    print(f"Packed {enc_seen} INT8 encoder blocks -> {out_pb_path}")
    print(f"  Size: {src_mb:.2f} MB (src .pb.gz) -> {dst_mb:.2f} MB (INT8 .pb.gz, saved {src_mb - dst_mb:.2f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-pb", required=True, help="Path to source FP16/FP32 Lc0 .pb.gz weights")
    ap.add_argument("--qdq-onnx", required=True, help="Path to calibrated W8A8 QDQ ONNX model")
    ap.add_argument("--out-pb", required=True, help="Output path for packed native INT8 .pb.gz weights")
    args = ap.parse_args()

    m_peek = onnx.load(args.qdq_onnx, load_external_data=False)
    node_names = {n.name for n in m_peek.graph.node}
    if "/encoder0/mha/Q/w" in node_names:
        encoders_q = extract_encoder_int8_from_qdq_onnx(args.qdq_onnx)
        print(f"Extracted {len(encoders_q)} calibrated W8A8 INT8 encoder blocks from {args.qdq_onnx}")
        for i, eq in enumerate(encoders_q):
            if i == 0 or i == len(encoders_q) - 1:
                print(
                    f"  enc[{i}]: Q={eq['q']['w_i8'].shape} qkv_act=({eq['q']['a_scale']:.5f}, zp={eq['q']['a_zp']}) | "
                    f"dense={eq['dense']['w_i8'].shape} ({eq['dense']['a_scale']:.5f}, zp={eq['dense']['a_zp']}) | "
                    f"ffn1={eq['ffn1']['w_i8'].shape} ({eq['ffn1']['a_scale']:.5f}, zp={eq['ffn1']['a_zp']}) | "
                    f"ffn2={eq['ffn2']['w_i8'].shape} ({eq['ffn2']['a_scale']:.5f}, zp={eq['ffn2']['a_zp']})"
                )
        pack_int8_pb_gz(args.src_pb, encoders_q, args.out_pb)
    elif "/block0/conv1" in node_names:
        blocks_q = extract_residual_int8_from_qdq_onnx(args.qdq_onnx)
        print(f"Extracted {len(blocks_q)} calibrated W8A8 INT8 residual blocks from {args.qdq_onnx}")
        for i, bq in enumerate(blocks_q):
            if i == 0 or i == len(blocks_q) - 1:
                print(
                    f"  blk[{i:2d}]: conv1={bq['conv1']['w_i8'].shape} act=({bq['conv1']['a_scale']:.5f}, zp={bq['conv1']['a_zp']}) | "
                    f"conv2={bq['conv2']['w_i8'].shape} act=({bq['conv2']['a_scale']:.5f}, zp={bq['conv2']['a_zp']})"
                )
        pack_residual_int8_pb_gz(args.src_pb, blocks_q, args.out_pb)
    else:
        raise RuntimeError(f"Unrecognized QDQ ONNX trunk architecture in {args.qdq_onnx}")


if __name__ == "__main__":
    main()
