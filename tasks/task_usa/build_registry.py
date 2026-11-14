#!/usr/bin/env python
"""Build embedded tokenizer vocabulary registry for main.py.

Collects vocabularies from:
  * tiktoken encodings (downloaded/cached locally)
  * HuggingFace `tokenizers`/`transformers` tokenizer.json files
  * SentencePiece .model files (parsed with a minimal protobuf reader)

and serialises them as zlib+base85 blobs inside a single self-contained main.py.
"""
import base64
import glob
import json
import os
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "vocab", "raw")


# ----------------------------------------------------------------------------
# serialisation helpers
# ----------------------------------------------------------------------------
def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def pack_pieces(pieces):
    """pieces: dict id -> str. Returns compressed+base85 payload."""
    n = max(pieces) + 1
    blob = bytearray()
    prev = -1
    for i in range(n):
        if i not in pieces:
            continue
        gap = i - prev - 1
        prev = i
        raw = pieces[i]
        piece = raw if isinstance(raw, bytes) else raw.encode("utf-8")
        blob += varint(gap)
        blob += varint(len(piece))
        blob += piece
    comp = zlib.compress(bytes(blob), 9)
    return base64.b85encode(comp).decode("ascii")


# ----------------------------------------------------------------------------
# SentencePiece .model protobuf (minimal parser)
# ----------------------------------------------------------------------------
def _read_varint(buf, i):
    shift = 0
    val = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7


def parse_sp_model(path):
    """Return dict id -> piece string for a SentencePiece .model file."""
    buf = open(path, "rb").read()
    pieces = []
    i = 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            _, i = _read_varint(buf, i)
        elif wire == 1:
            i += 8
        elif wire == 2:
            ln, i = _read_varint(buf, i)
            data = buf[i:i + ln]
            i += ln
            if field == 1:  # ModelProto.pieces
                piece = _parse_sp_piece(data)
                if piece is not None:
                    pieces.append(piece)
        elif wire == 5:
            i += 4
        else:
            raise ValueError("bad wire type %d" % wire)
    return {idx: p for idx, p in enumerate(pieces) if p is not None}


def _parse_sp_piece(data):
    piece = None
    ptype = 1
    i = 0
    while i < len(data):
        key, i = _read_varint(data, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _read_varint(data, i)
            if field == 3:
                ptype = val
        elif wire == 2:
            ln, i = _read_varint(data, i)
            val = data[i:i + ln]
            i += ln
            if field == 1:
                piece = val.decode("utf-8", "replace")
        elif wire == 5:
            i += 4
        elif wire == 1:
            i += 8
        else:
            raise ValueError("bad wire type %d" % wire)
    if piece is None:
        return None
    return piece, ptype


# ----------------------------------------------------------------------------
# HF tokenizer.json
# ----------------------------------------------------------------------------
def classifier_for_decoder(dec):
    def dtype(d):
        if not isinstance(d, dict):
            return None
        return d.get("type")

    if not isinstance(dec, dict):
        return "plain"
    t = dtype(dec)
    if t == "ByteLevel":
        return "bytelevel"
    if t == "Metaspace":
        return "metaspace"
    if t == "WordPiece":
        return "wordpiece"
    if t == "Sequence":
        types = [dtype(x) for x in dec.get("decoders", [])]
        if "ByteLevel" in types:
            return "bytelevel"
        if "WordPiece" in types:
            return "wordpiece"
        if "Metaspace" in types or "Replace" in types:
            return "metaspace"
        if "ByteFallback" in types:
            return "metaspace"
    return "plain"


def load_hf_json(path):
    d = json.load(open(path, encoding="utf-8"))
    model = d.get("model") or {}
    vocab = model.get("vocab")
    pieces = {}
    if isinstance(vocab, list):
        for i, p in enumerate(vocab):
            pieces[i] = p[0]
    elif isinstance(vocab, dict):
        for tok, i in vocab.items():
            pieces[i] = tok
    for a in d.get("added_tokens", []):
        pieces[a["id"]] = a["content"]
    return pieces, classifier_for_decoder(d.get("decoder"))


# ----------------------------------------------------------------------------
# tiktoken
# ----------------------------------------------------------------------------
def load_tiktoken():
    import tiktoken
    out = {}
    for name in ["gpt2", "r50k_base", "p50k_base", "p50k_edit",
                 "cl100k_base", "o200k_base"]:
        enc = tiktoken.get_encoding(name)
        pieces = {}
        for tok_bytes, rank in enc._mergeable_ranks.items():
            pieces[rank] = tok_bytes
        for tok, idx in enc._special_tokens.items():
            pieces[idx] = tok
        mode = "raw"
        out[name] = (pieces, mode)
    return out


def main():
    registry = []  # (blob_key, mode, pieces)

    # --- tiktoken ---
    for name, (pieces, mode) in load_tiktoken().items():
        registry.append((name, mode, pieces))

    # --- HF tokenizer.json files ---
    skip = {"deberta-v3-small.json", "mt5-small.json", "opt-125m.json"}
    for path in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        base = os.path.basename(path)
        if base in skip:
            continue
        try:
            pieces, mode = load_hf_json(path)
        except Exception as e:  # noqa
            print("skip", base, e, file=sys.stderr)
            continue
        key = base[:-5]
        registry.append((key, mode, pieces))

    # --- SentencePiece .model files ---
    for path in sorted(glob.glob(os.path.join(RAW, "*.model"))):
        base = os.path.basename(path)
        key = base.rsplit(".", 1)[0]
        pieces = parse_sp_model(path)
        registry.append((key, "metaspace", pieces))

    lines = []
    for key, mode, pieces in registry:
        payload = pack_pieces(pieces)
        print("%-45s %-10s n=%7d b85=%7.2fMB" % (
            key, mode, max(pieces) + 1, len(payload) / 1e6), file=sys.stderr)
        lines.append("%s|%s|%s" % (key, mode, payload))

    data_path = os.path.join(HERE, "vocab_data.txt")
    with open(data_path, "w") as f:
        f.write("\n".join(lines))
    print("wrote", data_path, os.path.getsize(data_path) / 1e6, "MB",
          file=sys.stderr)


if __name__ == "__main__":
    main()
