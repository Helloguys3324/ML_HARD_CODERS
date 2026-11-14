#!/usr/bin/env python
"""Build the self-contained main.py for task_usa (tokenizer mystery)."""
import base64
import glob
import hashlib
import json
import lzma
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "vocab", "raw")

sys.path.insert(0, HERE)
import build_registry as B  # noqa: E402  (reuse loaders)


# ---------------------------------------------------------------------------
# candidate list, in tie-break priority order
#   (source_kind, source_name, canonical_id, aliases)
# ---------------------------------------------------------------------------
CANDIDATES = [
    ("tiktoken", "gpt2", "gpt2",
     ["gpt-2", "r50k_base", "r50k", "gpt2-medium", "gpt2-large", "gpt2-xl",
      "distilgpt2", "gpt-neo-125m", "gpt-neo-1.3b", "gpt-neo-2.7b", "gpt-j-6b",
      "opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b"]),
    ("tiktoken", "p50k_base", "p50k_base",
     ["p50k", "p50k-edit", "p50k_edit", "text-davinci-002", "text-davinci-003",
      "code-davinci-002", "davinci", "curie", "babbage", "ada"]),
    ("tiktoken", "cl100k_base", "cl100k_base",
     ["cl100k", "gpt-3.5-turbo", "gpt-4", "gpt-4-32k", "gpt-4-turbo",
      "text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]),
    ("tiktoken", "o200k_base", "o200k_base",
     ["o200k", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "o1", "o1-mini", "o3", "o3-mini"]),
    ("tiktoken", "r50k_base", "r50k_base", []),  # dedupes into gpt2

    ("hf", "bert-base-uncased.json", "bert-base-uncased",
     ["bert", "bert-uncased", "distilbert-base-uncased", "distilbert",
      "electra-small-discriminator", "electra-base-discriminator",
      "all-minilm-l6-v2", "sentence-transformers/all-minilm-l6-v2",
      "bert-tiny", "bert-mini", "bert-small", "bert-medium",
      "mobilebert-uncased", "prajjwal1/bert-tiny", "google/electra-small-discriminator"]),
    ("hf", "roberta-base.json", "roberta-base",
     ["roberta", "roberta-large", "facebookai/roberta-base", "bart-base",
      "bart-large", "longformer-base-4096", "allenai/longformer-base-4096"]),
    ("hf", "t5-small.json", "t5-small",
     ["t5", "t5-base", "t5-large", "google-t5/t5-small", "google/t5-small",
      "flan-t5-small", "flan-t5-base", "flan-t5-large", "google/flan-t5-small",
      "google/flan-t5-base"]),
    ("hf", "Llama-2-7b-hf-nous.json", "llama-2-7b-hf",
     ["llama", "llama-2", "llama2", "llama-2-7b", "llama-2-13b",
      "meta-llama/llama-2-7b-hf", "nousresearch/llama-2-7b-hf",
      "tinyllama", "tinyllama-1.1b-chat-v1.0",
      "tinyllama/tinyllama-1.1b-chat-v1.0", "unsloth/llama-2-7b"]),
    ("hf", "Mistral-7B-v0.1.json", "mistral-7b-v0.1",
     ["mistral", "mistralai/mistral-7b-v0.1", "mistral-7b",
      "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.2",
      "mistral-7b-instruct-v0.3"]),
    ("hf", "albert-base-v2.json", "albert-base-v2",
     ["albert", "albert-base-v1", "albert/albert-base-v2"]),
    ("hf", "xlnet-base-cased.json", "xlnet-base-cased",
     ["xlnet", "xlnet/xlnet-base-cased"]),
    ("hf", "bert-base-cased.json", "bert-base-cased", ["bert-cased"]),
    ("hf", "bert-base-multilingual-cased.json", "bert-base-multilingual-cased",
     ["bert-multilingual", "bert-base-multilingual",
      "distilbert-base-multilingual-cased"]),
    ("hf", "Qwen2.5-0.5B.json", "qwen2.5-0.5b",
     ["qwen", "qwen2.5", "qwen2", "qwen/qwen2.5-0.5b",
      "qwen2.5-0.5b-instruct", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]),
    ("hf", "llama-3-8b.json", "llama-3-8b",
     ["llama-3", "llama3", "meta-llama/meta-llama-3-8b",
      "meta-llama/llama-3-8b", "llama-3.1-8b", "llama-3.2-1b", "llama-3.2-3b"]),
    ("hf", "phi-3-mini.json", "phi-3-mini-4k-instruct",
     ["phi-3", "phi-3-mini", "microsoft/phi-3-mini-4k-instruct"]),
    ("hf", "SmolLM2-135M.json", "smollm2-135m",
     ["smollm2", "huggingfacetb/smollm2-135m", "smollm2-360m", "smollm2-1.7b"]),
    ("hf", "falcon-7b.json", "falcon-7b", ["falcon", "tiiuae/falcon-7b", "falcon-40b"]),
    ("hf", "starcoder2-3b.json", "starcoder2-3b",
     ["starcoder2", "bigcode/starcoder2-3b", "starcoder2-7b", "starcoder2-15b"]),
    ("hf", "codegen-350M-mono.json", "codegen-350m-mono",
     ["codegen", "salesforce/codegen-350m-mono", "codegen-350m-nl",
      "codegen-2b-mono", "codegen-6b-mono", "codegen-16b-mono",
      "phi-2", "microsoft/phi-2"]),
    ("hf", "gpt2.json", "gpt2", []),  # dedupes
    ("hf", "distilbert-base-uncased.json", "distilbert-base-uncased", []),
    ("hf", "electra-small-discriminator.json", "electra", []),
    ("hf", "all-MiniLM-L6-v2.json", "all-minilm-l6-v2", []),
    ("hf", "gpt2-medium.json", "gpt2-medium", []),
    ("hf", "gpt-neo-125M.json", "gpt-neo", []),
    ("hf", "t5-base.json", "t5-base", []),
    ("hf", "TinyLlama-1.1B-Chat-v1.0.json", "tinyllama", []),
    ("hf", "phi-2.json", "phi-2", []),
    ("sp", "deberta-v3-small.model", "deberta-v3-small",
     ["microsoft/deberta-v3-small", "deberta-v3"]),
    ("hf", "xlm-roberta-base.json", "xlm-roberta-base",
     ["xlm-roberta", "xlmr", "facebookai/xlm-roberta-base", "xlm-roberta-large",
      "xlm-roberta-base-sentencepiece"]),
]


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = bytes_to_unicode()


def bytelevel_to_raw(pieces):
    out = {}
    for i, p in pieces.items():
        if isinstance(p, bytes):
            out[i] = p
        elif all(ch in BYTE_DECODER for ch in p):
            out[i] = bytes(BYTE_DECODER[ch] for ch in p)
        else:
            out[i] = p.encode("utf-8")
    return out


def load_source(kind, name):
    if kind == "tiktoken":
        import tiktoken
        enc = tiktoken.get_encoding(name)
        pieces = {rank: tok for tok, rank in enc._mergeable_ranks.items()}
        for tok, idx in enc._special_tokens.items():
            pieces[idx] = tok.encode("utf-8")
        return pieces, "raw", set(enc._special_tokens.values())
    if kind == "hf":
        path = os.path.join(RAW, name)
        pieces, mode = B.load_hf_json(path)
        d = json.load(open(path, encoding="utf-8"))
        special = {a["id"] for a in d.get("added_tokens", []) if a.get("special")}
        if mode == "bytelevel":
            pieces = bytelevel_to_raw(pieces)
            mode = "raw"
        return pieces, mode, special
    if kind == "sp":
        raw = B.parse_sp_model(os.path.join(RAW, name))
        pieces = {i: p for i, (p, t) in raw.items()}
        special = {i for i, (p, t) in raw.items() if t not in (1, 6)}
        return pieces, "metaspace", special
    raise ValueError(kind)


def to_list(pieces):
    n = max(pieces) + 1
    return [pieces.get(i) for i in range(n)]


def main():
    blobs = []      # (key, mode, b85)
    by_hash = {}    # hash -> blob key
    entries = []    # (canonical, blob_key, aliases)

    for kind, name, canonical, aliases in CANDIDATES:
        try:
            pieces, mode, special = load_source(kind, name)
        except Exception as exc:  # noqa
            print("skip %s/%s: %s" % (kind, name, exc), file=sys.stderr)
            continue
        lst = to_list(pieces)
        raw = pickle.dumps(lst, protocol=4)
        h = hashlib.sha256(raw).hexdigest() + ":" + mode
        if h in by_hash:
            # merge aliases into the existing entry
            key = by_hash[h]
            for e in entries:
                if e[1] == key:
                    e[2].extend(a for a in [canonical] + aliases if a not in e[2])
                    break
            print("%-30s DUP -> %s" % (canonical, key), file=sys.stderr)
            continue
        payload = base64.b85encode(lzma.compress(raw, preset=9)).decode("ascii")
        by_hash[h] = canonical
        blobs.append((canonical, mode, payload))
        entries.append([canonical, canonical, list(aliases), sorted(special), len(lst)])
        print("%-30s %-10s n=%7d b85=%6.2fMB special=%d" %
              (canonical, mode, len(lst), len(payload) / 1e6, len(special)),
              file=sys.stderr)
        del lst

    data = "\n".join("%s|%s|%s" % (k, m, p) for k, m, p in blobs)
    total = sum(len(p) for _, _, p in blobs)
    print("unique blobs=%d  total b85=%.2f MB" % (len(blobs), total / 1e6), file=sys.stderr)

    template = open(os.path.join(HERE, "main_template.py"), encoding="utf-8").read()
    registry_repr = json.dumps(entries, ensure_ascii=False)
    out = template.replace("__VOCAB_DATA__", data).replace("__REGISTRY__", registry_repr)
    out_path = os.path.join(HERE, "main.py")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print("wrote %s (%.2f MB)" % (out_path, os.path.getsize(out_path) / 1e6),
          file=sys.stderr)


if __name__ == "__main__":
    main()
