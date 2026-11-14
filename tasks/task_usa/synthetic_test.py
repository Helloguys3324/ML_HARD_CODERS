#!/usr/bin/env python
"""Synthetic end-to-end test: for every embedded tokenizer, generate a fake
test-input.json (calibration pair + challenge message), run the solver and
check that it identifies and decodes correctly.

Tests both `add_special_tokens=True` and `False` encodings (where supported).
"""
import importlib.util
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "vocab", "raw")

CALIB = (
    "Riverside dispatch log - Denver & Sacramento field test\n"
    "def sample_count(names: list[str]) -> int:\n"
    "    \"\"\"How many volunteers signed up in Boise, Idaho?\"\"\"\n"
    "    return len(names)  # 17 people, 3 cities\n"
)


def load_main():
    spec = importlib.util.spec_from_file_location("main_solver", os.path.join(HERE, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SRC = {
    "gpt2": ("tiktoken", "gpt2"),
    "p50k_base": ("tiktoken", "p50k_base"),
    "cl100k_base": ("tiktoken", "cl100k_base"),
    "o200k_base": ("tiktoken", "o200k_base"),
    "bert-base-uncased": ("hf", "bert-base-uncased.json"),
    "roberta-base": ("hf", "roberta-base.json"),
    "t5-small": ("hf", "t5-small.json"),
    "llama-2-7b-hf": ("hf", "Llama-2-7b-hf-nous.json"),
    "mistral-7b-v0.1": ("hf", "Mistral-7B-v0.1.json"),
    "albert-base-v2": ("hf", "albert-base-v2.json"),
    "xlnet-base-cased": ("hf", "xlnet-base-cased.json"),
    "bert-base-cased": ("hf", "bert-base-cased.json"),
    "bert-base-multilingual-cased": ("hf", "bert-base-multilingual-cased.json"),
    "qwen2.5-0.5b": ("hf", "Qwen2.5-0.5B.json"),
    "llama-3-8b": ("hf", "llama-3-8b.json"),
    "phi-3-mini-4k-instruct": ("hf", "phi-3-mini.json"),
    "smollm2-135m": ("hf", "SmolLM2-135M.json"),
    "falcon-7b": ("hf", "falcon-7b.json"),
    "starcoder2-3b": ("hf", "starcoder2-3b.json"),
    "codegen-350m-mono": ("hf", "codegen-350M-mono.json"),
    "deberta-v3-small": ("sp", "deberta-v3-small.model"),
    "xlm-roberta-base": ("hf", "xlm-roberta-base.json"),
}


def get_ref(kind, name):
    if kind == "tiktoken":
        import tiktoken
        return tiktoken.get_encoding(name)
    if kind == "hf":
        from tokenizers import Tokenizer
        return Tokenizer.from_file(os.path.join(RAW, name))
    import sentencepiece as spm
    return spm.SentencePieceProcessor(model_file=os.path.join(RAW, name))


def encode(kind, ref, text, specials):
    if kind == "tiktoken":
        return ref.encode(text)
    if kind == "hf":
        return ref.encode(text, add_special_tokens=specials).ids
    return ref.encode(text, add_bos=specials, add_eos=specials)


def decode(kind, ref, ids, specials):
    if kind == "tiktoken":
        return ref.decode(ids)
    if kind == "hf":
        return ref.decode(ids, skip_special_tokens=not specials)
    return ref.decode(ids)


def main():
    mod = load_main()
    failures = 0
    results = []
    for canonical, (kind, name) in SRC.items():
        ref = get_ref(kind, name)
        for specials in (False, True):
            msg = ("TEST_SAMPLE: Verification successful at the American NLP "
                   "Observatory. Token IDs map to %s vocabulary!" % canonical)
            calib_ids = encode(kind, ref, CALIB, specials)
            chal_ids = encode(kind, ref, msg, specials)
            # what the generator would store as decoded message
            expected = decode(kind, ref, chal_ids, specials)
            payload = {
                "calibration": {"text": CALIB, "token_ids": calib_ids},
                "challenge": {"token_ids": chal_ids},
            }
            with tempfile.TemporaryDirectory() as td:
                ip = os.path.join(td, "test-input.json")
                json.dump(payload, open(ip, "w"))
                result, best, allres = mod.solve(ip)
            ok_id = result["tokenizer_id"].lower() == canonical.lower()
            ok_msg = result["decoded_message"] == expected
            # allow trailing whitespace normalisation
            ok_msg = ok_msg or result["decoded_message"].rstrip() == expected.rstrip()
            status = "OK " if (ok_id and ok_msg) else "FAIL"
            if not (ok_id and ok_msg):
                failures += 1
            results.append((canonical, specials, status, result["tokenizer_id"],
                            best["sim"], result["decoded_message"][:60],
                            expected[:60]))
    for r in results:
        print("%-30s specs=%-5s %s id=%-28s sim=%.4f\n    got=%r\n    exp=%r" % r)
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
