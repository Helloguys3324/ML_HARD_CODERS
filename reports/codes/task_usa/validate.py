#!/usr/bin/env python
"""Validate the pure-python decoders embedded in main.py against the real
libraries (tiktoken / HF tokenizers / sentencepiece)."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "vocab", "raw")


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

TEXTS = [
    "Hello world! This is a test, with punctuation: commas, periods. And 'quotes'?",
    "The quick brown fox jumps over the lazy dog.\nNew line here.",
    "El rápido zorro marrón salta sobre el perro perezoso. ¡Hola, mundo!",
    "def train_sampler(city: str = \"Austin\"):\n    return f\"{city}: 42 samples\"\n",
    "Token IDs map to cl100k_base vocabulary!",
    "I don't think it's working — can't you see? We'll fix it.",
    "  leading and trailing spaces  ",
    "Numbers 1234567890 and symbols @#$%^&*()_+-=[]{}|;:',.<>/?",
    "Emoji test 🚀🔥 and accents: café, naïve, résumé",
]


def get_ref(kind, name):
    if kind == "tiktoken":
        import tiktoken
        return tiktoken.get_encoding(name)
    if kind == "hf":
        from tokenizers import Tokenizer
        return Tokenizer.from_file(os.path.join(RAW, name))
    if kind == "sp":
        import sentencepiece as spm
        return spm.SentencePieceProcessor(model_file=os.path.join(RAW, name))
    raise ValueError(kind)


def ref_decode(kind, ref, ids):
    if kind == "tiktoken":
        return ref.decode(ids)
    if kind == "hf":
        return ref.decode(ids, skip_special_tokens=False)
    return ref.decode(ids)


def ref_encode(kind, ref, text):
    if kind == "tiktoken":
        return ref.encode(text)
    if kind == "hf":
        return ref.encode(text).ids
    return ref.encode(text)


def main():
    mod = load_main()
    data = json.load(open(os.path.join(HERE, "vocab", "..", "..", "..", "tasks",
                                       "task_usa", "data", "train-input.json")))
    calib_ids = data["calibration"]["token_ids"]
    chal_ids = data["challenge"]["token_ids"]

    ok = True
    for entry in mod._REGISTRY:
        canonical, blob_key, aliases = entry[0], entry[1], entry[2]
        kind, name = SRC[canonical]
        mode, pieces = mod._load_blob(blob_key)
        ref = get_ref(kind, name)

        seqs = {
            "calibration_ids": calib_ids,
            "challenge_ids": chal_ids,
        }
        for i, t in enumerate(TEXTS):
            try:
                seqs["text%d" % i] = ref_encode(kind, ref, t)
            except Exception as exc:  # noqa
                print("encode fail", canonical, i, exc)
                continue

        mism = 0
        if kind == "tiktoken":
            vsize = ref.n_vocab
        elif kind == "hf":
            vsize = ref.get_vocab_size()
        else:
            vsize = ref.get_piece_size()
        for label, ids in seqs.items():
            if any((not isinstance(i, int)) or i < 0 or i >= vsize for i in ids):
                continue
            ours = mod.decode_ids(mode, pieces, ids)
            theirs = ref_decode(kind, ref, ids)
            if ours != theirs:
                mism += 1
                if mism <= 2:
                    print("MISMATCH %-30s %-16s\n  ours  =%r\n  theirs=%r" %
                          (canonical, label, ours, theirs))
        print("%-30s %-9s seqs=%d mismatches=%d" % (canonical, mode, len(seqs), mism))
        ok = ok and mism == 0

    print("ALL OK" if ok else "MISMATCHES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
