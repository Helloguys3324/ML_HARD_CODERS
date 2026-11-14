#!/usr/bin/env python
"""Hidden-test simulation for task_usa.

Builds test-input.json payloads in the documented format, runs the solver and
checks tokenizer_id + decoded_message against the reference library output.

Covers:
  * all tiktoken + HF tokenizer vocabularies embedded in main.py
  * calibration text present (the documented case)
  * calibration text empty / calibration object missing (defensive cases)
  * special tokens wrapped around the challenge ids
  * the specific cl100k regression: ordinary ids > 50256 must survive
  * a real subprocess run that reads test-input.json from cwd

SentencePiece is optional (the module is not installed here); its two
vocabularies are checked only if sentencepiece is importable.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "vocab", "raw")
MAIN = os.path.join(HERE, "main.py")

CALIB = (
    "# Brooklyn NLP Lab - New York & Chicago Hidden Calibration\n"
    "def hidden_sampler(city: str = \"Chicago\"):\n"
    "    \"\"\"Calibration probe for the hidden tokenizer split.\"\"\"\n"
    "    temps = [-7, 12, 31]  # Fahrenheit\n"
    "    return f\"{city}: {len(temps)} hidden samples recorded.\"\n"
)

MESSAGE = (
    "TEST_SAMPLE: Verification successful at the American NLP Observatory. "
    "Token IDs map to %s vocabulary!"
)

TIKTOKEN = ["gpt2", "p50k_base", "cl100k_base", "o200k_base"]

HF = {
    "bert-base-uncased": "bert-base-uncased.json",
    "roberta-base": "roberta-base.json",
    "t5-small": "t5-small.json",
    "llama-2-7b-hf": "Llama-2-7b-hf-nous.json",
    "mistral-7b-v0.1": "Mistral-7B-v0.1.json",
    "albert-base-v2": "albert-base-v2.json",
    "xlnet-base-cased": "xlnet-base-cased.json",
    "bert-base-cased": "bert-base-cased.json",
    "bert-base-multilingual-cased": "bert-base-multilingual-cased.json",
    "qwen2.5-0.5b": "Qwen2.5-0.5B.json",
    "llama-3-8b": "llama-3-8b.json",
    "phi-3-mini-4k-instruct": "phi-3-mini.json",
    "smollm2-135m": "SmolLM2-135M.json",
    "falcon-7b": "falcon-7b.json",
    "starcoder2-3b": "starcoder2-3b.json",
    "codegen-350m-mono": "codegen-350M-mono.json",
    "xlm-roberta-base": "xlm-roberta-base.json",
}

SP = {
    "deberta-v3-small": "deberta-v3-small.model",
}


def load_main():
    spec = importlib.util.spec_from_file_location("main_solver", MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def decode_clean(kind, ref, ids):
    if kind == "tiktoken":
        return ref.decode(ids)
    if kind == "hf":
        return ref.decode(ids, skip_special_tokens=True)
    return ref.decode(ids)


def run_solve(mod, payload):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test-input.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        result, best, results, reason = mod.solve(path)
    return result, reason


def check(mod, label, payload, expected, failures):
    result, reason = run_solve(mod, payload)
    ok_id = result["tokenizer_id"].lower() == expected["tokenizer_id"].lower()
    ok_msg = result["decoded_message"].rstrip() == expected["decoded_message"].rstrip()
    if not (ok_id and ok_msg):
        failures.append(label)
        print("FAIL %-52s reason=%-30s\n  got id=%r msg=%r\n  exp id=%r msg=%r" % (
            label, reason, result["tokenizer_id"], result["decoded_message"][:100],
            expected["tokenizer_id"], expected["decoded_message"][:100]))
    return ok_id and ok_msg


def main():
    mod = load_main()
    failures = []
    passed = 0

    refs = []
    for name in TIKTOKEN:
        refs.append((name, "tiktoken", name))
    for canonical, filename in HF.items():
        refs.append((canonical, "hf", filename))
    try:
        import sentencepiece  # noqa: F401
        for canonical, filename in SP.items():
            refs.append((canonical, "sp", filename))
    except Exception:
        pass

    for canonical, kind, name in refs:
        ref = get_ref(kind, name)
        msg = MESSAGE % canonical
        calib_ids = encode(kind, ref, CALIB, False)
        chal_ids = encode(kind, ref, msg, False)
        # The ground truth is what the reference library decodes the ids to
        # (case is unrecoverable for uncased tokenizers, WordPiece inserts
        # spaces around punctuation, ...).
        expected = {"tokenizer_id": canonical,
                    "decoded_message": decode_clean(kind, ref, chal_ids)}

        normal = {"calibration": {"text": CALIB, "token_ids": calib_ids},
                  "challenge": {"token_ids": chal_ids}}
        passed += check(mod, canonical + " / calibration present", normal, expected, failures)

        empty = {"calibration": {"text": "", "token_ids": calib_ids},
                 "challenge": {"token_ids": chal_ids}}
        passed += check(mod, canonical + " / empty calibration text", empty, expected, failures)

        missing = {"challenge": {"token_ids": chal_ids}}
        passed += check(mod, canonical + " / missing calibration", missing, expected, failures)

        # special-token wrapped challenge, expected plain text
        with_special = {"calibration": {"text": CALIB, "token_ids": calib_ids},
                        "challenge": {"token_ids": encode(kind, ref, msg, True)}}
        passed += check(mod, canonical + " / special tokens around message",
                        with_special, expected, failures)

    # ------------------------------------------------------------------
    # cl100k-specific regression: a message whose ids exceed 50256
    # ------------------------------------------------------------------
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tricky = (
        "TEST_SAMPLE: Verification successful at the American NLP Observatory. "
        "The city observatory verified 56573 labels; Token IDs map to cl100k_base vocabulary!"
    )
    tricky_ids = enc.encode(tricky)
    assert any(i > 50256 for i in tricky_ids), "regression case must contain ids > 50256"
    payload = {"calibration": {"text": CALIB, "token_ids": enc.encode(CALIB)},
               "challenge": {"token_ids": tricky_ids}}
    expected = {"tokenizer_id": "cl100k_base", "decoded_message": tricky}
    passed += check(mod, "cl100k / ordinary ids > 50256 survive", payload, expected, failures)

    # Verify the old buggy filter would have failed this case (document it).
    filtered = enc.decode([i for i in tricky_ids if i < 50256])
    assert filtered != tricky, "expected the >=50256 filter to garble the message"

    # multiline + unicode challenge message
    multiline = "TEST_SAMPLE: line one.\nLine two — café, naïve, résumé.\nToken IDs map to cl100k_base vocabulary!"
    payload = {"calibration": {"text": CALIB, "token_ids": enc.encode(CALIB)},
               "challenge": {"token_ids": enc.encode(multiline)}}
    expected = {"tokenizer_id": "cl100k_base", "decoded_message": multiline}
    passed += check(mod, "cl100k / multiline + unicode", payload, expected, failures)

    # ------------------------------------------------------------------
    # true subprocess run (documented behaviour: read test-input.json in cwd,
    # write test-output.json in cwd)
    # ------------------------------------------------------------------
    hidden = {
        "description": "hidden test simulation",
        "calibration": {"text": CALIB, "token_ids": enc.encode(CALIB),
                        "token_count": len(enc.encode(CALIB))},
        "challenge": {"token_ids": enc.encode(MESSAGE % "cl100k_base"),
                      "token_count": len(enc.encode(MESSAGE % "cl100k_base"))},
    }
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "test-input.json"), "w", encoding="utf-8") as fh:
            json.dump(hidden, fh)
        proc = subprocess.run([sys.executable, MAIN], cwd=td,
                              capture_output=True, text=True)
        out_path = os.path.join(td, "test-output.json")
        ok = proc.returncode == 0 and os.path.isfile(out_path)
        if ok:
            with open(out_path, encoding="utf-8") as fh:
                out = json.load(fh)
            ok = (out["tokenizer_id"].lower() == "cl100k_base"
                  and out["decoded_message"] == MESSAGE % "cl100k_base")
        if ok:
            passed += 1
        else:
            failures.append("subprocess / cwd contract")
            print("FAIL subprocess run: rc=%s out=%r stderr=%s" % (
                proc.returncode, out if 'out' in dir() else None, proc.stderr[-500:]))

    print("\nPASSED %d checks; FAILURES: %d" % (passed, len(failures)))
    if failures:
        for f in failures:
            print(" -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
