#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


def find_input_file():
    candidates = [
        Path("test-input.json"),
        Path("train-input.json"),
        Path("data/test-input.json"),
        Path("data/train-input.json"),
        Path("challenges/task_usa/data/test-input.json"),
        Path("challenges/task_usa/data/train-input.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find test-input.json or train-input.json")


def solve():
    input_path = find_input_file()
    print(f"Reading input from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    calib_text = data["calibration"]["text"]
    calib_tokens = data["calibration"]["token_ids"]
    challenge_tokens = data["challenge"]["token_ids"]

    text_variants = [calib_text]
    if "\r\n" in calib_text:
        text_variants.append(calib_text.replace("\r\n", "\n"))
    elif "\n" in calib_text:
        text_variants.append(calib_text.replace("\n", "\r\n"))

    identified_tokenizer = None
    decoded_message = None

    # 1. Tiktoken encodings
    try:
        import tiktoken

        tiktoken_encodings = [
            "cl100k_base",
            "o200k_base",
            "p50k_base",
            "r50k_base",
            "p50k_edit",
            "gpt2",
            "o200k_harmony",
        ]
        for name in tiktoken_encodings:
            try:
                enc = tiktoken.get_encoding(name)
                for t in text_variants:
                    encoded = enc.encode(t, allowed_special="all")
                    if encoded == calib_tokens:
                        identified_tokenizer = name
                        decoded_message = enc.decode(challenge_tokens)
                        print(f"Matched tiktoken encoding: {name}")
                        break
                if identified_tokenizer:
                    break
            except Exception as e:
                pass
    except ImportError:
        print("tiktoken not installed")

    # 2. Hugging Face tokenizers
    if not identified_tokenizer:
        try:
            from transformers import AutoTokenizer

            hf_candidates = [
                "gpt2",
                "bert-base-uncased",
                "bert-base-cased",
                "roberta-base",
                "xlm-roberta-base",
                "t5-base",
                "google/electra-base-discriminator",
                "facebook/bart-base",
                "google-bert/bert-base-uncased",
                "google-bert/bert-base-cased",
                "distilbert/distilbert-base-uncased",
                "distilbert-base-uncased",
                "distilbert/distilbert-base-cased",
                "distilbert-base-cased",
                "albert-base-v2",
                "openai-community/gpt2",
                "meta-llama/Llama-2-7b-hf",
                "google/t5-v1_1-base",
                "facebook/bart-large",
            ]

            for model_id in hf_candidates:
                tok = None
                try:
                    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                except Exception:
                    try:
                        tok = AutoTokenizer.from_pretrained(model_id)
                    except Exception:
                        continue

                for t in text_variants:
                    for add_special in [False, True]:
                        try:
                            encoded = tok.encode(t, add_special_tokens=add_special)
                            if encoded == calib_tokens:
                                identified_tokenizer = model_id
                                decoded = tok.decode(challenge_tokens, skip_special_tokens=True)
                                if not decoded:
                                    decoded = tok.decode(challenge_tokens, skip_special_tokens=False)
                                decoded_message = decoded
                                print(f"Matched HF tokenizer: {model_id} (add_special={add_special})")
                                break
                        except Exception:
                            pass
                    if identified_tokenizer:
                        break
                if identified_tokenizer:
                    break
        except ImportError:
            print("transformers not installed")

    if not identified_tokenizer or decoded_message is None:
        raise RuntimeError("No matching tokenizer found for calibration text!")

    result = {
        "tokenizer_id": identified_tokenizer,
        "decoded_message": decoded_message,
    }

    output_path = Path("test-output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Successfully wrote {output_path}:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    solve()
