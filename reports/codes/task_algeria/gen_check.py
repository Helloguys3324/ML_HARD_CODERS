"""
Quick generalization spot-check: generate a register reply for arbitrary prompts with the
base GGUF + a trained adapter, and print the reply, its conformance and its plain reading.
Run after solve.py produces adapter.gguf.
"""
from __future__ import annotations

import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
DATA = TASK / "data"
sys.path.insert(0, str(DATA))

import style_checker  # noqa: E402

BASE_MODEL = DATA / "models" / "smollm2-360m-instruct-q8_0.gguf"
SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def build_prompt(user):
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


def main():
    adapter = sys.argv[1] if len(sys.argv) > 1 else str(TASK / "adapter.gguf")
    import llama_cpp
    llm = llama_cpp.Llama(model_path=str(BASE_MODEL), n_ctx=1024, n_threads=6,
                          verbose=False, seed=0, lora_path=adapter)
    prompts = [
        "What is the capital of Germany?",
        "How many legs does a spider have?",
        "Who wrote the Odyssey?",
        "Which language is spoken in Japan?",
        "What is the longest river in South America?",
        "In which year did the Titanic sink? Answer with one noun.",
        "Which sea lies north of Africa? Answer with one noun.",
        "Why do camels live in the desert?",
        "What is cheese made from?",
        "Which planet is farthest from the Sun?",
    ]
    if len(sys.argv) > 2:
        prompts = [sys.argv[2]]
    for p in prompts:
        out = llm.create_completion(build_prompt(p), max_tokens=96,
                                    temperature=0.0, stop=["<|im_end|>"])
        rep = out["choices"][0]["text"].strip()
        res = style_checker.check_reply(rep)
        print(f"Q: {p}")
        print(f"  reply : {rep!r}")
        print(f"  plain : {style_checker.deregister(rep)!r}")
        print(f"  conf  : {res['conformance']:.2f}  viol={[s['violations'] for s in res['sentences']]}")
        print()


if __name__ == "__main__":
    main()