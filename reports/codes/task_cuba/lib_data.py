"""Shared data loading + eval helpers for task_cuba."""
import json, os, re, unicodedata
from pathlib import Path

DATA = Path("/home/kozak/mllm/MML/work/tasks/task_cuba/data")
CORPUS = DATA / "data" / "corpus"


def load_meta():
    return [json.loads(l) for l in open(DATA / "data/corpus_meta.jsonl", encoding="utf-8")]


def load_corpus():
    meta = load_meta()
    docs = {}
    for m in meta:
        p = CORPUS / f"{m['id']}.txt"
        docs[m["id"]] = p.read_text(encoding="utf-8", errors="replace")
    return meta, docs


def load_qs():
    qs = [json.loads(l) for l in open(DATA / "data/sample-input.jsonl", encoding="utf-8")]
    gold = {}
    for l in open(DATA / "data/sample-output.jsonl", encoding="utf-8"):
        r = json.loads(l)
        gold[r["qid"]] = r["doc_id"]
    return qs, gold


def hit_at_k(ranked, gold, k=3):
    return sum(1 for qid, g in gold.items() if g in ranked.get(qid, [])[:k]) / max(1, len(gold))


def evaluate(fn, qs, gold, k=3, verbose=False):
    ranked = {}
    for q in qs:
        ranked[q["qid"]] = fn(q["question"])
    score = hit_at_k(ranked, gold, k)
    if verbose:
        misses = [(q["qid"], q["question"], gold[q["qid"]], ranked[q["qid"]][:3])
                  for q in qs if gold[q["qid"]] not in ranked[q["qid"]][:k]]
        for m in misses[:20]:
            print("MISS", m[0], "|", m[1][:100], "| gold:", m[2], "| got:", m[3])
        print("n misses:", len(misses))
    return score
