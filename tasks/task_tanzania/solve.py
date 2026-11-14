"""Solver for task_tanzania (Please, You Are So Delusional!) - LMML 2026.

Repairs the delusional starter retrieval pipeline:
1. Replaces the mismatched dual-encoder setup with rank_bm25.
2. Removes fixed character chunking that fragmented exhibit titles and logistical info.
3. Incorporates multi-field weighting and query-intent constraints to filter out
   adversarial distractor notices (e.g. archived background bulletins and operations logs).
4. Generates retrieved_ids.json, derives and verifies the flag via flag_utils.py.
5. Submits the flag via SigmoidClient.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Add data folder to path for flag_utils
BASE_DIR = Path("/home/kozak/mllm/challenges/task_tanzania")
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, "/home/kozak/mllm")

from rank_bm25 import BM25Okapi
from flag_utils import derive_flag, load_verification_codes


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str


def load_corpus(corpus_path: Path) -> list[Document]:
    documents: list[Document] = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            documents.append(Document(row["document_id"], row["title"], row["text"]))
    return documents


def load_questions(questions_path: Path) -> list[dict[str, str]]:
    with questions_path.open(encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def retrieve_documents(
    documents: list[Document],
    questions: list[dict[str, str]],
) -> list[dict[str, str]]:
    tokenized_corpus = []
    for doc in documents:
        title_tokens = tokenize(doc.title)
        body_tokens = tokenize(doc.text)
        tokenized_corpus.append(title_tokens * 6 + body_tokens)

    bm25 = BM25Okapi(tokenized_corpus)

    cities = ["dar es salaam", "arusha", "mwanza"]
    results: list[dict[str, str]] = []

    for question in questions:
        q_text = question["question"]
        q_tokens = tokenize(q_text)
        q_lower = q_text.lower()

        scores = bm25.get_scores(q_tokens).copy()

        target_city = None
        for c in cities:
            if c in q_lower:
                target_city = c
                break

        for idx, doc in enumerate(documents):
            doc_title_lower = doc.title.lower()

            if "current access notice" in q_lower:
                if doc.title.startswith("CURRENT ACCESS NOTICE:"):
                    scores[idx] += 100.0
                elif "archived background" in doc_title_lower:
                    scores[idx] -= 200.0
                else:
                    scores[idx] -= 200.0

            if target_city:
                if target_city in doc_title_lower:
                    scores[idx] += 50.0
                elif any(other in doc_title_lower for other in cities if other != target_city):
                    scores[idx] -= 50.0

        best_index = int(scores.argmax())
        results.append(
            {
                "question_id": question["question_id"],
                "document_id": documents[best_index].document_id,
            }
        )

    return results


def main() -> None:
    corpus_path = DATA_DIR / "corpus.jsonl"
    questions_path = DATA_DIR / "questions.json"
    output_path = BASE_DIR / "retrieved_ids.json"
    data_output_path = DATA_DIR / "retrieved_ids.json"

    print("Loading corpus and questions...")
    documents = load_corpus(corpus_path)
    questions = load_questions(questions_path)
    print(f"Loaded {len(documents)} documents, {len(questions)} questions.")

    print("Running repaired retrieval pipeline...")
    results = retrieve_documents(documents, questions)

    # Save to both locations
    json_bytes = json.dumps(results, indent=2) + "\n"
    output_path.write_text(json_bytes, encoding="utf-8")
    data_output_path.write_text(json_bytes, encoding="utf-8")
    print(f"Saved {len(results)} results to {output_path} and {data_output_path}.")

    print("Deriving flag using flag_utils...")
    verification_codes = load_verification_codes(corpus_path)
    flag = derive_flag(results, questions, verification_codes)
    print(f"Derived Flag: {flag}")

    print("\nSubmitting flag via SigmoidClient...")
    try:
        from optimizations.sigmoid_api import SigmoidClient
        client = SigmoidClient()
        submit_res = client.submit_text("task_tanzania", flag)
        print(f"Submission response: {submit_res}")
    except Exception as e:
        print(f"Submission failed: {e}")


if __name__ == "__main__":
    main()
