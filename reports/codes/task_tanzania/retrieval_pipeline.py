"""Repaired retrieval pipeline for the Tanzania "Safari ya Sayansi" task.

Bugs fixed relative to the supplied starter
-------------------------------------------
1. **Scoring / embedding-space mismatch.**  The starter encoded documents with
   ``sentence-transformers/all-MiniLM-L6-v2`` but questions with
   ``intfloat/e5-small-v2``.  Vectors from two different models live in
   unrelated spaces, so their dot products are meaningless.  A single encoder
   is now used for both sides.
2. **Tokenizer / prefix mismatch.**  E5 models are trained with the asymmetric
   prefixes ``query: `` (questions) and ``passage: `` (documents).  The starter
   used ``query: `` for the questions but never added ``passage: `` to the
   documents.  Both prefixes are applied here.
3. **Index / chunking.**  The starter cut each document into arbitrary
   180-character slices, splitting sentences and the decisive "Access details"
   paragraph.  Documents are now split on paragraph boundaries (with the title
   prepended to each passage and a title-only passage added).
4. **Duplication / attribution.**  Scores are aggregated per ``document_id``
   with a max-pool instead of ranking raw chunks, and repeated passages are
   de-duplicated before encoding.  The returned value is therefore always a
   source-document ID, never a chunk index.

Run:
    python retrieval_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flag_utils import derive_flag, load_verification_codes

# One consistent encoder for both sides of the retrieval problem.
MODEL_NAME = "intfloat/e5-small-v2"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


@dataclass(frozen=True)
class Document:
    """One searchable source document."""

    document_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """One searchable passage and its source document ID."""

    document_id: str
    text: str


def load_corpus(corpus_path: Path) -> list[Document]:
    """Read source documents from a JSON Lines corpus.

    :param corpus_path: Corpus file supplied with the task.
    :return: Searchable documents in file order.
    """
    documents: list[Document] = []
    with corpus_path.open(encoding="utf-8") as corpus_file:
        for line in corpus_file:
            row = json.loads(line)
            documents.append(Document(row["document_id"], row["title"], row["text"]))
    return documents


def load_questions(questions_path: Path) -> list[dict[str, str]]:
    """Read the ordered retrieval questions.

    :param questions_path: JSON question file supplied with the task.
    :return: Question objects in official order.
    """
    with questions_path.open(encoding="utf-8") as questions_file:
        questions = json.load(questions_file)
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a list.")
    return questions


def split_into_chunks(documents: list[Document]) -> list[Chunk]:
    """Split documents into paragraph passages with title context.

    Fixed-width character slicing (the starter's bug) severs sentences and can
    separate the pickup place from the opening time.  Paragraph boundaries keep
    each operational statement intact.  The title is prepended to every passage
    so the document type ("CURRENT ACCESS NOTICE") and the exhibit/city stay in
    the passage representation.  A title-only passage is added as well.

    :param documents: Complete source documents.
    :return: De-duplicated searchable passages in corpus order.
    """
    chunks: list[Chunk] = []
    for document in documents:
        seen: set[str] = set()
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", document.text)
            if paragraph.strip()
        ]
        candidates = [f"{document.title}\n{paragraph}" for paragraph in paragraphs]
        candidates.append(document.title)
        for text in candidates:
            fingerprint = re.sub(r"\s+", " ", text).strip().casefold()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            chunks.append(Chunk(document.document_id, text))
    return chunks


def retrieve_documents(
    chunks: list[Chunk],
    questions: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Return the best source document for every question.

    Chunk similarities are max-pooled per document before the arg-max, so the
    winner is a document, not an isolated fragment of one.

    :param chunks: Searchable text passages.
    :param questions: Ordered task questions.
    :return: Top source document ID for each question.
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(MODEL_NAME)
    chunk_vectors = encoder.encode(
        [PASSAGE_PREFIX + chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_vectors = encoder.encode(
        [QUERY_PREFIX + question["question"] for question in questions],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    similarities = np.asarray(query_vectors) @ np.asarray(chunk_vectors).T

    document_ids: list[str] = []
    document_index: dict[str, int] = {}
    for chunk in chunks:
        if chunk.document_id not in document_index:
            document_index[chunk.document_id] = len(document_ids)
            document_ids.append(chunk.document_id)

    document_scores = np.full(
        (len(questions), len(document_ids)), -np.inf, dtype=similarities.dtype
    )
    for chunk_position, chunk in enumerate(chunks):
        column = document_index[chunk.document_id]
        np.maximum(
            document_scores[:, column],
            similarities[:, chunk_position],
            out=document_scores[:, column],
        )

    results: list[dict[str, str]] = []
    for question, row in zip(questions, document_scores):
        best_document_index = int(np.argmax(row))
        results.append(
            {
                "question_id": question["question_id"],
                "document_id": document_ids[best_document_index],
            }
        )
    return results


def parse_arguments() -> argparse.Namespace:
    """Parse command-line file locations.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("corpus.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("questions.json"))
    parser.add_argument("--output", type=Path, default=Path("retrieved_ids.json"))
    return parser.parse_args()


def main() -> None:
    """Run retrieval, save IDs, and print the derived flag."""
    arguments = parse_arguments()
    documents = load_corpus(arguments.corpus)
    questions = load_questions(arguments.questions)
    results = retrieve_documents(split_into_chunks(documents), questions)
    arguments.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    flag = derive_flag(results, questions, load_verification_codes(arguments.corpus))
    print(f"Saved {len(results)} results to {arguments.output}.")
    print(flag)


if __name__ == "__main__":
    main()
