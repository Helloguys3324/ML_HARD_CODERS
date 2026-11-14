#!/usr/bin/env python3
"""Reproduce and verify the Tanzania task answer.

This script is a lightweight verifier: it re-derives the flag from the repaired
``retrieved_ids.json`` and independently validates every retrieved document
against the question text (exhibit + city -> "CURRENT ACCESS NOTICE: ...").

Usage:
    /home/kozak/mllm/py solve.py [--data work] [--results retrieved_ids.json]

Use ``run_pipeline.sh`` (or run ``retrieval_pipeline.py`` directly) to
regenerate ``retrieved_ids.json`` from scratch with the repaired retriever.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from flag_utils import derive_flag_from_files, load_verification_codes

CITIES = ("Dar es Salaam", "Arusha", "Mwanza")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("work"))
    parser.add_argument("--results", type=Path, default=Path("retrieved_ids.json"))
    arguments = parser.parse_args()

    corpus_path = arguments.data / "corpus.jsonl"
    questions_path = arguments.data / "questions.json"
    documents = {
        row["document_id"]: row
        for row in (
            json.loads(line)
            for line in corpus_path.open(encoding="utf-8")
        )
    }
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    results = json.loads(arguments.results.read_text(encoding="utf-8"))

    current_notices = [
        document
        for document in documents.values()
        if document["title"].startswith("CURRENT ACCESS NOTICE:")
    ]
    exhibits = sorted(
        {
            document["title"].split(": ", 1)[1].rsplit(", ", 1)[0]
            for document in current_notices
        }
    )

    assert len(questions) == 72, len(questions)
    assert len(results) == len(questions), (len(results), len(questions))
    assert [row["question_id"] for row in results] == [
        question["question_id"] for question in questions
    ], "results are not in questions.json order"
    assert len({row["document_id"] for row in results}) == 72, "documents repeat"

    evidence_rows = []
    verified = 0
    for question, result in zip(questions, results):
        text = question["question"]
        matching_exhibits = [name for name in exhibits if name.lower() in text.lower()]
        matching_cities = [city for city in CITIES if city in text]
        assert len(matching_exhibits) == 1, (question["question_id"], matching_exhibits)
        assert len(matching_cities) == 1, (question["question_id"], matching_cities)
        expected_title = (
            f"CURRENT ACCESS NOTICE: {matching_exhibits[0]}, {matching_cities[0]}"
        )
        document = documents[result["document_id"]]
        assert document["title"] == expected_title, (
            question["question_id"],
            document["title"],
            expected_title,
        )
        verified += 1
        evidence_rows.append(
            {
                "question_id": question["question_id"],
                "document_id": result["document_id"],
                "document_title": document["title"],
                "verification_code": document["verification_code"],
            }
        )

    verification_codes = load_verification_codes(corpus_path)
    sequence = "".join(
        verification_codes[row["document_id"]] for row in results
    )
    flag = derive_flag_from_files(arguments.results, questions_path, corpus_path)

    evidence_path = arguments.results.parent / "retrieval_evidence.csv"
    with evidence_path.open("w", newline="", encoding="utf-8") as evidence_file:
        writer = csv.DictWriter(evidence_file, fieldnames=list(evidence_rows[0]))
        writer.writeheader()
        writer.writerows(evidence_rows)

    print(f"Validated {verified}/{len(questions)} retrieved documents.")
    print(f"Verification-character sequence ({len(sequence)} chars): {sequence}")
    print(f"Flag: {flag}")
    print(f"Evidence table: {evidence_path}")


if __name__ == "__main__":
    main()
