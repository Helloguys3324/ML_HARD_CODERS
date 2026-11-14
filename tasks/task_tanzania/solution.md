# task_tanzania — "Please, You Are So Delusional!"

- **Task ID:** `task_tanzania` (country TZ)
- **Difficulty:** Easy
- **Input:** text flag
- **Answer:** `SIGMOID_A69EE28C5B2214A9AD99`
- **Confidence:** **high**

## TL;DR

The corpus contains 744 documents: **600 generic operations bulletins** (decoys),
**72 `CURRENT ACCESS NOTICE` documents** (the authoritative source), and **72
`ARCHIVED BACKGROUND` documents** (withdrawn decoys that literally mention the
"current access notice" in their body). Each of the 72 questions names exactly
one exhibit and exactly one city, so its correct source is the unique
`CURRENT ACCESS NOTICE: <exhibit>, <city>` document. The repaired embedding
pipeline recovers that document for all 72 questions, and `flag_utils.py`
derives:

```
KZIP27S3IQ7W9YBSBPIEBPU955ZB4ZNWZQAXLUWNME79I8JRIT0DTOW4JG1OEI85K3KMX1M0
→ SIGMOID_A69EE28C5B2214A9AD99
```

The task description's own expected-output example gives `q001 → TZ-01-1A`;
our result for `q001` is exactly `TZ-01-1A`.

## Data inspection

`corpus.jsonl` (744 rows, all unique IDs/titles/texts):

| Title prefix | Count | Role |
|---|---:|---|
| `Lake Hall / Summit Hall / Harbour Hall operations bulletin #` | 600 | Generic decoy notices ("contains no equipment reservation or collection information") |
| `CURRENT ACCESS NOTICE: <exhibit>, <city>` | 72 | 24 exhibits × 3 cities — **authoritative access details** |
| `ARCHIVED BACKGROUND: <exhibit>, <city>` | 72 | Withdrawn historical material — explicitly says it "does not replace the current access notice" |

`questions.json`: 72 questions, `q001`…`q072`, in official order. Every question
contains exactly one canonical exhibit name (`Solar Spectroscope`, `Portable
Planetarium`, …) and exactly one city (`Dar es Salaam`, `Arusha`, `Mwanza`).
That gives a deterministic semantic oracle: 72 distinct `(exhibit, city)` pairs
→ 72 distinct current-access-notice IDs.

The archived decoys are a deliberate trap: their body text repeats the phrase
"current access notice" and "withdrawn", so naive full-text term matching ranks
them **above** the real notices. A plain TF-IDF title+text baseline scored
**0/72** and selected an archived document every time.

## Bugs in the supplied starter (`retrieval_pipeline.py`)

1. **Scoring / embedding-space mismatch (fatal).**
   `DOCUMENT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"` but
   `QUERY_MODEL = "intfloat/e5-small-v2"`. The document and query vectors come
   from two unrelated encoders, so the cosine/dot-product scores are
   meaningless. The starter therefore retrieves nonsense (6/72 correct, 41/72
   archived documents).
2. **Tokenizer / prefix mismatch.** E5 is trained with asymmetric prefixes. The
   starter adds `query: ` to questions but never adds `passage: ` to document
   chunks (and documents are not even E5-encoded).
3. **Broken chunking/index.** Fixed 180-character slices cut sentences in half
   and separate the "Access details" paragraph (pickup point + starting time)
   from its context. The title is duplicated inside every slice without
   normalisation.
4. **Chunk-level argmax instead of document ranking.** The code returns the
   document of the single best chunk. Scores are never aggregated per
   `document_id`, so a document with many chunks is implicitly favoured and the
   retrieval unit is a fragment, not a source document.

## Repair (`retrieval_pipeline.py`)

- One consistent encoder for **both** sides: `intfloat/e5-small-v2`.
- Correct E5 asymmetry: `query: ` for questions, `passage: ` for documents.
- Split documents on **paragraph boundaries** (keeping the title prepended to
  each passage and adding a title-only passage), so operational statements stay
  intact.
- De-duplicate identical passages per document.
- **Max-pool chunk similarities per `document_id`**, then take the arg-max over
  documents. The output is always a document ID, in `questions.json` order.
- Kept the starter CLI (`--corpus`, `--questions`, `--output`) and the
  `flag_utils.derive_flag` integration unchanged.

The pinned `sentence-transformers`/E5 model is explicitly part of the supplied
task materials (`requirements.txt`, starter constants), so using it is within
the task's fair-play rules; no external datasets or APIs were used.

## Evidence / validation

All commands were run with `/home/kozak/mllm/py` (Python 3.13 venv).

1. **Repaired pipeline run** (from the extracted participant directory):
   ```bash
   HF_HOME=/tmp/opencode/hf /home/kozak/mllm/py ../retrieval_pipeline.py \
       --output ../retrieved_ids.json
   ```
   Output:
   ```
   Saved 72 results to ../retrieved_ids.json.
   SIGMOID_A69EE28C5B2214A9AD99
   ```
2. **Structured verification** (`solve.py`) independently parses every question,
   locates its unique exhibit and city, and checks the retrieved title equals
   `CURRENT ACCESS NOTICE: <exhibit>, <city>`:
   ```
   Validated 72/72 retrieved documents.
   Verification-character sequence (72 chars): KZIP27...M0
   Flag: SIGMOID_A69EE28C5B2214A9AD99
   ```
   The repaired pipeline's 72 IDs are exactly the same 72 IDs as the
   independent structured oracle, all unique.
3. **Broken-starter baseline** (same data, original code):
   `SIGMOID_E6F6657DA55CB419CE78`, only **6/72** documents correct
   (41 archived, 22 current, 9 generic bulletins).
4. **Ablations** (accuracy against the structured oracle):
   | Configuration | Accuracy |
   |---|---:|
   | Broken starter (MiniLM docs + E5 queries, 180-char chunks) | 6/72 (8.3%) |
   | E5, single model, whole-document title+text | 60/72 (83.3%) |
   | E5, paragraph passages, max-pool per document (repaired) | **72/72** |
   | E5, title only | 72/72 |
   | TF-IDF title only | 72/72 |
   | TF-IDF title+text | 0/72 (always picks archived decoys) |

   This isolates the fixes: the single shared encoder + E5 prefixes +
   paragraph-level passages + document-level max-pooling are what recover the
   correct notices. Whole-document embedding still loses 12 questions to the
   archived decoys, which is why the paragraph index and title context matter.

`retrieval_evidence.csv` lists all 72 `(question, retrieved document title,
document ID, verification character)` tuples. `diagnostics.json` records the
baseline/ablation numbers.

## Why the answer is right

- The mapping is semantically forced: every question asks for the **current
  access notice** of a named exhibit in a named city; the corpus contains
  exactly one such document for each of the 72 `(exhibit, city)` pairs.
- Only the 72 `CURRENT ACCESS NOTICE` documents contain an `Access details:`
  paragraph and a clock time (72/72). None of the 72 archived documents and
  none of the 600 bulletins contains either — they cannot answer the
  "collection point and opening time" questions.
- The repaired retrieval pipeline returns exactly those 72 documents and
  nothing else.
- `q001 → TZ-01-1A` matches the task description's sample output.
- `flag_utils.derive_flag` (unmodified supplied helper) hashes the 72
  verification characters in `questions.json` order:
  `SHA-256(sequence)[:20].upper()` = `A69EE28C5B2214A9AD99`.

## Artifacts

- `answer.txt` — `SIGMOID_A69EE28C5B2214A9AD99`
- `retrieval_pipeline.py` — repaired pipeline
- `flag_utils.py` — unmodified supplied helper
- `retrieved_ids.json` — 72 results in official question order
- `solve.py` — independent validation + flag derivation
- `retrieval_evidence.csv` — per-question evidence table
- `diagnostics.json` — baseline/ablation evidence
- `run_pipeline.sh` — one-command rerun of the repaired pipeline
- `work/` — extracted participant materials and raw run outputs
  (`broken_ids.json`, `rule_results.json`)

No prompt-injection artifacts were observed in the task data.
