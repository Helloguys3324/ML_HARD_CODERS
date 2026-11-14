# Solution Report: task_egypt ("Always Valid, No Excuses")

## 1. Overview & Objective
- **Task ID:** `task_egypt` (Egypt).
- **Difficulty & Score:** Easy, 100 / 100 points.
- **Domain:** Structured Text Parsing, OCR Noise Extraction, Deterministic Data Sanitization.
- **Input Data:** `test-input.txt` (UTF-8 encoded line-delimited records). Each line contains degraded optical character recognition (OCR) output representing an Egyptological archive record.
- **Data Normalization Specifications:**
  1. *Segment Cleaning:* Delimited by pipe characters `|`; leading and trailing whitespaces must be stripped from keys and values.
  2. *Artifact Suppression:* Scanning noise must be filtered out: page markers (`page 22`), chapter headings (`Chapter IX`), pagination dashes (`- 45 -`), bracketed defect notes (`[torn]`, `[illegible]`, `[stain]`), illustration tags (`Fig 3`, `Plate VII`, `footnote 3`), and random delimiters (`###`, `~~~`).
  3. *Key Validation:* A segment is recognized as a valid field if and only if it follows `key: value` syntax and the key conforms to `^[a-zA-Z_][a-zA-Z0-9_]*$`.
  4. *Type Casting:*
     - Empty values $\rightarrow$ JSON `null` (`None` in Python).
     - Exact strings `"true"` / `"false"` $\rightarrow$ JSON booleans `true` / `false`.
     - Decimal numbers (e.g., `40`, `0`, `-1264`) $\rightarrow$ native JSON integers.
     - All other entries $\rightarrow$ JSON strings (preserving escaped quotes and slashes).
     - Partitioning executes strictly on the **first** colon `:` (e.g., `ratio: 3:1` $\rightarrow$ `"ratio": "3:1"`).
  5. *Output Invariant:* Exactly one strictly formatted JSON object per line written to `test-output.txt`. Format violations or line count discrepancies result in rejection.

---

## 2. Key Insights & Methodology

### 2.1 Deterministic Rule Engine vs. LLM Inference
While the task description suggested an optional local LLM oracle (Qwen3-1.7B-GGUF), generative language models introduce non-deterministic edge cases: hallucinated string quotes around integers, dropped sparse fields, corrupt escape sequences on file paths, and potential timeout risks on CPU. A deterministic regular expression engine in pure Python standard library was engineered instead, guaranteeing zero hallucination and microsecond throughput.

### 2.2 Noise Filtering via Identifier Validation
- Splitting lines along `|` and filtering segments lacking `:` eliminates raw text fragments.
- Partitioning along the first colon (`seg.split(":", 1)`) and asserting `^[a-zA-Z_][a-zA-Z0-9_]*$` on `key.strip()` automatically purges complex noise segments containing colons (e.g., `Chapter IX: The Tomb`, `page 22: note`, `[torn]: missing`).

### 2.3 Strict-JSON Serialization
- Records are serialized using `json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))`.
- Enforcing `ensure_ascii=True` escapes control characters and line breaks, preventing corrupt multi-line records.
- Deserialization round-trips via `json.loads` are verified in memory prior to file emission.

---

## 3. Implementation Details

1. **`main.py` (Production Parser):**
   - **`parse_record(line: str) -> dict`**:
     - Strips newline characters and splits along `|`.
     - Validates key structure against `^[a-zA-Z_][a-zA-Z0-9_]*$`.
     - Parses integers via `^-?(?:0|[1-9]\d*)$`, maps booleans and nulls, and casts remaining payloads to strings.
   - **`find_input_file(cwd, script_dir) -> Path`**:
     - Resolves paths via CLI arguments, standard working directory files (`test-input.txt`, `input.txt`), and content signatures (`|` and `:`).
   - **`main()`**:
     - Processes input records line-by-line and streams validated JSON objects into `test-output.txt`.
2. **`validate.py` (Regression Suite):**
   - Validates sample data against `train-output.txt` (13/13 exact JSON structural matches, correctly parsing Windows paths `C:\papyri\thoth`, inner quotes `"lord of the moon"`, and negative numbers).
   - Validates specification edge case:
     `page 22| king :Menes| title: ruled with "the Aten" |[torn]| alive :true | Chapter IX`
     $\implies$ `{"king":"Menes","title":"ruled with \"the Aten\"","alive":true}`.
   - Asserts zero schema deviations across empty strings, embedded colons, and numeric boundaries (`ALL OK`).
3. **`main.py.txt` (Platform Fallback):**
   - Byte-for-byte identical mirror (MD5 `c6693a0ce13bfd3abd6304fa431495b8`) catering to platform uploaders specifying `.txt` input acceptance.

---

## 4. Results & Artifacts
- **Validation Verdict:** `ALL OK` (`[sample] OK (13 lines)`, `[spec-example] OK`, `[edge] OK`).
- **Throughput:** 100 evaluation records parsed in **0.44 seconds** on CPU (allowable limit: 60.0 s).
- **Deliverables:**
  - `main.py`: Fully autonomous, dependency-free parsing engine.
  - `main.py.txt`: Secondary mirror file matching platform extension specifications.
  - `validate.py`: Test harness covering golden samples and edge cases.
  - `test-output.txt`: Validated output reference.
  - `solution.md`: Exhaustive specification and architecture writeup.
  - `answer.txt`: Submission execution instructions (`SUBMIT_FILE ...`).
