# Solution Report: task_usa ("The Great American Token Mystery")

## 1. Overview & Objective
- **Task ID:** `task_usa` (United States).
- **Difficulty:** Medium / Reverse Engineering.
- **Submission Type:** Executable script `main.py` evaluated offline in a sandboxed grading environment.
- **Contract & Requirements:**
  - Reads `test-input.json` containing:
    ```json
    {
      "calibration": {"text": "...", "token_ids": [...]},
      "challenge":   {"token_ids": [...]}
    }
    ```
  - Blindly identifies which subword tokenizer produced `calibration.token_ids` from `calibration.text`.
  - Decodes `challenge.token_ids` using the detected tokenizer.
  - Writes `test-output.json`:
    ```json
    {
      "tokenizer_id": "<id>",
      "decoded_message": "<text>"
    }
    ```
- **Platform Constraints:**
  - Strictly offline evaluation (no internet access; external libraries like `tiktoken`, `transformers`, `tokenizers`, and `sentencepiece` are unavailable).
  - Maximum file size: $\le 10.0$ MB. Execution time: under 10 seconds.
- **Goal:** Construct a standalone `main.py` in pure Python stdlib containing embedded, compressed vocabularies for all dominant NLP tokenizer families, capable of zero-dependency detection and exact string reconstruction.

---

## 2. Key Insights & Methodology

### 2.1 Baseline Calibration (`train-input.json`)
- On the training calibration split, token IDs matched **`p50k_base`** (OpenAI Davinci / Code-Davinci) with 73/73 tokens aligning perfectly.
- Decoding `challenge.token_ids` via `p50k_base` exactly reproduced the reference message from `train-output.json`:
  `"TRAIN_SAMPLE: Verification successful at the American NLP Observatory. Token IDs map to p50k_base vocabulary!"`.

### 2.2 Embedded Registry Architecture
- To bypass external dependency restrictions, vocabularies for **22 unique tokenizers** were serialized into a single binary payload via `lzma` + `base85` + `pickle`, embedded directly into `main.py`.
- Final script size: **8.67 MB** (comfortably under the 10.0 MB threshold).
- **Supported Families (22 Vocabularies):**
  - **tiktoken:** `gpt2` (`r50k_base`), `p50k_base`, `cl100k_base` (GPT-4), `o200k_base` (GPT-4o).
  - **HuggingFace Byte-level BPE:** `roberta-base`/`bart`, `Llama-3-8B` (128k), `Qwen2.5-0.5B` (151k), `SmolLM2-135M`, `Falcon-7B`, `StarCoder2-3B`, `CodeGen-350M`/`phi-2`.
  - **HuggingFace WordPiece:** `bert-base-uncased`, `bert-base-cased`, `bert-base-multilingual-cased`.
  - **SentencePiece / Metaspace:** `t5-small`, `albert-base-v2`, `xlnet-base-cased`, `Llama-2-7B`/`TinyLlama`, `Mistral-7B-v0.1`, `Phi-3-mini`, `DeBERTa-v3-small`, `XLM-RoBERTa-base`.

### 2.3 Pure-Python Tokenizer Engine & Identification
- **Custom Decoders:** Pure-Python implementations handling byte fallback tokens, UTF-8 byte assembly, SentencePiece metaspace ` `, byte-level BPE character mapping, and WordPiece `##` prefix concatenation.
- **Scoring Pipeline:** Decodes `calibration.token_ids` across candidate vocabularies and scores against `calibration.text` using:
  1. Exact byte matching and normalized whitespace/casing.
  2. Special token stripping (`[CLS]`, `<s>`, `<pad>`, `[SEP]`).
  3. `difflib.SequenceMatcher` similarity.
  4. Heavy penalties for Out-Of-Vocabulary (OOV) tokens, preventing superset vocabularies from overshadowing subsets (e.g., distinguishing `gpt2` from `p50k_base`).
- **Self-Verification:** Extracts the candidate name embedded in the message pattern (`"Token IDs map to <name> vocabulary!"`) to cross-validate candidate ranking.

### 2.4 Test Split Evaluation (`test-output.json`)
- Evaluated on the hidden test set, the algorithm conclusively identified **`cl100k_base`**.
- The decoded payload revealed the challenge flag:
  ```
  SIGMOID_TOKENIZATION
  Congratulations! You have successfully solved the American Tokenizer Mystery.
      Verification Hash: 2026-USA-NLP-BPE
          Cluster Node: Grand-Canyon-Compute-01
  Subword token IDs are arbitrary vocabulary matrix indices, not universal ASCII codes!
  ```

---

## 3. Implementation Details

1. **`main.py` (Submission Script, 8.67 MB):**
   - Unpacks embedded vocabularies:
     ```python
     raw_data = pickle.loads(lzma.decompress(base64.b85decode(VOCAB_DATA_B85)))
     ```
   - Features lightweight decoders: `TiktokenDecoder`, `ByteBpeDecoder`, `SentencePieceDecoder`, `WordPieceDecoder`.
   - Execution performance: ~3.5–4.5s total runtime, ~155 MB peak memory usage.

2. **`build_registry.py` & `build_main.py` (Artifact Builder):**
   - Ingests raw HuggingFace `tokenizer.json`, SentencePiece `.model`, and tiktoken dumps, serializing token-to-byte mappings into `main_template.py`.

3. **`validate.py` (Parity Verification):**
   - Compares the pure-Python decoders against native `tiktoken`, `tokenizers`, and `sentencepiece` across 11 complex edge-case sequences (emojis, multilingual text, apostrophes, and unusual spacing).
   - Achieved **0 mismatches across all 22 tokenizers**.

4. **`synthetic_test.py` (End-to-End Stress Test):**
   - Generates 22 synthetic challenges across each supported tokenizer.
   - Result: **22/22 tests passed** with exact `tokenizer_id` identification and 100% character recovery.

---

## 4. Results & Artifacts
- **Submission Artifact:** `main.py` (pure Python stdlib, 8.67 MB).
- **Detected Test Tokenizer:** `cl100k_base`
- **Extracted Flag:** `SIGMOID_TOKENIZATION` (verification hash: `2026-USA-NLP-BPE`).
- **Key Artifacts (`task_usa/`):**
  - `main.py`: Autonomous solver containing packed vocabularies.
  - `test-output.json`: Graded test output (`tokenizer_id: cl100k_base`, payload with flag).
  - `answer.txt`: Verification summary.
  - `answer_guess.json`: Local validation prediction.
  - `build_registry.py` & `build_main.py`: Binary compilation and packaging scripts.
  - `validate.py`: Parity audit script against official tokenizer libraries.
  - `synthetic_test.py`: 22-way synthetic benchmark harness.
  - `solution.md`: Comprehensive reverse-engineering documentation.
