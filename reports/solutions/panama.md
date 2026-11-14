# Solution Report: task_panama ("The Canal Knows Too Much")

## 1. Overview & Objective
- **Task ID:** `task_panama` (Panama).
- **Difficulty & Score:** Medium, 200 / 200 points.
- **Domain:** Natural Language Processing (NLP) / Data Contamination & Audit.
- **Output Format:** Audit report `report.json` and flag `SIGMOID_<NAME>` via `open_seal.py`.
- **Premise:**
  - A maritime QA model developed for Panama Canal pilots was fine-tuned on operational logs (English shift reports and Spanish *registros* from Gatún to Miraflores locks).
  - The model achieved an anomalous 73.75% accuracy on the internal **CanalQA** benchmark due to data contamination.
  - The objective is to audit the dataset, isolate questions leaked from the training corpus, compute the model's true **honest accuracy**, and decrypt the verification certificate `audit_seal.json`.

---

## 2. Key Insights & Audit Methodology
- **Cross-Lingual Contamination Profile:**
  - **Corpus:** 180 operational logs across English and Spanish.
  - **Benchmark:** 80 factual questions paired with reference `gold` answers.
  - Leakage occurred cross-lingually: several English test queries tested facts directly translated from Spanish maintenance logs.
- **Multi-Stage Normalization Pipeline:**
  1. *Numerical & Unit Canonicalization:* Replaced European decimal commas (`26,5` $\to$ `26.5`), stripped thousand separators (`1,432` $\to$ `1432`), standardized units (`metros`/`meters` $\to$ `m`, `hectáreas` $\to$ `ha`, `kilómetros cuadrados` $\to$ `km2`), and mapped spelled numbers to Arabic numerals (`cero`..`cincuenta`, `zero`..`ten`).
  2. *Domain Translation:* Translated Spanish documents into English using the domain lexicon `glossary_es_en`.
- **Contamination Scoring Metric:**
  - Evaluated question-document lexical similarity and exact substring containment of the normalized reference answer:
    $$\text{score}(Q, D) = \text{JaccardOverlap}(Q_{\text{tokens}}, D_{\text{tokens}}) + 1.0 \cdot \mathbb{I}(\text{gold} \in D)$$
  - Threshold $\text{score} \ge 1.5$ flags a candidate query as contaminated.
- **Cryptographic Seal Enforcement:**
  - `audit_seal.json` is secured with PBKDF2-HMAC-SHA256 (200,000 iterations) + AES-CTR keystream + HMAC-SHA256.
  - The decryption key requires the canonical sorted newline-separated string of leaked question IDs (`"\n".join(sorted(ids))`). A single false positive or negative prevents decryption.

---

## 3. Implementation Details
The detection pipeline is implemented in `analyze.py`, outputs to `report.json`, and verifies against `open_seal.py`:

1. **Preprocessing & Corpus Translation (`analyze.py`):**
   - Implements `normalize(text)` and `translate_es_to_en(text)`.
   - Filters English stopwords (`STOP = {"the", "of", "what", "which", ...}`) from queries.
2. **Leakage Identification:**
   - Automated screening identified 19 obvious leaks; deep inspection of borderline candidates added `CQA-026` and `CQA-038`, confirming exactly **21 contaminated questions**:
     ```text
     CQA-001, CQA-002, CQA-008, CQA-012, CQA-013, CQA-018, CQA-019, CQA-022,
     CQA-026, CQA-034, CQA-035, CQA-038, CQA-040, CQA-051, CQA-057, CQA-058,
     CQA-060, CQA-061, CQA-070, CQA-074, CQA-075
     ```
3. **Honest Accuracy Calculation:**
   - Decontaminated evaluation subset: $80 - 21 = 59$ independent questions.
   - The model answered 38 of these 59 questions correctly:
     $$\text{Honest Accuracy} = \frac{38}{59} \approx 64.41\% \quad (0.6441)$$
4. **Certificate Decryption (`open_seal.py`):**
   - Derives key from canonical list of 21 IDs, validates the HMAC authentication tag, and decrypts `audit_seal.json`.

---

## 4. Results & Artifacts
- **Status:** PASSED
- **Score:** 200 / 200 points
- **Recovered Flag:** `SIGMOID_DECONTAMINATION`
- **Audit Verification Output:**
  ```text
  Seal opened.
    certified honest accuracy: 64.41%
    flag: SIGMOID_DECONTAMINATION
  ```
- **Output Artifacts:**
  - `report.json` — contains `excluded_ids` and `honest_accuracy: 0.6441`;
  - `analyze.py` — complete audit and normalization script;
  - `open_seal.py` — cryptographic validation utility.
