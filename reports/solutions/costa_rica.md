# Solution Report: task_costa_rica ("Pura Vida, No PII")

## 1. Overview & Objective
- **Task ID:** `task_costa_rica` (Costa Rica).
- **Difficulty & Score:** Medium, 200 / 200 points.
- **Domain:** Natural Language Processing, Privacy Engineering, Named Entity De-identification.
- **Context:** Travel agency Pura Vida Voyages shares client support correspondence with third-party analytics. All Personally Identifiable Information (PII) must be redacted while preserving downstream analytics signals.
- **Evaluation Criteria:**
  - **Privacy:** Proportion of true PII spans redacted/masked ($\ge 0.95$ threshold).
  - **Utility:** Proportion of documents where downstream rule-based analytics accurately extracts three key attributes: **topic**, **urgency**, and **booking reference** ($\ge 0.85$ threshold).
- **Format:** Autonomous script `main.py` consuming `test-input.json` (`{"documents": [{"id": ..., "text": ...}]}`) and emitting `test-output.json` (`{"redacted_documents": {id: text}}`).

---

## 2. Key Insights & Methodology

### 2.1 Targeted Typed Masking
Rather than broad sentence deletion or blanket redaction, spans are substituted with category tags:
- `[NAME]`: Passenger, contact, and family member names.
- `[EMAIL]`: RFC-compliant email addresses.
- `[PHONE]`: International and local Costa Rican telephone formats.
- `[ADDRESS]`: Street lines, avenues, and numbered coordinates.
- `[ID]`: Costa Rican National Identity cards (cédulas).
- `[DOB]`: Explicit dates of birth.
- `[ACCOUNT]`: Costa Rican bank account identifiers (CR IBAN format).

### 2.2 Strict Rule Ordering
To prevent regex collisions (e.g., telephone rules greedily matching portions of IBAN or cédula numbers), patterns execute in fixed sequence:
$$\text{IBAN } (\texttt{CR...}) \longrightarrow \text{Intl. Phones } (\texttt{+506...}) \longrightarrow \text{Cédula} \longrightarrow \text{DOB} \longrightarrow \text{Local Phones} \longrightarrow \text{Address} \longrightarrow \text{Email} \longrightarrow \text{Names}$$

### 2.3 Downstream Utility Preservation
- **Booking Codes:** Identifiers matching `PV-[A-Z0-9]{5}` are explicitly protected from numeric/account masking.
- **Classification Keywords:** Critical intent tokens (`refund`, `complaint`, `urgent`, `reschedule`, `payment`, `time-sensitive`, `delay`) are retained.
- **Geographic & Brand Context:** Non-PII entities (`Pura Vida Voyages`, `San José`, `Manuel Antonio`, `Arenal`) are preserved.

---

## 3. Implementation Details

1. **`main.py` (Standalone Rule Engine):**
   - Built with zero external dependencies (pure Python `re`).
   - Lexicon dictionaries: `STREET_TYPES` (`Avenida`, `Calle`, `Carrera`...), `STREET_NAMES`, `MONTHS_EN`/`ES`, `SIGN_OFFS` (`Regards`, `Atentamente`...), and `RELATIONS` (`son`, `wife`, `husband`...).
   - `_extract_names(text)`:
     - Leverages diacritic normalization (`_ACCENT_FOLD`).
     - Extracts names via contextual intro patterns (*"My name is..."*, *"I am..."*).
     - Captures signatures following sign-off phrases.
     - Deconstructs email local parts to locate corresponding name occurrences in the body.
   - Regex transformations:
     - IBAN: `CR\d{2}\s*(?:\d{4}\s*){4}\d{2}` / `CR\d{20}` $\rightarrow$ `[ACCOUNT]`.
     - Cédula: `\b\d-\d{4}-\d{4}\b` / `\b\d{9}\b` $\rightarrow$ `[ID]`.
     - Phones: `\+?506\s*[-.\s]?\d{4}[-.\s]?\d{4}` $\rightarrow$ `[PHONE]`.
     - Dates of Birth: Numeric and worded calendar combinations $\rightarrow$ `[DOB]`.
     - Addresses: Street/Avenue coordinates $\rightarrow$ `[ADDRESS]`.
     - Emails: RFC patterns $\rightarrow$ `[EMAIL]`.
2. **`solve.py` (Golden Validation):**
   - Evaluated `main.redact` against all 20 training records in `train-input.json`.
   - Achieved **20 / 20 exact character-level matches** with `train-output.json`.

---

## 4. Results & Artifacts
- **Status:** `passed`
- **Score:** `200 / 200`
- **Privacy Score:** **1.000** (pass mark $\ge 0.95$) ✅
- **Utility Score:** **1.000** (pass mark $\ge 0.85$) ✅
- **Recovered Flag:** `SIGMOID_DEIDENTIFICATION`
- **Deliverables:**
  - `main.py`: Production-ready inference module.
  - `solve.py`: Golden-set validation and regression tester.
  - `writeup.md`: Technical documentation of rule definitions.
