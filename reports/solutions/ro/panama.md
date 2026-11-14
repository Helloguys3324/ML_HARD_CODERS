# Raport de Soluție: task_panama ("The Canal Knows Too Much")

## 1. Prezentare Generală și Obiectiv
- **Task ID:** `task_panama` (Panama).
- **Dificultate și Scor:** Medium, 200 / 200 puncte.
- **Domeniu:** Procesarea Limbajului Natural (NLP) / Contaminarea Datelor și Audit.
- **Format Ieșire:** Raport de audit `report.json` și flag `SIGMOID_<NAME>` via `open_seal.py`.
- **Premisă:**
  - Un model maritim de QA dezvoltat pentru piloții Canalului Panama a fost adaptat (fine-tuned) pe jurnale operaționale (rapoarte de tură în engleză și *registros* în spaniolă de la ecluzele Gatún până la Miraflores).
  - Modelul a obținut o acuratețe anomală de 73.75% pe benchmark-ul intern **CanalQA** din cauza contaminării datelor (data contamination).
  - Obiectivul este auditarea setului de date, izolarea întrebărilor scurse din corpusul de antrenare, calcularea adevăratei **acurateți oneste (honest accuracy)** a modelului și decriptarea certificatului de verificare `audit_seal.json`.

---

## 2. Perspective Cheie și Metodologia de Audit
- **Profilul de Contaminare Trans-Lingvistică:**
  - **Corpus:** 180 de jurnale operaționale în engleză și spaniolă.
  - **Benchmark:** 80 de întrebări factuale asociate cu răspunsuri de referință `gold`.
  - Scurgerea de date a fost trans-lingvistică: mai multe întrebări de test în engleză evaluau fapte traduse direct din jurnalele de mentenanță în spaniolă.
- **Pipeline de Normalizare în Mai Multe Etape:**
  1. *Canonizarea Numerelor și Unităților:* Înlocuirea virgulelor zecimale europene (`26,5` $\to$ `26.5`), eliminarea separatorilor de mii (`1,432` $\to$ `1432`), standardizarea unităților (`metros`/`meters` $\to$ `m`, `hectáreas` $\to$ `ha`, `kilómetros cuadrados` $\to$ `km2`) și maparea numerelor scrise în cuvinte la cifre arabe (`cero`..`cincuenta`, `zero`..`ten`).
  2. *Traducere de Domeniu:* Traducerea documentelor spaniole în engleză folosind lexiconul specializat `glossary_es_en`.
- **Metrica de Punctare a Contaminării:**
  - Evaluarea similarității lexicale întrebare-document și a incluziunii exacte a răspunsului de referință normalizat ca subșir:
    $$\text{score}(Q, D) = \text{JaccardOverlap}(Q_{\text{tokens}}, D_{\text{tokens}}) + 1.0 \cdot \mathbb{I}(\text{gold} \in D)$$
  - Pragul $\text{score} \ge 1.5$ marchează o întrebare candidată drept contaminată.
- **Securizarea Criptografică a Sigiliului:**
  - `audit_seal.json` este securizat cu PBKDF2-HMAC-SHA256 (200.000 de iterații) + flux de cheie AES-CTR + HMAC-SHA256.
  - Cheia de decriptare necesită șirul canonic sortat, separat prin newline, al ID-urilor întrebărilor scurse (`"\n".join(sorted(ids))`). Un singur fals pozitiv sau fals negativ împiedică decriptarea.

---

## 3. Detalii de Implementare
Pipeline-ul de detecție este implementat în `analyze.py`, exportă rezultatele în `report.json` și se verifică prin `open_seal.py`:

1. **Preprocesare și Traducerea Corpusului (`analyze.py`):**
   - Implementează funcțiile `normalize(text)` și `translate_es_to_en(text)`.
   - Filtrează cuvintele de legătură (stopwords) în engleză (`STOP = {"the", "of", "what", "which", ...}`) din întrebări.
2. **Identificarea Scurgerilor de Date:**
   - Filtrarea automată a identificat 19 scurgeri evidente; analiza detaliată a candidaților limitrofi a adăugat `CQA-026` și `CQA-038`, confirmând exact **21 de întrebări contaminate**:
     ```text
     CQA-001, CQA-002, CQA-008, CQA-012, CQA-013, CQA-018, CQA-019, CQA-022,
     CQA-026, CQA-034, CQA-035, CQA-038, CQA-040, CQA-051, CQA-057, CQA-058,
     CQA-060, CQA-061, CQA-070, CQA-074, CQA-075
     ```
3. **Calculul Acurateții Oneste:**
   - Subsetul de evaluare decontaminat: $80 - 21 = 59$ întrebări independente.
   - Modelul a răspuns corect la 38 din aceste 59 de întrebări:
     $$\text{Honest Accuracy} = \frac{38}{59} \approx 64.41\% \quad (0.6441)$$
4. **Decriptarea Certificatului (`open_seal.py`):**
   - Generează cheia din lista canonică a celor 21 de ID-uri, validează eticheta de autentificare HMAC și decriptează `audit_seal.json`.

---

## 4. Rezultate și Artefacte
- **Status:** PASSED
- **Scor:** 200 / 200 puncte
- **Flag Recuperat:** `SIGMOID_DECONTAMINATION`
- **Ieșirea Verificării Auditului:**
  ```text
  Seal opened.
    certified honest accuracy: 64.41%
    flag: SIGMOID_DECONTAMINATION
  ```
- **Artefacte Rezultate:**
  - `report.json` — conține `excluded_ids` și `honest_accuracy: 0.6441`;
  - `analyze.py` — script complet de audit și normalizare;
  - `open_seal.py` — utilitar de validare criptografică.
