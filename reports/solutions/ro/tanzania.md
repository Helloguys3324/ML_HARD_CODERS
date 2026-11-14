# Raport de soluție: task_tanzania ("Please, You Are So Delusional!")

## 1. Prezentare generală și obiectiv
- **Task ID:** `task_tanzania` (Tanzania).
- **Dificultate:** Ușor (100 puncte).
- **Domeniu:** Recuperarea informației / RAG (Retrieval-Augmented Generation) + Filtrarea capcanelor adversariale (Adversarial Decoys).
- **Format de ieșire:** Șir text `SIGMOID_<HASH>`.
- **Date de intrare:**
  - `corpus.jsonl`: 744 documente:
    - **600 buletine operaționale generice:** Cu prefixul `Lake Hall / Summit Hall / Harbour Hall operations bulletin #` (zgomot de fundal care precizează explicit lipsa informațiilor despre rezervări).
    - **72 notificări curente de acces:** Cu prefixul `CURRENT ACCESS NOTICE: <exhibit>, <city>` — **singurele surse autoritare** care conțin locațiile de preluare (`Access details:`) și orele de funcționare pentru 24 de exponate din Dar es Salaam, Arusha și Mwanza.
    - **72 buletine arhivate:** Cu prefixul `ARCHIVED BACKGROUND: <exhibit>, <city>` — capcane adversariale conținând date învechite și repetând expresiile literale „current access notice” și „withdrawn”.
  - `questions.json`: 72 de întrebări (`q001` până la `q072`) în ordine secvențială strictă. Fiecare interogare face referire exact la un exponat și la un oraș.
- **Obiectiv:** Maparea fiecăreia dintre cele 72 de interogări către `document_id`-ul său autoritar real, extragerea șirului ordonat de verificare de 72 de caractere și calcularea flag-ului pe baza hash-ului SHA-256.

---

## 2. Observații-cheie și metodologie

### 2.1 Diagnosticarea erorilor din codul de pornire (`retrieval_pipeline.py`)
- **Nepotrivire la nivelul arhitecturii Dual-Encoder:** Pipeline-ul inițial a generat embedding-uri pentru pasajele din corpus folosind `sentence-transformers/all-MiniLM-L6-v2`, dar a codificat interogările cu `intfloat/e5-small-v2`. Din cauza spațiilor vectoriale nealiniate, similitudinea cosinus a generat zgomot aleatoriu, obținând o acuratețe de doar 6/72 și alegând capcane de 41 de ori.
- **Încălcarea prefixelor asimetrice E5:** Modelele E5 impun utilizarea prefixului `query: ` pentru interogări și `passage: ` pentru documente. Codul de pornire a adăugat prefixul la interogări, dar l-a omis la documente.
- **Trunchierea entităților:** Segmentarea rigidă în blocuri de 180 de caractere a divizat denumirile compuse ale exponatelor și a separat detaliile de acces de anteturile documentelor.
- **Capcane adversariale pentru TF-IDF naiv:** Deoarece documentele-capcană arhivate repetau frecvent expresia-cheie „current access notice”, un TF-IDF neponderat a înregistrat un scor de **0/72** (100% prins în capcane).

### 2.2 Două arhitecturi de recuperare (Acuratețe 100%)
Au fost dezvoltate două pipeline-uri de recuperare independente și complementare:
1. **Retriever dens neural (`retrieval_pipeline.py`):**
   - Model simetric `intfloat/e5-small-v2` atât pentru interogări (`query: `), cât și pentru pasaje (`passage: `).
   - Împărțire în bucăți bazată pe limitele paragrafelor, cu titlul documentului adăugat la începutul fiecărui chunk.
   - **Max-pooling la nivel de document:** Agregarea scorurilor maxime de similaritate ale bucăților per `document_id` înainte de clasificare.
2. **Retriever lexical determinist cu filtrare pe bază de reguli (`solve.py`):**
   - BM25Okapi cu ponderare de 6× a titlului: $\text{tokens} = \text{title\_tokens} \times 6 + \text{body\_tokens}$.
   - Filtrarea intenției interogării: bonus de $+100.0$ pentru prefixul `CURRENT ACCESS NOTICE:`, penalizare de $-200.0$ pentru capcanele `ARCHIVED BACKGROUND` și penalizare de $-50.0$ pentru nepotrivirea orașului.
- Ambele pipeline-uri au atins o acuratețe de **72/72 (100%)**, potrivindu-se perfect cu referința de bază `q001 -> TZ-01-1A`.

### 2.3 Deducerea flag-ului
- Maparea celor 72 de ID-uri de documente autoritare în ordinea întrebărilor produce secvența de verificare de 72 de caractere:
  ```
  KZIP27S3IQ7W9YBSBPIEBPU955ZB4ZNWZQAXLUWNME79I8JRIT0DTOW4JG1OEI85K3KMX1M0
  ```
- Calcularea primelor 20 de caractere hexadecimale cu majuscule ale hash-ului SHA-256:
  $$\text{SHA-256}(\text{sequence})[:20].\text{upper}() = \mathbf{A69EE28C5B2214A9AD99}$$
- Flag final: **`SIGMOID_A69EE28C5B2214A9AD99`**.

---

## 3. Detalii de implementare

1. **`solve.py` (Solver BM25 autonom și rapid):**
   - Parsează `corpus.jsonl` și `questions.json`.
   - Construiește documente tokenizate ponderate (`title * 6 + text`).
   - Aplică scorul BM25 cu bonusuri de intenție și statut.
   - Salvează ID-urile potrivite în `retrieved_ids.json` și apelează `derive_flag()`.

2. **`retrieval_pipeline.py` (Retriever dens E5 corectat):**
   - Logică de embedding reparată folosind un model unificat E5-small-v2.
   - Segmentare structurată și max-pooling al scorurilor la nivel de document.

3. **`flag_utils.py` (Validare și hashing de flag):**
   - Încarcă maparea `document_id` $\to$ caracter de verificare.
   - Implementează funcția `derive_flag(sequence)`.

4. **`diagnostics.json` și `retrieval_evidence.csv`:**
   - Evaluări cantitative de ablațiune:
     - Starter inițial nefuncțional (decalaj între encodere): 6/72 (8.3%)
     - TF-IDF pe text nefiltrat: 0/72 (0.0% — capturat în întregime de capcane)
     - E5 reparat cu chunk pooling: 72/72 (100.0%)
     - BM25 ponderat + filtre de intenție: 72/72 (100.0%)
   - `retrieval_evidence.csv` înregistrează ID-urile întrebărilor, titlurile, ID-urile documentelor asociate și caracterele de verificare.

---

## 4. Rezultate și artefacte
- **Flag final:** `SIGMOID_A69EE28C5B2214A9AD99`
- **Nivel de încredere:** Absolut (ridicat) — 72/72 potriviri exacte verificate atât pe solverul neural, cât și pe cel lexical; verificarea inițială confirmă `q001 -> TZ-01-1A`.
- **Artefacte-cheie (`task_tanzania/`):**
  - `answer.txt`: Conține `SIGMOID_A69EE28C5B2214A9AD99`.
  - `retrieved_ids.json`: Vectorul celor 72 de ID-uri autoritare selectate.
  - `retrieval_evidence.csv`: Tabel complet cu întrebări, titluri de documente, ID-uri și coduri de verificare.
  - `diagnostics.json`: Comparații de performanță între metodele de recuperare.
  - `solve.py`: Solver BM25 gata de producție cu euristici de intenție.
  - `retrieval_pipeline.py`: Pipeline funcțional de căutare densă neurală.
  - `flag_utils.py`: Utilitar pentru sume de control și generarea flag-ului.
  - `solution.md`: Documentație tehnică detaliată și analiza capcanelor.
