# Raport de Soluție: task_cote_divoire („The Abidjan Transit Ledger”)

## 1. Prezentare Generală și Obiectiv
- **ID Sarcină:** `task_cote_divoire` (Coasta de Fildeș).
- **Dificultate și Scor:** Mediu, 200 / 200 puncte.
- **Domeniu:** Procesarea Limbajului Natural (NLP), Învățare Continuă (Continual Learning), Atenuarea Uitării Catastrofale (Catastrophic Forgetting Mitigation).
- **Context:**
  - Un router ușor de interogări, care clasifică intențiile utilizatorilor exprimate în limbaj natural în șabloane de interogare SQL, a fost ajustat fin (fine-tuned) pe un domeniu nou (**NEW task** — 5 tipuri de interogări spațiale/ferestre), provocând **uitarea catastrofală** a domeniului de bază (**OLD task** — 5 tipuri de interogări tranzacționale CRUD/join).
  - Punctele de control inițiale au prezentat o divergență totală:
    - `original_model.joblib`: $\text{Accuracy}_{\text{old}} = 1.00$, $\text{Accuracy}_{\text{new}} = 0.00$.
    - `finetuned_model.joblib`: $\text{Accuracy}_{\text{old}} = 0.00$, $\text{Accuracy}_{\text{new}} = 1.00$.
- **Criteriu de Promovare:** Modelul trebuie să depășească simultan o acuratețe de 85% pe ambele distribuții de sarcini pe setul de evaluare ascuns:
  $$\text{Accuracy}_{\text{old}} \ge 0.85 \quad \text{și} \quad \text{Accuracy}_{\text{new}} \ge 0.85$$
- **Spațiul Etichetelor Țintă (10 clase):**
  - *OLD task:* `OLD_JOIN_CUSTOMER_ORDERS`, `OLD_GROUPBY_REVENUE_BY_CATEGORY`, `OLD_CRUD_INSERT_ORDER`, `OLD_CRUD_UPDATE_INVENTORY`, `OLD_CRUD_DELETE_ACCOUNT`.
  - *NEW task:* `NEW_SPATIAL_NEAREST_WAREHOUSE`, `NEW_SPATIAL_WITHIN_RADIUS`, `NEW_WINDOW_LAG_DELIVERY_DELAY`, `NEW_WINDOW_LEAD_NEXT_SHIPMENT`, `NEW_WINDOW_PARTITION_RANK_FLEET`.
- **Format Trimitere:** Script autonom `main.py` care se execută într-un mediu izolat fără acces la rețea, preluând `test-input.json` și generând `predictions.json`.

---

## 2. Observații Cheie și Metodologie

### 2.1 Replay de Experiență / Buffer de Repetiție (Experience Replay / Rehearsal Buffer)
- Deoarece spațiile de etichete ale celor două sarcini erau complet disjuncte, niciunul dintre modelele preexistente nu putea fi utilizat direct.
- Seturile de date de antrenare au fost sintetizate într-un buffer unificat de repetiție:
  - Set public: `train-input.json` + `train-output.json` (250 exemple, câte 25 per clasă).
  - Exemple istorice de repetiție păstrate: `old_task_sample.json` (40 exemple, câte 8 per clasă veche).
  - După deduplicarea textului, s-a stabilit un set de 279 de interogări adnotate unice.

### 2.2 Unire de Caracteristici cu Granularitate Duală și SVM Liniar
- **Reprezentări de Text (`FeatureUnion`):**
  1. *N-grame de cuvinte:* `(1, 2)`-grame cu scalare sublineară a frecvenței termenilor.
  2. *N-grame de caractere:* `(3, 5)`-grame la limitele cuvintelor (`char_wb`). Captează afixele morfologice și rădăcinile verbelor (`join`, `insert`, `delete`, `nearest`, `radius`, `lag`, `partition`) indiferent de denumirile entităților sau variațiile de exprimare.
- **Clasificator:** `LinearSVC(C=1.0)` cu funcție de pierdere squared hinge. Se antrenează în $\approx 1.2$ secunde pe CPU, menține granițe decizionale deterministe și necesită un consum minim de memorie.

### 2.3 Implementare Autonomă
Setul de date de repetiție de 279 de eșantioane este comprimat cu `gzip` și codificat în base64 direct în `main.py`. La pornirea containerului, scriptul decomprimă datele în memorie, antrenează pipeline-ul dublu TF-IDF + `LinearSVC` în sub 2 secunde și realizează inferența pe `test-input.json`.

---

## 3. Detalii de Implementare

1. **`solve.py`:**
   - Îmbină seturile publice de antrenament și `old_task_sample.json`.
   - Construiește și antrenează pipeline-ul `FeatureUnion` + `LinearSVC`.
   - Salvează artefactele intermediare (`vectorizer.joblib`, `recovered_model.joblib`, `embedded_data.json`).
2. **`eval_cv.py` (Protocol de Validare):**
   - Validarea încrucișată 5-fold `StratifiedKFold` pe setul combinat de 279 eșantioane a atins acuratețea de **1.000 / 1.000** pe ambele sarcini.
   - Testul de generalizare pe `old_task_sample.json` la antrenarea exclusivă pe cele 250 de exemple de bază a obținut **40 / 40 (1.00)**.
   - Testele de robustețe la perturbarea entităților parafrazate au obținut **10 / 10 (1.00)**.
3. **`build_main.py`:**
   - Împachetează dicționarul comprimat și logica de șablon în `main.py`.
4. **`main.py`:**
   - Script autonom de trimitere: decomprimă datele, antrenează în RAM și generează `predictions.json`.

---

## 4. Rezultate și Artefacte
- **Trimitere pe Platformă:** `passed` (ID Trimitere: 1079)
- **Scor:** `200 / 200`
- **Acuratețe pe Testul Ascuns:**
  - `old-task accuracy`: **0.9733** (țintă $\ge 0.85$) ✅
  - `new-task accuracy`: **0.9867** (țintă $\ge 0.85$) ✅
- **Flag:** `SIGMOID_REHEARSAL`
- **Livrabile:**
  - `main.py`: Pipeline autonom de antrenare și inferență.
  - `solve.py`: Script de antrenare prin repetiție și generator de artefacte.
  - `eval_cv.py`: Modul de validare încrucișată și evaluare a robusteții.
  - `writeup.md`: Narativ complet al soluției.
