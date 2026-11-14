# Raport de soluție: task_mexico ("Ruta de las Identidades")

## 1. Prezentare generală & Obiectiv
- **ID sarcină:** `task_mexico` (Mexic).
- **Tip sarcină:** Rezoluția entităților / Legarea înregistrărilor (Entity Resolution / Record Linkage) între tabelele de evidență municipală.
- **Modul de trimitere:** Script autonom `main.py` evaluat offline.
- **Set de date:** `train-input.csv` (și cel ascuns `test-input.csv`) ce conține 19.520 înregistrări provenind din 3 surse independente:
  - Tabelul `a`: 6.609 înregistrări
  - Tabelul `b`: 6.446 înregistrări
  - Tabelul `c`: 6.465 înregistrări
  - Atribute: `source_table, record_id, first_name, last_name, address`.
- **Structura setului de referință (`train-output.json`):**
  - Conține 12.038 perechi potrivite etichetate, acoperind 10.000 de indivizi unici:
    - 4.484 clustere de dimensiune 2 (a-c: 1.549; a-b: 1.507; b-c: 1.428).
    - 2.518 clustere de dimensiune 3 (înregistrări prezente în toate cele trei tabele, generând câte $C_3^2 = 3$ muchii per pereche).
    - 2.998 înregistrări singulare.
  - **Invariant structural:** Cel mult o înregistrare per persoană în fiecare tabel sursă. Legătura pe perechi reprezintă o potrivire parțială 1-la-1 între tabele.
- **Contract de ieșire:** `main.py` citește fișierul CSV de intrare din argumentele CLI sau stdin/cwd, generează `test-output.json` și returnează `{"matches": [["a_1", "b_2"], ...]}`.

---

## 2. Observații cheie & Metodologie

1. **Asimetria zgomotului (Nume vs. Adrese):**
   - **Câmpurile de nume:** Puternic degradate de erori OCR, greșeli de tipar, inversări între prenume și nume, porecle, inițiale și trunchieri de prefixe (ex. `Mercedes` $\leftrightarrow$ `Lbarra`). Similitudinea fuzzy a numelor scade până la 0.50 în rândul perechilor reale, făcând numele inadecvate ca chei primare de blocare (blocking keys).
   - **Câmpurile de adrese:** Extrem de conservatoare; textul adresei suferă doar reformatări și abrevieri standard spaniole, păstrând multimea de bază a tokenilor semantici:
     - `C.jón` $\leftrightarrow$ `Callejón`, `Int.` $\leftrightarrow$ `Interior`, `Av.` $\leftrightarrow$ `Avenida`, `Peat.` $\leftrightarrow$ `Peatonal`, `Priv.` $\leftrightarrow$ `Privada`, `Depto.` $\leftrightarrow$ `Departamento`, `Edif.` $\leftrightarrow$ `Edificio`, `Prol.` $\leftrightarrow$ `Prolongación`, `Cor.` $\leftrightarrow$ `Corredor`, `Ret.` $\leftrightarrow$ `Retorno`, `Ampl.` $\leftrightarrow$ `Ampliación`, `Calz.` $\leftrightarrow$ `Calzada`, `Circ.` $\leftrightarrow$ `Circunvalación`, `And.` $\leftrightarrow$ `Andador`, `Blvd.` $\leftrightarrow$ `Boulevard`, `Q. ROO` $\leftrightarrow$ `QROO`.

2. **Diferența empirică de separabilitate a adreselor:**
   - Prin normalizarea canonică a tokenilor, **toate cele 12.038 perechi ground-truth împart multimulțimi identice de tokeni de adresă** (similitudine Jaccard $J = \mathbf{1.000}$).
   - Similitudinea maximă Jaccard pentru adrese în rândul tuturor celor 47.915 perechi candidate nepotrivite este de doar **0.417**.
   - Nu există două persoane distincte în același tabel sursă care să împartă aceeași adresă ($J \le 0.42$). Un prag de $J \ge 0.60$ garantează precizie de 100% și zero fals-pozitive.

---

## 3. Detalii de implementare

Implementat în `main.py` utilizând biblioteca standard Python (`sys`, `csv`, `json`, `re`, `unicodedata`, `collections`, `difflib`), fără dependențe externe:

1. **Normalizare canonică de text:**
   - Elimină diacriticele prin descompunere Unicode NFKD (`unicodedata.normalize('NFKD', ...)`).
   - Convertește la minuscule, transformă punctuația în spații și extinde abrevierile de adrese din spaniolă prin regex-uri compilate (ex. `c jon` $\to$ `callejon`, `q roo` $\to$ `qroo`).

2. **Motor de blocare cu chei multiple (Multi-Key Blocking):**
   - Perechile candidate sunt indexate pe mai multe chei redundante pentru a tolera zgomotul izolat:
     1. Cod poștal exact din 5 cifre (ZIP).
     2. Prefix de 4 cifre al codului poștal (protejează împotriva erorilor pe o singură cifră).
     3. Numărul casei + tokenul prefixului străzii.
     4. Tuplul exact al numelui complet.
     5. Tokeni rari de nume (frecvență în document $df \le 25$).
   - Produce 83.941 perechi candidate pe datele de antrenare, captând 100% dintre potrivirile reale.

3. **Scor & Ramură de salvare (Rescue Branch):**
   - Evaluează similitudinea Jaccard $J$ a tokenilor de adresă.
   - Dacă $J \ge 0.60$, perechea este acceptată necondiționat.
   - Pentru scoruri la limită ($0.20 \le J < 0.60$), o ramură de rezervă evaluează similitudinea Dice pe bigrame de caractere, scorul F1 fuzzy pe tokeni, rapoartele `SequenceMatcher` și acordul pe nume ($\ge 0.55$), cerând concordanță pe ancorele principale de adresă (codul poștal sau numărul casei + strada).

4. **Union-Find cu protecție la conflicte (Conflict-Guarded):**
   - Unește perechile candidate în ordinea descrescătoare a încrederii.
   - Impune o constrângere strictă: interzice unirea clusterelor ce conțin înregistrări din aceeași `source_table`.
   - Emite toate combinațiile de muchii pentru componentele de dimensiune 3 rezultate.

5. **Robustețe & Validare:**
   - Testat prin `stress_test.py` pe o corupere sintetică a 8% din înregistrări (greșeli de tipar, tokeni eliminați, coduri poștale modificate), recuperând 12.035 / 12.038 perechi reale cu 0 fals pozitive.
   - Cititorul robust de CSV gestionează BOM, CRLF, separatori personalizați și ordine variabilă a coloanelor.

---

## 4. Rezultate & Artefacte
- **Artefact de trimitere:** `main.py` (autonom, exclusiv biblioteca standard).
- **Performanță pe setul de antrenare (19.520 înregistrări):**
  - Potriviri recuperate: **12.038 / 12.038 (100.0%)**
  - Fals Pozitive: **0**
  - Fals Negative: **0**
  - **Scor F1: 1.0000**
- **Evaluare pe setul de test & Steag:**
  - Evaluat pe split-ul de test al platformei: **12.077 / 12.077 potriviri (100/100 puncte, PASSED)**.
  - **Steag:** **`SIGMOID_RECORD_LINKAGE`**
- **Timp de rulare:** $1.5 \dots 5.0$ secunde timp total de execuție pe un singur nucleu CPU.