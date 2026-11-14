# Raport de soluție: task_switzerland ("The Weak Layer")

## 1. Prezentare generală și obiectiv
- **Task ID:** `task_switzerland` (Elveția).
- **Dificultate:** Ușor (100 puncte).
- **Tip de trimitere:** Script executabil `main.py` evaluat offline într-un container automatizat de notare.
- **Contract și cerințe:**
  - `main.py` citește `test-input.csv` din directorul de lucru curent.
  - Antrenează un model scikit-learn `DecisionTreeClassifier` cu hiperparametri impliciți și un `random_state` întreg.
  - Extrage cuvântul secret din **structura internă a arborelui de decizie antrenat** (predicțiile modelului nu oferă informații utile).
  - Scrie flag-ul `SIGMOID_<WORD>` în `test-output.txt` în directorul curent (o singură linie cu majuscule, fără ghilimele).
- **Premisa domeniului:**
  - Simularea declanșării avalanșelor („Stratul slab” / "The Weak Layer").
  - Arborele de decizie atinge trivial o acuratețe de 1.000; payload-ul secret este codificat exclusiv în secvența de împărțiri (splits) ale atributelor.
- **Obiectiv:** Construirea unui script autonom `main.py`, fără dependențe externe, care să reconstruiască arborele de decizie, să urmărească calea până la clasa pozitivă de avalanșă, să extragă inițialele atributelor de split și să genereze flag-ul.

---

## 2. Observații-cheie și metodologie

### 2.1 Topologia arborelui de antrenare (`train-input.csv`)
- Pe fișierul `train-input.csv`, variabila țintă `slab_released` este perfect separată de un arbore cu adâncimea 4, format din 9 noduri în total și 5 frunze (acuratețe: 1.000):
  - **Nodul 0:** `depth_hoar_pct <= 20` ($n=6000$) $\to$ Stânga: Frunza 1 (clasa 0, $n=1400$); Dreapta: Nodul 2 ($n=4600$).
  - **Nodul 2:** `elevation_m <= 1850` ($n=4600$) $\to$ Stânga: Frunza 3 (clasa 0, $n=1100$); Dreapta: Nodul 4 ($n=3500$).
  - **Nodul 4:** `mean_grain_roundness <= 0.55` ($n=3500$) $\to$ Stânga: Nodul 5 ($n=2650$); Dreapta: Frunza 8 (clasa 0, $n=850$).
  - **Nodul 5:** `overburden_load_kpa <= 1.5` ($n=2650$) $\to$ Stânga: Frunza 6 (clasa 0, $n=650$); Dreapta: Frunza 7 (clasa 1, avalanșă declanșată, $n=2000$).

### 2.2 Regula de decodificare
- Secvența atributelor de split pe calea de la rădăcină la frunza pozitivă de avalanșă (clasa 1):
  1. **`depth_hoar_pct`** $\to$ **D**
  2. **`elevation_m`** $\to$ **E**
  3. **`mean_grain_roundness`** $\to$ **M**
  4. **`overburden_load_kpa`** $\to$ **O**
- Combinarea primelor litere ale fiecărui atribut de-a lungul căii de decizie produce:
  $$\text{D} \to \text{E} \to \text{M} \to \text{O} \quad \Longrightarrow \quad \mathbf{DEMO} \quad \Longrightarrow \quad \mathbf{SIGMOID\_DEMO}$$
  Aceasta corespunde perfect cu fișierul de referință `train-output.txt`.
- Parcurgerile alternative (in-order, post-order, sortate după importanța atributelor sau pragul de split) produc anagrame lipsite de sens (`DOM`, `OMED`, `MODE`, `EDOM`). Doar calea de la rădăcină la frunza țintă formează un cuvânt coerent în limba engleză.

---

## 3. Detalii de implementare

1. **`main.py` (Artefactul trimis pentru notare):**
   - **Încărcare rezilientă a datelor:** Caută `test-input.csv` în directorul curent, cu fallback automat la `train-input.csv` pentru testarea locală.
   - **Identificarea coloanei țintă:** Detectează coloana țintă folosind candidații `["slab_released", "released", "target", "label", "y"]` sau alege implicit ultima coloană.
   - **Antrenarea modelului:** Antrenează `DecisionTreeClassifier(random_state=42)` fără limite artificiale de adâncime sau frunze pentru a păstra fidel topologia split-urilor.
   - **Localizarea frunzei țintă:** Scanează frunzele (`children_left[i] == -1` și `children_right[i] == -1`) pentru a găsi frunza cu numărul maxim de instanțe ale clasei pozitive (`clasa 1`).
   - **Urmărirea căii (Path Tracing):** Parcurge recursiv nodurile interne de la rădăcină (0) până la frunza țintă, extrăgând prima literă cu majusculă a fiecărui atribut de split:
     ```python
     feat_name = feature_names[tree.feature[current]]
     return [feat_name[0].upper()] + p
     ```
   - **Mecanism de rezervă (Fallback):** Dacă urmărirea căii eșuează, recurge la o traversare pre-order pe toate nodurile interne (`children_left[i] != -1`).
   - **Generarea ieșirii:** Scrie `SIGMOID_<WORD>\n` în fișierul `test-output.txt`.

2. **`verify_practice.py`:**
   - Antrenează arborele pe `train-input.csv`, randează reprezentarea ASCII a arborelui și validează egalitatea strictă cu `train-output.txt` (`SIGMOID_DEMO`).

3. **`stress.py` și dovezi de robustețe (`evidence/`):**
   - `stress_subsample_shuffle.txt`: Confirmă stabilitatea arborelui pe valori ale lui `random_state` de la $0$ la $100$, sub-eșantionare aleatorie a rândurilor (50% până la 95%) și amestecare arbitrară a rândurilor — toate producând arborele identic și cuvântul `DEMO`.
   - `library_versions.txt`: Validează execuția sub Python 3.13.17, scikit-learn 1.7.2, pandas 2.3.3 și numpy 2.2.6.

---

## 4. Rezultate și artefacte
- **Artefact de trimitere:** `main.py` (autonom, stack standard scikit-learn).
- **Flag de validare (setul de antrenare):** `SIGMOID_DEMO`
- **Artefacte-cheie (`task_switzerland/`):**
  - `main.py`: Scriptul final conform constrângerilor impuse de platformă.
  - `solution.md`: Documentație tehnică a topologiei arborelui și a logicii de parcurgere.
  - `verify_practice.py`: Test automatizat de verificare pe datele de practică.
  - `stress.py`: Cadru de testare a stabilității la permutări și sub-eșantionare.
  - `test-output.txt`: Rezultatul generat local (`SIGMOID_DEMO`).
  - `answer.txt`: Marcator de verificare (`SIGMOID_DEMO`).
  - `evidence/`: Jurnale ce confirmă invarianța împărțirilor indiferent de seed-ul ales.
