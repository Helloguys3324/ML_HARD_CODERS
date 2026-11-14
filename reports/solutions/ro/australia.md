# Raport de Soluție: task_australia („The Drunken Mario Adventures”)

## 1. Prezentare Generală și Obiectiv
- **ID Sarcină:** `task_australia` (Australia).
- **Dificultate și Scor:** Medie, 200 / 200 puncte.
- **Domeniu / Track:** Inginerie Inversă pe Binare (Binary Reverse Engineering), Criminalistică în Învățarea prin Consolidare (Reinforcement Learning Forensics).
- **Context:** O platformă experimentală de neuroevoluție („Marathon Lab”) bazată pe un joc de tip platformer 2D în stil Mario.
- **Artefact:** Binar ELF compilat de sine stătător `marathon-lab-linux-x86_64` (27.990.042 octeți; SHA-256: `569dd52d9a4ff551798caff32c7eb3a1f5e462e14225d93f55a42dff25f2eca4`).
- **Anomalie:** Politica de referință antrenată (`lab/proven.npy`) navighează nivelul ocolind deliberat majoritatea monedelor vizuale și colectând doar un subset specific, deși toate monedele par vizual identice.
- **Obiectiv:** Aplicarea ingineriei inverse asupra mecanicii mediului, funcțiilor de recompensă și comportamentului politicii pentru a recupera flagul secret.
- **Format Flag:** `SIGMOID_<WORD>`.

---

## 2. Perspective Cheie și Metodologie

1. **Decompresia Containerului Nuitka:**
   Inspectarea sfârșitului fișierului binar a evidențiat un pachet Nuitka onefile ce conținea o sarcină utilă comprimată cu zstd (semnătură `KA`). Despachetarea a extras 448 de fișiere de aplicație, inclusiv executabilul ELF de bază `marathon-lab.bin` și ponderile modelului de referință `lab/proven.npy`.
2. **Introspecție la Rulare prin `LD_PRELOAD`:**
   Deoarece modulele au fost compilate în cod nativ C, interceptarea dinamică a simbolurilor a fost implementată prin injectarea unei componente C în proces folosind `LD_PRELOAD`. Apelurile API-ului Python C (`PyGILState_Ensure`, `PyRun_SimpleString`) au expus starea internă a modulelor `lab.level` și `lab.fitness`.
3. **Descoperirea Asimetriei de Recompensă:**
   Nivelul conține 20 de monede dispuse de-a lungul liniei $y=11$. Evaluatorul de fitness `lab.fitness` păstrează o tabelă ascunsă `_SIGN`:
   - 7 monede au `_SIGN[(x, 11)] == 1`, acordând o recompensă de $+25.0$;
   - 13 monede au `_SIGN[(x, 11)] == 0`, aplicând o penalizare de $-25.0$.
4. **Verificarea Politicii:**
   Executarea `lab/proven.npy` de-a lungul a 1.197 de pași de simulare (scor de fitness: 514.21) a confirmat că politica colectează cu exactitate cele 7 monede pozitive `[(7,11), (17,11), (80,11), (101,11), (110,11), (118,11), (172,11)]`, evitând toate cele 13 monede cu penalizare.
5. **Decodificarea Cifrului A1Z26:**
   Ordonarea semnelor celor 20 de monede cronologic de la stânga la dreapta produce 4 numere întregi pe 5 biți:
   - Monedele 1–5 ($x \in [7, 12, 17, 25, 32]$): `10100` $\to 20 \to$ **T**
   - Monedele 6–10 ($x \in [43, 61, 66, 73, 80]$): `00001` $\to 1 \to$ **A**
   - Monedele 11–15 ($x \in [93, 101, 110, 118, 130]$): `01110` $\to 14 \to$ **N**
   - Monedele 16–20 ($x \in [157, 172, 195, 201, 206]$): `01000` $\to 8 \to$ **H**

Cuvântul decodificat este **`TANH`**, rezultând flagul `SIGMOID_TANH`.

---

## 3. Detalii de Implementare Pas cu Pas

### Date de Introspecție a Modulelor
- **Constante `lab.level`:**
  - `WIDTH = 212`, `HEIGHT = 15`, `COIN_ROW = 11`.
  - `COIN_SITES = [7, 12, 17, 25, 32, 43, 61, 66, 73, 80, 93, 101, 110, 118, 130, 157, 172, 195, 201, 206]`.
- **Parametri `lab.fitness`:**
  - `_BASIS = 919247`, `PROGRESS = 1.0`, `TIME_COST = 0.01`, `COIN_WEIGHT = 25.0`, `WIN = 150.0`.
  - Tabela ascunsă de căutare `_SIGN` care asociază semne binare $\{0, 1\}$ fiecărei coordonate `(x, y)`.

### Unelte de Verificare (`task_australia/`)
- `probe_runtime.py` / `probe_reward_events.py`: Extrage tabelele modulelor și interceptează tranzițiile de recompensă la nivel de pas.
- `solve.py`: Reconstruiește vectorul de 20 de biți, verifică logica împărțirii pe biți și afișează șirul decodificat al flagului.
- `evidence_*.txt`: Loguri brute de execuție ce validează traiectoria de 1.197 de pași și evenimentele de colectare a monedelor.

---

## 4. Rezultate și Artefacte
- **Stare:** `PASSED` (Scor: **200 / 200** puncte).
- **Tentativă de Trimitere:** Tentativa 1 (`2026-11-14T12:27:02.112Z`, trimis de: `Kozak`).
- **Flag Recuperat:**
  ```text
  SIGMOID_TANH
  ```
- **Artefacte Cheie:**
  - `answer.txt` (`SIGMOID_TANH`);
  - `solution.md` (raport detaliat de analiză criminalistică);
  - `solve.py` (script decodificator reproductibil);
  - `coin_signs.csv` (tabel de mapare coordonate-biți);
  - `evidence_module_dump.txt` și `evidence_reward_events.txt`.
