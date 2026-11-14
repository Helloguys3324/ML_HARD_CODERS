# Raport de soluție: task_morocco ("The Dispatch Desk of Fez")

## 1. Prezentare generală & Obiectiv
- **ID sarcină:** `task_morocco` (Maroc, LMML 2026).
- **Scor & Status:** **200 / 200 puncte (PASSED)**.
- **Steag deblocat:** **`SIGMOID_ROUTING`**.
- **Tip sarcină:** Rutare LLM sub constrângeri de calitate și buget (LLM Routing under Quality and Budget Constraints); trimis sub formă de script autonom Python `main.py`.
- **Set de date:** `test-input.csv` (`request_id, prompt`) conținând 500 de interogări ascunse. Setul de antrenare conține 400 de interogări (`train-input.csv`) însoțite de răspunsurile reale și ieșirile ambelor modele candidate.
- **Nivelurile LLM vizate:**
  - Model ieftin: Qwen2.5-0.5B (Cost: 1 dirham/interogare).
  - Model scump: Qwen2.5-7B (Cost: 25 dirhami/interogare).
- **Criterii duale de promovare (Dual-Gate):**
  1. **Calitate:** Acuratețea față de ground truth trebuie să fie **$\ge 0.80$** (80%).
  2. **Buget:** Costul total pe cele 500 de interogări de test trebuie să fie **$\le 10.000$ dirhami** (medie $\le 20$ dirhami/interogare, permițând cel mult 395 de apeluri către modelul scump).
- **Formatul de ieșire:** Tabel de decizii de rutare `routing_log.csv` (`request_id, model`).

---

## 2. Observații cheie & Metodologie

1. **Profilarea performanței modelelor (400 de interogări de antrenare):**
   | Tip interogare | Număr mostre ($n$) | Acuratețe Ieftin | Acuratețe Scump | Câștig net ($\Delta = \text{exp} - \text{cheap}$) |
   |---|---|---|---|---|
   | `crates_remain` (aritmetică de inventar în mai mulți pași) | 24 | 0% | 100% | **+1.00** |
   | `toll` (calcule de taxe vamale condiționate) | 66 | 9% | 97% | **+0.88** |
   | `takings_minmax` (căutare extreme de venituri) | 39 | 15% | 100% | **+0.85** |
   | `time` (aritmetică de orare cu escale) | 66 | 14% | 82% | **+0.68** |
   | `board_indirect` (căutare pe avizier în doi pași) | 22 | 55% | 100% | **+0.45** |
   | `takings_fee` ("most" — comision maxim) | 13 | 23% | 69% | **+0.46** |
   | `walk_direct` (cel mai scurt drum este muchia directă) | 37 | 51% | 89% | **+0.38** |
   | `crates_total` (înmulțire într-un singur pas) | 43 | 70% | 100% | **+0.30** |
   | `takings_fee` ("least" — comision minim) | 15 | 27% | 40% | **+0.13** |
   | `walk_indirect` (cel mai scurt drum necesită ocolire) | 30 | 3% | 10% | **+0.07** |
   | `board_direct` (căutare directă de șiruri pe avizier) | 45 | 100% | 100% | **0.00** |

2. **Analiza grafică Dijkstra în cadrul ruterului:**
   - Pentru interogările de deplasare pietonală între cartierele din Fez, ruterul parsează muchiile orașului și timpii de tranziție direct din textul promptului și execută algoritmul lui Dijkstra.
   - Dacă muchia directă este într-adevăr ruta cea mai scurtă (`walk_direct`), modelul scump reușește în mod fiabil (89%).
   - Dacă un ocol indirect este mai scurt (`walk_indirect`), ambele modele eșuează aproape invariabil (10% vs. 3%). Alocarea a 25 de dirhami pentru interogări de mers indirect irosește bugetul pentru un câștig neglijabil; rutarea lor către modelul ieftin conservă bugetul pentru sarcini aritmetice.

3. **Alocarea optimă a bugetului (Rucsac / Knapsack):**
   - Fiecărei interogări i se atribuie o valoare de utilitate așteptată:
     $$\text{Value} = P(\text{correct} \mid \text{expensive}) - P(\text{correct} \mid \text{cheap})$$
   - Fie $E$ numărul de apeluri către modelul scump:
     $$\text{Cost} = 25E + (N - E) = N + 24E \le 20N \iff E \le \left\lfloor \frac{19N}{24} \right\rfloor$$
   - Pentru $N = 500$:
     $$E \le \left\lfloor \frac{9500}{24} \right\rfloor = 395 \implies \text{Cost} = 25 \times 395 + 105 = 9.980 \le 10.000$$
   - Interogările sunt sortate descrescător după `Value`: primele 395 de interogări sunt atribuite lui `expensive`, iar ultimele 105 (toate cele din `board_direct` și `walk_indirect`) sunt rutate către `cheap`.

---

## 3. Detalii de implementare

Implementat în `main.py` folosind biblioteca standard Python (`csv`, `re`, `heapq`, `collections`):

1. **Parsarea grafului & Motorul Dijkstra (`_walk_is_direct_shortest`):**
   - Parsează originea și destinația (`from <A> to <B>`).
   - Extrage muchiile prin regex-ul `([A-Za-z]+)-([A-Za-z]+)\s+(\d+)\s+minutes`.
   - Calculează cel mai scurt drum prin Dijkstra cu `heapq` și compară lungimea muchiei directe cu distanța optimă.

2. **Clasificator de interogări bazat pe regex & Calibrarea priorităților:**
   - Clasifică interogările și atribuie valori calibrate:
     - `crates_remain`: 1.00
     - `toll`: 0.88
     - `takings_minmax`: 0.85
     - `time`: 0.68
     - `board_indirect`: 0.46
     - `takings_fee_most`: 0.45
     - `walk_direct`: 0.38
     - `crates_total`: 0.30
     - `takings_fee_least`: 0.12
     - `walk_indirect`: 0.07
     - `board_direct`: 0.00

3. **Rutare în lot (Batch) & Generarea rezultatului:**
   - Încarcă `test-input.csv`, sortează după scorul de prioritate, etichetează primele 395 ca `expensive` și ultimele 105 ca `cheap`, și scrie `routing_log.csv`.

---

## 4. Rezultate & Artefacte
- **Script de trimitere:** `main.py` (determinist, zero dependențe externe).
- **Tabel de ieșire:** `routing_log.csv` (500 de decizii).
- **Evaluare oficială pe setul de test ascuns (500 de interogări):**
  - **Scor de calitate:** **0.8520 (85.2%)** (Prag de promovare: $\ge 0.80$).
  - **Cost total:** **9.980 dirhami** (Plafon bugetar: $\le 10.000$; 395 scumpe, 105 ieftine).
  - **Punctaj:** **200 / 200 puncte (PASSED)**.
  - **Steag:** **`SIGMOID_ROUTING`**.
- **Validare locală (400 de interogări de antrenare):**
  - Calitate = 84.75%, Cost = 7.984 dirhami (limită: 8.000).