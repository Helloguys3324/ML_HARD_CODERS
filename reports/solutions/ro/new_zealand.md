# Raport de Soluție: task_new_zealand ("The Braided River Run")

## 1. Prezentare Generală și Obiectiv
- **Task ID:** `task_new_zealand` (Noua Zeelandă).
- **Dificultate și Scor:** Hard, 300 / 300 puncte.
- **Domeniu:** Reinforcement Learning / Găsirea Deterministă a Căii pe Grafuri (A* Search).
- **Format Ieșire:** Șir de traiectorie a mișcărilor în `solution.txt`, deblocând flag-ul `SIGMOID_<NAME>`.
- **Premisă:**
  - Într-o vale cu albii de râu împletite reprezentată pe o grilă de $47 \times 61$, un agent trebuie să navigheze de la coordonata de start `S` (stânga-sus) până la cabana pădurarului `G` (dreapta-jos), traversând cinci canale de râu periculoase.
  - Pădurarul deține recordul traseului de 297 de minute. Obiectivul este calcularea unei rute admisibile, fără coliziuni, care ajunge la `G` în **$\le 296$ minute** (depășind recordul pădurarului).
  - Mecanica mișcării:
    - Spațiu discret de acțiuni: `U` (sus), `D` (jos), `L` (stânga), `R` (dreapta) și `W` (așteptare).
    - Cost de bază per pas: 1 minut pentru deplasare sau așteptare.
    - Celule de mlaștină (`,`): adaugă o penalizare de `swamp_cost = 3` minute (total 4 minute).
    - Variații de altitudine: urcarea în pantă adaugă `climb_cost = 1` minut per metru de diferență de nivel pozitivă ($\max(0, h_{\text{new}} - h_{\text{curr}})$).
    - Obstacole impracticabile: stânca `#` și apa adâncă `=` nu pot fi traversate.
    - Ciclul mareelor: vadurile (`~`) sunt practicabile exclusiv în timpul refluxului: $t \pmod{16} < 5$.
    - Poduri suspendate (`A`..`I`): toate podurile pornesc ridicate (impracticabile), cu excepția podului `I`. Călcarea pe stațiile cu troliu (`a`..`l`) comută starea podurilor asociate prin diferență simetrică (XOR).

---

## 2. Perspective Cheie și Metodologia de Căutare
- **Extinderea Spațiului Stărilor:**
  - Mediul determinist este modelat ca o problemă de drum minim pe un graf extins de stări, unde fiecare nod este definit prin:
    $$\text{State} = (\text{cell\_index}, \text{tide\_phase}, \text{bridge\_mask})$$
    unde:
    - $\text{cell\_index} \in [0, N_{\text{passable}}-1]$ ($1{,}146$ celule traversabile dintr-un total de $2{,}867$ celule ale grilei);
    - $\text{tide\_phase} = t \pmod{16} \in [0, 15]$ (16 faze discrete ale ciclului mareic);
    - $\text{bridge\_mask} \in [0, 2^8 - 1]$ (mască de 8 biți pentru podurile dinamice `A`..`H`, podul `I` fiind deschis static, generând 256 de configurații).
  - Cardinalitatea totală a spațiului stărilor:
    $$|\mathcal{S}| = 1{,}146 \times 16 \times 256 \approx \mathbf{4{,}694{,}016}\text{ stări}$$
- **Căutare Optimă prin A*:**
  - Tranzițiile depind strict de $(\text{cell}, \text{phase}, \text{mask})$, garantând costuri deterministe ale muchiilor și permițând o căutare exactă $A^*$ cu margini uniforme de cost.
- **Euristică Monotonă Admisibilă:**
  - Precalculează distanțele Breadth-First Search (BFS) invers de la celula țintă `G` pe o grilă simplificată unde toate celulele non-stâncă și non-apă costă 1 minut (ignorând mareele, podurile, mlaștinile și altitudinea).
  - Aceasta oferă o limită inferioară strict admisibilă și consistentă $h(s) \le h^*(s)$ pentru timpul rămas până la `G`.

---

## 3. Detalii de Implementare
Rezolvitorul este implementat în `solve.py` (harness de verificare) și `main.py` (rezolvitor de producție autonom, fără dependențe externe):

1. **Parsarea Hărții și Indexarea Grafului (`parse_map` în `main.py`):**
   - Parsează `test-input.txt`, extrăgând limitele grilei, tokenii de teren, matricea altitudinilor, parametrii mareei (`period=16, low=5`), podurile deschise inițial și tabelele de conectare troliu-poduri.
   - Atribuie indici întregi unici tuturor celulelor traversabile și precalculează listele de adiacență ale vecinilor.
2. **Precalcularea Euristicii prin BFS Invers:**
   - Evaluează distanțele minime în pași $h_{\text{dist}}[r, c]$ de la `G` utilizând `collections.deque`.
3. **Structură de Memorie de Înaltă Performanță:**
   - Pentru a evita overhead-ul dicționarelor și alocarea excesivă de obiecte Python pentru 4.7M stări potențiale, distanțele și tranzițiile sunt stocate în tablouri 1D C compacte prin `array('i', [-1] * TOTAL_STATES)`:
     - `dist`: înregistrează timpul minim scurs pentru a atinge fiecare stare extinsă;
     - `parent_state`: indexul stării predecesoare pentru reconstituirea rutei;
     - `parent_action`: caracterul acțiunii (`U`, `D`, `L`, `R`, `W`).
   - Coada de priorități este gestionată prin `heapq` cu cheia de prioritate $f = g + h$.
4. **Dinamica Tranzițiilor și Evaluarea Constrângerilor:**
   - Evaluează `W` (așteptare) și pașii direcționali `['U', 'D', 'L', 'R']`.
   - Validează condiția de reflux $(t + 1) \pmod{16} < 5$ pentru vadurile `~`.
   - Verifică starea măștii de biți pentru podurile `A`..`H`.
   - Actualizează `bridge_mask` prin XOR pe biți la pășirea pe celulele cu troliu `a`..`l`.
5. **Reconstrucția Traseului:**
   - La extragerea din coadă a oricărei stări situate la celula `G`, algoritmul se oprește imediat și reconstruiește secvența de acțiuni.

---

## 4. Rezultate și Artefacte
- **Status:** PASSED
- **Scor:** 300 / 300 puncte
- **Flag Recuperat:** `SIGMOID_BELLMAN`
- **Performanță și Verificare:**
  - **Total Acțiuni:** **246 pași**;
  - **Timp Scurs:** **293 minute** (cu 4 minute sub bugetul record de 297 de minute);
  - **Timp de Execuție al Rezolvitorului:** $\approx 0.5\text{ secunde}$ în Python standard library.
- **Artefacte Cheie:**
  - `solution.txt` — șir de traseu verificat de 246 de caractere;
  - `main.py` — rezolvitor autonom rapid;
  - `solve.py` — script de cercetare cu validare prin simulator local.
