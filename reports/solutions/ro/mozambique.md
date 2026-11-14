# Raport de soluție: task_mozambique ("The Professor of Maputo Bay")

## 1. Prezentare generală & Obiectiv
- **ID sarcină:** `task_mozambique` (Mozambic).
- **Tip sarcină:** Strategie multi-agent pe runde / Planificare logistică în condiții de competiție adversă (Turn-Based Multi-Agent Strategy / Logistics Planning under Adversarial Competition).
- **Modul de trimitere:** Script autonom de agent în Python `main.py`.
- **Mediu & Mecanici de joc:**
  - Se desfășoară pe durata a 80 de runde pe un graf conex de porturi din Golful Maputo între agentul jucătorului și un oponent determinist („Profesorul”).
  - Comunicarea se realizează prin binarul `mozambique-engine` și modulul `arena.py` prin JSON-RPC / stdio.
  - În fiecare rundă, agentul transmite o comandă către o ambarcațiune liberă. Capacitatea de încărcare a unei ambarcațiuni este de 3 unități.
  - Tranzitul între porturile $i$ și $j$ consumă un număr fix de runde `routes[i][j]`.
  - Rolurile porturilor:
    - Rol `0`: Recoltare cherestea / Timber ($T$).
    - Rol `1`: Recoltare fibre / Fibre ($F$).
    - Rol `2`: Atelier de producție (fabrică bunuri finite, tip 2).
    - Portul de bază acasă (`homes`).
  - Economie & Producție:
    - Construirea unui atelier (`build_workshop`) și inițierea producției (`craft`) consumă ambele câte 2 cherestea și 1 fibră depozitate la depozitul fabricii. Producția durează 3 runde.
    - Livrarea (`deliver`) îndeplinește contracte:
      - Contracte de materie primă (Raw): Necesită 2 unități brute; recompensă **5 puncte**.
      - Contracte de produse prelucrate (Crafted): Necesită 1 bun finisat; recompensă **12 puncte**.
      - Contractele au termene limită și expiră dacă nu sunt onorate.
- **Condiție de victorie:**
  - Fondul total de puncte disponibile este fixat la 68 de puncte (4 brute @ 5 pct + 4 prelucrate @ 12 pct: $4 \times 5 + 4 \times 12 = 68$).
  - Câștigarea necesită depășirea scorului Profesorului (rată de victorie $\ge 65\%$ într-o serie worst-of-5) pentru deblocarea steagului `SIGMOID_<WORD>`.

---

## 2. Observații cheie & Metodologie

1. **Aritmetica punctelor & Căi spre victorie:**
   - Deoarece totalul punctelor este de 68, victoria se poate obține pe două căi:
     - **Strategia celor 3 bunuri prelucrate:** Livrarea a 3 bunuri finite ($3 \times 12 = 36$ puncte), lăsând Profesorul cu cel mult $68 - 36 = 32$ puncte, garantând victoria ($36 > 32$).
     - **Strategia hibridă:** Livrarea a 2 bunuri finite și a cel puțin 3 contracte brute ($2 \times 12 + 3 \times 5 = 39$ puncte vs. max 29).
   - Profesorul execută o producție agresivă timpurie (prima livrare în rundele 18–31) și obține 39–68 puncte dacă este lăsat nestingherit.

2. **Strategia algoritmică (`main.py`):**
   - **Cele mai scurte drumuri pentru toate perechile:** Algoritmul Floyd–Warshall calculează `dist[i][j]` și tabela următorului pas `nxt[i][j]` în timpul inițializării.
   - **Plasarea optimă a fabricii:** Selectează nodul fabricii $F$ care minimizează distanța ponderată de tranzit:
     $$\text{Cost}(F) = \text{dist}(\text{home}, F) + 0.7 \cdot (\text{dist}_{T} + \text{dist}_{\text{fib}}) + 0.35 \cdot (\text{dist}_{C_1} + \text{dist}_{C_2})$$
   - **Deschidere coordonată:** Portul de bază pornește cu $5T + 3F$. Ambele ambarcațiuni încarcă $2T + 1F$ la bază (4 runde), navighează simultan către fabrica $F$, descarcă materialele, construiesc atelierul (`build_workshop`) și încep producția primului bun (`craft`).
   - **Buclă de aprovizionare și livrare navetă:** Ambarcațiunile fac naveta între porturile de materii prime și atelier pentru a menține activ fluxul de producție. De îndată ce un bun este produs, o ambarcațiune liberă îl livrează imediat către cel mai apropiat contract prelucrat valid cu cel mai scurt termen limită.
   - **Refuz oportunist al punctelor & Prevenirea blocajelor (Deadlock):** Preia contracte brute pentru a refuza puncte oponentului când linia de fabricație este tamponată. Încărcătura nelivrată este descărcată înapoi la fabrică pentru a preveni blocarea magaziilor navelor.

---

## 3. Detalii de implementare

Implementat în `main.py` (17 KB) folosind biblioteca standard Python, complet compatibil cu PyPy-3.10:

1. **Logica agentului (`main.py`):**
   - `prep(obs, memory)`: Precalcularea drumurilor prin Floyd–Warshall, punctarea nodurilor de fabrică și indexarea contractelor (`craft_idx`, `raw_idx`).
   - `agent(obs, memory)`: Automat cu stări ierarhic care generează comenzi atomice: `sail`, `gather`, `load`, `unload`, `build_workshop`, `craft` și `deliver`.
   - `serve()`: Buclă robustă JSON-RPC / stdio care gestionează comunicarea cu motorul.

2. **Module de testare și evaluare:**
   - `solve.py`: Rulaj de evaluare pe hărți multiple pe 24 de semințe (seeds) folosind modulul oficial `arena.py`.
   - `harness.py`: Apelator de arenă în cadrul aceluiași proces pentru profilare și identificarea blocajelor.
   - `eval.py`: Evaluator turneu multi-threaded în sistem worst-of-5.
   - `agent_dev.py`, `agent_v3.py`: Iterări incrementale ale logicii de agent.

---

## 4. Rezultate & Artefacte
- **Script de trimitere:** `main.py` (agent autonom PyPy/CPython).
- **Indicator răspuns:** `answer.txt` indicând către `FILE_SUBMISSION:main.py`.
- **Metrici de rulare:** Latența de execuție a unei runde este de $0.14 \dots 0.35$ s pe CPython (mai rapid pe PyPy), respectând cu ușurință limitele de timp ale motorului (`within_time_limit: true`).
- **Evaluarea meciurilor:**
  - Execută consecvent deschiderea, construiește atelierul și livrează 2 contracte prelucrate (24 puncte).
  - Obține victorii 36:32 pe configurații favorabile de hartă prin 3 livrări prelucrate.
  - În serii randomizate mai largi, Profesorul obține în medie 34–44 puncte prin colectarea paralelă de contracte brute, menținând rata globală de victorii sub ținta de 65% (încredere: **LOW**).
- **Direcții viitoare:** Introducerea unei planificări anticipative de tip beam-search pentru a prezice și intercepta activ rutele de colectare a contractelor brute ale Profesorului.