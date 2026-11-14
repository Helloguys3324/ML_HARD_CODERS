# Raport de Soluție: task_fiji („Walu Season”)

## 1. Prezentare Generală a Problemei
- **Sarcină și Domeniu (Track):** Ușoară (Easy, 100 pct). Problemă de bandiți cu mai multe brațe (multi-armed bandits) desfășurată pe parcursul unui sezon de pescuit de 100 de zile în Kadavu (apele *qoliqoli*). În fiecare zi sunt disponibile 10 locuri și 1.000 de aruncări.
- **Dinamica Mediului:**
  - Maparea locurilor de pescuit este amestecată aleatoriu în fiecare zi (fără transfer de memorie între zile).
  - Două distribuții distincte de recompense:
    - *Constantă (Steady):* Normală trunchiată $N(\mu, \sigma)$ cu media $\sim 0.8\text{--}2.6\,\text{kg}$.
    - *Banc de pești (School):* Coadă grea la dreapta cu probabilitate Bernoulli $p$ a unui banc de pești walu ($N(20, 5)\,\text{kg}$), iar în caz contrar captură de bază $N(0.6, 0.4)\,\text{kg}$.
    - Optimul de referință al recifului: locul cu bancuri la $p=0.35$ (media $7.46\,\text{kg}$/aruncare), în timp ce al doilea cel mai bun loc constant are o medie de doar $2.63\,\text{kg}$.
- **Obiectiv:** Implementarea unei strategii de bandiți cu mai multe brațe (`Strategy`) în `main.py` prin I/O streaming (`stdout` indexul locului, `stdin` recompensa), depășind captura medie țintă de **5.300 kg/zi** pe parcursul a 100 de zile, într-o fereastră de execuție de 300s.

## 2. Comparație cu Baseline-urile și Perspectivă Cheie
- **Eșecul Algoritmilor Standard de Bandiți:**
  - *Starter Greedy:* 3.847.4 kg/zi (se blochează în locuri constante suboptimale după câteva capturi norocoase timpurii).
  - *Selecție Aleatorie:* 2.394.9 kg/zi.
  - *UCB1 Standard ($c=1$):* 4.689.6 kg/zi (medii de eșantionare distorsionate de capturile excepționale rare de 20kg pe locurile cu bancuri cu $p$ scăzut).
  - *Eșantionare Thompson Gaussiană:* 4.940.7 kg/zi (colaps al dispersiei: o serie ghinionistă de zerouri pe brațul optim cu bancuri reduce prematur probabilitățile a posteriori).
- **Strategia de Bază: UCB cu Conștientizarea Varianței (UCB1-Tuned):**
  1. *Explorare Inițială Forțată:* Fiecare loc este explorat de $\ge 2$ ori înainte de calculul scorului UCB pentru a stabili o dispersie empirică de bază și a preveni împărțirea la zero.
  2. *UCB cu Varianță Adaptivă:* Utilizează dispersia empirică scalată în funcție de captura maximă înregistrată $S = \text{max\_catch}$:
     $$V_i = \text{Var}_i + S \cdot \sqrt{\frac{2 \ln t}{N_i}}, \quad \text{Bonus}_i = \sqrt{\frac{\ln t}{N_i} \cdot V_i}, \quad \text{Index}_i = \bar{X}_i + \text{Bonus}_i$$
  3. Implementare pură din biblioteca standard (`json`, `math`, `sys`), asigurând zero dependențe externe și o execuție ultrarapidă sub 10 secunde.

## 3. Detalii de Implementare (`main.py`)
- **Clasa `Strategy`:**
  - Stare monitorizată: `casts` ($N_i$), `total` ($\sum X$), `sq_total` ($\sum X^2$), `max_catch` ($S$) și pasul zilei `t`.
  - `choose()`:
    - Dacă oricare loc are `casts[s] < 2`, îl selectează imediat.
    - În caz contrar, calculează dispersia empirică $\text{Var}_i = \max(0.01, \frac{\sum X^2}{N_i} - \bar{X}_i^2)$ și selectează $\arg\max_i \text{Index}_i$.
  - `update(spot, catch)`: Actualizează sumele curente și captura maximă observată.
- **Bucla de Evaluare:** Comunică la fiecare aruncare prin `print(spot, flush=True)` și `sys.stdin.readline()`.

## 4. Rezultate și Artefacte
- **Metrici de Performanță:**
  - Captură Medie Zilnică: **7.039.7 kg/zi** (prag țintă: **5.300.0 kg/zi**, marjă **+33%**).
  - Cea mai slabă zi: **6.073 kg** (zero colapsuri catastrofale pe parcursul a 100 de zile simulate).
  - Raport față de Oracol: **95.8% din oracolul teoretic** (7.039.7 față de 7.349.6 kg).
  - Timp de Execuție: **9.6 secunde** pentru 100 de zile (mult sub limita de 300s).
- **Flag:** `SIGMOID_BAGGING`.
- **Fișiere:** `main.py`, `solution.md`, `bench.py`, `candidates.py`, `robust.py`.
