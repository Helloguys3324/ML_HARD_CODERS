# Raport de Soluție: task_germany („Black Box”)

## 1. Prezentare Generală a Problemei
- **Sarcină și Domeniu (Track):** Dificilă (Hard, 300 pct). Un stand de testare a senzorilor din Stuttgart este controlat de cinci intrări analogice $x_1, \dots, x_5 \in [0, 1]$ și generează un răspuns scalar determinist, fără zgomot $y = f(x_1, \dots, x_5)$. Schemele tehnice și documentația au fost pierdute.
- **Constrângeri și Criterii de Punctare:**
  - **Buget de Interogări (Query Budget):** Strict $\le 150$ interogări către `Oracle` (a 151-a interogare duce la descalificare imediată cu 0 puncte).
  - **Limită de Timp de Execuție:** 50 de secunde pentru `solve(oracle, points)`.
  - **Cerințe de Identificabilitate Structurală:**
    1. `relevant_variables` / `irrelevant_variables`: Clasificare binară exactă a tuturor celor 5 intrări.
    2. `interactions`: Identificarea perechii interactive $(x_p, x_q)$ și a coeficientului de interacțiune $c$ cu o marjă de eroare de cel mult 15%.
    3. `discontinuities`: Identificarea variabilelor de activare (gate), a normalei hiperplanului $g$, a pragului $T$ și a amplitudinii saltului $J$ cu o marjă de cel mult 15%.
    4. `model_expression`: Șir de caractere valid reprezentând o expresie analitică Python.
    5. `predictions`: Predicții pentru cele 300 de puncte de testare din `challenge_set.json`.
  - **Capcana OWAT (One-variable-at-a-time):** Variația câte unei singure variabile în jurul centrului hipercubului (0.5) generează răspunsuri liniare cu o eroare reziduală de $10^{-11}$, eșuând complet în detectarea termenilor de interacțiune centrați sau a discontinuității în treaptă.
  - Obținerea a $\ge 70/100$ puncte pe oracolul ascuns deblochează flag-ul `SIGMOID_<WORD>`.

## 2. Metodologie Cheie și Familia Analitică
Funcția de răspuns necunoscută aparține familiei parametrice:
$$f(x) = C + \sum_{i=1}^5 a_i x_i + c \cdot (x_p - 0.5)(x_q - 0.5) + J \cdot \mathbf{1}[g \cdot x \ge T]$$

O strategie de sondare adaptivă în 4 faze gestionează strict consumul de interogări:
1. **Faza 1: Șabloane de Diferențe Finite (64 interogări):**
   - $m = 4$ puncte de bază $B_k \in [0, 1]^5$ eșantionate prin Latin Hypercube Sampling (LHS), decalate de marginile domeniului cu $h = 0.05$.
   - Un șablon de 16 puncte este interogat pentru fiecare punct de bază: baza $x$, 5 deplasări individuale $x + h e_i$ și 10 deplasări pereche $x + h e_i + h e_j$ ($4 \times 16 = 64$ interogări).
   - *Diferențe mixte de ordinul doi:* $D_{ij} = f(x + h e_i + h e_j) - f(x + h e_i) - f(x + h e_j) + f(x)$. În regiunile netede, $D_{ij} \approx c \cdot h^2$ exclusiv pentru perechea reală $(p, q)$. Mediana pe șabloanele neafectate de salt ($|D_{ij}| < 0.5$) izolează coeficientul $c$.
   - *Diferențe de ordinul întâi:* Gradienții liniari $a_i$ sunt calculați scăzând termenul de interacțiune centrat:
     $$a_i = \text{median}\left(\frac{f(x + h e_i) - f(x)}{h} - c \cdot (x_q - 0.5)\right)$$
2. **Faza 2: Interogarea Colțurilor Hipercubului (32 interogări):**
   - Reziduul $r(x) = f(x) - \text{smooth}(x)$ evaluează la $0$ sau $J$.
   - Sunt interogate toate cele 32 de colțuri $\{0, 1\}^5$. Clustering-ul 2-means separă starea inactivă ($r \approx 0$) de cea activă ($r \approx J$), determinând mărimea saltului $J = c_{hi} - c_{lo}$ și variabilele candidate de poartă (gate) de-a lungul muchiilor opuse 1D ale hipercubului.
3. **Faza 3: Bisecție și Ajustarea Hiperplanului prin PCA (44 interogări):**
   - Sondarea direcțională a limitelor (`_probe_expand`) și bisecțiile în 9 pași localizează coordonatele de frontieră între stările comutate.
   - Descompunerea valorilor singulare (PCA / SVD) ajustează versorul normal $g$ al hiperplanului și pragul de deplasare $T$, orientate astfel încât $g \cdot x_{\text{jump}} \ge T$.
4. **Faza 4: Calibrarea Termenului Liber (Intercept) și Evaluare:**
   - Evaluează interceptul median $C$ pe puncte situate ferm în regiunea inactivă ($g \cdot x < T - 0.005$) și calculează predicțiile pentru cele 300 de puncte de testare.

## 3. Contabilitatea Interogărilor și Detalii de Implementare
- **Defalcarea Bugetului:**
  - Faza 1 (Șabloane): $4 \times 16 = 64$ interogări.
  - Faza 2 (Colțurile hipercubului): 32 interogări.
  - Faza 3 (Sonde de frontieră și bisecții): 44 interogări.
  - **Total Interogări:** Exact **140 / 150** interogări utilizate.
- **Structura Codului:**
  - `solve.py` / `main.py`: Încapsulează clasa `Solver` cu metodele `lhs`, `_two_means`, `_bisect` și generatorii de predicții.
  - `dev_test.py`: Suite de verificare pe oracole sintetice.

## 4. Rezultate și Artefacte
- **Performanță pe Oracol:**
  - Buget de Interogări: **140 / 150** interogări utilizate.
  - Detecție Caracteristici și Interacțiuni: 100% (40/40 pct).
  - Parametrizarea Discontinuității: 100% (30/30 pct).
  - Predicții Puncte de Test (RMSE): 100% (30/30 pct).
  - Scor Total: **100 / 100 puncte**.
- **Flag:** `SIGMOID_FACTORIAL`.
- **Fișiere:** `solve.py`, `main.py`, `dev_test.py`.
