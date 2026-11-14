# Raport de soluție: task_spain ("¡Feliz Navidad!")

## 1. Prezentare generală și obiectiv
- **Task ID:** `task_spain` (Spania).
- **Domeniu:** Analiza datelor tabulare + reconstrucție steganografică raster bitmap (Bitmap Raster Reconstruction).
- **Format de ieșire:** Șir text `SIGMOID_<WORD>`.
- **Date de intrare:**
  - `office_price_guide.csv`: Catalogul celor 6 suveniruri tradiționale spaniole (`List_no` de la 1 la 6), unitatea de măsură și intervalul de preț corect (fair) în EUR:
    1. Jamon Iberico (kg): 45.00 – 90.00 EUR
    2. Churros (duzină): 4.00 – 8.00 EUR
    3. Turron de Jijona (balsam/baton): 6.00 – 14.00 EUR
    4. Paella kit (kit): 18.00 – 35.00 EUR
    5. Saffron (g): 7.00 – 15.00 EUR
    6. Barquillos (cutie): 5.00 – 10.00 EUR
  - `purchases.csv`: 577 înregistrări de achiziții efectuate de 21 de piloți de linie (`Pilot_no` de la 1 la 21), conținând `Product`, `City`, `Quantity`, `Total_paid_eur` și caracteristici false / decoy (rating-ul magazinului, temperatura meteo, distanța de zbor etc.).
- **Reguli și mecanică:**
  - O chitanță este **corectă** (fair) dacă și numai dacă prețul unitar ($\text{unit\_price} = \text{Total\_paid\_eur} / \text{Quantity}$) se încadrează în intervalul $[\text{Fair\_min\_eur}, \text{Fair\_max\_eur}]$. Sub minim este contrafăcut („fake”); peste maxim este o „capcană pentru turiști” („tourist trap”).
  - O celulă a grilei `(product, pilot)` este bifată dacă acel pilot a achiziționat respectivul suvenir la un preț corect **cel puțin o dată**.
- **Obiectiv:** Clasificarea achizițiilor, construirea unei grile de coordonate $6 \times 21$ (rânduri: produse ordonate după `List_no`, coloane: piloți ordonați după `Pilot_no`) și decodificarea cuvântului secret din bitmap-ul raster.

---

## 2. Observații-cheie și metodologie

### 2.1 Clasificarea prețurilor
- Calcularea $\text{unit\_price} = \text{Total\_paid\_eur} / \text{Quantity}$ împarte toate cele 577 de chitanțe în trei grupuri distincte și clar separate:
  - **Capcane pentru turiști ($\text{unit\_price} > \text{Fair\_max\_eur}$):** 270 chitanțe (de 3.8× până la 8.6× peste valoarea normală).
  - **Falsuri ($\text{unit\_price} < \text{Fair\_min\_eur}$):** 230 chitanțe (între 0.2× și 0.55× din minim).
  - **Achiziții corecte / Fair ($\text{Fair\_min} \le \text{unit\_price} \le \text{Fair\_max}$):** Exact **77 de chitanțe**.
- Toate caracteristicile auxiliare (temperatura, rating-ul magazinului, distanța de zbor) sunt piste false (red herrings). Coloana `City` corespunde cu `Home_city` în 100% din rânduri.

### 2.2 Construirea matricei binare $6 \times 21$
- Se creează o matrice binară $M \in \{0, 1\}^{6 \times 21}$:
  - Rândurile $r \in \{0, \dots, 5\}$: suveniruri ordonate după `List_no` (1: Jamon, 2: Churros, 3: Turron, 4: Paella, 5: Saffron, 6: Barquillos).
  - Coloanele $c \in \{0, \dots, 20\}$: piloți ordonați după `Pilot_no` de la 1 la 21.
  - $M[r, c] = 1$ dacă pilotul $c+1$ are $\ge 1$ achiziție corectă (fair) a suvenirului $r+1$, altfel $0$.

### 2.3 Decodificarea glifelor bitmap
- Coloanele **4, 8, 12, 16 și 18** sunt exclusiv zero, funcționând ca spațiatori inter-literali de 1 pixel.
- Coloanele nenule formează glife raster monospace clare, cu înălțimea de 6 pixeli:
  - **Col 1–3 (lățime 3):** litera mică `c`
  - **Col 4 (lățime 1):** spațiator
  - **Col 5–7 (lățime 3):** litera mare `R`
  - **Col 8 (lățime 1):** spațiator
  - **Col 9–11 (lățime 3):** litera mică `u`
  - **Col 12 (lățime 1):** spațiator
  - **Col 13–15 (lățime 3):** litera mare `C`
  - **Col 16 (lățime 1):** spațiator
  - **Col 17 (lățime 1):** litera îngustă `i`
  - **Col 18 (lățime 1):** spațiator
  - **Col 19–21 (lățime 3):** litera mare `O`
- Defalcarea lățimii pe coloane: $3 + 1 + 3 + 1 + 3 + 1 + 3 + 1 + 1 + 1 + 3 = 21$ coloane.
- Cuvânt decodificat: **`CRUCIO`** $\to$ Flag: **`SIGMOID_CRUCIO`** (corespunde bibliotecii open-source de augmentare a datelor *Crucio* dezvoltată de Sigmoid și blestemului din Harry Potter).

---

## 3. Detalii de implementare

1. **`solve.py` (Pipeline reproductibil):**
   - Îmbină `purchases.csv` cu `office_price_guide.csv` pe coloana `Product`.
   - Filtrează chitanțele care respectă condiția `Fair_min_eur <= unit_price <= Fair_max_eur`.
   - Populează o matrice NumPy $6 \times 21$:
     ```python
     matrix = np.zeros((6, 21), dtype=int)
     for _, row in fair.iterrows():
         matrix[int(row["List_no"]) - 1, int(row["Pilot_no"]) - 1] = 1
     ```
   - Randează grila raster în format ASCII:
     ```
          123456789012345678901
        +---------------------+
     1  |....##......###...###|  (Jamon)
     2  |....#.#.....#...#.#.#|  (Churros)
     3  |###.#.#.#.#.#.....#.#|  (Turron)
     4  |#...##..#.#.#...#.#.#|  (Paella)
     5  |#...#.#.#.#.#...#.#.#|  (Saffron)
     6  |###.#.#.###.###.#.###|  (Barquillos)
     ```
   - Trimite `SIGMOID_CRUCIO` către API-ul platformei.

2. **`solution.md` (Verificarea robusteții):**
   - Pragurile alternative de agregare („toate fair” $\to 0$ px; „majoritatea fair” $\to 6$ puncte; „$\ge 2$ fair” $\to 20$ puncte răzlețe) distrug continuitatea glifelor; textul apare în mod unic doar la condiția „$\ge 1$ achiziție fair”.
   - Orice reordonare a rândurilor (alfabetic, după preț) sau a coloanelor generează zgomot ilizibil.

---

## 4. Rezultate și artefacte
- **Flag final:** `SIGMOID_CRUCIO`
- **Nivel de încredere:** Ridicat (randare neechivocă a caracterelor monospace cu delimitatori exacți de coloană).
- **Artefacte-cheie (`task_spain/`):**
  - `answer.txt`: Conține `SIGMOID_CRUCIO`.
  - `solve.py`: Script complet de procesare a datelor, rasterizare ASCII și trimitere a soluției.
  - `solution.md`: Demonstrații matematice, artă ASCII și analiza robusteții.
